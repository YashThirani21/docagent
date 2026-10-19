"""Pipeline B: technical drawing (PDF or image) -> DrawingExtraction JSON with every value.

Findings that shaped this (see NOTES.md):
- 10 of the 11 NIST PDFs (Acrobat Distiller) draw dimensions/GD&T as vector outlines: the text layer
  only holds the title block and notes. Only FTC-07 (AutoCAD) has the annotations as real text.
  => the VLM must read annotations from pixels; the text layer is used where it exists as exact truth.
- Qwen2.5-VL-7B on a whole page at ~100 dpi misses and invents values; on ~200 dpi tiles it reads
  them well.
- Asking the VLM for every field of every value is slow (output tokens dominate on a laptop GPU).
  So the work is split by what each step is good at:
    1. READ   (VLM, per tile)  : exact text + type + GD&T symbol + box. Short output.
    2. PARSE  (code)           : quantity / nominal / tolerances / datums / modifiers.
    3. LOCATE (VLM, per sheet) : numbered marks drawn on the sheet -> 'belongs_to' per mark + view/units.
    4. EXPLAIN (code)          : 'meaning' from templates - deterministic and consistent.
- Title block and notes come from the text layer deterministically when present.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from ..config import Config
from ..llm import VLM, fit_image
from ..pdf import doc_id_from_path, render
from ..schemas import Attachment, DrawingExtraction, DrawingPage, DrawingValue, ValueType

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ VLM output schemas

GdtSymbol = str  # free text, normalised by _CHARACTERISTICS below


class _Read(BaseModel):
    raw_text: str = Field(description="Exact text as printed, e.g. '4X Ø.250 +.003 -.000' or 'Ø0.75 A B C'")
    type: ValueType
    bbox_2d: list[int] = Field(description="[x1, y1, x2, y2] pixel box of the annotation in this image")


class _TileRead(BaseModel):
    annotations: list[_Read]


class _Locate(BaseModel):
    id: int
    belongs_to: str = Field(description="Feature the mark's annotation points to + where, max 12 words")
    gdt_symbol: str = Field(description="If the mark is a feature control frame: its geometric characteristic "
                                        "(position, flatness, profile of a surface, perpendicularity, ...). Else ''")


class _SheetLocate(BaseModel):
    view: str = Field(description="View name, e.g. 'View A', 'isometric view', 'section B-B'")
    part_description: str = Field(description="What the part looks like, one sentence")
    units: str = Field(description="mm, inch or unknown")
    items: list[_Locate]


READ_PROMPT = """This image is one tile of an engineering drawing sheet (ASME Y14.5 / GD&T).
Transcribe EVERY annotation in this tile, exactly as printed:
dimensions (with tolerances, stacked +/- values, quantity prefixes like 4X), diameters (Ø), radii (R),
angles, feature control frames (all compartments: tolerance with Ø/modifiers, datum letters),
datum feature symbols (boxed letter), datum targets, surface finish, threads, chamfers, and short
text attached to features (e.g. '2 SURFACES').
Skip the sheet border, title block and general notes{skip}.
Exact strings known to be on this tile (copy them verbatim when they match): {hints}
Only report text you can actually read."""

LOCATE_PROMPT = """This is an engineering drawing sheet. Red numbered boxes mark annotations that were
read from it:
{marks}
For EVERY id, say which feature of the part that annotation's leader/dimension line points to
(e.g. 'left Ø20 side hole', 'top face', 'large slot near centre', 'hexagonal boss').
For feature control frames also name the geometric characteristic symbol in the first compartment.
Also name the view, describe the part in one sentence and give the units."""


# ------------------------------------------------------------------ parsing helpers

_NORM = str.maketrans({"Φ": "Ø", "⌀": "Ø", "ø": "Ø", "−": "-", "–": "-", "x": "X", "×": "X"})
ANNOT_TOKEN_RE = re.compile(r"(\d*\.\d+|\d+|Ø|±|°|\bR\d)")

_CHARACTERISTICS = {
    "position": "the axis/center plane must lie within a tolerance zone around its true (basic) position",
    "flatness": "the surface must lie between two parallel planes this far apart",
    "straightness": "the line element/axis must lie within this straightness zone",
    "circularity": "each circular cross-section must lie between two concentric circles this far apart",
    "cylindricity": "the surface must lie between two coaxial cylinders this far apart",
    "profile of a surface": "the surface must lie within a uniform boundary of this width around the true profile",
    "profile of a line": "each line element must lie within a uniform boundary of this width around the true profile",
    "perpendicularity": "the feature must lie within a zone perpendicular to the datum",
    "parallelism": "the feature must lie within a zone parallel to the datum",
    "angularity": "the feature must lie within a zone at the basic angle to the datum",
    "concentricity": "median points must lie within a zone coaxial with the datum axis",
    "symmetry": "median points must lie within a zone symmetric about the datum center plane",
    "circular runout": "each circular element may vary at most this much when rotated about the datum axis",
    "total runout": "the whole surface may vary at most this much when rotated about the datum axis",
}
_MODIFIERS = {"Ⓜ": "MMC", "(M)": "MMC", "Ⓛ": "LMC", "(L)": "LMC", "Ⓢ": "RFS", "Ⓟ": "projected tolerance zone",
              "Ⓕ": "free state", "Ⓣ": "tangent plane", "ⓤ": "unequally disposed", "ST": "statistical tolerance",
              "MAX": "maximum", "MIN": "minimum", "BASIC": "basic"}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s.translate(_NORM)).upper()


def _characteristic(s: str) -> str | None:
    s = (s or "").lower().strip()
    if not s:
        return None
    for k in _CHARACTERISTICS:
        if k in s or s in k:
            return k
    return s


def parse_numbers(raw: str) -> dict:
    """quantity / nominal / tolerances from the printed text (deterministic, no VLM)."""
    s = raw.translate(_NORM)
    out: dict = {}
    if m := re.match(r"\s*(\d+)\s*X\b", s, re.I):
        out["quantity"] = int(m.group(1))
        s = s[m.end():]
    nums = re.findall(r"[+-]?\d*\.\d+|[+-]?\d+", s.replace("±", " ±"))
    if nums:
        out["nominal"] = abs(float(nums[0]))
    if m_pm := re.search(r"±\s*(\d*\.?\d+)", s):
        t = float(m_pm.group(1))
        out["tol_plus"], out["tol_minus"] = t, -t
    elif len(nums) >= 3 and not re.search(r"\b[A-Z]\b", s):    # nominal +upper -lower (stacked)
        a, b = float(nums[1]), float(nums[2])
        out["tol_plus"], out["tol_minus"] = max(a, b), min(a, b)
    return out


def parse_frame(raw: str) -> dict:
    """Feature control frame: tolerance value, datum letters, modifiers."""
    s = raw.translate(_NORM)
    out: dict = {"datums": re.findall(r"(?<![A-Z])([A-H])(?![A-Z])", s.upper())}
    if m := re.search(r"(\d*\.\d+|\d+)", s):
        out["nominal"] = float(m.group(1))
    out["modifiers"] = [v for k, v in _MODIFIERS.items() if k in s.upper() or k in s]
    return out


def _retype(raw: str, vtype: str) -> str:
    """Correct the VLM's type where the printed text is unambiguous."""
    s = re.sub(r"^\d+\s*X\s*", "", raw.translate(_NORM).upper().strip())   # drop "4X" prefix
    if re.fullmatch(r"[A-Z]", s):
        return "datum_feature"
    if re.fullmatch(r"[A-Z]\d{1,2}", s):
        return "datum_target"
    if re.search(r"\d", s) and re.search(r"\d[^A-Z]*\s[A-H](\s+[A-H]){0,2}\s*$", s):
        return "gdt_frame"                      # tolerance value followed by datum letters
    if "°" in s and vtype in ("dimension", "tolerance", "other"):
        return "angle"
    if s.startswith("Ø") and vtype in ("dimension", "tolerance", "other"):
        return "diameter"
    if re.match(r"R\d", s) and vtype in ("dimension", "other"):
        return "radius"
    return vtype


def explain(v: DrawingValue) -> str:
    """Plain-language meaning, generated from the parsed fields (consistent across drawings)."""
    u = f" {v.unit}" if v.unit else ""
    qty = f"{v.quantity} features: " if v.quantity else ""
    tol = ""
    if v.tol_plus is not None and v.nominal is not None:
        lo, hi = v.nominal + v.tol_minus, v.nominal + v.tol_plus
        tol = f", tolerance +{v.tol_plus:g}/{v.tol_minus:g} (acceptable {lo:g}–{hi:g}{u})"
    n = f"{v.nominal:g}{u}" if v.nominal is not None else v.raw_text
    t = v.type
    if t == "gdt_frame":
        ch = v.characteristic or "geometric"
        zone = "cylindrical zone Ø" if "Ø" in v.raw_text.translate(_NORM) else "zone "
        datums = f" relative to datums {', '.join(v.datums)}" if v.datums else ""
        mods = f" ({', '.join(v.modifiers)})" if v.modifiers else ""
        rule = _CHARACTERISTICS.get(ch, "the feature must stay within this geometric tolerance")
        return f"{ch.capitalize()} tolerance, {zone}{n}{datums}{mods}: {rule}."
    if t == "datum_feature":
        letter = re.sub(r"[^A-Z0-9]", "", v.raw_text.upper())[:3]
        return f"Identifies this feature as datum {letter}, a reference for GD&T controls."
    if t == "datum_target":
        return f"Datum target {v.raw_text}: a specific point/line/area used to establish a datum."
    if t == "diameter":
        return f"{qty}Diameter {n}{tol}."
    if t == "radius":
        return f"{qty}Radius {n}{tol}."
    if t == "angle":
        return f"{qty}Angle {v.raw_text}{tol}."
    if t in ("dimension", "tolerance"):
        return f"{qty}Size/location dimension {n}{tol}."
    if t == "surface_finish":
        return f"Surface texture requirement {v.raw_text}."
    if t == "thread":
        return f"Thread specification {v.raw_text}."
    if t == "chamfer":
        return f"Chamfer {v.raw_text}."
    return f"Annotation '{v.raw_text}' attached to the feature."


# ------------------------------------------------------------------ page sources (PDF page or image)


@dataclass
class _Line:
    text: str
    bbox: tuple[float, float, float, float]


class _PageSource:
    """Uniform access to a drawing sheet: size in coordinate units, rendering of a clip, text layer."""

    def __init__(self, page: pymupdf.Page | None = None, image: Image.Image | None = None):
        self.page, self.image = page, image

    @property
    def size(self) -> tuple[float, float]:
        return (self.page.rect.width, self.page.rect.height) if self.page else self.image.size

    def render(self, clip: tuple[float, float, float, float], dpi: int) -> Image.Image:
        if self.page:
            return render(self.page, dpi, clip=pymupdf.Rect(*clip))
        return self.image.crop(tuple(int(v) for v in clip))

    def lines(self) -> list[_Line]:
        if not self.page:
            return []
        out = []
        for b in self.page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                t = "".join(sp["text"] for sp in ln["spans"]).strip()
                if t:
                    out.append(_Line(t, tuple(ln["bbox"])))
        return out

    def text(self) -> str:
        return self.page.get_text() if self.page else ""


# ------------------------------------------------------------------ title block & notes


def _title_block(text: str) -> dict[str, str]:
    tb: dict[str, str] = {}
    pats = {
        "title": r"^(PMI .*Test Case \d+.*)$",
        "atomic_test_cases": r"^Includes Atomic Test Cases\s*-\s*(.+)$",
        "series": r"^(NIST PMI Test Models\s*-\s*\d{4})$",
        "model": r"^(Test Model \d+)$",
        "revision": r"^Rev\.?\s+([A-Z0-9]{1,2})$",
        "drawing_number": r"^(nist_[a-z]{3}_\d{2}_\w+?)$",
        "feature_spec_index": r"^(\S+_fsi\.pdf)$",
    }
    for line in text.splitlines():
        line = line.strip()
        for k, p in pats.items():
            if k not in tb and (m := re.match(p, line, re.I)):
                tb[k] = m.group(1).strip()
    return tb


def _notes(text: str) -> list[str]:
    m = re.search(r"NOTES?(?:\s*\(UNLESS OTHERWISE SPECIFIED\))?:?\s*\n(.*)", text, re.S | re.I)
    if not m:
        return []
    body = re.split(r"\n(?:Feature and Specification Index|nist_[a-z]{3}_\d{2})", m.group(1))[0]
    items = re.split(r"\n\s*(?=\d+\.\s)", "\n" + body)
    return [re.sub(r"\s+", " ", i).strip() for i in items if i.strip()]


def _units(text: str, values: list[DrawingValue], vlm_guess: str | None) -> str | None:
    if m := re.search(r"UNITS:\s*(\w+)", text, re.I):
        return m.group(1).lower()
    dims = [v for v in values if v.type in ("dimension", "diameter", "radius", "gdt_frame")]
    leading_dot = sum(bool(re.search(r"(^|[^\d])\.\d{2,3}\b", v.raw_text)) for v in dims)
    if dims and leading_dot / len(dims) > 0.3:               # US inch style ".250"
        return "inches"
    if re.search(r"Y14\.5M", text):                          # the M edition = metric
        return "millimeters"
    if vlm_guess and vlm_guess.lower().startswith(("mm", "milli")):
        return "millimeters"
    if vlm_guess and vlm_guess.lower().startswith("inch"):
        return "inches"
    return None


# ------------------------------------------------------------------ extractor


class DrawingExtractor:
    version = "drawing-read-locate-v2"

    def __init__(self, cfg: Config, vlm: VLM | None = None):
        self.cfg = cfg
        self.dc = cfg.drawing
        self.vlm = vlm or VLM(cfg.llm)

    def extract(self, att: Attachment | str | Path) -> DrawingExtraction:
        att = att if isinstance(att, Attachment) else Attachment.from_path(att)
        drawing_id = doc_id_from_path(att.path)
        doc = pymupdf.open(att.path) if att.is_pdf else None
        sources = ([_PageSource(page=p) for p in doc] if doc
                   else [_PageSource(image=Image.open(att.path).convert("RGB"))])

        full_text = "\n".join(s.text() for s in sources)
        title_block = _title_block(full_text)
        notes = _notes(full_text)

        values: list[DrawingValue] = []
        pages: list[DrawingPage] = []
        unmatched: list[str] = []
        unit_guess = None
        for pno, src in enumerate(sources, start=1):
            log.info("%s sheet %d/%d", drawing_id, pno, len(sources))
            vals, unm, sheet = self._sheet(drawing_id, pno, src, title_block, notes)
            values += vals
            unmatched += unm
            pages.append(DrawingPage(page=pno, view=sheet.view, description=sheet.part_description))
            unit_guess = unit_guess or (sheet.units if sheet.units != "unknown" else None)

        units = _units(full_text, values, unit_guess)
        for v in values:
            if v.type == "angle":
                v.unit = "degrees"
            elif v.type not in ("note", "title_block", "datum_feature", "datum_target"):
                v.unit = units
            v.meaning = explain(v)
        values += self._text_values(drawing_id, title_block, notes)
        if doc:
            doc.close()

        return DrawingExtraction(
            drawing_id=drawing_id, source_file=Path(att.path).name, source_sha256=att.sha256, units=units,
            coord_units="pt" if att.is_pdf else "px", title_block=title_block, notes=notes, pages=pages,
            values=values, unmatched_text_tokens=unmatched, extractor_version=self.version)

    # ------------------------------------------------------------ one sheet
    def _sheet(self, drawing_id, pno, src: _PageSource, title_block, notes):
        lines = src.lines()
        known = {norm(t) for t in list(title_block.values()) + notes}
        # text-layer annotation tokens (only some CAD exporters keep them, e.g. FTC-07)
        ann_lines = [ln for ln in lines if len(ln.text) <= 40 and ANNOT_TOKEN_RE.search(ln.text)
                     and not any(norm(ln.text) in k for k in known)]
        ann_ids = {id(ln) for ln in ann_lines}
        skip_text = known | {norm(ln.text) for ln in lines if id(ln) not in ann_ids}

        vals = self._dedupe(self._read_tiles(pno, src, ann_lines, skip_text, bool(title_block or notes)))
        unmatched = self._reconcile(vals, ann_lines, pno)
        for i, v in enumerate(vals, start=1):
            v.id = f"{drawing_id}-p{pno}-v{i:03d}"
        sheet = self._locate(src, vals)
        return vals, unmatched, sheet

    def _tiles(self, w: float, h: float) -> list[tuple[float, float, float, float]]:
        cols, rows = self.dc.grid
        ow, oh = w / cols * self.dc.overlap, h / rows * self.dc.overlap
        return [(max(0, c * w / cols - ow), max(0, r * h / rows - oh),
                 min(w, (c + 1) * w / cols + ow), min(h, (r + 1) * h / rows + oh))
                for r in range(rows) for c in range(cols)]

    def _read_tiles(self, pno, src, ann_lines, skip_text, has_text_blocks) -> list[DrawingValue]:
        w, h = src.size
        out: list[DrawingValue] = []
        skip = " (already captured from the PDF text)" if has_text_blocks else ""
        jobs = []
        for clip in self._tiles(w, h):        # render sequentially (PyMuPDF pages are not thread-safe)
            hints = [ln.text for ln in ann_lines if _inside(ln.bbox, clip)]
            img = fit_image(src.render(clip, self.dc.tile_dpi), 1400)
            prompt = READ_PROMPT.format(skip=skip, hints="; ".join(hints) or "(none - read from the image)")
            jobs.append((clip, img, prompt))

        def read(job):
            return self.vlm.structured(job[2], _TileRead, images=[job[1]], max_tokens=3000)

        # VLM calls in parallel: vLLM batches them on the GPU; Ollama with one slot simply queues them
        with ThreadPoolExecutor(max_workers=self.cfg.llm.max_concurrency) as ex:
            futures = [ex.submit(read, j) for j in jobs]
        for ti, ((clip, img, _), fut) in enumerate(zip(jobs, futures), start=1):
            try:
                res = fut.result()
            except Exception as e:
                log.warning("tile %d sheet %d failed: %s", ti, pno, e)
                continue
            sx, sy = (clip[2] - clip[0]) / img.width, (clip[3] - clip[1]) / img.height
            for a in res.annotations:
                raw = a.raw_text.strip()
                if not raw or norm(raw) in skip_text:
                    continue
                bbox = None
                if len(a.bbox_2d) == 4:
                    x1, y1, x2, y2 = a.bbox_2d
                    bbox = [round(clip[0] + x1 * sx, 1), round(clip[1] + y1 * sy, 1),
                            round(clip[0] + x2 * sx, 1), round(clip[1] + y2 * sy, 1)]
                if any(len(norm(raw)) > 3 and norm(raw) in t for t in skip_text):
                    continue                                    # title block / notes text leaking in
                vtype = _retype(raw, a.type)
                fields: dict = {}
                if vtype == "gdt_frame":
                    fields = parse_frame(raw)
                elif vtype not in ("datum_feature", "datum_target", "note"):
                    fields = parse_numbers(raw)
                out.append(DrawingValue(
                    id="", page=pno, raw_text=raw, type=vtype,
                    belongs_to="", meaning="", bbox=bbox, source="vlm", confidence=0.6, **fields))
        return out

    def _locate(self, src: _PageSource, vals: list[DrawingValue]) -> _SheetLocate:
        """Set-of-marks prompting: draw numbered boxes where values were read, ask what each points to."""
        w, h = src.size
        img = src.render((0, 0, w, h), int(self.dc.overview_dpi * 1.5))
        scale = img.width / w
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        marks = []
        for i, v in enumerate(vals, start=1):
            if not v.bbox:
                continue
            x0, y0, x1, y1 = (c * scale for c in v.bbox)
            draw.rectangle([x0, y0, x1, y1], outline=(220, 0, 0), width=2)
            draw.text((x0, max(0, y0 - 15)), str(i), fill=(220, 0, 0), font=font)
            marks.append(f"{i}: {v.raw_text}")
        empty = _SheetLocate(view="unknown", part_description="", units="unknown", items=[])
        if not marks:
            return empty
        try:
            res = self.vlm.structured(LOCATE_PROMPT.format(marks="\n".join(marks)), _SheetLocate,
                                      images=[fit_image(img, 1400)], max_tokens=3000)
        except Exception as e:
            log.warning("locate failed: %s", e)
            return empty
        by_id = {it.id: it for it in res.items}
        for i, v in enumerate(vals, start=1):
            it = by_id.get(i)
            v.belongs_to = (it.belongs_to if it else "") or v.belongs_to or "not determined"
            if it and v.type == "gdt_frame" and it.gdt_symbol:
                v.characteristic = _characteristic(it.gdt_symbol)
        return res

    def _dedupe(self, vals: list[DrawingValue]) -> list[DrawingValue]:
        """Tiles overlap, so one annotation can be read twice. Same text + nearby box = same value."""
        kept: list[DrawingValue] = []
        for v in vals:
            dup = any(norm(k.raw_text) == norm(v.raw_text) and k.bbox and v.bbox
                      and _center_dist(k.bbox, v.bbox) < self.dc.dedup_distance_pt for k in kept)
            if not dup:
                kept.append(v)
        return kept

    def _reconcile(self, vals: list[DrawingValue], ann_lines: list[_Line], pno: int) -> list[str]:
        """Where a text layer exists it is ground truth: confirm VLM values, add what the VLM missed."""
        if not ann_lines:
            return []
        unmatched = []
        for ln in ann_lines:
            t = norm(ln.text)
            hit = next((v for v in vals if t and (t in norm(v.raw_text) or norm(v.raw_text) in t)), None)
            if hit:
                hit.source, hit.confidence = "vlm+text_layer", 0.9
                hit.bbox = hit.bbox or [round(c, 1) for c in ln.bbox]
            elif len(t) >= 2:
                unmatched.append(ln.text)
                vals.append(DrawingValue(
                    id="", page=pno, raw_text=ln.text, type="other", belongs_to="", meaning="",
                    bbox=[round(c, 1) for c in ln.bbox], source="text_layer", confidence=0.5,
                    **parse_numbers(ln.text)))
        for v in vals:                     # text layer exists, but this VLM value is not in it
            if v.source == "vlm":
                v.confidence = 0.4
        return unmatched

    def _text_values(self, drawing_id: str, title_block: dict[str, str], notes: list[str]) -> list[DrawingValue]:
        out = []
        for k, val in title_block.items():
            out.append(DrawingValue(id=f"{drawing_id}-tb-{k}", page=1, raw_text=val, type="title_block",
                                    belongs_to="title block", meaning=f"{k.replace('_', ' ').capitalize()}: {val}",
                                    source="text_layer", confidence=1.0))
        for i, n in enumerate(notes, start=1):
            out.append(DrawingValue(id=f"{drawing_id}-note-{i}", page=1, raw_text=n, type="note",
                                    belongs_to="general notes (whole drawing)",
                                    meaning=f"General note {i}; applies to the whole part unless stated otherwise.",
                                    source="text_layer", confidence=1.0))
        return out


def _inside(b, clip) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return clip[0] <= cx <= clip[2] and clip[1] <= cy <= clip[3]


def _center_dist(a, b) -> float:
    return (((a[0] + a[2]) - (b[0] + b[2])) ** 2 + ((a[1] + a[3]) - (b[1] + b[3])) ** 2) ** 0.5 / 2
