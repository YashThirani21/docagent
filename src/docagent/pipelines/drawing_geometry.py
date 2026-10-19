"""Geometry-first drawing back end: annotation candidates -> one read each -> values with meaning.

Per sheet:
  1. candidates from the vector front end (PDF) or the raster front end (pictures / scanned PDFs)
  2. READ    one VLM call per candidate crop: its values, in reading order (short structured output)
  3. SYMBOL  one VLM call per feature-control-frame crop: geometric characteristic from a fixed list
  4. FEATURE one VLM call per leader end: a crop around the arrow/dot names the feature it points to
  5. code    parse numbers, merge stacked fragments and limit dimensions, reconcile with the text layer,
             consistency checks -> confidence; meaning from templates (explain)
All VLM calls of a sheet run in parallel (llm.max_concurrency); crops are rendered first (PyMuPDF is
not thread-safe). Every call is disk-cached.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pymupdf
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from ..llm import VLM, fit_image
from ..pdf import render
from ..schemas import DrawingValue, ValueType
from .drawing_annotations import Candidate, RasterFrontEnd, VectorFrontEnd, load_image
from .gdt_symbols import datum_letter, frame_symbols

log = logging.getLogger(__name__)

GdtChar = Literal["none", "position", "flatness", "straightness", "circularity", "cylindricity",
                  "profile of a surface", "profile of a line", "perpendicularity", "parallelism", "angularity",
                  "concentricity", "symmetry", "circular runout", "total runout", "unknown"]


class _Item(BaseModel):
    raw_text: str = Field(description="One value exactly as printed in the crop")
    type: ValueType
    characteristic: GdtChar = Field(description="For a feature control frame row: its geometric characteristic; else 'none'")


class _CalloutRead(BaseModel):
    items: list[_Item]


class _Symbol(BaseModel):
    characteristic: GdtChar


class _Letter(BaseModel):
    letter: str = Field(description="The single capital letter (or letter+digit) inside the box")


class _Feature(BaseModel):
    feature: str = Field(description="The feature the red circle marks, max 10 words")


READ_PROMPT = """Read the annotation in this crop of a mechanical engineering drawing (the text may be rotated).
Transcribe ONLY what is printed in this crop. Do not invent or reuse values from anywhere else.
One item per printed value, in reading order:
- dimension: the number with its prefix symbol, quantity prefix and tolerance in ONE item; stacked upper and
  lower tolerances belong to the same item; two numbers joined by a dash are limits of one item
- feature control frame: ONE item per row - symbol, tolerance value with its modifiers, then the datum letters
- a single boxed letter on its own: datum feature
- words attached to the annotation: type other; a paragraph of general notes: type note
Copy every digit and letter exactly as printed. If nothing is printed in the crop, return an empty list.{hints}"""

SYMBOL_PROMPT = """This crop shows a GD&T feature control frame. Which geometric characteristic symbol is in its
first compartment?
position = circle with a cross (⌖); flatness = parallelogram (▱); straightness = short horizontal line (—);
circularity = circle (○); cylindricity = circle between two slanted lines (⌭);
profile of a surface = closed half-circle / semicircle with a base line (⌓); profile of a line = open arc (⌒);
perpendicularity = upside-down T (⊥); parallelism = two slanted parallel lines (∥); angularity = angle (∠);
concentricity = two concentric circles (◎); symmetry = three stacked horizontal lines;
circular runout = one slanted arrow (↗); total runout = two slanted arrows.
If the first compartment holds no symbol (a second row of a composite frame), answer none."""

DATUM_PROMPT = """This crop shows a datum feature symbol from an engineering drawing: a small box containing one
capital letter (sometimes a letter followed by a digit), attached to a triangle or a leader. Which letter is
in the box? Answer with the letter only."""

FEATURE_PROMPT = """This is a crop of an engineering drawing of a part. A red circle marks where an annotation's
leader line ends. Name the feature of the part at the red circle in a few words: what kind of feature it is (for example a hole,
face, slot, boss, edge, groove, pad or fillet) and where it is on the part.
{hint}"""


def _num_set(s: str) -> list[float]:
    return [float(n) for n in re.findall(r"\d*\.\d+|\d+", s)]


class GeometryReader:
    """Shared back end for both front ends."""

    def __init__(self, cfg, vlm: VLM, parse_numbers, parse_frame, retype, characteristic, norm):
        self.cfg, self.vlm = cfg, vlm
        self.parse_numbers, self.parse_frame, self.retype = parse_numbers, parse_frame, retype
        self.characteristic, self.norm = characteristic, norm

    # ------------------------------------------------------------ candidates
    def sheet_candidates(self, page: pymupdf.Page | None, image: Image.Image | None, pno: int,
                         known: set[str]) -> tuple[list[Candidate], "_Cropper"]:
        if page is not None:
            cands = VectorFrontEnd().candidates(page, pno, known)
            if cands:
                return cands, _Cropper(page=page)
            image = render(page, 200)                     # scanned / flattened PDF page -> raster route
            cands, img = RasterFrontEnd().candidates(image, pno)
            return cands, _Cropper(image=img)
        cands, img = RasterFrontEnd().candidates(image, pno)
        return cands, _Cropper(image=img)

    # ------------------------------------------------------------ read one sheet
    def read_sheet(self, drawing_id: str, pno: int, cands: list[Candidate], crop: "_Cropper",
                   ) -> tuple[list[DrawingValue], list[str]]:
        jobs = []                                         # (kind, cand_idx, extra, image, prompt, schema)
        sym_rows: dict[int, list[tuple[str, str]]] = {}
        reads: dict[int, _CalloutRead] = {}
        for ci, c in enumerate(cands):
            hints = ""
            if c.exact_text:
                hints = "\nExact strings printed in this crop (copy their digits): " + "; ".join(c.exact_text[:20])
            img = crop.crop(c.box, pad=3, angle=c.angle)
            if _is_datum_box(c):
                letter, score = datum_letter(crop.crop(c.box, pad=4, min_side=200))
                if letter and score >= 0.6:            # un-sheared box + OCR: reliable, no VLM call
                    reads[ci] = _CalloutRead(items=[_Item(raw_text=letter, type="datum_feature", characteristic="none")])
                else:
                    jobs.append(("datum", ci, None, crop.crop(c.box, pad=4, min_side=320, angle=c.angle), DATUM_PROMPT, _Letter))
                continue
            sym_rows[ci] = frame_symbols(img)          # deterministic GD&T symbols (the VLM can't read them)
            jobs.append(("read", ci, None, img, READ_PROMPT.format(hints=hints), _CalloutRead))
            if c.leaders:
                end = c.leaders[0].end
                hint = f"Geometry hint: {c.hint}." if c.hint else ""
                jobs.append(("feature", ci, None, crop.marked(end, half=90), FEATURE_PROMPT.format(hint=hint), _Feature))

        def run(job):
            kind, ci, extra, img, prompt, schema = job
            try:
                res = self.vlm.structured(prompt, schema, images=[img], max_tokens=600 if kind == "read" else 60)
                if kind == "read" and not res.items:          # occasional empty answer: retry without hints
                    res = self.vlm.structured(READ_PROMPT.format(hints="") + "\nThere IS an annotation in this crop.",
                                              schema, images=[img], max_tokens=600)
                return res
            except Exception as e:
                log.warning("%s call failed (%s p%d cand %d): %s", kind, drawing_id, pno, ci, e)
                return None
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.llm.max_concurrency)) as ex:
            results = list(ex.map(run, jobs))

        symbols: dict[int, list[str]] = {}
        features: dict[int, str] = {}
        for (kind, ci, extra, *_), res in zip(jobs, results):
            if res is None:
                continue
            if kind == "read":
                reads[ci] = res
            elif kind == "datum":
                letter = re.sub(r"[^A-Z0-9]", "", res.letter.upper())[:3]
                if letter and letter[0].isalpha():
                    reads[ci] = _CalloutRead(items=[_Item(raw_text=letter, type="datum_feature", characteristic="none")])
            elif kind == "symbol":
                symbols.setdefault(ci, []).append(res.characteristic)
            else:
                features[ci] = res.feature.strip()

        values: list[DrawingValue] = []
        notes: list[str] = []
        for ci, c in enumerate(cands):
            r = reads.get(ci)
            if (not r or not r.items) and c.exact_text:   # model gave nothing: the text layer is exact
                r = _CalloutRead(items=[_Item(raw_text=t, type="dimension", characteristic="none") for t in c.exact_text])
            if not r:
                continue
            rows = sym_rows.get(ci, [])
            items = self._merge_fragments(_split_joined([it for it in r.items if it.raw_text.strip()]))
            items += _unread_words(c.exact_text, items)        # text-layer word annotations the model skipped
            items = self._coalesce_frames(items, len(rows))
            n_frames = sum(1 for it in items if self.retype(it.raw_text, it.type) == "gdt_frame")
            syms = _fit_rows([ch for ch, _ in rows], n_frames)
            frame_k = 0
            for it in items:
                if not _plausible(it):
                    continue
                if it.type == "note" or (len(it.raw_text) > 80 and it.type == "other"):
                    notes.append(re.sub(r"\s+", " ", it.raw_text).strip())
                    continue
                v = self._value(drawing_id, pno, c, it, features.get(ci))
                if v.type == "gdt_frame":
                    sym = syms[frame_k] if frame_k < len(syms) else None
                    frame_k += 1
                    if sym not in (None, "none", "unknown"):
                        v.characteristic = sym
                        v.checks.append("symbol: recognised from frame geometry")
                        v.confidence = round(min(0.99, v.confidence + 0.1), 2)
                    else:                              # fall back to the reading (weak)
                        v.characteristic = None if it.characteristic in ("none", "unknown") else it.characteristic
                        v.checks.append("symbol: from reading only (not verified)")
                values.append(v)
        values = self._limit_pairs(values)
        return values, notes

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _propagate(syms: list[str]) -> list[str]:
        """Composite frames: second rows have no symbol cell -> inherit the row above."""
        out, last = [], None
        for s in syms:
            if s in ("none", "unknown") and last:
                out.append(last)
            else:
                out.append(s)
                last = s if s not in ("none", "unknown") else last
        return out

    @staticmethod
    def _coalesce_frames(items: list[_Item], n_rows: int) -> list[_Item]:
        """The model often splits a frame row into '0.75' + 'A B C', or reads '0.2' as a dimension.
        Rebuild rows: tolerance value (+ modifiers, projected zone, MAX) + datum letters."""
        items = GeometryReader._join_frame_parts(items)
        if not n_rows:
            return items
        tol_re = re.compile(r"^[^\d±+]*\d*\.?\d+\s*(\(?[MLPSTFU]\)?|MAX|Ⓜ|Ⓛ|Ⓟ|Ⓢ)?[^\d±]*$", re.I)
        datums_re = re.compile(r"^\s*([A-H](\s*[-–]?\s*[A-H]){0,3}\s*)(\(?[MLS]\)?)?\s*$")
        out, used, i = [], 0, 0
        while i < len(items):
            it = items[i]
            already = it.type == "gdt_frame"
            nxt = items[i + 1] if i + 1 < len(items) else None
            if not already and used < n_rows and tol_re.match(it.raw_text) and "±" not in it.raw_text:
                if nxt and datums_re.match(nxt.raw_text):
                    out.append(_Item(raw_text=f"{it.raw_text} {nxt.raw_text}", type="gdt_frame", characteristic="none"))
                    used += 1
                    i += 2
                    continue
                if not re.search(r"\d+X|\bX\b", it.raw_text):     # bare tolerance in a frame (flatness 0.2)
                    out.append(_Item(raw_text=it.raw_text, type="gdt_frame", characteristic="none"))
                    used += 1
                    i += 1
                    continue
            if already:
                used += 1
            out.append(it)
            i += 1
        return out

    @staticmethod
    def _join_frame_parts(items: list[_Item]) -> list[_Item]:
        """Signals that do not need frame geometry: 'A B C' (2+ letters is never a datum feature) belongs to
        the value before it; 'Ø.020P' + '.425' is a projected tolerance zone; 'Ø.040M' + 'Ø.045 MAX'."""
        multi = re.compile(r"^\s*[A-H](\s*[-–]?\s*[A-H]){1,3}\s*$")
        proj = re.compile(r"(\(P\)|Ⓟ|\d\s*P)\s*$", re.I)
        out: list[_Item] = []
        for it in items:
            t = it.raw_text.strip()
            prev = out[-1] if out else None
            if prev and multi.match(t) and re.search(r"\d", prev.raw_text) and "±" not in prev.raw_text \
                    and not re.search(r"\d+\s*X\b", prev.raw_text):
                out[-1] = _Item(raw_text=f"{prev.raw_text} {t}", type="gdt_frame", characteristic=prev.characteristic)
                continue
            if prev and proj.search(prev.raw_text) and re.fullmatch(r"\.?\d*\.?\d+\s*([A-H](\s+[A-H]){0,3})?", t):
                out[-1] = _Item(raw_text=f"{prev.raw_text} {t}", type="gdt_frame", characteristic=prev.characteristic)
                continue
            if prev and re.search(r"(\(M\)|Ⓜ|\d\s*M)\s*$", prev.raw_text) and re.search(r"MAX\s*$", t, re.I):
                out[-1] = _Item(raw_text=f"{prev.raw_text} {t}", type="gdt_frame", characteristic=prev.characteristic)
                continue
            out.append(it)
        return out

    @staticmethod
    def _merge_fragments(items: list[_Item]) -> list[_Item]:
        """'Ø20' + '+0.05 -0.10' read as two items -> one dimension."""
        out: list[_Item] = []
        for it in items:
            t = it.raw_text.strip()
            if out and re.fullmatch(r"[+\-±]\s*\d*\.?\d+(\s*[+\-]\s*\d*\.?\d+)?", t) \
                    and out[-1].type not in ("gdt_frame", "datum_feature", "note", "other"):
                out[-1] = _Item(raw_text=out[-1].raw_text + " " + t, type=out[-1].type,
                                characteristic=out[-1].characteristic)
            else:
                out.append(it)
        return out

    def _limit_pairs(self, values: list[DrawingValue]) -> list[DrawingValue]:
        """Stacked limit dimension read as two values ('Ø35.2' above 'Ø34.8') -> one with limits."""
        out: list[DrawingValue] = []
        for v in values:
            prev = out[-1] if out else None
            if prev and prev.page == v.page and prev.bbox == v.bbox and prev.type == v.type \
                    and v.type in ("diameter", "dimension", "radius") and prev.tol_plus is None and v.tol_plus is None \
                    and prev.nominal and v.nominal and prev.lower_limit is None and v.lower_limit is None \
                    and abs(prev.nominal - v.nominal) / max(prev.nominal, v.nominal) < 0.05:
                lo, hi = sorted([prev.nominal, v.nominal])
                prev.raw_text = f"{prev.raw_text} {v.raw_text}"
                prev.lower_limit, prev.upper_limit, prev.nominal = lo, hi, None
                prev.checks.append("limits: stacked pair merged")
                continue
            out.append(v)
        return out

    def _value(self, drawing_id: str, pno: int, c: Candidate, it: _Item, feature: str | None) -> DrawingValue:
        raw = re.sub(r"\s+", " ", it.raw_text).strip()
        # a leader stub touching the text is read as a dash: "-Ø34.8", "-1.25 A B" (never valid)
        raw = re.sub(r"^[-–:·]+\s*(?=[ØΦ⌀ø]|\d+\s*X\b|R\s*\.?\d)", "", raw)
        if it.type == "gdt_frame":
            raw = re.sub(r"^[-–:·]+\s*(?=\d|\.)", "", raw)
        checks: list[str] = []
        conf = 0.55
        # text layer = exact digits: correct the reading where they disagree
        if c.exact_text:
            best = max(c.exact_text, key=lambda e: _overlap(raw, e))
            if _overlap(raw, best) >= 0.5:
                if sorted(_num_set(best)) and not set(_num_set(best)) <= set(_num_set(raw)):
                    checks.append(f"text layer corrected digits ('{best}')")
                    raw = best if len(_num_set(best)) >= len(_num_set(raw)) else raw
                    conf += 0.1
                else:
                    checks.append("text layer agrees")
                    conf += 0.3
        vtype = self.retype(raw, it.type)
        fields: dict = {}
        if vtype == "gdt_frame":
            fields = self.parse_frame(raw)
        elif vtype not in ("datum_feature", "datum_target", "note", "other"):
            fields = self.parse_numbers(raw)
            lim = _limits(raw)
            if lim:
                fields.update(lower_limit=lim[0], upper_limit=lim[1], nominal=None, tol_plus=None, tol_minus=None)
            if fields.get("nominal") is not None or lim:
                conf += 0.1
                checks.append("numbers parsed")
            n, tp, tm = fields.get("nominal"), fields.get("tol_plus"), fields.get("tol_minus")
            if n is not None and tp is not None and tm is not None:
                fields.update(lower_limit=round(n + tm, 6), upper_limit=round(n + tp, 6))
        if feature:
            conf += 0.1
            checks.append("leader end located and named")
        belongs = feature or ("not determined (no leader found)" if not c.leaders else "not determined")
        return DrawingValue(id="", page=pno, raw_text=raw, type=vtype, belongs_to=belongs, meaning="",
                            bbox=[round(x, 1) for x in c.box], source="vlm+text_layer" if c.exact_text else "vlm",
                            confidence=round(min(conf, 0.99), 2), checks=checks, **fields)


def _unread_words(exact: list[str], items: list[_Item]) -> list[_Item]:
    """Word annotations ('4 COPLANAR SURFACES', 'SIM REQT') are exact in the text layer; add the ones no
    reading covers. Consecutive word-only lines form one phrase (the layer splits wrapped lines)."""
    read = re.sub(r"\s+", " ", " ".join(it.raw_text for it in items)).upper()
    out, phrase = [], []
    for t in exact + [""]:
        words = t.strip()
        is_words = bool(words) and len(re.sub(r"[^A-Z]", "", words.upper())) >= 4 and not re.search(r"[Ø±.]\d|\d\.\d", words)
        if is_words:
            phrase.append(words)
            continue
        if phrase:
            text = " ".join(phrase)
            if text.upper() not in read and not all(p.upper() in read for p in phrase):
                out.append(_Item(raw_text=text, type="other", characteristic="none"))
            phrase = []
    return out


def _plausible(it: _Item) -> bool:
    """Drop readings that cannot be a value: punctuation only, a 'datum' that is not a single letter,
    a 'dimension' without a digit."""
    t = it.raw_text.strip()
    if not re.search(r"[A-Za-z0-9]", t):
        return False
    if it.type in ("datum_feature",) and not re.fullmatch(r"[A-Z]\d{0,2}", re.sub(r"\s", "", t.upper())):
        return re.fullmatch(r"[A-H](\s*[A-H]){1,3}", t.upper()) is not None   # kept: joined into a frame later
    if it.type in ("dimension", "diameter", "radius", "angle", "tolerance") and not re.search(r"\d", t):
        return False
    return True


def _split_joined(items: list[_Item]) -> list[_Item]:
    """The model sometimes joins two values with '|' - split them (tolerance fragments re-merge later)."""
    out = []
    for it in items:
        parts = [p.strip() for p in it.raw_text.split("|") if p.strip()]
        out += [it] if len(parts) <= 1 else [_Item(raw_text=p, type=it.type, characteristic=it.characteristic) for p in parts]
    return out


_COMMON = ("position", "profile of a surface", "flatness", "perpendicularity", "parallelism", "profile of a line")


def _fit_rows(rows: list[str], n: int) -> list[str]:
    """More symbol rows than frame values: stray boxes were taken for rows - drop rare classes first."""
    rows = list(rows)
    while len(rows) > n and n > 0:
        rare = [k for k, r in enumerate(rows) if r not in _COMMON]
        rows.pop(rare[-1] if rare else len(rows) - 1)
    out, last = [], None                                # composite second rows inherit
    for r in rows:
        if r in ("none", "unknown") and last:
            r = last
        out.append(r)
        last = r if r not in ("none", "unknown") else last
    return out


def datum_check(values: list[DrawingValue]) -> None:
    """Datum letters used in frames should be defined by datum features somewhere on the drawing."""
    defined = {re.sub(r"[^A-Z]", "", v.raw_text.upper()) for v in values if v.type == "datum_feature"}
    defined = {d for d in defined if len(d) == 1}
    if not defined:
        return
    for v in values:
        if v.type == "gdt_frame" and v.datums:
            if set(v.datums) <= defined:
                v.checks.append("datums defined on drawing")
                v.confidence = round(min(0.99, v.confidence + 0.1), 2)
            else:
                v.checks.append(f"datum(s) {sorted(set(v.datums) - defined)} not defined on drawing")
                v.confidence = round(max(0.05, v.confidence - 0.2), 2)


def _overlap(a: str, b: str) -> float:
    na, nb = _num_set(a), _num_set(b)
    if not na or not nb:
        return 0.0
    return len(set(na) & set(nb)) / max(len(set(na)), len(set(nb)))


def _limits(raw: str) -> tuple[float, float] | None:
    """'Ø34.8 - 35.2', 'Ø35.2-34.8', 'R.012 - .022' -> (lower, upper)."""
    s = raw.replace("Φ", "Ø").replace("⌀", "Ø")
    m = re.search(r"(\d*\.\d+|\d+)\s*[-–]\s*Ø?(\d*\.\d+|\d+)\s*$", s)
    if m and "±" not in s and "+" not in s:
        a, b = sorted((float(m.group(1)), float(m.group(2))))
        if b > 0 and a > 0 and (b - a) / b < 0.6:
            return a, b
    return None


def _is_datum_box(c: Candidate) -> bool:
    """One small box with near-equal SIDES (not bbox: isometric frames have square-ish bboxes) and nothing
    else around it: a datum feature symbol."""
    import math as _m
    polys = getattr(c, "polys", [])
    if len(polys) != 1 or c.exact_text and len(c.exact_text) > 1:
        return False
    q = polys[0]
    sides = [_m.dist(q[i], q[(i + 1) % 4]) for i in range(4)]
    long_side, short_side = max(sides), max(1e-6, min(sides))
    fx0, fy0, fx1, fy1 = c.frames[0] if c.frames else c.box
    bw, bh = c.box[2] - c.box[0], c.box[3] - c.box[1]
    return long_side < 20 and long_side / short_side < 1.6 and (fx1 - fx0) * (fy1 - fy0) > 0.45 * bw * bh


class _Cropper:
    """Crops in candidate coordinates from a PDF page (vector, re-rendered sharp) or an image (pixels)."""

    def __init__(self, page: pymupdf.Page | None = None, image: Image.Image | None = None):
        self.page, self.image = page, image

    def crop(self, box, pad: float = 3, min_side: int = 720, max_side: int = 1200, angle: float = 0.0) -> Image.Image:
        """Crop a candidate; rotate so its text reads horizontally (isometric sheets: +-30..60 deg)."""
        x0, y0, x1, y1 = box
        if self.page is not None:
            clip = pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad) & self.page.rect
            long_pt = max(clip.width, clip.height, 1)
            dpi = int(max(200, min(900, min_side / long_pt * 72)))
            img = render(self.page, dpi, clip=clip)
        else:
            w, h = self.image.size
            p = pad * 3
            img = self.image.crop((max(0, int(x0 - p)), max(0, int(y0 - p)), min(w, int(x1 + p)), min(h, int(y1 + p))))
        if angle:
            img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
            ink = Image.eval(img.convert("L"), lambda v: 255 if v < 200 else 0).getbbox()
            if ink:
                m = 12
                img = img.crop((max(0, ink[0] - m), max(0, ink[1] - m), min(img.width, ink[2] + m), min(img.height, ink[3] + m)))
        if max(img.size) < min_side:
            f = min_side / max(img.size)
            img = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)
        return fit_image(img, max_side)

    def marked(self, pt, half: float = 60) -> Image.Image:
        x, y = pt
        if self.page is not None:
            clip = pymupdf.Rect(x - half, y - half, x + half, y + half) & self.page.rect
            dpi = 220
            img = render(self.page, dpi, clip=clip)
            s = dpi / 72
            cx, cy = (x - clip.x0) * s, (y - clip.y0) * s
        else:
            w, h = self.image.size
            hp = half * 3
            x0, y0 = max(0, int(x - hp)), max(0, int(y - hp))
            img = self.image.crop((x0, y0, min(w, int(x + hp)), min(h, int(y + hp))))
            cx, cy = x - x0, y - y0
        dr = ImageDraw.Draw(img)
        r = max(8, img.width // 25)
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(230, 0, 0), width=max(2, r // 4))
        return fit_image(img, 700)


__all__ = ["GeometryReader", "datum_check", "load_image"]
