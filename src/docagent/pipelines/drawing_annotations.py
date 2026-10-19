"""Annotation candidates: where the values on a drawing sheet are, before anything is read.

Two front ends produce the same `Candidate` objects; the drawing back end (pipelines/drawing.py) reads
each candidate once, so duplicates from overlapping tiles cannot happen by construction.

- VectorFrontEnd (PDF with vector graphics): the annotation layer is chosen automatically - a PDF
  optional-content layer whose name looks like PMI/annotation, otherwise the stroke colour whose paths
  look most like text and frames (many small multi-segment glyph paths, rectangles/quads). Text comes
  from glyph outlines OR the real text layer. Frames are the layer's rectangles/quads; leaders are its
  straight segments that start at a candidate and end away from it.
- RasterFrontEnd (JPG/PNG/scan): text boxes from RapidOCR's detector (detection only), frames from
  OpenCV contours, leaders from Hough line segments. Coordinates are pixels.

Coordinates are always *displayed* page coordinates: PyMuPDF returns paths and text unrotated, and all
NIST sheets are stored as portrait pages with /Rotate 270, so everything goes through rotation_matrix.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pymupdf
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

Rect = tuple[float, float, float, float]          # x0, y0, x1, y1
Pt = tuple[float, float]


@dataclass
class Leader:
    start: Pt                       # at the annotation
    end: Pt                         # where it points
    terminal: str = "none"          # arrow | dot | none


@dataclass
class Candidate:
    page: int
    box: Rect
    frames: list[Rect] = field(default_factory=list)    # feature-control-frame / datum-box outlines inside
    leaders: list[Leader] = field(default_factory=list)
    exact_text: list[str] = field(default_factory=list)  # text-layer strings inside (exact, when available)
    source: str = "vector"                               # vector | raster
    hint: str = ""                                       # geometry at the leader end, e.g. "circular edge ~Ø20"
    angle: float = 0.0                                   # text baseline direction in degrees (image coords, y down)


# ------------------------------------------------------------------ geometry helpers

def _area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _union(a: Rect, b: Rect) -> Rect:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _grow(r: Rect, g: float) -> Rect:
    return r[0] - g, r[1] - g, r[2] + g, r[3] + g


def _intersects(a: Rect, b: Rect) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _contains(r: Rect, p: Pt) -> bool:
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def _center(r: Rect) -> Pt:
    return (r[0] + r[2]) / 2, (r[1] + r[3]) / 2


def _in_poly(p: Pt, poly: list[Pt]) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > p[1]) != (y2 > p[1]) and p[0] < (x2 - x1) * (p[1] - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def _grow_poly(poly: list[Pt], g: float) -> list[Pt]:
    cx, cy = sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)
    out = []
    for x, y in poly:
        d = math.hypot(x - cx, y - cy) or 1.0
        out.append((x + (x - cx) / d * g, y + (y - cy) / d * g))
    return out


def _polys_touch(a: list[Pt], b: list[Pt], g: float) -> bool:
    ga, gb = _grow_poly(a, g), _grow_poly(b, g)
    return any(_in_poly(p, ga) for p in b) or any(_in_poly(p, gb) for p in a)


def _cluster(rects: list[Rect], gap: float) -> list[list[int]]:
    """Union-find on rectangles closer than `gap`."""
    n = len(rects)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    grown = [_grow(r, gap) for r in rects]
    order = sorted(range(n), key=lambda i: rects[i][0])
    for a_i, i in enumerate(order):                       # sweep on x to stay ~linear on dense sheets
        for j in order[a_i + 1:]:
            if rects[j][0] > grown[i][2]:
                break
            if _intersects(grown[i], rects[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _bbox(rects: list[Rect]) -> Rect:
    b = rects[0]
    for r in rects[1:]:
        b = _union(b, r)
    return b


# ------------------------------------------------------------------ grouping (shared by both front ends)

@dataclass
class _Parts:
    text: list[Rect]                 # glyph / word boxes
    text_strings: list[str | None]   # exact text when known (text layer), else None
    frames: list[list[Pt]]           # frame / datum-box outlines as polygons (often rotated quads)
    segments: list[tuple[Pt, Pt]]    # straight line segments of the annotation layer
    marks: list[Rect]                # arrowheads / dots
    part_curves: list[Rect] = field(default_factory=list)   # closed curves of the part geometry (holes)
    text_angles: list[float | None] = field(default_factory=list)  # per text box (raster detector polygons)
    doc_text: list[Rect] = field(default_factory=list)      # notes / title-block lines
    axis_marks: list[Rect] = field(default_factory=list)    # coordinate-system triads (junk magnets)


def _gap(a: Rect, b: Rect) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def group(parts: _Parts, page: int, source: str, unit: float = 1.0,
          glyph_gap: float = 2.5, callout_gap: float = 7.5) -> list[Candidate]:
    """text -> words -> frames absorb their text -> nearby words/frames form one callout.

    Callout merge rule: groups closer than `callout_gap` merge unless BOTH already have a leader of
    their own - a callout points at one feature, so two leaders mean two callouts (FTC-07 p2 glued a
    profile stack to a flatness stack before this rule)."""
    g_gap, c_gap = glyph_gap * unit, callout_gap * unit
    words = [(_bbox([parts.text[i] for i in grp]),
              list(dict.fromkeys(parts.text_strings[i] for i in grp if parts.text_strings[i])))
             for grp in _cluster(parts.text, g_gap)] if parts.text else []
    items: list[Candidate] = []
    used = set()
    for poly in parts.frames:                     # every frame absorbs the words whose centre lies inside it
        inside = [k for k, (b, _) in enumerate(words) if k not in used and _in_poly(_center(b), poly)]
        if not inside:
            continue
        fr = (min(p[0] for p in poly), min(p[1] for p in poly), max(p[0] for p in poly), max(p[1] for p in poly))
        box, texts = fr, []
        for k in inside:
            box = _union(box, words[k][0])
            texts += words[k][1]
            used.add(k)
        c = Candidate(page=page, box=box, frames=[fr], exact_text=texts, source=source)
        c.polys = [poly]
        items.append(c)
    for k, (b, t) in enumerate(words):
        if k not in used:
            c = Candidate(page=page, box=b, exact_text=t, source=source)
            c.polys = []
            items.append(c)
    # frames that touch (stacked composite rows, symbol cell) belong together - by real shape, not bbox
    def touch(a, b):
        return bool(a.polys) and bool(b.polys) and any(_polys_touch(x, y, 1.0 * unit) for x in a.polys for y in b.polys)
    items = _merge_where(items, touch)
    items = [c for c in items if not _junk(c, parts, unit)]
    for c in items:
        c.leaders = _leaders(c, items, parts, unit)
    # constrained callout merge, closest pairs first
    pairs = sorted((_gap(items[i].box, items[j].box), i, j) for i in range(len(items)) for j in range(i + 1, len(items))
                   if _gap(items[i].box, items[j].box) < c_gap)
    parent = list(range(len(items)))
    has_leader = [bool(c.leaders) for c in items]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for _, i, j in pairs:
        a, b = find(i), find(j)
        if a != b and not (has_leader[a] and has_leader[b]):
            parent[b] = a
            has_leader[a] = has_leader[a] or has_leader[b]
    groups: dict[int, list[Candidate]] = {}
    for i, c in enumerate(items):
        groups.setdefault(find(i), []).append(c)
    cands = []
    for g in groups.values():
        c = Candidate(page=page, box=_bbox([x.box for x in g]), frames=[f for x in g for f in x.frames],
                      exact_text=[t for x in g for t in x.exact_text], source=source)
        c.polys = [q for x in g for q in getattr(x, "polys", [])]
        cands.append(c)
    # fragments (frame cells split off a stack) fold into the much larger candidate they overlap
    cands = _merge_where(cands, lambda a, b: _intersects(a.box, b.box)
                         and min(_area(a.box), _area(b.box)) < 0.2 * max(_area(a.box), _area(b.box)))
    for c in cands:
        c.leaders = _leaders(c, cands, parts, unit)
        c.hint = _hint(c, parts, unit)
        c.angle = _text_angle(c, parts)
    return cands


def _text_angle(c: Candidate, parts: _Parts) -> float:
    """Baseline direction: a frame's long edge if there is one, else the principal axis of the text centres."""
    polys = getattr(c, "polys", [])
    if polys:
        q = max(polys, key=lambda q: math.dist(q[0], q[1]) + math.dist(q[1], q[2]))
        e1 = (q[1][0] - q[0][0], q[1][1] - q[0][1])
        e2 = (q[2][0] - q[1][0], q[2][1] - q[1][1])
        vx, vy = e1 if math.hypot(*e1) >= math.hypot(*e2) else e2
    else:
        if parts.text_angles:
            angs = [a for r, a in zip(parts.text, parts.text_angles) if a is not None and _contains(c.box, _center(r))]
            if angs:
                return float(np.median(angs))
        pts = np.array([_center(r) for r in parts.text if _contains(c.box, _center(r))])
        if len(pts) < 3:
            return 0.0
        pts = pts - pts.mean(0)
        w, v = np.linalg.eigh(pts.T @ pts)
        if w[1] < 3 * max(w[0], 1e-9):                    # no clear direction (single glyph, square block)
            return 0.0
        vx, vy = v[:, 1]
    a = math.degrees(math.atan2(vy, vx))
    while a > 90:
        a -= 180
    while a <= -90:
        a += 180
    return a if abs(a) > 3 else 0.0


def _merge_where(items: list[Candidate], pred) -> list[Candidate]:
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if pred(items[i], items[j]):
                    a, b = items[i], items.pop(j)
                    a.box = _union(a.box, b.box)
                    a.frames += b.frames
                    a.exact_text += b.exact_text
                    a.polys = getattr(a, "polys", []) + getattr(b, "polys", [])
                    changed = True
                    break
            if changed:
                break
    return items


AXIS_LETTERS = {"X", "Y", "Z"}


def _junk(c: Candidate, parts: _Parts, unit: float) -> bool:
    if any(_contains(t, _center(c.box)) for t in parts.doc_text):
        return True                                            # sits on a notes / title-block line
    if c.frames:
        return False
    if c.exact_text and all(t.strip().upper() in AXIS_LETTERS for t in c.exact_text):
        return True                                            # coordinate-triad labels
    if not c.exact_text and _area(c.box) < 20 * unit * unit:   # leader dots, stray strokes
        return True
    near_axis = any(_intersects(_grow(c.box, 12 * unit), m) for m in parts.axis_marks)
    return near_axis and not c.exact_text and _area(c.box) < 120 * unit * unit   # triad arrowheads/labels


def _endpoint_index(parts: _Parts, cell: float) -> dict:
    idx: dict = {}
    for k, (a, b) in enumerate(parts.segments):
        for pt, other in ((a, b), (b, a)):
            idx.setdefault((int(pt[0] // cell), int(pt[1] // cell)), []).append((pt, other, k))
    return idx


def _leaders(c: Candidate, cands: list[Candidate], parts: _Parts, unit: float) -> list[Leader]:
    near = _grow(c.box, 3 * unit)
    others = [_grow(o.box, 1 * unit) for o in cands if o is not c]
    tol = 1.2 * unit
    cell = 4 * unit
    if getattr(parts, "_idx_cell", None) != cell:
        parts._idx, parts._idx_cell = _endpoint_index(parts, cell), cell
    idx = parts._idx
    out = []
    axis = [_grow(m, 4 * unit) for m in parts.axis_marks]
    for k0, (p, q) in enumerate(parts.segments):
        pin, qin = _contains(near, p), _contains(near, q)
        if pin == qin:
            continue
        if any(_contains(a, p) or _contains(a, q) for a in axis):
            continue                                       # coordinate-triad axis, not a leader
        start, end = (p, q) if pin else (q, p)
        seen = {k0}
        for _ in range(6):                                     # follow polyline leaders (shoulder + slope)
            nxt = None
            cx, cy = int(end[0] // cell), int(end[1] // cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for pt, other, k in idx.get((cx + dx, cy + dy), []):
                        if k not in seen and math.dist(pt, end) < tol:
                            nxt = other
                            seen.add(k)
                            break
                    if nxt is not None:
                        break
                if nxt is not None:
                    break
            if nxt is None or _contains(near, nxt):
                break
            end = nxt
        if math.dist(start, end) < 8 * unit or any(_contains(o, end) for o in others):
            continue                                           # frame divider / line to another callout
        term = "none"
        for m in parts.marks:
            if _contains(_grow(m, 2 * unit), end):
                term = "dot" if _area(m) < 3 * unit * unit else "arrow"
                break
        out.append(Leader(start=start, end=end, terminal=term))
    out.sort(key=lambda l: (l.terminal == "none", -math.dist(l.start, l.end)))
    return out[:3]


def _hint(c: Candidate, parts: _Parts, unit: float) -> str:
    if not c.leaders:
        return ""
    end = c.leaders[0].end
    for r in parts.part_curves:
        if _contains(_grow(r, 4 * unit), end):
            d = max(r[2] - r[0], r[3] - r[1])
            return f"the leader ends on a circular/elliptical edge about {d / unit:.0f} pt across (hole or round boss)"
    return "the leader ends on a straight edge or a face" if c.leaders[0].terminal != "dot" \
        else "the leader ends with a dot on a face/surface"


def _thin_polygon_segment(d, m) -> tuple[Pt, Pt] | None:
    """A filled polygon that is long and thin is a line drawn with width (AutoCAD export)."""
    pts = []
    for it in d["items"]:
        if it[0] == "l":
            pts += [pymupdf.Point(it[1]) * m, pymupdf.Point(it[2]) * m]
    if len(pts) < 4:
        return None
    a, b = max(((p, q) for i, p in enumerate(pts) for q in pts[i + 1:]), key=lambda pq: math.dist(pq[0], pq[1]))
    length = math.dist(a, b)
    if length < 15:                                   # arrowheads (~5-8 pt) stay marks: they mark leader ends
        return None
    # polygon area via shoelace on the item path
    poly = [pts[0]] + pts[1::2]
    area = abs(sum(poly[i].x * poly[(i + 1) % len(poly)].y - poly[(i + 1) % len(poly)].x * poly[i].y
                   for i in range(len(poly)))) / 2
    return ((a.x, a.y), (b.x, b.y)) if area / length < 1.2 else None


# ------------------------------------------------------------------ vector front end

TITLE_LINE_RE = re.compile(r"^(NOTES?\b.*|Rev\.?\s+\w{1,2}|.*Test Case \d+.*|Includes Atomic.*|No Atomic.*|NIST PMI.*|"
                           r"nist_[a-z]{3}_\d{2}\w*|Feature and Specification Index|\S+_fsi\.pdf|Test Model \d+)$", re.I)


def _color_key(c) -> tuple | None:
    return tuple(round(v, 2) for v in c) if c else None


class VectorFrontEnd:
    """Annotation candidates from a PDF page's vector graphics (+ its text layer)."""

    def __init__(self, min_score: int = 15):
        self.min_score = min_score

    def candidates(self, page: pymupdf.Page, pno: int, known_text: set[str]) -> list[Candidate] | None:
        m = page.rotation_matrix
        known_text = {k for k in known_text if len(k) >= 3}   # "D" (a revision) must not hide datum D
        drawings = page.get_drawings()
        layer_ok = self._annotation_layer(drawings)
        color = self._annotation_color(drawings) if layer_ok is None else None
        if layer_ok is None and color is None:
            return None                                       # no separable annotation layer -> raster route

        def ours(d) -> bool:
            if layer_ok is not None:
                return (d.get("layer") or "") in layer_ok
            return _color_key(d.get("color")) == color or _color_key(d.get("fill")) == color

        parts = _Parts([], [], [], [], [])
        for d in drawings:
            r = pymupdf.Rect(d["rect"]) * m
            rect = (r.x0, r.y0, r.x1, r.y1)
            if not ours(d):
                ck = _color_key(d.get("color"))               # strokes only: shading fills are saturated too
                if ck and d["type"] in ("s", "fs") and max(ck) - min(ck) > 0.45 and _area(rect) < 2500 \
                        and len(d["items"]) <= 3:
                    parts.axis_marks.append(rect)              # small saturated strokes: axis triads
                if any(it[0] == "c" for it in d["items"]) and 3 < max(r.width, r.height) < 120 \
                        and 0.15 < r.width / max(r.height, 1e-6) < 6:
                    parts.part_curves.append(rect)            # curved part edges: hole / boss outlines
                continue
            items = d["items"]
            # glyph outlines are stroked (+filled); pure fills are arrowheads, dots or wide lines
            if d["type"] == "fs" or (d["type"] == "s" and len(items) > 5 and max(r.width, r.height) < 20):
                parts.text.append(rect)
                parts.text_strings.append(None)
            elif d["type"] == "s":
                for it in items:
                    if it[0] == "l":
                        p, q = pymupdf.Point(it[1]) * m, pymupdf.Point(it[2]) * m
                        parts.segments.append(((p.x, p.y), (q.x, q.y)))
                    elif it[0] in ("qu", "re"):
                        qr = (pymupdf.Quad(it[1]) if it[0] == "qu" else pymupdf.Rect(it[1]).quad) * m
                        b = qr.rect
                        if b.width > 4 and b.height > 4:
                            parts.frames.append([(qr.ul.x, qr.ul.y), (qr.ur.x, qr.ur.y), (qr.lr.x, qr.lr.y), (qr.ll.x, qr.ll.y)])
            else:
                seg = _thin_polygon_segment(d, m)
                if seg:
                    parts.segments.append(seg)
                else:
                    parts.marks.append(rect)
        # real text (e.g. AutoCAD exports). Per-character boxes: rotated text lines have huge, overlapping
        # axis-aligned line boxes that would chain every callout together.
        lines = []
        for b in page.get_text("rawdict")["blocks"]:
            for ln in b.get("lines", []):
                chars = [ch for sp in ln["spans"] for ch in sp["chars"]]
                t = "".join(ch["c"] for ch in chars).strip()
                if t:
                    r = pymupdf.Rect(ln["bbox"]) * m
                    lines.append((t, (r.x0, r.y0, r.x1, r.y1), chars))
        doc_idx = set()
        for i, (t, r, _) in enumerate(lines):            # notes / title block: not annotations
            n = _norm(t)
            if len(t) > 60 or n in known_text or TITLE_LINE_RE.search(t) or (len(n) >= 12 and any(n in k for k in known_text)):
                doc_idx.add(i)
        for i, (t, r, _) in enumerate(lines):            # short note fragments ("SPECIFIED.") next to notes lines
            n = _norm(t)
            if i not in doc_idx and len(n) >= 4 and any(n in k for k in known_text) \
                    and any(_gap(r, lines[j][1]) < 6 for j in doc_idx):
                doc_idx.add(i)
        for i, (t, r, chars) in enumerate(lines):
            if i in doc_idx:
                parts.doc_text.append(r)
                continue
            for ch in chars:
                    if ch["c"].strip():
                        cr = pymupdf.Rect(ch["bbox"]) * m
                        parts.text.append((cr.x0, cr.y0, cr.x1, cr.y1))
                        parts.text_strings.append(t)
        if not parts.text:
            return None
        return group(parts, pno, "vector")

    @staticmethod
    def _annotation_layer(drawings) -> set[str] | None:
        names = {d.get("layer") for d in drawings if d.get("layer")}
        hit = {n for n in names if re.search(r"pmi|annot|dimension|gd&?t|toleranc|callout", n, re.I)}
        return hit or None

    def _annotation_color(self, drawings) -> tuple | None:
        """The colour whose paths look most like annotation text and frames."""
        score: Counter = Counter()
        for d in drawings:
            ck = _color_key(d.get("color")) or _color_key(d.get("fill"))
            if ck is None:
                continue
            r = d["rect"]
            if (d["type"] == "fs" or len(d["items"]) > 5) and max(r.width, r.height) < 20:
                score[ck] += 1                                 # glyph-like
            if any(it[0] in ("qu", "re") for it in d["items"]) and d["type"] == "s":
                score[ck] += 3                                 # frame-like outline
        if not score:
            return None
        ck, s = score.most_common(1)[0]
        log.info("annotation colour %s (score %d)", ck, s)
        return ck if s >= self.min_score else None


# ------------------------------------------------------------------ raster front end

def load_image(path: str) -> Image.Image:
    """Picture input: honour EXIF orientation (phone photos), RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _rectify(img: np.ndarray) -> np.ndarray:
    """Photo of a sheet: if a large 4-corner contour (the sheet border) is found, warp it flat."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = g.shape
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.5 * w * h:
            pts = approx.reshape(4, 2).astype(np.float32)
            s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
            src = np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)
            x_axis = np.linalg.norm(src[1] - src[0])
            skew = abs(src[0][1] - src[1][1]) + abs(src[0][0] - src[3][0])
            if skew < 0.01 * w:                                # already a flat scan / render
                return img
            W, H = int(x_axis), int(np.linalg.norm(src[3] - src[0]))
            dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
            return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H))
    return img


class RasterFrontEnd:
    """Annotation candidates from pixels: text boxes (RapidOCR detector), frames, leaders (OpenCV)."""

    def candidates(self, img: Image.Image, pno: int) -> tuple[list[Candidate], Image.Image]:
        import cv2
        arr = _rectify(np.array(img))
        g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        from ..ocr import detect
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        H, W = g.shape
        polys = []
        n, ov = 3, 0.15                       # overlapping tiles: small rotated annotation text is missed otherwise
        for r in range(n):
            for c in range(n):
                x0, x1 = int(max(0, (c - ov) / n * W)), int(min(W, (c + 1 + ov) / n * W))
                y0, y1 = int(max(0, (r - ov) / n * H)), int(min(H, (r + 1 + ov) / n * H))
                polys += [p + [x0, y0] for p in detect(np.ascontiguousarray(rgb[y0:y1, x0:x1]),
                                                        box_thresh=0.3, thresh=0.2)]
        text, angles = [], []
        for pts in polys:
            text.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
            (x0, y0), (x1, y1) = pts[0], pts[1]                 # detector polygon: first edge = baseline direction
            a = math.degrees(math.atan2(y1 - y0, x1 - x0))
            angles.append(((a + 90) % 180) - 90)
        if not text:
            return [], Image.fromarray(arr)
        heights = sorted(min(r[2] - r[0], r[3] - r[1]) for r in text)
        med = heights[len(heights) // 2]
        unit = max(1.0, med / 7.0)                             # px per "pt": annotation text is ~7 pt tall
        keep = [k for k, r in enumerate(text) if min(r[2] - r[0], r[3] - r[1]) < 1.8 * med]   # titles are bigger
        text, angles = [text[k] for k in keep], [angles[k] for k in keep]
        binary = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
        frames = []
        cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            if len(approx) == 4 and 6 * unit < max(w, h) < min(250 * unit, 0.3 * W) and min(w, h) > 5 * unit \
                    and w * h < 0.03 * W * H and cv2.contourArea(approx) > 0.25 * w * h:
                frames.append([(float(px), float(py)) for px, py in approx.reshape(4, 2)])
        # leaders: line segments outside the text boxes
        mask = binary.copy()
        for r in text:
            x0, y0, x1, y1 = (int(v) for v in _grow(r, 1))
            mask[max(0, y0):y1, max(0, x0):x1] = 0
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=int(30), minLineLength=int(8 * unit), maxLineGap=int(2 * unit))
        segs = [((float(a), float(b)), (float(c), float(d))) for a, b, c, d in (lines.reshape(-1, 4) if lines is not None else [])]
        parts = _Parts(text=text, text_strings=[None] * len(text), frames=frames, segments=segs, marks=[],
                       text_angles=angles)
        cands = group(parts, pno, "raster", unit=unit, glyph_gap=1.0, callout_gap=4.0)
        return cands, Image.fromarray(arr)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


def overlay(img: Image.Image, cands: list[Candidate], scale: float) -> Image.Image:
    """Debug view: candidate boxes (red), frames (orange), leaders (green)."""
    from PIL import ImageDraw
    img = img.copy()
    dr = ImageDraw.Draw(img)
    for i, c in enumerate(cands):
        b = [v * scale for v in c.box]
        dr.rectangle(b, outline=(230, 0, 0), width=2)
        dr.text((b[0], b[1] - 11), str(i), fill=(230, 0, 0))
        for f in c.frames:
            dr.rectangle([v * scale for v in f], outline=(255, 140, 0), width=1)
        for l in c.leaders:
            dr.line([l.start[0] * scale, l.start[1] * scale, l.end[0] * scale, l.end[1] * scale], fill=(0, 170, 0), width=2)
            dr.ellipse([l.end[0] * scale - 3, l.end[1] * scale - 3, l.end[0] * scale + 3, l.end[1] * scale + 3], outline=(0, 170, 0))
    return img
