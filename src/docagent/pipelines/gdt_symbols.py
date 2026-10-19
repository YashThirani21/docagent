"""Deterministic GD&T symbol recognition on a (deskewed) callout crop.

The 7B VLM reads digits well but does not recognise GD&T symbols, even on clean, horizontal crops
(it called a position symbol "circularity"). The symbols are standard shapes, so we find them with
image topology instead:
  1. frame cells = enclosed regions (contour holes) of the frame grid, grouped into rows
  2. the first (left-most) cell of a row holds the symbol
  3. classify the symbol strokes: enclosed regions, corners, aspect, line directions
Works on renders of vector PDFs and on pictures alike (only pixels are used).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image


def _binary(img: Image.Image) -> np.ndarray:
    import cv2
    g = np.array(img.convert("L"))
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def _unshear(b: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Map the (parallelogram) first cell onto an upright rectangle: isometric frames shear the symbol."""
    import cv2
    c = quad.mean(0)                                   # clockwise order (y down) - never mirrors the content
    order = np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))
    q = quad[order]
    start = int(np.argmin(q.sum(1)))                    # top-left-most corner first
    tl, tr, br, bl = (q[(start + k) % 4] for k in range(4))
    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if W < 8 or H < 8:
        return np.zeros((1, 1), np.uint8)
    M = cv2.getAffineTransform(np.float32([tl, tr, bl]), np.float32([[0, 0], [W, 0], [0, H]]))
    cell = cv2.warpAffine(b, M, (W, H))
    m = max(3, int(0.1 * min(W, H)))                  # drop the cell borders
    cell[:m, :] = 0
    cell[-m:, :] = 0
    cell[:, :m] = 0
    cell[:, -m:] = 0
    return cell


def frame_rows(img: Image.Image) -> list[dict]:
    """Rows of a feature control frame: [{'box': (x0,y0,x1,y1), 'cells': [...], 'symbol': mask}] top->bottom."""
    import cv2
    b = _binary(img)
    h, w = b.shape
    cnts, hier = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    cells = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0:                          # outer contour, not a hole
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if ch < 0.06 * h and ch < 18 or cw < 10 or area < 0.5 * 0.5 * cw * ch:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)   # straight sides only
        if len(approx) != 4 or not (0.4 < cw / ch < 6):
            continue
        if area < 0.9 * cv2.contourArea(approx) or ch < 0.12 * h and ch < 30:
            continue                                   # rounded digit counters (0, 6, 8, D) are not cells
        cells.append({"rect": (x, y, x + cw, y + ch), "contour": c, "quad": approx.reshape(4, 2).astype(np.float32)})
    if not cells:
        return []
    # a frame cell has neighbours of similar height in the same row
    cells.sort(key=lambda c: (c["rect"][1] + c["rect"][3]) / 2)
    rows: list[list[dict]] = []
    for c in cells:
        cy = (c["rect"][1] + c["rect"][3]) / 2
        hgt = c["rect"][3] - c["rect"][1]
        for r in rows:
            ry = np.mean([(x["rect"][1] + x["rect"][3]) / 2 for x in r])
            rh = np.mean([x["rect"][3] - x["rect"][1] for x in r])
            if abs(cy - ry) < 0.35 * rh and 0.6 < hgt / rh < 1.6:
                r.append(c)
                break
        else:
            rows.append([c])
    out = []
    for r in rows:
        if len(r) < 2:                                 # single boxes: datum letters, not frames
            continue
        r.sort(key=lambda c: c["rect"][0])
        first = r[0]
        sym = _unshear(b, first["quad"])
        box = (min(c["rect"][0] for c in r), min(c["rect"][1] for c in r),
               max(c["rect"][2] for c in r), max(c["rect"][3] for c in r))
        out.append({"box": box, "n_cells": len(r), "symbol": sym, "cell": first["rect"]})
    out.sort(key=lambda r: r["box"][1])
    return out


def classify(sym: np.ndarray) -> tuple[str, str]:
    """(characteristic, reason). 'none' for an empty first cell (2nd row of a composite frame)."""
    import cv2
    ys, xs = np.nonzero(sym)
    if len(xs) < 25:
        return "none", "empty first cell"
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    s = sym[y0:y1 + 1, x0:x1 + 1]
    hh, ww = s.shape
    aspect = ww / max(hh, 1)
    cnts, hier = cv2.findContours(s.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outers = [i for i in range(len(cnts)) if hier[0][i][3] < 0 and cv2.contourArea(cnts[i]) > 4 or
              (hier[0][i][3] < 0 and len(cnts[i]) > 6)]
    holes = [i for i in range(len(cnts)) if hier[0][i][3] >= 0 and cv2.contourArea(cnts[i]) > 0.01 * hh * ww]
    n_out, n_hole = len(outers), len(holes)
    lines = cv2.HoughLinesP(s, 1, np.pi / 180, threshold=max(8, int(0.25 * max(hh, ww))),
                            minLineLength=int(0.45 * max(hh, ww)), maxLineGap=3)
    segs = [] if lines is None else np.asarray(lines).reshape(-1, 4)   # OpenCV 4 vs 5 shapes
    angs = [abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180 for x1, y1, x2, y2 in segs]
    horiz = [a for a in angs if a < 12 or a > 168]
    vert = [a for a in angs if 78 < a < 102]
    slant = [a for a in angs if 15 < a < 75 or 105 < a < 165]
    reason = f"holes={n_hole} parts={n_out} aspect={aspect:.2f} lines h{len(horiz)} v{len(vert)} s{len(slant)}"
    if n_hole >= 3:
        return "position", reason                      # circle divided by a cross
    if n_hole == 2:
        big = sorted((cv2.contourArea(cnts[i]) for i in holes), reverse=True)
        if big[1] < 0.5 * big[0] and aspect < 1.3:
            return "concentricity", reason             # ring inside ring
        if aspect > 1.25 and not slant:
            return "profile of a surface", reason      # half-circle split by its base line (AutoCAD style)
        return ("cylindricity" if aspect > 1.25 else "position"), reason
    if n_hole == 1:
        hole = cnts[holes[0]]
        approx = cv2.approxPolyDP(hole, 0.03 * cv2.arcLength(hole, True), True)   # curves keep many vertices
        (_, _), (rw, rh), _ = cv2.minAreaRect(hole)
        hole_aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if len(approx) == 4 and hole_aspect < 4:
            return "flatness", reason                  # parallelogram outline
        if aspect > 1.3 and len(slant) >= 2 and hole_aspect < 1.4:
            return "cylindricity", reason              # circle between two slanted tangents
        if hole_aspect > 1.45 or aspect > 1.45:
            return "profile of a surface", reason      # closed semicircle (flat base)
        return "circularity", reason
    # no enclosed region
    if n_out >= 2 and len(slant) >= 2:
        return "parallelism", reason
    if aspect > 3 and horiz:
        return "straightness", reason
    if horiz and vert:
        return "perpendicularity", reason
    if aspect > 1.4 and not angs:
        return "profile of a line", reason             # open arc
    if len(slant) >= 1 and horiz:
        return "angularity", reason
    if len(slant) >= 2:
        return "parallelism", reason
    return "unknown", reason


def frame_symbols(img: Image.Image) -> list[tuple[str, str]]:
    """Characteristic per frame row (top -> bottom); composite second rows inherit the row above."""
    out, last = [], None
    for r in frame_rows(img):
        ch, why = classify(r["symbol"])
        if ch == "none" and last:
            ch, why = last, "composite row: inherits " + last
        out.append((ch, why))
        if ch not in ("none", "unknown"):
            last = ch
    return out




def datum_letter(img: Image.Image) -> tuple[str | None, float]:
    """Letter inside a datum feature box. The (isometric, sheared) box interior is mapped to an upright
    square first - that also un-shears the letter - then OCR-recognised at the four right angles."""
    import cv2
    b = _binary(img)
    cnts, hier = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None, 0.0
    best_q, best_a = None, 0.0
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        a = cv2.contourArea(c)
        if len(approx) == 4 and a > best_a:
            best_q, best_a = approx.reshape(4, 2).astype(np.float32), a
    if best_q is None:
        return None, 0.0
    cell = _unshear(b, best_q)
    if cell.size < 64 or cell.max() == 0:
        return None, 0.0
    letter = Image.fromarray(255 - cell).convert("RGB")
    w, h = letter.size
    canvas = Image.new("RGB", (w * 3, h * 2), "white")           # recogniser wants some margin
    canvas.paste(letter, (w, h // 2))
    from ..ocr import recognize
    best = (None, 0.0)
    for rot in (0, 90, 270, 180):
        for text, score in recognize(np.array(canvas.rotate(rot, expand=True, fillcolor="white"))):
            t = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
            if len(t) == 1 and score > best[1]:
                best = (t, score)
    return best
