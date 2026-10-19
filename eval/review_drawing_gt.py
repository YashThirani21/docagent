"""Draw a ground-truth file's values on its sheets for human review: python -m eval.review_drawing_gt [drawing ids]

Writes eval/datasets/drawings/review/<drawing_id>_p<page>.png: each value's (approximate) box with its number and
printed text, so a reviewer can tick values off against the sheet. Default: the unverified ground-truth files.
After checking, set "verified": true in the JSON (and fix anything wrong first).
"""
from __future__ import annotations

import json
import sys

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from docagent.config import PROJECT_ROOT

GT_DIR = PROJECT_ROOT / "eval" / "datasets" / "drawings"
OUT = GT_DIR / "review"
DPI = 150


def font(size: int):
    for name in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def review(gt: dict) -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    f, small = font(15), font(13)
    k = DPI / 72
    paths = []
    with pymupdf.open(PROJECT_ROOT / "data" / "drawings" / gt["source_file"]) as doc:
        for pno, page in enumerate(doc, 1):
            pix = page.get_pixmap(dpi=DPI)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGBA")
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            legend = []
            for i, v in enumerate(gt["values"], 1):
                if v["page"] != pno or not v.get("bbox"):
                    continue
                x0, y0, x1, y1 = (c * k for c in v["bbox"])
                col = (150, 150, 150, 255) if v.get("optional") else (220, 30, 30, 255)
                d.rectangle((x0, y0, x1, y1), outline=col, width=2)
                d.rectangle((x0 - 2, y0 - 20, x0 + 26, y0), fill=col)
                d.text((x0 + 2, y0 - 19), str(i), fill="white", font=f)
                legend.append(f"{i:>2}  {v['raw_text']}  [{v['type']}" + (f", {v['characteristic']}" if v.get(
                    "characteristic") else "") + (f", datums {''.join(v['datums'])}" if v.get("datums") else "") +
                    ("" if not v.get("optional") else ", optional") + "]")
            img = Image.alpha_composite(img, layer).convert("RGB")
            # legend strip under the sheet
            strip = Image.new("RGB", (img.width, 22 + 17 * len(legend)), "white")
            ds = ImageDraw.Draw(strip)
            ds.text((8, 3), f"{gt['drawing_id']} sheet {pno} - ground truth ({'VERIFIED' if gt.get('verified') else 'NOT verified'})",
                    fill="black", font=f)
            for j, line in enumerate(legend):
                ds.text((8, 22 + 17 * j), line, fill=(120, 0, 0), font=small)
            out = Image.new("RGB", (img.width, img.height + strip.height), "white")
            out.paste(img, (0, 0))
            out.paste(strip, (0, img.height))
            p = OUT / f"{gt['drawing_id']}_p{pno}.png"
            out.save(p)
            paths.append(p)
    return paths


if __name__ == "__main__":
    ids = sys.argv[1:]
    for p in sorted(GT_DIR.glob("*.json")):
        gt = json.loads(p.read_text())
        if (ids and not any(i in gt["drawing_id"] for i in ids)) or (not ids and gt.get("verified", True)):
            continue
        for out in review(gt):
            print(out.relative_to(PROJECT_ROOT))
