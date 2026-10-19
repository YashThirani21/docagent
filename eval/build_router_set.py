"""Generate the router eval set from the source documents: python -m eval.build_router_set

Every case's expected route follows from how it was made, so nothing is hand-labelled:
  drawing  - the 11 drawing PDFs; single sheets as 1-page PDFs; sheets rendered as pictures with rotation, blur,
             JPEG artefacts, perspective, crop and a "photo of a printout" look (grey background, uneven light, noise)
  report_qa - the 3 report PDFs; report pages as 1-page PDFs and as pictures (text pages, figure pages); report
             figures cropped WITHOUT their caption (stand-ins for part photos), augmented; the 18-061 load charts;
             text-only questions
Split: cases that are the 14 original PDFs (the routing rules were written looking at them) are dev; everything
generated is test. Deterministic (fixed seeds): re-running rewrites identical files.
Output: eval/datasets/router/cases.jsonl + files/ (pictures ~100-150 dpi: the router classifies at 768 px).
Real phone photos: eval/datasets/router/real/ (see its README) - picked up by the suite, not by this script.
"""
from __future__ import annotations

import io
import json
import random
import re
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

from docagent.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "eval" / "datasets" / "router"
DRAWINGS = sorted((PROJECT_ROOT / "data" / "drawings").glob("*.pdf"))
REPORTS = sorted((PROJECT_ROOT / "data" / "reports").glob("*.pdf"))
CAPTION_RE = re.compile(r"^\s*Figure\s+(\d+)\s*[.:]", re.I)
AUGS = ["clean", "rotate", "blur", "jpeg", "perspective", "crop", "photo"]
TEXT_QUESTIONS = [
    "What caused the throttle arm to fracture?",
    "Which bolts had the lowest loosening torque on the right spar?",
    "Is there a drawing of the spar attachment?",                 # mentions 'drawing' but is a question
    "What is the tolerance on a 2024-T3511 extrusion?",
    "hello",
    "Summarise report 16-006.",
]


def rel(p: Path) -> str:
    return str(p.relative_to(PROJECT_ROOT))


def render(page: pymupdf.Page, dpi: int, clip=None) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def augment(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    if kind == "rotate":
        return img.rotate(rng.choice([-1, 1]) * rng.uniform(3, 12), expand=True, fillcolor=(255, 255, 255),
                          resample=Image.BICUBIC)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(rng.uniform(1.2, 2.2)))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(12, 25))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "perspective":
        return _perspective(img, rng, 0.10, (255, 255, 255))
    if kind == "crop":
        w, h = img.size
        f = rng.uniform(0.55, 0.75)
        x0, y0 = rng.randint(0, int(w * (1 - f))), rng.randint(0, int(h * (1 - f)))
        return img.crop((x0, y0, x0 + int(w * f), y0 + int(h * f)))
    if kind == "photo":                     # printout on a desk: skew, grey surround, light falloff, sensor noise
        img = _perspective(img, rng, 0.08, (110, 105, 95))
        a = np.asarray(img).astype(np.float32)
        h, w = a.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        light = 1.0 - 0.35 * (((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        a = a * light[..., None] * rng.uniform(0.8, 0.95)
        a += np.random.default_rng(rng.randint(0, 2**31)).normal(0, 6, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    return img


def _perspective(img: Image.Image, rng: random.Random, amount: float, fill) -> Image.Image:
    w, h = img.size
    pad = int(max(w, h) * amount)
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), fill)
    canvas.paste(img, (pad, pad))
    W, H = canvas.size
    j = lambda: rng.uniform(0, pad * 1.6)
    dst = [(j(), j()), (W - j(), j()), (W - j(), H - j()), (j(), H - j())]     # where the corners land
    src = [(0, 0), (W, 0), (W, H), (0, H)]
    coeffs = _coeffs(dst, src)
    return canvas.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=fill)


def _coeffs(pa, pb):
    m = []
    for (x, y), (u, v) in zip(pa, pb):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    return np.linalg.solve(np.array(m, dtype=float), np.array(pb, dtype=float).reshape(8)).tolist()


def figure_rects(page: pymupdf.Page) -> list[tuple[int, pymupdf.Rect]]:
    """Each captioned figure: the area above its caption (down to just above the caption text), shrunk to the
    drawn content. The caption itself is excluded - a user's photo has none."""
    blocks = sorted((b for b in page.get_text("blocks")), key=lambda b: b[1])
    caps = [b for b in blocks if CAPTION_RE.match(" ".join(b[4].split()))]
    header = 95.0 if page.rect.height > 700 else 0.0
    out, prev = [], header
    for c in caps:
        area = pymupdf.Rect(0, prev, page.rect.width, c[1] - 2)
        boxes = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info() if pymupdf.Rect(i["bbox"]).intersects(area)]
        if boxes:
            r = boxes[0]
            for b in boxes[1:]:
                r |= b
            area = r & area
        if area.height > 60 and area.width > 60:
            out.append((int(CAPTION_RE.match(" ".join(c[4].split())).group(1)), area))
        prev = c[3] + 4
    return out


def save_img(img: Image.Image, name: str, max_side: int = 1100) -> Path:
    img = img.convert("RGB")
    s = min(1.0, max_side / max(img.size))
    if s < 1:
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    p = OUT / "files" / name
    img.save(p, "JPEG", quality=85)
    return p


def save_page_pdf(src: Path, pno: int, name: str) -> Path:
    p = OUT / "files" / name
    with pymupdf.open(src) as doc, pymupdf.open() as one:
        one.insert_pdf(doc, from_page=pno, to_page=pno)
        one.save(p, garbage=4, deflate=True)
    return p


def build() -> list[dict]:
    (OUT / "files").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "files").glob("*"):
        old.unlink()
    rng = random.Random(7)
    cases: list[dict] = []

    def add(cid, expected, made, path=None, text=None, split="test"):
        cases.append({"id": cid, "expected": expected, "made": made, "split": split,
                      **({"file": rel(path)} if path else {}), **({"text": text} if text else {})})

    # drawings: originals (dev), sheet 1 of each as a picture with one augmentation each, extra sheets
    for d in DRAWINGS:
        add(f"drw-pdf-{d.stem[5:11]}", "drawing", "original drawing PDF", d, split="dev")
    k = 0
    for d in DRAWINGS:
        with pymupdf.open(d) as doc:
            for pno in range(len(doc)):
                if pno > 1:
                    continue                                  # sheets 1-2: enough variety, keeps the set small
                aug = AUGS[k % len(AUGS)]
                k += 1
                img = augment(render(doc[pno], 110), aug, rng)
                p = save_img(img, f"drw_{d.stem[5:11]}_s{pno + 1}_{aug}.jpg")
                add(f"drw-img-{d.stem[5:11]}-s{pno + 1}", "drawing", f"sheet {pno + 1} rendered, {aug}", p)
                if pno == 1:
                    q = save_page_pdf(d, pno, f"drw_{d.stem[5:11]}_s2.pdf")
                    add(f"drw-page-{d.stem[5:11]}-s2", "drawing", "sheet 2 as a 1-page PDF", q)

    # reports: originals (dev), single pages as PDFs and pictures, figure crops, charts
    for r in REPORTS:
        add(f"rep-pdf-{r.stem[5:]}", "report_qa", "original report PDF", r, split="dev")
    page_picks = {"ntsb_16-006": [1, 3, 7, 12], "ntsb_18-061": [2, 6, 14, 25], "ntsb_24-013": [3, 4]}
    for r in REPORTS:
        with pymupdf.open(r) as doc:
            for pno in page_picks[r.stem]:
                page = doc[pno - 1]
                kind = "figure page" if any(CAPTION_RE.match(" ".join(b[4].split()))
                                            for b in page.get_text("blocks")) else "text page"
                aug = rng.choice(["clean", "rotate", "jpeg", "photo"])
                p = save_img(augment(render(page, 100), aug, rng), f"rep_{r.stem[5:]}_p{pno}_{aug}.jpg")
                add(f"rep-img-{r.stem[5:]}-p{pno}", "report_qa", f"report {kind} as a picture, {aug}", p)
                if pno in page_picks[r.stem][:2]:
                    q = save_page_pdf(r, pno - 1, f"rep_{r.stem[5:]}_p{pno}.pdf")
                    add(f"rep-page-{r.stem[5:]}-p{pno}", "report_qa", f"report {kind} as a 1-page PDF", q)
    fig_picks = {"ntsb_16-006": [3, 7, 12], "ntsb_18-061": [2, 5, 6, 8, 13, 15, 19, 23], "ntsb_24-013": [2, 3, 4]}
    for r in REPORTS:
        with pymupdf.open(r) as doc:
            for pno, page in enumerate(doc, 1):
                for fig, rect in figure_rects(page):
                    if fig not in fig_picks[r.stem]:
                        continue
                    aug = rng.choice(["clean", "rotate", "blur", "jpeg", "crop", "photo"])
                    p = save_img(augment(render(page, 150, clip=rect), aug, rng), f"fig_{r.stem[5:]}_f{fig}_{aug}.jpg")
                    add(f"fig-{r.stem[5:]}-f{fig}", "report_qa", f"report figure without caption, {aug}", p)
    with pymupdf.open(PROJECT_ROOT / "data" / "reports" / "ntsb_18-061.pdf") as doc:
        for pno in (37, 39):
            p = save_img(augment(render(doc[pno - 1], 100), "clean", rng), f"chart_18-061_p{pno}.jpg")
            add(f"chart-18-061-p{pno}", "report_qa", "load-extension chart page as a picture", p)
        q = save_page_pdf(PROJECT_ROOT / "data" / "reports" / "ntsb_18-061.pdf", 37, "chart_18-061_p38.pdf")
        add("chart-18-061-p38-pdf", "report_qa", "chart page as a 1-page PDF", q)
    for i, t in enumerate(TEXT_QUESTIONS, 1):
        add(f"text-{i:02d}", "report_qa", "text only", text=t, split="dev" if i <= 3 else "test")

    (OUT / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    return cases


if __name__ == "__main__":
    cs = build()
    size = sum(p.stat().st_size for p in (OUT / "files").glob("*")) / 1e6
    by = {}
    for c in cs:
        by[(c["split"], c["expected"])] = by.get((c["split"], c["expected"]), 0) + 1
    print(f"{len(cs)} cases, files {size:.1f} MB: {by}")
