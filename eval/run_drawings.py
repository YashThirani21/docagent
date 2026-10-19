"""Score drawing extraction against hand-labelled ground truth (eval/datasets/drawings/*.json).

  # score JSONs that already exist (e.g. the bulk run downloaded from Colab)
  python -m eval.run_drawings --pred-dir eval/results/drawings --label baseline-tiles

  # run the extractor on the ground-truth drawings, then score
  python -m eval.run_drawings --run --config configs/colab_vllm.yaml --method geometry --label geo
  python -m eval.run_drawings --run --method geometry --as-images --label geo-jpg   # picture route
  python -m eval.run_drawings --pred-dir ... --compare eval/results/<earlier>_drawings_*.json

Matching: per sheet, every predicted value is paired one-to-one with the most similar ground-truth value
(similarity of the printed text after normalising symbols, or identical numbers). Title-block and note
entries are scored separately.

Metrics
  recall        required ground-truth values found
  precision     predicted values that match a ground-truth value (incl. optional ones)
  numbers       matched values whose nominal/limits/tolerances/quantity agree (where labelled)
  type          matched values with the right coarse type (dimension-like / frame / datum / other)
  gdt_symbol    matched frames with the right geometric characteristic
  datums        matched frames with the right datum letters
  belongs_to    matched values whose belongs_to names an acceptable feature (labelled keywords)
  title/notes/units  document-level fields (PDF route)
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import tempfile
import time
from pathlib import Path

from docagent.config import PROJECT_ROOT, load_config

GT_DIR = PROJECT_ROOT / "eval" / "datasets" / "drawings"
RESULTS = PROJECT_ROOT / "eval" / "results"
DOC_TYPES = {"title_block", "note"}
_SYM = str.maketrans({"Φ": "Ø", "⌀": "Ø", "ø": "Ø", "−": "-", "–": "-", "×": "X", "x": "X"})


def core(s: str) -> str:
    s = s.translate(_SYM).upper()
    return re.sub(r"[^0-9.+\-±/A-Z]", "", s)


def nums(s: str) -> list[float]:
    out = []
    for n in re.findall(r"\d*\.\d+|\d+", s.translate(_SYM)):
        try:
            out.append(round(float(n), 6))
        except ValueError:
            pass
    return out


def similarity(p: str, g: str) -> float:
    a, b = core(p), core(g)
    if not a or not b:
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    na, nb = nums(p), nums(g)
    if na and na == nb:
        r = max(r, 0.9)
    return r


def coarse(t: str) -> str:
    if t in ("dimension", "diameter", "radius", "angle", "tolerance", "chamfer", "thread"):
        return "dim"
    if t == "gdt_frame":
        return "frame"
    if t in ("datum_feature", "datum_target"):
        return "datum"
    return "other"


def limits(v: dict) -> tuple | None:
    lo, hi = v.get("lower_limit", v.get("lower")), v.get("upper_limit", v.get("upper"))
    if lo is not None and hi is not None:
        return round(lo, 6), round(hi, 6)
    n, tp, tm = v.get("nominal"), v.get("tol_plus"), v.get("tol_minus")
    if n is not None and tp is not None and tm is not None:
        return round(n + tm, 6), round(n + tp, 6)
    return None


def numbers_ok(p: dict, g: dict) -> bool | None:
    gl = limits(g)
    checks = []
    if gl is not None:
        checks.append(limits(p) == gl)
    elif g.get("nominal") is not None:
        checks.append(p.get("nominal") is not None and abs(p["nominal"] - g["nominal"]) < 1e-6)
    if g.get("quantity"):
        checks.append(p.get("quantity") == g["quantity"])
    return all(checks) if checks else None


def match(preds: list[dict], gts: list[dict], thr: float = 0.7) -> list[tuple[int, int]]:
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if p["page"] == g["page"]:
                s = similarity(p["raw_text"], g["raw_text"])
                if s >= thr:
                    pairs.append((s, i, j))
    used_p, used_g, out = set(), set(), []
    for s, i, j in sorted(pairs, reverse=True):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j))
    return out


def score_drawing(pred: dict, gt: dict) -> dict:
    preds = [v for v in pred["values"] if v["type"] not in DOC_TYPES]
    gts = gt["values"]
    pairs = match(preds, gts)
    req = [j for j, g in enumerate(gts) if not g.get("optional")]
    matched_g = {j for _, j in pairs}
    rows, num, typ, sym, dat, bel = [], [], [], [], [], []
    for i, j in pairs:
        p, g = preds[i], gts[j]
        n_ok = numbers_ok(p, g)
        if n_ok is not None:
            num.append(n_ok)
        typ.append(coarse(p["type"]) == coarse(g["type"]))
        if g["type"] == "gdt_frame":
            sym.append((p.get("characteristic") or "") == g.get("characteristic"))
            dat.append(sorted(p.get("datums") or []) == sorted(g.get("datums") or []))
        if g.get("belongs"):
            b = (p.get("belongs_to") or "").lower()
            bel.append(any(k in b for k in g["belongs"]))
        rows.append({"pred": p["raw_text"], "gt": g["raw_text"], "page": g["page"], "numbers": n_ok,
                     "type_ok": typ[-1], "pred_char": p.get("characteristic"), "gt_char": g.get("characteristic"),
                     "belongs_to": p.get("belongs_to")})
    missed = [gts[j]["raw_text"] + f" (p{gts[j]['page']})" for j in req if j not in matched_g]
    extra = [preds[i]["raw_text"] + f" (p{preds[i]['page']})" for i in range(len(preds)) if i not in {a for a, _ in pairs}]
    tb_gt = gt.get("title_block", {})
    tb_ok = [pred.get("title_block", {}).get(k) == v for k, v in tb_gt.items()]

    def rate(x):
        return round(sum(x) / len(x), 3) if x else None
    return {
        "drawing": gt["drawing_id"], "n_pred": len(preds), "n_gt_required": len(req),
        "recall": rate([j in matched_g for j in req]), "precision": rate([True] * len(pairs) + [False] * len(extra)),
        "numbers": rate(num), "type": rate(typ), "gdt_symbol": rate(sym), "datums": rate(dat), "belongs_to": rate(bel),
        "title_block": rate(tb_ok), "notes": len(pred.get("notes", [])) == gt.get("n_notes"),
        "units": pred.get("units") == gt.get("units"),
        "missed": missed, "extra": extra, "pairs": rows,
    }


def aggregate(per: list[dict]) -> dict:
    keys = ["recall", "precision", "numbers", "type", "gdt_symbol", "datums", "belongs_to", "title_block"]
    out = {}
    for k in keys:      # micro-average weighted by the counts behind each rate
        w = {"recall": "n_gt_required", "precision": "n_pred"}.get(k)
        vals = [(d[k], d[w] if w else len(d["pairs"])) for d in per if d[k] is not None]
        tot = sum(n for _, n in vals)
        out[k] = round(sum(v * n for v, n in vals) / tot, 3) if tot else None
    out["notes"] = round(sum(d["notes"] for d in per) / len(per), 3)
    out["units"] = round(sum(d["units"] for d in per) / len(per), 3)
    out["n_pred"] = sum(d["n_pred"] for d in per)
    out["n_gt_required"] = sum(d["n_gt_required"] for d in per)
    return out


def run_extractor(cfg, method: str, as_images: bool, out_dir: Path, gts: list[dict]) -> dict[str, dict]:
    from docagent.pipelines.drawing import DrawingExtractor
    from docagent.schemas import Attachment
    cfg.drawing.method = method
    ex = DrawingExtractor(cfg)
    preds = {}
    for gt in gts:
        pdf = PROJECT_ROOT / "data" / "drawings" / gt["source_file"]
        t0 = time.time()
        if not as_images:
            d = ex.extract(Attachment.from_path(pdf)).model_dump(mode="json")
        else:                                  # picture route: each sheet as a 200 dpi JPG
            import pymupdf
            from docagent.pdf import render
            tmp = Path(tempfile.mkdtemp(prefix="drw_jpg_"))
            d = None
            with pymupdf.open(pdf) as doc:
                for pno, page in enumerate(doc, start=1):
                    jpg = tmp / f"{gt['drawing_id']}_p{pno}.jpg"
                    render(page, 200).save(jpg, quality=90)
                    part = ex.extract(Attachment.from_path(jpg)).model_dump(mode="json")
                    for v in part["values"]:
                        v["page"] = pno
                    if d is None:
                        d = part
                        d["drawing_id"] = gt["drawing_id"]
                    else:
                        d["values"] += part["values"]
                        d["notes"] += part["notes"]
                        d["pages"] += part["pages"]
        d["_seconds"] = round(time.time() - t0, 1)
        (out_dir / f"{gt['drawing_id']}.json").write_text(json.dumps(d, indent=1, ensure_ascii=False))
        preds[gt["drawing_id"]] = d
    return preds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", help="score existing <drawing_id>.json files in this folder")
    ap.add_argument("--run", action="store_true", help="run the extractor on the ground-truth drawings")
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default=None, help="geometry | tiles (default: from config)")
    ap.add_argument("--as-images", action="store_true", help="feed each sheet as a JPG (picture route)")
    ap.add_argument("--ids", nargs="*", default=None, help="subset of drawing ids")
    ap.add_argument("--label", default="")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--verbose", action="store_true", help="print missed / extra values")
    args = ap.parse_args(argv)

    gts = [json.loads(p.read_text()) for p in sorted(GT_DIR.glob("*.json"))]
    if args.ids:
        gts = [g for g in gts if any(i in g["drawing_id"] for i in args.ids)]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cfg = load_config(args.config)
    if args.run:
        method = args.method or cfg.drawing.method
        out_dir = RESULTS / "drawing_runs" / f"{stamp}_{args.label or method}{'_jpg' if args.as_images else ''}"
        out_dir.mkdir(parents=True, exist_ok=True)
        preds = run_extractor(cfg, method, args.as_images, out_dir, gts)
    else:
        pdir = Path(args.pred_dir)
        preds = {g["drawing_id"]: json.loads((pdir / f"{g['drawing_id']}.json").read_text()) for g in gts}

    per = [score_drawing(preds[g["drawing_id"]], g) for g in gts]
    summary = aggregate(per)
    cols = ["recall", "precision", "numbers", "type", "gdt_symbol", "datums", "belongs_to", "title_block", "notes", "units"]
    print(f"\n{'drawing':24} {'pred':>4} {'gt':>3} " + " ".join(f"{c[:9]:>9}" for c in cols))
    for d in per + [dict(summary, drawing="ALL")]:
        print(f"{d['drawing']:24} {d['n_pred']:>4} {d['n_gt_required']:>3} "
              + " ".join(f"{'-' if d[c] is None else (('✓' if d[c] else '✗') if isinstance(d[c], bool) else d[c]):>9}" for c in cols))
    if args.verbose:
        for d in per:
            print(f"\n{d['drawing']} missed: {d['missed']}\n  extra: {d['extra'][:40]}")
    secs = [p.get("_seconds") for p in preds.values() if p.get("_seconds") is not None]
    out = RESULTS / f"{stamp}_drawings_{args.label or 'run'}.json"
    out.write_text(json.dumps({"run": {"label": args.label, "method": args.method, "as_images": args.as_images,
                                       "pred_dir": args.pred_dir, "config": args.config, "seconds": secs},
                               "summary": summary, "drawings": per}, indent=1, ensure_ascii=False))
    print("\nsummary:", json.dumps(summary))
    print("results:", out.relative_to(PROJECT_ROOT))
    if args.compare:
        prev = json.loads(Path(args.compare).read_text())["summary"]
        print(f"\ncompared with {args.compare}:")
        for k, v in summary.items():
            if isinstance(v, (int, float)) and isinstance(prev.get(k), (int, float)) and v != prev[k]:
                print(f"  {k:12} {prev[k]} -> {v}")


if __name__ == "__main__":
    main()
