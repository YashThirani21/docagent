"""drawing: values read from a technical drawing (the extractor called directly - router bypassed).

Two scores, reported side by side (NOTES.md section 10):
- accuracy vs hand labels (eval/datasets/drawings/*.json): recall, precision, numbers, type, gdt_symbol, datums,
  belongs_to, belongs_primary, title block, notes, units. Headline = verified sheets only (CTC-01, FTC-07 dev;
  CTC-03 / FTC-11 test count once `verified: true` - until then shown separately as draft labels).
- agreement with a reference run (default eval/datasets/drawings/reference/geo-l4-fix4: model outputs, not labels)
  on all 11 drawings: agree_recall, agree_precision, numbers_agree, type_agree, gdt_agree, datums_agree,
  belongs_to_agree, with per-value diffs (dropped vs reference / new vs reference / changed reading). It measures
  CHANGE relative to the reference, not correctness: fixing a reference mistake lowers agreement.
`--compare` flips: per ground-truth value (found / not found) and per reference value (reproduced / dropped).
Options: --as-images (each sheet as a 200-dpi JPG: picture route; compared with the reference's jpg/ outputs where
they exist, else its PDF outputs), --method geometry|tiles (= --set drawing.method=...), --pred-dir DIR (score
existing <drawing_id>.json files, run nothing), --reference DIR, --no-reference.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from docagent.config import PROJECT_ROOT
from eval.core import mark
from eval.run_drawings import GT_DIR, aggregate, aggregate_agreement, agreement, score_drawing

NAME = "drawing"
DEFAULT_SPLIT = "all"                    # agreement covers every drawing; the GT files carry their own split
REFERENCE = GT_DIR / "reference" / "geo-l4-fix4"
DATASET = sorted(GT_DIR.glob("*.json")) + sorted(REFERENCE.glob("*.json"))
NOTE = ("agreement = change relative to the reference run (model outputs, not labels), not correctness; "
        "accuracy on the verified sheets is the correctness measure")
_args = None


def add_args(ap) -> None:
    ap.add_argument("--as-images", action="store_true", help="feed each sheet as a JPG (picture route)")
    ap.add_argument("--method", default=None, help="geometry | tiles (shortcut for --set drawing.method=...)")
    ap.add_argument("--pred-dir", default=None, help="score existing <drawing_id>.json files in this folder")
    ap.add_argument("--reference", default=str(REFERENCE),
                    help="reference run folder (<drawing_id>.json, optional jpg/) - default geo-l4-fix4")
    ap.add_argument("--no-reference", action="store_true", help="accuracy on the labelled sheets only")
    ap.add_argument("--run", action="store_true", help=argparse.SUPPRESS)   # old flag: running is the default


def prepare_args(args) -> None:
    global _args, DATASET
    _args = args
    if args.method:                      # recorded like any other override, applied before the config loads
        args.overrides.append(f"drawing.method={args.method}")
    ref = [] if args.no_reference else sorted(Path(args.reference).glob("*.json"))
    DATASET = sorted(GT_DIR.glob("*.json")) + ref       # what the run record's dataset hash covers


def load_cases() -> list[dict]:
    gts = {g["drawing_id"]: g for g in (json.loads(p.read_text()) for p in sorted(GT_DIR.glob("*.json")))}
    refs = {} if _args is None or _args.no_reference else {
        p.stem: p for p in sorted(Path(_args.reference).glob("*.json"))}
    cases = []
    for did in sorted(set(gts) | set(refs)):
        gt = gts.get(did)
        cases.append({"id": did, "split": gt.get("split", "dev") if gt else "dev", "gt": gt,
                      "verified": bool(gt and gt.get("verified", True)), "ref": refs.get(did),
                      "source_file": gt["source_file"] if gt else f"{did}.pdf"})
    return cases


def reference_for(ctx, case: dict) -> tuple[dict | None, str | None]:
    if case["ref"] is None:
        return None, None
    if ctx.args.as_images:
        jpg = case["ref"].parent / "jpg" / case["ref"].name
        if jpg.exists():
            return json.loads(jpg.read_text()), "jpg"
        return json.loads(case["ref"].read_text()), "pdf (no picture-route reference for this drawing)"
    return json.loads(case["ref"].read_text()), "pdf"


def extract(ctx, path: Path) -> dict:
    from docagent.schemas import Attachment, Request
    att = Attachment.from_path(path)
    with ctx.traced(Request(attachments=[att])):
        return ctx.agent.drawings.extract(att).model_dump(mode="json")


def predict(ctx, case: dict) -> dict:
    if ctx.args.pred_dir:
        return json.loads((Path(ctx.args.pred_dir) / f"{case['id']}.json").read_text())
    pdf = PROJECT_ROOT / "data" / "drawings" / case["source_file"]
    if not ctx.args.as_images:
        return extract(ctx, pdf)
    import pymupdf
    from docagent.pdf import render
    tmp = Path(tempfile.mkdtemp(prefix="drw_jpg_"))
    d = None
    with pymupdf.open(pdf) as doc:                   # picture route: each sheet on its own, pages renumbered
        for pno, page in enumerate(doc, start=1):
            jpg = tmp / f"{case['id']}_p{pno}.jpg"
            render(page, 200).save(jpg, quality=90)
            part = extract(ctx, jpg)
            for v in part["values"]:
                v["page"] = pno
            if d is None:
                d = part
                d["drawing_id"] = case["id"]
            else:
                d["values"] += part["values"]
                d["notes"] += part["notes"]
                d["pages"] += part["pages"]
    return d


def run_case(ctx, case: dict) -> dict:
    pred = predict(ctx, case)
    if not ctx.args.pred_dir:                         # keep the prediction next to the results
        out = ctx.run_dir / "drawings" / f"{case['id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(pred, indent=1, ensure_ascii=False))
    row = {"id": case["id"], "verified": case["verified"], "has_gt": case["gt"] is not None}
    if case["gt"]:
        row["gt"] = score_drawing(pred, case["gt"])
    ref, route = reference_for(ctx, case)
    if ref is not None:
        row["reference"] = agreement(pred, ref) | {"route": route}
    return row


def outcomes(row: dict) -> dict[str, bool]:
    out = dict(row.get("found") or row.get("gt", {}).get("found", {}))   # hand-label values: found
    out.update(row.get("reference", {}).get("reproduced", {}))           # reference values: reproduced
    return out


def summarize(rows: list[dict]) -> dict:
    gt_rows = [r["gt"] for r in rows if r.get("gt") and r.get("verified")]
    draft = [r["gt"] for r in rows if r.get("gt") and not r.get("verified")]
    agree = [r["reference"] for r in rows if r.get("reference")]
    s = aggregate(gt_rows) if gt_rows else {}            # headline accuracy: verified sheets (same keys as before)
    s["accuracy_drawings"] = [r["id"] for r in rows if r.get("gt") and r.get("verified")]
    if draft:
        s["draft_labels"] = aggregate(draft) | {"drawings": [r["id"] for r in rows if r.get("gt") and not r.get("verified")],
                                                "note": "unverified hand labels - not in the headline"}
    if agree:
        s.update(aggregate_agreement(agree))
        s["reference"] = str(Path(_args.reference).name) if _args else None
        s["agreement_drawings"] = len(agree)
        s["agreement_note"] = NOTE
    s["n"] = len(rows)
    return s


def print_rows(rows: list[dict], args) -> None:
    cols = ["recall", "precision", "numbers", "type", "gdt_symbol", "datums", "belongs_to", "belongs_primary",
            "title_block", "notes", "units"]
    gt_rows = [r for r in rows if r.get("gt")]
    if gt_rows:
        print(f"\nACCURACY vs hand labels\n{'drawing':24} {'split':5} {'pred':>4} {'gt':>3} "
              + " ".join(f"{c[:9]:>9}" for c in cols))
        for r in gt_rows:
            d = r["gt"]
            print(f"{r['id']:24} {r.get('split', ''):5} {d['n_pred']:>4} {d['n_gt_required']:>3} "
                  + " ".join(f"{(mark(d[c]) if isinstance(d[c], bool) else '-' if d[c] is None else d[c]):>9}"
                             for c in cols) + ("" if r.get("verified") else "   (draft labels, not in headline)"))
            if args.verbose:
                print(f"   missed: {d['missed']}\n   extra:  {d['extra'][:40]}")
    ref_rows = [r for r in rows if r.get("reference")]
    if ref_rows:
        acols = ["agree_recall", "agree_precision", "numbers_agree", "type_agree", "gdt_agree", "datums_agree",
                 "belongs_to_agree"]
        print(f"\nAGREEMENT with reference {Path(args.reference).name} ({NOTE})\n{'drawing':24} {'new':>4} {'ref':>4} "
              + " ".join(f"{c.replace('_agree', '').replace('agree_', '')[:9]:>9}" for c in acols)
              + "  dropped/new/changed")
        for r in ref_rows:
            a = r["reference"]
            print(f"{r['id']:24} {a['n_pred']:>4} {a['n_ref']:>4} "
                  + " ".join(f"{'-' if a[c] is None else a[c]:>9}" for c in acols)
                  + f"  {len(a['dropped'])}/{len(a['new'])}/{len(a['changed'])}"
                  + ("" if a["route"] == "pdf" or not args.as_images else f"  (vs {a['route']})"))
            if args.verbose:
                for x in a["dropped"]:
                    print(f"   dropped vs reference: {x}")
                for x in a["new"]:
                    print(f"   new vs reference:     {x}")
                for c in a["changed"]:
                    print(f"   changed reading:      {c['ref']!r} -> {c['new']!r} (p{c['page']}: {', '.join(c['what'])})")
    for r in rows:
        if "error" in r:
            print(f"{r['id']:24} error: {r['error']}")
