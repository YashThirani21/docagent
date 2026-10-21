"""Score drawing extraction (eval/datasets/drawings/*.json).

  python -m eval drawing --split all --label geo                 # run the extractor on the GT drawings, score
  python -m eval drawing --as-images --label geo-jpg             # picture route (each sheet as a JPG)
  python -m eval drawing --set drawing.method=tiles --label tiles
  python -m eval drawing --pred-dir outputs/drawings             # score JSONs that already exist
  python -m eval.run_drawings ...                                # old entry point (--run is now the default)

The scorer lives here; the runner is eval/suites/drawing.py + eval/core.py (NOTES.md section 10).

Matching: per sheet, every predicted value is paired one-to-one with the most similar label value
(similarity of the printed text after normalising symbols, or identical numbers). Title-block and note
entries are scored separately.

Metrics
  recall        required label values found
  precision     predicted values that match a label value (incl. optional ones)
  numbers       matched values whose nominal/limits/tolerances/quantity agree (where labelled)
  type          matched values with the right coarse type (dimension-like / frame / datum / other)
  gdt_symbol    matched frames with the right geometric characteristic
  datums        matched frames with the right datum letters
  belongs_to    matched values whose belongs_to names an acceptable feature (labelled keywords; lenient)
  belongs_primary  stricter: the GT's main feature word (`feature`, else the first keyword) is named
  title/notes/units  document-level fields (PDF route)
"""
from __future__ import annotations

import difflib
import re

from docagent.config import PROJECT_ROOT

GT_DIR = PROJECT_ROOT / "eval" / "datasets" / "drawings"
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
    rows, num, typ, sym, dat, bel, bel_p = [], [], [], [], [], [], []
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
            bel_p.append((g.get("feature") or g["belongs"][0]) in b)     # stricter: the main feature word
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
        "belongs_primary": rate(bel_p),
        "title_block": rate(tb_ok), "notes": len(pred.get("notes", [])) == gt.get("n_notes"),
        "units": pred.get("units") == gt.get("units"),
        "missed": missed, "extra": extra, "pairs": rows,
        # per required label value: found or not - what `--compare` flips are computed over
        "found": {f"{gt['drawing_id']}#{j + 1} {gts[j]['raw_text']} (p{gts[j]['page']})": j in matched_g for j in req},
    }


def aggregate(per: list[dict]) -> dict:
    keys = ["recall", "precision", "numbers", "type", "gdt_symbol", "datums", "belongs_to", "belongs_primary",
            "title_block"]
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



# ---------------------------------------------------------------- agreement with a reference run

AGREE_KEYS = ("agree_recall", "agree_precision", "numbers_agree", "type_agree", "gdt_agree", "datums_agree",
              "belongs_to_agree")


def agreement(pred: dict, ref: dict, pages: list[int] | None = None) -> dict:
    """How far a run reproduces a reference run (model outputs, not labels) - same matcher as against ground
    truth. Measures change, not correctness. pages: restrict both sides to these sheets (a photo of one sheet).
    Per-value diffs: dropped (reference value not reproduced), new (not in the reference), changed (paired, but
    the reading differs: text, numbers, type, GD&T characteristic, datums or belongs_to)."""
    def keep(vs):
        return [v for v in vs if v["type"] not in DOC_TYPES and (pages is None or v["page"] in pages)]
    preds, refs = keep(pred["values"]), keep(ref["values"])
    pairs = match(preds, refs)
    n_num = n_typ = n_gdt = n_dat = n_bel = 0
    c_num = c_typ = c_gdt = c_dat = c_bel = 0
    changed, reproduced = [], {}
    for i, j in pairs:
        p, r = preds[i], refs[j]
        what = []
        if core(p["raw_text"]) != core(r["raw_text"]):
            what.append("text")
        ok = numbers_ok(p, r)
        if ok is not None:
            n_num += 1
            c_num += ok
            what += [] if ok else ["numbers"]
        n_typ += 1
        c_typ += p["type"] == r["type"]
        what += [] if p["type"] == r["type"] else ["type"]
        if "gdt_frame" in (p["type"], r["type"]):
            n_gdt += 1
            n_dat += 1
            g = (p.get("characteristic") or "") == (r.get("characteristic") or "")
            d = sorted(p.get("datums") or []) == sorted(r.get("datums") or [])
            c_gdt += g
            c_dat += d
            what += ([] if g else ["characteristic"]) + ([] if d else ["datums"])
        n_bel += 1
        b = _bel(p) == _bel(r)
        c_bel += b
        what += [] if b else ["belongs_to"]
        if what:
            changed.append({"ref": r["raw_text"], "new": p["raw_text"], "page": r["page"], "what": what,
                            **({"ref_belongs_to": r.get("belongs_to"), "new_belongs_to": p.get("belongs_to")}
                               if "belongs_to" in what else {})})
    matched_r = {j for _, j in pairs}
    for j, r in enumerate(refs):
        reproduced[f"{ref['drawing_id']}#ref{j + 1} {r['raw_text']} (p{r['page']})"] = j in matched_r
    rate = lambda c, n: round(c / n, 3) if n else None
    return {
        "n_pred": len(preds), "n_ref": len(refs), "n_paired": len(pairs),
        "agree_recall": rate(len(pairs), len(refs)), "agree_precision": rate(len(pairs), len(preds)),
        "numbers_agree": rate(c_num, n_num), "type_agree": rate(c_typ, n_typ), "gdt_agree": rate(c_gdt, n_gdt),
        "datums_agree": rate(c_dat, n_dat), "belongs_to_agree": rate(c_bel, n_bel),
        "counts": {"num": [c_num, n_num], "typ": [c_typ, n_typ], "gdt": [c_gdt, n_gdt], "dat": [c_dat, n_dat],
                   "bel": [c_bel, n_bel]},
        "dropped": [f"{refs[j]['raw_text']} (p{refs[j]['page']})" for j in range(len(refs)) if j not in matched_r],
        "new": [f"{preds[i]['raw_text']} (p{preds[i]['page']})" for i in range(len(preds))
                if i not in {a for a, _ in pairs}],
        "changed": changed, "reproduced": reproduced,
    }


def _bel(v: dict) -> str:
    return re.sub(r"\s+", " ", (v.get("belongs_to") or "").strip().lower())


def aggregate_agreement(per: list[dict]) -> dict:
    """Micro-average over drawings from the raw counts."""
    if not per:
        return {}
    tot = lambda f: sum(f(a) for a in per)
    r = lambda c, n: round(c / n, 3) if n else None
    out = {"agree_recall": r(tot(lambda a: a["n_paired"]), tot(lambda a: a["n_ref"])),
           "agree_precision": r(tot(lambda a: a["n_paired"]), tot(lambda a: a["n_pred"]))}
    for key, k in (("numbers_agree", "num"), ("type_agree", "typ"), ("gdt_agree", "gdt"), ("datums_agree", "dat"),
                   ("belongs_to_agree", "bel")):
        out[key] = r(tot(lambda a: a["counts"][k][0]), tot(lambda a: a["counts"][k][1]))
    out.update(ref_values=tot(lambda a: a["n_ref"]), dropped_vs_reference=tot(lambda a: len(a["dropped"])),
               new_vs_reference=tot(lambda a: len(a["new"])), changed_reading=tot(lambda a: len(a["changed"])))
    return out

def main(argv=None):
    """Kept for the old command line: same as `python -m eval drawing` (--pred-dir, --method, --as-images work)."""
    from eval.core import run_suite
    from eval.suites import drawing
    run_suite(drawing, argv, drawing.add_args)


if __name__ == "__main__":
    main()
