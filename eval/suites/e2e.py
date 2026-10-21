"""e2e: the whole agent - Agent.handle on a request as a user sends it; nothing is fed in that the user wouldn't send.

Dataset eval/datasets/e2e/cases.jsonl: text only, photo only, photo + question, drawing PDF / picture, several
drawings, text + drawing, photo + drawing, a report PDF with a question. Expected answers and drawing labels
are referenced from the other suites (one label, used twice), each output scored with that suite's scorer.
A case succeeds when every attachment is routed as expected, every output passes, and nothing errored:
  drawing with labels: recall >= DRAWING_OK and precision >= DRAWING_OK; without: agreement with the fix4
    reference run >= AGREE_OK (agree_recall and agree_precision) - change relative to fix4, not correctness
  answer: the qa_text / qa_image pass criterion; `answer: null` = no report answer expected (text + drawing:
  current behaviour is drawing JSON only - the row records that the question went unanswered)
Metrics: success, route accuracy, component pass given correct routing, error rate, and the decomposition
route_accuracy x component_pass vs success (where the e2e losses come from). Latency / tokens: cost section.
"""
from __future__ import annotations

import json

from docagent.config import PROJECT_ROOT
from eval.core import mark, rate

NAME = "e2e"
ROOT = PROJECT_ROOT / "eval" / "datasets"
DATASET = [ROOT / "e2e" / "cases.jsonl", ROOT / "qa_text.jsonl", ROOT / "qa_image" / "cases.jsonl",
           *sorted((ROOT / "drawings").glob("*.json")), *sorted((ROOT / "drawings" / "reference" / "geo-l4-fix4").glob("*.json"))]
DRAWING_OK = 0.7
AGREE_OK = 0.7          # drawings without labels: agreement with the reference run (geo-l4-fix4)


def load_cases() -> list[dict]:
    return [json.loads(l) for l in (ROOT / "e2e" / "cases.jsonl").read_text().splitlines() if l.strip()]


def _refs():
    qa_text = {c["id"]: c for c in map(json.loads, (ROOT / "qa_text.jsonl").read_text().splitlines()) if c}
    qa_image = {c["id"]: c for c in map(json.loads, (ROOT / "qa_image" / "cases.jsonl").read_text().splitlines()) if c}
    gts = {g["drawing_id"]: g for g in (json.loads(p.read_text()) for p in (ROOT / "drawings").glob("*.json"))}
    return qa_text, qa_image, gts


def score_answer(spec: dict | None, ans: dict | None, refs) -> dict:
    from eval.run_qa import score as score_text
    from eval.suites.qa_image import score as score_image
    qa_text, qa_image, _ = refs
    if spec is None:
        return {"expected": None, "ok": ans is None, "text_answered": ans is not None}
    if ans is None:
        return {"expected": spec, "ok": False, "problem": "no report answer"}
    if "inline" in spec:
        c = score_text(spec["inline"], ans)
    elif spec["from"] == "qa_text":
        c = score_text(qa_text[spec["id"]], ans)
    else:
        c = score_image(qa_image[spec["id"]], ans)
    return {"expected": spec, "ok": bool(c["pass"]), "checks": c}


def score_drawings(case: dict, resp, refs, atts) -> list[dict]:
    from eval.run_drawings import agreement, score_drawing
    from eval.suites.drawing import REFERENCE
    _, _, gts = refs
    drawing_atts = [a for a, r in zip(atts, case["expect"]["routes"]) if r == "drawing"]
    ref_specs = iter(case["expect"].get("reference", []))
    out = []
    for att, gid in zip(drawing_atts, case["expect"].get("drawings", [])):
        d = next((x for x in resp.drawings if x.source_sha256 == att.sha256), None)
        if d is None:
            out.append({"file": att.name, "gt": gid, "ok": False, "problem": "no drawing JSON for this file"})
            continue
        pred = d.model_dump(mode="json")
        if gid is None:                  # no labels: agreement with the reference run (change, not correctness)
            spec = next(ref_specs, None)
            ref_file = REFERENCE / f"{spec['drawing_id']}.json" if spec else None
            if ref_file is None or not ref_file.exists():
                out.append({"file": att.name, "gt": None, "ok": len(pred["values"]) > 0, "values": len(pred["values"])})
                continue
            a = agreement(pred, json.loads(ref_file.read_text()), spec.get("pages"))
            out.append({"file": att.name, "gt": None, "reference": f"{REFERENCE.name}/{ref_file.name}",
                        "ok": (a["agree_recall"] or 0) >= AGREE_OK and (a["agree_precision"] or 0) >= AGREE_OK,
                        **{k: a[k] for k in ("agree_recall", "agree_precision", "numbers_agree", "n_ref", "n_pred")},
                        "dropped_vs_reference": a["dropped"], "new_vs_reference": a["new"],
                        "changed_reading": a["changed"]})
            continue
        s = score_drawing(pred, gts[gid])
        out.append({"file": att.name, "gt": gid, "verified": gts[gid].get("verified", True),
                    "ok": (s["recall"] or 0) >= DRAWING_OK and (s["precision"] or 0) >= DRAWING_OK,
                    **{k: s[k] for k in ("recall", "precision", "numbers", "gdt_symbol", "belongs_to", "missed")}})
    return out


def run_case(ctx, case: dict) -> dict:
    from docagent.schemas import Attachment, Request
    refs = getattr(ctx, "_e2e_refs", None) or _refs()
    ctx._e2e_refs = refs
    atts = [Attachment.from_path(PROJECT_ROOT / f) for f in case["files"]]
    req = Request(text=case.get("text"), attachments=atts)
    resp = ctx.agent.handle(req, ctx.batch_id)           # traced inside, under this run's batch
    routes = [d.route.value for d in resp.routes]
    route_ok = routes == case["expect"]["routes"]
    drawings = score_drawings(case, resp, refs, atts) if "drawings" in case["expect"] else []
    answer = score_answer(case["expect"]["answer"], resp.report_answer and resp.report_answer.model_dump(mode="json"),
                          refs) if "answer" in case["expect"] else None
    components = [d["ok"] for d in drawings] + ([answer["ok"]] if answer else [])
    comp_ok = all(components) if components else True
    return {"id": case["id"], "what": case["what"], "request_id": req.request_id, "routes": routes,
            "expected_routes": case["expect"]["routes"], "route_ok": route_ok, "drawings": drawings,
            "answer": answer, "errors": resp.errors, "component_ok": comp_ok,
            "pass": route_ok and comp_ok and not resp.errors}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "route_ok" in r]
    routed = [r for r in ok if r["route_ok"]]
    route_acc = rate(r["route_ok"] for r in ok)
    comp = rate(r["component_ok"] for r in routed)
    return {
        "n": len(rows), "success": rate(r.get("pass") for r in rows), "route_accuracy": route_acc,
        "component_pass_given_route": comp,
        "expected_success_if_independent": round(route_acc * comp, 3) if route_acc is not None and comp is not None else None,
        "error_rate": rate(bool(r.get("errors") or r.get("error")) for r in rows),
        "unanswered_text_with_drawing": sum(1 for r in ok if (r.get("answer") or {}).get("text_answered") is False),
    }


def print_rows(rows: list[dict], args) -> None:
    print(f"\n{'id':4} {'split':5} {'ok':3} {'route':6} {'parts':6}  what / problem")
    for r in rows:
        if "route_ok" not in r:
            print(f"{r['id']:4} {r.get('split', ''):5} {mark(False):3} error: {r.get('error')}")
            continue
        probs = []
        if not r["route_ok"]:
            probs.append(f"routes {r['routes']} != {r['expected_routes']}")
        probs += [f"{d['file']}: " + (d.get("problem") or (f"recall {d.get('recall')} precision {d.get('precision')}"
                                                         if d.get("gt") else
                                                         f"agreement with reference: recall {d.get('agree_recall')} "
                                                         f"precision {d.get('agree_precision')}" if d.get("reference")
                                                         else "empty JSON"))
                  for d in r["drawings"] if not d["ok"]]
        a = r.get("answer")
        if a and not a["ok"]:
            c = a.get("checks", {})
            failed = [k for k in ("facts", "pages", "doc_hit", "refusal") if c.get(k) is False]
            detail = ", ".join(filter(None, [f"missing {c['missing']}" if c.get("missing") else "",
                                             f"forbidden {c['forbidden_found']}" if c.get("forbidden_found") else ""]))
            probs.append(a.get("problem") or (f"answer failed {'/'.join(failed)}" + (f" ({detail})" if detail else ""))
                         if a.get("expected") else "unexpected report answer")
        probs += r["errors"]
        print(f"{r['id']:4} {r.get('split', ''):5} {mark(r['pass']):3} {mark(r['route_ok']):6} {mark(r['component_ok']):6}"
              f"  {r['what']}" + (f"  <- {'; '.join(probs)}" if probs else ""))
