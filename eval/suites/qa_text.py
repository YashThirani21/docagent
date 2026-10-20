"""qa_text: text questions answered from the reports (ReportQA called directly - no routing).

Dataset eval/datasets/qa_text.jsonl: q01-q20 split=dev (used for tuning, section 6), t01-t15 split=test.
Scoring (eval/run_qa.py `score`, unchanged): facts (must / must_not), pages (a cited page states the facts),
page_recall, refusal for unanswerable questions, filler (diagnosis/actions on a plain lookup). pass = facts and
(pages | refusal).
"""
from __future__ import annotations

import json

from docagent.config import PROJECT_ROOT
from eval.core import mark
from eval.run_qa import score, summarize as _summarize

NAME = "qa_text"
DATASET = PROJECT_ROOT / "eval" / "datasets" / "qa_text.jsonl"
EMPTY = {"answer": "", "diagnosis": None, "recommended_actions": [], "citations": [], "grounded": False,
         "retrieved": []}


def load_cases() -> list[dict]:
    return [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]


def setup(ctx) -> None:
    ctx.agent.qa            # load embedder + index before the first case, so its latency isn't counted in it


def extra_versions(ctx) -> dict:
    return {"retriever": ctx.agent.qa.retriever.version}


def run_case(ctx, item: dict) -> dict:
    from docagent.schemas import Request
    req = Request(text=item["question"])
    err = None
    try:
        with ctx.traced(req):
            ans = ctx.agent.qa.run(item["question"]).model_dump(mode="json")
    except Exception as e:                          # count it, keep going
        ans, err = dict(EMPTY), f"{type(e).__name__}: {e}"
    checks = score(item, ans)
    return {"id": item["id"], "request_id": req.request_id, "question": item["question"], "kind": item["kind"],
            "expected": item["expected"], "answer": ans, "error": err, "checks": checks, "pass": checks["pass"]}


def summarize(rows: list[dict]) -> dict:
    for r in rows:                                   # rows that crashed in the harness have no checks
        r.setdefault("checks", {"pass": False, "facts": False, "pages": None, "page_recall": None,
                                "refusal": None, "filler": False, "n_citations": 0})
        r.setdefault("kind", "lookup")
        r.setdefault("latency_s", 0)
    s = _summarize(rows)
    s.pop("avg_latency_s", None)                     # latency comes from the traces (cost section)
    return s


def print_rows(rows: list[dict], args) -> None:
    print(f"\n{'id':5} {'split':5} {'pass':5} {'facts':6} {'pages':6} {'filler':7} {'secs':>5}  problem")
    for r in rows:
        c = r["checks"]
        prob = r.get("error") or "; ".join(filter(None, [
            f"missing {c.get('missing')}" if c.get("missing") else "",
            f"forbidden {c.get('forbidden_found')}" if c.get("forbidden_found") else "",
            "no correct page cited" if c.get("pages") is False else "",
            "did not refuse" if c.get("refusal") is False else "",
            "filler" if c.get("filler") else ""]))
        print(f"{r['id']:5} {r['split']:5} {mark(c['pass']):5} {mark(c['facts']):6} {mark(c['pages']):6} "
              f"{'yes' if c.get('filler') else '':7} {r['latency_s']:5.0f}  {prob}")
