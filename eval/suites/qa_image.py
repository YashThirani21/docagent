"""qa_image: a picture (optionally with a question) answered from the reports (ReportQA called directly).

Dataset eval/datasets/qa_image/cases.jsonl from `python -m eval.build_qa_image_set` (report figures, caption cut
off, augmented) + eval/datasets/qa_image/out_of_scope/ (pictures the reports don't cover: must be refused).
Metrics
  doc_hit       a citation is in the report the figure comes from
  fig_in_context  the figure itself was retrieved into the context (hit@k, k = retrieval.top_k)
  page_cited    the figure's page is cited
  facts         every `must` fact in answer + diagnosis + actions, no `must_not` claim
  actions       recommended actions present when the picture shows a problem
  refusal       refusal / out-of-scope cases: grounded=false or an explicit "not covered / not stated"
  filler_on_lookups  lookup / wrong_aircraft questions answered with a diagnosis or actions (as in qa_text)
  fact_page_cited    picture + question: the page stating the expected fact is cited
  pass          doc_hit and facts (refusal kinds: refused and no forbidden claim)
Every metric overall, per input type (image | image+text) and per kind (figure, diagnosis, lookup, refusal,
wrong_aircraft, out_of_scope); --compare shows the deltas at every level. wrong_aircraft: the question names a
different aircraft than the pictured one - expected: answer for the named aircraft, never the pictured one's value.
"""
from __future__ import annotations

import json

from docagent.config import PROJECT_ROOT
from eval.core import mark, rate
from eval.run_qa import REFUSAL_RE, _norm

NAME = "qa_image"
ROOT = PROJECT_ROOT / "eval" / "datasets" / "qa_image"
DATASET = [ROOT / "cases.jsonl"]
EMPTY = {"answer": "", "diagnosis": None, "recommended_actions": [], "citations": [], "grounded": False,
         "retrieved": [], "image_observations": []}


def load_cases() -> list[dict]:
    cases = [json.loads(l) for l in (ROOT / "cases.jsonl").read_text().splitlines() if l.strip()]
    for f in sorted((ROOT / "out_of_scope").glob("*")):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            side = f.with_suffix(f.suffix + ".json")
            q = json.loads(side.read_text()).get("question") if side.exists() else None
            cases.append({"id": f"oos-{f.stem}", "image": str(f.relative_to(PROJECT_ROOT)), "question": q,
                          "kind": "out_of_scope", "split": "test", "must": [], "must_not": [],
                          "input": "image+text" if q else "image"})
    return cases


def setup(ctx) -> None:
    ctx.agent.qa            # load embedder + index before the first case, so its latency isn't counted in it


def extra_versions(ctx) -> dict:
    return {"retriever": ctx.agent.qa.retriever.version}


REFUSE_KINDS = ("out_of_scope", "refusal")      # the reports don't answer it: must be refused
LOOKUP_KINDS = ("lookup", "wrong_aircraft")     # a fact, no problem shown: diagnosis / actions would be filler


def score(case: dict, ans: dict) -> dict:
    text = _norm(" ".join([ans["answer"], ans.get("diagnosis") or "", " ".join(ans["recommended_actions"])]))
    missing = [m for m in case["must"] if _norm(m) not in text]
    forbidden = [m for m in case["must_not"] if _norm(m) in text]
    r = {"missing": missing, "forbidden_found": forbidden, "facts": not missing and not forbidden,
         "n_citations": len(ans["citations"])}
    # same filler rule as qa_text: a lookup answered with a diagnosis / recommended actions
    r["filler"] = bool(ans.get("diagnosis") or ans["recommended_actions"]) if case["kind"] in LOOKUP_KINDS else None
    if case["kind"] in REFUSE_KINDS:
        r["refusal"] = (not ans["grounded"]) or bool(REFUSAL_RE.search(ans["answer"] + " " + (ans.get("diagnosis") or "")))
        r["pass"] = r["refusal"] and not forbidden
        return r
    cited_docs = {c["doc_id"] for c in ans["citations"]}
    r["doc_hit"] = case["doc_id"] in cited_docs
    r["wrong_doc_citations"] = sum(1 for c in ans["citations"] if c["doc_id"] != case["doc_id"])
    r["fig_in_context"] = any(c["doc_id"] == case["doc_id"] and c.get("figure_no") == case["figure_no"]
                              for c in ans.get("retrieved", []))
    r["page_cited"] = any(c["doc_id"] == case["doc_id"] and c["page"] == case["page"] for c in ans["citations"])
    if case.get("pages"):                        # picture + question: the page that states the expected fact
        gold = {tuple(x) for x in case["pages"]}
        r["fact_page_cited"] = any((c["doc_id"], c["page"]) in gold for c in ans["citations"])
    r["actions"] = bool(ans["recommended_actions"]) if case.get("problem") else None
    r["pass"] = r["doc_hit"] and r["facts"]
    return r


def run_case(ctx, case: dict) -> dict:
    from PIL import Image
    from docagent.schemas import Attachment, Request
    path = PROJECT_ROOT / case["image"]
    req = Request(text=case.get("question"), attachments=[Attachment.from_path(path)])
    err = None
    try:
        with ctx.traced(req):
            ans = ctx.agent.qa.run(case.get("question"), [Image.open(path).convert("RGB")]).model_dump(mode="json")
    except Exception as e:
        ans, err = dict(EMPTY), f"{type(e).__name__}: {e}"
    checks = score(case, ans)
    return {"id": case["id"], "request_id": req.request_id, "kind": case["kind"],
            "input": case.get("input") or ("image+text" if case.get("question") else "image"),
            "question": case.get("question"),
            "expected": {k: case.get(k) for k in ("doc_id", "figure_no", "page")}, "answer": ans, "error": err,
            "checks": checks, "pass": checks["pass"]}


def _metrics(rows: list[dict]) -> dict:
    c = [r.get("checks", {}) for r in rows]
    ans = [x for r, x in zip(rows, c) if r.get("kind", "figure") not in REFUSE_KINDS]   # cases with an answer
    m = {
        "n": len(rows), "pass": rate(r.get("pass") for r in rows),
        "doc_hit": rate(x.get("doc_hit") for x in ans), "fig_in_context": rate(x.get("fig_in_context") for x in ans),
        "page_cited": rate(x.get("page_cited") for x in ans), "fact_page_cited": rate(x.get("fact_page_cited") for x in ans),
        "facts": rate(x.get("facts") for x in c),
        "forbidden_claims": sum(len(x.get("forbidden_found", [])) for x in c),
        "actions_when_problem": rate(x.get("actions") for x in ans),
        "filler_on_lookups": rate(x.get("filler") for x in c),
        "refusal": rate(x.get("refusal") for x in c),
        "avg_wrong_doc_citations": round(sum(x.get("wrong_doc_citations", 0) for x in ans) / max(1, len(ans)), 2),
    }
    return {k: v for k, v in m.items() if v is not None}


def summarize(rows: list[dict]) -> dict:
    """Overall, then per input type (image only / image + text) and per kind - compare shows every level."""
    s = _metrics(rows)
    s["by_input"] = {t: _metrics([r for r in rows if r.get("input") == t])
                     for t in ("image", "image+text") if any(r.get("input") == t for r in rows)}
    kinds = sorted({r.get("kind") for r in rows if r.get("kind")})
    s["by_kind"] = {k: _metrics([r for r in rows if r.get("kind") == k]) for k in kinds}
    return s


def print_rows(rows: list[dict], args) -> None:
    print(f"\n{'id':22} {'split':5} {'input':10} {'kind':14} {'pass':5} {'doc':4} {'fig':4} {'facts':6} {'acts':5} "
          f"{'filler':6}  problem")
    for r in rows:
        c = r.get("checks", {})
        prob = r.get("error") or "; ".join(filter(None, [
            f"missing {c.get('missing')}" if c.get("missing") else "",
            f"forbidden {c.get('forbidden_found')}" if c.get("forbidden_found") else "",
            "did not refuse" if c.get("refusal") is False else "", "filler" if c.get("filler") else ""]))
        print(f"{r['id']:22} {r.get('split', ''):5} {r.get('input', ''):10} {r.get('kind', ''):14} "
              f"{mark(r.get('pass')):5} {mark(c.get('doc_hit')):4} {mark(c.get('fig_in_context')):4} "
              f"{mark(c.get('facts')):6} {mark(c.get('actions')):5} {'yes' if c.get('filler') else '':6}  {prob}")
    print("\nby input: " + "  ".join(f"{t}: n={m['n']} pass={m.get('pass')}" for t, m in _by(rows, "input").items()))
    print("by kind:  " + "  ".join(f"{k}: n={m['n']} pass={m.get('pass')}" for k, m in _by(rows, "kind").items()))


def _by(rows, key) -> dict:
    return {v: _metrics([r for r in rows if r.get(key) == v]) for v in sorted({r.get(key) for r in rows if r.get(key)})}
