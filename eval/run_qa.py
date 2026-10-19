"""Run the report-QA question set and score it automatically.

  python -m eval qa_text                                   # dev questions (q01-q20), default config
  python -m eval qa_text --split test                      # held-out questions (t01-t15)
  python -m eval qa_text --config configs/colab_vllm.yaml --compare eval/results/<earlier>.json
  python -m eval.run_qa ...                                # old entry point, same options

The scorer lives here; the runner is eval/suites/qa_text.py + eval/core.py (NOTES.md section 10).

Checks (cheap and deterministic; an LLM judge can be added on top later):
  facts      every `must` string in the answer, no `must_not` string (catches wrong-aircraft values)
  pages      at least one citation on a page where the facts are stated
  page_recall  share of those pages that are cited
  refusal    for questions the reports cannot answer: grounded=false or an explicit "not stated"
  filler     a plain lookup answered with a diagnosis / recommended actions (should not happen)
  pass       facts and (refusal | pages)
"""
from __future__ import annotations

import re
from pathlib import Path

REFUSAL_RE = re.compile(r"not (stated|provided|specified|determined|included|available|covered)|"
                        r"do(es)? not (state|provide|specify|include|contain|cover|mention)|no information", re.I)


def _norm(s: str) -> str:
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)          # 64,000 -> 64000
    return re.sub(r"\s+", " ", s).lower()


def score(item: dict, ans: dict) -> dict:
    text = _norm(" ".join([ans["answer"], ans.get("diagnosis") or "", " ".join(ans["recommended_actions"])]))
    missing = [m for m in item["must"] if _norm(m) not in text]
    forbidden = [m for m in item["must_not"] if _norm(m) in text]
    cited = {(c["doc_id"], c["page"]) for c in ans["citations"]}
    gold = {tuple(p) for p in item["pages"]}
    r: dict = {
        "facts": not missing and not forbidden, "missing": missing, "forbidden_found": forbidden,
        "pages": bool(cited & gold) if gold else None,
        "page_recall": round(len(cited & gold) / len(gold), 2) if gold else None,
        "n_citations": len(cited),
        "off_doc_citations": sum(1 for d, _ in cited if gold and d not in {g[0] for g in gold}),
        "filler": item["kind"] == "lookup" and bool(ans.get("diagnosis") or ans["recommended_actions"]),
    }
    if item["kind"] == "refusal":
        r["refusal"] = (not ans["grounded"]) or bool(REFUSAL_RE.search(ans["answer"] + " " + (ans.get("diagnosis") or "")))
        r["pass"] = r["facts"] and r["refusal"]
    else:
        r["refusal"] = None
        r["pass"] = r["facts"] and bool(r["pages"])
    return r


def summarize(rows: list[dict]) -> dict:
    def rate(key, rows_=rows):
        vals = [r["checks"][key] for r in rows_ if r["checks"].get(key) is not None]
        return round(sum(map(bool, vals)) / len(vals), 3) if vals else None
    recalls = [r["checks"]["page_recall"] for r in rows if r["checks"].get("page_recall") is not None]
    lookups = [r for r in rows if r["kind"] == "lookup"]
    return {
        "n": len(rows), "pass": rate("pass"), "facts": rate("facts"), "pages": rate("pages"),
        "page_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "refusal": rate("refusal"), "filler_on_lookups": rate("filler", lookups),
        "avg_citations": round(sum(r["checks"]["n_citations"] for r in rows) / max(1, len(rows)), 1),
        "errors": sum(1 for r in rows if r.get("error")),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / max(1, len(rows)), 1),
    }


def main(argv=None):
    """Kept for the old command line: same as `python -m eval qa_text` (default --split dev = q01-q20)."""
    import sys
    from eval.core import run_suite
    from eval.suites import qa_text
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--dataset" in argv:                       # old option: another question file
        i = argv.index("--dataset")
        qa_text.DATASET = Path(argv[i + 1])
        del argv[i:i + 2]
    run_suite(qa_text, argv)


if __name__ == "__main__":
    main()
