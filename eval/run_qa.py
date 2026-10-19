"""Run the report-QA question set and score it automatically.

  python -m eval.run_qa                                   # all questions, default config
  python -m eval.run_qa --config configs/colab_vllm.yaml  # e.g. on Colab
  python -m eval.run_qa --n 5 --ids q10 q14               # subset
  python -m eval.run_qa --compare eval/results/<earlier>.json   # better or worse than a previous run?

Writes eval/results/<timestamp>_qa_<config>.json (every answer + every check) and prints a summary.

Checks (cheap and deterministic; an LLM judge can be added on top later):
  facts      every `must` string in the answer, no `must_not` string (catches wrong-aircraft values)
  pages      at least one citation on a page where the facts are stated
  page_recall  share of those pages that are cited
  refusal    for questions the reports cannot answer: grounded=false or an explicit "not stated"
  filler     a plain lookup answered with a diagnosis / recommended actions (should not happen)
  pass       facts and (refusal | pages)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docagent.config import PROJECT_ROOT, load_config

DATASET = PROJECT_ROOT / "eval" / "datasets" / "qa_text.jsonl"
RESULTS = PROJECT_ROOT / "eval" / "results"
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--n", type=int, default=None, help="first N questions")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--compare", default=None, help="earlier results file to diff against")
    ap.add_argument("--label", default="", help="free-text tag stored with the run, e.g. 'no-filler-prompt'")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    from docagent.pipelines.report_qa import ReportQA       # after config: sets offline env
    qa = ReportQA(cfg)

    items = [json.loads(l) for l in Path(args.dataset).read_text().splitlines() if l.strip()]
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    if args.n:
        items = items[: args.n]

    def run(item):
        t0 = time.perf_counter()
        try:
            ans = qa.run(item["question"]).model_dump(mode="json")
            err = None
        except Exception as e:                                # keep going; count it
            ans = {"answer": "", "diagnosis": None, "recommended_actions": [], "citations": [],
                   "grounded": False, "retrieved": []}
            err = f"{type(e).__name__}: {e}"
        row = {"id": item["id"], "question": item["question"], "kind": item["kind"], "expected": item["expected"],
               "answer": ans, "error": err, "latency_s": round(time.perf_counter() - t0, 1)}
        row["checks"] = score(item, ans)
        return row

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg.llm.max_concurrency) as ex:   # 1 on Ollama, 4 on vLLM
        rows = list(ex.map(run, items))
    summary = summarize(rows)

    print(f"\n{'id':5} {'pass':5} {'facts':6} {'pages':6} {'filler':7} {'secs':>5}  problem")
    for r in rows:
        c = r["checks"]
        prob = r["error"] or "; ".join(filter(None, [
            f"missing {c['missing']}" if c["missing"] else "",
            f"forbidden {c['forbidden_found']}" if c["forbidden_found"] else "",
            "no correct page cited" if c["pages"] is False else "",
            "did not refuse" if c["refusal"] is False else "",
            "filler" if c["filler"] else ""]))
        print(f"{r['id']:5} {'✓' if c['pass'] else '✗':5} {'✓' if c['facts'] else '✗':6} "
              f"{'-' if c['pages'] is None else '✓' if c['pages'] else '✗':6} {'yes' if c['filler'] else '':7} "
              f"{r['latency_s']:5.0f}  {prob}")
    print("\nsummary:", json.dumps(summary))

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = Path(args.config or "default").stem
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}_qa_{tag}.json"
    out.write_text(json.dumps({
        "run": {"config": args.config or "default", "label": args.label, "model": cfg.llm.model,
                "collection": cfg.store.collection, "qa_version": qa.version,
                "retriever_version": qa.retriever.version, "wall_s": round(time.time() - t0, 1)},
        "summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print("results:", out.relative_to(PROJECT_ROOT))

    if args.compare:
        prev = json.loads(Path(args.compare).read_text())
        print(f"\ncompared with {args.compare} ({prev['run'].get('label') or prev['run']['config']}):")
        for k, v in summary.items():
            pv = prev["summary"].get(k)
            if isinstance(v, (int, float)) and isinstance(pv, (int, float)) and v != pv:
                print(f"  {k:18} {pv} -> {v}")
        before = {r["id"]: r["checks"]["pass"] for r in prev["rows"]}
        for r in rows:
            if r["id"] in before and before[r["id"]] != r["checks"]["pass"]:
                print(f"  {r['id']}: {'fixed' if r['checks']['pass'] else 'BROKE'}")


if __name__ == "__main__":
    main()
