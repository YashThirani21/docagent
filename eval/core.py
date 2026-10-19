"""Shared eval harness: run record, case selection, config overrides, trace stats, results files, compare.

A suite module provides:
  NAME, DATASET (path or list of paths, hashed into the record), load_cases() -> list[dict] (each with id, split),
  add_args(ap) / prepare_args(args) / setup(ctx) (optional), run_case(ctx, case) -> row (dict with id, pass, ...),
  summarize(rows) -> dict,
  print_rows(rows) (optional), outcomes(row) -> {key: bool} (optional; default {id: pass}) used for flips.
Every case runs as a traced request under one batch id, so cost/latency come from the trace database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

import yaml

from docagent.config import PROJECT_ROOT, Config, load_config

RESULTS = PROJECT_ROOT / "eval" / "results"


# ---------------------------------------------------------------- arguments and config

def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=None, help="agent version (YAML); default configs/default.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE",
                    help="override one config value, e.g. --set drawing.method=tiles (repeatable)")
    ap.add_argument("--n", type=int, default=None, help="number of cases (deterministic sample with --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None, help="only these case ids")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"],
                    help="dev (used for tuning; default except drawing: all) | test (held out) | all")
    ap.add_argument("--label", default="", help="free-text tag stored with the run and in the file name")
    ap.add_argument("--compare", default=None, help="earlier results file: deltas + per-case flips")
    ap.add_argument("--fresh", action="store_true",
                    help="empty VLM cache for this run (needed for latency / cost numbers)")
    ap.add_argument("--results-dir", default=str(RESULTS), help="where results files go (default eval/results)")
    ap.add_argument("--verbose", action="store_true")


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """--set a.b=c on top of the loaded config; values parsed as YAML (numbers, booleans, lists)."""
    data = cfg.model_dump()
    for o in overrides:
        key, _, raw = o.partition("=")
        if not _:
            raise SystemExit(f"--set expects KEY.PATH=VALUE, got {o!r}")
        node = data
        *parents, leaf = key.strip().split(".")
        for p in parents:
            if not isinstance(node.get(p), dict):
                raise SystemExit(f"--set {key}: no config section {p!r}")
            node = node[p]
        if leaf not in node:
            raise SystemExit(f"--set {key}: unknown key {leaf!r} (have: {', '.join(node)})")
        node[leaf] = yaml.safe_load(raw)
    return Config(**data)


class Ctx:
    """What a suite's run_case gets: config, agent, tracer, batch id, run folder."""

    def __init__(self, args, suite):
        from docagent.trace import config_hash
        self.args, self.suite = args, suite
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.label = args.label or Path(args.config or "default").stem
        cfg = apply_overrides(load_config(args.config), args.overrides)
        # config hash of the version under test - before --fresh / run folders change paths that don't matter
        self.config_hash = config_hash(cfg)
        self.results = Path(args.results_dir)
        self.run_dir = self.results / "runs" / f"{self.stamp}_{suite.NAME}_{self.label}"
        if args.fresh:
            cfg.llm.cache_dir = tempfile.mkdtemp(prefix="eval_fresh_cache_")
        cfg.store.output_dir = str(self.run_dir / "outputs")      # result files of this run, not outputs/
        self.cfg = cfg
        self.batch_id = f"eval-{suite.NAME}-{self.label}-{self.stamp}"
        self._agent = None

    @property
    def agent(self):
        if self._agent is None:
            from docagent.agent import Agent
            self._agent = Agent(self.cfg)
        return self._agent

    @property
    def tracer(self):
        return self.agent.tracer

    def traced(self, req, versions: dict | None = None):
        """Context manager: one eval case = one traced request in this run's batch."""
        return self.tracer.request(req, versions or self.agent.versions, self.agent.config_hash, self.batch_id)


# ---------------------------------------------------------------- cases

def select(cases: list[dict], args) -> list[dict]:
    """split -> ids -> deterministic sample of n (kept in dataset order, so runs print alike)."""
    out = [c for c in cases if args.split == "all" or c.get("split", "dev") == args.split]
    if args.ids:
        want = set(args.ids)
        out = [c for c in out if c["id"] in want]
    if args.n and args.n < len(out):
        keep = set(random.Random(args.seed).sample([c["id"] for c in out], args.n))
        out = [c for c in out if c["id"] in keep]
    return out


def file_hash(paths) -> str:
    """Content hash of the dataset file(s): a changed dataset makes a comparison suspect."""
    h = hashlib.sha256()
    for p in sorted(Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- cost / latency from traces

def trace_stats(tracer, batch_id: str) -> dict:
    rows = tracer.query("SELECT latency_ms, prompt_tokens, completion_tokens, llm_calls, cache_hits, status "
                        "FROM requests WHERE batch_id=?", (batch_id,))
    lat = sorted(r["latency_ms"] / 1000 for r in rows if r["latency_ms"] is not None)

    def pct(p):
        return round(lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))], 2) if lat else None
    return {
        "requests": len(rows),
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None, "latency_p95_s": pct(0.95),
        "fresh_llm_calls": sum(r["llm_calls"] or 0 for r in rows),
        "cached_calls": sum(r["cache_hits"] or 0 for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
        "failed_requests": sum(1 for r in rows if r["status"] == "failed"),
    }


# ---------------------------------------------------------------- run, save, compare

def run_suite(suite, argv=None, extra_args=None) -> Path:
    ap = argparse.ArgumentParser(prog=f"python -m eval {suite.NAME}", description=suite.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    if extra_args:
        extra_args(ap)
    ap.set_defaults(split=getattr(suite, "DEFAULT_SPLIT", "dev"))     # drawing: all (agreement covers every sheet)
    args = ap.parse_args(argv)
    if hasattr(suite, "prepare_args"):          # suite options that are config overrides (drawing --method)
        suite.prepare_args(args)
    ctx = Ctx(args, suite)
    cases = select(suite.load_cases(), args)
    if not cases:
        raise SystemExit(f"no cases selected (split={args.split}, ids={args.ids})")
    if hasattr(suite, "setup"):
        suite.setup(ctx)

    t0 = time.time()
    rows = []
    for c in cases:                     # sequential: latency per case stays meaningful; drawings parallelise inside
        t1 = time.perf_counter()
        try:
            row = suite.run_case(ctx, c)
        except Exception as e:          # one broken case must not end the run; it counts as failed
            row = {"id": c["id"], "pass": False, "error": f"{type(e).__name__}: {e}"}
        row.setdefault("latency_s", round(time.perf_counter() - t1, 2))
        row.setdefault("split", c.get("split", "dev"))
        rows.append(row)

    summary = suite.summarize(rows)
    summary["errors"] = sum(1 for r in rows if r.get("error"))
    cost = trace_stats(ctx.tracer, ctx.batch_id) if ctx._agent is not None else {}
    record = {
        "suite": suite.NAME, "label": args.label, "timestamp": ctx.stamp, "config": args.config or "default",
        "overrides": args.overrides, "config_hash": ctx.config_hash,
        "versions": ctx.agent.versions | getattr(suite, "extra_versions", lambda c: {})(ctx),
        "model": ctx.cfg.llm.model, "dataset": [str(Path(p).relative_to(PROJECT_ROOT)) for p in _paths(suite)],
        "dataset_hash": file_hash(_paths(suite)), "split": args.split, "seed": args.seed, "n": len(rows),
        "case_ids": [r["id"] for r in rows], "fresh": args.fresh, "batch_id": ctx.batch_id,
        "wall_s": round(time.time() - t0, 1),
    }
    if hasattr(suite, "print_rows"):
        suite.print_rows(rows, args)
    print("\nsummary:", json.dumps(summary, ensure_ascii=False))
    print("cost:   ", json.dumps(cost))
    print(f"traces:  docagent requests --batch {ctx.batch_id}")

    ctx.results.mkdir(parents=True, exist_ok=True)
    out = ctx.results / f"{ctx.stamp}_{suite.NAME}_{ctx.label}.json"
    result = {"run": record, "summary": summary, "cost": cost, "rows": rows}
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    print("results:", _show(out))
    if args.compare:
        compare(result, json.loads(Path(args.compare).read_text()), args.compare,
                getattr(suite, "outcomes", None))
    return out


def outcomes_of(row: dict, fn=None) -> dict[str, bool]:
    if fn:
        return fn(row)
    p = row.get("pass", row.get("checks", {}).get("pass"))     # older run_qa files keep pass in checks
    return {row["id"]: bool(p)} if p is not None and "id" in row else {}


def compare(cur: dict, prev: dict, prev_name: str, outcomes_fn=None) -> dict:
    """Print metric deltas, per-case flips and cost deltas. Returns them (for tests)."""
    pr, cr = prev.get("run", {}), cur["run"]
    print(f"\ncompared with {prev_name} ({pr.get('label') or pr.get('config') or '?'}):")
    warn = []
    if pr.get("suite") and pr["suite"] != cr["suite"]:
        warn.append(f"different suite ({pr['suite']} vs {cr['suite']})")
    if pr.get("dataset_hash") and pr["dataset_hash"] != cr["dataset_hash"]:
        warn.append("dataset changed since the earlier run")
    if pr.get("split") and pr["split"] != cr["split"]:
        warn.append(f"different split ({pr['split']} vs {cr['split']})")
    if pr.get("case_ids") is not None and set(pr["case_ids"]) != set(cr["case_ids"]):
        warn.append(f"different cases ({len(set(pr['case_ids']) & set(cr['case_ids']))} in common): "
                    "flips only over common cases, averages not comparable")
    if not pr.get("suite"):
        warn.append("earlier file is from the old eval scripts: no run record, comparing what matches")
    for w in warn:
        print(f"  WARNING: {w}")

    deltas = {}
    before_m = dict(_flat(prev.get("summary", {})))
    for k, v in _flat(cur["summary"]):            # nested breakdowns too, e.g. by_input.image+text.pass
        pv = before_m.get(k)
        if pv is not None and v != pv:
            deltas[k] = (pv, v)
            print(f"  {k:22} {pv} -> {v}  ({'+' if v > pv else ''}{round(v - pv, 3)})")
    if not deltas:
        print("  metrics: no change")

    before: dict[str, bool] = {}
    for r in prev.get("rows", prev.get("drawings", [])):
        before.update(outcomes_of(r, outcomes_fn))
    flips = {"fixed": [], "broke": []}
    for r in cur["rows"]:
        for key, ok in outcomes_of(r, outcomes_fn).items():
            if key in before and before[key] != ok:
                flips["fixed" if ok else "broke"].append(key)
    print(f"  flips: {len(flips['fixed'])} newly passing, {len(flips['broke'])} newly failing")
    for k in flips["broke"]:
        print(f"    BROKE  {k}")
    for k in flips["fixed"]:
        print(f"    fixed  {k}")
    if not before:
        print("  (earlier file has no per-case outcomes to compare)")

    cost = {}
    for k, v in cur.get("cost", {}).items():
        pv = prev.get("cost", {}).get(k)
        if isinstance(v, (int, float)) and isinstance(pv, (int, float)) and v != pv:
            cost[k] = (pv, v)
    if cost:
        print("  cost/latency: " + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in cost.items()))
    if cur.get("cost") and prev.get("cost") and cur["run"].get("fresh") != pr.get("fresh"):
        print("  WARNING: one run used --fresh and the other did not: latency/tokens not comparable")
    return {"deltas": deltas, "flips": flips, "cost": cost, "warnings": warn}


def _flat(d: dict, prefix: str = ""):
    """Numeric leaves of a (nested) summary as (dotted key, value)."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{prefix}{k}", v


def _paths(suite) -> list[Path]:
    d = suite.DATASET
    return [Path(p) for p in (d if isinstance(d, (list, tuple)) else [d])]


def _show(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def rate(vals) -> float | None:
    vals = [bool(v) for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def mark(v) -> str:
    return "-" if v is None else "✓" if v else "✗"
