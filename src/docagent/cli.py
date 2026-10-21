"""Command line: ingest reports, ask, bulk-process uploads, inspect routing and the index.

  docagent ingest data/reports                 # index reports (idempotent; --force to rebuild)
  docagent ask "Why did the spar fail?"        # text question
  docagent ask -f photo.jpg                    # picture only - the agent decides what to do
  docagent ask -f d.pdf --force                # --force (ask/bulk/retry): reuse no earlier result
  docagent bulk data/drawings --out outputs    # many files at once (drawings -> JSON); done files skipped
  docagent bulk --resume <batch_id>            # finish a crashed / partly failed bulk run
  docagent batch <batch_id> [--zip out.zip]    # batch progress / results download
  docagent retry <request_id>                  # new attempt; finished results reused
  docagent route some.pdf other.png            # show routing decisions only
  docagent status                              # what is in the index
  docagent requests --status failed --since 1h # traced requests (--file nist_ctc_01: by attachment)
  docagent trace <request_id|last>             # what one request did and called
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import load_config
from .uploads import SUPPORTED

FORCE_HELP = ("reuse no earlier result: process every file again (default: a file already done with the same "
              "agent version and config is reused). Finished model calls still replay from the VLM cache; "
              "to bypass that too, use a config with another llm.cache_dir")


def _files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]
    return out


def cmd_ingest(args, cfg):
    from .ingest.reports import ReportIngestor
    res = ReportIngestor(cfg, describe_figures=not args.no_figures,
                         reuse_descriptions=not args.refresh_descriptions).ingest_paths(args.paths, force=args.force)
    for name, status in res.items():
        print(f"{name}: {status}")


def cmd_ask(args, cfg):
    from .agent import Agent, save_drawing_json
    resp = Agent(cfg).ask(args.text, args.files, force=args.force)
    if args.json:
        print(resp.model_dump_json(indent=2))
    else:
        from .render import render_markdown
        print(render_markdown(resp))
        for p in resp.outputs:
            print(f"written: {p}")
        if args.out:
            for p in save_drawing_json(resp, args.out):
                print(f"drawing JSON copied: {p}")
    if resp.errors and not (resp.drawings or resp.report_answer):
        sys.exit(1)                     # nothing came out: fail the command, as when the error used to raise


def cmd_bulk(args, cfg):
    """New batch: files / folders / zips -> upload store -> one queued request per file -> processed.
    --resume: finish a batch (queued run, failed / interrupted retried). Same code as the chat's batches."""
    from . import batch
    from .agent import Agent, save_drawing_json
    if not args.paths and not args.resume:
        sys.exit("give files/folders/zips, or --resume <batch_id>")
    agent = Agent(cfg)                                     # marks requests of dead processes 'interrupted'
    workers = args.workers or cfg.batch.workers
    out = Path(args.out)
    (out / "answers").mkdir(parents=True, exist_ok=True)
    if args.resume:
        batch_id = args.resume
        if not agent.tracer.query("SELECT 1 FROM requests WHERE batch_id=? LIMIT 1", (batch_id,)):
            sys.exit(f"no requests in batch {batch_id!r}")
        jobs, notes = batch.plan_resume(agent, batch_id)
    else:
        batch_id = args.batch_id or batch.new_batch_id("bulk")
        atts, notes = agent.store_files(args.paths)
        jobs, more = batch.plan_new(agent, atts, batch_id)
        notes += more
    for n in notes:
        print(n)
    print(f"batch_id={batch_id}: {len(jobs)} to process, {workers} workers")
    t0 = time.time()

    def one(job):
        kind, resp, notes = batch.run_job(agent, job, batch_id, force=args.force)
        written = save_drawing_json(resp, out / "drawings")
        if resp.report_answer:
            p = out / "answers" / f"{resp.request_id}.json"
            p.write_text(resp.report_answer.model_dump_json(indent=2))
            written.append(p)
        row = agent.tracer.query("SELECT * FROM requests WHERE request_id=?", (resp.request_id,))
        return kind, resp, written, batch.file_names(row[0]) if row else "?", notes

    summary = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                kind, resp, written, name, notes = fut.result()
                for n in notes:
                    print(f"  note: {n}")
                routes = ",".join(r.route.value for r in resp.routes)
                n_vals = sum(len(d.values) for d in resp.drawings)
                if resp.drawings and len(resp.reused) == len(resp.drawings) and not resp.errors:
                    print(f"[skip] {name}: skipped (done) - reused result from request "
                          f"{next(iter(resp.reused.values()))}; request={resp.request_id}")
                else:
                    print(f"[{'ok' if not resp.errors else 'fail'}] {name}: {kind} route={routes} values={n_vals} "
                          f"request={resp.request_id}" + (f" errors={resp.errors}" if resp.errors else ""))
                summary.append({"file": name, "request_id": resp.request_id, "batch_id": batch_id, "job": kind,
                                "routes": routes, "values": n_vals, "reused": bool(resp.reused),
                                "outputs": [str(w) for w in written], "errors": resp.errors})
            except Exception as e:                         # incl. RetryError (file changed / gone)
                print(f"[fail] {futs[fut][0]}: {e}", file=sys.stderr)
                summary.append({"file": "?", "errors": [str(e)]})
    (out / "bulk_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {time.time() - t0:.0f}s; summary: {out / 'bulk_summary.json'}; "
          f"status: docagent batch {batch_id}; finish failed ones: docagent bulk --resume {batch_id}")


def cmd_batch(args, cfg):
    from . import batch
    from .trace import Tracer
    tr = Tracer.from_config(cfg)
    st = batch.status(tr, args.batch_id)
    if not st["files"]:
        sys.exit(f"no requests in batch {args.batch_id!r}")
    print(batch.format_status(st))
    if args.zip:
        print(f"results zip: {batch.build_zip(tr, args.batch_id, args.zip)}")


def cmd_retry(args, cfg):
    from .agent import Agent, RetryError
    from .render import render_markdown
    agent = Agent(cfg)
    try:
        resp, notes = agent.retry(args.request_id, force=args.force)
    except RetryError as e:
        sys.exit(f"retry refused: {e}")
    for n in notes:
        print(f"note: {n}")
    st = [t["status"] for t in agent.tracer.query("SELECT status FROM tasks WHERE request_id=?", (resp.request_id,))]
    print(render_markdown(resp))
    print(f"{st.count('reused')} of {len(st)} attachment(s) reused, {st.count('failed')} failed; "
          f"trace: docagent trace {resp.request_id}")


def cmd_route(args, cfg):
    from .router import Router
    from .schemas import Attachment
    r = Router(cfg)
    for f in _files(args.paths):
        d = r.route_attachment(Attachment.from_path(f))
        print(f"{f.name}: {d.route.value} ({d.tier}, conf={d.confidence:.2f}) {d.reason}")


def cmd_status(args, cfg):
    from .store import VectorStore
    s = VectorStore(cfg)
    print(f"collection {cfg.store.collection}: {s.count()} chunks")
    for doc, n in sorted(s.documents().items()):
        print(f"  {doc}: {n}")


def _since(s: str) -> str:
    """'90m' / '2h' / '1d' -> ISO cutoff comparable with the stored timestamps."""
    from datetime import datetime, timedelta, timezone
    n, unit = float(s[:-1]), s[-1]
    delta = timedelta(**{{"m": "minutes", "h": "hours", "d": "days"}[unit]: n})
    return (datetime.now(timezone.utc) - delta).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _short(v, n: int = 70) -> str:
    t = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return t if len(t) <= n else t[: n - 3] + "..."


def cmd_requests(args, cfg):
    from .trace import Tracer
    tr = Tracer.from_config(cfg)
    where, params = [], []
    if args.status:
        where.append("status = ?"), params.append(args.status)
    if args.since:
        where.append("created_at >= ?"), params.append(_since(args.since))
    if args.batch:
        where.append("batch_id = ?"), params.append(args.batch)
    if args.file:                                          # by attachment path (drawing uploads have no text)
        # attachments JSON holds the original name (and the stored path); tasks.name / attachment for old rows
        where.append("(attachments LIKE ? OR request_id IN "
                     "(SELECT request_id FROM tasks WHERE name LIKE ? OR attachment LIKE ?))")
        params += [f"%{args.file}%"] * 3
    rows = tr.query("SELECT * FROM requests" + (" WHERE " + " AND ".join(where) if where else "")
                    + " ORDER BY created_at DESC LIMIT ?", (*params, args.limit))
    # llm / tok = fresh model calls (spend); cached = replies replayed from the VLM cache (no tokens)
    print(f"{'created (UTC)':19}  {'request_id':12}  {'status':11} {'try':>3} {'secs':>6} {'llm':>4} {'cached':>6} "
          f"{'tok in':>7} {'tok out':>7}  files / routes / text")
    for r in rows:
        summ = json.loads(r["summary"] or "{}")
        names = [a.get("name") or Path(a["path"]).name for a in json.loads(r["attachments"] or "[]")] or [
            t["name"] or Path(t["attachment"]).name for t in tr.query(
                "SELECT name, attachment FROM tasks WHERE request_id=?", (r["request_id"],)) if t["attachment"]]
        what = "  ".join(filter(None, [",".join(names), ",".join(summ.get("routes", [])), r["text"]]))
        secs = f"{r['latency_ms'] / 1000:.1f}" if r["latency_ms"] is not None else "-"
        print(f"{r['created_at'][:19]}  {r['request_id']:12}  {r['status']:11} {r['attempt'] or 1:>3} {secs:>6} "
              f"{r['llm_calls']:>4} {r['cache_hits']:>6} {r['prompt_tokens']:>7} {r['completion_tokens']:>7}  "
              f"{_short(what, 70)}")


def cmd_trace(args, cfg):
    from .trace import Tracer
    tr = Tracer.from_config(cfg)
    if args.request_id == "last":
        rows = tr.query("SELECT * FROM requests ORDER BY created_at DESC LIMIT 1")
    else:                                                  # a unique prefix is enough
        rows = tr.query("SELECT * FROM requests WHERE request_id LIKE ? || '%'", (args.request_id,))
    if len(rows) != 1:
        sys.exit(f"{len(rows)} requests match {args.request_id!r}")
    r = rows[0]
    secs = f"{r['latency_ms'] / 1000:.1f}s" if r["latency_ms"] is not None else "-"
    print(f"request {r['request_id']}  {r['status']}  {secs}  created {r['created_at']}"
          + (f"  batch={r['batch_id']}" if r["batch_id"] else ""))
    if r["text"]:
        print(f"  text:     {_short(r['text'], 100)}")
    chain = tr.chain(r["request_id"])
    if len(chain) > 1 or r["retry_of"]:
        print(f"  attempt:  {r['attempt'] or 1}" + (f" (retry of {r['retry_of']})" if r["retry_of"] else ""))
        for c in chain:
            mark = "  <- this" if c["request_id"] == r["request_id"] else ""
            secs_c = f"{c['latency_ms'] / 1000:.1f}s" if c["latency_ms"] is not None else "-"
            print(f"    attempt {c['attempt'] or 1}: {c['request_id']}  {c['status']:11} {secs_c:>7}  "
                  f"llm {c['llm_calls']} cached {c['cache_hits']}{mark}")
    print(f"  versions: {r['versions']}  config={r['config_hash']}")
    spans = tr.query("SELECT * FROM spans WHERE request_id=? ORDER BY started_at", (r["request_id"],))
    cached = [a for a in (json.loads(s_["attrs"] or "{}") for s_ in spans if s_["name"] == "llm.call")
              if a.get("cache_hit")]
    print(f"  llm:      {r['llm_calls']} model calls, tokens {r['prompt_tokens']} in / {r['completion_tokens']} out")
    if r["cache_hits"]:
        print(f"  cache:    {r['cache_hits']} replayed from cache (no spend; originally "
              f"{sum(a.get('orig_prompt_tokens') or 0 for a in cached)} in / "
              f"{sum(a.get('orig_completion_tokens') or 0 for a in cached)} out)")
    if r["error"]:
        print(f"  error:    {r['error']}")
    if r["summary"]:
        print(f"  outcome:  {_short(json.loads(r['summary']), 300)}")
    tasks = tr.query("SELECT * FROM tasks WHERE request_id=? ORDER BY started_at", (r["request_id"],))
    names = {}
    if tasks:
        print("tasks:")
    for i, t in enumerate(tasks, start=1):
        names[t["task_id"]] = f"task{i}"
        decision = f"{t['route']} ({t['tier']}, {t['confidence']:.2f})" if t["route"] else "not routed"
        extra = f" from task {t['reused_from']}" if t["reused_from"] else ""
        extra += f" -> {t['output_path']}" if t["output_path"] else ""
        print(f"  task{i} {t['name'] or Path(t['attachment'] or '-').name}: {decision} "
              f"{t['status']}{extra} stage={t['stage']}" + (f"\n    error: {t['error'].splitlines()[0]}" if t["error"] else ""))   # full text + traceback in the db
    children: dict = {}
    for s_ in spans:
        children.setdefault(s_["parent_id"], []).append(s_)

    def dur(s_):
        return f"{s_['duration_ms'] / 1000:.2f}s" if s_["duration_ms"] is not None else s_["status"]

    def show(parent, depth):
        kids = children.get(parent, [])
        llm = [k for k in kids if k["name"] == "llm.call"]
        if llm and not args.llm:                           # one line per parent instead of one per call
            a = [json.loads(k["attrs"] or "{}") for k in llm]
            fresh = [x for x in a if not x.get("cache_hit")]
            failed = sum(k["status"] != "succeeded" for k in llm)
            print(f"{'  ' * depth}llm.call x{len(llm)}  {sum(k['duration_ms'] or 0 for k in llm) / 1000:.2f}s  "
                  f"fresh {len(fresh)} tokens {sum(x.get('prompt_tokens') or 0 for x in fresh)}/"
                  f"{sum(x.get('completion_tokens') or 0 for x in fresh)}  cached {len(a) - len(fresh)}"
                  + (f"  FAILED {failed}" if failed else ""))
        for k in kids:
            if k["name"] == "llm.call" and not args.llm:
                continue
            attrs = {a: v for a, v in json.loads(k["attrs"] or "{}").items() if v not in (None, [], "")}
            task_tag = f" [{names[k['task_id']]}]" if k["task_id"] in names and depth == 1 else ""
            print(f"{'  ' * depth}{k['name']}{task_tag}  {dur(k)}  {_short(attrs, 110) if attrs else ''}"
                  + (f"  ERROR {k['error']}" if k["error"] else ""))
            show(k["span_id"], depth + 1)

    print("spans:")
    show(None, 1)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="docagent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (= agent version); default configs/default.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="index report PDFs (files or directories)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    p.add_argument("--no-figures", action="store_true", help="skip VLM figure descriptions (fast)")
    p.add_argument("--refresh-descriptions", action="store_true",
                   help="re-describe figures with the current VLM instead of reusing those in the index")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="one request: text and/or files")
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("-f", "--files", nargs="*", default=[])
    p.add_argument("--json", action="store_true", help="print the full Response JSON")
    p.add_argument("--out", default=None, help="also copy drawing JSON here (always written to store.output_dir)")
    p.add_argument("--force", action="store_true", help=FORCE_HELP)
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("bulk", help="process many uploaded files (each routed automatically)")
    p.add_argument("paths", nargs="*", help="files, folders or .zip archives")
    p.add_argument("--resume", metavar="BATCH_ID", help="finish a batch: run queued, retry failed/interrupted")
    p.add_argument("--force", action="store_true", help=FORCE_HELP)
    p.add_argument("--out", default="outputs")
    p.add_argument("--workers", type=int, default=None, help="files at once (default: batch.workers)")
    p.add_argument("--batch-id", default=None, help="tag for the traced requests (default: generated)")
    p.set_defaults(fn=cmd_bulk)

    p = sub.add_parser("batch", help="status of a bulk / chat batch; --zip: download its results")
    p.add_argument("batch_id")
    p.add_argument("--zip", metavar="OUT.zip", help="latest successful output per file + batch_summary.json")
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("retry", help="new attempt of a request; finished results are reused")
    p.add_argument("request_id", help="id or unique prefix (the latest attempt of its chain is retried)")
    p.add_argument("--force", action="store_true", help="allow retrying a succeeded request. " + FORCE_HELP)
    p.set_defaults(fn=cmd_retry)

    p = sub.add_parser("route", help="show routing decisions only")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=cmd_route)

    p = sub.add_parser("status", help="show what is indexed")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("requests", help="list traced requests")
    p.add_argument("--status", choices=["queued", "running", "succeeded", "partial", "failed", "interrupted"])
    p.add_argument("--since", help="e.g. 30m, 2h, 1d")
    p.add_argument("--batch", help="bulk / eval batch id")
    p.add_argument("--file", help="substring of an attachment path, e.g. nist_ctc_01")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(fn=cmd_requests)

    p = sub.add_parser("trace", help="what one request did and called (span tree)")
    p.add_argument("request_id", help="id, unique prefix, or 'last'")
    p.add_argument("--llm", action="store_true", help="list every llm.call instead of one line per step")
    p.set_defaults(fn=cmd_trace)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    from .trace import install_log_file
    install_log_file(cfg)
    args.fn(args, cfg)


if __name__ == "__main__":
    main()
