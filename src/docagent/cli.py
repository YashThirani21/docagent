"""Command line: ingest reports, ask, bulk-process uploads, inspect routing and the index.

  docagent ingest data/reports                 # index reports (idempotent; --force to rebuild)
  docagent ask "Why did the spar fail?"        # text question
  docagent ask -f photo.jpg                    # picture only - the agent decides what to do
  docagent bulk data/drawings --out outputs    # many files at once (drawings -> JSON)
  docagent route some.pdf other.png            # show routing decisions only
  docagent status                              # what is in the index
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

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]
    return out


def cmd_ingest(args, cfg):
    from .ingest.reports import ReportIngestor
    res = ReportIngestor(cfg, describe_figures=not args.no_figures).ingest_paths(args.paths, force=args.force)
    for name, status in res.items():
        print(f"{name}: {status}")


def cmd_ask(args, cfg):
    from .agent import Agent, save_drawing_json
    resp = Agent(cfg).ask(args.text, args.files)
    if args.json:
        print(resp.model_dump_json(indent=2))
        return
    from .render import render_markdown
    print(render_markdown(resp))
    for p in save_drawing_json(resp, args.out):
        print(f"drawing JSON written: {p}")


def cmd_bulk(args, cfg):
    from .agent import Agent, save_drawing_json
    agent = Agent(cfg)
    files = _files(args.paths)
    out = Path(args.out)
    (out / "answers").mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} files, {args.workers} workers")
    t0 = time.time()

    def one(f: Path):
        resp = agent.ask(None, [f])
        written = save_drawing_json(resp, out / "drawings")
        if resp.report_answer:
            p = out / "answers" / f"{f.stem}.json"
            p.write_text(resp.report_answer.model_dump_json(indent=2))
            written.append(p)
        return f, resp, written

    summary = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, f) for f in files]
        for fut in as_completed(futs):
            try:
                f, resp, written = fut.result()
                routes = ",".join(r.route.value for r in resp.routes)
                n_vals = sum(len(d.values) for d in resp.drawings)
                print(f"[ok] {f.name}: route={routes} values={n_vals} -> {[str(w) for w in written]}"
                      + (f" errors={resp.errors}" if resp.errors else ""))
                summary.append({"file": f.name, "request_id": resp.request_id, "routes": routes,
                                "values": n_vals, "outputs": [str(w) for w in written], "errors": resp.errors})
            except Exception as e:
                print(f"[fail] {e}", file=sys.stderr)
                summary.append({"file": "?", "errors": [str(e)]})
    (out / "bulk_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {time.time() - t0:.0f}s; summary: {out / 'bulk_summary.json'}")


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
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="one request: text and/or files")
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("-f", "--files", nargs="*", default=[])
    p.add_argument("--json", action="store_true", help="print the full Response JSON")
    p.add_argument("--out", default="outputs/drawings")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("bulk", help="process many uploaded files (each routed automatically)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default="outputs")
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(fn=cmd_bulk)

    p = sub.add_parser("route", help="show routing decisions only")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=cmd_route)

    p = sub.add_parser("status", help="show what is indexed")
    p.set_defaults(fn=cmd_status)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.fn(args, load_config(args.config))


if __name__ == "__main__":
    main()
