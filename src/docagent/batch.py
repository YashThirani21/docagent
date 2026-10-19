"""Batches: many files, each its own request, tracked under one batch id (NOTES.md section 9.1 / 9.2).

One implementation for `docagent bulk`, `bulk --resume`, `docagent batch` and the chat app's background batches.
All state is in the trace database, so anyone can read progress (`status`) and a restarted process can resume:
- plan_new     every file is enqueued as a 'queued' request first; a crash leaves the rest visibly not started.
               A file already done (same idem_key) still gets a request - it is reused instantly - so the batch
               lists every file and its results zip is complete.
- plan_resume  per file the latest attempt: queued -> run, interrupted (dead process) -> retry, failed/partial ->
               retry unless unfinished_only (the chat's automatic resume must not loop on a file that always fails).
"""
from __future__ import annotations

import json
import logging
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import Agent, RetryError
from .schemas import Attachment, Request, Response
from .trace import Tracer, new_id

log = logging.getLogger(__name__)

Job = tuple[str, object]          # ("run", Request) | ("retry", request_id)
TERMINAL = ("succeeded", "partial", "failed")


def new_batch_id(prefix: str = "bulk") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{new_id()[:4]}"


def plan_new(agent: Agent, atts: list[Attachment], batch_id: str) -> tuple[list[Job], list[str]]:
    """-> jobs, notes. Identical content twice in one batch runs once."""
    jobs: list[Job] = []
    notes: list[str] = []
    seen: dict[str, str] = {}
    for att in atts:
        if att.sha256 in seen:
            notes.append(f"{att.name}: same content as {seen[att.sha256]} - processed once")
            continue
        seen[att.sha256] = att.name
        req = Request(attachments=[att])
        agent.tracer.enqueue(req, batch_id)
        jobs.append(("run", req))
    return jobs, notes


def latest_per_file(tracer: Tracer, batch_id: str) -> dict[str, list]:
    """attachment path -> that file's requests in this batch, oldest first (bulk requests have one file)."""
    files: dict[str, list] = {}
    for r in tracer.query("SELECT * FROM requests WHERE batch_id=? ORDER BY created_at", (batch_id,)):
        atts = json.loads(r["attachments"] or "[]")
        files.setdefault(atts[0]["path"] if atts else r["request_id"], []).append(r)
    return files


def plan_resume(agent: Agent, batch_id: str, unfinished_only: bool = False) -> tuple[list[Job], list[str]]:
    jobs: list[Job] = []
    notes: list[str] = []
    for path, rows in latest_per_file(agent.tracer, batch_id).items():
        r = rows[-1]
        name = file_names(r)
        if r["status"] == "succeeded":
            notes.append(f"[skip] {name}: done ({r['request_id']})")
        elif r["status"] == "running":
            notes.append(f"[skip] {name}: still running in pid {r['pid']} ({r['request_id']})")
        elif r["status"] == "queued":
            try:
                jobs.append(("run", agent.request_from_row(r)))
            except RetryError as e:
                notes.append(f"[fail] {name}: {e}")
        elif r["status"] == "interrupted" or not unfinished_only:
            jobs.append(("retry", r["request_id"]))
    return jobs, notes


def run_job(agent: Agent, job: Job, batch_id: str, force: bool = False) -> tuple[str, Response, list[str]]:
    kind, x = job
    if kind == "run":
        return kind, agent.handle(x, batch_id, reuse=not force), []
    resp, notes = agent.retry(x, batch_id=batch_id, force=force)
    return kind, resp, notes


# ---------------------------------------------------------------- reading a batch

def file_names(r) -> str:
    atts = json.loads(r["attachments"] or "[]")
    return ", ".join(a.get("name") or Path(a["path"]).name for a in atts) or r["request_id"]


def _outputs(tracer: Tracer, request_id: str) -> list[str]:
    rows = tracer.query("SELECT DISTINCT output_path FROM tasks WHERE request_id=? AND status IN ('succeeded','reused') "
                        "AND output_path IS NOT NULL ORDER BY started_at", (request_id,))
    return [r["output_path"] for r in rows if Path(r["output_path"]).exists()]


def status(tracer: Tracer, batch_id: str) -> dict:
    """Per file: its latest attempt; totals over every request of the batch (all attempts)."""
    files = latest_per_file(tracer, batch_id)
    per_file, counts = [], {}
    for rows in files.values():
        r = rows[-1]
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        ok = next((x for x in reversed(rows) if x["status"] in ("succeeded", "partial")), None)
        tasks = tracer.query("SELECT route, status, error FROM tasks WHERE request_id=?", (r["request_id"],))
        err = r["error"] or next((t["error"] for t in tasks if t["error"]), None)
        per_file.append({"file": file_names(r), "status": r["status"], "attempts": len(rows),
                         "request_id": r["request_id"],
                         "route": ",".join(sorted({t["route"] for t in tasks if t["route"]})) or None,
                         "reused": any(t["status"] == "reused" for t in tasks),
                         "outputs": _outputs(tracer, ok["request_id"]) if ok else [],
                         "error": err.splitlines()[0][:200] if err else None})
    all_rows = [x for rows in files.values() for x in rows]
    ended = [x["ended_at"] for x in all_rows if x["ended_at"]]
    start = min((x["created_at"] for x in all_rows), default=None)
    secs = None
    if start and ended:
        from datetime import datetime
        f = "%Y-%m-%dT%H:%M:%S.%fZ"
        secs = round((datetime.strptime(max(ended), f) - datetime.strptime(start, f)).total_seconds(), 1)
    return {"batch_id": batch_id, "files": len(files), "counts": counts,
            "finished": bool(files) and not any(s in counts for s in ("queued", "running")),
            "seconds": secs, "llm_calls": sum(x["llm_calls"] or 0 for x in all_rows),
            "prompt_tokens": sum(x["prompt_tokens"] or 0 for x in all_rows),
            "completion_tokens": sum(x["completion_tokens"] or 0 for x in all_rows),
            "cache_hits": sum(x["cache_hits"] or 0 for x in all_rows), "per_file": per_file}


def format_status(st: dict, markdown: bool = False) -> str:
    order = ["succeeded", "partial", "failed", "interrupted", "running", "queued"]
    counts = ", ".join(f"{st['counts'][k]} {k}" for k in order if k in st["counts"]) or "no files"
    head = (f"batch {st['batch_id']}: {st['files']} files - {counts}"
            + (" - finished" if st["finished"] else " - in progress")
            + (f" in {st['seconds']}s" if st["finished"] and st["seconds"] is not None else ""))
    spend = (f"model calls {st['llm_calls']} (tokens {st['prompt_tokens']} in / {st['completion_tokens']} out), "
             f"{st['cache_hits']} replayed from cache")
    lines = []
    for f in st["per_file"]:
        out = f["outputs"][0] if len(f["outputs"]) == 1 else f"{len(f['outputs'])} files" if f["outputs"] else "-"
        extra = " (reused)" if f["reused"] else ""
        extra += f" x{f['attempts']} attempts" if f["attempts"] > 1 else ""
        extra += f"  error: {f['error']}" if f["error"] and f["status"] != "succeeded" else ""
        lines.append(f"{f['file']}: {f['status']}{extra}  route={f['route'] or '-'}  -> {out}")
    if markdown:
        return f"**{head}**\n\n{spend}\n\n" + "\n".join(f"- `{l}`" for l in lines)
    return "\n".join([head, spend] + ["  " + l for l in lines])


def build_zip(tracer: Tracer, batch_id: str, out_zip: str | Path) -> Path:
    """Latest successful output per file + batch_summary.json. Names inside: original file stem + .json
    (answers: .answer.json). Written to a temp file and renamed (never a half zip)."""
    st = status(tracer, batch_id)
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_name(f".{out_zip.name}.{os.getpid()}.tmp")
    used: set[str] = set()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in st["per_file"]:
            stem = Path(f["file"]).stem
            for p in f["outputs"]:
                base = f"{stem}.answer.json" if p.endswith("answer.json") else f"{stem}.json"
                arc, n = base, 2
                while arc in used:                         # two files with the same name
                    arc, n = f"{base[:-5]}_{n}.json", n + 1
                used.add(arc)
                z.write(p, arc)
                f.setdefault("zip_entries", []).append(arc)
        z.writestr("batch_summary.json", json.dumps(st, indent=2, ensure_ascii=False))
    os.replace(tmp, out_zip)
    return out_zip


# ---------------------------------------------------------------- background batches (chat app)

class BatchRunner:
    """Runs batches in a bounded thread pool inside a long-running process. Nothing lives only in memory:
    a closed browser tab just stops watching, a restart resumes from the database (resume_unfinished)."""

    def __init__(self, agent: Agent, workers: int):
        self.agent = agent
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="batch")

    def start(self, atts: list[Attachment], force: bool = False, prefix: str = "chat") -> tuple[str, list[str]]:
        batch_id = new_batch_id(prefix)
        jobs, notes = plan_new(self.agent, atts, batch_id)
        self._submit(jobs, batch_id, force)
        return batch_id, notes

    def resume_unfinished(self, prefix: str = "chat-") -> list[str]:
        """Batches of this prefix with files still queued or interrupted (process died): run those files."""
        rows = self.agent.tracer.query(
            "SELECT DISTINCT batch_id FROM requests WHERE batch_id LIKE ? || '%' AND status IN ('queued','interrupted')",
            (prefix,))
        resumed = []
        for r in rows:
            jobs, _ = plan_resume(self.agent, r["batch_id"], unfinished_only=True)
            if jobs:
                self._submit(jobs, r["batch_id"], False)
                resumed.append(r["batch_id"])
                log.warning("resuming batch %s: %d file(s)", r["batch_id"], len(jobs))
        return resumed

    def _submit(self, jobs: list[Job], batch_id: str, force: bool) -> None:
        for job in jobs:
            self.pool.submit(self._run, job, batch_id, force)

    def _run(self, job: Job, batch_id: str, force: bool) -> None:
        try:
            run_job(self.agent, job, batch_id, force)
        except Exception:                                  # recorded in the trace; the pool goes on
            log.exception("batch %s job failed", batch_id)
