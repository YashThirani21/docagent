"""Request tracing: what every request did and what it called, persisted in SQLite (NOTES.md section 9).

Three tables:
  requests  one per user request: lifecycle, versions, config hash, token / call / cache-hit totals
  tasks     one per attachment: route decision, status, error + traceback (the unit of retry / resume)
  spans     one per step, parent/child tree, attrs JSON (counts, chunk ids, tokens per llm.call)

Rows are written at state transitions (open -> close), not periodically: a crash loses at most the step
that was running, and that step stays visible as a 'running' span. The current request / task / span live
in contextvars, so pipelines just call `span(...)`; outside a traced request it is a no-op.
Tracing must never break the agent: database errors are logged and swallowed.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import socket
import sqlite3
import threading
import time
import traceback
import uuid
from concurrent.futures import Executor, Future
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)

STATUSES = ("queued", "running", "succeeded", "partial", "failed", "interrupted")   # task also: reused

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY, batch_id TEXT, status TEXT NOT NULL, text TEXT,
    created_at TEXT NOT NULL, started_at TEXT, ended_at TEXT, latency_ms REAL,
    versions TEXT, config_hash TEXT,
    prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,   -- fresh model calls only (spend)
    llm_calls INTEGER DEFAULT 0, cache_hits INTEGER DEFAULT 0,             -- calls sent / replies replayed
    summary TEXT, error TEXT,
    attachments TEXT, retry_of TEXT, attempt INTEGER DEFAULT 1, pid INTEGER, host TEXT);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, attachment TEXT, sha256 TEXT,
    route TEXT, tier TEXT, confidence REAL, reason TEXT,
    status TEXT NOT NULL, stage TEXT, attempts INTEGER DEFAULT 1, output_path TEXT,
    error TEXT, traceback TEXT, started_at TEXT, ended_at TEXT, heartbeat_at TEXT,
    idem_key TEXT, reused_from TEXT, name TEXT);
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY, parent_id TEXT, request_id TEXT NOT NULL, task_id TEXT,
    name TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT,
    duration_ms REAL, attrs TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS ix_requests_created ON requests(created_at);
CREATE INDEX IF NOT EXISTS ix_requests_status ON requests(status);
CREATE INDEX IF NOT EXISTS ix_requests_batch ON requests(batch_id);
CREATE INDEX IF NOT EXISTS ix_tasks_request ON tasks(request_id);
CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS ix_tasks_sha ON tasks(sha256);
CREATE INDEX IF NOT EXISTS ix_spans_request ON spans(request_id);
"""
# Columns added after the first schema: ALTER TABLE for databases created before them (indexes need them first).
MIGRATIONS = {"requests": {"attachments": "TEXT", "retry_of": "TEXT", "attempt": "INTEGER DEFAULT 1",
                           "pid": "INTEGER", "host": "TEXT"},
              "tasks": {"idem_key": "TEXT", "reused_from": "TEXT", "name": "TEXT"}}
INDEXES = """
CREATE INDEX IF NOT EXISTS ix_requests_retry ON requests(retry_of);
CREATE INDEX IF NOT EXISTS ix_tasks_idem ON tasks(idem_key);
"""
HOST = socket.gethostname()

_request: contextvars.ContextVar["_RequestCtx | None"] = contextvars.ContextVar("trace_request", default=None)
_task: contextvars.ContextVar["_TaskCtx | None"] = contextvars.ContextVar("trace_task", default=None)
_span: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("trace_span", default=None)


def now_iso() -> str:
    """UTC ISO-8601 with ms: sorts lexicographically, readable in any SQLite browser."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def config_hash(cfg) -> str:
    """Everything that can change an output. Excluded: where traces, results and uploads go, and how many
    files run at once - none of that changes a result."""
    exclude = {"trace": True, "batch": True, "store": {"output_dir", "upload_dir"}}
    return hashlib.sha256(cfg.model_dump_json(exclude=exclude).encode()).hexdigest()[:12]


def idem_key(sha256: str, versions: dict, cfg_hash: str) -> str:
    """Same file + same component versions + same config -> same output: safe to reuse instead of redo."""
    return hashlib.sha256(f"{sha256}|{_json(sorted(versions.items()))}|{cfg_hash}".encode()).hexdigest()[:24]


def _json(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


# ---------------------------------------------------------------- database

class Tracer:
    """One SQLite file. One connection per thread (sqlite3 connections must not be shared across threads);
    WAL lets bulk worker threads write while `docagent trace` reads, busy_timeout absorbs write contention."""

    def __init__(self, db_path: str | Path, enabled: bool = True):
        self.db_path = Path(db_path)
        self.enabled = enabled
        self._local = threading.local()
        self._warned = False
        if enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            c = self.conn()
            c.executescript(SCHEMA)
            for table, cols in MIGRATIONS.items():
                have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
                for col, decl in cols.items():
                    if col not in have:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            c.executescript(INDEXES)

    @classmethod
    def from_config(cls, cfg) -> "Tracer":
        return cls(cfg.path(cfg.trace.db_path), enabled=cfg.trace.enabled)

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)   # autocommit: 1 stmt = 1 txn
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=10000")
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    def execute(self, sql: str, params: tuple = ()) -> None:
        if not self.enabled:
            return
        try:
            self.conn().execute(sql, params)
        except sqlite3.Error as e:
            if not self._warned:                            # once per process: don't flood the log
                log.warning("trace write failed (tracing continues best-effort): %s", e)
                self._warned = True

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn().execute(sql, params).fetchall()

    # ------------------------------------------------------------ request lifecycle
    def enqueue(self, req, batch_id: str | None) -> None:
        """Record a request before it runs (bulk): a crash then leaves it 'queued', so --resume knows
        which files of the batch were never started."""
        self.execute("INSERT OR IGNORE INTO requests (request_id, batch_id, status, text, created_at, attachments) "
                     "VALUES (?,?,?,?,?,?)", (req.request_id, batch_id, "queued", req.text, now_iso(),
                                              _attachments_json(req)))

    @contextmanager
    def request(self, req, versions: dict | None = None, cfg_hash: str | None = None,
                batch_id: str | None = None, retry_of: str | None = None,
                attempt: int = 1) -> Iterator["_RequestCtx"]:
        """Trace one request. Status: failed on exception; else whatever the caller set (rc.status),
        else partial if a task failed, else succeeded. A queued row (bulk) is taken over, keeping created_at."""
        if not self.enabled:
            yield _RequestCtx(self, req.request_id, noop=True)
            return
        rc = _RequestCtx(self, req.request_id)
        t = now_iso()
        self.execute("INSERT INTO requests (request_id, batch_id, status, text, created_at, started_at, versions, "
                     "config_hash, attachments, retry_of, attempt, pid, host) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                     "ON CONFLICT(request_id) DO UPDATE SET status=excluded.status, started_at=excluded.started_at, "
                     "versions=excluded.versions, config_hash=excluded.config_hash, retry_of=excluded.retry_of, "
                     "attempt=excluded.attempt, pid=excluded.pid, host=excluded.host",
                     (rc.request_id, batch_id, "running", req.text, t, t, _json(versions or {}), cfg_hash,
                      _attachments_json(req), retry_of, attempt, os.getpid(), HOST))
        tokens = (_request.set(rc), _task.set(None), _span.set(None))
        t0 = time.perf_counter()
        try:
            yield rc
        except BaseException as e:
            rc.status, rc.error = "failed", f"{type(e).__name__}: {e}"
            raise
        finally:
            status = rc.status or ("partial" if rc.failed_tasks else "succeeded")
            self.execute("UPDATE requests SET status=?, ended_at=?, latency_ms=?, summary=?, error=? "
                         "WHERE request_id=?",
                         (status, now_iso(), (time.perf_counter() - t0) * 1000,
                          _json(rc.summary) if rc.summary is not None else None, rc.error, rc.request_id))
            _request.reset(tokens[0])
            _task.reset(tokens[1])
            _span.reset(tokens[2])

    def mark_interrupted(self, stale_s: float = 900) -> int:
        """Rows still 'running' that belong to a process that died: mark them 'interrupted' so they show up
        as failures (and become resumable) instead of 'in progress' forever. Dead = its pid no longer exists
        on this host (known at once), or - for other hosts / unknown pid - no activity for stale_s (latest
        span start, or the request's own start; default 15 min > one VLM call's timeout)."""
        if not self.enabled:
            return 0
        cutoff = datetime.fromtimestamp(time.time() - stale_s, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        rows = self.query(
            "SELECT r.request_id, r.pid, r.host, COALESCE((SELECT MAX(s.started_at) FROM spans s "
            "WHERE s.request_id=r.request_id), r.started_at) AS last FROM requests r WHERE r.status='running'")
        stale = [r["request_id"] for r in rows
                 if (r["host"] == HOST and r["pid"] and not _alive(r["pid"])) or (r["last"] or "") < cutoff]
        t = now_iso()
        for rid in stale:
            self.execute("UPDATE requests SET status='interrupted', ended_at=?, "
                         "error=COALESCE(error, 'process ended while running') WHERE request_id=?", (t, rid))
            self.execute("UPDATE tasks SET status='interrupted', ended_at=? WHERE request_id=? AND status='running'",
                         (t, rid))
            self.execute("UPDATE spans SET status='interrupted', ended_at=? WHERE request_id=? AND status='running'",
                         (t, rid))
        if stale:
            log.warning("marked %d stale running request(s) as interrupted", len(stale))
        return len(stale)


    # ------------------------------------------------------------ retry / reuse lookups
    def find_reusable(self, key: str) -> sqlite3.Row | None:
        """Latest finished drawing task with this idem_key whose output file still exists."""
        if not self.enabled:
            return None
        for r in self.query("SELECT * FROM tasks WHERE idem_key=? AND status IN ('succeeded','reused') "
                            "AND route='drawing' AND output_path IS NOT NULL ORDER BY ended_at DESC LIMIT 20", (key,)):
            if Path(r["output_path"]).exists():
                return r
        return None

    def chain(self, request_id: str) -> list[sqlite3.Row]:
        """All attempts of the retry chain this request belongs to, oldest first."""
        root = request_id
        while (r := self.query("SELECT retry_of FROM requests WHERE request_id=?", (root,))) and r[0]["retry_of"]:
            root = r[0]["retry_of"]
        return self.query(
            "WITH RECURSIVE c(id) AS (SELECT ? UNION SELECT r.request_id FROM requests r JOIN c ON r.retry_of=c.id) "
            "SELECT * FROM requests WHERE request_id IN (SELECT id FROM c) ORDER BY attempt, created_at", (root,))


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                                 # exists, owned by someone else
        return True
    return True


def _attachments_json(req) -> str:
    return _json([{"path": a.path, "sha256": a.sha256, "mime": a.mime, "name": a.name} for a in req.attachments])


# ---------------------------------------------------------------- contexts

class _RequestCtx:
    def __init__(self, tracer: Tracer, request_id: str, noop: bool = False):
        self.tracer, self.request_id, self.noop = tracer, request_id, noop
        self.status: str | None = None          # caller may decide (e.g. partial from Response.errors)
        self.error: str | None = None
        self.summary: dict | None = None        # small JSON shown by `docagent trace` (not the full output)
        self.failed_tasks = 0
        self._lock = threading.Lock()

    def add_llm_call(self, prompt_tokens: int | None, completion_tokens: int | None, cache_hit: bool) -> None:
        if self.noop:
            return
        # increments in SQL, not in memory: totals are right even if the process dies mid-request
        if cache_hit:          # replayed reply: no model work, so no tokens - spend totals are fresh calls only
            self.tracer.execute("UPDATE requests SET cache_hits=cache_hits+1 WHERE request_id=?", (self.request_id,))
            return
        self.tracer.execute("UPDATE requests SET llm_calls=llm_calls+1, "
                            "prompt_tokens=prompt_tokens+?, completion_tokens=completion_tokens+? WHERE request_id=?",
                            (prompt_tokens or 0, completion_tokens or 0, self.request_id))


class _TaskCtx:
    _COLUMNS = {"route", "tier", "confidence", "reason", "stage", "output_path", "attempts"}

    def __init__(self, rc: _RequestCtx | None, task_id: str):
        self.rc, self.task_id = rc, task_id
        self.failed = False

    def set(self, **fields) -> None:
        if not self.rc or self.rc.noop:
            return
        cols = [k for k in fields if k in self._COLUMNS]
        if cols:
            self.rc.tracer.execute(f"UPDATE tasks SET {', '.join(c + '=?' for c in cols)} WHERE task_id=?",
                                   (*[fields[c] for c in cols], self.task_id))

    def fail(self, exc: BaseException) -> None:
        """Record a handled failure (the caller continues with the next attachment)."""
        self.failed = True
        if not self.rc or self.rc.noop:
            return
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.rc.tracer.execute("UPDATE tasks SET status='failed', error=?, traceback=?, ended_at=? WHERE task_id=?",
                               (f"{type(exc).__name__}: {exc}", tb[-8000:], now_iso(), self.task_id))


@contextmanager
def task(attachment: str | None = None, sha256: str | None = None, decision=None,
         idem: str | None = None, name: str | None = None) -> Iterator[_TaskCtx]:
    """Trace one attachment. Unhandled exceptions mark it failed and propagate; `t.fail(e)` records a
    handled one."""
    rc = _request.get()
    t = _TaskCtx(rc, new_id())
    if rc is None or rc.noop:
        yield t
        return
    d = decision
    rc.tracer.execute(
        "INSERT INTO tasks (task_id, request_id, attachment, sha256, route, tier, confidence, reason, status, "
        "stage, started_at, heartbeat_at, idem_key, name) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (t.task_id, rc.request_id, attachment, sha256, d.route.value if d else None, d.tier if d else None,
         d.confidence if d else None, d.reason if d else None, "running", "started", now_iso(), now_iso(), idem,
         name or (Path(attachment).name if attachment else None)))
    token = _task.set(t)
    try:
        yield t
    except BaseException as e:
        t.fail(e)
        raise
    finally:
        _task.reset(token)
        if t.failed:
            with rc._lock:
                rc.failed_tasks += 1
        else:
            rc.tracer.execute("UPDATE tasks SET status='succeeded', stage='done', ended_at=? WHERE task_id=?",
                              (now_iso(), t.task_id))


def record_reused(attachment: str, sha256: str, src: sqlite3.Row, name: str | None = None) -> None:
    """A task satisfied by an earlier finished task with the same idem_key: no processing, the row points
    at the existing output (no copy - one file per result)."""
    rc = _request.get()
    if rc is None or rc.noop:
        return
    t = now_iso()
    rc.tracer.execute(
        "INSERT INTO tasks (task_id, request_id, attachment, sha256, route, tier, confidence, reason, status, stage, "
        "output_path, started_at, ended_at, idem_key, reused_from, name) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (new_id(), rc.request_id, attachment, sha256, src["route"], src["tier"], src["confidence"], src["reason"],
         "reused", "done", src["output_path"], t, t, src["idem_key"], src["reused_from"] or src["task_id"],
         name or Path(attachment).name))


class Span:
    """One step. Inserted on enter (status running), updated on exit with duration, attrs, status."""

    def __init__(self, name: str, attrs: dict):
        self.name, self.attrs = name, dict(attrs)
        self.rc = _request.get()
        self.active = self.rc is not None and not self.rc.noop
        self.span_id = new_id()

    def set(self, **attrs) -> "Span":
        self.attrs.update(attrs)
        return self

    def __enter__(self) -> "Span":
        if not self.active:
            return self
        parent, tk = _span.get(), _task.get()
        self._t0 = time.perf_counter()
        tr = self.rc.tracer
        tr.execute("INSERT INTO spans (span_id, parent_id, request_id, task_id, name, status, started_at, attrs) "
                   "VALUES (?,?,?,?,?,?,?,?)",
                   (self.span_id, parent.span_id if parent and parent.active else None, self.rc.request_id,
                    tk.task_id if tk else None, self.name, "running", now_iso(), _json(self.attrs)))
        if tk:                                              # activity on the task = its heartbeat
            tr.execute("UPDATE tasks SET heartbeat_at=?, stage=? WHERE task_id=?", (now_iso(), self.name, tk.task_id))
        self._token = _span.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.active:
            return False
        _span.reset(self._token)
        self.rc.tracer.execute(
            "UPDATE spans SET status=?, ended_at=?, duration_ms=?, attrs=?, error=? WHERE span_id=?",
            ("failed" if exc else "succeeded", now_iso(), (time.perf_counter() - self._t0) * 1000,
             _json(self.attrs), f"{exc_type.__name__}: {exc}"[:2000] if exc else None, self.span_id))
        return False                                        # never swallow the exception


def span(name: str, **attrs) -> Span:
    """`with span("qa.retrieve", query=q) as sp: ...; sp.set(n=len(items))` - no-op outside a request."""
    return Span(name, attrs)


def record_llm_call(prompt_tokens: int | None, completion_tokens: int | None, cache_hit: bool) -> None:
    """Add one model call to the current request's totals (the call's own span holds the details).
    Fresh call: llm_calls + tokens. Cache hit: cache_hits only - its tokens are never spend."""
    rc = _request.get()
    if rc is not None:
        rc.add_llm_call(prompt_tokens, completion_tokens, cache_hit)


def current_ids() -> dict[str, str | None]:
    rc, tk, sp = _request.get(), _task.get(), _span.get()
    return {"request_id": rc.request_id if rc else None, "task_id": tk.task_id if tk else None,
            "span_id": sp.span_id if sp and sp.active else None}


# ---------------------------------------------------------------- threads

def submit(executor: Executor, fn: Callable, *args, **kwargs) -> Future:
    """executor.submit that keeps the trace context: worker threads do not inherit contextvars, so without
    this every span opened in a worker would lose its request and parent."""
    return executor.submit(contextvars.copy_context().run, fn, *args, **kwargs)


def wrap(fn: Callable) -> Callable:
    """For executor.map: captures the context now, runs each call in its own copy (one Context object
    cannot be entered by two threads at once)."""
    ctx = contextvars.copy_context()
    return lambda *a, **kw: ctx.copy().run(fn, *a, **kw)


# ---------------------------------------------------------------- logging

class JsonLinesHandler(logging.Handler):
    """Log records as JSON lines stamped with the trace ids, so a log line joins to its request/span.
    Narrative only: metrics are read from the database, never parsed from logs."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)   # line-buffered: survives a crash
        self._lock2 = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
                    "level": record.levelname, "logger": record.name, "msg": record.getMessage(), **current_ids()}
            if record.exc_info:
                line["exc"] = "".join(traceback.format_exception(*record.exc_info))[-4000:]
            with self._lock2:
                self._fh.write(_json(line) + "\n")
        except Exception:
            self.handleError(record)


def install_log_file(cfg) -> None:
    """Add the JSON-lines file at INFO, keeping existing (console) handlers at the level they had."""
    if not cfg.trace.log_path:
        return
    root = logging.getLogger()
    if any(isinstance(h, JsonLinesHandler) for h in root.handlers):
        return
    for h in root.handlers:
        if h.level == logging.NOTSET:
            h.setLevel(root.level)
    root.setLevel(min(root.level, logging.INFO))
    h = JsonLinesHandler(cfg.path(cfg.trace.log_path))
    h.setLevel(logging.INFO)
    root.addHandler(h)
