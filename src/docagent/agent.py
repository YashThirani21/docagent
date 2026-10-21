"""Entry point: Request -> route each attachment -> run the pipelines -> Response. Retry of a traced request.

Result files go to <store.output_dir>/<request_id>/ (drawing JSON per drawing, answer.json), written by the
agent itself so chat, ask and bulk all record them the same way; the request id is the trace key, so a trace
leads straight to its files. A reused result is referenced where it already is, never copied.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from PIL import Image

from .config import Config, load_config
from .llm import VLM
from .pdf import attachment_images
from .pipelines.drawing import DrawingExtractor
from .pipelines.report_qa import ReportQA
from .router import Router
from .schemas import Attachment, DrawingExtraction, Request, Response, Route, RouteDecision
from .trace import Tracer, config_hash, idem_key, record_reused, span, task
from .uploads import expand

log = logging.getLogger(__name__)


class RetryError(Exception):
    """A retry that is refused or impossible; the message says why."""


class Agent:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.vlm = VLM(self.cfg.llm)
        self.router = Router(self.cfg, self.vlm)
        self.drawings = DrawingExtractor(self.cfg, self.vlm)
        self._qa: ReportQA | None = None          # lazy: loads the embedder + opens the index
        self._qa_lock = threading.Lock()           # chat batches run in threads: open the (embedded) index once
        self.tracer = Tracer.from_config(self.cfg)
        self.tracer.mark_interrupted()             # leftovers of a process that died mid-request
        self.config_hash = config_hash(self.cfg)
        self.out_root = self.cfg.path(self.cfg.store.output_dir)

    @property
    def qa(self) -> ReportQA:
        with self._qa_lock:
            if self._qa is None:
                with span("qa.load_index"):        # first request only: embedder + vector store
                    self._qa = ReportQA(self.cfg, vlm=self.vlm)
        return self._qa

    @property
    def versions(self) -> dict[str, str]:
        drawing = self.drawings.geometry_version if self.cfg.drawing.method == "geometry" else self.drawings.version
        return {"router": self.router.version, "drawing": drawing,
                "report_qa": ReportQA.version, "llm": self.cfg.llm.model}

    def idem_key(self, att: Attachment) -> str:
        """What one attachment's result depends on: its bytes, router + drawing versions, model, config."""
        v = self.versions
        return idem_key(att.sha256, {k: v[k] for k in ("router", "drawing", "llm")}, self.config_hash)

    def store_files(self, files: list[str | Path]) -> tuple[list[Attachment], list[str]]:
        """Submitted files (folders, zips expanded) -> copies in the upload store + notes on skipped members.
        From here on nothing depends on the original location (Gradio temp dirs vanish, users move files)."""
        return expand(files, self.cfg)

    def handle(self, req: Request, batch_id: str | None = None, reuse: bool = False,
               retry_of: str | None = None, attempt: int = 1) -> Response:
        """reuse=True: a drawing whose result already exists for the same idem_key is taken from there
        instead of being processed again (ask / chat / retry / bulk by default; `force` turns it off)."""
        with self.tracer.request(req, self.versions, self.config_hash, batch_id, retry_of, attempt) as rc:
            resp = self._handle(req, reuse)
            rc.summary = _summary(resp)
            if resp.errors:                           # some attachment failed: partial if anything came out
                rc.status = "partial" if resp.drawings or resp.report_answer else "failed"
                rc.error = "; ".join(resp.errors)[:2000]
            return resp

    def _handle(self, req: Request, reuse: bool = False) -> Response:
        resp = Response(request_id=req.request_id, routes=[])
        out_dir = self.out_root / req.request_id
        if not req.attachments:
            with span("router", n_attachments=0) as sp:
                resp.routes = self.router.route(req)
                sp.set(route=resp.routes[0].route.value, tier=resp.routes[0].tier)

        # Each attachment is one task: routed and processed in isolation, so a corrupt PDF, an unreadable
        # image or a failed classify call fails that task only; the others still run (request -> partial).
        qa_images: list[Image.Image] = []
        qa_tasks = []
        for att in req.attachments:
            key = self.idem_key(att)
            if reuse and (src := self.tracer.find_reusable(key)):
                record_reused(att.path, att.sha256, src, att.name)
                origin = Path(src["output_path"]).parent.name        # the request that produced the file
                resp.routes.append(RouteDecision(route=Route(src["route"]), attachment=att.path, tier=src["tier"],
                                                 confidence=src["confidence"],
                                                 reason=f"reused result from request {origin}: {src['reason']}"))
                drawing = DrawingExtraction.model_validate_json(Path(src["output_path"]).read_text())
                resp.drawings.append(drawing)
                resp.reused[drawing.drawing_id] = origin
                resp.outputs.append(src["output_path"])
                continue
            with task(att.path, att.sha256, idem=key, name=att.name) as t:
                try:
                    with span("router", attachment=att.name) as sp:
                        d = self.router.route_attachment(att)
                        sp.set(route=d.route.value, tier=d.tier, confidence=d.confidence)
                    t.set(route=d.route.value, tier=d.tier, confidence=d.confidence, reason=d.reason)
                    resp.routes.append(d)
                    if d.route == Route.DRAWING:
                        drawing = self.drawings.extract(att)
                        path = write_json_atomic(_unique(out_dir / f"{drawing.drawing_id}.json"),
                                                 drawing.model_dump(mode="json"))
                        t.set(output_path=str(path))     # before the task closes: succeeded only with its file
                        resp.drawings.append(drawing)
                        resp.outputs.append(str(path))
                    else:
                        qa_images += attachment_images(att)
                        t.set(stage="images_to_report_qa")
                        qa_tasks.append(t)
                except Exception as e:
                    log.exception("attachment failed: %s (%s)", att.name, att.path)
                    resp.errors.append(f"{att.name}: {type(e).__name__}: {e}")
                    t.fail(e)

        # answer from the reports when there is a question or a non-drawing picture
        if qa_images or (req.text and not resp.drawings):
            # isolated like the attachments above: a failed answer must not take the drawings with it
            try:
                with span("report_qa", has_text=bool(req.text), n_images=len(qa_images)):
                    resp.report_answer = self.qa.run(req.text, qa_images)
                # one joint answer for the question + all pictures: never reused, its tasks just point at it
                path = write_json_atomic(out_dir / "answer.json", resp.report_answer.model_dump(mode="json"))
                resp.outputs.append(str(path))
                for t in qa_tasks:
                    t.set(output_path=str(path))
            except Exception as e:
                log.exception("report answer failed (request %s)", req.request_id)
                resp.errors.append(f"report answer: {type(e).__name__}: {e}")
                for t in qa_tasks:          # closed already; fail() is an update by task id
                    t.fail(e)
        return resp

    # ------------------------------------------------------------ retry
    def request_from_row(self, row) -> Request:
        """Rebuild a traced request (same text, same files). The files must still be there, unchanged."""
        atts = json.loads(row["attachments"]) if row["attachments"] else [
            {"path": t["attachment"], "sha256": t["sha256"]}       # traces from before the attachments column
            for t in self.tracer.query("SELECT * FROM tasks WHERE request_id=? ORDER BY started_at",
                                       (row["request_id"],))]
        problems, out = [], []
        for a in atts:
            shown = f"{a.get('name') or Path(a['path']).name} ({a['path']})"
            if not Path(a["path"]).exists():
                problems.append(f"{shown}: file no longer exists")
                continue
            now = Attachment.from_path(a["path"], name=a.get("name"))   # old rows: no name = file name
            if now.sha256 != a["sha256"]:
                problems.append(f"{shown}: file changed since the request (sha256 {a['sha256'][:12]} -> "
                                f"{now.sha256[:12]}); submit it as a new request instead")
            out.append(now)
        if problems:
            raise RetryError("cannot retry " + row["request_id"] + ":\n  " + "\n  ".join(problems))
        return Request(request_id=row["request_id"], text=row["text"], attachments=out)

    def retry(self, request_id: str, force: bool = False, batch_id: str | None = None) -> tuple[Response, list[str]]:
        """New attempt of a traced request (retry_of = latest attempt, attempt + 1). Finished results with the
        same idem_key are reused; the rest re-runs, its finished model calls replaying from the VLM cache.
        Returns the response and notes for the user."""
        rows = self.tracer.query("SELECT * FROM requests WHERE request_id LIKE ? || '%'", (request_id,))
        if len(rows) != 1:
            raise RetryError(f"{len(rows)} requests match {request_id!r}")
        prev = self.tracer.chain(rows[0]["request_id"])[-1]           # always continue from the latest attempt
        notes = [] if prev["request_id"] == rows[0]["request_id"] else [
            f"{rows[0]['request_id']} was already retried; continuing from attempt {prev['attempt']} ({prev['request_id']})"]
        if prev["status"] == "running":
            raise RetryError(f"{prev['request_id']} is still running (pid {prev['pid']} on {prev['host']})")
        if prev["status"] == "succeeded" and not force:
            raise RetryError(f"{prev['request_id']} (attempt {prev['attempt']}) already succeeded; --force to run it again")
        req = self.request_from_row(prev).model_copy(update={"request_id": Request().request_id})
        changed = prev["config_hash"] != self.config_hash or json.loads(prev["versions"] or "{}") != self.versions
        if changed:
            notes.append(f"config/version changed since attempt {prev['attempt']} (config {prev['config_hash']} -> "
                         f"{self.config_hash}): nothing is reused, every attachment runs again")
        elif force:
            notes.append("--force: nothing is reused, every attachment runs again (model calls still cached)")
        resp = self.handle(req, batch_id or prev["batch_id"], reuse=not (changed or force),
                           retry_of=prev["request_id"], attempt=(prev["attempt"] or 1) + 1)
        return resp, notes

    def ask(self, text: str | None = None, files: list[str | Path] | None = None,
            batch_id: str | None = None, force: bool = False) -> Response:
        """force=True: process every attachment again instead of reusing a finished result for the same file.
        The VLM response cache still applies (bypassing it = a config change: llm.cache_dir)."""
        atts, notes = self.store_files(files or [])
        resp = self.handle(Request(text=text or None, attachments=atts), batch_id, reuse=not force)
        resp.notes += notes
        return resp


def _summary(resp: Response) -> dict:
    """What `docagent trace` shows as the outcome - small; the full drawing JSON lives in outputs/."""
    out: dict = {"routes": [d.route.value for d in resp.routes], "errors": resp.errors,
                 "drawings": [{"drawing_id": d.drawing_id, "values": len(d.values), "pages": len(d.pages)}
                              for d in resp.drawings]}
    if a := resp.report_answer:
        out["answer"] = a.answer[:1000]
        out["grounded"] = a.grounded
        out["citations"] = [f"{c.doc_id} p{c.page}" + (f" fig{c.figure_no}" if c.figure_no else "")
                            for c in a.citations]
    return out


def write_json_atomic(path: Path, obj) -> Path:
    """Write to a temp file in the same directory, then rename: a crash never leaves a half-written result
    that a later retry would take for a finished one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)
    return path


def _unique(path: Path) -> Path:
    """Same drawing id twice in one request (e.g. a PDF and a photo of it): keep both."""
    n, p = 2, path
    while p.exists():
        p, n = path.with_name(f"{path.stem}_{n}{path.suffix}"), n + 1
    return p


def save_drawing_json(resp: Response, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for d in resp.drawings:
        p = out_dir / f"{d.drawing_id}.json"
        p.write_text(json.dumps(d.model_dump(mode="json"), indent=2, ensure_ascii=False))
        paths.append(p)
    return paths
