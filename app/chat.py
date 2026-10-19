"""Local chat UI. Run: .venv/bin/python app/chat.py  (http://127.0.0.1:7860, bound to localhost only)

The user types text and/or attaches pictures / PDFs / zips; the agent decides what to do.
- One file, or files with a question: answered in the reply - one request (the question covers all pictures).
- Several files and no question (batch.chat_min_files), or a zip: a background batch - each file its own
  request, the same code as `docagent bulk`. The reply shows the batch id at once and streams progress read
  from the trace database; a reload or closed tab loses nothing: send `status <batch_id>` to watch it again
  and get the results zip.
- On start: requests of a crashed run are marked interrupted and unfinished chat batches resume (only files
  still queued or interrupted - a file that failed is not retried automatically; `docagent bulk --resume`).
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # privacy: no telemetry calls

import gradio as gr

from docagent import batch
from docagent.agent import Agent
from docagent.render import render_markdown
from docagent.trace import install_log_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "httpcore", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

agent = Agent()                      # marks requests of a crashed run 'interrupted'
install_log_file(agent.cfg)          # JSON lines with request ids; traces: docagent trace <request_id>
runner = batch.BatchRunner(agent, agent.cfg.batch.workers)
RESUMED = runner.resume_unfinished("chat-")
STATUS_RE = re.compile(r"^\s*status\s+((?:chat|bulk)-[\w-]+)\s*$", re.I)
POLL_S = 1.0


def watch(batch_id: str, intro: str = ""):
    """Stream a batch's progress from the database until it is finished, then offer the results zip.
    The generator only reads: stopping it (tab closed) does not touch the processing."""
    shown = None
    while True:
        st = batch.status(agent.tracer, batch_id)
        if not st["files"]:
            yield f"No batch `{batch_id}` found."
            return
        text = intro + batch.format_status(st, markdown=True)
        if st["finished"]:
            z = batch.build_zip(agent.tracer, batch_id,
                                agent.cfg.path(agent.cfg.store.output_dir).parent / "batches" / f"{batch_id}.zip")
            yield [text, gr.File(str(z), label=z.name)]
            return
        if text != shown:
            shown = text
            yield (text + f"\n\n_Running in the background: closing this tab does not stop it. "
                          f"Send `status {batch_id}` to check again._")
        time.sleep(POLL_S)


def respond(message: dict, history, force: bool = False):
    text = (message.get("text") or "").strip()
    files = [f if isinstance(f, str) else f.get("path") for f in message.get("files", [])]
    if not files and (m := STATUS_RE.match(text)):
        yield from watch(m.group(1))
        return
    has_zip = any(str(f).lower().endswith(".zip") for f in files)
    if has_zip or (not text and len(files) >= agent.cfg.batch.chat_min_files):
        atts, notes = agent.store_files(files)
        if text:
            notes.append("the question is not applied to a batch: each file is processed on its own")
        if not atts:
            yield "Nothing to process. " + "; ".join(notes)
            return
        batch_id, more = runner.start(atts, force=force)
        intro = f"Batch `{batch_id}` started with {len(atts)} file(s).\n\n"
        if notes + more:
            intro += "_Notes: " + "; ".join(notes + more) + "_\n\n"
        yield intro
        yield from watch(batch_id, intro)
        return
    # a drawing already processed (same file, agent version, config) is reused unless forced; the reply says so
    resp = agent.ask(text or None, files, force=force)
    out: list = [render_markdown(resp)]
    for p in resp.outputs:                     # written by the agent: <output_dir>/<request_id>/ (in the trace)
        if not p.endswith("answer.json"):
            out.append(gr.File(p, label=Path(p).name))
    yield out


demo = gr.ChatInterface(
    respond,
    multimodal=True,
    title="Document agent (local)",
    description="Ask about the NTSB materials reports, send a photo of a damaged part, or upload technical "
                "drawings to get their values as JSON. Several files without a question (or a zip) run as a "
                "background batch: `status <batch_id>` shows its progress again.",
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image", ".pdf", ".zip"]),
    additional_inputs=[gr.Checkbox(False, label="Force re-run (ignore previous results)",
                                   info="Processes drawings again instead of reusing an earlier result for the same "
                                        "file. Model calls still replay from the response cache.")],
    additional_inputs_accordion=gr.Accordion("Options", open=True),
)

if __name__ == "__main__":
    if RESUMED:
        print(f"resumed unfinished batches: {', '.join(RESUMED)}")
    demo.launch(server_name="127.0.0.1", share=False)
