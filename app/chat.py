"""Local chat UI. Run: .venv/bin/python app/chat.py  (http://127.0.0.1:7860, bound to localhost only)

The user types text and/or attaches pictures/PDFs (several at once = bulk); the agent decides what to do.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # privacy: no telemetry calls

import gradio as gr

from docagent.agent import Agent, save_drawing_json
from docagent.config import PROJECT_ROOT
from docagent.render import render_markdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

agent = Agent()
OUT = PROJECT_ROOT / "outputs" / "chat"


def respond(message: dict, history):
    files = [f if isinstance(f, str) else f.get("path") for f in message.get("files", [])]
    resp = agent.ask(message.get("text") or None, files)
    out: list = [render_markdown(resp)]
    for p in save_drawing_json(resp, OUT / resp.request_id):
        out.append(gr.File(str(p), label=Path(p).name))
    return out


demo = gr.ChatInterface(
    respond,
    multimodal=True,
    title="Document agent (local)",
    description="Ask about the NTSB materials reports, send a photo of a damaged part, "
                "or upload technical drawings to get their values as JSON.",
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image", ".pdf"]),
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
