"""Entry point: Request -> route each attachment -> run the pipelines -> Response."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image

from .config import Config, load_config
from .llm import VLM
from .pdf import attachment_images
from .pipelines.drawing import DrawingExtractor
from .pipelines.report_qa import ReportQA
from .router import Router
from .schemas import Request, Response, Route

log = logging.getLogger(__name__)


class Agent:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.vlm = VLM(self.cfg.llm)
        self.router = Router(self.cfg, self.vlm)
        self.drawings = DrawingExtractor(self.cfg, self.vlm)
        self._qa: ReportQA | None = None          # lazy: loads the embedder + opens the index

    @property
    def qa(self) -> ReportQA:
        if self._qa is None:
            self._qa = ReportQA(self.cfg, vlm=self.vlm)
        return self._qa

    @property
    def versions(self) -> dict[str, str]:
        return {"router": self.router.version, "drawing": self.drawings.version,
                "report_qa": ReportQA.version, "llm": self.cfg.llm.model}

    def handle(self, req: Request) -> Response:
        routes = self.router.route(req)
        resp = Response(request_id=req.request_id, routes=routes)
        by_path = {a.path: a for a in req.attachments}

        qa_images: list[Image.Image] = []
        for d in routes:
            if d.attachment is None:
                continue
            att = by_path[d.attachment]
            if d.route == Route.DRAWING:
                try:
                    resp.drawings.append(self.drawings.extract(att))
                except Exception as e:                    # one bad file must not kill a bulk request
                    log.exception("drawing extraction failed for %s", att.path)
                    resp.errors.append(f"{Path(att.path).name}: {e}")
            else:
                qa_images += attachment_images(att)

        # answer from the reports when there is a question or a non-drawing picture
        if qa_images or (req.text and not resp.drawings):
            resp.report_answer = self.qa.run(req.text, qa_images)
        return resp

    def ask(self, text: str | None = None, files: list[str | Path] | None = None) -> Response:
        return self.handle(Request.build(text, files))


def save_drawing_json(resp: Response, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for d in resp.drawings:
        p = out_dir / f"{d.drawing_id}.json"
        p.write_text(json.dumps(d.model_dump(mode="json"), indent=2, ensure_ascii=False))
        paths.append(p)
    return paths
