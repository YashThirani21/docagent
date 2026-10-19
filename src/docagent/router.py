"""Decide per attachment which pipeline applies. Cheap deterministic rules first, VLM only if unsure.

- no attachment                       -> report QA (text question)
- PDF with drawing signals            -> drawing     (title block / ASME / tolerance text, landscape,
                                                      dense vector linework, little raster imagery)
- PDF with report-like signals        -> report QA   (long prose, portrait, photos)
- images, or PDFs the rules can't call -> VLM classification on a downscaled render
"""
from __future__ import annotations

import logging
import re
from typing import Literal

import pymupdf
from PIL import Image
from pydantic import BaseModel, Field

from .config import Config
from .llm import VLM, fit_image
from .pdf import page_stats, render
from .schemas import Attachment, Request, Route, RouteDecision

log = logging.getLogger(__name__)

DRAWING_TEXT_RE = re.compile(
    r"(ASME\s+Y14|UNLESS OTHERWISE SPECIFIED|TOLERANC|\bUNITS?:|\bSCALE\b|\bSHEET\b|\bDWG\b|"
    r"\bREV\b\.?\s*[A-Z]\b|Ø|±|\bPMI\b|\bDATUM)", re.I)


class _Kind(BaseModel):
    kind: Literal["technical_drawing", "photo_of_part", "document_page", "chart_or_plot", "other"]
    confidence: float = Field(description="0..1")
    reason: str


CLASSIFY_PROMPT = """Classify this image.
technical_drawing = an engineering/CAD drawing with dimensions, tolerances or GD&T (including a photo
or screenshot of such a drawing). photo_of_part = a photograph, radiograph or micrograph of a physical
part, damage or fracture. document_page = a page of a text report. chart_or_plot = a graph."""


class Router:
    version = "rules+vlm-v1"

    def __init__(self, cfg: Config, vlm: VLM | None = None):
        self.cfg = cfg
        self.vlm = vlm or VLM(cfg.llm)

    def route(self, req: Request) -> list[RouteDecision]:
        if not req.attachments:
            return [RouteDecision(route=Route.REPORT_QA, confidence=1.0, tier="rule",
                                  reason="text only, no attachment")]
        return [self.route_attachment(a) for a in req.attachments]

    def route_attachment(self, att: Attachment) -> RouteDecision:
        if att.is_pdf:
            d = self._pdf_rules(att)
            if d:
                return d
            with pymupdf.open(att.path) as doc:
                img = render(doc[0], 80)
            return self._vlm(att, img)
        if att.is_image:
            return self._vlm(att, Image.open(att.path).convert("RGB"))
        return RouteDecision(route=Route.REPORT_QA, attachment=att.path, confidence=0.3, tier="rule",
                             reason=f"unsupported type {att.mime}; treated as context for a question")

    def _pdf_rules(self, att: Attachment) -> RouteDecision | None:
        with pymupdf.open(att.path) as doc:
            s = page_stats(doc[0])
        hits = len(set(m.group(0).upper() for m in DRAWING_TEXT_RE.finditer(s.text)))
        score = (hits >= 2) + s.landscape + (s.vector_paths > 300) + (s.image_coverage < 0.1)
        prose = s.chars > 1500 and not s.landscape
        why = (f"text_signals={hits} landscape={s.landscape} vector_paths={s.vector_paths} "
               f"image_coverage={s.image_coverage:.2f} chars={s.chars}")
        if score >= 3 and not prose:
            return RouteDecision(route=Route.DRAWING, attachment=att.path, confidence=0.6 + 0.1 * score,
                                 tier="rule", reason="drawing signals: " + why)
        if prose or (score <= 1 and s.image_coverage > 0.3):
            return RouteDecision(route=Route.REPORT_QA, attachment=att.path, confidence=0.8, tier="rule",
                                 reason="report/photo signals: " + why)
        return None

    def _vlm(self, att: Attachment, img: Image.Image) -> RouteDecision:
        k = self.vlm.structured(CLASSIFY_PROMPT, _Kind,
                                images=[fit_image(img, self.cfg.router.classify_max_side)])
        route = Route.DRAWING if k.kind == "technical_drawing" else Route.REPORT_QA
        return RouteDecision(route=route, attachment=att.path, confidence=k.confidence, tier="vlm",
                             reason=f"{k.kind}: {k.reason}")
