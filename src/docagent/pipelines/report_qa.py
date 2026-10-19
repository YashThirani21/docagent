"""Pipeline A: answer text and/or picture requests from the NTSB report corpus.

Flow: (pictures -> VLM observations) -> retrieval query -> hybrid search (+ linked figures)
      -> VLM answer over numbered sources (+ user pictures + top figure crops) -> citations checked.
"""
from __future__ import annotations

import logging

from PIL import Image
from pydantic import BaseModel, Field

from ..config import Config
from ..llm import VLM, fit_image
from ..retriever import HybridRetriever
from ..schemas import Citation, ReportAnswer, RetrievedItem

log = logging.getLogger(__name__)


class ImageObservation(BaseModel):
    component: str = Field(description="What part/object is shown, as specifically as possible")
    observations: list[str] = Field(description="Visible conditions: fracture features, deformation, "
                                                "corrosion, wear, cracks, markings")
    visible_text: str = Field(description="Any text/labels/scale markings readable in the picture, or ''")


class _Generated(BaseModel):
    answer: str
    diagnosis: str = Field(description="What is wrong, if the user shows/describes a problem; else ''")
    recommended_actions: list[str] = Field(description="What to do; [] if not applicable")
    cited_sources: list[int] = Field(description="Numbers of the sources the answer relies on")
    grounded: bool = Field(description="true only if the sources actually support the answer")


OBSERVE_PROMPT = """You are a failure-analysis engineer. Look at this picture of an aircraft/mechanical
component (it may be a photo, radiograph, micrograph or fracture surface).
Report only what is visible, precisely and factually. Use metallurgical terms only for features you can
actually see; do not infer the cause of failure."""

SYSTEM = """You are a materials-failure assistant for aircraft components. You answer ONLY from the numbered
sources, which are excerpts and figures from NTSB Materials Laboratory Factual Reports.
Rules:
- Every factual claim must come from the sources; cite them by number in cited_sources.
- The report's conclusions (e.g. "consistent with bending overstress", "fatigue cracking") are
  authoritative. Never contradict them or substitute a different failure mode.
- Several aircraft/parts can appear in one report: keep registration numbers and parts straight.
- If the user provided a picture, compare it with similar cases in the sources: say what is wrong
  (diagnosis) and what to do (recommended_actions). The reports state findings, not instructions, so
  recommendations must follow from the findings (e.g. fatigue at a bolt hole -> inspect that location on
  similar parts) - say they are inferred from the report findings.
- If the sources do not cover the question, set grounded=false and say so plainly. Do not guess.
- Be concise and specific: part names, failure modes, locations, measurements, report numbers."""


def _format_sources(items: list[RetrievedItem]) -> str:
    lines = []
    for n, it in enumerate(items, start=1):
        c = it.chunk
        tag = f"Figure {c.figure_no}" if c.kind == "figure" and c.figure_no else c.kind
        lines.append(f"[{n}] report {c.doc_id}, page {c.page}, {tag}"
                     f"{', section ' + c.section if c.section else ''}\n{c.text}")
    return "\n\n".join(lines)


class ReportQA:
    version = "report-qa-v1"

    def __init__(self, cfg: Config, retriever: HybridRetriever | None = None, vlm: VLM | None = None):
        self.cfg = cfg
        self.retriever = retriever or HybridRetriever(cfg)
        self.vlm = vlm or VLM(cfg.llm)

    # ------------------------------------------------------------ steps (public for notebooks/eval)
    def observe(self, image: Image.Image) -> ImageObservation:
        return self.vlm.structured(OBSERVE_PROMPT, ImageObservation, images=[fit_image(image, 1036)])

    def build_query(self, text: str | None, observations: list[ImageObservation]) -> str:
        parts = [text.strip()] if text and text.strip() else []
        for o in observations:
            # observations only - no failure-mode guess: it biased retrieval and the answer
            parts.append(f"{o.component}: {'; '.join(o.observations)}")
        return "\n".join(parts) or "What failed and why?"

    def retrieve(self, query: str) -> list[RetrievedItem]:
        return self.retriever.search(query)

    def generate(self, question: str | None, items: list[RetrievedItem], user_images: list[Image.Image],
                 observations: list[ImageObservation]) -> ReportAnswer:
        figs = [it for it in items if it.chunk.kind == "figure" and it.chunk.image_path]
        figs = figs[: self.cfg.retrieval.max_context_figures]
        images = [fit_image(im, 896) for im in user_images]
        fig_note = ""
        for k, it in enumerate(figs, start=len(images) + 1):
            images.append(fit_image(Image.open(self.cfg.path(it.chunk.image_path)), 784))
            fig_note += f"\nImage {k} is source [{items.index(it) + 1}]."

        user_note = (f"The first {len(user_images)} image(s) are the user's picture(s). "
                     f"Observations of them: " + " | ".join(o.model_dump_json() for o in observations)
                     if user_images else "The user sent no picture.")
        prompt = (f"SOURCES:\n{_format_sources(items)}\n\n{user_note}{fig_note}\n\n"
                  f"USER REQUEST: {question or '(no text - the user only sent the picture)'}")
        out = self.vlm.structured(prompt, _Generated, images=images, system=SYSTEM)

        valid = [n for n in out.cited_sources if 1 <= n <= len(items)]   # drop hallucinated source ids
        cites = [_cite(items[n - 1]) for n in dict.fromkeys(valid)]
        return ReportAnswer(
            answer=out.answer, diagnosis=out.diagnosis or None, recommended_actions=out.recommended_actions,
            citations=cites, grounded=out.grounded and bool(cites),
            image_observations=[o.model_dump_json() for o in observations],
            retrieved=[_cite(it) for it in items])

    # ------------------------------------------------------------ end to end
    def run(self, text: str | None, images: list[Image.Image] | None = None) -> ReportAnswer:
        images = images or []
        observations = [self.observe(im) for im in images]
        query = self.build_query(text, observations)
        log.info("retrieval query: %s", query[:300])
        items = self.retrieve(query)
        return self.generate(text, items, images, observations)


def _cite(it: RetrievedItem) -> Citation:
    c = it.chunk
    return Citation(doc_id=c.doc_id, page=c.page, chunk_id=c.chunk_id, figure_no=c.figure_no)
