"""Pipeline A: answer text and/or picture requests from the NTSB report corpus.

Flow: (pictures -> VLM observations) -> retrieval query -> hybrid search (+ linked figures)
      -> VLM answer over numbered sources (+ user pictures + top figure crops) -> citations checked.
"""
from __future__ import annotations

import logging
import re

from PIL import Image
from pydantic import BaseModel, Field

from ..config import Config
from ..llm import VLM, fit_image
from ..retriever import HybridRetriever
from ..schemas import Citation, ReportAnswer, RetrievedItem
from ..trace import span

log = logging.getLogger(__name__)


class ImageObservation(BaseModel):
    component: str = Field(description="What part/object is shown, as specifically as possible")
    observations: list[str] = Field(description="Visible conditions: fracture features, deformation, "
                                                "corrosion, wear, cracks, markings")
    visible_text: str = Field(description="Any text/labels/scale markings readable in the picture, or ''")


class _Generated(BaseModel):
    user_reports_problem: bool = Field(description="true only if the user shows or describes a damaged/failed/"
                                                   "malfunctioning part and wants to know what is wrong or what "
                                                   "to do; false for factual questions about the reports")
    answer: str
    diagnosis: str = Field(description="What is wrong - ONLY if user_reports_problem, else ''")
    recommended_actions: list[str] = Field(description="What to do - ONLY if user_reports_problem, else []")
    cited_sources: list[int] = Field(description="Numbers of the sources the answer relies on")
    grounded: bool = Field(description="true only if the sources actually support the answer")


# Observations run 102 tokens (median) to 347 (max) in the traces. Uncapped, one looped until the context was full
# (7,584 tokens, 150 s on AWQ; the BF16 qa_image cases at ~460 s): the cap ends a loop early and the retry in
# VLM.structured takes over, as it did after the full-context run. The answer call stays uncapped: it never looped,
# and with 4 images its prompt is close enough to an 8k context that a cap could get the request rejected.
OBSERVE_MAX_TOKENS = 512

OBSERVE_PROMPT = """You are a failure-analysis engineer. Look at this picture of an aircraft/mechanical
component (it may be a photo, radiograph, micrograph or fracture surface).
Report only what is visible, precisely and factually. Use metallurgical terms only for features you can
actually see; do not infer the cause of failure."""

SYSTEM = """You are a materials-failure assistant for aircraft components. You answer ONLY from the numbered
sources, which are excerpts and figures from NTSB Materials Laboratory Factual Reports.
Rules:
- Every factual claim must come from the sources; cite every source you use by number in cited_sources.
- The report's findings (e.g. "consistent with bending overstress", "fatigue") are authoritative. Never
  contradict them or substitute a different failure mode. If the sources describe several failure modes
  for the part asked about (e.g. overstress AND fatigue cracks), report all of them.
- These are factual laboratory reports. They do not state accident causes or "probable cause". Never
  present a component finding as the cause of an accident, and never attribute a determination to the NTSB
  that the sources do not state.
- Each source says which aircraft/part it is "about". Answer for the aircraft/part the user names. If the
  sources only give the value for a different one, say the reports do not give it for the one asked.
  Never tell the user their question is wrong.
- For "why" questions, give only the facts the sources state; if they do not state the reason, say so.
  Do not add your own explanations.
- Factual question -> answer it directly; diagnosis '' and recommended_actions [].
  User shows or describes a problem (e.g. sends a photo of a damaged part) -> also give the diagnosis
  and recommended_actions. The reports state findings, not instructions, so recommendations must follow
  from the findings (e.g. fatigue at a bolt hole -> inspect that location on similar parts).
- If the sources do not cover the question, set grounded=false and say so plainly. Do not guess.
- Be complete: when asked what was found, include every relevant finding from all sources (findings
  about one part are often spread over several pages), and give identifiers exactly (registration,
  report number, hole/bolt IDs, measurements with units).
- Be concise and specific: part names, failure modes, locations, measurements, report numbers."""


CITING_MARK = "\nReport text citing this figure:"


def evidence_text(it: RetrievedItem) -> str:
    """What a source itself says. Figure chunks also carry the paragraphs that cite them (useful for
    retrieval), but those paragraphs are separate sources: showing them twice steered answers (q16) and
    let a figure 'back' a number that is really on a text page (q06)."""
    t = it.chunk.text
    return t.split(CITING_MARK, 1)[0] if it.chunk.kind == "figure" else t


def _format_sources(items: list[RetrievedItem]) -> str:
    lines = []
    for n, it in enumerate(items, start=1):
        c = it.chunk
        tag = f"Figure {c.figure_no}" if c.kind == "figure" and c.figure_no else c.kind
        lines.append(f"[{n}] report {c.doc_id}, page {c.page}, {tag}"
                     f"{', section ' + c.section if c.section else ''}"
                     f"{', about: ' + c.subject if c.subject else ''}\n{evidence_text(it)}")
    return "\n\n".join(lines)


class ReportQA:
    version = "report-qa-v5"

    def __init__(self, cfg: Config, retriever: HybridRetriever | None = None, vlm: VLM | None = None):
        self.cfg = cfg
        self.retriever = retriever or HybridRetriever(cfg)
        self.vlm = vlm or VLM(cfg.llm)

    # ------------------------------------------------------------ steps (public for notebooks/eval)
    def observe(self, image: Image.Image) -> ImageObservation:
        return self.vlm.structured(OBSERVE_PROMPT, ImageObservation, images=[fit_image(image, 1036)],
                                   max_tokens=OBSERVE_MAX_TOKENS)

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
        # Image budget: the server rejects a call with more than llm.max_images images (e2e e18: a report PDF
        # rendered as 3 pages + 2 figure crops = 5). The user's pictures come first, figures fill what is left.
        # Within the budget nothing changes (same images, same prompt); a dropped picture is still described
        # by its observation, which every picture gets.
        limit = self.cfg.llm.max_images
        shown = user_images[:limit]
        dropped = len(user_images) - len(shown)
        figs = [it for it in items if it.chunk.kind == "figure" and it.chunk.image_path]
        figs = figs[: max(0, min(self.cfg.retrieval.max_context_figures, limit - len(shown)))]
        images = [fit_image(im, 896) for im in shown]
        fig_note = ""
        for k, it in enumerate(figs, start=len(images) + 1):
            images.append(fit_image(Image.open(self.cfg.path(it.chunk.image_path)), 784))
            fig_note += f"\nImage {k} is source [{items.index(it) + 1}]."

        user_note = (f"The first {len(shown)} image(s) are the user's picture(s). "
                     f"Observations of them: " + " | ".join(o.model_dump_json() for o in observations)
                     + (f" ({dropped} more picture(s) were sent but are not shown; only their observations "
                        f"above describe them.)" if dropped else "")
                     if user_images else "The user sent no picture.")
        scope = ""
        if question and CAUSE_RE.search(question):          # lab reports never state accident causes
            scope = ("\n\nNOTE: The user asks for the cause of an accident. The sources are laboratory factual "
                     "reports and do not state accident causes. Start your answer by saying the reports do not "
                     "state the cause of the accident, then summarise what the report found about the examined "
                     "component. Do not write 'probable cause' or 'determined by'.")
        prompt = (f"SOURCES:\n{_format_sources(items)}\n\n{user_note}{fig_note}{scope}\n\n"
                  f"USER REQUEST: {question or '(no text - the user only sent the picture)'}")
        with span("qa.generate", n_sources=len(items), n_images=len(images), n_user_images=len(user_images),
                  images_dropped=dropped, cause_question=bool(scope)) as sp:
            out = self.vlm.structured(prompt, _Generated, images=images, system=SYSTEM)
            sp.set(model_cited=out.cited_sources, grounded=out.grounded, user_reports_problem=out.user_reports_problem)

        valid = [n for n in out.cited_sources if 1 <= n <= len(items)]   # drop hallucinated source ids
        problem = bool(user_images) or out.user_reports_problem
        diagnosis = out.diagnosis if problem else ""                      # no filler on factual questions
        actions = out.recommended_actions if problem else []
        answer_text = " ".join([out.answer, diagnosis, *actions])
        with span("qa.verify_citations", model_cited=[n - 1 for n in dict.fromkeys(valid)]) as sp:
            cited = verify_citations(answer_text, items, [n - 1 for n in dict.fromkeys(valid)], question)
            sp.set(kept=cited)                                # indices into the sources: diff = what code changed
        cites = _dedupe([_cite(items[i]) for i in cited])
        return ReportAnswer(
            answer=out.answer, diagnosis=diagnosis or None, recommended_actions=actions,
            citations=cites, grounded=out.grounded and bool(cites),
            image_observations=[o.model_dump_json() for o in observations],
            retrieved=[_cite(it) for it in items])

    # ------------------------------------------------------------ end to end
    def run(self, text: str | None, images: list[Image.Image] | None = None) -> ReportAnswer:
        images = images or []
        observations = []
        for im in images:
            with span("qa.observe", size=list(im.size)) as sp:
                observations.append(self.observe(im))
                sp.set(component=observations[-1].component)
        query = self.build_query(text, observations)
        log.info("retrieval query: %s", query[:300])
        with span("qa.retrieve", query=query[:1000], retriever=self.retriever.version) as sp:
            items = self.retrieve(query)
            sp.set(n_items=len(items), items=[f"{it.chunk.doc_id} p{it.chunk.page} {it.chunk.kind}"
                                              f"{' fig' + str(it.chunk.figure_no) if it.chunk.figure_no else ''}"
                                              f" {it.via} {it.score:.3f}" for it in items],
                   chunk_ids=[it.chunk.chunk_id for it in items])
        return self.generate(text, items, images, observations)


CAUSE_RE = re.compile(r"probable cause|cause of (the|this) (accident|crash|incident)|why did .* crash", re.I)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_ID_RE = re.compile(r"\b(?:[A-Z]{1,2}-\d{1,2}|N\d{1,5}[A-Z]{0,2}|\d{4}-T\d{1,4})\b")


def _nums(text: str) -> set[str]:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)                      # 7,690.6 -> 7690.6
    # distinctive numbers only: decimals or 3+ digits (not "1", "2", figure numbers)
    return {n for n in _NUM_RE.findall(text) if "." in n or len(n) >= 3}


def verify_citations(answer: str, items: list[RetrievedItem], cited: list[int], question: str | None) -> list[int]:
    """Make citations follow the evidence, not the model's habit of citing whatever it saw.

    1. Drop cited sources about a different aircraft than the one(s) the question names.
    2. Every distinctive number in the answer must be backed by a cited source; if none of the cited
       sources contains it, add the highest-ranked retrieved source that does (e.g. a table page).
    3. If the answer has distinctive numbers/IDs, drop cited sources that contain none of them,
       unless that would leave no citation.
    """
    from ..retriever import registrations
    wanted = registrations(question)
    texts = [re.sub(r"(?<=\d),(?=\d{3})", "", evidence_text(it)) for it in items]
    keep = [i for i in cited if not (wanted and registrations(items[i].chunk.subject)
                                     and not registrations(items[i].chunk.subject) & wanted)]
    nums = _nums(answer)
    ids = set(_ID_RE.findall(answer))
    for n in sorted(nums):
        if not any(n in texts[i] for i in keep):
            backer = next((i for i in range(len(items)) if n in texts[i] and i not in keep), None)
            if backer is not None:
                keep.append(backer)
    if nums or ids:
        anchored = [i for i in keep if any(a in texts[i] for a in nums | ids)]
        keep = anchored or keep
    # wording: the paragraph the answer is (nearly) quoting must be cited (q01 quoted p.3, cited figures)
    ans_tri = _trigrams(answer)
    overlap = [len(ans_tri & _trigrams(t)) for t in texts]
    best = max(range(len(items)), key=lambda i: overlap[i], default=None)
    if best is not None and best not in keep and overlap[best] >= 3 \
            and overlap[best] > max((overlap[i] for i in keep), default=0):
        keep.append(best)
    return keep


def _trigrams(text: str) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9.%-]+", text.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def _dedupe(cites: list[Citation]) -> list[Citation]:
    seen, out = set(), []
    for c in cites:
        key = (c.doc_id, c.page, c.figure_no)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _cite(it: RetrievedItem) -> Citation:
    c = it.chunk
    return Citation(doc_id=c.doc_id, page=c.page, chunk_id=c.chunk_id, figure_no=c.figure_no)
