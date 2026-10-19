"""Report ingestion: PDF -> text chunks + figure chunks (crop + VLM description) -> Qdrant.

Findings that shaped this (see NOTES.md):
- The NTSB reports are born-digital: full text layer, so no OCR is needed for the corpus.
- A "figure" is a composite of many small raster tiles plus vector arrows/labels, so extracting the
  embedded images individually is useless. Instead we crop the page region above each
  "Figure N." caption and render it - exactly what a reader sees.
- Figures are made searchable by text: caption + in-figure labels + VLM description + the body
  sentences that cite the figure, all embedded in the same space as the text chunks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from ..config import Config
from ..embeddings import get_embedder
from ..llm import VLM, fit_image
from ..pdf import doc_id_from_path, page_stats, render
from ..schemas import Chunk, sha256_file
from ..store import VectorStore

log = logging.getLogger(__name__)

HEADER_RE = re.compile(r"(Page No\.\s*\d+|Pg\s+\d+\s+of\s+\d+|^\s*NATIONAL TRANSPORTATION SAFETY BOARD\s*$)", re.I)
CAPTION_RE = re.compile(r"^\s*Figure\s+(\d+)\s*[.:]", re.I)
BLANK_RE = re.compile(r"intentionally left blank", re.I)
SECTION_RE = re.compile(r"^\s*([A-H])\.\s+([A-Z][A-Z ,&/-]{3,})")
FIG_REF_RE = re.compile(r"\bfig(?:ure)?s?\.?\s+((?:\d+[\s,]*(?:and|through|-|–)?\s*)+)", re.I)

FIGURE_PROMPT_VERSION = "fig-v1"
FIGURE_PROMPT = """This figure comes from an NTSB Materials Laboratory factual report about an aircraft accident.
Report context: {header}
Caption: {caption}
Labels printed on the figure: {labels}

Describe factually what the figure shows, for a search index: the component(s), the view, and every
visible condition such as fracture surface features (beach marks, ratchet marks, origins, shear lips,
dimples), deformation, corrosion, wear, cracks, missing material, measurement markings. Mention what
the arrows/labels point at. 3-6 sentences, no speculation beyond what is visible or stated."""


@dataclass
class _Block:
    y0: float
    y1: float
    text: str


@dataclass
class _DocMeta:
    doc_id: str
    report_no: str | None = None
    vehicle: str | None = None
    location: str | None = None
    date: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def header(self) -> str:
        parts = [f"NTSB Materials Laboratory Factual Report {self.report_no or self.doc_id}"]
        parts += [p for p in (self.vehicle, self.location, self.date) if p]
        return " | ".join(parts)


def _meta(doc: pymupdf.Document, doc_id: str) -> _DocMeta:
    text = "\n".join(doc[i].get_text() for i in range(min(3, len(doc))))
    def grab(pat: str) -> str | None:
        m = re.search(pat, text, re.I)
        return re.sub(r"\s+", " ", m.group(1)).strip(" :") if m else None
    return _DocMeta(
        doc_id=doc_id,
        report_no=grab(r"Report\s+(?:No\.\s*)?(\d{2}-\d{3})"),
        vehicle=grab(r"Vehicle\s*:?\s*\n?\s*:?\s*([^\n]+)"),
        location=grab(r"(?:Place|Location)\s*:?\s*\n?\s*:?\s*([^\n]+)"),
        date=grab(r"\nDate\s*:?\s*\n?\s*:?\s*([^\n]+)"),
    )


def _blocks(page: pymupdf.Page) -> list[_Block]:
    out = []
    for b in page.get_text("blocks"):
        t = re.sub(r"[ \t]+", " ", b[4]).strip()
        t = re.sub(r"\s*\n\s*", " ", t)
        if t and not HEADER_RE.search(t):
            out.append(_Block(b[1], b[3], t))
    return sorted(out, key=lambda b: b.y0)


def _figure_refs(text: str) -> list[int]:
    refs: set[int] = set()
    for m in FIG_REF_RE.finditer(text):
        nums = [int(n) for n in re.findall(r"\d+", m.group(1))]
        if "through" in m.group(1) and len(nums) == 2:
            refs.update(range(nums[0], nums[1] + 1))
        else:
            refs.update(nums)
    return sorted(n for n in refs if n < 200)


def _split_tokens(texts: list[str], max_tokens: int, count) -> list[str]:
    """Greedy merge of paragraphs up to max_tokens; over-long paragraphs are split on sentences."""
    pieces: list[str] = []
    for t in texts:
        if count(t) <= max_tokens:
            pieces.append(t)
        else:
            pieces.extend(s for s in re.split(r"(?<=[.;])\s+", t) if s)
    chunks, cur = [], ""
    for p in pieces:
        cand = f"{cur} {p}".strip()
        if cur and count(cand) > max_tokens:
            chunks.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


class _CaptionCache:
    """VLM figure descriptions are the expensive step; cache by (image hash, model, prompt version)."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def key(self, png: bytes, model: str) -> str:
        return hashlib.sha256(png + model.encode() + FIGURE_PROMPT_VERSION.encode()).hexdigest()

    def get(self, k: str) -> str | None:
        return self.data.get(k)

    def put(self, k: str, v: str) -> None:
        self.data[k] = v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))


class ReportIngestor:
    def __init__(self, cfg: Config, describe_figures: bool = True):
        self.cfg = cfg
        self.describe_figures = describe_figures
        self.embedder = get_embedder(cfg.embedding)
        self.store = VectorStore(cfg, dim=self.embedder.dim)
        self.vlm = VLM(cfg.llm)
        self.image_dir = cfg.path(cfg.store.image_dir)
        self.cache = _CaptionCache(cfg.path(cfg.store.cache_dir) / "figure_descriptions.json")

    # ------------------------------------------------------------ public
    def ingest_paths(self, paths: list[str | Path], force: bool = False) -> dict[str, str]:
        files: list[Path] = []
        for p in map(Path, paths):
            files += sorted(p.rglob("*.pdf")) if p.is_dir() else [p]
        return {f.name: self.ingest_file(f, force=force) for f in files}

    def ingest_file(self, path: Path, force: bool = False) -> str:
        sha = sha256_file(path)
        doc_id = doc_id_from_path(path)
        if not force and self.store.has_document(sha):
            log.info("skip %s (already ingested)", path.name)
            return "skipped"
        self.store.delete_document(doc_id)            # replaces an older version of the same doc
        chunks = self.parse(path, sha)
        vectors = self.embedder.embed([c.embed_text for c in chunks])
        self.store.upsert(chunks, vectors)
        n_fig = sum(c.kind == "figure" for c in chunks)
        log.info("ingested %s: %d chunks (%d figures)", path.name, len(chunks), n_fig)
        return f"{len(chunks)} chunks ({n_fig} figures)"

    # ------------------------------------------------------------ parsing
    def parse(self, path: Path, sha: str) -> list[Chunk]:
        doc_id = doc_id_from_path(path)
        doc = pymupdf.open(path)
        meta = _meta(doc, doc_id)
        text_chunks: list[Chunk] = []
        fig_chunks: list[Chunk] = []
        section: str | None = None

        for pno, page in enumerate(doc, start=1):
            blocks = _blocks(page)
            captions = [b for b in blocks if CAPTION_RE.match(b.text)]
            stats = page_stats(page)
            chart_like = not captions and stats.vector_paths > 40 and stats.chars < 600
            runs: list[tuple[str | None, list[str]]] = []      # body text grouped by section
            for b in blocks:
                if m := SECTION_RE.match(b.text):
                    section = f"{m.group(1)}. {m.group(2).strip().title()}"
                if b in captions:
                    continue
                # on figure pages, short blocks are labels printed on the figure, not body text
                if (captions or chart_like) and len(b.text) < 200:
                    continue
                if len(b.text) < 40 and not SECTION_RE.match(b.text) or BLANK_RE.search(b.text):
                    continue
                if not runs or runs[-1][0] != section:
                    runs.append((section, []))
                runs[-1][1].append(b.text)

            i = 0
            for run_section, texts in runs:
                for txt in _split_tokens(texts, self.cfg.embedding.max_chunk_tokens,
                                         self.embedder.count_tokens):
                    text_chunks.append(Chunk(
                        chunk_id=f"{doc_id}:p{pno}:t{i}", doc_id=doc_id, doc_sha256=sha, page=pno,
                        kind="text", text=txt, section=run_section, figure_refs=_figure_refs(txt),
                        embed_text=f"{meta.header}\nSection: {run_section or '-'}\n{txt}"))
                    i += 1

            regions = self._figure_regions(page, blocks, captions, chart_like)
            for fig_no, caption, rect, labels in regions:
                fig_chunks.append(self._figure_chunk(doc_id, sha, pno, page, fig_no, caption,
                                                     rect, labels, meta, section))

        # link figures back to the body text that cites them: this text explains the figure
        for fc in fig_chunks:
            if fc.figure_no is None:
                continue
            citing = [t.text for t in text_chunks if fc.figure_no in t.figure_refs]
            if citing:
                ctx = " ".join(citing)[:1500]
                fc.text += f"\nReport text citing this figure: {ctx}"
                fc.embed_text += f"\nReport text citing this figure: {ctx}"
        return text_chunks + fig_chunks

    def _figure_regions(self, page, blocks, captions, chart_like):
        """One region per caption: from the previous caption (or the header) down to this caption."""
        regions = []
        header_bottom = 95.0 if page.rect.height > 700 else 0.0
        prev = header_bottom
        for c in captions:
            rect = pymupdf.Rect(0, prev, page.rect.width, c.y1 + 4)
            labels = [b.text for b in blocks if b not in captions and b.y0 >= prev and b.y1 <= c.y0
                      and len(b.text) < 200]
            fig_no = int(CAPTION_RE.match(c.text).group(1))
            regions.append((fig_no, c.text, rect, labels))
            prev = c.y1 + 4
        if chart_like:
            labels = [b.text for b in blocks]
            title = labels[0] if labels else "Chart"
            regions.append((None, f"Chart/plot page: {title}", page.rect, labels))
        return regions

    def _figure_chunk(self, doc_id, sha, pno, page, fig_no, caption, rect, labels, meta, section) -> Chunk:
        img = render(page, 150, clip=rect)
        out_dir = self.image_dir / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"fig{fig_no}_p{pno}.png" if fig_no is not None else f"chart_p{pno}.png"
        img_path = out_dir / name
        img.save(img_path)

        description = ""
        if self.describe_figures:
            png = img_path.read_bytes()
            key = self.cache.key(png, self.cfg.llm.model)
            description = self.cache.get(key) or ""
            if not description:
                prompt = FIGURE_PROMPT.format(header=meta.header, caption=caption,
                                              labels="; ".join(labels[:40]) or "-")
                description = self.vlm.chat(prompt, images=[fit_image(img, 1288)], max_tokens=400).strip()
                self.cache.put(key, description)

        # text = the report's own words (evidence); description = VLM words (findability only)
        text = f"{caption}\nLabels: {'; '.join(labels[:40]) or '-'}"
        cid = f"{doc_id}:p{pno}:fig{fig_no if fig_no is not None else 'chart'}"
        return Chunk(chunk_id=cid, doc_id=doc_id, doc_sha256=sha, page=pno, kind="figure", text=text,
                     description=description or None,
                     embed_text=f"{meta.header}\nSection: {section or '-'}\n{text}\nVisual description: {description or '-'}",
                     section=section,
                     figure_no=fig_no, image_path=str(img_path.relative_to(self.cfg.path("."))))
