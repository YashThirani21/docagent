"""Hybrid retriever: bge-m3 dense + sparse -> Qdrant RRF fusion -> context assembly.

Context assembly (v2, from the QA baseline):
- Text first. Figure chunks (caption + labels) ranked above the text pages holding the actual findings,
  so answers missed details (fatigue at RC-1, roll groove). A text-only search guarantees `min_text`
  report paragraphs in context; text precedes figures in the prompt.
- Registration-aware. A report can cover several aircraft. When the question names one (N106ER), chunks
  about a *different* named aircraft are dropped (Q10 answered with N104ER's values).
- Report focus. When the question names something in exactly one report's header (place, registration,
  report number, aircraft type), only that report is used (Q19 blended 24-013 with 16-006).
- Next paragraph. A finding often continues in the chunk after the one that matched (Q14: the fatigue
  at RC-1 sat in the chunk after the "overstress from RD-1" chunk), so the top text hits bring their
  successor along.
- Figure links: figures cited by the top text hits are added after.
"""
from __future__ import annotations

import re

from qdrant_client import models as qm

from .config import Config
from .embeddings import get_embedder
from .schemas import RetrievedItem
from .store import VectorStore

REG_RE = re.compile(r"\bN\d{1,5}[A-Z]{0,2}\b")
_TERM_RE = re.compile(r"\b(?:[A-Z][a-z]{3,}|[A-Z0-9]+-[A-Z0-9-]+|N\d{1,5}[A-Z]{0,2}|\d{2}-\d{3})\b")
_STOP = {"what", "which", "where", "when", "were", "does", "with", "from", "that", "this", "report", "reports",
         "figure", "about", "airplane", "aircraft", "accident", "national", "transportation", "safety", "board"}


def registrations(text: str | None) -> set[str]:
    return set(REG_RE.findall((text or "").upper()))


class HybridRetriever:
    version = "hybrid-rrf-v3-textfirst-reg-focus-next"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = get_embedder(cfg.embedding)
        self.store = VectorStore(cfg, dim=self.embedder.dim)

    def _search(self, q, k: int, flt: qm.Filter | None = None) -> list[RetrievedItem]:
        rc = self.cfg.retrieval
        return [RetrievedItem(chunk=c, score=s) for c, s in
                self.store.hybrid_search(q, k, rc.dense_k, rc.sparse_k, flt=flt)]

    def search(self, query: str, top_k: int | None = None, expand_figures: bool = True) -> list[RetrievedItem]:
        rc = self.cfg.retrieval
        k = top_k or rc.top_k
        q = self.embedder.embed([query])[0]
        text_only = qm.Filter(must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value="text"))])
        pool = self._search(q, k * 2) + self._search(q, k * 2, flt=text_only)

        wanted = registrations(query)
        seen: set[str] = set()
        items: list[RetrievedItem] = []
        for it in sorted(pool, key=lambda i: -i.score):
            if it.chunk.chunk_id in seen:
                continue
            seen.add(it.chunk.chunk_id)
            about = registrations(it.chunk.subject)
            if wanted and about and not (about & wanted):
                continue                                  # explicitly about another aircraft
            items.append(it)

        items = self._focus(query, items)
        texts = [i for i in items if i.chunk.kind == "text"]
        figs = [i for i in items if i.chunk.kind != "text"]
        # at least min_text paragraphs; figures fill the rest but never more than half of the context
        chosen_text = texts[: max(rc.min_text, k - min(len(figs), k // 2))]
        chosen = chosen_text + figs[: max(0, k - len(chosen_text))]
        chosen = self._with_next(chosen, wanted)
        if expand_figures:
            chosen += self._linked_figures(chosen, wanted)
        return chosen

    @staticmethod
    def _focus(query: str, items: list[RetrievedItem]) -> list[RetrievedItem]:
        """Keep one report if the question names something only that report's header contains."""
        terms = {t.lower() for t in _TERM_RE.findall(query) if t.lower() not in _STOP}
        if not terms:
            return items
        headers = {i.chunk.doc_id: i.chunk.embed_text.split("\n", 1)[0].lower() for i in items}
        pats = [re.compile(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])") for t in terms]   # whole words
        hit = {d for d, h in headers.items() if any(p.search(h) for p in pats)}
        return [i for i in items if i.chunk.doc_id in hit] if len(hit) == 1 else items

    def _with_next(self, items: list[RetrievedItem], wanted: set[str]) -> list[RetrievedItem]:
        """Add the paragraph that follows each of the top text hits (same report, reading order)."""
        have = {i.chunk.chunk_id for i in items}
        nxt = []
        for it in [i for i in items if i.chunk.kind == "text"][:3]:
            m = re.match(r"(.+):p(\d+):t(\d+)$", it.chunk.chunk_id)
            if not m:
                continue
            doc, page, n = m.group(1), int(m.group(2)), int(m.group(3))
            for cand in (f"{doc}:p{page}:t{n + 1}", f"{doc}:p{page + 1}:t0"):
                if cand in have:
                    break
                got = self.store.get([cand])
                if got:
                    have.add(cand)
                    about = registrations(got[0].subject)
                    if not (wanted and about and not (about & wanted)):
                        nxt.append((items.index(it), RetrievedItem(chunk=got[0], score=it.score, via="linked")))
                    break
        for offset, (pos, new) in enumerate(nxt):              # place right after the chunk it continues
            items.insert(pos + 1 + offset, new)
        return items

    def _linked_figures(self, items: list[RetrievedItem], wanted: set[str]) -> list[RetrievedItem]:
        """A text hit that says 'see figure 5' is much more useful with figure 5 next to it."""
        have = {(i.chunk.doc_id, i.chunk.figure_no) for i in items if i.chunk.kind == "figure"}
        want: dict[str, list[int]] = {}
        for it in [i for i in items if i.chunk.kind == "text"][:3]:   # only the strongest text hits
            for n in it.chunk.figure_refs[:2]:
                if (it.chunk.doc_id, n) not in have:
                    want.setdefault(it.chunk.doc_id, []).append(n)
                    have.add((it.chunk.doc_id, n))
        out = []
        for doc_id, nums in want.items():
            for ch in self.store.figures(doc_id, nums):
                about = registrations(ch.subject)
                if wanted and about and not (about & wanted):
                    continue
                out.append(RetrievedItem(chunk=ch, score=0.0, via="linked"))
        return out
