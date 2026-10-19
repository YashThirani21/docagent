"""Hybrid retriever: bge-m3 dense + sparse -> Qdrant RRF fusion -> figure-link expansion."""
from __future__ import annotations

from .config import Config
from .embeddings import get_embedder
from .schemas import RetrievedItem
from .store import VectorStore


class HybridRetriever:
    version = "hybrid-rrf-v1"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = get_embedder(cfg.embedding)
        self.store = VectorStore(cfg, dim=self.embedder.dim)

    def search(self, query: str, top_k: int | None = None, expand_figures: bool = True) -> list[RetrievedItem]:
        rc = self.cfg.retrieval
        q = self.embedder.embed([query])[0]
        hits = self.store.hybrid_search(q, top_k or rc.top_k, rc.dense_k, rc.sparse_k)
        items = [RetrievedItem(chunk=c, score=s) for c, s in hits]
        if expand_figures:
            items += self._linked_figures(items)
        return items

    def _linked_figures(self, items: list[RetrievedItem]) -> list[RetrievedItem]:
        """A text hit that says 'see figure 5' is much more useful with figure 5 next to it."""
        have = {(i.chunk.doc_id, i.chunk.figure_no) for i in items if i.chunk.kind == "figure"}
        want: dict[str, list[int]] = {}
        for it in items[:4]:                                  # only the strongest text hits
            for n in it.chunk.figure_refs[:3]:
                if (it.chunk.doc_id, n) not in have:
                    want.setdefault(it.chunk.doc_id, []).append(n)
                    have.add((it.chunk.doc_id, n))
        out = []
        for doc_id, nums in want.items():
            for ch in self.store.figures(doc_id, nums):
                out.append(RetrievedItem(chunk=ch, score=0.0, via="linked"))
        return out
