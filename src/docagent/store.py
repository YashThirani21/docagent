"""Qdrant wrapper: one collection, named dense + sparse vectors, chunk payload, hybrid search."""
from __future__ import annotations

import atexit
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .config import Config
from .embeddings import Embedded
from .schemas import Chunk

log = logging.getLogger(__name__)

_CLIENTS: dict[str, QdrantClient] = {}


def _client(cfg: Config) -> QdrantClient:
    key = cfg.store.qdrant_url or str(cfg.path(cfg.store.qdrant_path))
    if key not in _CLIENTS:
        # NB: embedded mode takes a file lock - only one process at a time (use qdrant_url to share)
        _CLIENTS[key] = (QdrantClient(url=cfg.store.qdrant_url) if cfg.store.qdrant_url
                         else QdrantClient(path=key))
        atexit.register(_CLIENTS[key].close)      # release the local-mode lock cleanly
    return _CLIENTS[key]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self, cfg: Config, dim: int = 1024):
        self.cfg = cfg
        self.client = _client(cfg)
        self.collection = cfg.store.collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config={"dense": qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
                sparse_vectors_config={"sparse": qm.SparseVectorParams()},
            )
            for field in ("doc_id", "doc_sha256", "kind"):
                self.client.create_payload_index(self.collection, field, qm.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection, "figure_no", qm.PayloadSchemaType.INTEGER)

    # ------------------------------------------------------------ write
    def has_document(self, doc_sha256: str) -> bool:
        hits, _ = self.client.scroll(self.collection, limit=1, with_payload=False, scroll_filter=qm.Filter(
            must=[qm.FieldCondition(key="doc_sha256", match=qm.MatchValue(value=doc_sha256))]))
        return bool(hits)

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(self.collection, points_selector=qm.FilterSelector(filter=qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])))

    def upsert(self, chunks: list[Chunk], vectors: list[Embedded]) -> None:
        points = [
            qm.PointStruct(
                id=point_id(c.chunk_id),
                vector={"dense": v.dense,
                        "sparse": qm.SparseVector(indices=v.sparse_indices, values=v.sparse_values)},
                payload=c.model_dump(),
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points=points)

    # ------------------------------------------------------------ read
    def hybrid_search(self, q: Embedded, top_k: int, dense_k: int, sparse_k: int,
                      flt: qm.Filter | None = None) -> list[tuple[Chunk, float]]:
        res = self.client.query_points(
            self.collection,
            prefetch=[
                qm.Prefetch(query=q.dense, using="dense", limit=dense_k, filter=flt),
                qm.Prefetch(query=qm.SparseVector(indices=q.sparse_indices, values=q.sparse_values),
                            using="sparse", limit=sparse_k, filter=flt),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [(Chunk(**p.payload), p.score) for p in res.points]

    def figures(self, doc_id: str, figure_nos: list[int]) -> list[Chunk]:
        if not figure_nos:
            return []
        hits, _ = self.client.scroll(self.collection, limit=len(figure_nos) * 2, scroll_filter=qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
            qm.FieldCondition(key="kind", match=qm.MatchValue(value="figure")),
            qm.FieldCondition(key="figure_no", match=qm.MatchAny(any=figure_nos)),
        ]))
        return [Chunk(**h.payload) for h in hits]

    def count(self) -> int:
        return self.client.count(self.collection).count

    def documents(self) -> dict[str, int]:
        out: dict[str, int] = {}
        offset = None
        while True:
            hits, offset = self.client.scroll(self.collection, limit=256, offset=offset,
                                              with_payload=["doc_id"])
            for h in hits:
                out[h.payload["doc_id"]] = out.get(h.payload["doc_id"], 0) + 1
            if offset is None:
                return out
