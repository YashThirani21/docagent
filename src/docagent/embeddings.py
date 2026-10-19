"""bge-m3: one forward pass gives a dense vector (semantic) and sparse lexical weights (keywords)."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from .config import EmbeddingCfg

log = logging.getLogger(__name__)


@dataclass
class Embedded:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class Embedder:
    def __init__(self, cfg: EmbeddingCfg):
        from FlagEmbedding import BGEM3FlagModel  # heavy import, keep lazy

        self.cfg = cfg
        log.info("loading embedder %s on %s", cfg.model, cfg.device)
        self.model = BGEM3FlagModel(cfg.model, use_fp16=cfg.device == "cuda", devices=cfg.device)
        self.tokenizer = self.model.tokenizer
        self._lock = threading.Lock()
        self._self_check()

    def _self_check(self) -> None:
        """Batched and single encodings must agree. Guards against silent backend bugs
        (seen: MPS returned near-identical vectors for padded batches of long texts)."""
        texts = [("fatigue crack origin at the bolt hole in the lower spar cap " * 40),
                 "bending overstress fracture of the threaded section", "radiograph of a lock coil"]
        batch = self.model.encode(texts, batch_size=len(texts), max_length=1024)["dense_vecs"]
        for i, t in enumerate(texts):
            single = self.model.encode([t], max_length=1024)["dense_vecs"][0]
            if float(batch[i] @ single) < 0.99:
                raise RuntimeError(f"embedder self-check failed on device {self.cfg.device}: batched "
                                   "and single vectors differ. Use device: cpu (or cuda).")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def embed(self, texts: list[str]) -> list[Embedded]:
        with self._lock:                  # one model, many request threads (eval, bulk, chat)
            out = self.model.encode(texts, batch_size=self.cfg.batch_size, max_length=1024,
                                    return_dense=True, return_sparse=True)
        res = []
        for dense, lex in zip(out["dense_vecs"], out["lexical_weights"]):
            idx = [int(k) for k in lex]
            res.append(Embedded(dense.tolist(), idx, [float(lex[k]) for k in lex]))
        return res

    @property
    def dim(self) -> int:
        return 1024


_EMBEDDERS: dict[tuple[str, str], Embedder] = {}


def get_embedder(cfg: EmbeddingCfg) -> Embedder:
    """Load once per process (~2 GB); every component shares the same instance."""
    key = (cfg.model, cfg.device)
    if key not in _EMBEDDERS:
        _EMBEDDERS[key] = Embedder(cfg)
    return _EMBEDDERS[key]
