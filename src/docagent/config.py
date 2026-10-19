"""Typed configuration loaded from YAML. One config file = one agent version."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class PrivacyCfg(BaseModel):
    offline: bool = True


class LLMCfg(BaseModel):
    backend: str = "ollama"                     # ollama | vllm (both OpenAI-compatible)
    max_concurrency: int = 1                    # parallel VLM calls; >1 pays off with vLLM
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5vl:7b"
    num_ctx: int = 16384
    temperature: float = 0.0
    timeout_s: int = 600
    cache_dir: str | None = "store/cache/vlm"   # disk cache for temperature-0 calls; null disables


class EmbeddingCfg(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 8
    max_chunk_tokens: int = 350


class StoreCfg(BaseModel):
    qdrant_path: str | None = "store/qdrant"
    qdrant_url: str | None = None
    collection: str = "reports_bge-m3_v1"
    image_dir: str = "store/images"
    cache_dir: str = "store/cache"


class RetrievalCfg(BaseModel):
    dense_k: int = 30
    sparse_k: int = 30
    top_k: int = 8
    max_context_figures: int = 2


class DrawingCfg(BaseModel):
    tile_dpi: int = 200
    grid: tuple[int, int] = (2, 2)
    overlap: float = 0.12
    overview_dpi: int = 100
    dedup_distance_pt: float = 25.0


class RouterCfg(BaseModel):
    classify_max_side: int = 768


class Config(BaseModel):
    privacy: PrivacyCfg = PrivacyCfg()
    llm: LLMCfg = LLMCfg()
    embedding: EmbeddingCfg = EmbeddingCfg()
    store: StoreCfg = StoreCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    drawing: DrawingCfg = DrawingCfg()
    router: RouterCfg = RouterCfg()

    def path(self, p: str) -> Path:
        """Resolve a config path relative to the project root."""
        q = Path(p)
        return q if q.is_absolute() else PROJECT_ROOT / q


def load_config(path: str | Path | None = None) -> Config:
    """Explicit path > $DOCAGENT_CONFIG > configs/default.yaml (so CLI, chat app and notebooks agree)."""
    path = Path(path or os.environ.get("DOCAGENT_CONFIG") or DEFAULT_CONFIG)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    cfg = Config(**(yaml.safe_load(path.read_text()) or {}))
    if cfg.privacy.offline:
        # Models are downloaded once by scripts/download_models.py; afterwards nothing may call out.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return cfg
