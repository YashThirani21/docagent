"""Thin client for the local VLM over the OpenAI-compatible API (Ollama now, vLLM later)."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import time
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ValidationError

from .config import PROJECT_ROOT, LLMCfg

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fit_image(img: Image.Image, max_side: int, multiple: int = 28) -> Image.Image:
    """Downscale so the long side <= max_side, snapping to the VLM's patch size (Qwen2.5-VL: 28px)
    so the server does not resize again (keeps returned pixel coordinates meaningful)."""
    scale = min(1.0, max_side / max(img.size))
    w = max(multiple, int(img.width * scale) // multiple * multiple)
    h = max(multiple, int(img.height * scale) // multiple * multiple)
    return img if (w, h) == img.size else img.resize((w, h), Image.LANCZOS)


class VLM:
    def __init__(self, cfg: LLMCfg, cache_dir: str | Path | None = None):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key="local", timeout=cfg.timeout_s)
        # Deterministic calls (temperature 0) are cached on disk: re-runs of eval/dev cost nothing.
        cache_dir = cache_dir or cfg.cache_dir
        self.cache_dir = (PROJECT_ROOT / cache_dir) if cache_dir else None

    def _cache_key(self, kwargs: dict) -> str:
        return hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()

    def _messages(self, prompt: str, images: list[Image.Image] | None, system: str | None):
        content: list[dict] = [{"type": "image_url", "image_url": {"url": image_to_data_url(im)}}
                               for im in images or []]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": content})
        return msgs

    def chat(self, prompt: str, images: list[Image.Image] | None = None, system: str | None = None,
             json_schema: dict | None = None, max_tokens: int | None = None) -> str:
        kwargs: dict = dict(
            model=self.cfg.model,
            messages=self._messages(prompt, images, system),
            temperature=self.cfg.temperature,
            extra_body={"options": {"num_ctx": self.cfg.num_ctx}},  # ignored by vLLM, needed by Ollama
        )
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_schema:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "output", "schema": json_schema}}
        cache_file = None
        if self.cache_dir and self.cfg.temperature == 0:
            cache_file = self.cache_dir / f"{self._cache_key(kwargs)}.txt"
            if cache_file.exists():
                return cache_file.read_text()
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        u = resp.usage
        log.info("vlm call %.1fs prompt_tokens=%s completion_tokens=%s images=%d",
                 time.perf_counter() - t0, getattr(u, "prompt_tokens", None),
                 getattr(u, "completion_tokens", None), len(images or []))
        text = resp.choices[0].message.content or ""
        if cache_file and resp.choices[0].finish_reason == "stop":
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(text)
        return text

    def structured(self, prompt: str, model: type[T], images: list[Image.Image] | None = None,
                   system: str | None = None, retries: int = 1, max_tokens: int | None = None) -> T:
        """Constrained JSON decoding against a pydantic schema, with one repair retry."""
        schema = model.model_json_schema()
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            raw = self.chat(prompt, images, system, json_schema=schema, max_tokens=max_tokens)
            try:
                return model.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                log.warning("structured output invalid (attempt %d): %s", attempt + 1, str(e)[:300])
        raise ValueError(f"VLM did not return valid {model.__name__}: {last_err}")
