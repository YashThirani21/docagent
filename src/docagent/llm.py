"""Thin client for the self-hosted VLM over the OpenAI-compatible API (Ollama or vLLM)."""
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

    def _complete(self, prompt: str, images: list[Image.Image] | None, system: str | None,
                  json_schema: dict | None, max_tokens: int | None) -> tuple[str, str]:
        """One call -> (text, finish_reason). finish_reason 'length' = output was cut off."""
        kwargs: dict = dict(
            model=self.cfg.model,
            messages=self._messages(prompt, images, system),
            temperature=self.cfg.temperature,
        )
        if self.cfg.backend == "ollama":
            kwargs["extra_body"] = {"options": {"num_ctx": self.cfg.num_ctx}}  # context is fixed server-side in vLLM
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_schema:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "output", "schema": json_schema}}
        cache_file = None
        if self.cache_dir and self.cfg.temperature == 0:
            cache_file = self.cache_dir / f"{self._cache_key(kwargs)}.txt"
            if cache_file.exists():
                return cache_file.read_text(), "stop"
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        u = resp.usage
        finish = resp.choices[0].finish_reason or "stop"
        log.info("vlm call %.1fs prompt_tokens=%s completion_tokens=%s images=%d finish=%s",
                 time.perf_counter() - t0, getattr(u, "prompt_tokens", None),
                 getattr(u, "completion_tokens", None), len(images or []), finish)
        text = resp.choices[0].message.content or ""
        if cache_file and finish == "stop":
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(text)
        return text, finish

    def chat(self, prompt: str, images: list[Image.Image] | None = None, system: str | None = None,
             json_schema: dict | None = None, max_tokens: int | None = None) -> str:
        return self._complete(prompt, images, system, json_schema, max_tokens)[0]

    def structured(self, prompt: str, model: type[T], images: list[Image.Image] | None = None,
                   system: str | None = None, retries: int = 1, max_tokens: int | None = None) -> T:
        """Constrained JSON decoding against a pydantic schema.

        - Output cut off (max_tokens hit, often a repetition loop): keep every complete item instead of
          discarding the whole reply. Retrying the same prompt at temperature 0 would fail identically.
        - Otherwise-invalid output: one retry with a changed prompt (so the retry can actually differ).
        """
        schema = model.model_json_schema()
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            p = prompt if attempt == 0 else prompt + "\n\nReturn compact, valid JSON. Do not repeat items."
            raw, finish = self._complete(p, images, system, schema, max_tokens)
            try:
                return model.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
            for salvaged in truncated_json_candidates(raw):
                try:
                    obj = model.model_validate(salvaged)
                except ValidationError:
                    continue
                log.warning("%s output %s after %d chars; kept the complete part",
                            model.__name__, "truncated (max_tokens)" if finish == "length" else "malformed",
                            len(raw))
                return obj
            self._dump_failure(model.__name__, raw, finish, last_err)
            log.warning("structured output invalid (attempt %d, finish=%s): %s",
                        attempt + 1, finish, str(last_err)[:200])
            if finish == "length" and self.cfg.temperature == 0 and attempt == 0:
                continue                      # the retry prompt differs, so a second try is not identical
        raise ValueError(f"VLM did not return valid {model.__name__}: {last_err}")

    def _dump_failure(self, name: str, raw: str, finish: str, err: Exception | None) -> None:
        """Keep the raw reply of failed calls for inspection (store/cache/vlm_failures/)."""
        if not self.cache_dir:
            return
        d = self.cache_dir.parent / "vlm_failures"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_{finish}.txt"
        f.write_text(f"# error: {err}\n{raw}")
        log.warning("raw VLM output saved to %s", f)


def repair_truncated_json(raw: str):
    """Longest valid-JSON prefix of a cut-off reply (parsed), or None."""
    return next(truncated_json_candidates(raw), None)


def truncated_json_candidates(raw: str):
    """Yield parsed JSON from successively shorter prefixes of a cut-off reply: cut after a complete
    object/array element and close whatever is still open. The caller keeps the first one that also
    validates against its schema (a prefix may end inside an incomplete item)."""
    stack: list[str] = []
    cuts: list[tuple[int, list[str]]] = []
    in_str = esc = False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            cuts.append((i, stack.copy()))
    for i, open_ in reversed(cuts[-200:]):
        try:
            yield json.loads(raw[: i + 1] + "".join(reversed(open_)))
        except json.JSONDecodeError:
            continue
