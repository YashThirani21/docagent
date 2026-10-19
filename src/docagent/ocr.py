"""Offline OCR helpers (RapidOCR, ONNX models bundled in the wheel - no download at runtime).

Two package generations exist and neither covers every Python:
  rapidocr-onnxruntime 1.x  (Python <= 3.12; used locally)   ->  engine(img) returns (result, elapsed)
  rapidocr 3.x              (Python 3.13, e.g. current Colab) ->  engine(img) returns an output object
Only these two functions are used by the pipelines, so the difference stays here.
"""
from __future__ import annotations

import numpy as np

_ENGINES: dict[tuple, tuple[str, object]] = {}


def _engine(box_thresh: float | None = None, thresh: float | None = None) -> tuple[str, object]:
    key = (box_thresh, thresh)
    if key not in _ENGINES:
        try:
            from rapidocr_onnxruntime import RapidOCR
            kw = {k: v for k, v in (("det_box_thresh", box_thresh), ("det_thresh", thresh)) if v is not None}
            _ENGINES[key] = ("v1", RapidOCR(**kw))
        except ImportError:
            from rapidocr import RapidOCR
            params = {"Global.log_level": "error"}
            if box_thresh is not None:
                params["Det.box_thresh"] = box_thresh
            if thresh is not None:
                params["Det.thresh"] = thresh
            _ENGINES[key] = ("v3", RapidOCR(params=params))
    return _ENGINES[key]


def detect(img: np.ndarray, box_thresh: float | None = None, thresh: float | None = None) -> list[np.ndarray]:
    """Text-box detection only: list of polygons (N x 2 arrays, first edge along the text baseline)."""
    kind, eng = _engine(box_thresh, thresh)
    if kind == "v1":
        res, _ = eng(img, use_det=True, use_cls=False, use_rec=False)
        return [np.array(p, dtype=float).reshape(-1, 2) for p in res or []]
    out = eng(img, use_det=True, use_cls=False, use_rec=False)
    return [np.array(p, dtype=float).reshape(-1, 2) for p in (out.boxes if out.boxes is not None else [])]


def recognize(img: np.ndarray) -> list[tuple[str, float]]:
    """Recognition only (the image is one text line): list of (text, score)."""
    kind, eng = _engine()
    if kind == "v1":
        res, _ = eng(img, use_det=False, use_cls=False, use_rec=True)
        return [(str(r[0]), float(r[1])) for r in res or []]
    out = eng(img, use_det=False, use_cls=False, use_rec=True)
    return [(str(t), float(s)) for t, s in zip(out.txts or (), out.scores or ())]
