"""PDF / image helpers: rendering, cheap page statistics used by ingestion and routing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image

from .schemas import Attachment


def render(page: pymupdf.Page, dpi: int, clip: pymupdf.Rect | None = None) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


@dataclass
class PageStats:
    width: float
    height: float
    chars: int
    vector_paths: int
    image_count: int
    image_coverage: float        # fraction of page area covered by placed raster images
    text: str

    @property
    def landscape(self) -> bool:
        return self.width > self.height


def page_stats(page: pymupdf.Page) -> PageStats:
    area = page.rect.width * page.rect.height
    infos = page.get_image_info()
    covered = sum(abs(pymupdf.Rect(i["bbox"]) & page.rect) for i in infos)
    text = page.get_text()
    return PageStats(page.rect.width, page.rect.height, len(text.strip()), len(page.get_drawings()),
                     len(infos), min(1.0, covered / area), text)


def attachment_images(att: Attachment, dpi: int = 110, max_pages: int = 3) -> list[Image.Image]:
    """Pictures of an attachment as the VLM should see them (PDF pages rendered, images as-is)."""
    if att.is_pdf:
        with pymupdf.open(att.path) as doc:
            return [render(doc[i], dpi) for i in range(min(max_pages, len(doc)))]
    return [Image.open(att.path).convert("RGB")]


def doc_id_from_path(path: str | Path) -> str:
    return Path(path).stem
