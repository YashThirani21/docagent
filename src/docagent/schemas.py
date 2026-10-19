"""Contracts shared by every component. Pipelines exchange these objects, never raw dicts."""
from __future__ import annotations

import hashlib
import mimetypes
import uuid
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- inputs

class Attachment(BaseModel):
    path: str
    mime: str
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> "Attachment":
        path = Path(path)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return cls(path=str(path), mime=mime, sha256=sha256_file(path))

    @property
    def is_pdf(self) -> bool:
        return self.mime == "application/pdf"

    @property
    def is_image(self) -> bool:
        return self.mime.startswith("image/")


class Request(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str | None = None
    attachments: list[Attachment] = []

    @classmethod
    def build(cls, text: str | None = None, files: list[str | Path] | None = None) -> "Request":
        return cls(text=text or None, attachments=[Attachment.from_path(f) for f in files or []])


# ---------------------------------------------------------------- routing

class Route(str, Enum):
    REPORT_QA = "report_qa"   # answer from the report corpus (text and/or pictures)
    DRAWING = "drawing"       # extract every value from a technical drawing into JSON


class RouteDecision(BaseModel):
    route: Route
    attachment: str | None = None          # path of the attachment this decision is about
    confidence: float
    tier: Literal["rule", "vlm"]
    reason: str


# ---------------------------------------------------------------- corpus

class Chunk(BaseModel):
    chunk_id: str                      # deterministic: same input -> same id (idempotent ingestion)
    doc_id: str                        # e.g. "ntsb_18-061"
    doc_sha256: str
    page: int                          # 1-based
    kind: Literal["text", "figure"]
    text: str                          # what the user sees / the LLM reads
    embed_text: str                    # text + context header, what gets embedded
    section: str | None = None
    subject: str | None = None         # what this text is about, e.g. "2. Airplane N104ER Left Main Wing Spar"
    figure_no: int | None = None       # for figure chunks
    figure_refs: list[int] = []        # for text chunks: figures this text mentions
    image_path: str | None = None      # figure crop on disk
    description: str | None = None     # VLM figure description: used for retrieval only, never as evidence


class Citation(BaseModel):
    doc_id: str
    page: int
    chunk_id: str | None = None
    figure_no: int | None = None


class RetrievedItem(BaseModel):
    chunk: Chunk
    score: float
    via: Literal["hybrid", "linked"] = "hybrid"   # linked = pulled in via figure reference


# ---------------------------------------------------------------- pipeline A output

class ReportAnswer(BaseModel):
    answer: str                               # direct answer in plain language
    diagnosis: str | None = None              # what is wrong (for picture / problem inputs)
    recommended_actions: list[str] = []       # what to do; inferred from report findings
    citations: list[Citation] = []
    grounded: bool = True                     # False -> the reports do not cover the question
    image_observations: list[str] = []        # what the VLM saw in the user's pictures
    retrieved: list[Citation] = []            # everything that was put in context (for debugging/eval)


# ---------------------------------------------------------------- pipeline B output

ValueType = Literal[
    "dimension", "angle", "radius", "diameter", "tolerance", "gdt_frame", "datum_feature",
    "datum_target", "surface_finish", "thread", "chamfer", "note", "title_block", "other",
]


class DrawingValue(BaseModel):
    id: str
    page: int
    raw_text: str                              # exactly as printed
    type: ValueType
    quantity: int | None = None                # "4X" -> 4
    nominal: float | None = None
    unit: str | None = None
    tol_plus: float | None = None
    tol_minus: float | None = None
    characteristic: str | None = None          # GD&T: position, flatness, profile of a surface, ...
    datums: list[str] = []
    modifiers: list[str] = []                  # MMC, LMC, basic, statistical, ...
    belongs_to: str                            # feature / view it is attached to
    meaning: str                               # plain-language interpretation
    bbox: list[float] | None = None            # [x0,y0,x1,y1] page coordinates (see coord_units)
    source: Literal["vlm", "text_layer", "vlm+text_layer"]
    confidence: float = 0.5


class DrawingPage(BaseModel):
    page: int
    view: str | None = None
    description: str | None = None


class DrawingExtraction(BaseModel):
    drawing_id: str
    source_file: str
    source_sha256: str
    units: str | None = None
    coord_units: Literal["pt", "px"] = "pt"   # bbox units: PDF points, or pixels for image inputs
    title_block: dict[str, str] = {}
    notes: list[str] = []
    pages: list[DrawingPage] = []
    values: list[DrawingValue] = []
    unmatched_text_tokens: list[str] = []     # text-layer annotation tokens no value accounts for
    extractor_version: str = ""


# ---------------------------------------------------------------- final response

class Response(BaseModel):
    request_id: str
    routes: list[RouteDecision]
    report_answer: ReportAnswer | None = None
    drawings: list[DrawingExtraction] = []
    errors: list[str] = []
