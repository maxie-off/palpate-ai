"""Shared data shapes passed between pipeline stages."""

from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Rectangle in PDF user space (origin bottom-left, y increasing upward)."""

    x0: float
    y0: float
    x1: float
    y1: float


class TextBlock(BaseModel):
    text: str
    emphasised: bool = False
    bbox: BBox | None = None


class TextLine(BaseModel):
    """One visual line, the unit a highlight anchor is attached to."""

    text: str
    bbox: BBox
    emphasised: bool = False


class FigureBlock(BaseModel):
    """A placed image. Coordinates are PDF user space, same as text lines."""

    xref: int
    bbox: BBox
    width: int = 0
    height: int = 0
    ext: str = "png"
    image_bytes: bytes = b""


class PageContent(BaseModel):
    page: int
    width: float = 0
    height: float = 0
    blocks: list[TextBlock] = Field(default_factory=list)
    lines: list[TextLine] = Field(default_factory=list)
    figures: list[FigureBlock] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    text: str
    t_start: float
    t_end: float


class Transcript(BaseModel):
    """Speech-to-text output. Timestamps are seconds from the start of the recording."""

    model: str
    language: str = ""
    segments: list[TranscriptSegment] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    doc_id: str
    page: int
    text: str
    kind: Literal["text", "image_caption"] = "text"
    emphasised: bool = False
    image_path: str | None = None
    rule_version: str = "v1"


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    chunk_id: str
    quote: str
    doc_id: str = ""
    page: int | None = None
    image_path: str | None = None


class ParsedQuestion(BaseModel):
    id: str
    stem: str
    options: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)


class QuestionAnswer(BaseModel):
    question_id: str
    status: Literal["answered", "insufficient_evidence"]
    summary: str = ""
    citations: list[Citation] = Field(default_factory=list)
