"""Shared data shapes passed between pipeline stages."""

from typing import Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    text: str
    emphasised: bool = False


class PageContent(BaseModel):
    page: int
    blocks: list[TextBlock] = Field(default_factory=list)


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
