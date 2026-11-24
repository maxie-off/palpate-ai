"""Shared data shapes passed between pipeline stages."""

from typing import Literal

from pydantic import BaseModel, Field

# (x0, y0, x1, y1) in PDF user space: origin bottom-left, y up, valid at any zoom.
BBox = tuple[float, float, float, float]


class Line(BaseModel):
    """One source line with its position, captured while parsing (anchors cannot be recovered)."""

    text: str
    bbox: BBox
    emphasised: bool = False


class TextBlock(BaseModel):
    text: str
    emphasised: bool = False
    lines: list[Line] = Field(default_factory=list)


class PageContent(BaseModel):
    page: int
    blocks: list[TextBlock] = Field(default_factory=list)


class ChunkSpan(BaseModel):
    """A `page_bbox` anchor: one row of `chunk_spans`."""

    page: int
    bbox: BBox


class Chunk(BaseModel):
    """A retrievable unit. `topic` chunks are parents, `section` chunks are what gets embedded.

    ``ordinal`` is the position among siblings: topics within the document, sections within
    their topic. Only sections carry spans; a topic is the union of its sections.
    """

    id: str
    document_id: str
    kind: Literal["topic", "section", "quiz_item"] = "section"
    parent_id: str | None = None
    ordinal: int = 0
    breadcrumb: str = ""
    heading: str | None = None
    body: str
    emphasised: bool = False
    spans: list[ChunkSpan] = Field(default_factory=list)

    @property
    def page(self) -> int | None:
        return self.spans[0].page if self.spans else None

    @property
    def char_count(self) -> int:
        return len(self.body)

    @property
    def embed_text(self) -> str:
        """What gets embedded: the breadcrumb gives a section context its body may lack."""
        return f"{self.breadcrumb}\n{self.body}" if self.breadcrumb else self.body


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    chunk_id: str
    quote: str
    document_id: str = ""
    page: int | None = None


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
