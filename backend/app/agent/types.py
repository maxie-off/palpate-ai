"""Typed shapes for the agent pipeline, including highlightable citations.

Anchors follow the three shapes in the schema draft: a PDF rectangle, an audio time range,
or a character range. A frontend can switch on ``kind`` and draw or seek without further
parsing.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

DocType = Literal["note", "slides", "handout", "video", "textbook", "quiz"]
ChunkKind = Literal["text", "image_caption", "transcript"]
Modality = Literal["text", "image", "audio"]
TaskName = Literal["ingest", "draft_notes", "qa", "quiz", "summarize"]
AnchorKind = Literal["page_bbox", "time_range", "char_range"]
RunStatus = Literal["ok", "answered", "insufficient_evidence", "not_implemented", "failed"]

PARSER_PDF = "pymupdf_layout_v1"
PARSER_AUDIO = "stt_v1"


class PageBBoxAnchor(BaseModel):
    kind: Literal["page_bbox"] = "page_bbox"
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class TimeRangeAnchor(BaseModel):
    kind: Literal["time_range"] = "time_range"
    t_start: float
    t_end: float


class CharRangeAnchor(BaseModel):
    """Character offsets into the chunk body. ``c_end`` is exclusive."""

    kind: Literal["char_range"] = "char_range"
    c_start: int
    c_end: int


SourceAnchor = Annotated[
    PageBBoxAnchor | TimeRangeAnchor | CharRangeAnchor,
    Field(discriminator="kind"),
]


class ChunkSpan(BaseModel):
    """One source line (or one audio segment) captured at parse time."""

    anchor: AnchorKind
    page: int | None = None
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    t_start: float | None = None
    t_end: float | None = None
    c_start: int | None = None
    c_end: int | None = None
    excerpt: str = ""

    def to_anchor(self) -> SourceAnchor | None:
        if self.anchor == "page_bbox":
            page, x0, y0, x1, y1 = self.page, self.x0, self.y0, self.x1, self.y1
            if page is None or x0 is None or y0 is None or x1 is None or y1 is None:
                return None
            return PageBBoxAnchor(page=page, x0=x0, y0=y0, x1=x1, y1=y1)
        if self.anchor == "time_range":
            if self.t_start is None or self.t_end is None:
                return None
            return TimeRangeAnchor(t_start=self.t_start, t_end=self.t_end)
        if self.c_start is None or self.c_end is None:
            return None
        return CharRangeAnchor(c_start=self.c_start, c_end=self.c_end)


class StoredChunk(BaseModel):
    id: str
    doc_id: str
    owner_id: str
    lesson_id: str = ""
    doc_type: DocType = "handout"
    kind: ChunkKind = "text"
    modality: Modality = "text"
    page: int | None = None
    text: str
    breadcrumb: str = ""
    search_text: str = ""
    emphasised: bool = False
    image_path: str | None = None
    rule_version: str = "v1"
    spans: list[ChunkSpan] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list, repr=False)


class IngestedDocument(BaseModel):
    doc_id: str
    owner_id: str
    lesson_id: str = ""
    doc_type: DocType = "handout"
    title: str
    source_path: str = ""
    mime_type: str = "application/pdf"
    content_hash: str
    page_count: int = 0
    chunk_count: int = 0
    figure_count: int = 0
    cached: bool = False
    parser: str = PARSER_PDF
    parser_version: str = "v1"


class SearchHit(BaseModel):
    chunk: StoredChunk
    fused_score: float = 0.0
    dense_similarity: float | None = None
    lexical_overlap: int = 0


class SourceCitation(BaseModel):
    """A verified quote plus the anchors a frontend needs to highlight it."""

    id: str
    rank: int
    claim: str
    quote: str
    doc_id: str
    chunk_id: str
    doc_type: str
    modality: Modality
    anchors: list[SourceAnchor] = Field(default_factory=list)
    image_path: str | None = None
    verified: bool = True
    fused_score: float | None = None


class Claim(BaseModel):
    text: str
    citations: list[SourceCitation] = Field(default_factory=list)


class Highlight(BaseModel):
    doc_id: str
    chunk_id: str
    quote: str
    anchor: SourceAnchor


class NoteSection(BaseModel):
    key: str
    heading: str
    body_md: str = ""
    claims: list[Claim] = Field(default_factory=list)


class IngestSummary(BaseModel):
    documents: list[IngestedDocument] = Field(default_factory=list)


class AgentRequest(BaseModel):
    """One call into the pipeline. Set ``task`` when the caller already knows it."""

    task: TaskName | None = None
    message: str = ""
    owner_id: str = "local"
    lesson_id: str = ""
    pdf_path: str = ""
    audio_path: str = ""
    doc_id: str = ""
    title: str = ""
    doc_type: DocType = "handout"


class AgentResult(BaseModel):
    task: str
    status: RunStatus
    error: str = ""
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    sections: list[NoteSection] = Field(default_factory=list)
    citations: list[SourceCitation] = Field(default_factory=list)
    highlights: list[Highlight] = Field(default_factory=list)
    ingest: IngestSummary | None = None
    attempts: int = 0
    trace: list[str] = Field(default_factory=list)
