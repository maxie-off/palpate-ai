"""Span-aware chunking for the agent pipeline.

The legacy ``chunk_pages`` helper still splits text for the original quiz path. This module
keeps the same heading and size rules and also records a page rectangle and a character
range for every source line, which is the data highlighting needs and cannot be recovered
later without re-parsing.
"""

from collections.abc import Sequence

from app.agent.store import with_search_text
from app.agent.types import ChunkKind, ChunkSpan, DocType, Modality, StoredChunk
from app.core.config import ChunkingConfig
from app.ingestion.chunker import HEADING_RE
from app.models.schemas import BBox, PageContent, TextLine, Transcript


def _line_chunk(
    lines: list[TextLine],
    pages: list[int],
    *,
    doc_id: str,
    owner_id: str,
    lesson_id: str,
    doc_type: DocType,
    ordinal: int,
    breadcrumb: str,
    rule_version: str,
) -> StoredChunk:
    parts: list[str] = []
    spans: list[ChunkSpan] = []
    cursor = 0
    emphasised = False
    for index, (line, page) in enumerate(zip(lines, pages, strict=True)):
        if index:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(line.text)
        cursor += len(line.text)
        emphasised = emphasised or line.emphasised
        spans.append(
            ChunkSpan(
                anchor="page_bbox",
                page=page,
                x0=line.bbox.x0,
                y0=line.bbox.y0,
                x1=line.bbox.x1,
                y1=line.bbox.y1,
                excerpt=line.text,
            )
        )
        spans.append(ChunkSpan(anchor="char_range", c_start=start, c_end=cursor, excerpt=line.text))
    return with_search_text(
        StoredChunk(
            id=f"{doc_id}-{ordinal:04d}",
            doc_id=doc_id,
            owner_id=owner_id,
            lesson_id=lesson_id,
            doc_type=doc_type,
            kind="text",
            modality="text",
            page=pages[0] if pages else None,
            text="".join(parts),
            breadcrumb=breadcrumb,
            emphasised=emphasised,
            rule_version=rule_version,
            spans=spans,
        )
    )


def chunk_pages_for_index(
    pages: Sequence[PageContent],
    *,
    doc_id: str,
    owner_id: str,
    lesson_id: str,
    doc_type: DocType,
    title: str,
    config: ChunkingConfig,
) -> list[StoredChunk]:
    """Split extracted lines into section chunks, carrying one span per line."""
    flat: list[tuple[int, TextLine]] = []
    for page in pages:
        source_lines = page.lines
        if not source_lines and page.blocks:
            source_lines = [
                TextLine(
                    text=block.text.strip(),
                    bbox=block.bbox or BBox(x0=0, y0=0, x1=0, y1=0),
                    emphasised=block.emphasised,
                )
                for block in page.blocks
                if block.text.strip()
            ]
        for line in source_lines:
            text = line.text.strip()
            if text:
                flat.append((page.page, line.model_copy(update={"text": text})))

    chunks: list[StoredChunk] = []
    buffer: list[tuple[int, TextLine]] = []
    breadcrumb = title

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        chunks.append(
            _line_chunk(
                [line for _page, line in buffer],
                [page for page, _line in buffer],
                doc_id=doc_id,
                owner_id=owner_id,
                lesson_id=lesson_id,
                doc_type=doc_type,
                ordinal=len(chunks),
                breadcrumb=breadcrumb,
                rule_version=config.rule_version,
            )
        )
        buffer = []

    for page_number, line in flat:
        text = line.text
        current_len = sum(len(item.text) for _page, item in buffer)
        is_heading = HEADING_RE.match(text) is not None
        if buffer and (is_heading or current_len + len(text) > config.max_chars):
            flush()
        if is_heading:
            breadcrumb = text
        elif not breadcrumb:
            breadcrumb = title
        buffer.append((page_number, line))
    flush()
    return chunks


def make_media_chunk(
    *,
    doc_id: str,
    owner_id: str,
    lesson_id: str,
    doc_type: DocType,
    ordinal: int,
    text: str,
    kind: ChunkKind,
    modality: Modality,
    breadcrumb: str,
    rule_version: str,
    spans: list[ChunkSpan],
    page: int | None = None,
    image_path: str | None = None,
) -> StoredChunk:
    return with_search_text(
        StoredChunk(
            id=f"{doc_id}-{ordinal:04d}",
            doc_id=doc_id,
            owner_id=owner_id,
            lesson_id=lesson_id,
            doc_type=doc_type,
            kind=kind,
            modality=modality,
            page=page,
            text=text,
            breadcrumb=breadcrumb,
            image_path=image_path,
            rule_version=rule_version,
            spans=spans,
        )
    )


def chunks_from_transcript(
    transcript: Transcript,
    *,
    doc_id: str,
    owner_id: str,
    lesson_id: str,
    title: str,
    rule_version: str,
) -> list[StoredChunk]:
    chunks: list[StoredChunk] = []
    for segment in transcript.segments:
        text = segment.text.strip()
        if not text:
            continue
        chunks.append(
            make_media_chunk(
                doc_id=doc_id,
                owner_id=owner_id,
                lesson_id=lesson_id,
                doc_type="video",
                ordinal=len(chunks),
                text=text,
                kind="transcript",
                modality="audio",
                breadcrumb=title,
                rule_version=rule_version,
                page=None,
                spans=[
                    ChunkSpan(
                        anchor="time_range",
                        t_start=segment.t_start,
                        t_end=segment.t_end,
                        excerpt=text,
                    ),
                    ChunkSpan(anchor="char_range", c_start=0, c_end=len(text), excerpt=text),
                ],
            )
        )
    return chunks
