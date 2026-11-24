"""Heading-aware, two-level chunking (small-to-big).

* A ``topic`` starts at a top-level heading (壹、 / # ) and holds the whole topic: this is what
  generation receives for context.
* A ``section`` starts at a sub-heading (一、 / ## ), or when the current one would exceed
  ``max_chars``. Sections are what get embedded and matched, for precision.

Each section keeps the source line positions of its text so a citation can be drawn on the
original page.
"""

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.config import ChunkingConfig
from app.ingestion.furniture import is_furniture
from app.models.schemas import Chunk, ChunkSpan, PageContent

_UPPER_NUMERALS = "壹貳參肆伍陸柒捌玖拾"
_LOWER_NUMERALS = "一二三四五六七八九十"

TOPIC_RE = re.compile(rf"^\s*(?:[{_UPPER_NUMERALS}]+、|#\s)")
SECTION_RE = re.compile(rf"^\s*(?:[{_LOWER_NUMERALS}]+、|#{{2,6}}\s)")

_MAX_HEADING_CHARS = 80


@dataclass
class _Section:
    heading: str | None
    texts: list[str] = field(default_factory=list)
    spans: list[ChunkSpan] = field(default_factory=list)
    emphasis_chars: float = 0.0
    heading_only: bool = False

    @property
    def length(self) -> int:
        return sum(len(t) for t in self.texts)


@dataclass
class _Topic:
    heading: str | None
    sections: list[_Section] = field(default_factory=list)


def _heading_text(block_text: str) -> str:
    lines = [ln.strip() for ln in block_text.strip().splitlines() if ln.strip()]
    # A bare numeral line ("壹拾、") keeps its title, which PDFs often put on the next line.
    first = f"{lines[0]} {lines[1]}" if lines[0].endswith("、") and len(lines) > 1 else lines[0]
    return first[:_MAX_HEADING_CHARS]


def _breadcrumb(*parts: str | None) -> str:
    out: list[str] = []
    for part in parts:
        if part and (not out or out[-1] != part):
            out.append(part)
    return " > ".join(out)


def chunk_pages(
    pages: Sequence[PageContent], document_id: str, doc_title: str, config: ChunkingConfig
) -> list[Chunk]:
    """Split parsed pages into topic chunks, each followed by its section chunks.

    ``document_id`` must be a UUID string; chunk ids are derived from it deterministically so
    re-chunking the same document yields the same ids.
    """
    topics: list[_Topic] = []

    def current_topic() -> _Topic:
        if not topics:
            topics.append(_Topic(heading=None))
        return topics[-1]

    def new_section(heading: str | None, *, heading_only: bool = False) -> _Section:
        section = _Section(heading=heading, heading_only=heading_only)
        current_topic().sections.append(section)
        return section

    section: _Section | None = None
    for page in pages:
        for block in page.blocks:
            text = block.text.strip()
            if not text or is_furniture(text):
                continue
            if TOPIC_RE.match(text):
                topics.append(_Topic(heading=_heading_text(text)))
                section = new_section(_heading_text(text), heading_only=True)
            elif SECTION_RE.match(text):
                if section is not None and section.heading_only:
                    # A topic heading directly followed by a sub-heading: keep them together.
                    section.heading = _heading_text(text)
                else:
                    section = new_section(_heading_text(text), heading_only=True)
            elif section is None:
                section = new_section(None)
            elif section.length and section.length + len(text) > config.max_chars:
                section = new_section(section.heading)
            if section is None:
                section = new_section(None)
            section.texts.append(text)
            section.emphasis_chars += block.emphasis * len(text)
            if section.length > len(text):
                section.heading_only = False
            section.spans.extend(ChunkSpan(page=page.page, bbox=ln.bbox) for ln in block.lines)
    return _emit(topics, document_id, doc_title, config.emphasis_threshold)


def _emit(topics: list[_Topic], document_id: str, doc_title: str, threshold: float) -> list[Chunk]:
    namespace = uuid.UUID(document_id)
    chunks: list[Chunk] = []
    for t_index, topic in enumerate(topics):
        sections = [s for s in topic.sections if s.texts]
        if not sections:
            continue
        topic_chars = sum(s.length for s in sections)
        topic_emphasis = sum(s.emphasis_chars for s in sections) / topic_chars
        topic_id = str(uuid.uuid5(namespace, f"topic:{t_index}"))
        topic_crumb = _breadcrumb(doc_title, topic.heading)
        chunks.append(
            Chunk(
                id=topic_id,
                document_id=document_id,
                kind="topic",
                ordinal=t_index,
                breadcrumb=topic_crumb,
                heading=topic.heading,
                body="\n".join("\n".join(s.texts) for s in sections),
                emphasis=topic_emphasis,
                emphasised=topic_emphasis >= threshold,
            )
        )
        for s_index, sec in enumerate(sections):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid5(namespace, f"section:{t_index}:{s_index}")),
                    document_id=document_id,
                    kind="section",
                    parent_id=topic_id,
                    ordinal=s_index,
                    breadcrumb=_breadcrumb(doc_title, topic.heading, sec.heading),
                    heading=sec.heading,
                    body="\n".join(sec.texts),
                    emphasis=sec.emphasis_chars / sec.length,
                    emphasised=sec.emphasis_chars / sec.length >= threshold,
                    spans=sec.spans,
                )
            )
    return chunks
