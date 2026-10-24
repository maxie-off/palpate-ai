"""Heading-aware chunking. Each chunk keeps its start page and a highlight flag."""

import re
from collections.abc import Sequence

from app.core.config import ChunkingConfig
from app.models.schemas import Chunk, PageContent

# Chinese-numbered section headings such as "壹、 Milia" or "一、 病因：", and markdown headings.
HEADING_RE = re.compile(r"^\s*(?:[壹貳參肆伍陸柒捌玖拾]+|[一二三四五六七八九十]+)、|^\s*#{1,6}\s")


def chunk_pages(pages: Sequence[PageContent], doc_id: str, config: ChunkingConfig) -> list[Chunk]:
    """Split pages into chunks at headings or when a chunk would exceed ``max_chars``.

    A single block longer than ``max_chars`` stays whole rather than being cut mid-sentence.
    """
    chunks: list[Chunk] = []
    texts: list[str] = []
    start_page = 1
    emphasised = False

    def flush() -> None:
        nonlocal texts, emphasised
        if texts:
            chunks.append(
                Chunk(
                    id=f"{doc_id}-{len(chunks):04d}",
                    doc_id=doc_id,
                    page=start_page,
                    text="\n".join(texts),
                    emphasised=emphasised,
                    rule_version=config.rule_version,
                )
            )
        texts = []
        emphasised = False

    for page in pages:
        for block in page.blocks:
            text = block.text.strip()
            if not text:
                continue
            current_len = sum(len(t) for t in texts)
            if texts and (HEADING_RE.match(text) or current_len + len(text) > config.max_chars):
                flush()
            if not texts:
                start_page = page.page
            texts.append(text)
            emphasised = emphasised or block.emphasised
    flush()
    return chunks
