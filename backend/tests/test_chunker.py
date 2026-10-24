from app.core.config import ChunkingConfig
from app.ingestion.chunker import chunk_pages
from app.models.schemas import PageContent, TextBlock


def _page(n: int, *blocks: tuple[str, bool]) -> PageContent:
    return PageContent(page=n, blocks=[TextBlock(text=t, emphasised=e) for t, e in blocks])


def test_splits_on_chinese_numbered_headings() -> None:
    pages = [
        _page(
            1,
            ("壹、 Milia 粟丘疹", False),
            ("body one", False),
            ("貳、 Teratoma 畸胎瘤", False),
            ("body two", True),
        )
    ]
    chunks = chunk_pages(pages, "d1", ChunkingConfig(max_chars=500))
    assert len(chunks) == 2
    assert chunks[0].id == "d1-0000"
    assert not chunks[0].emphasised
    assert chunks[1].emphasised


def test_splits_when_size_exceeded() -> None:
    pages = [_page(1, ("a" * 60, False), ("b" * 60, False))]
    chunks = chunk_pages(pages, "d1", ChunkingConfig(max_chars=100))
    assert len(chunks) == 2


def test_chunk_records_start_page_and_rule_version() -> None:
    pages = [_page(1, ("intro", False)), _page(2, ("貳、 next", False), ("more", False))]
    chunks = chunk_pages(pages, "d1", ChunkingConfig(rule_version="v9"))
    assert [c.page for c in chunks] == [1, 2]
    assert all(c.rule_version == "v9" for c in chunks)


def test_ignores_empty_blocks() -> None:
    pages = [_page(1, ("  ", False), ("text", False))]
    chunks = chunk_pages(pages, "d1", ChunkingConfig())
    assert len(chunks) == 1
    assert chunks[0].text == "text"
