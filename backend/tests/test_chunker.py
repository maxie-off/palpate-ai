import uuid

from app.core.config import ChunkingConfig
from app.ingestion.chunker import chunk_pages
from app.models.schemas import Chunk, Line, PageContent, TextBlock

DOC = str(uuid.UUID(int=1))


def _block(text: str, emphasised: bool = False, y: float = 700.0) -> TextBlock:
    emphasis = 1.0 if emphasised else 0.0
    line = Line(text=text, bbox=(72.0, y, 300.0, y + 12), emphasis=emphasis)
    return TextBlock(text=text, emphasis=emphasis, lines=[line])


def _page(n: int, *blocks: tuple[str, bool]) -> PageContent:
    return PageContent(page=n, blocks=[_block(t, e) for t, e in blocks])


def _sections(chunks: list[Chunk]) -> list[Chunk]:
    return [c for c in chunks if c.kind == "section"]


def test_topics_split_on_upper_numeral_headings() -> None:
    pages = [
        _page(
            1,
            ("壹、 Milia 粟丘疹", False),
            ("body one", False),
            ("貳、 Teratoma 畸胎瘤", False),
            ("a long emphasised body paragraph", True),
        )
    ]
    chunks = chunk_pages(pages, DOC, "Notes", ChunkingConfig(max_chars=500))
    topics = [c for c in chunks if c.kind == "topic"]
    assert [t.heading for t in topics] == ["壹、 Milia 粟丘疹", "貳、 Teratoma 畸胎瘤"]
    assert [t.emphasised for t in topics] == [False, True]


def test_sections_are_children_of_their_topic_with_breadcrumb() -> None:
    pages = [
        _page(
            1,
            ("壹、 Milia", False),
            ("一、 病因", False),
            ("cause text", False),
            ("二、 臨床表現", False),
            ("clinical text", False),
        )
    ]
    chunks = chunk_pages(pages, DOC, "口病共筆", ChunkingConfig())
    topic = chunks[0]
    sections = _sections(chunks)
    assert topic.kind == "topic" and topic.parent_id is None
    assert [s.heading for s in sections] == ["一、 病因", "二、 臨床表現"]
    assert all(s.parent_id == topic.id for s in sections)
    assert sections[0].breadcrumb == "口病共筆 > 壹、 Milia > 一、 病因"
    assert [s.ordinal for s in sections] == [0, 1]
    # The topic heading stays with the first sub-section rather than becoming an empty chunk.
    assert len(sections) == 2
    assert "壹、 Milia" in sections[0].body
    assert "cause text" in topic.body and "clinical text" in topic.body


def test_embed_text_prepends_breadcrumb() -> None:
    pages = [_page(1, ("壹、 Milia", False), ("50% of newborns show lesions", False))]
    section = _sections(chunk_pages(pages, DOC, "Notes", ChunkingConfig()))[0]
    assert section.embed_text.startswith("Notes > 壹、 Milia\n")
    assert "50% of newborns" in section.embed_text


def test_splits_section_when_size_exceeded_keeping_heading() -> None:
    pages = [_page(1, ("壹、 Topic", False), ("a" * 60, False), ("b" * 60, False))]
    sections = _sections(chunk_pages(pages, DOC, "N", ChunkingConfig(max_chars=100)))
    assert len(sections) == 2
    assert {s.heading for s in sections} == {"壹、 Topic"}
    assert [s.ordinal for s in sections] == [0, 1]


def test_section_records_line_spans_and_start_page() -> None:
    pages = [_page(1, ("壹、 One", False)), _page(2, ("body on page two", False))]
    section = _sections(chunk_pages(pages, DOC, "N", ChunkingConfig()))[0]
    assert section.page == 1
    assert [sp.page for sp in section.spans] == [1, 2]
    assert section.spans[0].bbox == (72.0, 700.0, 300.0, 712.0)


def test_text_before_first_heading_gets_an_intro_topic() -> None:
    pages = [_page(1, ("preface text", False), ("壹、 First", False), ("x", False))]
    chunks = chunk_pages(pages, DOC, "Notes", ChunkingConfig())
    topics = [c for c in chunks if c.kind == "topic"]
    assert [t.heading for t in topics] == [None, "壹、 First"]
    assert topics[0].breadcrumb == "Notes"


def test_ids_are_deterministic_uuids() -> None:
    pages = [_page(1, ("壹、 A", False), ("text", False))]
    first = chunk_pages(pages, DOC, "N", ChunkingConfig())
    second = chunk_pages(pages, DOC, "N", ChunkingConfig())
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
    uuid.UUID(first[0].id)


def test_ignores_empty_blocks() -> None:
    pages = [_page(1, ("  ", False), ("text", False))]
    sections = _sections(chunk_pages(pages, DOC, "N", ChunkingConfig()))
    assert len(sections) == 1
    assert sections[0].body == "text"


def test_chunk_is_emphasised_only_above_the_character_threshold() -> None:
    plain, stressed = "p" * 80, "s" * 20  # 20% of the characters are stressed
    pages = [_page(1, ("壹、 T", False), (plain, False), (stressed, True))]
    low = _sections(chunk_pages(pages, DOC, "N", ChunkingConfig(emphasis_threshold=0.3)))[0]
    high = _sections(chunk_pages(pages, DOC, "N", ChunkingConfig(emphasis_threshold=0.1)))[0]
    assert 0.15 < low.emphasis < 0.25
    assert not low.emphasised and high.emphasised


def test_page_numbers_and_header_block_are_not_chunked() -> None:
    pages = [
        _page(
            1, ("- 1 -", False), ("撰寫者：林昱瑄", False), ("壹、 First", False), ("body", False)
        )
    ]
    chunks = chunk_pages(pages, DOC, "N", ChunkingConfig())
    assert [c.heading for c in chunks if c.kind == "topic"] == ["壹、 First"]
    assert all("撰寫者" not in c.body and "- 1 -" not in c.body for c in chunks)


def test_bare_numeral_heading_keeps_the_title_on_its_second_line() -> None:
    block = TextBlock(
        text="壹拾、\nTuberous sclerosis",
        lines=[
            Line(text="壹拾、", bbox=(0, 0, 1, 1)),
            Line(text="Tuberous sclerosis", bbox=(0, 0, 1, 1)),
        ],
    )
    pages = [PageContent(page=1, blocks=[block, _block("body")])]
    topic = chunk_pages(pages, DOC, "N", ChunkingConfig())[0]
    assert topic.heading == "壹拾、 Tuberous sclerosis"
