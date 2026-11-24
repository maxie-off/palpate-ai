from app.answering.verifier import verify_citations
from app.models.schemas import Chunk, ChunkSpan, Citation

CHUNK = Chunk(
    id="c1",
    document_id="d1",
    spans=[ChunkSpan(page=7, bbox=(0, 0, 1, 1))],
    body="Dermoid cyst 皮樣囊腫 是一種 developmental cystic malformation",
)


def test_valid_quote_ignores_whitespace_and_case() -> None:
    cites = [Citation(chunk_id="c1", quote="皮樣囊腫是一種DEVELOPMENTAL cystic")]
    result = verify_citations(cites, {CHUNK.id: CHUNK})
    assert len(result.valid) == 1
    assert result.dropped == []


def test_valid_citation_is_enriched_with_source_location() -> None:
    cites = [Citation(chunk_id="c1", quote="Dermoid cyst")]
    valid = verify_citations(cites, {CHUNK.id: CHUNK}).valid[0]
    assert (valid.document_id, valid.page) == ("d1", 7)


def test_hallucinated_quote_is_dropped() -> None:
    cites = [Citation(chunk_id="c1", quote="a sentence that is not there")]
    result = verify_citations(cites, {CHUNK.id: CHUNK})
    assert result.valid == []
    assert len(result.dropped) == 1


def test_unknown_chunk_id_is_dropped() -> None:
    cites = [Citation(chunk_id="nope", quote="Dermoid cyst")]
    assert verify_citations(cites, {CHUNK.id: CHUNK}).valid == []


def test_empty_quote_is_dropped() -> None:
    cites = [Citation(chunk_id="c1", quote="   ")]
    assert verify_citations(cites, {CHUNK.id: CHUNK}).valid == []
