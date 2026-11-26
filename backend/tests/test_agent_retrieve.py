"""Hybrid retrieval: owner scope, highlight boost, glossary expansion, lexical bigrams."""

from pathlib import Path

from app.agent.lexical import render_search_text, rrf_component, search_tokens
from app.agent.service import run_agent
from app.agent.standins import HashingEmbedder, SourceCopyLLM, ZeroEmbedder
from app.agent.store import (
    InMemoryKnowledgeStore,
    PgVectorKnowledgeStore,
    SearchParams,
    vector_literal,
    with_search_text,
)
from app.agent.types import AgentRequest, ChunkSpan, IngestedDocument, StoredChunk
from app.retrieval.glossary import GlossaryEntry
from tests.test_agent_pipeline import make_deps


def _chunk(text: str, **overrides: object) -> StoredChunk:
    chunk_id = str(overrides.pop("id", "c1"))
    embedding = overrides.pop("embedding", HashingEmbedder().embed([text])[0])
    return with_search_text(
        StoredChunk(
            id=chunk_id,
            doc_id=str(overrides.pop("doc_id", "d1")),
            owner_id=str(overrides.pop("owner_id", "local")),
            text=text,
            page=int(overrides.pop("page", 1)),  # type: ignore[arg-type]
            embedding=embedding,  # type: ignore[arg-type]
            emphasised=bool(overrides.pop("emphasised", False)),
            lesson_id=str(overrides.pop("lesson_id", "")),
            spans=[
                ChunkSpan(anchor="char_range", c_start=0, c_end=len(text), excerpt=text),
            ],
        )
    )


def _doc(**overrides: object) -> IngestedDocument:
    return IngestedDocument(
        doc_id=str(overrides.pop("doc_id", "d1")),
        owner_id=str(overrides.pop("owner_id", "local")),
        title=str(overrides.pop("title", "notes")),
        content_hash=str(overrides.pop("content_hash", "hash")),
        lesson_id=str(overrides.pop("lesson_id", "")),
    )


def test_cjk_bigrams_and_rrf() -> None:
    tokens = search_tokens("人類乳突病毒 HPV")
    assert "乳突" in tokens
    assert "hpv" in tokens
    assert "人類乳突病毒" not in tokens
    assert rrf_component(1, 60) == 1 / 61
    assert rrf_component(None, 60) == 1 / 1060
    assert "乳突" in render_search_text("人類乳突病毒")


def test_search_never_crosses_owners_or_lessons() -> None:
    store = InMemoryKnowledgeStore()
    store.upsert(
        _doc(doc_id="da", owner_id="a", content_hash="ha"),
        [_chunk("milia keratin cysts", id="a1", doc_id="da", owner_id="a", lesson_id="L1")],
    )
    store.upsert(
        _doc(doc_id="db", owner_id="b", content_hash="hb"),
        [_chunk("milia keratin cysts", id="b1", doc_id="db", owner_id="b", lesson_id="L1")],
    )
    store.upsert(
        _doc(doc_id="da2", owner_id="a", content_hash="ha2"),
        [_chunk("milia keratin cysts", id="a2", doc_id="da2", owner_id="a", lesson_id="L2")],
    )
    vector = HashingEmbedder().embed(["milia keratin"])[0]
    owned = store.search("milia keratin", vector, SearchParams(owner_id="a", lesson_id="L1"))
    assert [hit.chunk.id for hit in owned] == ["a1"]


def test_emphasised_chunk_outranks_an_identical_passage() -> None:
    plain = _chunk("milia keratin cysts", id="plain")
    marked = _chunk("milia keratin cysts", id="marked", emphasised=True)
    marked = marked.model_copy(update={"embedding": list(plain.embedding)})
    store = InMemoryKnowledgeStore()
    store.upsert(_doc(), [plain, marked])
    hits = store.search(
        "milia keratin cysts",
        plain.embedding,
        SearchParams(owner_id="local", top_k=2),
    )
    assert [hit.chunk.id for hit in hits] == ["marked", "plain"]
    assert hits[0].fused_score > hits[1].fused_score


def test_lexical_hit_survives_a_dense_score_gate(tmp_path: Path) -> None:
    chinese = with_search_text(
        StoredChunk(
            id="zh",
            doc_id="d1",
            owner_id="local",
            text="粟丘疹常見於新生兒",
            page=1,
            embedding=[0.0, 0.0, 0.0, 0.0],
        )
    )
    other = with_search_text(
        StoredChunk(
            id="other",
            doc_id="d1",
            owner_id="local",
            text="unrelated enamel rods",
            page=2,
            embedding=[0.0, 0.0, 0.0, 0.0],
        )
    )
    store = InMemoryKnowledgeStore()
    store.upsert(_doc(), [chinese, other])
    gated = SearchParams(owner_id="local", min_dense_score=0.99, top_k=5)
    hits = store.search("粟丘疹", [0.0, 0.0, 0.0, 0.0], gated)
    assert [hit.chunk.id for hit in hits] == ["zh"]
    assert store.search("zzzz-no-such-token", [0.0, 0.0, 0.0, 0.0], gated) == []

    answer = run_agent(
        AgentRequest(task="qa", message="What is milia?"),
        make_deps(
            tmp_path,
            store=store,
            embedder=ZeroEmbedder(),
            llm=SourceCopyLLM(),
            glossary=(GlossaryEntry(zh="粟丘疹", en="Milia"),),
        ),
    )
    assert answer.status == "answered"
    assert any(citation.chunk_id == "zh" for citation in answer.citations)


class _Cursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, object]] = []

    def execute(self, sql: str, params: object = None) -> None:
        self.statements.append((" ".join(sql.split()), params))

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> list[object]:
        return []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Conn:
    def __init__(self) -> None:
        self.cursor_obj = _Cursor()
        self.commits = 0

    def cursor(self) -> _Cursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        return None


def test_pgvector_writes_spans_and_filters_by_owner() -> None:
    conn = _Conn()
    store = PgVectorKnowledgeStore(conn, embedding_dim=2)
    chunk = _chunk("milia", id="c1", embedding=[0.2, 0.4])
    store.upsert(_doc(), [chunk])
    sql = "\n".join(statement for statement, _params in conn.cursor_obj.statements)
    assert "INSERT INTO chunk_spans" in sql
    assert "owner_id" in sql
    assert conn.commits == 1

    store.search("milia", [0.2, 0.4], SearchParams(owner_id="student", lesson_id="L3"))
    search_sql, params = conn.cursor_obj.statements[-1]
    assert "<=>" in search_sql
    assert "owner_id" in search_sql
    assert "student" in params  # type: ignore[operator]
    assert vector_literal([0.5, 0.25]) == "[0.50000000,0.25000000]"


def test_pgvector_rejects_the_wrong_embedding_width() -> None:
    store = PgVectorKnowledgeStore(_Conn(), embedding_dim=2)
    try:
        store.upsert(_doc(), [_chunk("milia", embedding=[0.1])])
    except ValueError as exc:
        assert "PALPATE_EMBEDDING_DIM" in str(exc)
    else:
        raise AssertionError("expected a dimension error")
