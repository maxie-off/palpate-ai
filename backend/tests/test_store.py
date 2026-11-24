import hashlib
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from app.core.config import DomainConfig
from app.ingestion.store import embed_document, ingest_document, user_transaction


class FakeEmbedder:
    def __init__(self, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls = 0
        self.input_types: list[str] = []

    def complete(self, *, system: str, user: str, model_id: str, max_tokens: int = 1024) -> str:
        raise AssertionError("not used")

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        self.calls += 1
        self.input_types.append(input_type)
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("bedrock throttled")
        out = []
        for text in texts:
            vec = [0.0] * 1024
            vec[int(hashlib.md5(text.encode()).hexdigest(), 16) % 1024] = 1.0
            out.append(vec)
        return out


def _pdf(path: Path, marker: str = "") -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 80
    for line in [
        "# Milia",
        "## Cause",
        f"Retained keratin forms small cysts. {marker}",
        "## Clinical features",
        "White papules on the face of newborns.",
        "# Teratoma",
        "A germ cell tumour with several tissue types.",
    ]:
        page.insert_text((72, y), line)
        y += 30
    doc.save(path)
    doc.close()
    return path


def _count(conn: Any, owner: str, sql: str, *args: Any) -> int:
    with user_transaction(conn, owner):
        return int(conn.execute(sql, args).fetchone()[0])


def _ingest(conn: Any, owner: str, path: Path) -> Any:
    return ingest_document(
        conn, owner_id=owner, path=path, doc_type="note", config=DomainConfig(), title="Notes"
    )


def test_ingest_stores_document_chunks_and_spans(db: Any, tmp_path: Path) -> None:
    owner = db.new_user()
    result = _ingest(db, owner, _pdf(tmp_path / "a.pdf"))

    assert result.status == "parsed" and not result.duplicate
    with user_transaction(db, owner):
        rows = db.execute(
            "SELECT kind, heading, breadcrumb, parent_id IS NOT NULL, embedding IS NULL, "
            "search_text, doc_type::text FROM chunks WHERE document_id = %s "
            "ORDER BY breadcrumb",
            (result.document_id,),
        ).fetchall()
    sections = [r for r in rows if r[0] == "section"]
    assert len([r for r in rows if r[0] == "topic"]) == 2
    assert [r[1] for r in sections] == ["## Cause", "## Clinical features", "# Teratoma"]
    assert sections[0][2] == "Notes > # Milia > ## Cause"
    assert all(r[3] and r[4] for r in sections)  # has parent, no embedding yet
    assert "keratin" in sections[0][5] and sections[0][6] == "note"
    assert _count(db, owner, "SELECT count(*) FROM chunk_spans") > 0


def test_reupload_of_identical_bytes_is_a_noop(db: Any, tmp_path: Path) -> None:
    owner = db.new_user()
    path = _pdf(tmp_path / "a.pdf")
    first = _ingest(db, owner, path)
    second = _ingest(db, owner, path)

    assert second.duplicate and second.document_id == first.document_id
    assert second.chunk_count == first.chunk_count
    assert _count(db, owner, "SELECT count(*) FROM documents") == 1


def test_embed_document_fills_sections_only_and_is_idempotent(db: Any, tmp_path: Path) -> None:
    owner = db.new_user()
    doc = _ingest(db, owner, _pdf(tmp_path / "a.pdf")).document_id
    llm = FakeEmbedder()

    assert embed_document(db, llm, owner_id=owner, document_id=doc, embed_model="m") == 3
    assert embed_document(db, llm, owner_id=owner, document_id=doc, embed_model="m") == 0

    assert llm.input_types == ["document"]
    assert _count(db, owner, "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL") == 3
    assert (
        _count(
            db, owner, "SELECT count(*) FROM chunks WHERE kind='topic' AND embedding IS NOT NULL"
        )
        == 0
    )
    with user_transaction(db, owner):
        status, model = db.execute(
            "SELECT d.ingest_status, c.embed_model FROM documents d JOIN chunks c "
            "ON c.document_id = d.id WHERE c.kind = 'section' LIMIT 1"
        ).fetchone()
    assert (status, model) == ("embedded", "m")


def test_failed_embedding_is_recorded_and_can_be_retried(db: Any, tmp_path: Path) -> None:
    owner = db.new_user()
    doc = _ingest(db, owner, _pdf(tmp_path / "a.pdf")).document_id

    with pytest.raises(RuntimeError):
        embed_document(
            db, FakeEmbedder(fail_after=0), owner_id=owner, document_id=doc, embed_model="m"
        )
    with user_transaction(db, owner):
        status, error = db.execute(
            "SELECT ingest_status, ingest_error FROM documents WHERE id = %s", (doc,)
        ).fetchone()
    assert status == "failed" and "throttled" in error

    assert embed_document(db, FakeEmbedder(), owner_id=owner, document_id=doc, embed_model="m") == 3
    assert _count(db, owner, "SELECT count(*) FROM documents WHERE ingest_status='embedded'") == 1


def test_users_cannot_see_each_others_documents(db: Any, tmp_path: Path) -> None:
    alice, bob = db.new_user(), db.new_user()
    _ingest(db, alice, _pdf(tmp_path / "a.pdf"))

    assert _count(db, alice, "SELECT count(*) FROM chunks") > 0
    assert _count(db, bob, "SELECT count(*) FROM chunks") == 0
    assert _count(db, bob, "SELECT count(*) FROM documents") == 0
    # The same file is not a duplicate for another user: each knowledge base is private.
    assert not _ingest(db, bob, tmp_path / "a.pdf").duplicate


def test_cannot_insert_rows_for_another_owner(db: Any) -> None:
    alice, bob = db.new_user(), db.new_user()
    with pytest.raises(Exception, match="row-level security"), user_transaction(db, alice):
        db.execute("INSERT INTO subjects (owner_id, code, name) VALUES (%s, 'X', 'x')", (bob,))
