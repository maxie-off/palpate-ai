"""Persist a parsed document: dedup, chunks, spans, then embeddings.

Every function takes an open psycopg connection and the owner's id. Each transaction first sets
``app.current_user_id`` so Postgres row-level security applies even if a ``WHERE owner_id``
is forgotten. The id must come from the verified login token, never from a request parameter.

Ingest is two phases so a failed embedding call never loses the parsed document:
``ingest_document`` commits the document and chunks as ``parsed``; ``embed_document`` fills the
vectors and flips the status to ``embedded`` (or ``failed``, and can simply be called again).
"""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.core.config import DomainConfig
from app.core.llm import LLMClient
from app.ingestion.chunker import chunk_pages
from app.ingestion.parser import DocumentParser, PyMuPDFParser
from app.ingestion.text import make_search_text
from app.models.schemas import Chunk

EMBED_BATCH = 96  # Cohere on Bedrock accepts at most 96 texts per call
PARSER_NAME = "pymupdf"

DocType = Literal["note", "slides", "handout", "video", "textbook", "quiz"]


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    status: str  # ingest_status of the stored document
    chunk_count: int
    duplicate: bool


@contextmanager
def user_transaction(conn: Any, owner_id: str) -> Iterator[Any]:
    """One transaction scoped to a user: commits on success, rolls back on error."""
    with conn.transaction():
        conn.execute("SELECT set_config('app.current_user_id', %s, true)", (owner_id,))
        yield conn


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def ingest_document(
    conn: Any,
    *,
    owner_id: str,
    path: Path,
    doc_type: DocType,
    config: DomainConfig,
    title: str | None = None,
    lesson_id: str | None = None,
    subject_id: str | None = None,
    parser: DocumentParser | None = None,
) -> IngestResult:
    """Parse and chunk a PDF and store it. A re-upload of identical bytes is a no-op."""
    content_hash = file_sha256(path)
    with user_transaction(conn, owner_id):
        row = conn.execute(
            "SELECT id, ingest_status, (SELECT count(*) FROM chunks WHERE document_id = d.id) "
            "FROM documents d WHERE owner_id = %s AND content_hash = %s",
            (owner_id, content_hash),
        ).fetchone()
    if row is not None:
        return IngestResult(str(row[0]), row[1], int(row[2]), duplicate=True)

    pages = (parser or PyMuPDFParser(config.parsing)).parse(path)
    title = title or path.stem

    with user_transaction(conn, owner_id):
        doc_id = str(
            conn.execute(
                "INSERT INTO documents (owner_id, lesson_id, doc_type, title, filename, "
                "storage_uri, mime_type, page_count, content_hash, has_text_layer, parser, "
                "parser_version, ingest_status) "
                "VALUES (%s, %s, %s::doc_type_t, %s, %s, %s, 'application/pdf', %s, %s, %s, "
                "%s, %s, 'parsed') RETURNING id",
                (
                    owner_id,
                    lesson_id,
                    doc_type,
                    title,
                    path.name,
                    path.resolve().as_uri(),
                    len(pages),
                    content_hash,
                    any(block.text for page in pages for block in page.blocks),
                    PARSER_NAME,
                    config.chunking.rule_version,
                ),
            ).fetchone()[0]
        )
        chunks = chunk_pages(pages, doc_id, title, config.chunking)
        _insert_chunks(conn, chunks, owner_id, doc_type, lesson_id, subject_id)
    return IngestResult(doc_id, "parsed", len(chunks), duplicate=False)


def _insert_chunks(
    conn: Any,
    chunks: list[Chunk],
    owner_id: str,
    doc_type: str,
    lesson_id: str | None,
    subject_id: str | None,
) -> None:
    # Topics come before their sections, so parent_id always points at an existing row.
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (id, owner_id, document_id, lesson_id, subject_id, doc_type, "
            "kind, parent_id, ordinal, breadcrumb, heading, body, char_count, emphasised, "
            "search_text) VALUES (%s, %s, %s, %s, %s, %s::doc_type_t, %s, %s, %s, %s, %s, %s, "
            "%s, %s, %s)",
            [
                (
                    c.id,
                    owner_id,
                    c.document_id,
                    lesson_id,
                    subject_id,
                    doc_type,
                    c.kind,
                    c.parent_id,
                    c.ordinal,
                    c.breadcrumb,
                    c.heading,
                    c.body,
                    c.char_count,
                    c.emphasised,
                    make_search_text(c.embed_text),
                )
                for c in chunks
            ],
        )
        cur.executemany(
            "INSERT INTO chunk_spans (chunk_id, anchor, page, x0, y0, x1, y1) "
            "VALUES (%s, 'page_bbox', %s, %s, %s, %s, %s)",
            [(c.id, sp.page, *sp.bbox) for c in chunks for sp in c.spans],
        )


def embed_document(
    conn: Any, llm: LLMClient, *, owner_id: str, document_id: str, embed_model: str
) -> int:
    """Embed every section of a document that has no vector yet. Returns how many were embedded.

    Progress is committed per batch, so a retry after a failure only does the remaining work.
    """
    try:
        with user_transaction(conn, owner_id):
            todo = conn.execute(
                "SELECT id, breadcrumb || E'\\n' || body FROM chunks "
                "WHERE document_id = %s AND kind = 'section' AND embedding IS NULL "
                "ORDER BY ordinal, id",
                (document_id,),
            ).fetchall()
        done = 0
        for start in range(0, len(todo), EMBED_BATCH):
            batch = todo[start : start + EMBED_BATCH]
            vectors = llm.embed([text for _, text in batch], input_type="document")
            with user_transaction(conn, owner_id), conn.cursor() as cur:
                cur.executemany(
                    "UPDATE chunks SET embedding = %s::vector, embed_model = %s WHERE id = %s",
                    [
                        (_vector_literal(v), embed_model, chunk_id)
                        for (chunk_id, _), v in zip(batch, vectors, strict=True)
                    ],
                )
            done += len(batch)
        with user_transaction(conn, owner_id):
            conn.execute(
                "UPDATE documents SET ingest_status = 'embedded', ingest_error = NULL "
                "WHERE id = %s",
                (document_id,),
            )
        return done
    except Exception as exc:
        with user_transaction(conn, owner_id):
            conn.execute(
                "UPDATE documents SET ingest_status = 'failed', ingest_error = %s WHERE id = %s",
                (str(exc)[:500], document_id),
            )
        raise
