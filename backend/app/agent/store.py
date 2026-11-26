"""Knowledge store: hybrid retrieval over the caller's materials.

``InMemoryKnowledgeStore`` is what tests and the offline sample use. ``PgVectorKnowledgeStore``
is the same contract on the repo's Postgres + pgvector schema. Both filter by ``owner_id``
on every read. Row-level security from the schema draft is not enabled yet; the filter is
still applied in the query so a forgotten caller-side check is not the only line of defence
inside this package.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.agent.lexical import cosine, lexical_overlap, render_search_text, rrf_component
from app.agent.types import ChunkSpan, IngestedDocument, SearchHit, StoredChunk


@dataclass(frozen=True)
class SearchParams:
    owner_id: str
    lesson_id: str = ""
    top_k: int = 8
    rrf_k: int = 60
    candidate_k: int = 50
    highlight_boost: float = 1.2
    min_dense_score: float = 0.0


class KnowledgeStore(Protocol):
    def find_by_hash(self, owner_id: str, content_hash: str) -> IngestedDocument | None: ...

    def upsert(
        self, document: IngestedDocument, chunks: Sequence[StoredChunk]
    ) -> IngestedDocument: ...

    def list_chunks(
        self,
        *,
        owner_id: str,
        lesson_id: str = "",
        doc_ids: Sequence[str] | None = None,
    ) -> list[StoredChunk]: ...

    def search(
        self, query: str, vector: Sequence[float], params: SearchParams
    ) -> list[SearchHit]: ...


def embed_input(chunk: StoredChunk) -> str:
    """Breadcrumb plus body, so a section that never repeats its heading is still findable."""
    if chunk.breadcrumb:
        return f"{chunk.breadcrumb}\n{chunk.text}"
    return chunk.text or " "


def with_search_text(chunk: StoredChunk) -> StoredChunk:
    if chunk.search_text:
        return chunk
    return chunk.model_copy(update={"search_text": render_search_text(embed_input(chunk))})


def _public_chunk(chunk: StoredChunk) -> StoredChunk:
    if not chunk.embedding:
        return chunk
    return chunk.model_copy(update={"embedding": []})


def _passes_dense_gate(similarity: float | None, overlap: int, min_dense_score: float) -> bool:
    if min_dense_score <= 0:
        return True
    if overlap > 0:
        return True
    if similarity is None:
        return True
    return similarity >= min_dense_score


def fuse_hits(
    chunks: Sequence[StoredChunk],
    query: str,
    vector: Sequence[float],
    params: SearchParams,
) -> list[SearchHit]:
    """Dense cosine and lexical overlap, fused by reciprocal rank, then highlight-boosted."""
    eligible = [chunk for chunk in chunks if chunk.owner_id == params.owner_id]
    if params.lesson_id:
        eligible = [chunk for chunk in eligible if chunk.lesson_id == params.lesson_id]
    if not eligible:
        return []

    dense_pool = [chunk for chunk in eligible if chunk.embedding]
    dense_ordered = sorted(
        dense_pool,
        key=lambda chunk: (-cosine(vector, chunk.embedding), chunk.id),
    )[: params.candidate_k]
    dense_rank = {chunk.id: rank for rank, chunk in enumerate(dense_ordered, start=1)}
    dense_sim = {chunk.id: cosine(vector, chunk.embedding) for chunk in dense_ordered}

    lexical_scored = [
        (chunk, lexical_overlap(query, chunk.search_text))
        for chunk in eligible
        if lexical_overlap(query, chunk.search_text) > 0
    ]
    lexical_scored.sort(key=lambda item: (-item[1], item[0].id))
    lexical_scored = lexical_scored[: params.candidate_k]
    lexical_rank = {
        chunk.id: rank for rank, (chunk, _overlap) in enumerate(lexical_scored, start=1)
    }
    lexical_count = {chunk.id: overlap for chunk, overlap in lexical_scored}

    seen: dict[str, StoredChunk] = {chunk.id: chunk for chunk in dense_ordered}
    for chunk, _overlap in lexical_scored:
        seen.setdefault(chunk.id, chunk)

    hits: list[SearchHit] = []
    for chunk in seen.values():
        overlap = lexical_count.get(chunk.id, 0)
        similarity = dense_sim.get(chunk.id)
        if not _passes_dense_gate(similarity, overlap, params.min_dense_score):
            continue
        fused = rrf_component(dense_rank.get(chunk.id), params.rrf_k) + rrf_component(
            lexical_rank.get(chunk.id), params.rrf_k
        )
        if chunk.emphasised:
            fused *= params.highlight_boost
        hits.append(
            SearchHit(
                chunk=_public_chunk(chunk),
                fused_score=fused,
                dense_similarity=similarity,
                lexical_overlap=overlap,
            )
        )
    hits.sort(key=lambda hit: (-hit.fused_score, hit.chunk.id))
    return hits[: params.top_k]


class InMemoryKnowledgeStore:
    def __init__(self) -> None:
        self._documents: dict[str, IngestedDocument] = {}
        self._chunks: dict[str, StoredChunk] = {}
        self._by_hash: dict[tuple[str, str], str] = {}

    def find_by_hash(self, owner_id: str, content_hash: str) -> IngestedDocument | None:
        doc_id = self._by_hash.get((owner_id, content_hash))
        if doc_id is None:
            return None
        document = self._documents.get(doc_id)
        if document is None:
            return None
        return document.model_copy(update={"cached": True})

    def upsert(self, document: IngestedDocument, chunks: Sequence[StoredChunk]) -> IngestedDocument:
        prepared = [with_search_text(chunk) for chunk in chunks]
        figure_count = sum(1 for chunk in prepared if chunk.kind == "image_caption")
        stored = document.model_copy(
            update={"chunk_count": len(prepared), "figure_count": figure_count, "cached": False}
        )
        current = self._documents.get(stored.doc_id)
        if current is not None and current.owner_id != stored.owner_id:
            raise ValueError(f"document {stored.doc_id} belongs to another owner")
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if chunk.doc_id != stored.doc_id
        }
        for chunk in prepared:
            self._chunks[chunk.id] = chunk
        self._by_hash = {
            key: doc_id for key, doc_id in self._by_hash.items() if doc_id != stored.doc_id
        }
        if stored.content_hash:
            self._by_hash[(stored.owner_id, stored.content_hash)] = stored.doc_id
        self._documents[stored.doc_id] = stored
        return stored

    def list_chunks(
        self,
        *,
        owner_id: str,
        lesson_id: str = "",
        doc_ids: Sequence[str] | None = None,
    ) -> list[StoredChunk]:
        wanted = set(doc_ids) if doc_ids is not None else None
        rows = [
            chunk
            for chunk in self._chunks.values()
            if chunk.owner_id == owner_id
            and (not lesson_id or chunk.lesson_id == lesson_id)
            and (wanted is None or chunk.doc_id in wanted)
        ]
        rows.sort(key=lambda chunk: (chunk.page or 0, chunk.id))
        return [_public_chunk(chunk) for chunk in rows]

    def search(self, query: str, vector: Sequence[float], params: SearchParams) -> list[SearchHit]:
        return fuse_hits(list(self._chunks.values()), query, vector, params)


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


class PgVectorKnowledgeStore:
    """Postgres implementation. The connection is expected to be a psycopg connection.

    The store commits on write and rolls back if a write fails. Search reads the
    columns created in ``db/init.sql`` / ``db/migrations/001_agent_pipeline.sql``.
    """

    def __init__(self, conn: Any, *, embedding_dim: int = 1024) -> None:
        self._conn = conn
        self._dim = embedding_dim

    def find_by_hash(self, owner_id: str, content_hash: str) -> IngestedDocument | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, owner_id, lesson_id, doc_type, title, source_path, mime_type,
                       content_hash, page_count, parser, parser_version
                FROM documents
                WHERE owner_id = %s AND content_hash = %s
                """,
                (owner_id, content_hash),
            )
            row = cur.fetchone()
            if row is None:
                return None
            doc_id = row[0]
            cur.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,))
            chunk_count = int(cur.fetchone()[0])
            cur.execute(
                "SELECT count(*) FROM chunks WHERE doc_id = %s AND kind = 'image_caption'",
                (doc_id,),
            )
            figure_count = int(cur.fetchone()[0])
        return IngestedDocument(
            doc_id=row[0],
            owner_id=row[1],
            lesson_id=row[2] or "",
            doc_type=row[3],
            title=row[4],
            source_path=row[5] or "",
            mime_type=row[6] or "application/octet-stream",
            content_hash=row[7] or "",
            page_count=int(row[8] or 0),
            chunk_count=chunk_count,
            figure_count=figure_count,
            cached=True,
            parser=row[9] or "",
            parser_version=row[10] or "",
        )

    def upsert(self, document: IngestedDocument, chunks: Sequence[StoredChunk]) -> IngestedDocument:
        prepared = [with_search_text(chunk) for chunk in chunks]
        for chunk in prepared:
            if chunk.embedding and len(chunk.embedding) != self._dim:
                raise ValueError(
                    f"embedding dim {len(chunk.embedding)} does not match "
                    f"PALPATE_EMBEDDING_DIM={self._dim}"
                )
        figure_count = sum(1 for chunk in prepared if chunk.kind == "image_caption")
        stored = document.model_copy(
            update={"chunk_count": len(prepared), "figure_count": figure_count, "cached": False}
        )
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT owner_id FROM documents WHERE id = %s", (stored.doc_id,))
                existing = cur.fetchone()
                if existing is not None and existing[0] != stored.owner_id:
                    raise ValueError(f"document {stored.doc_id} belongs to another owner")
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (stored.doc_id,))
                cur.execute(
                    """
                    INSERT INTO documents (
                        id, owner_id, lesson_id, doc_type, title, source_path, mime_type,
                        content_hash, page_count, parser, parser_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        owner_id = EXCLUDED.owner_id,
                        lesson_id = EXCLUDED.lesson_id,
                        doc_type = EXCLUDED.doc_type,
                        title = EXCLUDED.title,
                        source_path = EXCLUDED.source_path,
                        mime_type = EXCLUDED.mime_type,
                        content_hash = EXCLUDED.content_hash,
                        page_count = EXCLUDED.page_count,
                        parser = EXCLUDED.parser,
                        parser_version = EXCLUDED.parser_version
                    """,
                    (
                        stored.doc_id,
                        stored.owner_id,
                        stored.lesson_id,
                        stored.doc_type,
                        stored.title,
                        stored.source_path,
                        stored.mime_type,
                        stored.content_hash,
                        stored.page_count,
                        stored.parser,
                        stored.parser_version,
                    ),
                )
                for chunk in prepared:
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            id, doc_id, owner_id, lesson_id, doc_type, page, kind, modality,
                            text, breadcrumb, search_text, emphasised, image_path,
                            rule_version, embedding
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s::vector
                        )
                        """,
                        (
                            chunk.id,
                            chunk.doc_id,
                            chunk.owner_id,
                            chunk.lesson_id,
                            chunk.doc_type,
                            chunk.page,
                            chunk.kind,
                            chunk.modality,
                            chunk.text,
                            chunk.breadcrumb,
                            chunk.search_text,
                            chunk.emphasised,
                            chunk.image_path,
                            chunk.rule_version,
                            vector_literal(chunk.embedding) if chunk.embedding else None,
                        ),
                    )
                    for span in chunk.spans:
                        cur.execute(
                            """
                            INSERT INTO chunk_spans (
                                chunk_id, anchor, page, x0, y0, x1, y1,
                                t_start, t_end, c_start, c_end, excerpt
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                chunk.id,
                                span.anchor,
                                span.page,
                                span.x0,
                                span.y0,
                                span.x1,
                                span.y1,
                                span.t_start,
                                span.t_end,
                                span.c_start,
                                span.c_end,
                                span.excerpt,
                            ),
                        )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return stored

    def list_chunks(
        self,
        *,
        owner_id: str,
        lesson_id: str = "",
        doc_ids: Sequence[str] | None = None,
    ) -> list[StoredChunk]:
        sql = "SELECT id FROM chunks WHERE owner_id = %s"
        args: list[Any] = [owner_id]
        if lesson_id:
            sql += " AND lesson_id = %s"
            args.append(lesson_id)
        if doc_ids is not None:
            sql += " AND doc_id = ANY(%s)"
            args.append(list(doc_ids))
        sql += " ORDER BY page NULLS FIRST, id"
        with self._conn.cursor() as cur:
            cur.execute(sql, args)
            ids = [row[0] for row in cur.fetchall()]
            return self._load_chunks(cur, ids)

    def search(self, query: str, vector: Sequence[float], params: SearchParams) -> list[SearchHit]:
        if len(vector) != self._dim:
            raise ValueError(
                "query embedding dim "
                f"{len(vector)} does not match PALPATE_EMBEDDING_DIM={self._dim}"
            )
        literal = vector_literal(vector)
        lexical_query = render_search_text(query)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                WITH dense AS (
                    SELECT id,
                           row_number() OVER (ORDER BY embedding <=> %s::vector) AS r,
                           1 - (embedding <=> %s::vector) AS sim
                    FROM chunks
                    WHERE owner_id = %s
                      AND embedding IS NOT NULL
                      AND (%s = '' OR lesson_id = %s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                ),
                lexical AS (
                    SELECT id,
                           row_number() OVER (
                               ORDER BY ts_rank(
                                   to_tsvector('simple', search_text),
                                   plainto_tsquery('simple', %s)
                               ) DESC
                           ) AS r
                    FROM chunks
                    WHERE owner_id = %s
                      AND (%s = '' OR lesson_id = %s)
                      AND %s <> ''
                      AND to_tsvector('simple', search_text)
                          @@ plainto_tsquery('simple', %s)
                    LIMIT %s
                )
                SELECT c.id,
                       1.0 / (%s + COALESCE(d.r, 1000))
                           + 1.0 / (%s + COALESCE(l.r, 1000)) AS fused,
                       d.sim
                FROM chunks c
                LEFT JOIN dense d ON d.id = c.id
                LEFT JOIN lexical l ON l.id = c.id
                WHERE d.id IS NOT NULL OR l.id IS NOT NULL
                ORDER BY fused DESC
                LIMIT %s
                """,
                (
                    literal,
                    literal,
                    params.owner_id,
                    params.lesson_id,
                    params.lesson_id,
                    literal,
                    params.candidate_k,
                    lexical_query,
                    params.owner_id,
                    params.lesson_id,
                    params.lesson_id,
                    lexical_query,
                    lexical_query,
                    params.candidate_k,
                    params.rrf_k,
                    params.rrf_k,
                    params.candidate_k,
                ),
            )
            ranked = [(row[0], float(row[1]), row[2]) for row in cur.fetchall()]
            ids = [row[0] for row in ranked]
            loaded = {chunk.id: chunk for chunk in self._load_chunks(cur, ids)}
        hits: list[SearchHit] = []
        for chunk_id, fused, similarity in ranked:
            chunk = loaded.get(chunk_id)
            if chunk is None:
                continue
            overlap = lexical_overlap(query, chunk.search_text)
            dense_similarity = float(similarity) if similarity is not None else None
            if not _passes_dense_gate(dense_similarity, overlap, params.min_dense_score):
                continue
            if chunk.emphasised:
                fused *= params.highlight_boost
            hits.append(
                SearchHit(
                    chunk=chunk,
                    fused_score=fused,
                    dense_similarity=dense_similarity,
                    lexical_overlap=overlap,
                )
            )
        hits.sort(key=lambda hit: (-hit.fused_score, hit.chunk.id))
        return hits[: params.top_k]

    def _load_chunks(self, cur: Any, ids: Sequence[str]) -> list[StoredChunk]:
        if not ids:
            return []
        cur.execute(
            """
            SELECT id, doc_id, owner_id, lesson_id, doc_type, page, kind, modality,
                   text, breadcrumb, search_text, emphasised, image_path, rule_version
            FROM chunks
            WHERE id = ANY(%s)
            """,
            (list(ids),),
        )
        rows = cur.fetchall()
        cur.execute(
            """
            SELECT chunk_id, anchor, page, x0, y0, x1, y1, t_start, t_end, c_start, c_end, excerpt
            FROM chunk_spans
            WHERE chunk_id = ANY(%s)
            ORDER BY id
            """,
            (list(ids),),
        )
        spans_by_chunk: dict[str, list[ChunkSpan]] = {}
        for span_row in cur.fetchall():
            spans_by_chunk.setdefault(span_row[0], []).append(
                ChunkSpan(
                    anchor=span_row[1],
                    page=span_row[2],
                    x0=span_row[3],
                    y0=span_row[4],
                    x1=span_row[5],
                    y1=span_row[6],
                    t_start=span_row[7],
                    t_end=span_row[8],
                    c_start=span_row[9],
                    c_end=span_row[10],
                    excerpt=span_row[11] or "",
                )
            )
        chunks: list[StoredChunk] = []
        for row in rows:
            chunks.append(
                StoredChunk(
                    id=row[0],
                    doc_id=row[1],
                    owner_id=row[2],
                    lesson_id=row[3] or "",
                    doc_type=row[4],
                    page=row[5],
                    kind=row[6],
                    modality=row[7],
                    text=row[8],
                    breadcrumb=row[9] or "",
                    search_text=row[10] or "",
                    emphasised=bool(row[11]),
                    image_path=row[12],
                    rule_version=row[13],
                    spans=spans_by_chunk.get(row[0], []),
                )
            )
        return chunks
