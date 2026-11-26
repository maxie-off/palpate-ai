-- Palpate schema. Embedding dimension 1024 matches Cohere Embed Multilingual v3
-- (PALPATE_EMBEDDING_DIM). Changing the model means altering this column and re-embedding.
-- Fresh databases get this file from docker-entrypoint-initdb.d. Existing volumes need
-- db/migrations/001_agent_pipeline.sql.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    owner_id        TEXT NOT NULL DEFAULT 'local',
    lesson_id       TEXT NOT NULL DEFAULT '',
    doc_type        TEXT NOT NULL DEFAULT 'handout',
    title           TEXT NOT NULL,
    source_path     TEXT,
    mime_type       TEXT NOT NULL DEFAULT 'application/octet-stream',
    content_hash    TEXT,
    page_count      INT,
    parser          TEXT NOT NULL DEFAULT 'pymupdf_layout_v1',
    parser_version  TEXT NOT NULL DEFAULT 'v1',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Same bytes uploaded again by the same student are a no-op, not a second copy.
CREATE UNIQUE INDEX IF NOT EXISTS documents_owner_hash
    ON documents (owner_id, content_hash)
    WHERE content_hash IS NOT NULL;

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    owner_id     TEXT NOT NULL DEFAULT 'local',
    lesson_id    TEXT NOT NULL DEFAULT '',
    doc_type     TEXT NOT NULL DEFAULT 'handout',
    page         INT,
    kind         TEXT NOT NULL DEFAULT 'text',
    modality     TEXT NOT NULL DEFAULT 'text',
    text         TEXT NOT NULL,
    breadcrumb   TEXT NOT NULL DEFAULT '',
    search_text  TEXT NOT NULL DEFAULT '',
    emphasised   BOOLEAN NOT NULL DEFAULT FALSE,
    image_path   TEXT,
    rule_version TEXT NOT NULL,
    embedding    vector(1024)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_scope
    ON chunks (owner_id, lesson_id, kind, doc_type);

-- search_text is Latin words plus CJK bigrams, so the simple config can match Chinese
-- without the pg_bigm extension.
CREATE INDEX IF NOT EXISTS chunks_search
    ON chunks USING gin (to_tsvector('simple', search_text));

CREATE TABLE IF NOT EXISTS chunk_spans (
    id        BIGSERIAL PRIMARY KEY,
    chunk_id  TEXT NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    anchor    TEXT NOT NULL CHECK (anchor IN ('page_bbox', 'time_range', 'char_range')),
    page      INT,
    x0        REAL,
    y0        REAL,
    x1        REAL,
    y1        REAL,
    t_start   DOUBLE PRECISION,
    t_end     DOUBLE PRECISION,
    c_start   INT,
    c_end     INT,
    excerpt   TEXT NOT NULL DEFAULT '',
    CHECK (
        (anchor = 'page_bbox' AND page IS NOT NULL AND x0 IS NOT NULL) OR
        (anchor = 'time_range' AND t_start IS NOT NULL) OR
        (anchor = 'char_range' AND c_start IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS spans_lookup ON chunk_spans (chunk_id, anchor);

CREATE TABLE IF NOT EXISTS glossary_terms (
    id      SERIAL PRIMARY KEY,
    doc_id  TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    zh      TEXT NOT NULL,
    en      TEXT NOT NULL,
    aliases TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (doc_id, zh, en)
);
