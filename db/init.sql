-- Palpate schema. Embedding dimension 1024 matches Cohere Embed Multilingual v3.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source_path TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    page         INT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'text',
    text         TEXT NOT NULL,
    emphasised   BOOLEAN NOT NULL DEFAULT FALSE,
    image_path   TEXT,
    rule_version TEXT NOT NULL,
    embedding    vector(1024)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS glossary_terms (
    id      SERIAL PRIMARY KEY,
    doc_id  TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    zh      TEXT NOT NULL,
    en      TEXT NOT NULL,
    aliases TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (doc_id, zh, en)
);
