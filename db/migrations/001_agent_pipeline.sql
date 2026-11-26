-- Upgrade a database created from the original init.sql (documents/chunks/glossary only).
-- Safe to re-run. Fresh installs already have these columns from db/init.sql.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'local';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS lesson_id TEXT NOT NULL DEFAULT '';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_type TEXT NOT NULL DEFAULT 'handout';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS mime_type TEXT NOT NULL DEFAULT 'application/octet-stream';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count INT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parser TEXT NOT NULL DEFAULT 'pymupdf_layout_v1';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parser_version TEXT NOT NULL DEFAULT 'v1';

CREATE UNIQUE INDEX IF NOT EXISTS documents_owner_hash
    ON documents (owner_id, content_hash)
    WHERE content_hash IS NOT NULL;

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'local';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS lesson_id TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_type TEXT NOT NULL DEFAULT 'handout';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS modality TEXT NOT NULL DEFAULT 'text';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS breadcrumb TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_text TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks ALTER COLUMN page DROP NOT NULL;

CREATE INDEX IF NOT EXISTS chunks_scope
    ON chunks (owner_id, lesson_id, kind, doc_type);

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
