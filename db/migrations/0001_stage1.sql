-- Palpate schema v0.2, stage 1 subset. See docs/schema.md.
-- Deferred to later migrations: note_templates, note_drafts, watched_folders, annotations.
-- Embedding dimension 1024 matches Cohere Embed Multilingual v3 (open decision 1 in the schema doc).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid() on older builds

CREATE TYPE doc_type_t AS ENUM ('note', 'slides', 'handout', 'video', 'textbook', 'quiz');
CREATE TYPE anchor_t   AS ENUM ('page_bbox', 'time_range', 'char_range');
CREATE TYPE scope_t    AS ENUM ('lesson', 'course', 'cross_subject');
CREATE TYPE verdict_t  AS ENUM ('supported', 'insufficient', 'contradicted');
CREATE TYPE stance_t   AS ENUM ('agrees', 'differs', 'unique');

-- Identity and settings ---------------------------------------------------------------------

CREATE TABLE users (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email         text UNIQUE NOT NULL,
    display_name  text,
    school_domain text NOT NULL,
    locale        text NOT NULL DEFAULT 'zh-TW',
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz
);

CREATE TABLE user_settings (
    user_id               uuid PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
    source_priority       doc_type_t[] NOT NULL
        DEFAULT ARRAY['note', 'slides', 'video', 'handout', 'textbook']::doc_type_t[],
    show_all_sources      boolean NOT NULL DEFAULT false,
    cross_subject_enabled boolean NOT NULL DEFAULT false,
    updated_at            timestamptz NOT NULL DEFAULT now()
);

-- Curriculum scope --------------------------------------------------------------------------

CREATE TABLE subjects (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id   uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    code       text NOT NULL,
    name       text NOT NULL,
    region     text NOT NULL DEFAULT 'TW',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (owner_id, code)
);

CREATE TABLE lessons (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id   uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    subject_id uuid NOT NULL REFERENCES subjects (id) ON DELETE CASCADE,
    lesson_no  int,
    title      text NOT NULL,
    taught_on  date,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (owner_id, subject_id, lesson_no)
);

-- Documents and ingestion -------------------------------------------------------------------

-- Created now (empty) because documents.client_id references it; used by the desktop companion.
CREATE TABLE ingest_clients (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id     uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    kind         text NOT NULL CHECK (kind IN ('web', 'desktop', 'ipad')),
    device_name  text,
    app_version  text,
    last_sync_at timestamptz,
    revoked_at   timestamptz,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE documents (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id         uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    lesson_id        uuid REFERENCES lessons (id) ON DELETE SET NULL,
    doc_type         doc_type_t NOT NULL,
    title            text NOT NULL,
    filename         text NOT NULL,
    storage_uri      text NOT NULL,
    mime_type        text NOT NULL,
    page_count       int,
    duration_s       int,
    content_hash     char(64) NOT NULL,
    source           text NOT NULL DEFAULT 'upload'
        CHECK (source IN ('upload', 'folder_sync', 'school_connector')),
    client_id        uuid REFERENCES ingest_clients (id) ON DELETE SET NULL,
    has_text_layer   boolean,
    transcript_model text,
    parser           text NOT NULL,
    parser_version   text NOT NULL,
    ingest_status    text NOT NULL DEFAULT 'pending'
        CHECK (ingest_status IN ('pending', 'parsed', 'embedded', 'failed')),
    ingest_error     text,
    ingested_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (owner_id, content_hash)
);

-- Chunks, anchors and concepts --------------------------------------------------------------

CREATE TABLE chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id    uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    document_id uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    lesson_id   uuid REFERENCES lessons (id) ON DELETE SET NULL,
    subject_id  uuid REFERENCES subjects (id) ON DELETE SET NULL,
    doc_type    doc_type_t NOT NULL,
    kind        text NOT NULL CHECK (kind IN ('topic', 'section', 'quiz_item')),
    parent_id   uuid REFERENCES chunks (id) ON DELETE CASCADE,
    ordinal     int NOT NULL,
    breadcrumb  text NOT NULL,
    heading     text,
    body        text NOT NULL,
    char_count  int NOT NULL,
    emphasised  boolean NOT NULL DEFAULT false,  -- teacher highlight signal (design outline)
    embedding   vector(1024),
    embed_model text,
    search_text text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunk_spans (
    id       bigserial PRIMARY KEY,
    chunk_id uuid NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    anchor   anchor_t NOT NULL,
    page     int,
    x0 real, y0 real, x1 real, y1 real,
    t_start  numeric(9, 3), t_end numeric(9, 3),
    c_start  int, c_end int,
    CHECK (
        (anchor = 'page_bbox' AND page IS NOT NULL AND x0 IS NOT NULL) OR
        (anchor = 'time_range' AND t_start IS NOT NULL) OR
        (anchor = 'char_range' AND c_start IS NOT NULL)
    )
);

-- Created now because citations.via_concept_id references concepts; filled in stage 2+.
CREATE TABLE concepts (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id      uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    canonical     text NOT NULL,
    surface_forms text[] NOT NULL DEFAULT '{}',
    kind          text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (owner_id, canonical)
);

CREATE TABLE chunk_concepts (
    chunk_id    uuid NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    concept_id  uuid NOT NULL REFERENCES concepts (id) ON DELETE CASCADE,
    occurrences int NOT NULL DEFAULT 1,
    PRIMARY KEY (chunk_id, concept_id)
);

-- Quiz items --------------------------------------------------------------------------------

CREATE TABLE quiz_items (
    chunk_id     uuid PRIMARY KEY REFERENCES chunks (id) ON DELETE CASCADE,
    document_id  uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    item_no      int NOT NULL,
    stem         text NOT NULL,
    options      jsonb NOT NULL,
    answer_label text,   -- NULL = sheet gives no key (distinct from empty)
    rationale    text,
    UNIQUE (document_id, item_no)
);

-- Queries, citations and source divergence --------------------------------------------------

CREATE TABLE queries (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id      uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    mode          text NOT NULL CHECK (mode IN ('qa', 'quiz_item', 'quiz_option', 'note_section')),
    question      text NOT NULL,
    lesson_id     uuid REFERENCES lessons (id) ON DELETE SET NULL,
    subject_id    uuid REFERENCES subjects (id) ON DELETE SET NULL,
    quiz_chunk_id uuid REFERENCES chunks (id) ON DELETE CASCADE,
    option_label  text,
    note_draft_id uuid,  -- FK added with the note_drafts table
    scope_used    scope_t NOT NULL,
    verdict       verdict_t NOT NULL,
    answer_md     text,
    answer_model  text,
    latency_ms    int,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE citations (
    id              bigserial PRIMARY KEY,
    owner_id        uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    query_id        uuid NOT NULL REFERENCES queries (id) ON DELETE CASCADE,
    source_chunk_id uuid REFERENCES chunks (id) ON DELETE SET NULL,
    rank            int NOT NULL,
    dense_score     real,
    lexical_score   real,
    fused_score     real,
    margin          real,
    via_concept_id  uuid REFERENCES concepts (id) ON DELETE SET NULL,
    verdict         verdict_t NOT NULL,
    verdict_model   text,
    verdict_reason  text,
    quote           text,  -- verified quote from the chunk (see ADR 0002)
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE answer_sources (
    id           bigserial PRIMARY KEY,
    owner_id     uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    query_id     uuid NOT NULL REFERENCES queries (id) ON DELETE CASCADE,
    document_id  uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    doc_type     doc_type_t NOT NULL,
    claim_group  int NOT NULL,
    stance       stance_t NOT NULL,
    summary_md   text NOT NULL,
    display_rank int NOT NULL,
    UNIQUE (query_id, document_id)
);

-- Indexes -----------------------------------------------------------------------------------

CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_scope    ON chunks (owner_id, lesson_id, kind, doc_type);
CREATE INDEX chunks_subject  ON chunks (owner_id, subject_id, kind);
CREATE INDEX chunks_parent   ON chunks (parent_id);
CREATE INDEX chunks_search   ON chunks USING gin (to_tsvector('simple', search_text));
CREATE INDEX spans_lookup    ON chunk_spans (chunk_id, anchor);
CREATE INDEX cc_by_concept   ON chunk_concepts (concept_id);
CREATE INDEX concepts_forms  ON concepts USING gin (surface_forms);
CREATE INDEX citations_query ON citations (query_id, rank);
CREATE INDEX asrc_query      ON answer_sources (query_id, display_rank);
CREATE INDEX queries_recent  ON queries (owner_id, created_at DESC);

-- Row-level isolation -----------------------------------------------------------------------
-- The API connects as palpate_app (not a superuser, so RLS applies) and sets
-- app.current_user_id once per transaction from the verified token. Unset => no rows.

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'subjects', 'lessons', 'ingest_clients', 'documents', 'chunks', 'concepts',
        'queries', 'citations', 'answer_sources'
    ] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format(
            'CREATE POLICY %I ON %I USING (owner_id = nullif(current_setting(''app.current_user_id'', true), '''')::uuid)',
            t || '_owner', t
        );
    END LOOP;
END $$;

-- Dev-only role; replace the password outside local development.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'palpate_app') THEN
        CREATE ROLE palpate_app LOGIN PASSWORD 'palpate';
    END IF;
END $$;
GRANT USAGE ON SCHEMA public TO palpate_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO palpate_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO palpate_app;
