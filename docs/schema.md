# Palpate — Data Schema (v0.2 draft)

Target store: **PostgreSQL 16 + pgvector** (AWS RDS).

Covers the three product modes agreed so far: **Q&A with multi-source citation**, **cross-subject retrieval**, and **共筆 note drafting**. Columns for later stages are present where adding them later would mean a full re-ingest.

Not every table here is needed in week one — see [Stage 1 minimum](#stage-1-minimum) at the end.

---

## What changed from v0.1

1. `doc_type` became a real enum and a richer taxonomy, because source *kind* now drives display order.
2. Source priority moved to a per-user setting, not a column on the document. Students choose their own order; the default is 共筆 first.
3. `chunk_spans` became polymorphic. A PDF anchor, a video timestamp and a `.docx` character range are three different shapes.
4. New `concepts` / `chunk_concepts` tables. These are the gate that lets cross-subject retrieval be precise instead of merely loose.
5. `citations` now hangs off a `queries` row rather than off a quiz item, because questions also arrive as free text and as note-section prompts.
6. New `answer_sources` table. This is where "the PPT and the 共筆 say different things" is actually represented.
7. New `note_templates` / `note_drafts` for 共筆 mode.
8. New `ingest_clients` / `watched_folders` for the desktop companion.

---

## Design decisions encoded here

**Two chunk levels, one table.** Retrieval matches small `section` chunks for precision; generation receives the whole `topic` parent for context. `parent_id` implements this *small-to-big retrieval*.

**Markdown is the body, not the only artifact.** Converting a document to markdown discards position. Highlighting needs position. So the normalizer emits both: clean text into `chunks.body`, and a position anchor per source line into `chunk_spans`. Re-deriving anchors later would mean re-parsing every document every student has ever uploaded.

**Scope is data, not a heuristic.** Our retrieval test linked an Epithelial(I) worksheet to cyst-chapter notes with plausible-looking scores, because a retriever always returns *something*. `lessons` exists so the default scope is a `WHERE` clause.

**Cross-subject needs a passport, not a lower threshold.** Loosening the similarity threshold cannot distinguish real topic overlap from shared generic vocabulary — unrelated chunks scored 5.3 on words like 細胞 and 組織 while a true hit scored 13.2. Instead, a chunk from another subject is only eligible when it shares a *named concept* with the question or with the in-scope results. `chunk_concepts` is that passport.

**The server never reads a student's disk.** `watched_folders.local_path` is stored for display only. The desktop client scans locally and pushes through the same ingest API the web uploader uses. One ingest contract, two clients.

---

## Enums

```sql
CREATE TYPE doc_type_t AS ENUM (
  'note',        -- 共筆, student collaborative notes  (default top priority)
  'slides',      -- teacher's PPT
  'handout',     -- teacher's PDF/doc handout
  'video',       -- class recording, stored as a transcript
  'textbook',    -- Regezi, Neville, etc.
  'quiz'         -- worksheet / 課後測驗
);

CREATE TYPE anchor_t AS ENUM ('page_bbox', 'time_range', 'char_range');

CREATE TYPE scope_t AS ENUM ('lesson', 'course', 'cross_subject');

CREATE TYPE verdict_t AS ENUM ('supported', 'insufficient', 'contradicted');

CREATE TYPE stance_t AS ENUM ('agrees', 'differs', 'unique');
```

`doc_type_t` is an enum rather than free text because the user's display-order preference is an *array of these values*. An enum array is checked by Postgres; a `text[]` would silently accept a typo and quietly break a student's ordering.

---

## Identity and settings

```sql
CREATE TABLE users (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email         text UNIQUE NOT NULL,
  display_name  text,
  school_domain text NOT NULL,                 -- 'tmu.edu.tw'
  locale        text NOT NULL DEFAULT 'zh-TW',
  created_at    timestamptz NOT NULL DEFAULT now(),
  last_seen_at  timestamptz
);
```

No password column, by design. Identity comes from Google via OIDC, with Cognito as the broker. We never store or verify credentials, so a breach here leaks no password. `school_domain` is persisted after a pre-signup Lambda verifies the email domain — Google's `hd` claim is a hint, not a guarantee, so the check is server-side and the result is recorded.

```sql
CREATE TABLE user_settings (
  user_id         uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  source_priority doc_type_t[] NOT NULL
                  DEFAULT ARRAY['note','slides','video','handout','textbook']::doc_type_t[],
  show_all_sources     boolean NOT NULL DEFAULT false,
  cross_subject_enabled boolean NOT NULL DEFAULT false,
  updated_at      timestamptz NOT NULL DEFAULT now()
);
```

`source_priority` is the ordered list a student sets for themselves. Default leads with `note` (共筆), as agreed.

`show_all_sources` is the sticky state of the "show every source" toggle. The toggle does not change *retrieval* — all sources in scope are always retrieved and all of them are stored in `answer_sources`. It only changes *presentation*: off, the answer shows the highest-priority source and a count of the others; on, it shows them side by side. Keeping retrieval constant means toggling the view costs nothing and never re-queries the model.

`cross_subject_enabled` stays off until the abstention gate is trusted. It is a per-user switch so it can be enabled for a few volunteers first.

---

## Curriculum scope

```sql
CREATE TABLE subjects (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  code        text NOT NULL,              -- 'OMP', 'OMB'
  name        text NOT NULL,              -- '口腔病理學', '口腔微生物學'
  region      text NOT NULL DEFAULT 'TW',
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (owner_id, code)
);

CREATE TABLE lessons (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  subject_id  uuid NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
  lesson_no   int,                         -- 3
  title       text NOT NULL,               -- 'Epithelial (I)'
  taught_on   date,
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (owner_id, subject_id, lesson_no)
);
```

`subjects` was split out of v0.1's `lessons` because cross-subject retrieval needs *subject* as its own level. The escalation ladder is lesson → subject → other subjects, and that middle rung needs something to be a rung.

`region` is here for the long-term goal of other countries' curricula. It costs nothing now and means the filter dimension already exists.

---

## Documents and ingestion

```sql
CREATE TABLE documents (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id       uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  lesson_id      uuid REFERENCES lessons(id) ON DELETE SET NULL,
  doc_type       doc_type_t NOT NULL,
  title          text NOT NULL,
  filename       text NOT NULL,
  storage_uri    text NOT NULL,            -- s3://palpate-uploads/<owner>/<hash>
  mime_type      text NOT NULL,
  page_count     int,
  duration_s     int,                      -- video only
  content_hash   char(64) NOT NULL,        -- sha256 of the raw bytes
  source         text NOT NULL DEFAULT 'upload'
                 CHECK (source IN ('upload','folder_sync','school_connector')),
  client_id      uuid REFERENCES ingest_clients(id) ON DELETE SET NULL,
  has_text_layer boolean,                  -- false -> OCR was required
  transcript_model text,                   -- video only
  parser         text NOT NULL,            -- 'lecture_font_v1', 'quiz_format_v1', …
  parser_version text NOT NULL,
  ingest_status  text NOT NULL DEFAULT 'pending'
                 CHECK (ingest_status IN ('pending','parsed','embedded','failed')),
  ingest_error   text,
  ingested_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (owner_id, content_hash)
);
```

`UNIQUE (owner_id, content_hash)` is the dedup rule. Classmates re-upload the same shared 共筆 constantly; this makes a re-upload a no-op instead of a second copy that doubles cost and splits retrieval across duplicates. Scoped per user, because each knowledge base is private.

`parser` plus `parser_version` records *how* a document was parsed. When a parser bug is fixed — the duplicate-question-number bug, for instance — this is how we find which documents need re-ingesting, instead of reprocessing everything.

`has_text_layer` distinguishes real extracted text from OCR output. OCR is less trustworthy and the answer layer should be able to know that. Same reasoning for `transcript_model`: a transcript is a derived, error-prone artifact, and a citation into one deserves to be marked as such.

```sql
CREATE TABLE ingest_clients (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id     uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind         text NOT NULL CHECK (kind IN ('web','desktop','ipad')),
  device_name  text,
  app_version  text,
  last_sync_at timestamptz,
  revoked_at   timestamptz,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE watched_folders (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id     uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  client_id    uuid NOT NULL REFERENCES ingest_clients(id) ON DELETE CASCADE,
  local_path   text NOT NULL,              -- display only; the server never reads it
  default_lesson_id uuid REFERENCES lessons(id) ON DELETE SET NULL,
  last_scan_at timestamptz,
  file_count   int,
  UNIQUE (client_id, local_path)
);
```

`revoked_at` on the client matters more than it looks: a student who loses a laptop needs a way to cut that device off without changing their Google password.

`default_lesson_id` lets a folder imply a lesson, so files dropped into `.../Lesson 3/` land in the right scope without the student tagging each one.

---

## Chunks, anchors and concepts

```sql
CREATE TABLE chunks (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  lesson_id   uuid REFERENCES lessons(id) ON DELETE SET NULL,
  subject_id  uuid REFERENCES subjects(id) ON DELETE SET NULL,
  doc_type    doc_type_t NOT NULL,          -- denormalised for filter + display order
  kind        text NOT NULL CHECK (kind IN ('topic','section','quiz_item')),
  parent_id   uuid REFERENCES chunks(id) ON DELETE CASCADE,
  ordinal     int NOT NULL,
  breadcrumb  text NOT NULL,
  heading     text,
  body        text NOT NULL,                -- markdown
  char_count  int NOT NULL,
  embedding   vector(1024),                 -- NULL until the embed step runs
  embed_model text,
  search_text text NOT NULL,                -- CJK bigrams + latin words
  created_at  timestamptz NOT NULL DEFAULT now()
);
```

What gets embedded is `breadcrumb || E'\n' || body`, never `body` alone. A section reading *"50% of newborns show primary milia…"* contains no occurrence of "Milia", so embedded in isolation it is unreachable by the obvious query. Prepending the breadcrumb is *contextual retrieval*; the usual implementation pays an LLM call per chunk to synthesise that context, but our notes carry an explicit heading hierarchy, so we assemble it by string concatenation at zero cost.

`doc_type` and `subject_id` are denormalised onto the chunk deliberately. Every retrieval filters on them, and a three-table join on the hot path to recover a value that never changes is a cost with no benefit.

`search_text` exists because Postgres full-text search has no Chinese tokenizer by default — `to_tsvector` treats a Chinese sentence as a single token. Two workable options: the `pg_bigm` extension (available on RDS), or precompute bigrams into `search_text` at ingest and index with the `simple` config, which needs no extension and is exactly what the retrieval prototype already does.

```sql
CREATE TABLE chunk_spans (
  id        bigserial PRIMARY KEY,
  chunk_id  uuid NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  anchor    anchor_t NOT NULL,
  page      int,                            -- page_bbox
  x0 real, y0 real, x1 real, y1 real,       -- page_bbox, PDF user space
  t_start numeric(9,3), t_end numeric(9,3), -- time_range, seconds
  c_start int, c_end int,                   -- char_range
  CHECK (
    (anchor = 'page_bbox'  AND page IS NOT NULL AND x0 IS NOT NULL) OR
    (anchor = 'time_range' AND t_start IS NOT NULL) OR
    (anchor = 'char_range' AND c_start IS NOT NULL)
  )
);
```

One row per source line. Three anchor shapes, because the sources genuinely differ:

- PDF and slides give `page_bbox`, which draws directly on the original file.
- Video gives `time_range`, so a citation becomes "jump to 14:32" — the same idea as highlighting, a different interaction.
- `.docx` has no page geometry at all, so it gives `char_range`. Highlighting one means either rendering to PDF at ingest or highlighting in a text view. **Open decision.**

Coordinates are stored in PDF user space, not viewport pixels, so they stay valid at any zoom and in any viewer.

```sql
CREATE TABLE concepts (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id      uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  canonical     text NOT NULL,              -- 'human papillomavirus'
  surface_forms text[] NOT NULL DEFAULT '{}',  -- ['HPV','人類乳突病毒']
  kind          text,                       -- 'pathogen','gene','procedure','structure','drug'
  created_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (owner_id, canonical)
);

CREATE TABLE chunk_concepts (
  chunk_id    uuid NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  concept_id  uuid NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
  occurrences int NOT NULL DEFAULT 1,
  PRIMARY KEY (chunk_id, concept_id)
);
```

This pair is the cross-subject passport. Named entities are extracted at ingest — `HPV`, `candida`, `keratin`, `USP6`, `Sistrunk procedure` — and a chunk from another subject becomes eligible only when it shares a `concept_id` with the question or with the in-scope results.

The reason this beats a looser similarity threshold: 口腔病理學 and 口腔微生物學 genuinely overlap on *specific things*, not on abstract semantic closeness. Requiring a shared named concept is a precise, cheap, inspectable rule. It is also debuggable in a way a threshold is not — you can look at which concept let a result through.

`surface_forms` carries the aliases, which is what makes a Chinese query match an English term in the notes.

---

## Quiz items

```sql
CREATE TABLE quiz_items (
  chunk_id     uuid PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  item_no      int NOT NULL,
  stem         text NOT NULL,
  options      jsonb NOT NULL,              -- [{"label":"A","text":"…"}, …]
  answer_label text,                         -- NULL when the sheet gives no key
  rationale    text,
  UNIQUE (document_id, item_no)
);
```

`UNIQUE (document_id, item_no)` is deliberate: the current quiz parser emits a duplicate item number (two items numbered 1), and without this constraint the second row would silently shadow the first. The constraint turns quiet data corruption into a loud ingest failure.

`answer_label` is nullable because it must be. Of the 20 items in the sample worksheet, 3 carry an answer key. The agent behaves differently in each case — with a key it explains the given answer, without one it reasons from retrieved evidence and says the sheet provides no official answer — so the schema has to distinguish "no key" from "not yet parsed". A `NOT NULL DEFAULT ''` would destroy that distinction.

`options` is `jsonb` because options are always read as a set with their item. Per-option *citations* do need their own rows, and those live below.

---

## Queries, citations and source divergence

```sql
CREATE TABLE queries (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id      uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  mode          text NOT NULL CHECK (mode IN ('qa','quiz_item','quiz_option','note_section')),
  question      text NOT NULL,
  lesson_id     uuid REFERENCES lessons(id) ON DELETE SET NULL,   -- scope at ask time
  subject_id    uuid REFERENCES subjects(id) ON DELETE SET NULL,
  quiz_chunk_id uuid REFERENCES chunks(id) ON DELETE CASCADE,     -- quiz modes
  option_label  text,                                             -- quiz_option mode
  note_draft_id uuid,                                             -- note_section mode
  scope_used    scope_t NOT NULL,
  verdict       verdict_t NOT NULL,
  answer_md     text,
  answer_model  text,
  latency_ms    int,
  created_at    timestamptz NOT NULL DEFAULT now()
);
```

In v0.1 citations hung off a quiz item. That no longer covers the product: a question can be free text, a worksheet item, a single clickable option, or a prompt for one section of a 共筆 draft. `queries` is the common parent for all four, so citation, verdict and divergence logic is written once.

`scope_used` records which rung of the escalation ladder produced the answer. This is the single most useful column for evaluation later — it tells you how often the default scope sufficed and how often widening actually helped.

```sql
CREATE TABLE citations (
  id              bigserial PRIMARY KEY,
  owner_id        uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  query_id        uuid NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
  source_chunk_id uuid REFERENCES chunks(id) ON DELETE SET NULL,
  rank            int NOT NULL,
  dense_score     real,
  lexical_score   real,
  fused_score     real,                      -- after RRF
  margin          real,                       -- fused(top1) - fused(top2)
  via_concept_id  uuid REFERENCES concepts(id) ON DELETE SET NULL,
  verdict         verdict_t NOT NULL,
  verdict_model   text,
  verdict_reason  text,
  created_at      timestamptz NOT NULL DEFAULT now()
);
```

`source_chunk_id` is nullable: a verdict of `insufficient` is a real, storable outcome with no source attached. That is the row shape for "the notes do not cover this", which was the correct answer for all 20 items in the cross-topic test.

Three scores are kept separately rather than collapsed, because the prototype showed an absolute score cannot distinguish a real hit from a lexical coincidence. `margin` is the relative signal that does, and keeping the components lets the gate be tuned on real data instead of a guessed threshold.

`via_concept_id` records *why* a cross-subject result was admitted. Without it, a surprising cross-subject citation is unexplainable; with it, the answer can say "linked through HPV" and a bad link is traceable to a bad concept extraction.

`verdict` is written by a reranking/groundedness pass, not by the retriever. Retrieval ranks; the verdict decides. Separating them is what makes abstention possible.

```sql
CREATE TABLE answer_sources (
  id          bigserial PRIMARY KEY,
  query_id    uuid NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  doc_type    doc_type_t NOT NULL,
  claim_group int NOT NULL,                  -- same number = saying the same thing
  stance      stance_t NOT NULL,
  summary_md  text NOT NULL,
  display_rank int NOT NULL,                 -- from the student's source_priority
  UNIQUE (query_id, document_id)
);
```

This table is the multi-source feature. After retrieval, the chunks are grouped by *claim*: `claim_group` is the same integer for every source asserting the same thing. A group of size one where others disagree is `differs`; a point only one source raises is `unique`.

That turns "the PPT, the 共筆 and the recording say slightly different things" from an essay the model has to write into structured data the interface can render — collapsed to the top-priority source by default, expanded side by side when the student hits the toggle. `display_rank` is computed from `user_settings.source_priority` at answer time, so re-ordering the preference does not require re-answering.

---

## 共筆 mode

```sql
CREATE TABLE note_templates (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id   uuid REFERENCES users(id) ON DELETE CASCADE,  -- NULL = shared default
  name       text NOT NULL,
  sections   jsonb NOT NULL,       -- ordered [{key, heading, instruction}]
  rules_md   text,                 -- house style: what to include, cite, omit
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE note_drafts (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  lesson_id   uuid NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
  template_id uuid REFERENCES note_templates(id) ON DELETE SET NULL,
  status      text NOT NULL DEFAULT 'draft'
              CHECK (status IN ('draft','review','final')),
  body_md     text,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);
```

`sections` plus `rules_md` are the porting target for the existing note-taker skill. A Claude Skill is a set of instructions Claude reads; it does not execute inside Palpate. What transfers is the content — the section order goes into `sections`, the writing rules into `rules_md` — and the pipeline renders them into its prompt. The rules themselves need no rewriting.

Note drafting needs no new retrieval machinery. Each section becomes a `queries` row with `mode = 'note_section'`, which means every drafted paragraph gets the same citations and the same abstention behaviour as an answer. That is why this mode is cheap to add, and why it belongs before cross-subject work in the build order.

---

## Isolation

Every user-owned table carries `owner_id`, read from the verified JWT on the server — never from a request parameter. A client that can pass its own user id can read its classmates' notes.

Belt and braces, with row-level security:

```sql
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY chunks_owner ON chunks
  USING (owner_id = current_setting('app.current_user_id')::uuid);
```

The API sets `app.current_user_id` once per transaction from the token. After that, a forgotten `WHERE owner_id = …` returns nothing rather than leaking. Highest-value safeguard in a multi-tenant system, and about ten lines per table.

---

## Indexes

```sql
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_scope    ON chunks (owner_id, lesson_id, kind, doc_type);
CREATE INDEX chunks_subject  ON chunks (owner_id, subject_id, kind);
CREATE INDEX chunks_search   ON chunks USING gin (to_tsvector('simple', search_text));
CREATE INDEX spans_lookup    ON chunk_spans (chunk_id, anchor);
CREATE INDEX cc_by_concept   ON chunk_concepts (concept_id);
CREATE INDEX concepts_forms  ON concepts USING gin (surface_forms);
CREATE INDEX citations_query ON citations (query_id, rank);
CREATE INDEX asrc_query      ON answer_sources (query_id, display_rank);
CREATE INDEX annot_page      ON annotations (owner_id, document_id, page);
CREATE INDEX queries_recent  ON queries (owner_id, created_at DESC);
```

`cc_by_concept` is indexed by concept rather than by chunk because the cross-subject lookup runs in that direction: given the concepts in scope, find chunks elsewhere that carry them.

A caveat: pgvector's HNSW index and a restrictive `WHERE` interact badly, because the graph is walked before the filter applies and can return fewer rows than requested. At stage-1 volumes a plain scan over tens of thousands of rows is fast enough that this does not matter. If it starts to, the fix is partial indexes per subject, not a different database.

---

## User highlights

```sql
CREATE TABLE annotations (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  anchor      anchor_t NOT NULL,
  page        int,
  rects       jsonb,                -- page_bbox
  t_start numeric(9,3), t_end numeric(9,3),   -- time_range, for video notes
  quoted_text text,
  note        text,
  color       text NOT NULL DEFAULT 'yellow',
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);
```

Student highlights anchor to the document and position, never to a chunk. A student highlights what they see, and that selection may straddle chunk boundaries or sit in a figure caption we never chunked. Anchoring to a chunk would lose every highlight whenever we re-ingest with a better parser; anchoring to position survives re-ingest.

---

## The retrieval path

Scope 1, the default — all source types inside one lesson, which is what makes multi-source comparison safe:

```sql
WITH dense AS (
  SELECT id, row_number() OVER (ORDER BY embedding <=> $qvec) AS r
  FROM chunks
  WHERE owner_id = $uid AND kind = 'section' AND lesson_id = $lesson
  ORDER BY embedding <=> $qvec LIMIT 50
),
lexical AS (
  SELECT id, row_number() OVER (
           ORDER BY ts_rank(to_tsvector('simple', search_text),
                            plainto_tsquery('simple', $qbigrams)) DESC) AS r
  FROM chunks
  WHERE owner_id = $uid AND kind = 'section' AND lesson_id = $lesson
  LIMIT 50
)
SELECT c.id, c.doc_type, c.document_id,
       1.0/(60 + COALESCE(d.r, 1000)) + 1.0/(60 + COALESCE(l.r, 1000)) AS fused
FROM chunks c
LEFT JOIN dense d ON d.id = c.id
LEFT JOIN lexical l ON l.id = c.id
WHERE d.id IS NOT NULL OR l.id IS NOT NULL
ORDER BY fused DESC LIMIT 12;
```

Two retrievers fused by *reciprocal rank fusion*. Dense vectors catch paraphrase and cross-language matches; lexical matching catches exact identifiers such as `USP6`, `NFATC2`, `Sistrunk procedure`, which dense embeddings handle unreliably. RRF combines by rank, not score, because the two scores are on incomparable scales.

Scope 3, cross-subject — reached only when the lesson and subject rungs came back `insufficient`, and gated on a shared concept rather than on a loosened threshold:

```sql
WITH scope_concepts AS (        -- concepts present in the question or in-scope hits
  SELECT DISTINCT cc.concept_id
  FROM chunk_concepts cc
  WHERE cc.chunk_id = ANY($in_scope_chunk_ids)
)
SELECT c.id, c.subject_id, sc.concept_id AS via_concept,
       (c.embedding <=> $qvec) AS dist
FROM chunks c
JOIN chunk_concepts cc ON cc.chunk_id = c.id
JOIN scope_concepts sc ON sc.concept_id = cc.concept_id
WHERE c.owner_id = $uid
  AND c.kind = 'section'
  AND c.subject_id <> $subject          -- genuinely another subject
ORDER BY dist LIMIT 12;
```

The join on `scope_concepts` is the passport: a chunk from another subject cannot appear unless it carries a concept the current question already involves. Everything retrieved still passes the groundedness verdict before a student sees it.

---

## Stage 1 minimum

Not all of the above is needed to ship something real to ten classmates. The stage-1 subset:

`users`, `user_settings`, `subjects`, `lessons`, `documents`, `chunks`, `chunk_spans`, `quiz_items`, `queries`, `citations`, `answer_sources`.

Deferred, but defined now so the columns referencing them already exist: `concepts`, `chunk_concepts` (cross-subject), `note_templates`, `note_drafts` (共筆 mode), `ingest_clients`, `watched_folders` (desktop companion), `annotations` (in-PDF highlighting).

The split matters: `chunk_spans` is in the minimum set even though stage 1 does not highlight anything, because anchors can only be captured while the document is being parsed.

---

## Open decisions

1. Embedding model, which fixes `vector(1024)`. Must be settled before the first real ingest — changing it rewrites the column and the index.
2. `pg_bigm` extension versus precomputed `search_text` bigrams.
3. The gate for `verdict = 'insufficient'`: margin threshold alone, or margin plus an LLM groundedness check. The 20 cross-topic worksheet items are a ready-made negative set, since every correct answer there is "insufficient".
4. Whether `textbook` documents respect the lesson filter at all. A textbook spans every lesson, so filtering it by lesson would hide it entirely.
5. `.docx` highlighting: render to PDF at ingest, or highlight in a text view.
6. How concepts are extracted — LLM pass at ingest, a curated dental term list, or both.

---

## Changelog

- **v0.2** — three product modes encoded. Added `subjects`, polymorphic `chunk_spans`, `concepts`/`chunk_concepts`, `queries` as the common citation parent, `answer_sources` for source divergence, 共筆 tables, ingestion-client tables, per-user `source_priority` defaulting to 共筆 first.
- **v0.1** — initial draft, from parsing 口病共筆第二週 (10 topics, 38 sections) and 0910課後測驗 Epithelial(I) (20 items), and from the cross-topic retrieval test that returned 19 plausible but wrong matches.
