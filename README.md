# Palpate

*Exam study agent that finds the source behind every answer.*

Palpate helps dental students study from their **own** course material. Upload your lecture notes, slides, and textbooks, then upload a past exam. For every question, Palpate finds the relevant passages and figures in your knowledge base and shows exactly where they came from: document, page, image, and a short grounded summary.

If your sources don't support an answer, Palpate says so instead of guessing.

> **Status:** stage 1 scaffold (dentistry only). See [Roadmap](#roadmap).

## Why this is different from a general chatbot

- **Grounded or silent.** Answers come only from your sources, otherwise "insufficient evidence".
- **Verified citations.** The model returns chunk IDs and quotes; a deterministic check confirms each quote really exists in the cited chunk. Failed citations are dropped.
- **Bilingual by design.** Taiwanese dental students study in Traditional Chinese with English terms mixed in. A bilingual glossary lets an English exam option find Chinese notes and the reverse.
- **Teacher emphasis as a signal.** Highlighted passages in lecture notes rank higher in retrieval.

## How it works

```
 INGESTION (offline, per upload)
 pdf/pptx ─▶ Parser ─▶ Chunker ─▶ Embed + Index (Postgres + pgvector)
             text,       by heading,
             images,     keeps page,
             highlights  highlight flag

 QUERY (online, per exam sheet)
 exam sheet ─▶ Question extractor ─▶ Retriever ─▶ Answerer ─▶ Citation verifier ─▶ UI
                (stem, options,      hybrid +     Claude       quote must exist
                 knowledge terms)    glossary     (Bedrock)    in cited chunk
```

It is a pipeline, not a free-roaming agent: the flow is predictable, which keeps answers verifiable. See [docs/decisions](docs/decisions).

## Quickstart

Prerequisites: Docker Desktop, Python 3.12 with [uv](https://docs.astral.sh/uv/), and (for LLM calls) AWS credentials with Bedrock model access.

```bash
cp .env.example .env          # fill in Bedrock model IDs
make up                       # Postgres (pgvector) + backend on :8000
curl localhost:8000/health    # {"status":"ok"}
```

Run checks locally:

```bash
make install   # uv sync with dev extras
make check     # ruff + mypy + pytest
```

## Repo layout

```
palpate/
├── backend/app/
│   ├── api/          routes only, no business logic
│   ├── ingestion/    parser, chunker, (stage 2) image captioner
│   ├── retrieval/    glossary expansion, retriever interface
│   ├── answering/    prompts, answerer, citation verifier
│   ├── models/       pydantic schemas
│   └── core/         settings, LLM client, DB helper
├── backend/tests/
├── config/           domain config (dentistry): chunking + retrieval settings
├── db/init.sql       Postgres schema (pgvector)
├── eval/             gold questions + metrics (stage 2)
├── data/sample/      small shareable demo data only
├── docs/             design notes and decision records
└── frontend/         Next.js UI (stage 1, in progress)
```

## What is stored vs. what is code

| Item | Lives in |
|---|---|
| Chunks, embeddings, bilingual glossary | Postgres (persistent) |
| Parsing and chunking rules | Code + `config/` in git; each chunk records its `rule_version` |
| Retrieval settings (top-k, thresholds, highlight boost) | `config/dentistry.yaml` in git |

## Data and copyright

This repo contains code and a small shareable demo corpus only. Textbooks, class slides, and classmates' notes are **never committed**: keep them under `data/private/` (gitignored). Only aggregate metrics from private data are published.

## Roadmap

1. **Stage 1:** local vertical slice. Ingest, retrieve, answer with verified citations, abstain when unsure.
2. **Stage 2:** image captioning, highlight ranking, bilingual glossary, hybrid search + rerank, eval harness with published metrics.
3. **Stage 3:** AWS deployment (Bedrock, S3, RDS, Fargate), LangGraph workflow, tracing, first real-user test.
4. **Stage 4:** auth, review prompts, other majors via per-domain config.
