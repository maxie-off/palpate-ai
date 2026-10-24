# Palpate: Design Outline (2026-10-10)

*Exam study agent that finds the source behind every answer.*

Repo name: `palpate` (use `palpate-ai` if taken). Check GitHub, domain, and trademark availability before publishing; none of these were searched.

---

## 1. What we're building

A quiz-oriented study agent for dental students. The student uploads their own course material (PDF/PPTX), then uploads an exam sheet. For each question, Palpate retrieves the relevant knowledge and images from the student's private knowledge base and shows citations: `{doc, page, image, description}`.

- Click a **question** to see a grounded summary of the answer, with citations.
- Click a **knowledge term** inside an option to see a summary of that term, with citations.
- The knowledge base is private and self-customised, so students stick to what their school actually taught instead of the whole internet.

Why this differs from ChatGPT/Claude chat: answers come only from the student's own sources, every claim is traceable, and the system says "insufficient evidence" when the sources don't support an answer.

**Scope:** Stage 1 is dentistry only (Oral and Maxillofacial Pathology). Domain adaptation to other majors and countries comes later.

**Audience:** Taiwanese dental students. Material is Traditional Chinese mixed with English terms. UI language is English.

**Goal of the project:** resume-presentable outcome in about a week, working prototype in 1 to 2 months, possible founding project later. Code and structure must stay clean for GitHub.

---

## 2. Design principles

1. **Applied AI, no model training.** We call foundation models through APIs and engineer the system around them.
2. **A pipeline, not a free-roaming agent.** Quiz help is predictable: parse, retrieve, answer, verify. This keeps answers verifiable.
3. **Grounded or silent.** Answer only when the sources support it; otherwise abstain.
4. **Every citation is verified.** The model returns chunk IDs; a deterministic check confirms the cited text really exists in that chunk. Failed citations are dropped.
5. **Frameworks at the edges.** LangChain/LangGraph sit behind our own interfaces; prompts and domain logic stay ours.
6. **Measure everything.** An eval harness with a hand-labelled gold set gives resume-ready numbers.

---

## 3. System design

```
 INGESTION (offline, per upload)
 ┌────────────┐   ┌──────────────┐   ┌───────────────────┐   ┌────────────────┐
 │ pdf / pptx │──▶│ Parser       │──▶│ Chunker           │──▶│ Embed + Index  │
 └────────────┘   │ text → md    │   │ by heading, keeps │   │ vector + BM25  │
                  │ images →     │   │ page, highlight   │   │ in Postgres    │
                  │ files+caption│   │ flag, image refs  │   └────────────────┘
                  └──────────────┘   └───────────────────┘
                         │ images get a vision-model description,
                         ▼ stored as their own retrievable chunks

 QUERY (online, per exam sheet)
 ┌────────────┐   ┌───────────────┐   ┌─────────────────────────────┐
 │ exam sheet │──▶│ Question      │──▶│ Retriever                   │
 │ pdf        │   │ extractor     │   │ hybrid search + term        │
 └────────────┘   │ stem, options,│   │ glossary expansion + rerank │
                  │ knowledge     │   └──────────────┬──────────────┘
                  │ terms         │                  ▼
                  └───────────────┘   ┌─────────────────────────────┐
                                      │ Answerer (Claude Sonnet)    │
                                      │ grounded summary, returns   │
                                      │ chunk IDs as citations      │
                                      └──────────────┬──────────────┘
                                                     ▼
                                      ┌─────────────────────────────┐
                                      │ Citation verifier           │
                                      │ cited span must exist in    │
                                      │ the source chunk            │
                                      └──────────────┬──────────────┘
                                                     ▼
                                          API ──▶ web UI
                                   (click question or term → popup
                                    summary + citations + images)
```

### Key techniques and why

- **Layout-aware parsing (Docling or MinerU + PyMuPDF).** Keeps figures next to their page and nearby text, which plain text extraction loses.
- **Highlight signal (PyMuPDF, no AI).** Yellow highlights and bold mark what the teacher stressed. We compute an "emphasised" flag per chunk and use it to rank retrieval. Check early that highlights survive as annotations or drawn rectangles in the exported PDFs.
- **Vision via a multimodal LLM, no vision transformer to host.**
  - At ingestion: Haiku writes a short bilingual description of each figure, stored as a searchable chunk linked to the image.
  - At query time: image-based questions (e.g. a histology photo) go to Sonnet together with the retrieved context.
  - Later (stage 3): multimodal embeddings for visual similarity search.
- **Bilingual glossary.** Haiku extracts a term table (中文, English, aliases) from the notes, e.g. 鰓裂囊腫 ↔ Branchial Cleft Cyst. At query time we expand question terms through it so English options find Chinese notes and the reverse. This is the main differentiator.
- **Hybrid retrieval.** Vector search plus BM25 keyword search, then reranking. Chinese keyword search uses jieba segmentation, because Postgres full-text search handles Chinese poorly.
- **Abstention.** If retrieved evidence is weak or verification fails, return "insufficient evidence" instead of guessing.

---

## 4. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | Python 3.12, FastAPI, `uv` | Typed, fast to build, familiar to recruiters |
| Database | Postgres + pgvector | Vectors, chunks, glossary, citations in one place |
| Keyword search | jieba + BM25 | Chinese segmentation |
| LLM | Claude via Bedrock: Sonnet 5.5 (answers), Haiku 5.5 (bulk: captions, extraction, glossary), Opus 5.5 (eval judge only) | Right model per job, cost control |
| Embeddings | Cohere Embed Multilingual on Bedrock (default); local bge-m3 as an ablation | Claude has no embedding model; need Chinese/English |
| Orchestration | LangChain (`langchain-aws`, splitters, pgvector store) at the edges; LangGraph for the query workflow (stage 3) | Resume keywords, and the query flow is a real graph with a retry loop |
| Frontend | Next.js | Clickable question and term popups |
| Infra | Docker Compose locally; AWS CDK (Python) or Terraform later | Reproducible |
| Quality | Ruff, mypy, pytest, pre-commit, GitHub Actions | Clean repo |

All LLM calls go through one thin `LLMClient` interface so the provider can be swapped by config. Start on Bedrock from day one; if new-account quotas block us, the same interface lets us start on the Anthropic API.

### AWS architecture (stage 3)

```
 Browser ──▶ CloudFront ──▶ S3 (frontend)
    │
    ▼
 API Gateway / App Runner ──▶ Fargate: FastAPI + LangGraph workflow
                                  │            │              │
                                  ▼            ▼              ▼
                        RDS Postgres     S3 (raw docs,   Bedrock: Claude
                        + pgvector       page images,    Sonnet/Haiku +
                        (chunks, terms,  figures)        embedding model
                         citations)
 Ingestion worker: same image, triggered by S3 upload
 Secrets Manager · CloudWatch · IAM roles
```

---

## 5. What is stored vs. what is code

| Item | Where it lives | Notes |
|---|---|---|
| Bilingual glossary | **Stored data** (Postgres / RDS) | Generated per corpus; persistent because rebuilding costs LLM calls and entries can be corrected by hand |
| Parsing and chunking rules | **Code and config in git** | Only their output (chunks, captions) is stored. Record the rule version on each chunk so changes can be re-ingested |
| Retrieval settings (top-k, rerank, highlight weight) | **YAML config in git** | Keep eval results per settings version in the repo as proof of improvement |

In stage 1 nothing touches AWS: Postgres runs in Docker locally.

---

## 6. Repo structure

```
palpate/
├── README.md              architecture diagram, demo gif, eval results
├── docker-compose.yml
├── backend/
│   ├── app/
│   │   ├── api/           routes only, no logic
│   │   ├── ingestion/     parsers, chunker, image captioner
│   │   ├── retrieval/     hybrid search, glossary, reranker
│   │   ├── answering/     prompts, answerer, citation verifier
│   │   ├── models/        pydantic schemas + DB models
│   │   └── core/          config, db, LLM client
│   └── tests/
├── frontend/
├── config/                dentistry prompts and retrieval settings (YAML)
├── eval/                  gold questions, metrics script
├── data/sample/           small shareable demo data only
└── docs/                  design decisions
```

Stage 1 stays dentistry-only. When other majors come in (stage 4), `config/` becomes per-domain folders (prompts, glossary builder, parsing rules), so adding a field means adding a folder instead of changing core code.

---

## 7. Data and legal

- The public repo contains code, an architecture write-up, and a small demo corpus that is legal to publish.
- Demo corpus options: open-licensed sources (PMC Open Access articles under CC BY, Wikipedia under CC BY-SA with attribution) plus a self-written bilingual mini course and worksheet on 5 or 6 lesions.
- Real material (Regezi, Neville, class slides, classmates' shared notes) stays **local and gitignored**. Publish only aggregate metrics (recall, citation accuracy), never the content.
- Verify each source's license; this is not legal advice.
- The sample files used while designing (口病共筆 notes on cysts, Epithelial(i) quiz) were only format examples. Their topics don't match each other, which is why the final demo needs the Lesson 3 slides and textbook chapters.

---

## 8. Cost estimate, stage 1

Prices (per million tokens, input/output): Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50, Opus 5.5 $4/$20. Assumptions: about 1,500 textbook pages, 2,000 images, 20 questions per worksheet, 30% allowance for Chinese token inflation.

| Item | Estimate |
|---|---|
| Embeddings (about 1.2M tokens, one time) | about $0.12 ($0 with local bge-m3) |
| Image captions (one time) | about $0.50 with Haiku, about $10 with Sonnet |
| One worksheet run (20 questions with popups) | about $1.2 to $1.5 (roughly $0.06 per question) |
| About 25 dev iterations | about $20 to $40 |
| About 10 Opus-judged eval runs | about $18 |
| **Stage 1 total** | **about $30 to $70; hard cap $100** |

AWS hosting is $0 in stage 1. In stage 3 expect roughly $30 to $60 per month (RDS, Fargate, S3); this is unverified, so use the AWS Pricing Calculator. Bedrock regional endpoints carry a 10% premium over global ones. Cost per question is also a useful unit-economics number for the founding-project story.

Source: https://platform.claude.com/docs/en/about-claude/pricing

---

## 9. Stages and goals

### Stage 1, week 1: resume-presentable vertical slice (dentistry only)
- Run everything locally with one `docker compose up`.
- Ingest PDF/PPTX into markdown chunks, with images saved separately.
- Upload a worksheet and get per-question citations (document, page, image, description).
- Click a question to see a grounded summary.
- Answer only when sources support it, otherwise say "insufficient evidence".
- Add the citation verifier.
- Clean repo, README with the architecture diagram, short demo gif.

### Stage 2, weeks 2 to 4: quality
- Image captioning and the teacher-highlight signal.
- Bilingual (Traditional Chinese / English) glossary.
- Hybrid search with jieba, plus reranking.
- Clickable knowledge terms with summaries and citations.
- Eval harness with a hand-labelled gold set.
- Publish metrics: recall@5, citation accuracy, abstention precision.

### Stage 3, weeks 5 to 8: prototype
- Deploy to AWS (Bedrock, S3, RDS, Fargate) with CDK or Terraform.
- CI/CD, tests, tracing (Langfuse or CloudWatch).
- Move the query workflow to LangGraph.
- UI polish.
- First real-user test with a dental student; fix the top issues.

### Stage 4, month 3 and on: product
- Auth and multi-user support.
- Scheduled review prompts and weak-point analytics.
- Per-domain config folders for other majors.
- Support other countries' curricula.
- Decide whether to continue as a founding project.

---

## 10. Environment setup (in order)

1. Install Python 3.12 with `uv`, Node 20 with pnpm, Docker Desktop, git, VS Code, and AWS CLI v2. VS Code extensions: Python, Pylance, Ruff, Docker, Jupyter. Run Claude Code in its terminal.
2. Create the GitHub repo with `pyproject.toml`, Ruff, mypy, pytest, and pre-commit. Add a GitHub Actions workflow for lint, type checks, and tests.
3. Write `docker-compose.yml` with Postgres (pgvector image), backend, and frontend. Keep config in `.env` loaded through pydantic-settings; never commit it.
4. Set up AWS: enable MFA, use IAM Identity Center instead of the root user, add budget alerts at $20 and $50, request Bedrock model access in us-east-1 or us-west-2, and create an S3 bucket.
5. Smoke test: one Sonnet call and one embedding call through `LLMClient`.

---

## 11. Evaluation plan

- Hand-label which source page each worksheet question should cite (the gold set).
- Metrics: retrieval recall@k, citation accuracy (verifier pass rate and correctness), abstention precision.
- Use Opus as the judge for answer quality; keep deterministic checks for citations.
- Ablations worth reporting: glossary expansion on/off, highlight weighting on/off, Cohere vs. bge-m3 embeddings, reranker on/off.
- Example resume line: "retrieval recall@5 improved from X to Y after bilingual glossary expansion."

---

## 12. Open items

1. AWS account status and Bedrock access enabled?
2. Confirm the demo corpus plan: open or self-written for the public repo, real data kept private.
3. Which 2 or 3 chapters or slide decks to design the parsers against first (10 to 20 page excerpts are enough).
4. Check name, GitHub, domain, and trademark availability for Palpate.
5. Add a project folder from the "+" menu so the repo can be scaffolded on your machine, or generate the files here to download.
