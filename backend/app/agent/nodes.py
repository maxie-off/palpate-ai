"""Graph nodes for ingest, transcription, retrieval, drafting, and citation checks.

Each factory closes over ``PipelineDeps`` and returns a function LangGraph can call.
Nodes never raise for expected failures (missing file, unconfigured speech, bad model
JSON). They set ``status`` to ``failed`` or an abstention and let the edges stop the run.
"""

import hashlib
import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from app.agent.chunking import chunk_pages_for_index, chunks_from_transcript, make_media_chunk
from app.agent.deps import PipelineDeps
from app.agent.jsonutil import parse_json_object
from app.agent.prompts import build_notes_prompt, build_qa_prompt
from app.agent.state import AgentState
from app.agent.store import SearchParams, embed_input
from app.agent.types import (
    PARSER_AUDIO,
    PARSER_PDF,
    ChunkSpan,
    Claim,
    DocType,
    Highlight,
    IngestedDocument,
    NoteSection,
    SearchHit,
    SourceAnchor,
    SourceCitation,
    StoredChunk,
)
from app.answering.verifier import normalise_text, verify_citations
from app.core.config import NoteSectionSpec
from app.models.schemas import BBox, Chunk, Citation, PageContent
from app.retrieval.glossary import GlossaryEntry, expand_terms

_VISION_PROMPT = (
    "Describe this figure from a dental lecture in one or two sentences. "
    "Use the language of the surrounding notes. Do not invent labels that are not visible."
)
_CAPTION_RE = re.compile(r"^(figure|fig\.|圖)\b", re.IGNORECASE)
_DOC_TYPES = {"note", "slides", "handout", "video", "textbook", "quiz"}
_EMBED_BATCH = 64


def route_node(state: AgentState) -> dict[str, Any]:
    return {"trace": ["route"]}


def quiz_node(state: AgentState) -> dict[str, Any]:
    return {
        "status": "not_implemented",
        "output": {
            "status": "not_implemented",
            "summary": (
                "Quiz answering is not implemented yet. "
                "It will reuse the qa branch with a parsed stem and options."
            ),
        },
        "trace": ["quiz"],
    }


def make_summarize() -> Any:
    def summarize(state: AgentState) -> dict[str, Any]:
        return {
            "status": "not_implemented",
            "output": {
                "status": "not_implemented",
                "summary": (
                    "Summarize is not implemented yet. "
                    "It will reuse retrieval and the same citation check as qa."
                ),
            },
            "trace": ["summarize"],
        }

    return summarize


def make_ingest_pdf(deps: PipelineDeps) -> Any:
    def ingest_pdf(state: AgentState) -> dict[str, Any]:
        if not state["pdf_path"]:
            if state["task"] == "ingest":
                return {
                    "status": "failed",
                    "error": "ingest requires pdf_path",
                    "trace": ["ingest_pdf"],
                }
            return {"trace": ["ingest_pdf"]}
        path = Path(state["pdf_path"])
        if not path.is_file():
            return {
                "status": "failed",
                "error": f"PDF not found: {path}",
                "trace": ["ingest_pdf"],
            }
        try:
            content_hash = _sha256(path)
            existing = deps.store.find_by_hash(state["owner_id"], content_hash)
            if existing is not None:
                return {
                    "active_doc_id": existing.doc_id,
                    "ingested": _append_docs(state["ingested"], existing),
                    "trace": ["ingest_pdf"],
                }
            doc_id = state["doc_id"] or str(uuid.uuid4())
            pages = deps.parser.parse(path)
            doc_type = _doc_type(state["doc_type"])
            title = state["title"] or path.stem
            text_chunks = chunk_pages_for_index(
                pages,
                doc_id=doc_id,
                owner_id=state["owner_id"],
                lesson_id=state["lesson_id"],
                doc_type=doc_type,
                title=title,
                config=deps.domain.chunking,
            )
            figures = _figure_chunks(
                deps,
                pages,
                doc_id=doc_id,
                owner_id=state["owner_id"],
                lesson_id=state["lesson_id"],
                doc_type=doc_type,
                title=title,
                rule_version=deps.domain.chunking.rule_version,
                start_ordinal=len(text_chunks),
            )
            document = IngestedDocument(
                doc_id=doc_id,
                owner_id=state["owner_id"],
                lesson_id=state["lesson_id"],
                doc_type=doc_type,
                title=title,
                source_path=str(path),
                mime_type="application/pdf",
                content_hash=content_hash,
                page_count=len(pages),
                parser=PARSER_PDF,
                parser_version=deps.domain.chunking.rule_version,
            )
            chunks = text_chunks + figures
            return {
                "active_doc_id": doc_id,
                "pending_chunks": [_dump(chunk) for chunk in chunks],
                "pending_docs": [_dump(document)],
                "trace": ["ingest_pdf"],
            }
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"Could not read PDF: {exc}",
                "trace": ["ingest_pdf"],
            }

    return ingest_pdf


def make_transcribe(deps: PipelineDeps) -> Any:
    def transcribe_audio(state: AgentState) -> dict[str, Any]:
        if state["status"] == "failed" or not state["audio_path"]:
            return {"trace": ["transcribe_audio"]}
        path = Path(state["audio_path"])
        if not path.is_file():
            return {
                "status": "failed",
                "error": f"Audio not found: {path}",
                "trace": ["transcribe_audio"],
            }
        try:
            transcript = deps.speech.transcribe(path, language=deps.domain.locale)
            content_hash = _sha256(path)
            existing = deps.store.find_by_hash(state["owner_id"], content_hash)
            if existing is not None:
                return {
                    "ingested": _append_docs(state["ingested"], existing),
                    "trace": ["transcribe_audio"],
                }
            doc_id = _audio_doc_id(state)
            title = state["title"] or path.stem
            if state["pdf_path"]:
                title = f"{title} (recording)"
            chunks = chunks_from_transcript(
                transcript,
                doc_id=doc_id,
                owner_id=state["owner_id"],
                lesson_id=state["lesson_id"],
                title=title,
                rule_version=deps.domain.chunking.rule_version,
            )
            document = IngestedDocument(
                doc_id=doc_id,
                owner_id=state["owner_id"],
                lesson_id=state["lesson_id"],
                doc_type="video",
                title=title,
                source_path=str(path),
                mime_type=_audio_mime(path),
                content_hash=content_hash,
                page_count=0,
                parser=PARSER_AUDIO,
                parser_version=transcript.model or "unknown",
            )
            pending_chunks = [_load_chunk(raw) for raw in state["pending_chunks"]]
            pending_chunks.extend(chunks)
            pending_docs = [_load_doc(raw) for raw in state["pending_docs"]]
            pending_docs.append(document)
            return {
                "pending_chunks": [_dump(chunk) for chunk in pending_chunks],
                "pending_docs": [_dump(doc) for doc in pending_docs],
                "trace": ["transcribe_audio"],
            }
        except Exception as exc:
            return {
                "status": "failed",
                "error": str(exc),
                "trace": ["transcribe_audio"],
            }

    return transcribe_audio


def make_index(deps: PipelineDeps) -> Any:
    def index_chunks(state: AgentState) -> dict[str, Any]:
        if state["status"] == "failed":
            return {"trace": ["index_chunks"]}
        pending_chunks = [_load_chunk(raw) for raw in state["pending_chunks"]]
        pending_docs = [_load_doc(raw) for raw in state["pending_docs"]]
        if pending_chunks:
            vectors = _embed_batched(deps, [embed_input(chunk) for chunk in pending_chunks])
            pending_chunks = [
                chunk.model_copy(update={"embedding": vector})
                for chunk, vector in zip(pending_chunks, vectors, strict=True)
            ]
        by_doc: dict[str, list[StoredChunk]] = {}
        for chunk in pending_chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        ingested = [_load_doc(raw) for raw in state["ingested"]]
        for document in pending_docs:
            ingested.append(deps.store.upsert(document, by_doc.get(document.doc_id, [])))
        output = None
        if state["task"] == "ingest":
            output = {"status": "ok", "summary": _ingest_blurb(ingested)}
        return {
            "pending_chunks": [],
            "pending_docs": [],
            "ingested": [_dump(doc) for doc in ingested],
            "output": output,
            "trace": ["index_chunks"],
        }

    return index_chunks


def make_retrieve(deps: PipelineDeps) -> Any:
    def retrieve(state: AgentState) -> dict[str, Any]:
        if state["status"] == "failed":
            return {"trace": ["retrieve"]}
        if state["task"] == "draft_notes":
            hits = _notes_context(state, deps)
        else:
            hits = _search(state, deps, state["message"])
        return {"hits": [_dump(hit) for hit in hits], "trace": ["retrieve"]}

    return retrieve


def make_draft_answer(deps: PipelineDeps) -> Any:
    def draft_answer(state: AgentState) -> dict[str, Any]:
        if state["status"] == "failed":
            return {"trace": ["draft_answer"]}
        hits = [_load_hit(raw) for raw in state["hits"]]
        if not hits:
            return {
                "draft": _empty_draft("qa"),
                "retryable": False,
                "trace": ["draft_answer"],
            }
        system, user = build_qa_prompt(state["message"], hits, state["rejection"])
        raw = deps.llm.complete(
            system=system,
            user=user,
            model_id=deps.answer_model_id,
            max_tokens=1024,
        )
        return {
            "attempts": state["attempts"] + 1,
            "draft": _parse_qa(raw),
            "retryable": False,
            "trace": ["draft_answer"],
        }

    return draft_answer


def make_draft_notes(deps: PipelineDeps) -> Any:
    def draft_notes(state: AgentState) -> dict[str, Any]:
        if state["status"] == "failed":
            return {"trace": ["draft_notes"]}
        hits = [_load_hit(raw) for raw in state["hits"]]
        if not hits:
            return {
                "draft": _empty_draft("notes"),
                "retryable": False,
                "trace": ["draft_notes"],
            }
        system, user = build_notes_prompt(
            state["message"],
            hits,
            deps.domain.notes.sections,
            deps.domain.notes.rules_md,
            state["rejection"],
        )
        raw = deps.llm.complete(
            system=system,
            user=user,
            model_id=deps.answer_model_id,
            max_tokens=2048,
        )
        return {
            "attempts": state["attempts"] + 1,
            "draft": _parse_notes(raw, deps.domain.notes.sections),
            "retryable": False,
            "trace": ["draft_notes"],
        }

    return draft_notes


def make_verify(deps: PipelineDeps) -> Any:
    def verify(state: AgentState) -> dict[str, Any]:
        if state["status"] == "failed":
            return {"trace": ["verify"]}
        draft = state["draft"] or _empty_draft("qa")
        hits = [_load_hit(raw) for raw in state["hits"]]
        shown = {hit.chunk.id: hit for hit in hits}
        surviving, dropped_quotes = _check_claims(draft, shown)
        status = "answered" if surviving else "insufficient_evidence"
        retryable = (
            status != "answered"
            and bool(hits)
            and (bool(draft.get("malformed")) or _draft_had_citations(draft))
        )
        sections = (
            _rebuild_sections(deps.domain.notes.sections, surviving)
            if draft.get("kind") == "notes"
            else []
        )
        flat_claims = [claim for claims in surviving.values() for claim in claims]
        summary = str(draft.get("summary") or "") if status == "answered" else ""
        if status == "answered" and not summary:
            summary = " ".join(claim.text for claim in flat_claims)
        if sections:
            summary = summary or "\n\n".join(
                f"## {section.heading}\n{section.body_md}".rstrip()
                for section in sections
                if section.body_md
            )
        citations = [citation for claim in flat_claims for citation in claim.citations]
        highlights = [
            Highlight(
                doc_id=citation.doc_id,
                chunk_id=citation.chunk_id,
                quote=citation.quote,
                anchor=anchor,
            )
            for citation in citations
            for anchor in citation.anchors
        ]
        return {
            "retryable": retryable,
            "rejection": (
                "\n".join(f"- {quote}" for quote in dropped_quotes[:8]) if retryable else ""
            ),
            "output": {
                "status": status,
                "summary": summary,
                "claims": [_dump(claim) for claim in flat_claims],
                "sections": [_dump(section) for section in sections],
                "citations": [_dump(citation) for citation in citations],
                "highlights": [_dump(highlight) for highlight in highlights],
            },
            "trace": ["verify"],
        }

    return verify


def _check_claims(
    draft: dict[str, Any],
    shown: dict[str, SearchHit],
) -> tuple[dict[str, list[Claim]], list[str]]:
    legacy = {chunk_id: _legacy_chunk(hit.chunk) for chunk_id, hit in shown.items()}
    grouped: dict[str, list[Claim]] = {}
    dropped: list[str] = []
    rank = 1
    for section_key, raw_claim in _iter_claims(draft):
        text = str(raw_claim.get("text", "")).strip()
        if not text:
            continue
        kept: list[SourceCitation] = []
        for raw_cite in raw_claim.get("citations", []):
            chunk_id = str(raw_cite.get("chunk_id", ""))
            quote = str(raw_cite.get("quote", ""))
            checked = verify_citations([Citation(chunk_id=chunk_id, quote=quote)], legacy)
            if not checked.valid:
                if quote.strip():
                    dropped.append(quote.strip())
                continue
            hit = shown.get(chunk_id)
            if hit is None:
                dropped.append(quote.strip())
                continue
            kept.append(
                SourceCitation(
                    id=f"c{rank}",
                    rank=rank,
                    claim=text,
                    quote=quote,
                    doc_id=hit.chunk.doc_id,
                    chunk_id=chunk_id,
                    doc_type=hit.chunk.doc_type,
                    modality=hit.chunk.modality,
                    anchors=_anchors_for_quote(hit.chunk, quote),
                    image_path=hit.chunk.image_path,
                    verified=True,
                    fused_score=hit.fused_score,
                )
            )
            rank += 1
        if kept:
            grouped.setdefault(section_key, []).append(Claim(text=text, citations=kept))
    return grouped, dropped


def _iter_claims(draft: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if draft.get("kind") == "notes":
        pairs: list[tuple[str, dict[str, Any]]] = []
        for section in draft.get("sections", []):
            key = str(section.get("key", ""))
            claims = section.get("claims", [])
            if isinstance(claims, list):
                for claim in claims:
                    if isinstance(claim, dict):
                        pairs.append((key, claim))
        return pairs
    claims = draft.get("claims", [])
    if not isinstance(claims, list):
        return []
    return [("", claim) for claim in claims if isinstance(claim, dict)]


def _anchors_for_quote(chunk: StoredChunk, quote: str) -> list[SourceAnchor]:
    matched: list[SourceAnchor] = []
    for span in chunk.spans:
        if span.excerpt and _contains(quote, span.excerpt):
            anchor = span.to_anchor()
            if anchor is not None:
                matched.append(anchor)
    if matched:
        return _dedupe_anchors(matched)
    regional = [
        anchor
        for span in chunk.spans
        if span.anchor != "char_range" and (anchor := span.to_anchor()) is not None
    ]
    if regional:
        return _dedupe_anchors(regional)
    return _dedupe_anchors(
        [anchor for span in chunk.spans if (anchor := span.to_anchor()) is not None]
    )


def _dedupe_anchors(anchors: list[SourceAnchor]) -> list[SourceAnchor]:
    seen: set[str] = set()
    unique: list[SourceAnchor] = []
    for anchor in anchors:
        key = anchor.model_dump_json()
        if key in seen:
            continue
        seen.add(key)
        unique.append(anchor)
    return unique


def _contains(quote: str, text: str) -> bool:
    normalised = normalise_text(quote)
    return bool(normalised) and normalised in normalise_text(text)


def _rebuild_sections(
    specs: Sequence[NoteSectionSpec],
    grouped: dict[str, list[Claim]],
) -> list[NoteSection]:
    sections: list[NoteSection] = []
    seen: set[str] = set()
    for spec in specs:
        seen.add(spec.key)
        claims = grouped.get(spec.key, [])
        sections.append(
            NoteSection(
                key=spec.key,
                heading=spec.heading,
                body_md="\n".join(f"- {claim.text}" for claim in claims),
                claims=claims,
            )
        )
    for key, claims in grouped.items():
        if not key or key in seen:
            continue
        sections.append(
            NoteSection(
                key=key,
                heading=key,
                body_md="\n".join(f"- {claim.text}" for claim in claims),
                claims=claims,
            )
        )
    return sections


def _notes_context(state: AgentState, deps: PipelineDeps) -> list[SearchHit]:
    doc_ids = [str(doc.get("doc_id", "")) for doc in state["ingested"] if doc.get("doc_id")]
    fresh = deps.store.list_chunks(owner_id=state["owner_id"], doc_ids=doc_ids) if doc_ids else []
    query = state["message"] or state["title"] or "class notes"
    searched = _search(state, deps, query)
    ordered: list[SearchHit] = [SearchHit(chunk=chunk, fused_score=0.0) for chunk in fresh]
    seen = {chunk.id for chunk in fresh}
    for hit in searched:
        if hit.chunk.id not in seen:
            ordered.append(hit)
            seen.add(hit.chunk.id)
    return ordered[: deps.domain.retrieval.notes_context_k]


def _search(state: AgentState, deps: PipelineDeps, message: str) -> list[SearchHit]:
    query = _expand_query(message, deps.glossary) or state["title"] or "lecture"
    vectors = deps.embedder.embed([query], input_type="search_query")
    if len(vectors) != 1:
        raise RuntimeError("embedder returned the wrong number of vectors")
    retrieval = deps.domain.retrieval
    return deps.store.search(
        query,
        vectors[0],
        SearchParams(
            owner_id=state["owner_id"],
            lesson_id=state["lesson_id"],
            top_k=retrieval.top_k,
            rrf_k=retrieval.rrf_k,
            candidate_k=retrieval.candidate_k,
            highlight_boost=retrieval.highlight_boost,
            min_dense_score=retrieval.min_dense_score,
        ),
    )


def _expand_query(message: str, glossary: Sequence[GlossaryEntry]) -> str:
    if not glossary or not message.strip():
        return message
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", message)
    expanded = expand_terms([*tokens, message], glossary)
    extras = [term for term in expanded if term.casefold() not in message.casefold()]
    if not extras:
        return message
    return f"{message} {' '.join(extras)}"


def _figure_chunks(
    deps: PipelineDeps,
    pages: Sequence[PageContent],
    *,
    doc_id: str,
    owner_id: str,
    lesson_id: str,
    doc_type: DocType,
    title: str,
    rule_version: str,
    start_ordinal: int,
) -> list[StoredChunk]:
    chunks: list[StoredChunk] = []
    for page in pages:
        for index, figure in enumerate(page.figures, start=1):
            ext = figure.ext.lower().lstrip(".") or "png"
            dest = deps.figures_dir / doc_id / f"p{page.page}-{index}.{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if figure.image_bytes:
                dest.write_bytes(figure.image_bytes)
            caption = _vision_caption(deps, figure.image_bytes, ext)
            nearby = _nearby_caption(page, figure.bbox)
            lines = [line for line in (caption, nearby) if line]
            if not lines:
                lines = [f"Figure on page {page.page}."]
            text = "\n".join(lines)
            chunks.append(
                make_media_chunk(
                    doc_id=doc_id,
                    owner_id=owner_id,
                    lesson_id=lesson_id,
                    doc_type=doc_type,
                    ordinal=start_ordinal + len(chunks),
                    text=text,
                    kind="image_caption",
                    modality="image",
                    breadcrumb=title,
                    rule_version=rule_version,
                    page=page.page,
                    image_path=str(dest),
                    spans=[
                        ChunkSpan(
                            anchor="page_bbox",
                            page=page.page,
                            x0=figure.bbox.x0,
                            y0=figure.bbox.y0,
                            x1=figure.bbox.x1,
                            y1=figure.bbox.y1,
                            excerpt=text,
                        ),
                        ChunkSpan(anchor="char_range", c_start=0, c_end=len(text), excerpt=text),
                    ],
                )
            )
    return chunks


def _vision_caption(deps: PipelineDeps, image: bytes, ext: str) -> str:
    if not image:
        return ""
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    described = deps.vision.describe(image=image, mime_type=mime, prompt=_VISION_PROMPT)
    return described.strip()


def _nearby_caption(page: PageContent, bbox: BBox) -> str:
    image_mid = (bbox.y0 + bbox.y1) / 2
    candidates: list[tuple[float, str]] = []
    for line in page.lines:
        text = line.text.strip()
        if not _CAPTION_RE.match(text):
            continue
        line_mid = (line.bbox.y0 + line.bbox.y1) / 2
        candidates.append((abs(line_mid - image_mid), text))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _parse_qa(raw: str) -> dict[str, Any]:
    payload = parse_json_object(raw)
    if payload is None:
        return {**_empty_draft("qa"), "malformed": True}
    claims = _raw_claims(payload.get("claims"))
    if payload.get("status") == "insufficient_evidence" or not claims:
        return {"kind": "qa", "status": "insufficient_evidence", "summary": "", "claims": claims}
    return {
        "kind": "qa",
        "status": "answered",
        "summary": str(payload.get("summary", "")),
        "claims": claims,
    }


def _parse_notes(raw: str, specs: Sequence[NoteSectionSpec]) -> dict[str, Any]:
    payload = parse_json_object(raw)
    if payload is None:
        return {**_empty_draft("notes"), "malformed": True}
    by_key: dict[str, dict[str, Any]] = {}
    raw_sections = payload.get("sections")
    if isinstance(raw_sections, list):
        for section in raw_sections:
            if isinstance(section, dict) and section.get("key"):
                by_key[str(section["key"])] = section
    sections: list[dict[str, Any]] = []
    for spec in specs:
        source = by_key.get(spec.key, {})
        sections.append(
            {
                "key": spec.key,
                "heading": spec.heading,
                "body_md": str(source.get("body_md", "")),
                "claims": _raw_claims(source.get("claims")),
            }
        )
    return {"kind": "notes", "status": payload.get("status", "answered"), "sections": sections}


def _raw_claims(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    claims: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        citations = []
        for cite in item.get("citations", []):
            if isinstance(cite, dict) and cite.get("chunk_id") and cite.get("quote"):
                citations.append({"chunk_id": str(cite["chunk_id"]), "quote": str(cite["quote"])})
        claims.append({"text": text, "citations": citations})
    return claims


def _empty_draft(kind: str) -> dict[str, Any]:
    draft: dict[str, Any] = {
        "kind": kind,
        "status": "insufficient_evidence",
        "summary": "",
        "claims": [],
        "malformed": False,
    }
    if kind == "notes":
        draft["sections"] = []
    return draft


def _draft_had_citations(draft: dict[str, Any]) -> bool:
    for _key, claim in _iter_claims(draft):
        citations = claim.get("citations", [])
        if isinstance(citations, list) and citations:
            return True
    return False


def _legacy_chunk(chunk: StoredChunk) -> Chunk:
    kind: Literal["text", "image_caption"] = (
        "image_caption" if chunk.kind == "image_caption" else "text"
    )
    return Chunk(
        id=chunk.id,
        doc_id=chunk.doc_id,
        page=chunk.page or 0,
        text=chunk.text,
        kind=kind,
        emphasised=chunk.emphasised,
        image_path=chunk.image_path,
        rule_version=chunk.rule_version,
    )


def _embed_batched(deps: PipelineDeps, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start : start + _EMBED_BATCH]
        embedded = deps.embedder.embed(batch, input_type="search_document")
        if len(embedded) != len(batch):
            raise RuntimeError("embedder returned the wrong number of vectors")
        vectors.extend(embedded)
    return vectors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audio_doc_id(state: AgentState) -> str:
    if state["pdf_path"]:
        base = state["active_doc_id"] or state["doc_id"] or str(uuid.uuid4())
        return f"{base}-audio"
    return state["doc_id"] or str(uuid.uuid4())


def _audio_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix in (".m4a", ".mp4"):
        return "audio/mp4"
    return "application/octet-stream"


def _doc_type(value: str) -> DocType:
    if value in _DOC_TYPES:
        return cast(DocType, value)
    return "handout"


def _append_docs(
    existing: list[dict[str, Any]], document: IngestedDocument
) -> list[dict[str, Any]]:
    docs = [_load_doc(raw) for raw in existing]
    docs.append(document)
    return [_dump(doc) for doc in docs]


def _ingest_blurb(documents: Sequence[IngestedDocument]) -> str:
    chunks = sum(doc.chunk_count for doc in documents)
    figures = sum(doc.figure_count for doc in documents)
    docs = len(documents)
    return (
        f"Indexed {chunks} {_plural(chunks, 'chunk')} "
        f"({figures} {_plural(figures, 'figure')}) "
        f"from {docs} {_plural(docs, 'document')}."
    )


def _plural(count: int, noun: str) -> str:
    return f"{noun}s" if count != 1 else noun


def _dump(model: Any) -> dict[str, Any]:
    dumped: dict[str, Any] = model.model_dump(mode="json")
    return dumped


def _load_chunk(raw: dict[str, Any]) -> StoredChunk:
    return StoredChunk.model_validate(raw)


def _load_doc(raw: dict[str, Any]) -> IngestedDocument:
    return IngestedDocument.model_validate(raw)


def _load_hit(raw: dict[str, Any]) -> SearchHit:
    return SearchHit.model_validate(raw)
