"""Agent graph: routing, ingest, grounded answers, notes, and the sample fixture."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.agent.demo import run_sample, write_silent_wav
from app.agent.deps import PipelineDeps
from app.agent.graph import build_graph
from app.agent.sample_pdf import build_sample_lecture
from app.agent.service import run_agent
from app.agent.standins import (
    HashingEmbedder,
    SequenceLLM,
    SourceCopyLLM,
    StaticSpeech,
    StaticVision,
)
from app.agent.store import InMemoryKnowledgeStore
from app.agent.types import AgentRequest
from app.core.config import REPO_ROOT, load_domain_config
from app.core.providers import UnconfiguredSpeechToText
from app.ingestion.parser import PyMuPDFParser

FIXTURE = Path(__file__).parent / "fixtures" / "sample_lecture.pdf"


def make_deps(tmp_path: Path, **overrides: Any) -> PipelineDeps:
    return PipelineDeps(
        llm=overrides.pop("llm", SourceCopyLLM()),
        embedder=overrides.pop("embedder", HashingEmbedder()),
        vision=overrides.pop("vision", StaticVision()),
        speech=overrides.pop("speech", StaticSpeech()),
        store=overrides.pop("store", InMemoryKnowledgeStore()),
        domain=overrides.pop("domain", load_domain_config(REPO_ROOT / "config" / "dentistry.yaml")),
        figures_dir=tmp_path / "figures",
        answer_model_id="test-model",
        glossary=overrides.pop("glossary", ()),
        **overrides,
    )


def test_graph_has_a_branch_per_task(tmp_path: Path) -> None:
    graph = build_graph(make_deps(tmp_path))
    assert {
        "route",
        "ingest_pdf",
        "transcribe_audio",
        "index_chunks",
        "retrieve",
        "draft_answer",
        "draft_notes",
        "verify",
        "quiz",
        "summarize",
    } <= set(graph.nodes)


def test_message_routes_without_a_model_call(tmp_path: Path) -> None:
    llm = SourceCopyLLM()
    deps = make_deps(tmp_path, llm=llm)
    quiz = run_agent(AgentRequest(message="please make a quiz on cysts"), deps)
    assert quiz.task == "quiz"
    assert quiz.status == "not_implemented"
    assert quiz.trace == ["route", "quiz"]

    summary = run_agent(AgentRequest(message="summarize the lecture"), deps)
    assert summary.task == "summarize"
    assert summary.status == "not_implemented"
    assert llm.calls == 0


def test_empty_corpus_abstains_without_calling_the_model(tmp_path: Path) -> None:
    llm = SourceCopyLLM()
    result = run_agent(
        AgentRequest(task="qa", message="What are milia?"),
        make_deps(tmp_path, llm=llm),
    )
    assert result.status == "insufficient_evidence"
    assert result.trace == ["route", "retrieve", "draft_answer", "verify"]
    assert llm.calls == 0
    assert result.citations == []


def test_missing_pdf_stops_before_transcription(tmp_path: Path) -> None:
    result = run_agent(
        AgentRequest(task="ingest", pdf_path=str(tmp_path / "missing.pdf")),
        make_deps(tmp_path),
    )
    assert result.status == "failed"
    assert "PDF not found" in result.error
    assert result.trace == ["route", "ingest_pdf"]


def test_ingest_extracts_text_figure_and_highlight(tmp_path: Path) -> None:
    pdf = tmp_path / "lecture.pdf"
    build_sample_lecture(pdf)
    vision = StaticVision("Small pale papules on the cheek.")
    embedder = HashingEmbedder()
    store = InMemoryKnowledgeStore()
    deps = make_deps(tmp_path, vision=vision, embedder=embedder, store=store)
    result = run_agent(
        AgentRequest(
            task="ingest",
            pdf_path=str(pdf),
            doc_id="lecture",
            title="Demo lecture: oral cysts",
            owner_id="student",
            lesson_id="L3",
            doc_type="slides",
        ),
        deps,
    )
    assert result.status == "ok"
    assert result.trace == ["route", "ingest_pdf", "transcribe_audio", "index_chunks"]
    assert result.ingest is not None
    document = result.ingest.documents[0]
    assert document.doc_id == "lecture"
    assert document.page_count == 2
    assert document.figure_count == 1
    assert document.cached is False
    assert vision.calls == 1
    assert embedder.calls == 1

    chunks = store.list_chunks(owner_id="student", lesson_id="L3")
    assert any(chunk.emphasised and chunk.modality == "text" for chunk in chunks)
    text = next(chunk for chunk in chunks if chunk.modality == "text")
    assert any(span.anchor == "page_bbox" and span.page == 1 for span in text.spans)
    assert any(span.anchor == "char_range" for span in text.spans)
    image = next(chunk for chunk in chunks if chunk.modality == "image")
    assert image.image_path
    assert Path(image.image_path).is_file()
    box = next(span for span in image.spans if span.anchor == "page_bbox")
    assert box.x0 == pytest.approx(72)
    assert box.x0 is not None and box.x1 is not None and box.x0 < box.x1
    assert box.y0 is not None and box.y1 is not None and box.y0 < box.y1

    again = run_agent(
        AgentRequest(task="ingest", pdf_path=str(pdf), doc_id="other-id", owner_id="student"),
        deps,
    )
    assert again.ingest is not None
    assert again.ingest.documents[0].cached is True
    assert again.ingest.documents[0].doc_id == "lecture"
    assert vision.calls == 1
    assert embedder.calls == 1


def test_same_file_for_another_owner_is_a_separate_copy(tmp_path: Path) -> None:
    pdf = tmp_path / "lecture.pdf"
    build_sample_lecture(pdf)
    deps = make_deps(tmp_path)
    first = run_agent(AgentRequest(task="ingest", pdf_path=str(pdf), owner_id="a"), deps)
    second = run_agent(AgentRequest(task="ingest", pdf_path=str(pdf), owner_id="b"), deps)
    assert first.ingest is not None and second.ingest is not None
    assert first.ingest.documents[0].cached is False
    assert second.ingest.documents[0].cached is False
    assert first.ingest.documents[0].doc_id != second.ingest.documents[0].doc_id
    assert deps.store.list_chunks(owner_id="a")
    assert deps.store.list_chunks(owner_id="b")


def test_verified_answer_carries_a_page_box(tmp_path: Path) -> None:
    pdf = tmp_path / "lecture.pdf"
    build_sample_lecture(pdf)
    deps = make_deps(tmp_path)
    run_agent(
        AgentRequest(task="ingest", pdf_path=str(pdf), doc_id="lecture", owner_id="student"),
        deps,
    )
    answer = run_agent(
        AgentRequest(task="qa", message="What are milia?", owner_id="student"),
        deps,
    )
    assert answer.status == "answered"
    assert answer.trace == ["route", "retrieve", "draft_answer", "verify"]
    assert answer.citations
    assert any(
        anchor.kind == "page_bbox" and anchor.page == 1
        for citation in answer.citations
        for anchor in citation.anchors
    )
    milia = next(citation for citation in answer.citations if "keratin-filled" in citation.quote)
    assert milia.verified
    assert milia.doc_id == "lecture"
    assert any(highlight.quote == milia.quote for highlight in answer.highlights)


def test_hallucinated_quote_is_retried_once_then_accepted(tmp_path: Path) -> None:
    from app.agent.store import with_search_text
    from app.agent.types import ChunkSpan, IngestedDocument, StoredChunk

    embedder = HashingEmbedder()
    chunk = with_search_text(
        StoredChunk(
            id="d1-0000",
            doc_id="d1",
            owner_id="local",
            text="Milia are keratin cysts",
            page=3,
            embedding=embedder.embed(["Milia are keratin cysts"])[0],
            spans=[
                ChunkSpan(
                    anchor="page_bbox",
                    page=3,
                    x0=1,
                    y0=2,
                    x1=8,
                    y1=9,
                    excerpt="Milia are keratin cysts",
                )
            ],
        )
    )
    store = InMemoryKnowledgeStore()
    store.upsert(
        IngestedDocument(doc_id="d1", owner_id="local", title="notes", content_hash="h1"),
        [chunk],
    )
    llm = SequenceLLM(
        [
            json.dumps(
                {
                    "status": "answered",
                    "summary": "invented",
                    "claims": [
                        {
                            "text": "nope",
                            "citations": [{"chunk_id": "d1-0000", "quote": "invented text"}],
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "status": "answered",
                    "summary": "Milia are keratin cysts.",
                    "claims": [
                        {
                            "text": "Milia are keratin cysts.",
                            "citations": [{"chunk_id": "d1-0000", "quote": "keratin cysts"}],
                        }
                    ],
                }
            ),
        ]
    )
    result = run_agent(
        AgentRequest(task="qa", message="What are milia?"),
        make_deps(tmp_path, llm=llm, embedder=embedder, store=store),
    )
    assert llm.calls == 2
    assert result.status == "answered"
    assert result.attempts == 2
    assert result.trace.count("verify") == 2
    assert result.citations[0].quote == "keratin cysts"
    box = next(anchor for anchor in result.citations[0].anchors if anchor.kind == "page_bbox")
    assert box.page == 3


def test_explicit_abstention_is_not_retried(tmp_path: Path) -> None:
    from app.agent.store import with_search_text
    from app.agent.types import IngestedDocument, StoredChunk

    embedder = HashingEmbedder()
    store = InMemoryKnowledgeStore()
    store.upsert(
        IngestedDocument(doc_id="d1", owner_id="local", title="notes", content_hash="h1"),
        [
            with_search_text(
                StoredChunk(
                    id="d1-0000",
                    doc_id="d1",
                    owner_id="local",
                    text="Milia are keratin cysts",
                    page=1,
                    embedding=embedder.embed(["Milia are keratin cysts"])[0],
                )
            )
        ],
    )
    llm = SequenceLLM(['{"status": "insufficient_evidence", "claims": []}'])
    result = run_agent(
        AgentRequest(task="qa", message="What are milia?"),
        make_deps(tmp_path, llm=llm, embedder=embedder, store=store),
    )
    assert result.status == "insufficient_evidence"
    assert llm.calls == 1


def test_malformed_json_retries_then_abstains(tmp_path: Path) -> None:
    from app.agent.store import with_search_text
    from app.agent.types import IngestedDocument, StoredChunk

    embedder = HashingEmbedder()
    store = InMemoryKnowledgeStore()
    store.upsert(
        IngestedDocument(doc_id="d1", owner_id="local", title="notes", content_hash="h1"),
        [
            with_search_text(
                StoredChunk(
                    id="d1-0000",
                    doc_id="d1",
                    owner_id="local",
                    text="Milia are keratin cysts",
                    page=1,
                    embedding=embedder.embed(["Milia are keratin cysts"])[0],
                )
            )
        ],
    )
    llm = SequenceLLM(["not json at all"])
    result = run_agent(
        AgentRequest(task="qa", message="What are milia?"),
        make_deps(tmp_path, llm=llm, embedder=embedder, store=store),
    )
    assert result.status == "insufficient_evidence"
    assert llm.calls == 2


def test_notes_cite_the_pdf_region_and_the_recording(tmp_path: Path) -> None:
    pdf = tmp_path / "lecture.pdf"
    wav = tmp_path / "lecture.wav"
    build_sample_lecture(pdf)
    write_silent_wav(wav)
    result = run_agent(
        AgentRequest(
            task="draft_notes",
            message="Draft class notes about milia",
            pdf_path=str(pdf),
            audio_path=str(wav),
            doc_id="lecture",
            title="Demo lecture: oral cysts",
            lesson_id="L3",
        ),
        make_deps(tmp_path),
    )
    assert result.status == "answered"
    assert result.trace[:6] == [
        "route",
        "ingest_pdf",
        "transcribe_audio",
        "index_chunks",
        "retrieve",
        "draft_notes",
    ]
    assert [section.key for section in result.sections] == [
        "overview",
        "key_concepts",
        "clinical",
        "figures",
    ]
    assert result.sections[0].claims
    modalities = {citation.modality for citation in result.citations}
    assert {"text", "image", "audio"} <= modalities
    audio = next(citation for citation in result.citations if citation.modality == "audio")
    timing = next(anchor for anchor in audio.anchors if anchor.kind == "time_range")
    assert (timing.t_start, timing.t_end) == (pytest.approx(12.5), pytest.approx(18.0))
    assert result.ingest is not None
    by_id = {document.doc_id: document for document in result.ingest.documents}
    assert by_id["lecture"].figure_count == 1
    assert by_id["lecture-audio"].doc_type == "video"


def test_notes_fail_when_speech_is_not_configured(tmp_path: Path) -> None:
    wav = tmp_path / "lecture.wav"
    write_silent_wav(wav)
    result = run_agent(
        AgentRequest(task="draft_notes", audio_path=str(wav), message="Draft class notes"),
        make_deps(tmp_path, speech=UnconfiguredSpeechToText()),
    )
    assert result.status == "failed"
    assert "PALPATE_STT_PROVIDER" in result.error


def test_fixture_matches_the_builder_and_the_sample_run(tmp_path: Path) -> None:
    built = tmp_path / "built.pdf"
    build_sample_lecture(built)
    assert _page_texts(built) == _page_texts(FIXTURE)

    wav = tmp_path / "lecture.wav"
    write_silent_wav(wav)
    results = run_sample(
        FIXTURE,
        question="What are milia?",
        audio_path=wav,
        figures_dir=tmp_path / "figures",
    )
    assert results["ingest"].status == "ok"
    assert results["ingest"].ingest is not None
    assert results["ingest"].ingest.documents[0].figure_count == 1
    assert results["qa"].status == "answered"
    assert any(
        anchor.kind == "page_bbox" for cite in results["qa"].citations for anchor in cite.anchors
    )
    assert results["notes"].status == "answered"
    kinds = {anchor.kind for cite in results["notes"].citations for anchor in cite.anchors}
    assert {"page_bbox", "time_range"} <= kinds


def _page_texts(path: Path) -> list[list[str]]:
    return [[line.text for line in page.lines] for page in PyMuPDFParser().parse(path)]
