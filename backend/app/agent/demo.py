"""Offline end-to-end sample: ingest the synthetic PDF, answer, and draft notes.

Run from ``backend/``::

    uv run python examples/sample_run.py

No AWS credentials and no database. Embeddings are the hashing stand-in, the "model"
copies source lines, and speech returns a fixed transcript.
"""

import wave
from pathlib import Path

from app.agent.deps import PipelineDeps
from app.agent.service import run_agent
from app.agent.standins import HashingEmbedder, SourceCopyLLM, StaticSpeech, StaticVision
from app.agent.store import InMemoryKnowledgeStore
from app.agent.types import AgentRequest, AgentResult
from app.core.config import REPO_ROOT, load_domain_config


def write_silent_wav(path: Path, duration_s: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(8000 * duration_s)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * frames)


def run_sample(
    pdf_path: Path,
    *,
    question: str,
    audio_path: Path | None = None,
    figures_dir: Path,
) -> dict[str, AgentResult]:
    """Ingest ``pdf_path``, answer ``question``, and draft notes when ``audio_path`` is set."""
    deps = PipelineDeps(
        llm=SourceCopyLLM(),
        embedder=HashingEmbedder(),
        vision=StaticVision("Small pale papules on the cheek."),
        speech=StaticSpeech(),
        store=InMemoryKnowledgeStore(),
        domain=load_domain_config(REPO_ROOT / "config" / "dentistry.yaml"),
        figures_dir=figures_dir,
        answer_model_id="offline-copy",
    )
    owner = "demo"
    lesson = "demo-lesson"
    results: dict[str, AgentResult] = {}
    results["ingest"] = run_agent(
        AgentRequest(
            task="ingest",
            pdf_path=str(pdf_path),
            doc_id="demo-lecture",
            title="Demo lecture: oral cysts",
            owner_id=owner,
            lesson_id=lesson,
            doc_type="slides",
        ),
        deps,
    )
    results["qa"] = run_agent(
        AgentRequest(
            task="qa",
            message=question,
            owner_id=owner,
            lesson_id=lesson,
        ),
        deps,
    )
    if audio_path is not None:
        results["notes"] = run_agent(
            AgentRequest(
                task="draft_notes",
                message="Draft class notes about milia",
                audio_path=str(audio_path),
                owner_id=owner,
                lesson_id=lesson,
                title="Demo lecture: oral cysts",
            ),
            deps,
        )
    return results
