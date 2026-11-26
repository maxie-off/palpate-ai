"""Run one request through the compiled graph."""

from typing import Any, cast

from app.agent.deps import PipelineDeps
from app.agent.graph import build_graph
from app.agent.state import AgentState, initial_state
from app.agent.store import InMemoryKnowledgeStore, KnowledgeStore, PgVectorKnowledgeStore
from app.agent.types import (
    AgentRequest,
    AgentResult,
    Claim,
    Highlight,
    IngestedDocument,
    IngestSummary,
    NoteSection,
    RunStatus,
    SourceCitation,
)
from app.core.config import Settings, get_settings, load_domain_config
from app.core.db import connect
from app.core.llm import BedrockLLMClient
from app.core.providers import BedrockVisionClient, NullVision, VisionClient, build_speech_client

_STATUSES: set[str] = {
    "ok",
    "answered",
    "insufficient_evidence",
    "not_implemented",
    "failed",
}


def build_deps(settings: Settings | None = None) -> PipelineDeps:
    """Adapters from environment.

    Speech stays unconfigured until ``PALPATE_STT_PROVIDER`` is set.
    """
    settings = settings or get_settings()
    domain = load_domain_config(settings.config_path)
    llm = BedrockLLMClient(settings.aws_region, settings.embedding_model_id)
    store = _build_store(settings)
    vision: VisionClient
    if settings.vision_model_id:
        vision = BedrockVisionClient(llm, settings.vision_model_id)
    else:
        vision = NullVision()
    return PipelineDeps(
        llm=llm,
        embedder=llm,
        vision=vision,
        speech=build_speech_client(
            settings.stt_provider,
            region=settings.aws_region,
            model_id=settings.stt_model_id,
        ),
        store=store,
        domain=domain,
        figures_dir=settings.figures_dir,
        answer_model_id=settings.answer_model_id or settings.bulk_model_id,
    )


def run_agent(request: AgentRequest, deps: PipelineDeps | None = None) -> AgentResult:
    """Execute one task. Pass ``deps`` in tests; otherwise settings choose the providers."""
    bound = deps or build_deps()
    graph = build_graph(bound)
    state = initial_state(request, max_attempts=bound.max_verify_attempts)
    final = cast(AgentState, graph.invoke(state))
    return result_from_state(final)


def result_from_state(state: AgentState) -> AgentResult:
    output: dict[str, Any] = state["output"] or {}
    documents = [IngestedDocument.model_validate(raw) for raw in state["ingested"]]
    return AgentResult(
        task=state["task"],
        status=_status(state, output),
        error=state["error"],
        summary=str(output.get("summary") or ""),
        claims=[Claim.model_validate(raw) for raw in output.get("claims", [])],
        sections=[NoteSection.model_validate(raw) for raw in output.get("sections", [])],
        citations=[SourceCitation.model_validate(raw) for raw in output.get("citations", [])],
        highlights=[Highlight.model_validate(raw) for raw in output.get("highlights", [])],
        ingest=IngestSummary(documents=documents) if documents else None,
        attempts=state["attempts"],
        trace=list(state["trace"]),
    )


def _build_store(settings: Settings) -> KnowledgeStore:
    if settings.vector_store == "memory":
        return InMemoryKnowledgeStore()
    if settings.vector_store == "pgvector":
        return PgVectorKnowledgeStore(
            connect(settings.database_url),
            embedding_dim=settings.embedding_dim,
        )
    raise ValueError(
        f"Unknown PALPATE_VECTOR_STORE {settings.vector_store!r}. Expected 'pgvector' or 'memory'."
    )


def _status(state: AgentState, output: dict[str, Any]) -> RunStatus:
    if state["status"] == "failed":
        return "failed"
    raw = str(output.get("status") or state["status"])
    if raw in _STATUSES:
        return cast(RunStatus, raw)
    return "failed"
