"""Retriever interface. The implementation plugs in behind it."""

from typing import Protocol

from app.models.schemas import ScoredChunk


class Retriever(Protocol):
    def retrieve(self, queries: list[str], top_k: int) -> list[ScoredChunk]: ...


class PgVectorRetriever:
    """Legacy entry point. The agent pipeline retrieves through ``app.agent.store``.

    ``KnowledgeStore.search`` is hybrid dense + lexical fused with reciprocal rank
    fusion, backed by Postgres pgvector when ``PALPATE_VECTOR_STORE=pgvector``.
    This class remains so older call sites keep a stable import; it is not the
    path the LangGraph workflow runs.
    """

    def retrieve(self, queries: list[str], top_k: int) -> list[ScoredChunk]:
        raise NotImplementedError(
            "Use app.agent.store.KnowledgeStore.search via the agent pipeline."
        )
