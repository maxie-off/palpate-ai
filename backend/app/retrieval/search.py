"""Retriever interface. The implementation plugs in behind it."""

from typing import Protocol

from app.models.schemas import ScoredChunk


class Retriever(Protocol):
    def retrieve(self, queries: list[str], top_k: int) -> list[ScoredChunk]: ...


class PgVectorRetriever:
    """Hybrid vector + BM25 retrieval over Postgres.

    TODO(stage 1): vector search via pgvector.
    TODO(stage 2): jieba BM25, highlight boost, reranking.
    """

    def retrieve(self, queries: list[str], top_k: int) -> list[ScoredChunk]:
        raise NotImplementedError
