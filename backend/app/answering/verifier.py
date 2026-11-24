"""Deterministic citation check: a cited quote must really exist in the cited chunk.

The model can hallucinate a quote or a chunk ID. This check does not trust it. Comparison ignores
whitespace and width/case differences, because PDF extraction often inserts or drops spaces
between Chinese and English text.
"""

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from app.models.schemas import Chunk, Citation


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


@dataclass(frozen=True)
class VerificationResult:
    valid: list[Citation]
    dropped: list[Citation]


def verify_citations(citations: list[Citation], chunks: Mapping[str, Chunk]) -> VerificationResult:
    valid: list[Citation] = []
    dropped: list[Citation] = []
    for citation in citations:
        chunk = chunks.get(citation.chunk_id)
        quote = _normalise(citation.quote)
        if chunk is not None and quote and quote in _normalise(chunk.body):
            valid.append(
                citation.model_copy(update={"document_id": chunk.document_id, "page": chunk.page})
            )
        else:
            dropped.append(citation)
    return VerificationResult(valid=valid, dropped=dropped)
