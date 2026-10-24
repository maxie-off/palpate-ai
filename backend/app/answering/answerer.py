"""Grounded answering: retrieve evidence in, verified citations out, abstain when unsure."""

import json
from collections.abc import Sequence
from typing import Any

from app.answering.prompts import SYSTEM_PROMPT, build_user_prompt
from app.answering.verifier import verify_citations
from app.core.config import RetrievalConfig
from app.core.llm import LLMClient
from app.models.schemas import Citation, ParsedQuestion, QuestionAnswer, ScoredChunk


def _parse_json(raw: str) -> dict[str, Any] | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


class Answerer:
    def __init__(self, llm: LLMClient, model_id: str, config: RetrievalConfig) -> None:
        self._llm = llm
        self._model_id = model_id
        self._config = config

    def answer(self, question: ParsedQuestion, evidence: Sequence[ScoredChunk]) -> QuestionAnswer:
        insufficient = QuestionAnswer(question_id=question.id, status="insufficient_evidence")

        usable = [e for e in evidence if e.score >= self._config.min_evidence_score]
        if not usable:
            return insufficient

        raw = self._llm.complete(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(question, usable),
            model_id=self._model_id,
            max_tokens=1024,
        )
        payload = _parse_json(raw)
        if payload is None or payload.get("status") == "insufficient_evidence":
            return insufficient

        citations = [
            Citation(chunk_id=str(c["chunk_id"]), quote=str(c["quote"]))
            for c in payload.get("citations", [])
            if isinstance(c, dict) and "chunk_id" in c and "quote" in c
        ]
        result = verify_citations(citations, {e.chunk.id: e.chunk for e in usable})
        if not result.valid:
            return insufficient

        return QuestionAnswer(
            question_id=question.id,
            status="answered",
            summary=str(payload.get("summary", "")),
            citations=result.valid,
        )
