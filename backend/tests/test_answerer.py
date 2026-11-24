import json

from app.answering.answerer import Answerer
from app.core.config import RetrievalConfig
from app.models.schemas import Chunk, ChunkSpan, ParsedQuestion, ScoredChunk


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, *, system: str, user: str, model_id: str, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self.reply

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


QUESTION = ParsedQuestion(id="q1", stem="Which lesion is Epstein-like?", options=["A", "B"])
EVIDENCE = [
    ScoredChunk(
        chunk=Chunk(
            id="c1",
            document_id="d1",
            spans=[ChunkSpan(page=3, bbox=(0, 0, 1, 1))],
            body="Milia are 1-3 mm keratin cysts",
        ),
        score=0.9,
    )
]


def _answerer(reply: str) -> tuple[Answerer, FakeLLM]:
    llm = FakeLLM(reply)
    return Answerer(llm, "model-x", RetrievalConfig(min_evidence_score=0.35)), llm


def test_answers_with_verified_citation() -> None:
    reply = json.dumps(
        {
            "status": "answered",
            "summary": "Milia are small keratin cysts.",
            "citations": [{"chunk_id": "c1", "quote": "1-3 mm keratin cysts"}],
        }
    )
    answerer, _ = _answerer(reply)
    result = answerer.answer(QUESTION, EVIDENCE)
    assert result.status == "answered"
    assert result.citations[0].page == 3


def test_hallucinated_quote_leads_to_abstention() -> None:
    reply = json.dumps(
        {
            "status": "answered",
            "summary": "x",
            "citations": [{"chunk_id": "c1", "quote": "invented text"}],
        }
    )
    answerer, _ = _answerer(reply)
    assert answerer.answer(QUESTION, EVIDENCE).status == "insufficient_evidence"


def test_low_scores_skip_the_llm_call() -> None:
    answerer, llm = _answerer("{}")
    weak = [ScoredChunk(chunk=EVIDENCE[0].chunk, score=0.1)]
    assert answerer.answer(QUESTION, weak).status == "insufficient_evidence"
    assert llm.calls == 0


def test_malformed_json_leads_to_abstention() -> None:
    answerer, _ = _answerer("not json at all")
    assert answerer.answer(QUESTION, EVIDENCE).status == "insufficient_evidence"


def test_model_can_abstain_itself() -> None:
    answerer, _ = _answerer('```json\n{"status": "insufficient_evidence"}\n```')
    assert answerer.answer(QUESTION, EVIDENCE).status == "insufficient_evidence"
