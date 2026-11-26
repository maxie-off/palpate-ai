"""Dependencies the graph closes over. Tests pass fakes; ``build_deps`` wires real adapters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.agent.store import KnowledgeStore
from app.core.config import DomainConfig
from app.core.llm import LLMClient
from app.core.providers import SpeechToTextClient, VisionClient
from app.ingestion.parser import DocumentParser, PyMuPDFParser
from app.retrieval.glossary import GlossaryEntry


class Embedder(Protocol):
    def embed(
        self, texts: list[str], *, input_type: str = "search_document"
    ) -> list[list[float]]: ...


@dataclass
class PipelineDeps:
    llm: LLMClient
    embedder: Embedder
    vision: VisionClient
    speech: SpeechToTextClient
    store: KnowledgeStore
    domain: DomainConfig
    figures_dir: Path
    answer_model_id: str
    parser: DocumentParser = field(default_factory=PyMuPDFParser)
    glossary: tuple[GlossaryEntry, ...] = ()
    max_verify_attempts: int = 2
