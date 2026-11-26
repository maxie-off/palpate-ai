"""Environment settings (secrets, endpoints) and tunable domain config (YAML in git)."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class RetrievalConfig(BaseModel):
    top_k: int = 8
    rerank: bool = False
    highlight_boost: float = 1.2
    # Used by the legacy Answerer, which treats scores as roughly cosine-like.
    # The agent graph does not compare reciprocal-rank-fusion scores against this cutoff.
    min_evidence_score: float = 0.35
    rrf_k: int = 60
    candidate_k: int = 50
    # 0 disables the dense-score gate. Lexical-only hits are kept either way, because
    # identifiers such as gene names are exactly what vector search tends to miss.
    min_dense_score: float = 0.0
    notes_context_k: int = 20


class ChunkingConfig(BaseModel):
    max_chars: int = 900
    rule_version: str = "v1"


class NoteSectionSpec(BaseModel):
    key: str
    heading: str
    instruction: str


def _default_note_sections() -> list[NoteSectionSpec]:
    return [
        NoteSectionSpec(
            key="overview",
            heading="Overview",
            instruction="What this lecture covered, in a short paragraph.",
        ),
        NoteSectionSpec(
            key="key_concepts",
            heading="Key concepts",
            instruction="Named lesions, structures, and definitions the class introduced.",
        ),
        NoteSectionSpec(
            key="clinical",
            heading="Clinical points",
            instruction="Presentation, treatment, and points the lecturer emphasised.",
        ),
        NoteSectionSpec(
            key="figures",
            heading="Figures",
            instruction="What each figure shows. Cite the image region.",
        ),
    ]


class NotesConfig(BaseModel):
    rules_md: str = (
        "Write in the language of the sources. Keep technical terms in the language the "
        "source uses. Every claim must quote a source span. If the sources do not cover a "
        "section, leave that section empty rather than filling it from outside knowledge."
    )
    sections: list[NoteSectionSpec] = Field(default_factory=_default_note_sections)


class DomainConfig(BaseModel):
    domain: str = "dentistry"
    locale: str = "zh-TW"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PALPATE_", extra="ignore")

    database_url: str = "postgresql://palpate:palpate@localhost:5432/palpate"
    aws_region: str = "us-east-1"
    answer_model_id: str = ""
    bulk_model_id: str = ""
    embedding_model_id: str = "cohere.embed-multilingual-v3"
    # Cohere Embed Multilingual v3. Must match the vector(1024) column. Changing it
    # means re-embedding every chunk and altering the column.
    embedding_dim: int = 1024
    vision_model_id: str = ""
    # "" or "none" leaves speech unconfigured. "transcribe" selects the Amazon
    # Transcribe adapter, which is stubbed until a bucket workflow is wired.
    stt_provider: str = ""
    stt_model_id: str = ""
    # "pgvector" (Postgres) or "memory" (process-local, for tests and the offline sample).
    vector_store: str = "pgvector"
    figures_dir: Path = REPO_ROOT / "data" / "figures"
    config_path: Path = REPO_ROOT / "config" / "dentistry.yaml"


def load_domain_config(path: Path) -> DomainConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return DomainConfig.model_validate(raw)


@lru_cache
def get_settings() -> Settings:
    return Settings()
