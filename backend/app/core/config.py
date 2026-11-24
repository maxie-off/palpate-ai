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
    min_evidence_score: float = 0.35


class ParsingConfig(BaseModel):
    """How teacher emphasis is read from a PDF (colour and bold, besides highlights)."""

    min_color_saturation: float = 0.25  # text coloured at least this much counts as emphasised
    emphasis_bold: bool = False  # black bold is usually just a heading; opt in per domain


class ChunkingConfig(BaseModel):
    max_chars: int = 900
    rule_version: str = "v1"
    emphasis_threshold: float = 0.3  # share of emphasised characters that flags a chunk


class DomainConfig(BaseModel):
    domain: str = "dentistry"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PALPATE_", extra="ignore")

    database_url: str = "postgresql://palpate:palpate@localhost:5432/palpate"
    aws_region: str = "us-east-1"
    answer_model_id: str = ""
    bulk_model_id: str = ""
    embedding_model_id: str = "cohere.embed-multilingual-v3"
    config_path: Path = REPO_ROOT / "config" / "dentistry.yaml"


def load_domain_config(path: Path) -> DomainConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return DomainConfig.model_validate(raw)


@lru_cache
def get_settings() -> Settings:
    return Settings()
