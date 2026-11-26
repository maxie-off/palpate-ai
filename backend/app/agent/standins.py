"""Offline stand-ins for the sample script and unit tests.

These are not model providers. HashingEmbedder is a bag-of-tokens vector so retrieval tests
can run without Bedrock. SourceCopyLLM quotes the source blocks it was shown, which is how
the sample demonstrates citation checking without calling a model.
"""

import hashlib
import json
import math
import re
from typing import Any

from app.agent.lexical import search_tokens
from app.models.schemas import Transcript, TranscriptSegment


class HashingEmbedder:
    """Deterministic bag-of-tokens embedding. Shared tokens land in the same dimensions."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str], *, input_type: str = "search_document") -> list[list[float]]:
        self.calls += 1
        return [_embed_one(text, self.dim) for text in texts]


def _embed_one(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in search_tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % dim] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class SourceCopyLLM:
    """Drafts claims by copying the first line of each source block in the prompt."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, user: str, model_id: str, max_tokens: int = 1024) -> str:
        self.calls += 1
        chunks = _parse_source_blocks(user)
        if not chunks:
            return json.dumps(
                {"status": "insufficient_evidence", "summary": "", "claims": [], "sections": []}
            )
        claims: list[dict[str, Any]] = [
            {"text": quote, "citations": [{"chunk_id": chunk_id, "quote": quote}]}
            for chunk_id, quote in chunks
        ]
        if user.startswith("TASK: notes"):
            sections = _sections_from_prompt(user, claims)
            return json.dumps({"status": "answered", "sections": sections})
        return json.dumps({"status": "answered", "summary": claims[0]["text"], "claims": claims})

    def embed(self, texts: list[str], *, input_type: str = "search_document") -> list[list[float]]:
        return HashingEmbedder().embed(texts, input_type=input_type)


class SequenceLLM:
    """Returns scripted replies, one per ``complete`` call. The last reply repeats."""

    def __init__(self, replies: list[str]) -> None:
        if not replies:
            raise ValueError("SequenceLLM needs at least one reply")
        self._replies = replies
        self.calls = 0

    def complete(self, *, system: str, user: str, model_id: str, max_tokens: int = 1024) -> str:
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return reply

    def embed(self, texts: list[str], *, input_type: str = "search_document") -> list[list[float]]:
        return HashingEmbedder().embed(texts, input_type=input_type)


class StaticVision:
    def __init__(self, caption: str = "Small pale papules on the cheek.") -> None:
        self.caption = caption
        self.calls = 0

    def describe(self, *, image: bytes, mime_type: str, prompt: str) -> str:
        self.calls += 1
        return self.caption


class StaticSpeech:
    def __init__(
        self,
        segments: list[tuple[str, float, float]] | None = None,
        *,
        model: str = "fake-stt",
    ) -> None:
        self._segments = segments or [
            (
                "The lecturer said milia are keratin-filled cysts common in newborns.",
                12.5,
                18.0,
            )
        ]
        self.model = model
        self.calls = 0

    def transcribe(self, audio_path: object, *, language: str | None = None) -> Transcript:
        self.calls += 1
        return Transcript(
            model=self.model,
            language=language or "",
            segments=[
                TranscriptSegment(text=text, t_start=start, t_end=end)
                for text, start, end in self._segments
            ],
        )


class ZeroEmbedder:
    """Every text maps to the origin, so only lexical overlap can rank a hit."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed(self, texts: list[str], *, input_type: str = "search_document") -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


_BLOCK_RE = re.compile(
    r"<<<CHUNK id=(?P<id>\S+) [^\n]*>>>\n(?P<body>.*?)\n<<<END>>>",
    re.DOTALL,
)
_SECTION_RE = re.compile(r"^SECTION key=(\S+) heading=(.*)$", re.MULTILINE)


def _parse_source_blocks(user: str) -> list[tuple[str, str]]:
    question = user
    for label in ("Question:", "Request:"):
        if label in user:
            question = user.split(label, 1)[1]
            break
    wanted = set(search_tokens(question))
    found: list[tuple[str, str]] = []
    for match in _BLOCK_RE.finditer(user):
        lines = [line.strip() for line in match.group("body").splitlines() if line.strip()]
        if not lines:
            continue
        quote = max(lines, key=lambda line: (len(wanted & set(search_tokens(line))), len(line)))
        found.append((match.group("id"), quote))
    return found


def _sections_from_prompt(user: str, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = _SECTION_RE.findall(user)
    if not keys:
        return [{"key": "overview", "heading": "Overview", "body_md": "", "claims": claims}]
    sections: list[dict[str, Any]] = []
    for index, (key, heading) in enumerate(keys):
        sections.append(
            {
                "key": key,
                "heading": heading.strip(),
                "body_md": "",
                "claims": claims if index == 0 else [],
            }
        )
    return sections
