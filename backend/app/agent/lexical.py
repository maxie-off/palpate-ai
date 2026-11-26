"""Lexical helpers shared by indexing and in-memory retrieval.

Postgres full-text search has no Chinese tokenizer. We precompute Latin words and CJK
bigrams into ``search_text`` and query that with the ``simple`` config, which needs no
extension. This is the alternative to ``pg_bigm`` noted in the schema draft.
"""

import math
import re
import unicodedata
from collections.abc import Sequence

_LATIN = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def search_tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold()
    tokens = _LATIN.findall(folded)
    for run in _CJK_RUN.findall(folded):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def render_search_text(text: str) -> str:
    return " ".join(search_tokens(text))


def lexical_overlap(query: str, search_text: str) -> int:
    if not query or not search_text:
        return 0
    return len(set(search_tokens(query)) & set(search_text.split()))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def rrf_component(rank: int | None, k: int) -> float:
    """Reciprocal rank. A missing list contributes the same small term as rank 1000."""
    position = 1000 if rank is None else rank
    return 1.0 / (k + position)
