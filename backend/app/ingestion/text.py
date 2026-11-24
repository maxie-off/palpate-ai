"""Lexical text helpers shared by ingestion and retrieval (they must tokenise identically)."""

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+|[㐀-鿿豈-﫿]+")
_CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")


def make_search_text(text: str) -> str:
    """CJK character bigrams plus lowercase latin words, space separated.

    Postgres full-text search has no Chinese tokenizer, so we segment ourselves and index the
    result with the ``simple`` config. A single CJK character is kept as a unigram.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(unicodedata.normalize("NFKC", text).lower()):
        token = match.group()
        if _CJK_RE.match(token):
            tokens.extend(token[i : i + 2] for i in range(max(len(token) - 1, 1)))
        else:
            tokens.append(token)
    return " ".join(tokens)
