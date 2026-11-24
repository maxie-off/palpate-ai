"""Prompt templates for grounded answering."""

from collections.abc import Sequence

from app.models.schemas import ParsedQuestion, ScoredChunk

SYSTEM_PROMPT = """\
You are Palpate, a study assistant for dental students.

Answer ONLY from the numbered sources provided. Never use outside knowledge. Sources may be in \
Traditional Chinese, English, or both; keep original terms in whichever language the source uses.

If the sources do not clearly support an answer, set status to "insufficient_evidence".

Return a single JSON object and nothing else:
{"status": "answered" | "insufficient_evidence",
 "summary": "short grounded explanation",
 "citations": [{"chunk_id": "<source id>", "quote": "<passage copied verbatim from it>"}]}

Every quote must be copied exactly from the cited source.
"""


def build_user_prompt(question: ParsedQuestion, evidence: Sequence[ScoredChunk]) -> str:
    sources = "\n\n".join(
        f"[{e.chunk.id}] {e.chunk.breadcrumb} (p.{e.chunk.page})\n{e.chunk.body}" for e in evidence
    )
    options = "\n".join(f"- {o}" for o in question.options)
    return f"Sources:\n{sources}\n\nQuestion: {question.stem}\nOptions:\n{options}"
