"""Prompt text for grounded answers and class-note drafts.

The source block format is also what the offline stand-in parses. Keep the markers stable.
"""

from collections.abc import Sequence

from app.agent.types import SearchHit
from app.core.config import NoteSectionSpec

_QA_SYSTEM = """\
You are Palpate, a study assistant for dental students.

Answer ONLY from the source blocks below. Never use outside knowledge. Sources may be \
Traditional Chinese, English, or both; keep a term in the language the source uses.

Each claim must quote a verbatim span from a source. If the sources do not support an \
answer, set status to "insufficient_evidence" and return no claims.

Return a single JSON object and nothing else:
{"status": "answered" | "insufficient_evidence",
 "summary": "short grounded explanation",
 "claims": [{"text": "one sentence", "citations": [{"chunk_id": "<id>", "quote": "<verbatim>"}]}]}
"""

_NOTES_SYSTEM = """\
You are Palpate, drafting class notes for a dental student from their own materials.

Use ONLY the source blocks. Reading material (PDFs) and the lecture transcript are both \
sources. Cite a PDF by quoting it, and cite the recording by quoting the transcript. \
Do not add facts that are not in the sources. If a section has no support, return it with \
an empty claims list.

Return a single JSON object and nothing else:
{"status": "answered" | "insufficient_evidence",
 "sections": [
   {"key": "<section key>", "heading": "<heading>", "body_md": "<markdown>",
    "claims": [{"text": "one sentence",
                "citations": [{"chunk_id": "<id>", "quote": "<verbatim>"}]}]}
 ]}
"""


def _source_block(hit: SearchHit) -> str:
    chunk = hit.chunk
    page = "-" if chunk.page is None else str(chunk.page)
    return (
        f"<<<CHUNK id={chunk.id} doc={chunk.doc_id} modality={chunk.modality} page={page}>>>\n"
        f"{chunk.text}\n"
        "<<<END>>>"
    )


def _sources(hits: Sequence[SearchHit]) -> str:
    return "\n\n".join(_source_block(hit) for hit in hits)


def build_qa_prompt(question: str, hits: Sequence[SearchHit], rejection: str) -> tuple[str, str]:
    user = f"TASK: qa\n\nSources:\n{_sources(hits)}\n\nQuestion: {question}"
    if rejection:
        user += (
            "\n\nThe previous draft was rejected because these quotes were not found "
            f"verbatim in the cited source. Copy a shorter exact span.\n{rejection}"
        )
    return _QA_SYSTEM, user


def build_notes_prompt(
    message: str,
    hits: Sequence[SearchHit],
    sections: Sequence[NoteSectionSpec],
    rules_md: str,
    rejection: str,
) -> tuple[str, str]:
    section_lines = "\n".join(
        f"SECTION key={section.key} heading={section.heading}\n{section.instruction}"
        for section in sections
    )
    user = (
        f"TASK: notes\n\nHouse style:\n{rules_md}\n\n"
        f"Sections to draft:\n{section_lines}\n\n"
        f"Sources:\n{_sources(hits)}\n\n"
        f"Request: {message or 'Draft class notes from the sources.'}"
    )
    if rejection:
        user += (
            "\n\nThe previous draft was rejected because these quotes were not found "
            f"verbatim in the cited source. Copy a shorter exact span.\n{rejection}"
        )
    return _NOTES_SYSTEM, user
