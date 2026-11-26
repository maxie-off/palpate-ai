"""Graph state. Nodes return partial updates; LangGraph merges them.

``trace`` appends. Every other field is replaced by the node that owns it.
"""

import operator
from typing import Annotated, Any, TypedDict

from app.agent.routing import infer_task
from app.agent.types import AgentRequest


class AgentState(TypedDict):
    task: str
    owner_id: str
    lesson_id: str
    message: str
    pdf_path: str
    audio_path: str
    doc_id: str
    title: str
    doc_type: str
    active_doc_id: str
    pending_chunks: list[dict[str, Any]]
    pending_docs: list[dict[str, Any]]
    ingested: list[dict[str, Any]]
    hits: list[dict[str, Any]]
    draft: dict[str, Any] | None
    output: dict[str, Any] | None
    status: str
    error: str
    attempts: int
    max_attempts: int
    retryable: bool
    rejection: str
    trace: Annotated[list[str], operator.add]


def initial_state(request: AgentRequest, *, max_attempts: int) -> AgentState:
    return AgentState(
        task=infer_task(request),
        owner_id=request.owner_id or "local",
        lesson_id=request.lesson_id,
        message=request.message,
        pdf_path=request.pdf_path,
        audio_path=request.audio_path,
        doc_id=request.doc_id,
        title=request.title,
        doc_type=request.doc_type,
        active_doc_id="",
        pending_chunks=[],
        pending_docs=[],
        ingested=[],
        hits=[],
        draft=None,
        output=None,
        status="ok",
        error="",
        attempts=0,
        max_attempts=max_attempts,
        retryable=False,
        rejection="",
        trace=[],
    )
