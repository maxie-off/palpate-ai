"""Compile the task graph.

The shape is fixed: route, then one branch. Question answering and note drafting share
retrieve, draft, and verify. Verify may send the draft back once when every quote failed
the deterministic check. Quiz and summarize are named branches with no model call yet,
so adding them later does not require a new entry point.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.deps import PipelineDeps
from app.agent.nodes import (
    make_draft_answer,
    make_draft_notes,
    make_index,
    make_ingest_pdf,
    make_retrieve,
    make_summarize,
    make_transcribe,
    make_verify,
    quiz_node,
    route_node,
)
from app.agent.state import AgentState


def _entry(state: AgentState) -> str:
    if state["status"] == "failed":
        return "end"
    task = state["task"]
    if task in ("ingest", "draft_notes"):
        return "ingest_pdf"
    if task == "qa":
        return "retrieve"
    if task == "quiz":
        return "quiz"
    if task == "summarize":
        return "summarize"
    return "end"


def _after_ingest(state: AgentState) -> str:
    if state["status"] == "failed":
        return "end"
    return "transcribe_audio"


def _after_transcribe(state: AgentState) -> str:
    if state["status"] == "failed":
        return "end"
    return "index_chunks"


def _after_index(state: AgentState) -> str:
    if state["status"] == "failed" or state["task"] == "ingest":
        return "end"
    return "retrieve"


def _after_retrieve(state: AgentState) -> str:
    if state["status"] == "failed":
        return "end"
    if state["task"] == "draft_notes":
        return "draft_notes"
    return "draft_answer"


def _after_verify(state: AgentState) -> str:
    if state["status"] == "failed" or not state["retryable"]:
        return "end"
    if state["attempts"] >= state["max_attempts"]:
        return "end"
    if state["task"] == "draft_notes":
        return "draft_notes"
    if state["task"] == "qa":
        return "draft_answer"
    return "end"


def build_graph(deps: PipelineDeps) -> Any:
    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("ingest_pdf", make_ingest_pdf(deps))
    graph.add_node("transcribe_audio", make_transcribe(deps))
    graph.add_node("index_chunks", make_index(deps))
    graph.add_node("retrieve", make_retrieve(deps))
    graph.add_node("draft_answer", make_draft_answer(deps))
    graph.add_node("draft_notes", make_draft_notes(deps))
    graph.add_node("verify", make_verify(deps))
    graph.add_node("quiz", quiz_node)
    graph.add_node("summarize", make_summarize())

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        _entry,
        {
            "ingest_pdf": "ingest_pdf",
            "retrieve": "retrieve",
            "quiz": "quiz",
            "summarize": "summarize",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "ingest_pdf",
        _after_ingest,
        {"transcribe_audio": "transcribe_audio", "end": END},
    )
    graph.add_conditional_edges(
        "transcribe_audio",
        _after_transcribe,
        {"index_chunks": "index_chunks", "end": END},
    )
    graph.add_conditional_edges(
        "index_chunks",
        _after_index,
        {"retrieve": "retrieve", "end": END},
    )
    graph.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {"draft_answer": "draft_answer", "draft_notes": "draft_notes", "end": END},
    )
    graph.add_edge("draft_answer", "verify")
    graph.add_edge("draft_notes", "verify")
    graph.add_conditional_edges(
        "verify",
        _after_verify,
        {"draft_answer": "draft_answer", "draft_notes": "draft_notes", "end": END},
    )
    graph.add_edge("quiz", END)
    graph.add_edge("summarize", END)
    return graph.compile()
