"""LangGraph pipeline: ingest, class notes, and grounded question answering.

The public entry point is ``run_agent``. The graph routes on an explicit task. The model
writes claims; it does not choose which steps run.
"""

from app.agent.service import run_agent

__all__ = ["run_agent"]
