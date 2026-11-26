"""HTTP routes. Routes only translate HTTP to service calls; no business logic here."""

from fastapi import APIRouter, HTTPException

from app.agent import run_agent
from app.agent.types import AgentRequest, AgentResult

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/agent/run", response_model=AgentResult)
def agent_run(body: AgentRequest) -> AgentResult:
    """Run one pipeline task: ingest, draft_notes, qa, quiz, or summarize.

    Local paths are a stage-1 seam. The desktop client in the schema draft pushes bytes
    through the same ingest contract; this route does not read a student's disk on its own.
    """
    return run_agent(body)


@router.post("/documents")
def upload_document() -> None:
    """Ingest a PDF/PPTX into the knowledge base. TODO(stage 1): wire ingestion pipeline."""
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/worksheets")
def upload_worksheet() -> None:
    """Upload an exam sheet and get per-question answers. TODO(stage 1): wire query pipeline."""
    raise HTTPException(status_code=501, detail="Not implemented yet")
