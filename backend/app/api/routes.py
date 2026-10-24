"""HTTP routes. Routes only translate HTTP to service calls; no business logic here."""

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/documents")
def upload_document() -> None:
    """Ingest a PDF/PPTX into the knowledge base. TODO(stage 1): wire ingestion pipeline."""
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/worksheets")
def upload_worksheet() -> None:
    """Upload an exam sheet and get per-question answers. TODO(stage 1): wire query pipeline."""
    raise HTTPException(status_code=501, detail="Not implemented yet")
