"""Task selection.

The branch is chosen here, before any model call. A caller that already knows the task
(the API, a button, a test) sets ``request.task`` and this function returns it unchanged.
A free-text message is classified with keywords so the graph still has one entry point.
The model is not asked which node to run.
"""

from app.agent.types import AgentRequest, TaskName

_QUIZ_MARKERS = ("quiz", "worksheet", "課後測驗")
_NOTE_MARKERS = ("draft note", "draft notes", "class note", "class notes", "共筆", "筆記")


def infer_task(request: AgentRequest) -> TaskName:
    if request.task is not None:
        return request.task
    text = request.message.casefold()
    if any(marker in text for marker in _QUIZ_MARKERS):
        return "quiz"
    if "summarize" in text or "summarise" in text or "摘要" in text:
        return "summarize"
    if any(marker in text for marker in _NOTE_MARKERS):
        return "draft_notes"
    if request.audio_path and not request.message:
        return "draft_notes"
    if request.pdf_path and not request.message and not request.audio_path:
        return "ingest"
    return "qa"
