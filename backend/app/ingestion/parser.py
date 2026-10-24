"""PDF parsing: text blocks per page plus a teacher-emphasis (highlight) flag.

Highlights are detected without any AI: PDF highlight annotations, and filled yellow rectangles
(which is how Word exports highlighted text). Emphasis is flagged per text block, which is coarse
but enough to boost ranking; finer span-level detection is a stage 2 refinement.
"""

from pathlib import Path
from typing import Any, Protocol

import pymupdf

from app.models.schemas import PageContent, TextBlock

_MIN_OVERLAP = 0.3
_PDF_ANNOT_HIGHLIGHT = 8  # pymupdf.PDF_ANNOT_HIGHLIGHT, which its type stubs do not export


class DocumentParser(Protocol):
    def parse(self, path: Path) -> list[PageContent]: ...


def _is_yellowish(fill: tuple[float, ...]) -> bool:
    if len(fill) != 3:
        return False
    r, g, b = fill
    return r > 0.8 and g > 0.8 and b < 0.5


def _highlight_rects(page: Any) -> list[Any]:
    rects: list[Any] = []
    for annot in page.annots() or []:
        if annot.type[0] == _PDF_ANNOT_HIGHLIGHT:
            rects.append(annot.rect)
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if fill and _is_yellowish(tuple(fill)):
            rects.append(pymupdf.Rect(drawing["rect"]))
    return rects


def _is_emphasised(block_rect: Any, highlights: list[Any]) -> bool:
    block_area = block_rect.get_area()
    if block_area <= 0:
        return False
    return any((block_rect & hl).get_area() / block_area >= _MIN_OVERLAP for hl in highlights)


class PyMuPDFParser:
    def parse(self, path: Path) -> list[PageContent]:
        pages: list[PageContent] = []
        with pymupdf.open(path) as doc:
            for index, page in enumerate(doc, start=1):
                highlights = _highlight_rects(page)
                blocks: list[TextBlock] = []
                for x0, y0, x1, y1, text, _no, block_type in page.get_text("blocks"):
                    if block_type != 0 or not text.strip():
                        continue
                    rect = pymupdf.Rect(x0, y0, x1, y1)
                    blocks.append(
                        TextBlock(text=text.strip(), emphasised=_is_emphasised(rect, highlights))
                    )
                pages.append(PageContent(page=index, blocks=blocks))
        return pages
