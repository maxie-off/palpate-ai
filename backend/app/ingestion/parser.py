"""PDF parsing: text lines with positions, plus a teacher-emphasis (highlight) flag.

Highlights are detected without any AI: PDF highlight annotations, and filled yellow rectangles
(which is how Word exports highlighted text). Every line keeps its bounding box in PDF user space,
because positions cannot be recovered later without re-parsing the original file.
"""

from pathlib import Path
from typing import Any, Protocol

import pymupdf

from app.core.config import ParsingConfig
from app.ingestion.furniture import is_page_number
from app.models.schemas import Line, PageContent, TextBlock

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


def _is_highlighted(rect: Any, highlights: list[Any]) -> bool:
    area = rect.get_area()
    if area <= 0:
        return False
    return any((rect & hl).get_area() / area >= _MIN_OVERLAP for hl in highlights)


def _saturation(rgb: int) -> float:
    r, g, b = (rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255
    top = max(r, g, b)
    return (top - min(r, g, b)) / top if top else 0.0


def _span_emphasised(span: dict[str, Any], config: ParsingConfig) -> bool:
    if _saturation(span["color"]) >= config.min_color_saturation:
        return True
    is_bold = bool(span["flags"] & 16) or "bold" in span["font"].lower()
    return config.emphasis_bold and is_bold


def _text_emphasis(raw_line: dict[str, Any], config: ParsingConfig) -> float:
    total = marked = 0
    for span in raw_line["spans"]:
        n = len("".join(span["text"].split()))
        total += n
        if _span_emphasised(span, config):
            marked += n
    return marked / total if total else 0.0


def _to_pdf_space(rect: Any, page: Any) -> tuple[float, float, float, float]:
    """Convert a PyMuPDF rect (origin top-left, y down) to PDF user space (origin bottom-left).

    Assumes the crop box starts at the origin, which holds for ordinary exported PDFs.
    """
    r = rect * page.derotation_matrix
    height = page.mediabox.height
    return (round(r.x0, 2), round(height - r.y1, 2), round(r.x1, 2), round(height - r.y0, 2))


def _weighted(items: list[tuple[str, float]]) -> float:
    total = sum(len(t) for t, _ in items)
    return sum(len(t) * e for t, e in items) / total if total else 0.0


class PyMuPDFParser:
    def __init__(self, config: ParsingConfig | None = None) -> None:
        self._config = config or ParsingConfig()

    def parse(self, path: Path) -> list[PageContent]:
        pages: list[PageContent] = []
        with pymupdf.open(path) as doc:
            for index, page in enumerate(doc, start=1):
                highlights = _highlight_rects(page)
                blocks: list[TextBlock] = []
                for raw in page.get_text("dict")["blocks"]:
                    if raw["type"] != 0:
                        continue
                    lines: list[Line] = []
                    for raw_line in raw["lines"]:
                        text = "".join(s["text"] for s in raw_line["spans"]).strip()
                        if not text or is_page_number(text):
                            continue
                        rect = pymupdf.Rect(raw_line["bbox"])
                        emphasis = _text_emphasis(raw_line, self._config)
                        if _is_highlighted(rect, highlights):
                            emphasis = 1.0
                        lines.append(
                            Line(
                                text=text,
                                bbox=_to_pdf_space(rect, page),
                                emphasis=emphasis,
                            )
                        )
                    if not lines:
                        continue
                    blocks.append(
                        TextBlock(
                            text="\n".join(ln.text for ln in lines),
                            emphasis=_weighted([(ln.text, ln.emphasis) for ln in lines]),
                            lines=lines,
                        )
                    )
                pages.append(PageContent(page=index, blocks=blocks))
        return pages
