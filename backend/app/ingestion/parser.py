"""PDF parsing: text lines, figure regions, and a teacher-emphasis flag.

Highlights are detected without any AI: PDF highlight annotations, and filled yellow rectangles
(which is how Word exports highlighted text). Coordinates stored on lines and figures are PDF
user space (origin at the bottom-left). PyMuPDF reports a top-left origin, so this module
converts before anything leaves the parser. Highlighting later can draw those rectangles at
any zoom.
"""

from pathlib import Path
from typing import Any, Protocol

import pymupdf

from app.models.schemas import BBox, FigureBlock, PageContent, TextBlock, TextLine

_MIN_OVERLAP = 0.3
_MIN_FIGURE_PX = 8
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


def _to_user_space(rect: Any, page_height: float) -> BBox:
    """Convert a PyMuPDF rect (origin top-left, y downward) to PDF user space."""
    x0, y0, x1, y1 = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    left, right = sorted((x0, x1))
    top, bottom = min(y0, y1), max(y0, y1)
    return BBox(x0=left, y0=page_height - bottom, x1=right, y1=page_height - top)


def _extract_lines(page: Any, highlights: list[Any], page_height: float) -> list[TextLine]:
    lines: list[TextLine] = []
    raw = page.get_text("dict")
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text:
                continue
            mu_rect = pymupdf.Rect(line["bbox"])
            lines.append(
                TextLine(
                    text=text,
                    bbox=_to_user_space(mu_rect, page_height),
                    emphasised=_is_emphasised(mu_rect, highlights),
                )
            )
    return lines


def _extract_figures(doc: Any, page: Any, page_height: float) -> list[FigureBlock]:
    figures: list[FigureBlock] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for info in page.get_image_info(xrefs=True):
        xref = int(info.get("xref") or 0)
        bbox = info.get("bbox")
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        if xref <= 0 or not bbox or width < _MIN_FIGURE_PX or height < _MIN_FIGURE_PX:
            continue
        user = _to_user_space(bbox, page_height)
        key = (xref, round(user.x0), round(user.y0), round(user.x1), round(user.y1))
        if key in seen:
            continue
        seen.add(key)
        try:
            extracted: dict[str, Any] = doc.extract_image(xref)
        except Exception:
            continue
        image_bytes = extracted.get("image") or b""
        if not image_bytes:
            continue
        figures.append(
            FigureBlock(
                xref=xref,
                bbox=user,
                width=int(extracted.get("width") or width),
                height=int(extracted.get("height") or height),
                ext=str(extracted.get("ext") or "png"),
                image_bytes=image_bytes,
            )
        )
    return figures


class PyMuPDFParser:
    def parse(self, path: Path) -> list[PageContent]:
        pages: list[PageContent] = []
        with pymupdf.open(path) as doc:
            for index, page in enumerate(doc, start=1):
                highlights = _highlight_rects(page)
                page_height = float(page.rect.height)
                blocks: list[TextBlock] = []
                for x0, y0, x1, y1, text, _no, block_type in page.get_text("blocks"):
                    if block_type != 0 or not text.strip():
                        continue
                    rect = pymupdf.Rect(x0, y0, x1, y1)
                    blocks.append(
                        TextBlock(
                            text=text.strip(),
                            emphasised=_is_emphasised(rect, highlights),
                            bbox=_to_user_space(rect, page_height),
                        )
                    )
                pages.append(
                    PageContent(
                        page=index,
                        width=float(page.rect.width),
                        height=page_height,
                        blocks=blocks,
                        lines=_extract_lines(page, highlights, page_height),
                        figures=_extract_figures(doc, page, page_height),
                    )
                )
        return pages
