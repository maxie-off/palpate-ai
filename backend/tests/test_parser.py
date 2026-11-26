from pathlib import Path

import pymupdf
import pytest

from app.ingestion.parser import PyMuPDFParser


def test_highlighted_block_is_flagged_as_emphasised(tmp_path: Path) -> None:
    pdf_path = tmp_path / "notes.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Verruciform xanthoma")
    page.insert_text((72, 400), "Plain paragraph")
    page.add_highlight_annot(page.search_for("Verruciform xanthoma")[0])
    doc.save(pdf_path)
    doc.close()

    pages = PyMuPDFParser().parse(pdf_path)

    flags = {b.text: b.emphasised for b in pages[0].blocks}
    assert flags["Verruciform xanthoma"] is True
    assert flags["Plain paragraph"] is False
    assert pages[0].page == 1
    assert pages[0].lines
    assert pages[0].blocks[0].bbox is not None
    assert pages[0].blocks[0].bbox.y0 < pages[0].blocks[0].bbox.y1


def test_figure_bbox_is_pdf_user_space(tmp_path: Path) -> None:
    from app.agent.sample_pdf import build_sample_lecture

    pdf_path = tmp_path / "lecture.pdf"
    build_sample_lecture(pdf_path)
    pages = PyMuPDFParser().parse(pdf_path)

    figure = pages[0].figures[0]
    assert figure.width == 80
    assert figure.height == 60
    assert figure.image_bytes
    assert figure.bbox.x0 == pytest.approx(72)
    assert figure.bbox.x1 == pytest.approx(152)
    assert figure.bbox.y0 == pytest.approx(pages[0].height - 220)
    assert figure.bbox.y1 == pytest.approx(pages[0].height - 160)
    emphasised = {line.text: line.emphasised for line in pages[0].lines}
    assert emphasised["They are common in newborns."] is True
    assert emphasised["Milia are 1-2 mm keratin-filled cysts on the face."] is False
