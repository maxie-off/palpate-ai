from pathlib import Path

import pymupdf

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


def test_lines_carry_bbox_in_pdf_user_space(tmp_path: Path) -> None:
    pdf_path = tmp_path / "pos.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Near the top")  # 100pt below the top edge
    doc.save(pdf_path)
    doc.close()

    line = PyMuPDFParser().parse(pdf_path)[0].blocks[0].lines[0]

    x0, y0, x1, y1 = line.bbox
    assert line.text == "Near the top"
    assert x0 == 72.0 and x1 > x0
    # PDF user space has y pointing up, so "near the top" means a large y.
    assert 730 < y0 < y1 < 760
