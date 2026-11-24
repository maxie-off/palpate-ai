from pathlib import Path

import pymupdf

from app.core.config import ParsingConfig
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

    flags = {b.text: b.emphasis for b in pages[0].blocks}
    assert flags["Verruciform xanthoma"] == 1.0
    assert flags["Plain paragraph"] == 0.0
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


def _colour_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "key point in blue", color=(0, 0.44, 0.75))
    page.insert_text((72, 140), "ordinary black text")
    page.insert_text((72, 180), "bold black heading", fontname="hebo")
    page.insert_text((72, 220), "- 7 -")
    doc.save(path)
    doc.close()


def test_coloured_text_counts_as_emphasis_but_black_bold_does_not(tmp_path: Path) -> None:
    pdf_path = tmp_path / "colour.pdf"
    _colour_pdf(pdf_path)

    blocks = {b.text: b.emphasis for b in PyMuPDFParser().parse(pdf_path)[0].blocks}

    assert blocks["key point in blue"] == 1.0
    assert blocks["ordinary black text"] == 0.0
    assert blocks["bold black heading"] == 0.0


def test_bold_counts_when_the_domain_opts_in(tmp_path: Path) -> None:
    pdf_path = tmp_path / "colour.pdf"
    _colour_pdf(pdf_path)

    parser = PyMuPDFParser(ParsingConfig(emphasis_bold=True))
    blocks = {b.text: b.emphasis for b in parser.parse(pdf_path)[0].blocks}

    assert blocks["bold black heading"] == 1.0


def test_page_numbers_are_dropped(tmp_path: Path) -> None:
    pdf_path = tmp_path / "colour.pdf"
    _colour_pdf(pdf_path)
    texts = [b.text for b in PyMuPDFParser().parse(pdf_path)[0].blocks]
    assert "- 7 -" not in texts
