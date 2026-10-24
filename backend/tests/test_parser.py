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
