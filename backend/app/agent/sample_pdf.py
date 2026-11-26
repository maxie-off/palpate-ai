"""Build the synthetic lecture PDF used by the sample run.

The text is written for this repository. It is not taken from a textbook, a lecture the
author attended, or a classmate's notes.
"""

from pathlib import Path

import pymupdf


def build_sample_lecture(path: Path) -> None:
    """Two pages: a short oral-cyst note, one diagram, and a highlight on a studied line."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Demo lecture: oral cysts")
    page.insert_text((72, 100), "Milia are 1-2 mm keratin-filled cysts on the face.")
    page.insert_text((72, 124), "They are common in newborns.")
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 60))
    pixmap.set_rect(pixmap.irect, (220, 180, 170))
    page.insert_image(pymupdf.Rect(72, 160, 152, 220), pixmap=pixmap)
    page.insert_text((72, 240), "Figure 1. Synthetic diagram of milia on the cheek.")
    page.insert_text((72, 280), "A branchial cleft cyst presents as a lateral neck mass.")
    hits = page.search_for("They are common in newborns.")
    if hits:
        page.add_highlight_annot(hits[0])

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Treatment is surgical excision of the cyst.")
    page2.insert_text((72, 100), "Incision and drainage is not definitive.")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
