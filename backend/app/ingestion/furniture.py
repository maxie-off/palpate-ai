"""Page furniture in student notes: page numbers and the author/date header block."""

import re
from collections.abc import Sequence

from app.models.schemas import PageContent

PAGE_NUMBER_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")
_METADATA_RE = re.compile(r"^\s*(撰寫者|檢查者|複審者|授課老師|上課日期)\s*[：:]\s*(.*)$")


def is_page_number(text: str) -> bool:
    return bool(PAGE_NUMBER_RE.match(text))


def metadata_field(text: str) -> tuple[str, str] | None:
    """Return (label, value) for a header line such as ``授課老師：巫仰哲``."""
    match = _METADATA_RE.match(text)
    return (match.group(1), match.group(2).strip()) if match else None


def is_furniture(text: str) -> bool:
    return is_page_number(text) or metadata_field(text) is not None


def extract_metadata(pages: Sequence[PageContent]) -> dict[str, str]:
    """Collect the header fields (e.g. 上課日期) so ingest can store them on the document."""
    found: dict[str, str] = {}
    for page in pages:
        for block in page.blocks:
            for line in block.text.splitlines():
                field = metadata_field(line)
                if field and field[1]:
                    found.setdefault(field[0], field[1])
    return found
