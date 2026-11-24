from app.ingestion.furniture import extract_metadata, is_page_number
from app.models.schemas import PageContent, TextBlock


def test_page_number_forms() -> None:
    assert is_page_number("- 12 -") and is_page_number(" -3- ")
    assert not is_page_number("12 patients") and not is_page_number("- item")


def test_extract_metadata_reads_header_fields() -> None:
    page = PageContent(
        page=1,
        blocks=[
            TextBlock(text="授課老師：巫仰哲"),
            TextBlock(text="上課日期：2026/11/19"),
            TextBlock(text="檢查者："),
            TextBlock(text="壹、 body"),
        ],
    )
    assert extract_metadata([page]) == {"授課老師": "巫仰哲", "上課日期": "2026/11/19"}
