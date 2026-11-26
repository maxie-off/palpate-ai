"""Citation objects round-trip, including every anchor shape."""

from app.agent.types import (
    CharRangeAnchor,
    PageBBoxAnchor,
    SourceCitation,
    TimeRangeAnchor,
)
from app.core.providers import AmazonTranscribeClient


def test_citation_keeps_page_time_and_character_anchors() -> None:
    citation = SourceCitation(
        id="c1",
        rank=1,
        claim="Milia are keratin-filled cysts.",
        quote="keratin-filled cysts",
        doc_id="lecture",
        chunk_id="lecture-0000",
        doc_type="slides",
        modality="text",
        anchors=[
            PageBBoxAnchor(page=2, x0=10, y0=20, x1=30, y1=40),
            TimeRangeAnchor(t_start=14.5, t_end=18.0),
            CharRangeAnchor(c_start=0, c_end=12),
        ],
    )
    restored = SourceCitation.model_validate(citation.model_dump())
    assert restored.anchors[0].kind == "page_bbox"
    assert restored.anchors[0].page == 2
    assert restored.anchors[1].kind == "time_range"
    assert restored.anchors[1].t_end == 18.0
    assert restored.anchors[2].kind == "char_range"
    assert restored.anchors[2].c_end == 12
    assert restored.verified is True


def test_transcribe_adapter_is_explicitly_unwired() -> None:
    client = AmazonTranscribeClient(region="us-east-1", model_id="")
    try:
        client.transcribe(__import__("pathlib").Path("lecture.wav"), language="zh-TW")
    except NotImplementedError as exc:
        assert "not implemented" in str(exc).lower()
    else:
        raise AssertionError("expected the stub to refuse")
