from app.ingestion.text import make_search_text


def test_cjk_becomes_bigrams_and_latin_becomes_words() -> None:
    assert make_search_text("鰓裂囊腫 Branchial Cyst") == "鰓裂 裂囊 囊腫 branchial cyst"


def test_width_and_case_are_normalised() -> None:
    assert make_search_text("ＵＳＰ６ gene") == "usp6 gene"


def test_single_cjk_char_is_kept_and_punctuation_splits_runs() -> None:
    assert make_search_text("癌、瘤") == "癌 瘤"
