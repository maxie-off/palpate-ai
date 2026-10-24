from app.retrieval.glossary import GlossaryEntry, expand_terms

GLOSSARY = [
    GlossaryEntry(
        zh="鰓裂囊腫", en="Branchial Cleft Cyst", aliases=("cervical lymphoepithelial cyst",)
    ),
    GlossaryEntry(zh="粟丘疹", en="Milia"),
]


def test_english_term_expands_to_chinese() -> None:
    assert "鰓裂囊腫" in expand_terms(["branchial cleft cyst"], GLOSSARY)


def test_chinese_term_expands_to_english_and_aliases() -> None:
    out = expand_terms(["鰓裂囊腫"], GLOSSARY)
    assert "Branchial Cleft Cyst" in out
    assert "cervical lymphoepithelial cyst" in out


def test_unknown_term_is_kept_and_output_is_deduplicated() -> None:
    out = expand_terms(["unknown", "Milia", "粟丘疹"], GLOSSARY)
    assert out[0] == "unknown"
    assert len(out) == len(set(t.casefold() for t in out))
