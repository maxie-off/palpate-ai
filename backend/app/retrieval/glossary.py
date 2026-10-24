"""Bilingual glossary expansion: an English term finds Chinese notes, and the reverse."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryEntry:
    zh: str
    en: str
    aliases: tuple[str, ...] = ()

    def forms(self) -> tuple[str, ...]:
        return (self.zh, self.en, *self.aliases)


def expand_terms(terms: Iterable[str], glossary: Iterable[GlossaryEntry]) -> list[str]:
    """Return the input terms plus every known translation or alias, de-duplicated in order."""
    index: dict[str, GlossaryEntry] = {}
    for entry in glossary:
        for form in entry.forms():
            index[form.casefold()] = entry

    out: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            out.append(term)

    for term in terms:
        add(term)
        match = index.get(term.casefold())
        if match:
            for form in match.forms():
                add(form)
    return out
