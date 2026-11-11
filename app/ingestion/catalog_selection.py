"""
Deterministic selection of a small catalog from the large multi-source one.

Rule (explainable in a few lines):

1. Popularity score of a work::

       score = wikipedia_editions + gutenberg_downloads / 1000

   i.e. one Wikipedia language edition with an article weighs as much as a
   thousand monthly Project Gutenberg downloads. Works with neither signal
   (most DOAB and Open Library records) score 0.

2. Works are grouped by language (``Book.language``: the work's language,
   or its primary description's when the work's is unknown). Each group gets
   a quota of the target size: English 40%, Spanish 25%, and the remaining
   35% split equally among the other configured languages.

3. Inside each group, works whose language is stated by a source come
   before works whose language was only inferred from their description (an
   English novel described only in Spanish is not a Spanish work); then
   works are taken by decreasing score; ties are broken by the number of
   description languages (more first), then by id, so the result never
   depends on input order.

4. If a group has fewer works than its quota, the unused places go to the
   best remaining works of all groups, by the same ordering.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import UUID

from app.domain.entities import Book

DOWNLOADS_PER_EDITION = 1000
ENGLISH_SHARE = 0.40
SPANISH_SHARE = 0.25


@dataclass(frozen=True)
class SelectionCandidate:
    """The few fields of a work the selection needs (keeps memory small)."""

    book_id: UUID
    language: str
    score: float
    description_languages: int
    language_stated: bool = True
    """False when Book.language was inferred from the primary description"""


def popularity_score(book: Book) -> float:
    """Wikipedia editions plus Gutenberg downloads in thousands (0 when neither is known)."""
    metadata = book.metadata
    if metadata is None:
        return 0.0
    editions = metadata.wikipedia_editions or 0
    downloads = metadata.gutenberg_downloads or 0
    return editions + downloads / DOWNLOADS_PER_EDITION


def to_candidate(book: Book) -> SelectionCandidate:
    """Reduce a book to its selection fields."""
    languages = len(book.metadata.descriptions) if book.metadata and book.metadata.descriptions else 1
    stated = bool(book.metadata and book.metadata.work_language and book.metadata.work_language == book.language)
    return SelectionCandidate(
        book_id=book.id,
        language=book.language or "und",
        score=popularity_score(book),
        description_languages=languages,
        language_stated=stated,
    )


def _order_key(candidate: SelectionCandidate) -> Tuple[int, float, int, str]:
    return (
        int(not candidate.language_stated),
        -candidate.score,
        -candidate.description_languages,
        str(candidate.book_id),
    )


def language_quotas(size: int, languages: Sequence[str]) -> Dict[str, int]:
    """
    Places per language: en 40%, es 25%, the rest shared equally by the others.

    Rounding leftovers go to the languages in priority order.
    """
    others = [lang for lang in languages if lang not in ("en", "es")]
    quotas = {"en": int(size * ENGLISH_SHARE), "es": int(size * SPANISH_SHARE)}
    rest = size - quotas["en"] - quotas["es"]
    for lang in others:
        quotas[lang] = rest // len(others) if others else 0
    leftover = size - sum(quotas.values())
    for lang in [*languages, "en"]:
        if leftover <= 0:
            break
        if lang in quotas:
            quotas[lang] += 1
            leftover -= 1
    return quotas


def select_books(
    candidates: Iterable[SelectionCandidate],
    size: int,
    languages: Sequence[str],
) -> List[SelectionCandidate]:
    """
    Apply the selection rule.

    Args:
        candidates: One entry per work of the large catalog
        size: Number of works to select
        languages: Configured languages in priority order

    Returns:
        The selected candidates, ordered by language group and then rank
    """
    by_language: Dict[str, List[SelectionCandidate]] = {}
    for candidate in candidates:
        by_language.setdefault(candidate.language, []).append(candidate)
    for group in by_language.values():
        group.sort(key=_order_key)

    quotas = language_quotas(size, languages)
    selected: List[SelectionCandidate] = []
    chosen = set()
    for lang in languages:
        for candidate in by_language.get(lang, [])[: quotas.get(lang, 0)]:
            selected.append(candidate)
            chosen.add(candidate.book_id)

    remaining = size - len(selected)
    if remaining > 0:
        pool = sorted(
            (c for group in by_language.values() for c in group if c.book_id not in chosen),
            key=_order_key,
        )
        selected.extend(pool[:remaining])
    return selected[:size]


def language_mix(selected: Iterable[SelectionCandidate]) -> Dict[str, int]:
    """Number of selected works per language, largest first."""
    counts: Dict[str, int] = {}
    for candidate in selected:
        counts[candidate.language] = counts.get(candidate.language, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def describe_rule(size: int, languages: Sequence[str], quotas: Optional[Dict[str, int]] = None) -> str:
    """One-paragraph description of the rule with the quotas used."""
    quotas = quotas or language_quotas(size, languages)
    parts = ", ".join(f"{lang} {quotas[lang]}" for lang in languages if lang in quotas)
    return (
        f"Score = Wikipedia editions + Gutenberg downloads/{DOWNLOADS_PER_EDITION}. "
        f"Quotas for {size} works: {parts}. Within a language, works whose language is "
        "stated by a source first, then highest score (ties: more description languages, "
        "then id); unused places go to the best remaining works of any language."
    )
