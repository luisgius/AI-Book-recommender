"""
Multi-source catalog builder: normalise, filter, de-duplicate and merge works.

This is the bulk counterpart of ``IngestionService`` (which ingests the
results of one search query). It reads every record of a ``BulkBooksSource``
and, in batches:

1. Normalisation (``normalize_record``): title (entities, MARC markers,
   whitespace), author names
   ("Austen, Jane, 1775-1817" -> "Jane Austen"), ISO 639-1 language codes,
   publication year, subjects (cleaned, at most ten) and descriptions
   (markup stripped, cut to about 1,200 characters at a sentence end).
2. Quality filter: a record is kept only if it has a title, an author and at
   least one description of 200 or more characters in one of the configured
   languages (the language comes from the source or, if absent, from the
   description text). An implausible year is dropped, not the record.
3. Primary description (``choose_primary_description``): the description in
   the work's own language if there is one, else English, else the first
   available in the configured language priority. It becomes
   ``Book.description``; the others stay in ``metadata.descriptions``.
4. De-duplication (``match_keys`` + ``DuplicateKeyIndex``): the record is
   merged into an existing work that shares, in this order, a Wikidata id, an
   Open Library id, a Project Gutenberg id, an ISBN-13, or the same
   normalised title + first-author surname with years at most two apart
   (when both are known). Two records with different Wikidata ids are never
   merged.
5. Merge (``merge_books``): descriptions (one per language, preferring
   human-written text and the more reliable source), identifiers, subjects
   and provenance are unioned; popularity signals keep the maximum.
6. Storage: one ``save_many`` per batch and one key-index update, so memory
   stays bounded by the batch size whatever the size of the source.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import UUID

from app.domain.book_normalization import (
    author_surname,
    clean_isbn13,
    clean_title,
    clean_subjects,
    is_plausible_year,
    normalize_author_name,
    strip_markup,
    title_key,
    to_iso639_1,
    truncate_text,
)
from app.domain.entities import Book
from app.domain.ports import BookCatalogRepository, BulkBooksSource, DuplicateKeyIndex
from app.domain.value_objects import BookDescription, BookMetadata
from app.ingestion.language_detection import detect_language

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES: Tuple[str, ...] = (
    "en", "es", "fr", "de", "it", "pt", "ca", "nl", "ru", "zh", "ja", "ar",
)

# Preference between two descriptions in the same language (lower = better).
SOURCE_PRIORITY: Tuple[str, ...] = ("wikipedia", "doab", "open_library", "gutenberg")

# Order of the de-duplication rules; also the prefixes of the match keys.
MATCH_RULES: Tuple[str, ...] = ("wikidata", "openlibrary", "gutenberg", "isbn13", "title")

MAX_AUTHORS = 10


@dataclass(frozen=True)
class CatalogRules:
    """Parameters of the normalisation and quality filter."""

    languages: Tuple[str, ...] = DEFAULT_LANGUAGES
    """Accepted description languages, in priority order"""

    min_description_chars: int = 200
    """Minimum length of a description after markup is stripped"""

    max_description_chars: int = 1200
    """Descriptions are truncated to this length (at a sentence end)"""

    max_subjects: int = 10
    """Maximum number of subjects kept per work"""

    max_year: int = datetime.now(UTC).year + 1
    """Latest plausible publication year"""

    year_tolerance: int = 2
    """Maximum year difference for the title/author match rule"""


@dataclass
class SourceIngestionStats:
    """What happened to the records of one source during ingestion."""

    source: str
    read: int = 0
    """Records read from the source (including those it could not convert)"""

    rejected: Counter = field(default_factory=Counter)
    """Records rejected, by reason"""

    inserted: int = 0
    """Records stored as new works"""

    merged: int = 0
    """Records merged into an existing work"""

    merged_by_rule: Counter = field(default_factory=Counter)
    """Merges by the rule that matched ('wikidata', ..., 'title')"""

    merged_cross_source: int = 0
    """Merges into a work that had no record from this source yet"""

    already_present: int = 0
    """Records whose work was stored by a previous run (re-run)"""

    years_dropped: int = 0
    """Implausible years removed from kept records"""

    descriptions_detected: int = 0
    """Descriptions whose language was detected from the text"""

    @property
    def kept(self) -> int:
        """Records that passed the quality filter."""
        return self.inserted + self.merged + self.already_present

    def to_dict(self) -> Dict:
        """Plain dict for JSON storage."""
        return {
            "source": self.source,
            "read": self.read,
            "kept": self.kept,
            "rejected": dict(self.rejected),
            "inserted": self.inserted,
            "merged": self.merged,
            "merged_by_rule": dict(self.merged_by_rule),
            "merged_cross_source": self.merged_cross_source,
            "already_present": self.already_present,
            "years_dropped": self.years_dropped,
            "descriptions_detected": self.descriptions_detected,
        }


# --------------------------------------------------------------- descriptions

def description_preference(description: BookDescription) -> Tuple[int, int, int]:
    """
    Sort key to choose between two descriptions in the same language.

    Human-written text first, then the more reliable source (Wikipedia, DOAB,
    Open Library, Gutenberg), then the longer text.
    """
    try:
        source_rank = SOURCE_PRIORITY.index(description.source)
    except ValueError:
        source_rank = len(SOURCE_PRIORITY)
    return (int(description.auto_generated), source_rank, -len(description.text))


def choose_primary_description(
    descriptions: Sequence[BookDescription],
    work_language: Optional[str],
    language_priority: Sequence[str],
) -> Optional[BookDescription]:
    """
    Pick the description shown as ``Book.description``.

    Rule: the description in the work's own language if present; else the
    English one; else the first available in the language priority order;
    else the first one.

    Args:
        descriptions: Candidate descriptions (at most one per language)
        work_language: Language of the work, if known
        language_priority: Configured languages in priority order

    Returns:
        The primary description, or None if there are none
    """
    by_language = {d.language: d for d in descriptions}
    for language in [work_language, "en", *language_priority]:
        if language and language in by_language:
            return by_language[language]
    return descriptions[0] if descriptions else None


def order_descriptions(
    descriptions: Iterable[BookDescription],
    work_language: Optional[str],
    language_priority: Sequence[str],
) -> Tuple[BookDescription, ...]:
    """
    Keep one description per language and put the primary one first.

    When two descriptions share a language, ``description_preference`` picks
    the one kept. The rest follow in language priority order.
    """
    best: Dict[Optional[str], BookDescription] = {}
    for description in descriptions:
        current = best.get(description.language)
        if current is None or description_preference(description) < description_preference(current):
            best[description.language] = description
    kept = list(best.values())
    primary = choose_primary_description(kept, work_language, language_priority)
    if primary is None:
        return ()
    rank: Dict[Optional[str], int] = {lang: i for i, lang in enumerate(language_priority)}
    others = sorted(
        (d for d in kept if d is not primary),
        key=lambda d: (rank.get(d.language, len(rank)), d.language or ""),
    )
    return (primary, *others)


# -------------------------------------------------------------- normalisation

def normalize_record(
    book: Book,
    rules: CatalogRules,
    detector: Callable[[str], Optional[str]] = detect_language,
    stats: Optional[SourceIngestionStats] = None,
) -> Tuple[Optional[Book], Optional[str]]:
    """
    Normalise one raw record and apply the quality filter.

    Args:
        book: Raw book from a BulkBooksSource
        rules: Filter parameters
        detector: Function guessing the language of a text
        stats: Optional counters updated with detected languages / dropped years

    Returns:
        (normalised book, None) if kept, or (None, rejection reason)
    """
    title = clean_title(book.title)
    if not title:
        return None, "no_title"

    authors: List[str] = []
    for name in book.authors:
        normalized = normalize_author_name(name)
        if normalized and normalized not in authors:
            authors.append(normalized)
    if not authors:
        return None, "no_author"

    metadata = book.metadata or BookMetadata()
    work_language = to_iso639_1(metadata.work_language or book.language)

    raw_descriptions = list(metadata.descriptions)
    if not raw_descriptions and book.description:
        raw_descriptions = [BookDescription(text=book.description, language=book.language, source=book.source)]
    if not raw_descriptions:
        return None, "no_description"

    descriptions: List[BookDescription] = []
    too_short = unsupported = 0
    for description in raw_descriptions:
        text = truncate_text(strip_markup(description.text), rules.max_description_chars)
        if len(text) < rules.min_description_chars:
            too_short += 1
            continue
        language = to_iso639_1(description.language) if description.language else None
        if language is None:
            language = detector(text)
            if language is None:
                language = work_language  # undecided text: assume the book's language
            elif stats is not None:
                stats.descriptions_detected += 1
        if language is None or language not in rules.languages:
            unsupported += 1
            continue
        descriptions.append(replace(description, text=text, language=language))
    if not descriptions:
        return None, "short_description" if too_short >= max(unsupported, 1) else "unsupported_language"

    ordered = order_descriptions(descriptions, work_language, rules.languages)
    primary = ordered[0]

    published = book.published_date
    if published is not None and not is_plausible_year(published.year, rules.max_year):
        published = None
        if stats is not None:
            stats.years_dropped += 1

    isbn13s = tuple(dict.fromkeys(
        i for i in (clean_isbn13(v) for v in (*metadata.isbn13s, metadata.isbn13, metadata.isbn)) if i
    ))
    new_metadata = replace(
        metadata,
        work_language=work_language,
        descriptions=ordered,
        isbn13=isbn13s[0] if isbn13s else None,
        isbn13s=isbn13s,
    )
    normalized = replace(
        book,
        title=title,
        authors=authors[:MAX_AUTHORS],
        description=primary.text,
        language=work_language or primary.language,
        categories=clean_subjects(book.categories, rules.max_subjects),
        published_date=published,
        metadata=new_metadata,
    )
    return normalized, None


# ------------------------------------------------------------ de-duplication

def match_keys(book: Book) -> List[Tuple[str, str]]:
    """
    Match keys of a normalised book, as (rule, key) pairs in rule order.

    Keys: 'wikidata:<QID>', 'openlibrary:<OLID>', 'gutenberg:<n>',
    'isbn13:<13 digits>' and 'title:<normalised title>|<first-author surname>'.
    """
    metadata = book.metadata or BookMetadata()
    keys: List[Tuple[str, str]] = []
    if metadata.wikidata_id:
        keys.append(("wikidata", f"wikidata:{metadata.wikidata_id}"))
    keys += [("openlibrary", f"openlibrary:{olid}") for olid in metadata.openlibrary_ids]
    keys += [("gutenberg", f"gutenberg:{gid}") for gid in metadata.gutenberg_ids]
    keys += [("isbn13", f"isbn13:{isbn}") for isbn in metadata.isbn13s]
    key = title_key(book.title)
    surname = author_surname(book.authors[0]) if book.authors else ""
    if key and surname:
        keys.append(("title", f"title:{key}|{surname}"))
    return keys


def can_merge(existing: Book, incoming: Book, rule: str, year_tolerance: int = 2) -> bool:
    """
    Check that a candidate found through a match key is really the same work.

    - Two works with different Wikidata ids are different works.
    - For the title/author rule, the years must be at most ``year_tolerance``
      apart when both are known.
    """
    old_qid = existing.metadata.wikidata_id if existing.metadata else None
    new_qid = incoming.metadata.wikidata_id if incoming.metadata else None
    if old_qid and new_qid and old_qid != new_qid:
        return False
    if rule == "title":
        old_year, new_year = existing.get_published_year(), incoming.get_published_year()
        if old_year is not None and new_year is not None and abs(old_year - new_year) > year_tolerance:
            return False
    return True


def merge_books(existing: Book, incoming: Book, rules: CatalogRules) -> Book:
    """
    Merge a normalised record into an existing work.

    The existing work keeps its id, source, title, authors and creation time.
    Descriptions, subjects, alternative titles, identifiers and provenance
    are unioned; missing scalar fields are filled from the incoming record;
    popularity signals keep the maximum. The primary description is chosen
    again with the merged descriptions.
    """
    old = existing.metadata or BookMetadata()
    new = incoming.metadata or BookMetadata()
    work_language = old.work_language or new.work_language
    descriptions = order_descriptions((*old.descriptions, *new.descriptions), work_language, rules.languages)

    def union(a: Tuple, b: Tuple) -> Tuple:
        return tuple(dict.fromkeys((*a, *b)))

    def maximum(a: Optional[int], b: Optional[int]) -> Optional[int]:
        values = [v for v in (a, b) if v is not None]
        return max(values) if values else None

    isbn13s = union(old.isbn13s, new.isbn13s)
    titles = union(old.titles, new.titles)
    if incoming.title != existing.title:
        titles = union(titles, ((incoming.language or "", incoming.title),))
    provenance = tuple({(r.source, r.record_id): r for r in (*old.provenance, *new.provenance)}.values())
    metadata = replace(
        old,
        isbn=old.isbn or new.isbn,
        isbn13=isbn13s[0] if isbn13s else None,
        publisher=old.publisher or new.publisher,
        page_count=old.page_count or new.page_count,
        thumbnail_url=old.thumbnail_url or new.thumbnail_url,
        preview_link=old.preview_link or new.preview_link,
        work_language=work_language,
        descriptions=descriptions,
        titles=titles,
        wikidata_id=old.wikidata_id or new.wikidata_id,
        openlibrary_ids=union(old.openlibrary_ids, new.openlibrary_ids),
        gutenberg_ids=union(old.gutenberg_ids, new.gutenberg_ids),
        isbn13s=isbn13s,
        doi=old.doi or new.doi,
        wikipedia_editions=maximum(old.wikipedia_editions, new.wikipedia_editions),
        gutenberg_downloads=maximum(old.gutenberg_downloads, new.gutenberg_downloads),
        provenance=provenance,
    )
    categories = clean_subjects([*existing.categories, *incoming.categories], rules.max_subjects)
    primary = descriptions[0] if descriptions else None
    return replace(
        existing,
        description=primary.text if primary else existing.description,
        language=work_language or (primary.language if primary else existing.language),
        categories=categories,
        published_date=existing.published_date or incoming.published_date,
        metadata=metadata,
        updated_at=datetime.now(UTC),
    )


def _source_family(source: str) -> str:
    """Catalog source of a provenance record ('wikipedia' text comes with 'wikidata')."""
    return "wikidata" if source == "wikipedia" else source


# ------------------------------------------------------------------- service

class CatalogBuildService:
    """
    Ingest bulk sources into a catalog with normalisation, filtering and merging.

    Sources must be ingested in a fixed order (Wikidata, Gutenberg, DOAB,
    Open Library): the first source that stores a work gives it its id,
    title and authors; later sources enrich it.
    """

    def __init__(
        self,
        catalog_repo: BookCatalogRepository,
        key_index: DuplicateKeyIndex,
        rules: CatalogRules = CatalogRules(),
        detector: Callable[[str], Optional[str]] = detect_language,
        batch_size: int = 1000,
    ) -> None:
        """
        Args:
            catalog_repo: Catalog where works are stored
            key_index: Persistent match-key index used for de-duplication
            rules: Normalisation and filter parameters
            detector: Language guesser for descriptions without a language
            batch_size: Records per batch (bounds memory use)
        """
        self._catalog = catalog_repo
        self._index = key_index
        self._rules = rules
        self._detector = detector
        self._batch_size = batch_size

    def ingest_source(
        self,
        source: BulkBooksSource,
        limit: Optional[int] = None,
        progress_every: int = 50_000,
    ) -> SourceIngestionStats:
        """
        Read, normalise, filter, de-duplicate and store every record of a source.

        Args:
            source: The bulk source (its dataset must be downloaded)
            limit: Optional maximum number of records read (quick trials)
            progress_every: Log progress every this many records

        Returns:
            Counters describing what happened to the source's records
        """
        name = source.get_source_name()
        stats = SourceIngestionStats(source=name)
        batch: List[Book] = []
        for raw in source.iter_books(limit=limit):
            stats.read += 1
            book, reason = normalize_record(raw, self._rules, self._detector, stats)
            if book is None:
                stats.rejected[reason] += 1
            else:
                batch.append(book)
            if len(batch) >= self._batch_size:
                self._store_batch(batch, stats)
                batch = []
            if stats.read % progress_every == 0:
                logger.info("%s: %d read, %d kept, %d merged", name, stats.read, stats.kept, stats.merged)
        if batch:
            self._store_batch(batch, stats)

        for reason, count in source.get_rejected_counts().items():
            stats.rejected[reason] += count
            stats.read += count
        logger.info("%s done: %s", name, stats.to_dict())
        return stats

    def _store_batch(self, batch: List[Book], stats: SourceIngestionStats) -> None:
        """De-duplicate one batch against the index (and itself) and save it."""
        keys_per_book = [match_keys(book) for book in batch]
        all_keys = list(dict.fromkeys(key for keys in keys_per_book for _, key in keys))
        found = self._index.lookup(all_keys)
        candidate_ids = list(dict.fromkeys(i for ids in found.values() for i in ids))
        existing = self._catalog.get_many(candidate_ids) if candidate_ids else {}

        pending: Dict[UUID, Book] = {}
        local_keys: Dict[str, List[UUID]] = {}
        new_entries: List[Tuple[str, UUID]] = []

        for book, keys in zip(batch, keys_per_book):
            match = self._find_match(book, keys, found, local_keys, existing, pending)
            if match is None:
                target_id = book.id
                pending[target_id] = book
                stats.inserted += 1
            else:
                target, rule = match
                target_id = target.id
                if target.id == book.id:
                    stats.already_present += 1
                else:
                    stats.merged += 1
                    stats.merged_by_rule[rule] += 1
                    families = {_source_family(r.source) for r in (target.metadata.provenance if target.metadata else ())}
                    if stats.source not in families:
                        stats.merged_cross_source += 1
                pending[target_id] = merge_books(target, book, self._rules)
            for _, key in keys:
                ids = local_keys.setdefault(key, [])
                if target_id not in ids:
                    ids.append(target_id)
                    new_entries.append((key, target_id))

        self._catalog.save_many(list(pending.values()))
        self._index.add(new_entries)

    def _find_match(
        self,
        book: Book,
        keys: List[Tuple[str, str]],
        found: Dict[str, List[UUID]],
        local_keys: Dict[str, List[UUID]],
        existing: Dict[UUID, Book],
        pending: Dict[UUID, Book],
    ) -> Optional[Tuple[Book, str]]:
        """First compatible work sharing a key, trying the rules in order."""
        for rule, key in keys:
            candidates = list(dict.fromkeys([*local_keys.get(key, []), *found.get(key, [])]))
            for candidate_id in candidates:
                candidate = pending.get(candidate_id) or existing.get(candidate_id)
                if candidate is not None and can_merge(candidate, book, rule, self._rules.year_tolerance):
                    return candidate, rule
        return None
