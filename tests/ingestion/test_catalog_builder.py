"""
Tests for the multi-source catalog builder: quality filter, primary
description rule, de-duplication rules, merge and batching.
"""

from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple
from uuid import UUID, uuid4

import pytest

from app.domain.entities import Book
from app.domain.value_objects import BookDescription, BookMetadata, SourceLicense, SourceRecord
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
from app.infrastructure.db.sqlite_catalog_build_state import SqliteDuplicateKeyIndex
from app.ingestion.catalog_builder import (
    CatalogBuildService,
    CatalogRules,
    choose_primary_description,
    match_keys,
    merge_books,
    normalize_record,
)

LONG_EN = (
    "Don Quixote is a Spanish novel by Miguel de Cervantes. It follows the adventures of a hidalgo "
    "who reads so many chivalric romances that he loses his mind and decides to become a knight-errant "
    "to revive chivalry and serve his nation under the name Don Quixote de la Mancha."
)
LONG_ES = (
    "Don Quijote de la Mancha es una novela escrita por el español Miguel de Cervantes Saavedra. "
    "Publicada su primera parte con el título de El ingenioso hidalgo don Quijote de la Mancha a "
    "comienzos de 1605, es la obra más destacada de la literatura española y una de las principales."
)
LONG_FR = (
    "L'Ingénieux Hidalgo Don Quichotte de la Manche est un roman écrit par Miguel de Cervantes et "
    "publié à Madrid en deux parties, la première en 1605 et la seconde en 1615. Ce livre est considéré "
    "comme l'un des romans les plus importants de la littérature mondiale et du Siècle d'or."
)
RULES = CatalogRules()


def no_detection(_: str) -> Optional[str]:
    """Detector stub: never decides (descriptions keep the language they came with)."""
    return None


def make_book(
    source: str = "wikidata",
    source_id: Optional[str] = None,
    title: str = "Don Quijote de la Mancha",
    authors: Optional[List[str]] = None,
    descriptions: Tuple[BookDescription, ...] = (BookDescription(LONG_EN, "en", "wikipedia"),),
    year: Optional[int] = 1605,
    language: Optional[str] = None,
    **metadata,
) -> Book:
    """Raw book as a source adapter would yield it."""
    source_id = source_id or str(uuid4())
    meta = BookMetadata(
        descriptions=descriptions,
        work_language=language,
        provenance=(SourceRecord(source, source_id, "test", None),),
        **metadata,
    )
    return Book(
        id=uuid4(),
        title=title,
        authors=authors or ["Cervantes Saavedra, Miguel de, 1547-1616"],
        language=language,
        published_date=datetime(year, 1, 1) if year else None,
        source=source,
        source_id=source_id,
        metadata=meta,
    )


class ListSource:
    """BulkBooksSource over an in-memory list."""

    def __init__(self, name: str, books: List[Book], rejected: Optional[Dict[str, int]] = None) -> None:
        self._name = name
        self._books = books
        self._rejected = rejected or {}

    def get_source_name(self) -> str:
        return self._name

    def get_license(self) -> SourceLicense:
        return SourceLicense(self._name, self._name, "CC0", "https://example.org", "none")

    def iter_books(self, limit: Optional[int] = None) -> Iterator[Book]:
        yield from self._books[:limit]

    def get_rejected_counts(self) -> Dict[str, int]:
        return dict(self._rejected)


class InMemoryKeyIndex:
    """DuplicateKeyIndex kept in a dict (the SQLite one is tested separately)."""

    def __init__(self) -> None:
        self.keys: Dict[str, List[UUID]] = {}

    def lookup(self, keys: List[str]) -> Dict[str, List[UUID]]:
        return {k: list(self.keys[k]) for k in keys if k in self.keys}

    def add(self, entries: List[Tuple[str, UUID]]) -> None:
        for key, book_id in entries:
            ids = self.keys.setdefault(key, [])
            if book_id not in ids:
                ids.append(book_id)


@pytest.fixture
def catalog(tmp_path):
    return SqliteBookCatalogRepository(tmp_path / "catalog.db", compact_json=True)


def build(catalog, sources, batch_size=1000, index=None):
    service = CatalogBuildService(catalog, index or InMemoryKeyIndex(), RULES, no_detection, batch_size)
    return [service.ingest_source(source) for source in sources]


# ----------------------------------------------------------------- filter

class TestQualityFilter:
    def test_keeps_a_good_record_and_normalises_it(self):
        book, reason = normalize_record(make_book(), RULES, no_detection)
        assert reason is None
        assert book.authors == ["Miguel de Cervantes Saavedra"]
        assert book.description == LONG_EN
        assert book.language == "en"  # no work language: primary description's

    def test_rejects_without_description(self):
        _, reason = normalize_record(make_book(descriptions=()), RULES, no_detection)
        assert reason == "no_description"

    def test_rejects_short_description(self):
        short = (BookDescription("A short novel.", "en", "wikipedia"),)
        _, reason = normalize_record(make_book(descriptions=short), RULES, no_detection)
        assert reason == "short_description"

    def test_length_is_measured_after_stripping_markup(self):
        padded = "<p>" + "<b></b>" * 100 + "Short text.</p>"
        assert len(padded) > 200
        _, reason = normalize_record(make_book(descriptions=(BookDescription(padded, "en", "x"),)), RULES, no_detection)
        assert reason == "short_description"

    def test_rejects_description_in_unconfigured_language(self):
        polish = (BookDescription(LONG_EN, "pl", "doab"),)
        _, reason = normalize_record(make_book(descriptions=polish), RULES, no_detection)
        assert reason == "unsupported_language"

    def test_rejects_when_language_cannot_be_determined(self):
        unknown = (BookDescription(LONG_EN, None, "open_library"),)
        _, reason = normalize_record(make_book(descriptions=unknown), RULES, no_detection)
        assert reason == "unsupported_language"

    def test_detects_missing_description_language(self):
        unknown = (BookDescription(LONG_ES, None, "open_library"),)
        book, _ = normalize_record(make_book(descriptions=unknown), RULES, lambda text: "es")
        assert book.metadata.descriptions[0].language == "es"

    def test_rejects_without_author(self):
        book = make_book()
        book.authors = ["  "]
        _, reason = normalize_record(book, RULES, no_detection)
        assert reason == "no_author"

    def test_truncates_long_descriptions(self):
        long = (BookDescription(LONG_EN * 10, "en", "wikipedia"),)
        book, _ = normalize_record(make_book(descriptions=long), RULES, no_detection)
        assert 200 <= len(book.description) <= RULES.max_description_chars

    def test_implausible_year_is_dropped_not_the_record(self):
        book, reason = normalize_record(make_book(year=2999), RULES, no_detection)
        assert reason is None and book.published_date is None

    def test_cleans_and_caps_subjects(self):
        raw = make_book()
        raw.categories = ["Accessible book"] + [f"Subject {i}" for i in range(15)]
        book, _ = normalize_record(raw, RULES, no_detection)
        assert book.categories[0] == "Subject 0" and len(book.categories) == 10

    def test_language_codes_are_normalised(self):
        upper = (BookDescription(LONG_ES, "ES", "doab"),)
        book, _ = normalize_record(make_book(descriptions=upper, language="ES"), RULES, no_detection)
        assert book.language == "es" and book.metadata.work_language == "es"
        assert book.metadata.descriptions[0].language == "es"


# ----------------------------------------------------- primary description

class TestPrimaryDescription:
    descriptions = (
        BookDescription(LONG_FR, "fr", "wikipedia"),
        BookDescription(LONG_EN, "en", "wikipedia"),
        BookDescription(LONG_ES, "es", "wikipedia"),
    )

    def test_prefers_the_work_language(self):
        assert choose_primary_description(self.descriptions, "es", RULES.languages).language == "es"

    def test_falls_back_to_english(self):
        assert choose_primary_description(self.descriptions, "ru", RULES.languages).language == "en"
        assert choose_primary_description(self.descriptions, None, RULES.languages).language == "en"

    def test_then_language_priority(self):
        no_english = tuple(d for d in self.descriptions if d.language != "en")
        assert choose_primary_description(no_english, None, RULES.languages).language == "es"

    def test_normalised_book_lists_primary_first(self):
        book, _ = normalize_record(make_book(descriptions=self.descriptions, language="es"), RULES, no_detection)
        assert book.description == LONG_ES
        assert [d.language for d in book.metadata.descriptions] == ["es", "en", "fr"]

    def test_human_text_beats_generated_text_in_the_same_language(self):
        auto = BookDescription(LONG_EN + " Generated.", "en", "gutenberg", auto_generated=True)
        human = BookDescription(LONG_EN, "en", "open_library")
        book, _ = normalize_record(make_book(descriptions=(auto, human)), RULES, no_detection)
        assert book.metadata.descriptions == (human,)


# ------------------------------------------------------------ dedup rules

class TestDeduplication:
    def test_match_keys_in_rule_order(self):
        book, _ = normalize_record(
            make_book(wikidata_id="Q480", openlibrary_ids=("OL1W",), gutenberg_ids=("2000",),
                      isbn13s=("9788437604947",)),
            RULES, no_detection,
        )
        assert [rule for rule, _ in match_keys(book)] == ["wikidata", "openlibrary", "gutenberg", "isbn13", "title"]
        assert match_keys(book)[-1][1] == "title:don quijote de la mancha|saavedra"

    @pytest.mark.parametrize(
        "rule, first, second",
        [
            ("wikidata", {"wikidata_id": "Q480"}, {"wikidata_id": "Q480"}),
            ("openlibrary", {"openlibrary_ids": ("OL1W",)}, {"openlibrary_ids": ("OL1W",)}),
            ("gutenberg", {"gutenberg_ids": ("2000",)}, {"gutenberg_ids": ("2000",)}),
            ("isbn13", {"isbn13s": ("9788437604947",)}, {"isbn13s": ("9788437604947",)}),
        ],
    )
    def test_identifier_rules_merge(self, catalog, rule, first, second):
        a = make_book("wikidata", title="Title A", **first)
        b = make_book("gutenberg", title="Completely different title", **second)
        stats = build(catalog, [ListSource("wikidata", [a]), ListSource("gutenberg", [b])])
        assert catalog.count() == 1
        assert stats[1].merged == 1 and stats[1].merged_by_rule == {rule: 1}
        assert stats[1].merged_cross_source == 1

    def test_title_rule_merges_within_two_years(self, catalog):
        a = make_book("wikidata", year=1605)
        b = make_book("doab", year=1607, title="Don Quijote de la Mancha: edición crítica")
        stats = build(catalog, [ListSource("wikidata", [a]), ListSource("doab", [b])])
        assert catalog.count() == 1 and stats[1].merged_by_rule == {"title": 1}

    def test_title_rule_does_not_merge_distant_years(self, catalog):
        a = make_book("wikidata", year=1605)
        b = make_book("doab", year=1980)
        build(catalog, [ListSource("wikidata", [a]), ListSource("doab", [b])])
        assert catalog.count() == 2

    def test_title_rule_merges_when_a_year_is_missing(self, catalog):
        a = make_book("wikidata", year=1605)
        b = make_book("gutenberg", year=None)
        build(catalog, [ListSource("wikidata", [a]), ListSource("gutenberg", [b])])
        assert catalog.count() == 1

    def test_different_wikidata_ids_never_merge(self, catalog):
        a = make_book("wikidata", wikidata_id="Q1", gutenberg_ids=("10",))
        b = make_book("wikidata", wikidata_id="Q2", gutenberg_ids=("10",))
        build(catalog, [ListSource("wikidata", [a, b])])
        assert catalog.count() == 2

    def test_duplicates_inside_one_batch_are_merged(self, catalog):
        a = make_book("gutenberg", gutenberg_ids=("1",), title="Faust")
        b = make_book("gutenberg", gutenberg_ids=("2",), title="Faust: Der Tragödie erster Teil", year=None)
        stats = build(catalog, [ListSource("gutenberg", [a, b])])
        assert catalog.count() == 1
        assert stats[0].merged == 1 and stats[0].merged_cross_source == 0

    def test_rerun_of_the_same_records_is_idempotent(self, catalog):
        index = InMemoryKeyIndex()
        book = make_book("wikidata", wikidata_id="Q480")
        build(catalog, [ListSource("wikidata", [book])], index=index)
        stats = build(catalog, [ListSource("wikidata", [book])], index=index)
        assert catalog.count() == 1
        assert stats[0].already_present == 1 and stats[0].merged == 0


# ------------------------------------------------------------------ merge

class TestMerge:
    def test_union_of_descriptions_identifiers_and_provenance(self, catalog):
        a = make_book("wikidata", wikidata_id="Q480", language="es",
                      descriptions=(BookDescription(LONG_ES, "es", "wikipedia"),), wikipedia_editions=125)
        b = make_book("gutenberg", gutenberg_ids=("2000",), wikidata_id=None, year=None,
                      descriptions=(BookDescription(LONG_EN, "en", "gutenberg", auto_generated=True),),
                      gutenberg_downloads=900)
        a_norm, _ = normalize_record(a, RULES, no_detection)
        b_norm, _ = normalize_record(b, RULES, no_detection)
        merged = merge_books(a_norm, b_norm, RULES)
        meta = merged.metadata
        assert merged.id == a_norm.id and merged.source == "wikidata"
        assert merged.description == LONG_ES  # the work is Spanish
        assert [d.language for d in meta.descriptions] == ["es", "en"]
        assert meta.wikidata_id == "Q480" and meta.gutenberg_ids == ("2000",)
        assert meta.wikipedia_editions == 125 and meta.gutenberg_downloads == 900
        assert [r.source for r in meta.provenance] == ["wikidata", "gutenberg"]

    def test_merged_book_round_trips_through_sqlite(self, catalog):
        a = make_book("wikidata", wikidata_id="Q480", language="es",
                      descriptions=(BookDescription(LONG_ES, "es", "wikipedia", "CC BY-SA 4.0", "https://es.wikipedia.org/wiki/X"),))
        b = make_book("gutenberg", wikidata_id="Q480", descriptions=(BookDescription(LONG_FR, "fr", "gutenberg"),))
        build(catalog, [ListSource("wikidata", [a]), ListSource("gutenberg", [b])])
        [stored] = catalog.get_all()
        assert stored.metadata.description_languages() == ["es", "fr"]
        assert stored.metadata.descriptions[0].attribution_url == "https://es.wikipedia.org/wiki/X"
        assert stored.metadata.sources() == ["wikidata", "gutenberg"]


# --------------------------------------------------------------- batching

class TestBatching:
    def make_records(self) -> List[Book]:
        books = []
        for i in range(25):
            books.append(make_book("gutenberg", title=f"Work number {i}", gutenberg_ids=(str(i),), year=None))
        # Duplicates of every fifth work, by Gutenberg id.
        for i in range(0, 25, 5):
            books.append(make_book("gutenberg", title=f"Another edition {i}", gutenberg_ids=(str(i),), year=None))
        return books

    @pytest.mark.parametrize("batch_size", [1, 4, 7, 1000])
    def test_batch_size_does_not_change_the_result(self, tmp_path, batch_size):
        repo = SqliteBookCatalogRepository(tmp_path / f"c{batch_size}.db", compact_json=True)
        index = SqliteDuplicateKeyIndex(tmp_path / f"k{batch_size}.db")
        [stats] = build(repo, [ListSource("gutenberg", self.make_records())], batch_size, index)
        assert repo.count() == 25
        assert stats.inserted == 25 and stats.merged == 5
        titles = sorted(book.title for book in repo.get_all())
        assert titles == sorted(f"Work number {i}" for i in range(25))

    def test_source_rejections_are_counted_as_read(self, catalog):
        source = ListSource("doab", [make_book("doab")], rejected={"not_a_book": 3})
        [stats] = build(catalog, [source])
        assert stats.read == 4 and stats.rejected["not_a_book"] == 3 and stats.kept == 1
