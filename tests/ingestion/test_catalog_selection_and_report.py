"""Tests for the small-catalog selection rule and the catalog report."""

import random
from datetime import datetime
from uuid import UUID, uuid5, NAMESPACE_URL

from app.domain.entities import Book
from app.domain.value_objects import BookDescription, BookMetadata, SourceRecord
from app.ingestion.catalog_report import compute_catalog_report, render_markdown
from app.ingestion.catalog_selection import (
    SelectionCandidate,
    language_mix,
    language_quotas,
    popularity_score,
    select_books,
    to_candidate,
)

LANGUAGES = ("en", "es", "fr", "de", "it", "pt", "ca", "nl", "ru", "zh", "ja", "ar")


def candidate(i: int, language: str, score: float, langs: int = 1) -> SelectionCandidate:
    return SelectionCandidate(uuid5(NAMESPACE_URL, f"book:{i}"), language, score, langs)


def make_pool():
    pool = []
    i = 0
    for language, n in [("en", 500), ("es", 100), ("fr", 40), ("de", 5), ("la", 30)]:
        for _ in range(n):
            pool.append(candidate(i, language, float(i % 37), 1 + i % 3))
            i += 1
    return pool


class TestSelection:
    def test_quotas_add_up_and_favour_english_and_spanish(self):
        quotas = language_quotas(5000, LANGUAGES)
        assert sum(quotas.values()) == 5000
        assert quotas["en"] == 2000 and quotas["es"] == 1250
        assert all(quotas[lang] == 175 for lang in LANGUAGES[2:])

    def test_selection_is_deterministic_whatever_the_input_order(self):
        pool = make_pool()
        first = [c.book_id for c in select_books(pool, 200, LANGUAGES)]
        shuffled = pool[:]
        random.Random(7).shuffle(shuffled)
        assert [c.book_id for c in select_books(shuffled, 200, LANGUAGES)] == first

    def test_quotas_then_leftovers_by_score(self):
        selected = select_books(make_pool(), 200, LANGUAGES)
        mix = language_mix(selected)
        assert len(selected) == 200
        assert mix["es"] == 50  # 25% quota filled
        assert mix["de"] == 5  # all German works (quota 7, only 5 exist)
        assert mix["fr"] >= 7  # quota, plus any leftover places won by score
        assert mix["en"] >= 80  # 40% quota plus part of the leftovers

    def test_a_larger_selection_contains_every_smaller_one(self):
        """
        V18: enlarging the deployed catalog keeps the works already loaded.
        Quotas only grow with the size and each language's ranking is fixed,
        so the quota places are nested; the places left unused grow at least
        as much, so the leftover works chosen before are chosen again.
        """
        pool = make_pool()
        # es, fr and de run short of their quotas at the larger sizes, so the
        # leftovers (the Latin works among them) matter here.
        sizes = [60, 120, 200, 300, 400, 600, 675]
        selections = [{c.book_id for c in select_books(pool, size, LANGUAGES)} for size in sizes]
        for size, smaller, larger in zip(sizes, selections, selections[1:]):
            assert len(smaller) == size and smaller <= larger

    def test_highest_scores_first_within_a_language(self):
        pool = [candidate(1, "es", 1.0), candidate(2, "es", 9.0), candidate(3, "es", 5.0, langs=4),
                candidate(4, "es", 5.0, langs=1)]
        ids = [c.book_id for c in select_books(pool, 3, ["es"])]
        assert ids == [pool[1].book_id, pool[2].book_id, pool[3].book_id]

    def test_stated_language_comes_before_inferred_language(self):
        inferred = SelectionCandidate(uuid5(NAMESPACE_URL, "a"), "es", 50.0, 1, language_stated=False)
        stated = SelectionCandidate(uuid5(NAMESPACE_URL, "b"), "es", 2.0, 1, language_stated=True)
        assert select_books([inferred, stated], 1, ["es"]) == [stated]

    def test_language_is_stated_only_when_the_source_gave_it(self):
        stated = Book(id=uuid5(NAMESPACE_URL, "s"), title="T", authors=["A"], language="es",
                      metadata=BookMetadata(work_language="es"))
        inferred = Book(id=uuid5(NAMESPACE_URL, "i"), title="T", authors=["A"], language="es",
                        metadata=BookMetadata())
        assert to_candidate(stated).language_stated and not to_candidate(inferred).language_stated

    def test_popularity_score(self):
        book = Book(id=uuid5(NAMESPACE_URL, "x"), title="T", authors=["A"], language="es",
                    metadata=BookMetadata(wikipedia_editions=12, gutenberg_downloads=2500))
        assert popularity_score(book) == 14.5
        assert to_candidate(book).language == "es"


def report_books():
    es = BookDescription("Texto " * 50, "es", "wikipedia", "CC BY-SA 4.0")
    en = BookDescription("Text " * 60, "en", "gutenberg", None, None, True)
    return [
        Book(id=UUID(int=1), title="Obra", authors=["A"], description=es.text, language="es",
             categories=["Novela"], published_date=datetime(1605, 1, 1),
             metadata=BookMetadata(work_language="es", descriptions=(es, en), wikidata_id="Q1",
                                   gutenberg_ids=("1",), wikipedia_editions=3,
                                   provenance=(SourceRecord("wikidata", "Q1"), SourceRecord("wikipedia", "es:Obra"),
                                               SourceRecord("gutenberg", "1")))),
        Book(id=UUID(int=2), title="Work", authors=["B"], description=en.text, language="en",
             categories=["Novela", "History"],
             metadata=BookMetadata(descriptions=(en,), isbn13s=("9788437604947",),
                                   provenance=(SourceRecord("doab", "u1"),))),
    ]


class TestReport:
    def test_counts(self):
        sources = [{"source": "wikidata", "status": "done", "stats": {"read": 10, "kept": 1, "rejected": {"no_author": 9}},
                    "license": {"name": "Wikidata", "license": "CC0"}}]
        report = compute_catalog_report(report_books(), sources, LANGUAGES)
        assert report["total_works"] == 2
        assert report["works_with_more_than_one_source"] == 1
        assert report["works_by_source_combination"] == {"gutenberg + wikidata": 1, "doab": 1}
        assert report["work_language"]["distribution"] == {"es": 1, "en": 1}
        assert report["work_language"]["stated_by_a_source"] == 1
        dl = report["description_language"]
        assert dl["works_with_a_description_in"]["en"] == 2 and dl["works_with_a_description_in"]["es"] == 1
        assert dl["works_with_more_than_one_description_language"] == 1
        assert dl["primary_description_auto_generated"] == 1
        assert report["coverage"]["year"] == 0.5 and report["coverage"]["isbn13"] == 0.5
        assert report["top_subjects"]["Novela"] == 2
        assert report["sources"][0]["works_with_this_source"] == 1

    def test_markdown_renders(self):
        report = compute_catalog_report(report_books(), [], LANGUAGES)
        text = render_markdown(report)
        assert text.startswith("# Multi-source catalog report")
        assert "## Licences and attribution" in text
