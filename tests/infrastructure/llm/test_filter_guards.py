"""Tests for the deterministic guards on LLM-extracted filters (v2)."""

import pytest

from app.domain.value_objects import SearchFilters
from app.infrastructure.llm.filter_guards import (
    guard_filters,
    query_requests_language,
    query_states_years,
)


@pytest.mark.parametrize("query,language", [
    ("novelas en español", "es"),
    ("libros en castellano sobre cocina", "es"),
    ("Spanish civil war novels in English translation", "en"),
    ("novels translated into Spanish", "es"),
    ("poesía traducida al inglés", "en"),
])
def test_explicit_language_requests_are_recognised(query, language):
    assert query_requests_language(query, language)


@pytest.mark.parametrize("query,language", [
    ("Italian pasta recipes homemade sauce", "it"),      # cuisine, not language
    ("French pastry desserts professional techniques", "fr"),
    ("cocina espanola tapas y pinchos", "es"),            # nationality
    ("novela guerra civil espanola", "es"),
    ("libros para aprender machine learning con python", "es"),  # query written in Spanish
    ("Spanish novels", "es"),
])
def test_nationality_or_query_language_is_not_a_request(query, language):
    assert not query_requests_language(query, language)


@pytest.mark.parametrize("query,expected", [
    ("novelas publicadas después de 1950", True),
    ("Spanish novels from the 90s", True),
    ("literatura espanola siglo XIX realismo", False),   # topic period, not publication date
    ("Victorian era mystery novels", False),
])
def test_years_need_digits_in_the_query(query, expected):
    assert query_states_years(query) is expected


def test_guard_keeps_only_stated_values_and_is_idempotent():
    extracted = SearchFilters(language="es", category="Fiction", min_year=1800, max_year=1899)
    guarded = guard_filters(extracted, "literatura espanola siglo XIX")
    assert guarded == SearchFilters(category="Fiction")
    assert guard_filters(guarded, "literatura espanola siglo XIX") == guarded

    stated = SearchFilters(language="en", min_year=1990, max_year=1999)
    assert guard_filters(stated, "novels in English from the 1990s") == stated
