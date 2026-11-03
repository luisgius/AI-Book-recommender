"""
Deterministic guards for filters extracted from a query by the LLM.

LLM output is a suggestion, not a fact: a filter only restricts the results
if the user's own words state it. The rules come from the LLM-path evaluation
(data/evaluation/results_llm_path_k10.json, v1 guards), where extracted hard
filters lowered nDCG@10 significantly. Typical errors were:

- nationality read as language: "Italian pasta recipes" -> language=it,
  "cocina española" -> language=es (a Spanish cookbook may be in English);
- topic period read as publication date: "literatura española siglo XIX"
  -> 1800-1899, which excludes modern studies about the 19th century.

v2 rules:
- a language filter is kept only if the query asks for the language with a
  preposition: "en español", "in English", "translated into Spanish";
- year filters are kept only if the query literally contains digits.

The guards are pure functions so they can be tested in isolation and
re-applied to cached LLM output during evaluation.
"""

import re
import unicodedata

from app.domain.value_objects import SearchFilters

FILTER_GUARDS_VERSION = "v2"

# Accent-folded language names per ISO 639-1 code (Spanish and English names)
LANGUAGE_NAMES = {
    "es": ("espanol", "castellano", "spanish"),
    "en": ("ingles", "english"),
    "ca": ("catalan", "valenciano", "valencian"),
    "fr": ("frances", "french"),
    "de": ("aleman", "german"),
    "it": ("italiano", "italian"),
    "pt": ("portugues", "portuguese"),
}

# "en español", "in English", "translated into Spanish", "traducido al inglés"
_LANGUAGE_REQUEST = r"\b(?:en|in|into|al|to)\s+(?:el\s+)?{name}\b"
_DIGITS = re.compile(r"\d{2,4}")


def fold(text: str) -> str:
    """Lowercase and strip accents, for accent-insensitive matching."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def query_requests_language(query: str, language: str) -> bool:
    """True if the query explicitly asks for books in that language."""
    folded = fold(query)
    return any(
        re.search(_LANGUAGE_REQUEST.format(name=name), folded)
        for name in LANGUAGE_NAMES.get(language, ())
    )


def query_states_years(query: str) -> bool:
    """True if the query literally contains a year or decade ("1950", "90s")."""
    return _DIGITS.search(query) is not None


def guard_filters(filters: SearchFilters, query: str) -> SearchFilters:
    """
    Drop the extracted filter values the query does not state explicitly.

    Category handling is separate (see LangChainLLMClient._sanitize_filters).
    Applying the guard twice gives the same result (idempotent).
    """
    language = filters.language
    if language is not None and not query_requests_language(query, language):
        language = None
    keep_years = query_states_years(query)
    return SearchFilters(
        language=language,
        category=filters.category,
        min_year=filters.min_year if keep_years else None,
        max_year=filters.max_year if keep_years else None,
    )
