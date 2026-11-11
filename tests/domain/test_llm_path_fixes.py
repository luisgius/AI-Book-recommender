"""
Tests for the fixes that make the LLM search path work end to end.

Covers:
1. One filter definition (Book.matches_filters) shared by service and repositories.
2. Multi-query fusion returns the whole pool, keeps component scores, reports
   real candidate counts and is deterministic.
3. Filters and MMR in search_with_understanding work on the whole fused pool.
4. Explanations: configurable count, degraded mode, original query wording.
5. Explicit user filters take precedence over LLM-extracted ones.
"""

import time
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import pytest

from app.domain.entities import Book, Explanation, SearchResult
from app.domain.services import SearchService
from app.domain.value_objects import (
    QueryIntent,
    RetrievalStrategy,
    SearchFilters,
    SearchQuery,
)


# =============================================================================
# Fakes
# =============================================================================


def make_book(title: str, category: str = "Computers", language: str = "en",
              year: Optional[int] = 2020) -> Book:
    from datetime import datetime
    return Book(
        id=uuid4(),
        title=title,
        authors=["Author"],
        categories=[category],
        language=language,
        published_date=datetime(year, 1, 1) if year else None,
    )


class OrderedRepo:
    """
    Lexical/vector fake: returns its books in a fixed order.

    Like the real repositories, it applies the filters before cutting to
    max_results (using the shared Book.matches_filters rule).
    """

    def __init__(self, books: List[Book], source: str):
        self._books = books
        self._source = source
        self.calls: List[Dict] = []

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        self.calls.append({"query_text": query_text, "max_results": max_results})
        matching = [b for b in self._books if b.matches_filters(filters)]
        score_field = "lexical_score" if self._source == "lexical" else "vector_score"
        return [
            SearchResult(book=book, final_score=1.0 / (i + 1), rank=i + 1,
                         source=self._source, **{score_field: 1.0 / (i + 1)})
            for i, book in enumerate(matching[:max_results])
        ]

    def is_ready(self) -> bool:
        return True


class NotReady:
    def is_ready(self) -> bool:
        return False


class FakeEmbeddings:
    """Same embedding for every book unless a per-book vector is given."""

    def __init__(self, vectors: Optional[Dict[UUID, List[float]]] = None):
        self._vectors = vectors or {}

    def generate_embedding(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

    generate_query_embedding = generate_embedding  # same vector for queries

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return self._vectors.get(book_id, [0.1, 0.2, 0.3])

    def is_ready(self) -> bool:
        return True


class StubLLM:
    """LLM client double: fixed intent, records explanation requests."""

    def __init__(self, intent: Optional[QueryIntent] = None):
        self.intent = intent
        self.explained_queries: List[str] = []

    def extract_query_intent(self, query_text: str) -> QueryIntent:
        return self.intent

    def generate_grounded_explanation(self, query_text: str, book: Book) -> Explanation:
        self.explained_queries.append(query_text)
        return Explanation(book_id=book.id, query_text=query_text, text="because")

    def get_model_name(self) -> str:
        return "stub"


def intent(filters: SearchFilters = SearchFilters(), intent_type="recommendation",
           original="original words", reformulated="reformulated words", variations=None):
    return QueryIntent(
        intent_type=intent_type, original_query=original, reformulated_query=reformulated,
        extracted_filters=filters, confidence=0.9, reasoning="stub",
        query_variations=variations or [],
    )


# =============================================================================
# 1. Filter semantics
# =============================================================================


class TestBookMatchesFilters:
    def test_category_is_case_insensitive_substring(self):
        book = make_book("Novela", category="Spanish fiction", language="es")
        assert book.matches_filters(SearchFilters(category="fiction"))
        assert book.matches_filters(SearchFilters(category="FICTION"))
        assert not book.matches_filters(SearchFilters(category="History"))

    def test_language_is_case_insensitive(self):
        book = make_book("Novela", language="es")
        assert book.matches_filters(SearchFilters(language="ES"))
        assert not book.matches_filters(SearchFilters(language="en"))

    def test_book_without_year_fails_year_filters(self):
        book = make_book("Undated", year=None)
        assert not book.matches_filters(SearchFilters(min_year=1900))
        assert not book.matches_filters(SearchFilters(max_year=2000))

    def test_empty_or_missing_filters_match(self):
        book = make_book("Any")
        assert book.matches_filters(None)
        assert book.matches_filters(SearchFilters())


def test_service_keeps_what_repositories_accept():
    """Regression: the service used exact, case-sensitive category matching."""
    books = [make_book("A", category="Spanish fiction"), make_book("B", category="Historical fiction"),
             make_book("C", category="Cooking")]
    service = SearchService(OrderedRepo(books, "lexical"), OrderedRepo(books, "vector"), FakeEmbeddings())
    results = [SearchResult(book=b, final_score=1.0, rank=i + 1, source="hybrid") for i, b in enumerate(books)]

    kept = service._apply_filters(results, SearchFilters(category="fiction"))

    assert [r.book.title for r in kept] == ["A", "B"]


class TestOverriddenBy:
    def test_explicit_values_win_field_by_field(self):
        extracted = SearchFilters(language="es", category="Fiction", min_year=1950)
        merged = extracted.overridden_by(SearchFilters(language="en", max_year=2000))
        assert merged == SearchFilters(language="en", category="Fiction", min_year=1950, max_year=2000)

    def test_no_explicit_filters_keeps_extracted(self):
        extracted = SearchFilters(language="es")
        assert extracted.overridden_by(None) is extracted
        assert extracted.overridden_by(SearchFilters()) is extracted

    def test_contradictory_combination_falls_back_to_explicit(self):
        explicit = SearchFilters(max_year=1900)
        assert SearchFilters(min_year=1950).overridden_by(explicit) == explicit


# =============================================================================
# 2. Multi-query fusion
# =============================================================================


class TestMultiQueryFusion:
    def test_returns_whole_pool_not_top_k(self):
        books = [make_book(f"B{i}") for i in range(30)]
        service = SearchService(OrderedRepo(books, "lexical"), OrderedRepo(books, "vector"), FakeEmbeddings())

        fused, _ = service._multi_query_search(
            variations=["q1"], strategy=RetrievalStrategy.BALANCED, query=SearchQuery(text="q", max_results=5)
        )

        assert len(fused) > 5

    def test_keeps_component_scores_and_counts_candidates(self):
        lexical_books = [make_book(f"L{i}") for i in range(4)]
        vector_books = lexical_books[:2] + [make_book(f"V{i}") for i in range(3)]
        service = SearchService(OrderedRepo(lexical_books, "lexical"), OrderedRepo(vector_books, "vector"),
                                FakeEmbeddings())

        fused, meta = service._multi_query_search(
            variations=["q1", "q2"], strategy=RetrievalStrategy.BALANCED, query=SearchQuery(text="q")
        )

        assert meta["candidates_lexical"] == 4
        assert meta["candidates_vector"] == 5
        both = {b.id for b in lexical_books[:2]}
        assert all(r.lexical_score is not None and r.vector_score is not None
                   for r in fused if r.book.id in both)

    def test_ranking_follows_variation_order_not_completion_order(self):
        a, b = make_book("A"), make_book("B")
        # Variation "first" finishes last; its book must still win the RRF tie
        lexical = {"first": [a], "second": [b]}

        class PerQueryRepo(OrderedRepo):
            def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
                time.sleep(0.05 if query_text == "first" else 0.0)
                return [SearchResult(book=lexical[query_text][0], final_score=1.0, rank=1,
                                     source="lexical", lexical_score=1.0)]

        service = SearchService(PerQueryRepo([], "lexical"), OrderedRepo([], "vector"), FakeEmbeddings())

        fused, _ = service._multi_query_search(
            variations=["first", "second"], strategy=RetrievalStrategy.BALANCED, query=SearchQuery(text="q")
        )

        assert [r.book.title for r in fused] == ["A", "B"]

    def test_fuse_cut_off_is_optional(self):
        books = [make_book(f"B{i}") for i in range(6)]
        service = SearchService(OrderedRepo(books, "lexical"), OrderedRepo(books, "vector"), FakeEmbeddings())
        ranked = [SearchResult(book=b, final_score=1.0, rank=i + 1, source="hybrid") for i, b in enumerate(books)]

        assert len(service._fuse_multi_query_results([ranked])) == 6
        assert len(service._fuse_multi_query_results([ranked], max_results=2)) == 2


# =============================================================================
# 3. Filters and selection on the whole pool
# =============================================================================


def test_understanding_returns_only_matching_books_ranked_from_one():
    others = [make_book(f"Other{i}", category="Cooking") for i in range(20)]
    matching = [make_book(f"Match{i}", category="Spanish fiction") for i in range(10)]
    books = others + matching
    llm = StubLLM(intent(filters=SearchFilters(category="fiction")))
    service = SearchService(OrderedRepo(books, "lexical"), OrderedRepo(books, "vector"), FakeEmbeddings(),
                            llm_client=llm)

    response, _ = service.search_with_understanding("novelas", max_results=5)

    assert len(response.results) == 5
    assert all(r.book.title.startswith("Match") for r in response.results)
    assert [r.rank for r in response.results] == [1, 2, 3, 4, 5]


def test_mmr_can_select_beyond_the_rrf_top_k():
    """
    Regression: the fused pool was cut to max_results before MMR, so MMR
    could only reorder the same books. Here the RRF top-5 are near-duplicates
    and lower-ranked books are diverse: MMR must reach past the top-5.
    """
    books = [make_book(f"B{i}") for i in range(10)]
    vectors = {b.id: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] for b in books[:5]}
    for i, b in enumerate(books[5:], start=1):
        one_hot = [0.0] * 6
        one_hot[i] = 1.0
        vectors[b.id] = one_hot
    llm = StubLLM(intent(intent_type="exploratory"))  # exploratory forces MMR
    service = SearchService(OrderedRepo(books, "lexical"), OrderedRepo(books, "vector"),
                            FakeEmbeddings(vectors), llm_client=llm)

    response, _ = service.search_with_understanding("q", max_results=5)

    titles = {r.book.title for r in response.results}
    assert response.metadata.diversification_enabled
    assert titles - {f"B{i}" for i in range(5)}, "MMR never left the RRF top-5"


def test_understanding_metadata_reports_candidates_not_variations():
    books = [make_book(f"B{i}") for i in range(8)]
    llm = StubLLM(intent(variations=["v1", "v2"]))
    service = SearchService(OrderedRepo(books, "lexical"), OrderedRepo(books, "vector"), FakeEmbeddings(),
                            llm_client=llm)

    response, _ = service.search_with_understanding("q", max_results=3)

    # recommendation -> VECTOR_HEAVY: BM25 pool 3x1, vector pool 3x4 (capped by 8 books);
    # the old code reported the number of variations (2) for both
    assert response.metadata.candidates_lexical == 3
    assert response.metadata.candidates_vector == 8


# =============================================================================
# 4. Explanations
# =============================================================================


def test_max_explanations_limits_llm_calls():
    books = [make_book(f"B{i}") for i in range(6)]
    llm = StubLLM()
    service = SearchService(OrderedRepo(books, "lexical"), OrderedRepo(books, "vector"), FakeEmbeddings(),
                            llm_client=llm, max_explanations=2)

    results = service.search(SearchQuery(text="q", max_results=5, use_explanations=True))

    assert len(llm.explained_queries) == 2
    assert [r.explanation is not None for r in results] == [True, True, False, False, False]


def test_negative_max_explanations_is_rejected():
    with pytest.raises(ValueError):
        SearchService(NotReady(), NotReady(), FakeEmbeddings(), max_explanations=-1)


def test_degraded_lexical_mode_still_explains():
    books = [make_book(f"B{i}") for i in range(4)]
    llm = StubLLM()
    service = SearchService(OrderedRepo(books, "lexical"), NotReady(), FakeEmbeddings(), llm_client=llm,
                            max_explanations=3)

    response = service.search_with_fallback(SearchQuery(text="q", max_results=4, use_explanations=True))

    assert response.search_mode == "lexical_only"
    assert sum(r.explanation is not None for r in response.results) == 3


def test_understanding_fallback_explains_with_original_query():
    books = [make_book(f"B{i}") for i in range(4)]
    llm = StubLLM(intent(original="libros de cocina", reformulated="cocina recetas"))
    service = SearchService(OrderedRepo(books, "lexical"), NotReady(), FakeEmbeddings(), llm_client=llm,
                            max_explanations=2)

    response, _ = service.search_with_understanding("libros de cocina", max_results=4, use_explanations=True)

    assert response.degraded
    assert llm.explained_queries == ["libros de cocina", "libros de cocina"]


# =============================================================================
# 5. Explicit filters win over extracted ones
# =============================================================================


def test_explicit_filters_override_extracted_ones():
    books = [make_book("Spanish", language="es"), make_book("English", language="en")]
    llm = StubLLM(intent(filters=SearchFilters(language="es")))
    service = SearchService(OrderedRepo(books, "lexical"), OrderedRepo(books, "vector"), FakeEmbeddings(),
                            llm_client=llm)

    response, _ = service.search_with_understanding("q", max_results=5, filters=SearchFilters(language="en"))

    assert [r.book.title for r in response.results] == ["English"]
