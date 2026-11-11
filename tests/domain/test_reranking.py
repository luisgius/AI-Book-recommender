"""
Tests for the optional re-ranking stage (Reranker port) in SearchService.

Pipeline under test: retrieve -> fuse (RRF) -> filter -> re-rank the top
`rerank_depth` -> optional MMR -> cut to max_results.

The fakes make the fused order predictable: both repositories return the
books in catalog order ("Book 00", "Book 01", ...), so RRF keeps that order.
The fake reranker scores each result with its fused rank, which reverses the
head it receives. No model is loaded.
"""

import logging
from dataclasses import replace
from datetime import datetime
from typing import Callable, List, Optional
from uuid import UUID

import pytest

from app.domain.entities import Book, SearchResult
from app.domain.services import DEFAULT_RERANK_DEPTH, SearchService
from app.domain.value_objects import (
    QueryIntent,
    SearchFilters,
    SearchQuery,
    STRATEGY_POOL_SIZES,
    RetrievalStrategy,
)


# =============================================================================
# Fakes
# =============================================================================


def make_books(n: int = 50) -> List[Book]:
    """Books 'Book 00'..'Book NN'; even ones in English, odd ones in Spanish."""
    return [
        Book.create_new(
            title=f"Book {i:02d}",
            authors=["Author"],
            language="en" if i % 2 == 0 else "es",
            categories=["Fiction"],
            published_date=datetime(2000 + i % 20, 1, 1),
            description=f"Description of book {i}",
        )
        for i in range(n)
    ]


class RecordingRepo:
    """
    Lexical or vector repository returning the books in catalog order.

    It ignores filters on purpose, so the tests can check that the service's
    own filter pass runs before the reranker. Every requested max_results is
    recorded.
    """

    def __init__(self, books: List[Book], source: str, ready: bool = True):
        self._books = books
        self._source = source
        self._ready = ready
        self.requested: List[int] = []

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        self.requested.append(max_results)
        score_field = "lexical_score" if self._source == "lexical" else "vector_score"
        return [
            SearchResult(
                book=book,
                final_score=1.0 / (i + 1),
                rank=i + 1,
                source=self._source,
                **{score_field: 1.0 / (i + 1)},
            )
            for i, book in enumerate(self._books[:max_results])
        ]

    def is_ready(self) -> bool:
        return self._ready


class FakeEmbeddings:
    """Gives every book a distinct embedding (needed by MMR)."""

    def __init__(self, books: List[Book]):
        self._vectors = {book.id: [1.0, float(i)] for i, book in enumerate(books)}

    def generate_query_embedding(self, text: str) -> List[float]:
        return [1.0, 0.0]

    generate_embedding = generate_query_embedding

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return self._vectors.get(book_id)

    def is_ready(self) -> bool:
        return True


class FakeReranker:
    """
    Records what it receives and orders results by score_of (descending).

    The default score is the fused rank, which reverses the received head.
    """

    def __init__(
        self,
        score_of: Callable[[SearchResult], float] = lambda r: float(r.rank),
        fail: bool = False,
        drop_one: bool = False,
    ):
        self._score_of = score_of
        self._fail = fail
        self._drop_one = drop_one
        self.calls: List[tuple[str, List[str]]] = []

    def rerank(self, query_text: str, results: List[SearchResult]) -> List[SearchResult]:
        self.calls.append((query_text, [r.book.title for r in results]))
        if self._fail:
            raise RuntimeError("cross-encoder crashed")
        scored = sorted(results, key=self._score_of, reverse=True)
        if self._drop_one:
            scored = scored[:-1]
        return [
            replace(r, rank=i, rerank_score=self._score_of(r))
            for i, r in enumerate(scored, start=1)
        ]

    def get_model_name(self) -> str:
        return "fake-reranker"

    def is_ready(self) -> bool:
        return not self._fail


class StubLLM:
    """Understanding stub whose reformulation and variations differ from the query."""

    def extract_query_intent(self, query_text: str) -> QueryIntent:
        return QueryIntent(
            intent_type="factual",  # no forced MMR
            original_query=query_text,
            reformulated_query="reformulated text",
            extracted_filters=SearchFilters(),
            confidence=0.9,
            reasoning="test",
            query_variations=["reformulated text", "variation two"],
        )

    def generate_grounded_explanation(self, query_text: str, book: Book):
        raise AssertionError("explanations are not requested in these tests")

    def get_model_name(self) -> str:
        return "stub-llm"


def build_service(
    reranker=None,
    rerank_depth: int = DEFAULT_RERANK_DEPTH,
    books: Optional[List[Book]] = None,
    vector_ready: bool = True,
    llm_client=None,
) -> tuple[SearchService, RecordingRepo, RecordingRepo]:
    books = books if books is not None else make_books()
    lexical = RecordingRepo(books, "lexical")
    vector = RecordingRepo(books, "vector", ready=vector_ready)
    kwargs = {} if reranker is None else {"reranker": reranker, "rerank_depth": rerank_depth}
    service = SearchService(
        lexical_search=lexical,
        vector_search=vector,
        embeddings_store=FakeEmbeddings(books),
        llm_client=llm_client,
        **kwargs,
    )
    return service, lexical, vector


def titles(results: List[SearchResult]) -> List[str]:
    return [r.book.title for r in results]


def book_titles(numbers) -> List[str]:
    return [f"Book {i:02d}" for i in numbers]


# =============================================================================
# Ordering, ranks and scores
# =============================================================================


class TestRerankOrderAndScores:
    def test_reranker_changes_the_order(self):
        service, _, _ = build_service(FakeReranker())

        results = service.search(SearchQuery(text="query", max_results=10))

        # Head = fused top 30 (Book 00..29), reversed by the fake reranker
        assert titles(results) == book_titles(range(29, 19, -1))

    def test_ranks_are_renumbered_and_both_scores_are_kept(self):
        baseline, _, _ = build_service()
        fused_by_title = {
            r.book.title: r for r in baseline.search(SearchQuery(text="query", max_results=50))
        }
        service, _, _ = build_service(FakeReranker())

        results = service.search(SearchQuery(text="query", max_results=10))

        assert [r.rank for r in results] == list(range(1, 11))
        for result in results:
            fused = fused_by_title[result.book.title]
            assert result.rerank_score == float(fused.rank)  # the fake's score
            assert result.final_score == pytest.approx(fused.final_score)  # RRF kept
            assert result.lexical_score == fused.lexical_score
            assert result.vector_score == fused.vector_score
            assert result.source == "hybrid"

    def test_search_with_fallback_reranks_with_the_query_text(self):
        reranker = FakeReranker()
        service, _, _ = build_service(reranker)

        response = service.search_with_fallback(SearchQuery(text="user words", max_results=10))

        assert response.degraded is False
        assert [call[0] for call in reranker.calls] == ["user words"]
        assert titles(response.results) == book_titles(range(29, 19, -1))
        assert all(r.has_rerank_score() for r in response.results)


# =============================================================================
# Depth and candidate pool
# =============================================================================


class TestRerankDepth:
    def test_reranker_receives_exactly_depth_results(self):
        reranker = FakeReranker()
        service, _, _ = build_service(reranker, rerank_depth=30)

        service.search(SearchQuery(text="query", max_results=10))

        assert len(reranker.calls) == 1
        assert reranker.calls[0][1] == book_titles(range(30))

    def test_results_beyond_depth_keep_the_fused_order(self):
        reranker = FakeReranker()
        service, _, _ = build_service(reranker, rerank_depth=5)

        results = service.search(SearchQuery(text="query", max_results=8))

        assert reranker.calls[0][1] == book_titles(range(5))
        assert titles(results) == book_titles([4, 3, 2, 1, 0, 5, 6, 7])
        assert [r.has_rerank_score() for r in results] == [True] * 5 + [False] * 3
        assert [r.rank for r in results] == list(range(1, 9))

    def test_hybrid_retrieval_fetches_at_least_depth_candidates(self):
        service, lexical, vector = build_service(FakeReranker(), rerank_depth=30)

        service.search(SearchQuery(text="query", max_results=5))

        assert min(lexical.requested) >= 30
        assert min(vector.requested) >= 30

    def test_multi_query_retrieval_fetches_at_least_depth_candidates(self):
        service, lexical, vector = build_service(
            FakeReranker(), rerank_depth=30, llm_client=StubLLM()
        )

        service.search_with_understanding("query", max_results=5)

        # factual -> lexical heavy: BM25 factor 4, vector factor 1
        assert lexical.requested and vector.requested
        assert min(lexical.requested) >= 30
        assert min(vector.requested) >= 30

    def test_lexical_only_retrieval_fetches_at_least_depth_candidates(self):
        reranker = FakeReranker()
        service, lexical, _ = build_service(reranker, rerank_depth=30, vector_ready=False)

        response = service.search_with_fallback(SearchQuery(text="query", max_results=5))

        assert response.search_mode == "lexical_only"
        assert lexical.requested == [30]
        assert len(reranker.calls[0][1]) == 30
        assert titles(response.results) == book_titles(range(29, 24, -1))
        assert [r.rank for r in response.results] == [1, 2, 3, 4, 5]

    def test_rerank_depth_must_be_positive(self):
        with pytest.raises(ValueError, match="rerank_depth"):
            build_service(FakeReranker(), rerank_depth=0)


# =============================================================================
# The reranker always sees the user's original query text
# =============================================================================


class TestOriginalQueryText:
    def test_understanding_path_uses_original_text(self):
        reranker = FakeReranker()
        service, _, _ = build_service(reranker, llm_client=StubLLM())

        response, intent = service.search_with_understanding("original user words", max_results=10)

        assert intent.reformulated_query == "reformulated text"
        assert [call[0] for call in reranker.calls] == ["original user words"]
        assert all(r.has_rerank_score() for r in response.results)

    def test_understanding_fallback_path_uses_original_text(self):
        reranker = FakeReranker()
        service, _, _ = build_service(reranker, llm_client=StubLLM(), vector_ready=False)

        response, _ = service.search_with_understanding("original user words", max_results=10)

        assert response.degraded is True
        assert [call[0] for call in reranker.calls] == ["original user words"]

    def test_understanding_without_llm_uses_the_query_text(self):
        reranker = FakeReranker()
        service, _, _ = build_service(reranker)

        service.search_with_understanding("plain words", max_results=10)

        assert [call[0] for call in reranker.calls] == ["plain words"]


# =============================================================================
# Filters, MMR and paths without a query
# =============================================================================


class TestRerankWithFiltersAndMMR:
    def test_filters_are_applied_before_reranking(self):
        reranker = FakeReranker()
        service, _, _ = build_service(reranker)

        results = service.search(
            SearchQuery(text="query", max_results=10, filters=SearchFilters(language="es"))
        )

        received = reranker.calls[0][1]
        # Only odd (Spanish) books reach the reranker, even though the fake
        # repositories ignore the filter
        assert received and all(int(title.split()[1]) % 2 == 1 for title in received)
        assert all(r.book.language == "es" for r in results)

    def test_mmr_uses_the_rerank_score_as_relevance(self):
        service, _, _ = build_service(FakeReranker())

        results = service.search(
            SearchQuery(text="query", max_results=5, use_diversification=True, diversity_lambda=0.9)
        )

        # MMR's first pick is the most relevant result: the reranker's top
        # (Book 29), not the fused top (Book 00)
        assert results[0].book.title == "Book 29"
        assert all(r.has_rerank_score() for r in results)
        assert [r.rank for r in results] == list(range(1, 6))

    def test_mmr_fills_with_the_fused_tail_when_more_results_than_depth(self):
        service, _, _ = build_service(FakeReranker(), rerank_depth=5)

        results = service.search(
            SearchQuery(text="query", max_results=8, use_diversification=True)
        )

        assert len(results) == 8
        assert all(r.has_rerank_score() for r in results[:5])
        assert sorted(titles(results[:5])) == book_titles(range(5))
        assert titles(results[5:]) == book_titles([5, 6, 7])
        assert [r.rank for r in results] == list(range(1, 9))

    def test_find_similar_books_does_not_rerank(self):
        reranker = FakeReranker()
        books = make_books()
        service, _, _ = build_service(reranker, books=books)

        service.find_similar_books(books[0].id, max_results=5)

        assert reranker.calls == []


# =============================================================================
# Graceful degradation
# =============================================================================


class TestRerankFailure:
    def test_failure_returns_the_fused_order(self, caplog):
        baseline, _, _ = build_service()
        expected = titles(baseline.search(SearchQuery(text="query", max_results=10)))
        service, _, _ = build_service(FakeReranker(fail=True))

        with caplog.at_level(logging.WARNING, logger="app.domain.services"):
            results = service.search(SearchQuery(text="query", max_results=10))

        assert titles(results) == expected
        assert [r.rank for r in results] == list(range(1, 11))
        assert not any(r.has_rerank_score() for r in results)
        assert "Re-ranking failed" in caplog.text

    def test_failure_in_understanding_path_returns_the_fused_order(self):
        service, _, _ = build_service(FakeReranker(fail=True), llm_client=StubLLM())

        response, _ = service.search_with_understanding("query", max_results=10)

        assert response.degraded is False
        assert len(response.results) == 10
        assert not any(r.has_rerank_score() for r in response.results)

    def test_a_reranker_that_loses_results_is_treated_as_a_failure(self):
        baseline, _, _ = build_service()
        expected = titles(baseline.search(SearchQuery(text="query", max_results=10)))
        service, _, _ = build_service(FakeReranker(drop_one=True))

        results = service.search(SearchQuery(text="query", max_results=10))

        assert titles(results) == expected
        assert not any(r.has_rerank_score() for r in results)


# =============================================================================
# Without a reranker nothing changes
# =============================================================================


class TestWithoutReranker:
    def test_hybrid_candidate_pool_and_order_are_unchanged(self):
        service, lexical, vector = build_service()

        results = service.search(SearchQuery(text="query", max_results=10))

        assert lexical.requested == [20] and vector.requested == [20]  # max_results * 2
        assert titles(results) == book_titles(range(10))
        assert not any(r.has_rerank_score() for r in results)

    def test_multi_query_pool_sizes_are_unchanged(self):
        service, lexical, vector = build_service(llm_client=StubLLM())

        service.search_with_understanding("query", max_results=5)

        pools = STRATEGY_POOL_SIZES[RetrievalStrategy.LEXICAL_HEAVY]
        assert set(lexical.requested) == {5 * pools["bm25_factor"]}
        assert set(vector.requested) == {5 * pools["vector_factor"]}

    def test_lexical_only_pool_is_unchanged(self):
        service, lexical, _ = build_service(vector_ready=False)

        response = service.search_with_fallback(SearchQuery(text="query", max_results=5))

        assert lexical.requested == [5]
        assert titles(response.results) == book_titles(range(5))


# =============================================================================
# Health and introspection
# =============================================================================


class TestRerankerHealth:
    def test_health_reports_a_configured_reranker(self):
        service, _, _ = build_service(FakeReranker())

        status = service.get_health_status()

        assert service.has_reranker() is True
        assert service.get_reranker_model_name() == "fake-reranker"
        assert status["reranker"] is True
        assert status["overall"] is True

    def test_broken_reranker_does_not_affect_overall(self):
        service, _, _ = build_service(FakeReranker(fail=True))

        status = service.get_health_status()

        assert status["reranker"] is False
        assert status["overall"] is True

    def test_health_without_reranker(self):
        service, _, _ = build_service()

        status = service.get_health_status()

        assert service.has_reranker() is False
        assert service.get_reranker_model_name() is None
        assert status["reranker"] is False
        assert status["overall"] is True
