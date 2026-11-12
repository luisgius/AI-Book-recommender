"""
Vector candidate mode of SearchService (candidates="vector").

The application's default since the evaluation on the deployed catalog: the
vector search alone supplies the candidates that are re-ranked, and keyword
search (BM25) runs only as the fallback. These tests check that:
- in the normal case BM25 is never called and no result claims a keyword position;
- the fallback still reaches BM25 when the embedding fails, when the vector
  index is unavailable and when the daily model budget is spent, with the same
  degraded flag and notice as in the hybrid mode;
- the understanding path labels its results the same way;
- the hybrid mode stays selectable and is the domain default.
"""

from typing import Dict, List

import pytest

from app.domain.entities import Book, SearchResult
from app.domain.ports import BUDGET_MODEL_SEARCH
from app.domain.services import NOTICE_SEARCH_BUDGET, SearchService
from app.domain.value_objects import CANDIDATES_HYBRID, CANDIDATES_VECTOR, SearchQuery
from tests.domain.test_reranking import FakeEmbeddings, FakeReranker, StubLLM, make_books


class CountingRepo:
    """A repository with a fixed list (numbered 1..n) that counts its calls."""

    def __init__(self, books: List[Book], source: str, ready: bool = True):
        self._books = books
        self._source = source
        self._ready = ready
        self.calls = 0

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        self.calls += 1
        field = "lexical_score" if self._source == "lexical" else "vector_score"
        return [
            SearchResult(book=b, final_score=1.0 / i, rank=i, source=self._source, **{field: 1.0 / i})
            for i, b in enumerate(self._books[:max_results], start=1)
        ]

    def is_ready(self) -> bool:
        return self._ready


class FailingEmbeddings(FakeEmbeddings):
    """The embedding provider fails (for example a read timeout)."""

    def generate_query_embedding(self, text: str) -> List[float]:
        raise RuntimeError("ReadTimeout (after 2 attempts)")


class SpentBudget:
    """A daily budget whose model-search units are spent."""

    def try_consume(self, kind: str) -> bool:
        return kind != BUDGET_MODEL_SEARCH

    def limits(self) -> Dict[str, int]:
        return {BUDGET_MODEL_SEARCH: 0}


def build(candidates: str = CANDIDATES_VECTOR, *, embeddings=None, vector_ready: bool = True, **kwargs):
    books = make_books(12)
    lexical = CountingRepo(books[::-1], "lexical")
    vector = CountingRepo(books, "vector", ready=vector_ready)
    service = SearchService(
        lexical_search=lexical,
        vector_search=vector,
        embeddings_store=embeddings or FakeEmbeddings(books),
        candidates=candidates,
        **kwargs,
    )
    return service, lexical, vector, books


def test_the_domain_default_is_the_hybrid_mode_of_the_published_experiments():
    service = SearchService(
        lexical_search=CountingRepo([], "lexical"),
        vector_search=CountingRepo([], "vector"),
        embeddings_store=FakeEmbeddings([]),
    )
    assert service.candidate_mode == CANDIDATES_HYBRID


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="candidates must be one of"):
        build("bm25_only")


def test_vector_mode_does_not_call_keyword_search_and_reports_it():
    service, lexical, vector, books = build(reranker=FakeReranker(), rerank_depth=4)

    response = service.search_with_fallback(SearchQuery(text="q", max_results=5))

    assert lexical.calls == 0 and vector.calls == 1
    assert not response.degraded
    assert response.search_mode == "vector_only"
    assert response.metadata.fusion_method == "none" and response.metadata.rrf_k is None
    assert response.metadata.candidates_lexical == 0
    assert response.metadata.candidates_vector > 0
    assert "lexical" not in response.metadata.timings_ms
    for result in response.results:
        assert result.source == "vector"
        assert result.lexical_rank is None and result.lexical_score is None
        assert result.vector_rank is not None
        assert result.final_score == result.vector_score
    # The reranker reversed the head of the vector list: positions stay truthful
    assert [r.rerank_rank for r in response.results[:4]] == [1, 2, 3, 4]
    assert response.results[0].vector_rank == 4
    assert response.results[0].book.title == books[3].title


def test_vector_mode_falls_back_to_keyword_search_when_the_embedding_fails():
    books = make_books(12)
    service, lexical, vector, _ = build(embeddings=FailingEmbeddings(books))

    response = service.search_with_fallback(SearchQuery(text="q", max_results=5))

    assert lexical.calls == 1 and vector.calls == 0
    assert response.degraded
    assert response.search_mode == "lexical_only"
    assert "ReadTimeout" in response.degradation_reason
    assert response.metadata.candidates_vector == 0
    # The keyword list's own order and positions
    assert [r.book.title for r in response.results] == [b.title for b in books[::-1][:5]]
    assert [r.lexical_rank for r in response.results] == [1, 2, 3, 4, 5]
    assert all(r.vector_rank is None for r in response.results)


def test_vector_mode_falls_back_to_keyword_search_when_the_index_is_unavailable():
    service, lexical, vector, _ = build(vector_ready=False)

    response = service.search_with_fallback(SearchQuery(text="q", max_results=5))

    assert lexical.calls == 1 and vector.calls == 0
    assert response.degraded and response.search_mode == "lexical_only"


def test_vector_mode_uses_keyword_search_without_models_when_the_budget_is_spent():
    reranker = FakeReranker()
    service, lexical, vector, _ = build(reranker=reranker, usage_budget=SpentBudget())

    response = service.search_with_fallback(SearchQuery(text="q", max_results=5))

    assert lexical.calls == 1 and vector.calls == 0
    assert reranker.calls == []
    assert response.degraded and response.search_mode == "lexical_only"
    assert NOTICE_SEARCH_BUDGET in response.notices


def test_understanding_in_vector_mode_labels_results_as_vector_and_skips_keywords():
    service, lexical, vector, _ = build(llm_client=StubLLM())

    response, _intent = service.search_with_understanding("q", max_results=5)

    assert lexical.calls == 0 and vector.calls == 2  # two query variations
    assert response.search_mode == "vector_only"
    assert response.metadata.candidates_lexical == 0
    # Two variation lists are fused with RRF (no keyword list among them)
    assert response.metadata.fusion_method == "rrf"
    assert all(r.source == "vector" and r.lexical_rank is None for r in response.results)


def test_understanding_in_vector_mode_falls_back_to_keywords_when_the_embedding_fails():
    books = make_books(12)
    service, lexical, _, _ = build(embeddings=FailingEmbeddings(books), llm_client=StubLLM())

    response, _intent = service.search_with_understanding("q", max_results=5)

    assert lexical.calls >= 1
    assert response.degraded and response.search_mode == "lexical_only"


def test_hybrid_mode_still_fuses_both_lists():
    service, lexical, vector, _ = build(CANDIDATES_HYBRID)

    response = service.search_with_fallback(SearchQuery(text="q", max_results=5))

    assert lexical.calls == 1 and vector.calls == 1
    assert response.search_mode == "hybrid"
    assert response.metadata.fusion_method == "rrf" and response.metadata.rrf_k == 60
    assert all(r.source == "hybrid" for r in response.results)
