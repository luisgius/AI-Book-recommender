"""
The daily budget of paid model calls inside SearchService (V17).

With a UsageBudget wired, each search takes one model-search unit (without
it: lexical-only, no re-ranking, no embedding call) and each request that
uses the LLM takes one LLM unit (without it: no understanding, no
explanations, a notice). Fakes only: no network, no database.
"""

from datetime import UTC, datetime
from typing import Dict, List
from uuid import uuid4

from app.domain.entities import Book, Explanation, SearchResult
from app.domain.ports import BUDGET_LLM_REQUEST, BUDGET_MODEL_SEARCH
from app.domain.services import (
    NOTICE_LLM_BUDGET,
    NOTICE_SEARCH_BUDGET,
    SEARCH_BUDGET_REASON,
    SearchService,
)
from app.domain.value_objects import QueryIntent, SearchFilters, SearchQuery
from app.infrastructure.usage_budget import InMemoryUsageBudget


def _book(title: str) -> Book:
    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    return Book(id=uuid4(), title=title, authors=["A. Author"], description=f"About {title}.", language="en",
                categories=["Fiction"], source="test", source_id=title, created_at=stamp, updated_at=stamp)


BOOKS = [_book(f"Book {i}") for i in range(5)]


def _results(source: str) -> List[SearchResult]:
    return [SearchResult(book=b, final_score=1.0 - i / 10, rank=i + 1, source=source,
                         lexical_score=1.0 if source == "lexical" else None,
                         vector_score=0.9 if source == "vector" else None)
            for i, b in enumerate(BOOKS)]


class Lexical:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query_text, max_results=10, filters=None):
        self.calls += 1
        return _results("lexical")[:max_results]

    def is_ready(self) -> bool:
        return True


class Vector:
    def search(self, query_embedding, max_results=10, filters=None):
        return list(reversed(_results("vector")))[:max_results]

    def get_embeddings_batch(self, ids):
        return {i: [1.0, 0.0] for i in ids}

    def is_ready(self) -> bool:
        return True


class Embeddings:
    """Counts query embeddings (an API call in the cloud profile)."""

    def __init__(self) -> None:
        self.query_calls = 0

    def generate_query_embedding(self, text: str) -> List[float]:
        self.query_calls += 1
        return [1.0, 0.0]

    def get_embedding(self, book_id):
        return [1.0, 0.0]

    def get_embeddings_batch(self, ids):
        return {i: [1.0, 0.0] for i in ids}

    def is_ready(self) -> bool:
        return True


class Reranker:
    """Counts rerank requests (an API call in the cloud profile); reverses the order."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query_text, results):
        self.calls += 1
        out = list(reversed(results))
        for i, r in enumerate(out, start=1):
            r.rank = i
            r.rerank_score = 1.0 / i
        return out

    def get_model_name(self) -> str:
        return "fake-reranker"

    def is_ready(self) -> bool:
        return True


class Llm:
    """Counts LLM calls."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_grounded_explanation(self, query_text, book):
        self.calls += 1
        return Explanation(book_id=book.id, query_text=query_text, text="Because.", model="fake-llm")

    def extract_query_intent(self, query_text):
        self.calls += 1
        return QueryIntent(intent_type="factual", original_query=query_text, reformulated_query=query_text,
                           extracted_filters=SearchFilters(), confidence=0.9, reasoning="fake")


class BrokenBudget:
    def try_consume(self, kind: str) -> bool:
        raise RuntimeError("database unreachable")

    def limits(self) -> Dict[str, int]:
        return {"model_search": 1}


def service(budget, *, llm=None, reranker=None, embeddings=None, lexical=None) -> SearchService:
    return SearchService(lexical_search=lexical or Lexical(), vector_search=Vector(),
                         embeddings_store=embeddings or Embeddings(), llm_client=llm,
                         reranker=reranker, rerank_depth=5, usage_budget=budget)


def test_searches_are_hybrid_and_reranked_while_the_budget_lasts_then_lexical_without_models():
    embeddings, reranker = Embeddings(), Reranker()
    svc = service(InMemoryUsageBudget({BUDGET_MODEL_SEARCH: 2}), embeddings=embeddings, reranker=reranker)

    first = svc.search_with_fallback(SearchQuery(text="sea voyages", max_results=3))
    second = svc.search_with_fallback(SearchQuery(text="sea voyages", max_results=3))
    third = svc.search_with_fallback(SearchQuery(text="sea voyages", max_results=3))

    assert [r.search_mode for r in (first, second)] == ["hybrid", "hybrid"]
    assert first.notices == () and not first.degraded
    assert third.search_mode == "lexical_only" and third.degraded
    assert third.degradation_reason == SEARCH_BUDGET_REASON
    assert third.notices == (NOTICE_SEARCH_BUDGET,)
    assert len(third.results) == 3 and all(r.rerank_score is None for r in third.results)
    # No model call once the budget is spent: two embeddings and two reranks, not three
    assert embeddings.query_calls == 2 and reranker.calls == 2


def test_explanations_are_refused_with_a_notice_when_the_llm_budget_is_spent():
    llm = Llm()
    svc = service(InMemoryUsageBudget({BUDGET_LLM_REQUEST: 1}), llm=llm)

    explained = svc.search_with_fallback(SearchQuery(text="poems", max_results=2, use_explanations=True))
    refused = svc.search_with_fallback(SearchQuery(text="poems", max_results=2, use_explanations=True))

    assert all(r.explanation is not None for r in explained.results) and explained.notices == ()
    assert all(r.explanation is None for r in refused.results)
    assert refused.notices == (NOTICE_LLM_BUDGET,)
    assert refused.search_mode == "hybrid"  # the search itself is not limited (no search limit set)
    assert llm.calls == 2  # max_explanations defaults to 5, two results explained once


def test_a_search_without_explanations_takes_no_llm_unit():
    budget = InMemoryUsageBudget({BUDGET_LLM_REQUEST: 1, BUDGET_MODEL_SEARCH: 10})
    svc = service(budget, llm=Llm())

    svc.search_with_fallback(SearchQuery(text="poems", max_results=2))

    assert budget.used_today() == {BUDGET_MODEL_SEARCH: 1}


def test_understanding_takes_one_llm_unit_and_falls_back_without_the_llm_when_spent():
    llm, embeddings = Llm(), Embeddings()
    budget = InMemoryUsageBudget({BUDGET_LLM_REQUEST: 1, BUDGET_MODEL_SEARCH: 10})
    svc = service(budget, llm=llm, embeddings=embeddings)

    response, intent = svc.search_with_understanding("dragons", max_results=2, use_explanations=True)
    assert intent.intent_type == "factual" and response.notices == ()
    calls_with_budget = llm.calls  # understanding + two explanations

    response, intent = svc.search_with_understanding("dragons", max_results=2, use_explanations=True)

    assert llm.calls == calls_with_budget  # no LLM call without a unit
    assert response.notices == (NOTICE_LLM_BUDGET,)
    assert intent.confidence == 0.0 and "budget" in intent.reasoning
    assert all(r.explanation is None for r in response.results)
    assert budget.used_today() == {BUDGET_LLM_REQUEST: 1, BUDGET_MODEL_SEARCH: 2}


def test_understanding_takes_one_search_unit_and_runs_lexical_without_models_when_spent():
    embeddings, reranker = Embeddings(), Reranker()
    budget = InMemoryUsageBudget({BUDGET_MODEL_SEARCH: 0})
    svc = service(budget, llm=Llm(), embeddings=embeddings, reranker=reranker)

    response, _ = svc.search_with_understanding("dragons", max_results=2)

    assert response.search_mode == "lexical_only" and response.notices == (NOTICE_SEARCH_BUDGET,)
    assert embeddings.query_calls == 0 and reranker.calls == 0


def test_a_budget_that_fails_counts_as_spent():
    embeddings = Embeddings()
    svc = service(BrokenBudget(), embeddings=embeddings, llm=Llm())

    response = svc.search_with_fallback(SearchQuery(text="anything", max_results=2, use_explanations=True))

    assert response.search_mode == "lexical_only"
    assert set(response.notices) == {NOTICE_SEARCH_BUDGET, NOTICE_LLM_BUDGET}
    assert embeddings.query_calls == 0


def test_without_a_budget_nothing_changes():
    svc = service(None, reranker=Reranker())

    response = svc.search_with_fallback(SearchQuery(text="sea", max_results=3))

    assert response.search_mode == "hybrid" and response.notices == () and not response.degraded


def test_stage_timings_are_reported():
    svc = service(None, reranker=Reranker())

    hybrid = svc.search_with_fallback(SearchQuery(text="sea", max_results=3))
    lexical = service(InMemoryUsageBudget({BUDGET_MODEL_SEARCH: 0})).search_with_fallback(
        SearchQuery(text="sea", max_results=3))

    assert {"lexical", "embedding", "vector", "rerank"} <= set(hybrid.metadata.timings_ms)
    assert set(lexical.metadata.timings_ms) == {"lexical"}
