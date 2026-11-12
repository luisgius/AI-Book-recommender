"""
Explanations of a result list are generated concurrently, with a bounded number of workers.

The fake LLM sleeps a little in each call and counts how many calls are in
flight at once, so the tests can check the bound, the overlap, the order of
the results and that one failed explanation does not lose the others.
"""

import threading
import time
from datetime import UTC, datetime
from typing import List
from uuid import uuid4

import pytest

from app.domain.entities import Book, Citation, Explanation, SearchResult
from app.domain.ports import BUDGET_LLM_REQUEST
from app.domain.services import SearchService
from app.domain.value_objects import SearchQuery


def make_results(n: int) -> List[SearchResult]:
    books = [Book(id=uuid4(), title=f"Book {i}", authors=["A"], description=f"About {i}") for i in range(n)]
    return [SearchResult(book=b, final_score=1.0 - i / 10, rank=i + 1, source="hybrid", lexical_score=1.0)
            for i, b in enumerate(books)]


class SlowLLM:
    """Each explanation takes `delay` seconds; titles in `fail_titles` raise."""

    def __init__(self, delay: float = 0.05, fail_titles: tuple = ()) -> None:
        self.delay = delay
        self.fail_titles = fail_titles
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0
        self._lock = threading.Lock()

    def generate_grounded_explanation(self, query_text: str, book: Book) -> Explanation:
        with self._lock:
            self.calls += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            if book.title in self.fail_titles:
                raise RuntimeError("provider error")
            return Explanation(
                book_id=book.id, query_text=query_text, text=f"Explains {book.title}",
                citations=[Citation(book_id=book.id, chunk_id="title", snippet=book.title, relevance_score=0.9)],
                model="fake", created_at=datetime.now(UTC),
            )
        finally:
            with self._lock:
                self.in_flight -= 1


class Repo:
    def __init__(self, results):
        self._results = results

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        return list(self._results)

    def is_ready(self):
        return True


class Embeddings:
    def generate_query_embedding(self, text):
        return [0.1, 0.2]

    generate_embedding = generate_query_embedding

    def is_ready(self):
        return True


class CountingBudget:
    def __init__(self):
        self.taken: List[str] = []

    def try_consume(self, kind: str) -> bool:
        self.taken.append(kind)
        return True


def service(llm, max_explanations=3, workers=3, results=None, budget=None) -> SearchService:
    results = results or []
    return SearchService(Repo(results), Repo(results), Embeddings(), llm_client=llm,
                         max_explanations=max_explanations, explanation_workers=workers, usage_budget=budget)


def test_explanations_overlap_and_keep_the_order_of_the_results():
    llm = SlowLLM(delay=0.1)
    results = make_results(5)

    started = time.perf_counter()
    service(llm)._add_explanations("query", results)
    elapsed = time.perf_counter() - started

    assert [r.book.title for r in results] == [f"Book {i}" for i in range(5)]
    assert [r.explanation.text for r in results[:3]] == ["Explains Book 0", "Explains Book 1", "Explains Book 2"]
    assert all(r.explanation is None for r in results[3:])  # only max_explanations are explained
    assert llm.max_in_flight == 3
    assert elapsed < 0.25  # three 0.1 s calls at once, not 0.3 s one after another


def test_the_worker_bound_is_respected():
    llm = SlowLLM(delay=0.05)
    results = make_results(6)

    service(llm, max_explanations=6, workers=2)._add_explanations("query", results)

    assert llm.calls == 6
    assert llm.max_in_flight == 2
    assert all(r.explanation is not None for r in results)


def test_one_failed_explanation_does_not_lose_the_others():
    llm = SlowLLM(fail_titles=("Book 1",))
    results = make_results(3)

    service(llm)._add_explanations("query", results)

    assert results[0].explanation.text == "Explains Book 0"
    assert results[1].explanation is None
    assert results[2].explanation.text == "Explains Book 2"


def test_one_worker_generates_them_one_after_another():
    llm = SlowLLM(delay=0.02)
    results = make_results(3)

    service(llm, workers=1)._add_explanations("query", results)

    assert llm.max_in_flight == 1
    assert all(r.explanation is not None for r in results)


def test_workers_must_be_positive():
    with pytest.raises(ValueError):
        service(SlowLLM(), workers=0)


def test_a_search_with_explanations_still_takes_one_llm_unit():
    results = make_results(4)
    budget = CountingBudget()
    llm = SlowLLM(delay=0.01)

    response = service(llm, results=results, budget=budget).search_with_fallback(
        SearchQuery(text="query", max_results=4, use_explanations=True)
    )

    assert budget.taken.count(BUDGET_LLM_REQUEST) == 1
    assert llm.calls == 3
    assert [r.explanation is not None for r in response.results] == [True, True, True, False]
