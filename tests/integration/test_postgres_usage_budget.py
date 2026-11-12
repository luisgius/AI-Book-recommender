"""
Integration tests of the daily budget against a real Postgres (V17).

Skipped unless TEST_DATABASE_URL is set (use a throwaway database or Neon
branch). Every run uses its own schema (it_budget_<random>) and drops it.

- Units are granted up to the limit and then refused, per kind.
- Several "instances" (separate connection pools) asking at once never pass
  the limit: the increment is one atomic statement.
- A SearchService over the real budget degrades the search once it is spent.
"""

import os
import threading
from uuid import uuid4

import pytest

from app.domain.ports import BUDGET_LLM_REQUEST, BUDGET_MODEL_SEARCH

TEST_URL = os.environ.get("TEST_DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set (no test Postgres)")


@pytest.fixture
def schema():
    from app.infrastructure.db.postgres_connection import PostgresDatabase

    name = f"it_budget_{uuid4().hex[:10]}"
    yield name
    db = PostgresDatabase(TEST_URL, pool_max_size=1)
    try:
        db.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")
    finally:
        db.close()


def _budget(schema, limits):
    from app.infrastructure.db.postgres_connection import PostgresDatabase
    from app.infrastructure.db.postgres_usage_budget import PostgresUsageBudget

    db = PostgresDatabase(TEST_URL, pool_max_size=2)
    return db, PostgresUsageBudget(db, schema, limits)


def test_units_are_granted_up_to_the_limit_then_refused(schema):
    db, budget = _budget(schema, {BUDGET_MODEL_SEARCH: 3, BUDGET_LLM_REQUEST: 1})
    try:
        searches = [budget.try_consume(BUDGET_MODEL_SEARCH) for _ in range(5)]
        llm = [budget.try_consume(BUDGET_LLM_REQUEST) for _ in range(2)]

        assert searches == [True, True, True, False, False]
        assert llm == [True, False]
        assert budget.used_today() == {BUDGET_MODEL_SEARCH: 3, BUDGET_LLM_REQUEST: 1}
    finally:
        db.close()


def test_concurrent_instances_never_pass_the_limit(schema):
    limit, instances, attempts_each = 25, 6, 10
    budgets = [_budget(schema, {BUDGET_MODEL_SEARCH: limit}) for _ in range(instances)]
    budgets[0][1].used_today()  # create the table first, as the deployment does before going public
    granted = []
    lock = threading.Lock()

    def ask(budget):
        for _ in range(attempts_each):
            ok = budget.try_consume(BUDGET_MODEL_SEARCH)
            with lock:
                granted.append(ok)

    threads = [threading.Thread(target=ask, args=(b,)) for _, b in budgets]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(granted) == instances * attempts_each
        assert sum(granted) == limit
        assert budgets[0][1].used_today() == {BUDGET_MODEL_SEARCH: limit}
    finally:
        for db, _ in budgets:
            db.close()


def test_search_service_degrades_on_the_real_budget(schema):
    from datetime import UTC, datetime

    from app.domain.entities import Book, SearchResult
    from app.domain.services import NOTICE_SEARCH_BUDGET, SearchService
    from app.domain.value_objects import SearchQuery

    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    books = [Book(id=uuid4(), title=f"Book {i}", authors=["A"], description="About the sea.", language="en",
                  categories=["Fiction"], source="test", source_id=str(i), created_at=stamp, updated_at=stamp)
             for i in range(3)]

    class Repo:
        def __init__(self, source):
            self.source = source

        def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
            return [SearchResult(book=b, final_score=1.0, rank=i + 1, source=self.source)
                    for i, b in enumerate(books[:max_results])]

        def is_ready(self):
            return True

    class Embeddings:
        calls = 0

        def generate_query_embedding(self, text):
            Embeddings.calls += 1
            return [1.0]

        def is_ready(self):
            return True

    db, budget = _budget(schema, {BUDGET_MODEL_SEARCH: 1})
    try:
        service = SearchService(Repo("lexical"), Repo("vector"), Embeddings(), usage_budget=budget)
        first = service.search_with_fallback(SearchQuery(text="sea", max_results=2))
        second = service.search_with_fallback(SearchQuery(text="sea", max_results=2))

        assert first.search_mode == "hybrid" and first.notices == ()
        assert second.search_mode == "lexical_only" and second.notices == (NOTICE_SEARCH_BUDGET,)
        assert Embeddings.calls == 1
    finally:
        db.close()
