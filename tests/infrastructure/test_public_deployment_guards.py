"""
Adapters and settings of the public-deployment guards (V17), without network.

- InMemoryUsageBudget and PostgresUsageBudget (with a fake database that
  records each statement: one round trip per unit, the table created once).
- UsageBudgetSettings, RateLimitSettings, EvaluateEndpointSettings,
  ApiQuerySettings and LLM_CLIENT_MAX_RETRIES: local defaults unchanged.
- OpenRouterHttp per-call timeout, retries and deadline, with a fake clock.
- The query embedding and the rerank call use the short deadlines; document
  batches keep the long ones.
"""

from datetime import date

import httpx
import pytest

from app.domain.ports import BUDGET_LLM_REQUEST, BUDGET_MODEL_SEARCH
from app.infrastructure.config.settings import (
    ApiQuerySettings,
    EvaluateEndpointSettings,
    LLMSettings,
    MCPSettings,
    RateLimitSettings,
    UsageBudgetSettings,
)
from app.infrastructure.db.postgres_usage_budget import PostgresUsageBudget, consume_sql
from app.infrastructure.search.openrouter_http import OpenRouterError, OpenRouterHttp
from app.infrastructure.usage_budget import InMemoryUsageBudget

# ---------------------------------------------------------------- budgets


def test_in_memory_budget_counts_per_kind_and_starts_again_the_next_day():
    day = [date(2026, 11, 11)]
    budget = InMemoryUsageBudget({BUDGET_MODEL_SEARCH: 2, BUDGET_LLM_REQUEST: 0}, today=lambda: day[0])

    assert [budget.try_consume(BUDGET_MODEL_SEARCH) for _ in range(3)] == [True, True, False]
    assert budget.try_consume(BUDGET_LLM_REQUEST) is False  # a limit of 0 always refuses
    day[0] = date(2026, 11, 12)
    assert budget.try_consume(BUDGET_MODEL_SEARCH) is True
    assert budget.used_today() == {BUDGET_MODEL_SEARCH: 1}


def test_a_kind_without_a_limit_is_always_granted_and_unknown_kinds_are_rejected():
    budget = InMemoryUsageBudget({BUDGET_MODEL_SEARCH: 1})

    assert all(budget.try_consume(BUDGET_LLM_REQUEST) for _ in range(100))
    with pytest.raises(ValueError):
        budget.try_consume("tokens")
    with pytest.raises(ValueError):
        InMemoryUsageBudget({"tokens": 3})


class FakeDatabase:
    """Records statements; answers the increment with the rows the test sets."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query(self, sql, params=None, *, settings=None):
        self.calls.append((sql, params))
        return self.rows


def test_postgres_budget_takes_a_unit_in_one_round_trip_and_creates_the_table_once():
    db = FakeDatabase([{"used": 1}])
    budget = PostgresUsageBudget(db, "usage_budget", {BUDGET_MODEL_SEARCH: 400})

    assert budget.try_consume(BUDGET_MODEL_SEARCH) is True
    assert budget.try_consume(BUDGET_MODEL_SEARCH) is True

    assert len(db.calls) == 2  # one statement (one round trip) per unit
    first, second = db.calls[0][0], db.calls[1][0]
    assert first.startswith("CREATE SCHEMA IF NOT EXISTS usage_budget;") and consume_sql("usage_budget") in first
    assert second == consume_sql("usage_budget")
    assert db.calls[1][1] == {"kind": BUDGET_MODEL_SEARCH, "limit": 400}
    assert "ON CONFLICT (day, kind) DO UPDATE SET used = u.used + 1" in second
    assert "WHERE u.used < %(limit)s" in second and "RETURNING used" in second


def test_postgres_budget_refuses_when_no_row_comes_back_and_skips_the_database_without_a_limit():
    db = FakeDatabase([])
    budget = PostgresUsageBudget(db, "usage_budget", {BUDGET_MODEL_SEARCH: 400, BUDGET_LLM_REQUEST: 0})

    assert budget.try_consume(BUDGET_MODEL_SEARCH) is False  # the row is at the limit
    assert budget.try_consume(BUDGET_LLM_REQUEST) is False  # limit 0: refused without a call
    assert len(db.calls) == 1
    assert PostgresUsageBudget(db, "usage_budget", {}).try_consume(BUDGET_LLM_REQUEST) is True
    assert len(db.calls) == 1


def test_postgres_budget_rejects_unsafe_schema_names():
    with pytest.raises(ValueError):
        PostgresUsageBudget(FakeDatabase([]), "budget; DROP TABLE x", {BUDGET_MODEL_SEARCH: 1})


# ---------------------------------------------------------------- settings


def test_guards_are_off_by_default_and_read_from_the_environment():
    assert not UsageBudgetSettings.from_env({}).enabled
    assert RateLimitSettings.from_env({}) == RateLimitSettings(0, 0, None)
    assert EvaluateEndpointSettings.from_env({}).enabled is True
    assert LLMSettings.from_env({}).client_max_retries is None
    assert MCPSettings.from_env({}).client_ip_header is None

    env = {"DAILY_SEARCH_LIMIT": "400", "DAILY_LLM_LIMIT": "60", "SEARCH_RATE_LIMIT_PER_MINUTE": "20",
           "LLM_RATE_LIMIT_PER_MINUTE": "4", "CLIENT_IP_HEADER": "X-Real-IP", "EVALUATE_ENDPOINT_ENABLED": "false",
           "LLM_CLIENT_MAX_RETRIES": "0"}
    assert UsageBudgetSettings.from_env(env).limits() == {BUDGET_MODEL_SEARCH: 400, BUDGET_LLM_REQUEST: 60}
    assert RateLimitSettings.from_env(env) == RateLimitSettings(20, 4, "x-real-ip")
    assert EvaluateEndpointSettings.from_env(env).enabled is False
    assert LLMSettings.from_env(env).client_max_retries == 0
    assert MCPSettings.from_env(env).client_ip_header == "x-real-ip"
    with pytest.raises(ValueError):
        UsageBudgetSettings.from_env({"DAILY_SEARCH_LIMIT": "-1"})


def test_query_deadlines_apply_in_the_cloud_profile_only():
    assert ApiQuerySettings.from_env({}).call_options() == {}
    cloud = ApiQuerySettings.from_env({"SEARCH_PROFILE": "cloud"})
    assert cloud.call_options() == {"timeout_s": 3.0, "max_retries": 1, "deadline_s": 5.0}
    tuned = ApiQuerySettings.from_env({"SEARCH_PROFILE": "cloud", "API_QUERY_TIMEOUT_S": "2",
                                       "API_QUERY_MAX_RETRIES": "0", "API_QUERY_DEADLINE_S": "2.5"})
    assert tuned.call_options() == {"timeout_s": 2.0, "max_retries": 0, "deadline_s": 2.5}


# ---------------------------------------------------------------- deadlines


class FakeClock:
    """A clock that advances only when the fake transport or sleep says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _http(clock: FakeClock, handler, **kwargs) -> OpenRouterHttp:
    return OpenRouterHttp(api_key="test-key", transport=httpx.MockTransport(handler), sleep=clock.sleep,
                          clock=clock, **kwargs)


def test_a_query_call_makes_at_most_one_retry():
    clock, attempts = FakeClock(), []

    def handler(request):
        attempts.append(clock.now)
        clock.now += 0.1
        return httpx.Response(503, json={"error": {"message": "busy"}})

    http = _http(clock, handler)  # client defaults: 30 s, 3 retries (document batches)
    with pytest.raises(OpenRouterError) as failure:
        http.post("embeddings", {"input": ["q"]}, timeout_s=3.0, max_retries=1, deadline_s=5.0)

    assert len(attempts) == 2 and failure.value.transient
    assert "after 2 attempts" in str(failure.value)


def test_no_retry_starts_without_time_left_before_the_deadline():
    clock, attempts = FakeClock(), []

    def handler(request):
        attempts.append(clock.now)
        clock.now += 3.0  # the attempt took 3 s before failing
        raise httpx.ReadTimeout("slow", request=request)

    http = _http(clock, handler, backoff_s=1.0)
    with pytest.raises(OpenRouterError) as failure:
        http.post("rerank", {"query": "q"}, timeout_s=3.0, max_retries=1, deadline_s=4.0)

    assert len(attempts) == 1  # 3 s spent + 1 s backoff leaves no time for a retry
    assert "deadline" in str(failure.value)


def test_an_attempt_is_cut_to_the_time_left(monkeypatch):
    clock, timeouts = FakeClock(), []
    http = _http(clock, lambda request: httpx.Response(200, json={}))

    def post_once(path, body, timeout_s=None):
        timeouts.append(timeout_s)
        clock.now += timeout_s
        raise httpx.ConnectError("down")

    monkeypatch.setattr(http, "_post_once", post_once)
    with pytest.raises(OpenRouterError):
        http.post("embeddings", {}, timeout_s=3.0, max_retries=1, deadline_s=5.0)

    assert timeouts == [3.0, pytest.approx(1.0)]  # 5 s deadline - 3 s - 1 s backoff


def test_document_calls_keep_the_client_defaults():
    clock, attempts = FakeClock(), []

    def handler(request):
        attempts.append(clock.now)
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    http = _http(clock, handler, max_retries=3)
    with pytest.raises(OpenRouterError):
        http.post("embeddings", {"input": ["doc"]})

    assert len(attempts) == 4


def test_query_embeddings_use_the_short_deadlines_and_documents_the_long_ones(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from app.infrastructure.config.settings import EmbeddingSettings
    from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder

    class RecordingHttp:
        def __init__(self) -> None:
            self.calls = []

        def post(self, path, body, **options):
            self.calls.append(options)
            from app.infrastructure.search.openrouter_http import ApiResponse

            data = [{"index": i, "embedding": [0.1] * 1024} for i in range(len(body["input"]))]
            return ApiResponse({"data": data, "usage": {}}, 0.01, 1)

    settings = EmbeddingSettings.from_env({"SEARCH_PROFILE": "cloud"})
    query_settings = ApiQuerySettings.from_env({"SEARCH_PROFILE": "cloud"})
    encoder = OpenRouterEmbeddingEncoder.from_settings(settings, query_settings=query_settings)
    http = RecordingHttp()
    encoder._http = http  # noqa: SLF001 - swap the HTTP client for the recording one

    encoder.generate_query_embedding("a question")
    encoder.generate_embeddings_batch(["a document"])

    assert http.calls == [{"timeout_s": 3.0, "max_retries": 1, "deadline_s": 5.0}, {}]


def test_the_api_reranker_gets_the_short_timeout_retry_and_deadline(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from app.infrastructure.config.settings import RerankerSettings
    from app.infrastructure.search.api_reranker import ApiReranker

    env = {"SEARCH_PROFILE": "cloud"}
    reranker = ApiReranker.from_settings(RerankerSettings.from_env(env), query_settings=ApiQuerySettings.from_env(env))

    assert (reranker._http._timeout_s, reranker._http._max_retries) == (3.0, 1)  # noqa: SLF001
    assert reranker._call == {"deadline_s": 5.0}  # noqa: SLF001
