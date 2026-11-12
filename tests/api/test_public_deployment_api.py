"""
The public-deployment guards through the HTTP API (V17), without network.

- Per-client rate limits of POST /search, GET /books/{id}/similar and the LLM
  endpoints: 429 with Retry-After, counted per client address (a trusted
  header when CLIENT_IP_HEADER is set).
- The daily budget: notices in the search response, 429 from the agent.
- POST /evaluate answers 404 when switched off; GET /usage is off by default.
- Degradation with the real API adapters whose provider fails: the query
  embedding fails -> lexical-only; the rerank call fails -> fused order; the
  LLM fails -> results without explanations. Each failing call is tried at
  most twice.
- Health reports the database (cloud only), the budget and the uptime; each
  result carries the source and licence of its description.
"""

from datetime import UTC, datetime
from typing import List
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.v1 import dependencies as deps
from app.domain.entities import Book, SearchResult
from app.domain.ports import BUDGET_LLM_REQUEST, BUDGET_MODEL_SEARCH
from app.domain.services import SearchService
from app.domain.value_objects import BookDescription, BookMetadata, SourceRecord
from app.infrastructure.config.settings import (
    EvaluateEndpointSettings,
    LLMSettings,
    RateLimitSettings,
    UsageBudgetSettings,
)
from app.infrastructure.search.api_reranker import ApiReranker
from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder
from app.infrastructure.search.openrouter_http import OpenRouterHttp
from app.infrastructure.usage_budget import InMemoryUsageBudget
from app.main import app

STAMP = datetime(2025, 1, 1, tzinfo=UTC)
QUERY_CALL = {"timeout_s": 3.0, "max_retries": 1, "deadline_s": 5.0}


def _book(title: str, metadata=None) -> Book:
    return Book(id=uuid4(), title=title, authors=["Author"], description=f"A book about {title}.", language="en",
                categories=["Fiction"], source="wikidata", source_id=title, metadata=metadata,
                created_at=STAMP, updated_at=STAMP)


WIKI = BookMetadata(
    descriptions=(BookDescription("A book about Moby Dick.", "en", "wikipedia", "CC BY-SA 4.0",
                                  "https://en.wikipedia.org/wiki/Moby-Dick"),),
    provenance=(SourceRecord("wikidata", "Q174596", "CC0 1.0", "https://www.wikidata.org/wiki/Q174596"),
                SourceRecord("wikipedia", "en:Moby-Dick", "CC BY-SA 4.0", "https://en.wikipedia.org/wiki/Moby-Dick")),
)
BOOKS = [_book("Moby Dick", WIKI), _book("Sea stories"), _book("Whales")]


class Repo:
    def __init__(self, source: str):
        self.source = source

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        books = BOOKS if self.source == "lexical" else list(reversed(BOOKS))
        field = "lexical_score" if self.source == "lexical" else "vector_score"
        return [SearchResult(book=b, final_score=1.0 / (i + 1), rank=i + 1, source=self.source,
                             **{field: 1.0 / (i + 1)}) for i, b in enumerate(books[:max_results])]

    def is_ready(self) -> bool:
        return True


class Embeddings:
    """Embeddings store over an encoder (as PostgresEmbeddingsStore does)."""

    def __init__(self, encoder=None):
        self.encoder = encoder

    def generate_query_embedding(self, text: str) -> List[float]:
        return self.encoder.generate_query_embedding(text) if self.encoder else [0.1, 0.2]

    def get_embedding(self, book_id):
        return [0.1, 0.2]

    def get_embeddings_batch(self, ids):
        return {i: [0.1, 0.2] for i in ids}

    def is_ready(self) -> bool:
        return True


class FailingLLM:
    """Every LLM call fails (provider down)."""

    def __init__(self):
        self.calls = 0

    def generate_grounded_explanation(self, query_text, book):
        self.calls += 1
        raise RuntimeError("LLM provider unavailable")

    def get_model_name(self) -> str:
        return "failing-llm"


def failing_http(attempts: List[str], **options) -> OpenRouterHttp:
    """An OpenRouter client whose every request fails with a connection error, without waiting."""
    def handler(request):
        attempts.append(request.url.path)
        raise httpx.ConnectError("provider unreachable", request=request)

    return OpenRouterHttp(api_key="test-key", transport=httpx.MockTransport(handler), sleep=lambda s: None,
                          **options)


@pytest.fixture
def client_with(monkeypatch):
    """A TestClient around a SearchService built by the test, with chosen guard settings."""
    def build(service: SearchService, *, rate=RateLimitSettings(), budget=None, evaluate=True):
        app.dependency_overrides[deps.get_search_service] = lambda: service
        monkeypatch.setattr(deps, "_llm_settings", LLMSettings(provider="ollama"))
        monkeypatch.setattr(deps, "_rate_limit_settings", rate)
        monkeypatch.setattr(deps, "_rest_rate_limits", None)
        monkeypatch.setattr(deps, "_evaluate_settings", EvaluateEndpointSettings(enabled=evaluate))
        monkeypatch.setattr(deps, "_budget_settings", UsageBudgetSettings(*budget) if budget else UsageBudgetSettings())
        monkeypatch.setattr(deps, "_usage_budget", InMemoryUsageBudget(UsageBudgetSettings(*budget).limits())
                            if budget else None)
        monkeypatch.setattr(deps, "_budget_initialized", True)
        return TestClient(app)
    yield build
    app.dependency_overrides.clear()


def plain_service(**kwargs) -> SearchService:
    return SearchService(Repo("lexical"), Repo("vector"), Embeddings(), **kwargs)


# ---------------------------------------------------------------- rate limits


def test_search_rate_limit_answers_429_with_retry_after(client_with):
    client = client_with(plain_service(), rate=RateLimitSettings(search_per_minute=2))

    codes = [client.post("/api/v1/search", json={"text": "sea"}).status_code for _ in range(3)]
    third = client.post("/api/v1/search", json={"text": "sea"})

    assert codes == [200, 200, 429]
    assert third.status_code == 429 and int(third.headers["Retry-After"]) >= 1
    assert "at most 2 per minute" in third.json()["detail"]


def test_similar_books_count_toward_the_search_limit(client_with):
    client = client_with(plain_service(), rate=RateLimitSettings(search_per_minute=1))

    assert client.post("/api/v1/search", json={"text": "sea"}).status_code == 200
    assert client.get(f"/api/v1/books/{BOOKS[0].id}/similar").status_code == 429


def test_llm_requests_have_their_own_tighter_limit(client_with):
    client = client_with(plain_service(), rate=RateLimitSettings(search_per_minute=10, llm_per_minute=1))

    first = client.post("/api/v1/search", json={"text": "sea", "use_explanations": True})
    second = client.post("/api/v1/search", json={"text": "sea", "use_explanations": True})
    plain = client.post("/api/v1/search", json={"text": "sea"})

    assert (first.status_code, second.status_code, plain.status_code) == (200, 429, 200)
    assert "LLM requests" in second.json()["detail"]


def test_clients_are_told_apart_by_the_trusted_header(client_with):
    client = client_with(plain_service(), rate=RateLimitSettings(search_per_minute=1, client_ip_header="x-real-ip"))

    a1 = client.post("/api/v1/search", json={"text": "sea"}, headers={"x-real-ip": "203.0.113.1"})
    b1 = client.post("/api/v1/search", json={"text": "sea"}, headers={"x-real-ip": "203.0.113.2"})
    a2 = client.post("/api/v1/search", json={"text": "sea"}, headers={"x-real-ip": "203.0.113.1"})

    assert (a1.status_code, b1.status_code, a2.status_code) == (200, 200, 429)


def test_no_limits_by_default(client_with):
    client = client_with(plain_service())

    assert all(client.post("/api/v1/search", json={"text": "sea"}).status_code == 200 for _ in range(30))


# ---------------------------------------------------------------- budget and switches


def test_search_response_carries_the_budget_notices(client_with):
    budget = InMemoryUsageBudget({BUDGET_MODEL_SEARCH: 1})
    client = client_with(plain_service(usage_budget=budget), budget=(1, None))

    first = client.post("/api/v1/search", json={"text": "sea"}).json()
    second = client.post("/api/v1/search", json={"text": "sea"}).json()

    assert first["notices"] == [] and first["search_mode"] == "hybrid"
    assert second["notices"] == ["search_budget_spent"] and second["search_mode"] == "lexical_only"
    assert second["degraded"] is True and "budget" in second["degradation_reason"]


def test_agent_is_refused_with_a_clear_message_when_the_llm_budget_is_spent(client_with, monkeypatch):
    client = client_with(plain_service(), budget=(None, 0))
    monkeypatch.setattr(deps, "_search_agent", object())
    monkeypatch.setattr(deps, "_agent_initialized", True)

    response = client.post("/api/v1/agent/ask", json={"question": "novels about whales"})

    assert response.status_code == 429
    assert "budget of LLM requests" in response.json()["detail"]


def test_evaluate_is_404_when_switched_off_and_usage_is_off_by_default(client_with, monkeypatch):
    client = client_with(plain_service(), evaluate=False)
    monkeypatch.setattr(deps, "_usage_settings", None)
    monkeypatch.delenv("USAGE_ENDPOINT_ENABLED", raising=False)
    monkeypatch.setattr(deps, "load_dotenv", lambda *a, **k: False)

    assert client.post("/api/v1/evaluate", json={}).status_code == 404
    assert client.get("/api/v1/usage").status_code == 404


def test_health_reports_budget_uptime_and_no_database_in_the_local_profile(client_with):
    client = client_with(plain_service(), budget=(400, 60))

    health = client.get("/api/v1/health").json()

    assert health["budget"] == {BUDGET_MODEL_SEARCH: 400, BUDGET_LLM_REQUEST: 60}
    assert health["database"] is None and health["uptime_s"] >= 0


def test_results_name_the_source_licence_and_attribution_link(client_with):
    client = client_with(plain_service())

    results = client.post("/api/v1/search", json={"text": "moby", "include_metadata": True}).json()["results"]
    moby = next(r["book"] for r in results if r["book"]["title"] == "Moby Dick")

    assert moby["description_source"] == {
        "source": "wikipedia", "language": "en", "license": "CC BY-SA 4.0",
        "attribution_url": "https://en.wikipedia.org/wiki/Moby-Dick", "auto_generated": False,
    }
    assert [p["source"] for p in moby["provenance"]] == ["wikidata", "wikipedia"]


# ---------------------------------------------------------------- degradation with the API adapters


def test_failing_query_embedding_degrades_to_lexical_only_after_one_retry(client_with):
    attempts: List[str] = []
    encoder = OpenRouterEmbeddingEncoder("qwen/qwen3-embedding-8b", dimensions=1024, upstream_provider="deepinfra",
                                         http=failing_http(attempts), query_call=QUERY_CALL)
    client = client_with(SearchService(Repo("lexical"), Repo("vector"), Embeddings(encoder)))

    body = client.post("/api/v1/search", json={"text": "sea"}).json()

    assert body["search_mode"] == "lexical_only" and body["degraded"] is True
    assert len(body["results"]) == 3
    assert attempts == ["/api/v1/embeddings"] * 2  # one attempt and at most one retry


def test_failing_rerank_keeps_the_fused_order_after_one_retry(client_with):
    attempts: List[str] = []
    # Built as the cloud profile builds it: 3 s per attempt, one retry, 5 s deadline
    reranker = ApiReranker("qwen/qwen3-reranker-8b", upstream_provider="primeintellect",
                           http=failing_http(attempts, timeout_s=3.0, max_retries=1), deadline_s=5.0)
    reranked = SearchService(Repo("lexical"), Repo("vector"), Embeddings(), reranker=reranker, rerank_depth=3)
    fused = SearchService(Repo("lexical"), Repo("vector"), Embeddings())

    body = client_with(reranked).post("/api/v1/search", json={"text": "sea"}).json()
    expected = client_with(fused).post("/api/v1/search", json={"text": "sea"}).json()

    assert body["search_mode"] == "hybrid" and not body["degraded"]
    assert [r["book"]["id"] for r in body["results"]] == [r["book"]["id"] for r in expected["results"]]
    assert all(r["rerank_score"] is None for r in body["results"])
    assert attempts == ["/api/v1/rerank"] * 2  # one attempt and at most one retry


def test_failing_llm_returns_the_results_without_explanations(client_with):
    llm = FailingLLM()
    client = client_with(SearchService(Repo("lexical"), Repo("vector"), Embeddings(), llm_client=llm,
                                       max_explanations=2))

    response = client.post("/api/v1/search", json={"text": "sea", "use_explanations": True})

    assert response.status_code == 200
    assert len(response.json()["results"]) == 3
    assert all(r["explanation"] is None for r in response.json()["results"])
    assert llm.calls == 2
