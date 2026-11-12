"""
Contract of POST /api/v1/assistant/chat, without network or model.

- With an assistant, the reply is mapped to the response (route, message,
  options, results, recommendations, steps, timings, usage).
- Without an LLM, or with the day's LLM budget spent, the endpoint runs the
  plain search of the last user message and says why in `notices`.
- Each request takes one LLM budget unit and counts toward the per-client
  LLM rate limit.
- The request is bounded: at most 50 messages of 5000 characters (422), and
  it must end with a user message (400).
"""

from datetime import UTC, datetime
from typing import List
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import dependencies as deps
from app.domain.entities import (
    AgentAnswer,
    AgentStep,
    AssistantRecommendation,
    AssistantReply,
    Book,
    Citation,
    SearchResult,
)
from app.domain.ports import BUDGET_LLM_REQUEST
from app.domain.services import SearchService
from app.domain.value_objects import ChatMessage, SearchFilters, SearchResponse
from app.infrastructure.config.settings import LLMSettings, RateLimitSettings, UsageBudgetSettings
from app.infrastructure.usage_budget import InMemoryUsageBudget
from app.main import app

BOOK = Book(id=uuid4(), title="Moby Dick", authors=["Herman Melville"], language="en",
            description="A novel about a whale.", published_date=datetime(1851, 1, 1, tzinfo=UTC))


class Repo:
    def __init__(self, source: str):
        self._source = source

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        field = "lexical_score" if self._source == "lexical" else "vector_score"
        return [SearchResult(book=BOOK, final_score=1.0, rank=1, source=self._source, **{field: 1.0})]

    def is_ready(self) -> bool:
        return True


class Embeddings:
    def generate_query_embedding(self, text: str) -> List[float]:
        return [0.1, 0.2]

    generate_embedding = generate_query_embedding

    def is_ready(self) -> bool:
        return True


class FakeAssistant:
    def __init__(self, reply: AssistantReply):
        self._reply = reply
        self.received: List[List[ChatMessage]] = []

    def reply(self, messages, max_results=10):
        self.received.append(messages)
        if not messages or messages[-1].role != "user":
            raise ValueError("the conversation must end with a user message")
        return self._reply


@pytest.fixture
def client_with(monkeypatch):
    def build(assistant, *, budget=None, rate=RateLimitSettings()):
        service = SearchService(Repo("lexical"), Repo("vector"), Embeddings())
        app.dependency_overrides[deps.get_search_service] = lambda: service
        app.dependency_overrides[deps.get_assistant] = lambda: assistant
        monkeypatch.setattr(deps, "_llm_settings", LLMSettings(provider="ollama"))
        monkeypatch.setattr(deps, "_rate_limit_settings", rate)
        monkeypatch.setattr(deps, "_rest_rate_limits", None)
        monkeypatch.setattr(deps, "_budget_settings", UsageBudgetSettings(*budget) if budget else UsageBudgetSettings())
        monkeypatch.setattr(deps, "_usage_budget", InMemoryUsageBudget(UsageBudgetSettings(*budget).limits())
                            if budget else None)
        monkeypatch.setattr(deps, "_budget_initialized", True)
        return TestClient(app)
    yield build
    app.dependency_overrides.clear()


def conversation(*texts: str) -> dict:
    roles = ["user", "assistant"]
    return {"messages": [{"role": roles[i % 2], "content": t, **({"route": "clarify"} if i % 2 else {})}
                         for i, t in enumerate(texts)]}


def search_reply() -> AssistantReply:
    result = SearchResult(book=BOOK, final_score=0.9, rank=1, source="hybrid", lexical_score=1.0)
    return AssistantReply(
        route="search", message="Moby Dick fits.", search_query="novels about whales",
        filters=SearchFilters(language="en"), search_response=SearchResponse(results=[result]),
        recommendations=[AssistantRecommendation(book=BOOK, reason="It is about a whale.", citations=[
            Citation(book_id=BOOK.id, chunk_id="description", snippet="a whale", relevance_score=0.9)])],
        timings_ms={"route": 800.0, "retrieve": 120.0, "answer": 900.0, "verify": 0.1},
        model="fake", usage={"llm_calls": 2, "input_tokens": 1500, "output_tokens": 200, "cost_usd": 0.0004},
    )


def test_search_reply_contract(client_with):
    client = client_with(FakeAssistant(search_reply()))

    response = client.post("/api/v1/assistant/chat", json=conversation("novels about whales"))

    body = response.json()
    assert response.status_code == 200
    assert body["route"] == "search" and body["message"] == "Moby Dick fits."
    assert body["search_query"] == "novels about whales"
    assert body["filters"] == {"language": "en", "category": None, "min_year": None, "max_year": None}
    assert body["results"][0]["book"]["title"] == "Moby Dick"
    assert body["recommendations"][0]["citations"] == [
        {"field": "description", "snippet": "a whale", "relevance_score": 0.9}]
    assert body["timings_ms"] == {"route": 800.0, "retrieve": 120.0, "answer": 900.0, "verify": 0.1}
    assert body["usage"] == {"llm_calls": 2, "failed_calls": 0, "input_tokens": 1500, "output_tokens": 200,
                             "cost_usd": 0.0004}
    assert body["options"] == [] and body["steps"] == [] and body["notices"] == []
    assert "latency_ms" in body


def test_clarify_and_agent_replies(client_with):
    clarify = AssistantReply(route="clarify", message="¿De qué tema?", options=["Historia", "Terror"], clarifications=1)
    body = client_with(FakeAssistant(clarify)).post("/api/v1/assistant/chat", json=conversation("novela")).json()
    assert (body["route"], body["options"], body["clarifications"], body["results"]) == (
        "clarify", ["Historia", "Terror"], 1, [])

    agent = AssistantReply(route="agent", message="Moby Dick is longer.", search_query="compare",
                           agent_answer=AgentAnswer(question="compare", answer="Moby Dick is longer.",
                                                    recommendations=[], model="fake",
                                                    steps=[AgentStep(tool="get_book_details",
                                                                     arguments={"book_id": "x"}, result_count=1)]),
                           recommendations=[AssistantRecommendation(book=BOOK, reason="Longer.")])
    body = client_with(FakeAssistant(agent)).post("/api/v1/assistant/chat", json=conversation("compare")).json()
    assert body["steps"] == [{"tool": "get_book_details", "arguments": {"book_id": "x"}, "result_count": 1, "books": []}]
    assert body["recommendations"][0]["citations"] == []


def test_the_conversation_and_routes_reach_the_assistant(client_with):
    assistant = FakeAssistant(search_reply())
    client = client_with(assistant)

    client.post("/api/v1/assistant/chat", json=conversation("novela", "¿De qué tema?", "ballenas"))

    sent = assistant.received[0]
    assert [(m.role, m.content, m.route) for m in sent] == [
        ("user", "novela", None), ("assistant", "¿De qué tema?", "clarify"), ("user", "ballenas", None)]


def test_without_llm_it_degrades_to_the_plain_search(client_with):
    client = client_with(None)

    body = client.post("/api/v1/assistant/chat", json=conversation("novelas de ballenas")).json()

    assert body["route"] == "search"
    assert body["notices"] == ["llm_unavailable"]
    assert body["search_query"] == "novelas de ballenas"
    assert body["results"][0]["book"]["title"] == "Moby Dick"
    assert body["message"].startswith("El asistente no está disponible")
    assert body["usage"]["llm_calls"] == 0


def test_spent_llm_budget_degrades_to_the_plain_search(client_with):
    client = client_with(FakeAssistant(search_reply()), budget=(None, 1))

    first = client.post("/api/v1/assistant/chat", json=conversation("whale novels")).json()
    second = client.post("/api/v1/assistant/chat", json=conversation("whale novels")).json()

    assert first["notices"] == [] and first["message"] == "Moby Dick fits."
    assert second["notices"] == ["llm_budget_spent"]
    assert second["message"].startswith("The assistant is not available")


def test_each_request_takes_one_llm_unit(client_with):
    client = client_with(FakeAssistant(search_reply()), budget=(None, 10))

    client.post("/api/v1/assistant/chat", json=conversation("novela", "¿Tema?", "ballenas"))

    assert deps.get_usage_budget().used_today() == {BUDGET_LLM_REQUEST: 1}


def test_llm_rate_limit_applies(client_with):
    client = client_with(FakeAssistant(search_reply()), rate=RateLimitSettings(llm_per_minute=1))

    assert client.post("/api/v1/assistant/chat", json=conversation("a")).status_code == 200
    second = client.post("/api/v1/assistant/chat", json=conversation("a"))
    assert second.status_code == 429 and "Retry-After" in second.headers


def test_a_conversation_ending_with_the_assistant_is_a_bad_request(client_with):
    client = client_with(FakeAssistant(search_reply()))

    response = client.post("/api/v1/assistant/chat", json=conversation("novela", "¿De qué tema?"))

    assert response.status_code == 400


@pytest.mark.parametrize("body", [
    {"messages": []},
    {"messages": [{"role": "user", "content": "x"}] * 51},
    {"messages": [{"role": "user", "content": "x" * 5001}]},
    {"messages": [{"role": "system", "content": "x"}]},
    {"messages": [{"role": "assistant", "content": "x", "route": "chat"}, {"role": "user", "content": "x"}]},
    {"messages": [{"role": "user", "content": "x"}], "max_results": 0},
])
def test_invalid_requests_are_rejected(client_with, body):
    assert client_with(FakeAssistant(search_reply())).post("/api/v1/assistant/chat", json=body).status_code == 422


def test_the_existing_endpoints_are_unchanged(client_with):
    paths = {route.path for route in app.routes}
    assert {"/api/v1/search", "/api/v1/agent/ask", "/api/v1/assistant/chat"} <= paths
