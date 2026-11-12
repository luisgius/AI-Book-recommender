"""Tests for POST /agent/ask (agentic search endpoint)."""

from datetime import datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import dependencies as deps
from app.domain.entities import AgentAnswer, AgentRecommendation, AgentStep, AgentStepBook, Book
from app.main import app


class FakeAgent:
    def __init__(self, error: Exception = None):
        self.error = error

    def answer(self, question: str) -> AgentAnswer:
        if self.error:
            raise self.error
        book = Book(id=uuid4(), title="War Novel", authors=["A"], language="en",
                    published_date=datetime(2001, 1, 1))
        return AgentAnswer(
            question=question, answer="One book fits.",
            recommendations=[AgentRecommendation(book=book, reason="It is about war.")],
            steps=[AgentStep(tool="search_books", arguments={"query": "war"}, result_count=1,
                             books=[AgentStepBook(book_id="b1", title="War Novel")])],
            model="fake", dropped_recommendations=1, reflections=0,
        )


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_returns_answer_recommendations_and_trace(client):
    app.dependency_overrides[deps.get_search_agent] = lambda: FakeAgent()

    response = client.post("/api/v1/agent/ask", json={"question": "a war novel"})

    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "One book fits."
    assert body["recommendations"][0]["book"]["title"] == "War Novel"
    assert body["recommendations"][0]["reason"] == "It is about war."
    assert body["steps"] == [{"tool": "search_books", "arguments": {"query": "war"}, "result_count": 1,
                              "books": [{"book_id": "b1", "title": "War Novel"}]}]
    assert body["dropped_recommendations"] == 1


def test_without_llm_returns_503_with_how_to_enable_it(client):
    app.dependency_overrides[deps.get_search_agent] = lambda: None

    response = client.post("/api/v1/agent/ask", json={"question": "a war novel"})

    assert response.status_code == 503
    assert "OPENROUTER_API_KEY" in response.json()["detail"]
    assert "Plain search still works" in response.json()["detail"]


def test_unreachable_llm_returns_503(client):
    app.dependency_overrides[deps.get_search_agent] = lambda: FakeAgent(RuntimeError("LLM unreachable"))
    assert client.post("/api/v1/agent/ask", json={"question": "a war novel"}).status_code == 503


def test_too_short_question_is_a_validation_error(client):
    app.dependency_overrides[deps.get_search_agent] = lambda: FakeAgent()
    assert client.post("/api/v1/agent/ask", json={"question": "a"}).status_code == 422
