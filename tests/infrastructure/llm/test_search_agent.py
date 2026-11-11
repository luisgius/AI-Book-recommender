"""
Tests for the LangGraph search agent (Phase 4: tool use, ReAct, self-reflection).

The LLM is a scripted fake: it returns predefined AIMessages (with or without
tool calls) and predefined final recommendation lists. The tools run for real
over a SearchService with in-memory fakes, so the graph's routing, the
ToolNode, the observation parsing and the verification are all exercised.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage

from app.domain.entities import Book, SearchResult
from app.domain.services import SearchService
from app.infrastructure.llm.agents.search_agent import (
    LangGraphSearchAgent,
    RecommendationListLLM,
    RecommendedBookLLM,
)


def make_book(title: str, language: str = "en") -> Book:
    return Book(id=uuid4(), title=title, authors=["Author"], language=language, categories=["Fiction"],
                published_date=datetime(2001, 1, 1), description=f"A novel called {title}.")


BOOKS = [make_book("War Novel"), make_book("Novela de la guerra", "es"), make_book("Cookbook")]


class Repo:
    def __init__(self, source: str):
        self._source = source

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        matching = [b for b in BOOKS if b.matches_filters(filters)][:max_results]
        field = "lexical_score" if self._source == "lexical" else "vector_score"
        return [SearchResult(book=b, final_score=1.0 / (i + 1), rank=i + 1, source=self._source,
                             **{field: 1.0}) for i, b in enumerate(matching)]

    def is_ready(self) -> bool:
        return True


class Embeddings:
    def generate_embedding(self, text: str) -> List[float]:
        return [0.1, 0.2]

    generate_query_embedding = generate_embedding  # same vector for queries

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return [0.1, 0.2]

    def get_embeddings_batch(self, book_ids):
        return {i: v for i in book_ids if (v := self.get_embedding(i)) is not None}

    def is_ready(self) -> bool:
        return True


class Catalog:
    def get_by_id(self, book_id: UUID) -> Optional[Book]:
        return next((b for b in BOOKS if b.id == book_id), None)


class Scripted:
    """Returns the next scripted output on each invoke and records the inputs."""

    def __init__(self, outputs: list):
        self._outputs = list(outputs)
        self.inputs: list = []

    def invoke(self, messages):
        self.inputs.append(messages)
        if not self._outputs:
            raise AssertionError("model called more times than scripted")
        return self._outputs.pop(0)


class FakeModel:
    """Chat model double with bind_tools and with_structured_output."""

    def __init__(self, turns: list, finals: list):
        self.turns = Scripted(turns)
        self.finals = Scripted(finals)

    def bind_tools(self, tools):
        return self.turns

    def with_structured_output(self, schema):
        return self.finals


def tool_call(name: str, call_id: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def final(answer: str, *book_ids: str) -> RecommendationListLLM:
    return RecommendationListLLM(
        answer=answer, recommendations=[RecommendedBookLLM(book_id=b, reason="fits") for b in book_ids]
    )


def build_agent(model: FakeModel, **kwargs) -> LangGraphSearchAgent:
    service = SearchService(Repo("lexical"), Repo("vector"), Embeddings())
    return LangGraphSearchAgent(model, service, Catalog(), model_name="fake", **kwargs)


def test_react_loop_searches_then_recommends_verified_books():
    war = str(BOOKS[0].id)
    model = FakeModel(
        turns=[tool_call("search_books", "c1", query="war novel"), AIMessage(content="Found them")],
        finals=[final("Here you go", war, str(uuid4()))],  # second id was never returned by a tool
    )

    result = build_agent(model).answer("novels about war")

    assert [r.book.title for r in result.recommendations] == ["War Novel"]
    assert result.dropped_recommendations == 1
    assert result.reflections == 0
    assert result.steps[0].tool == "search_books"
    assert result.steps[0].arguments == {"query": "war novel"}
    assert result.steps[0].result_count == 3
    # finalize sees the tool observations, not the agent's free text
    finalize_prompt = model.finals.inputs[0][0].content
    assert war in finalize_prompt and "Cookbook" in finalize_prompt


def test_tool_filters_are_passed_to_the_search():
    model = FakeModel(
        turns=[tool_call("search_books", "c1", query="guerra", language="ES"), AIMessage(content="ok")],
        finals=[final("Aquí está", str(BOOKS[1].id))],
    )

    result = build_agent(model).answer("novelas en español sobre la guerra")

    assert result.steps[0].result_count == 1
    assert [r.book.title for r in result.recommendations] == ["Novela de la guerra"]


def test_reflection_sends_the_agent_back_once_when_nothing_is_verified():
    model = FakeModel(
        turns=[
            AIMessage(content="I already know a great book"),          # no tool use
            tool_call("search_books", "c2", query="war"),              # after reflection
            AIMessage(content="done"),
        ],
        finals=[final("Invented", str(uuid4())), final("Real", str(BOOKS[0].id))],
    )

    result = build_agent(model).answer("a war novel")

    assert result.reflections == 1
    assert result.dropped_recommendations == 1
    assert [r.book.title for r in result.recommendations] == ["War Novel"]
    assert "Verification failed" in model.turns.inputs[1][-1].content


def test_gives_up_with_no_recommendations_after_max_reflections():
    model = FakeModel(
        turns=[AIMessage(content="no search"), AIMessage(content="still no search")],
        finals=[final("Invented", str(uuid4())), final("Invented again", str(uuid4()))],
    )

    result = build_agent(model, max_reflections=1).answer("anything")

    assert result.recommendations == []
    assert result.dropped_recommendations == 2
    assert result.reflections == 1


def test_tool_round_limit_stops_the_loop():
    model = FakeModel(
        turns=[tool_call("search_books", f"c{i}", query=f"q{i}") for i in range(5)],
        finals=[final("Enough", str(BOOKS[2].id))],
    )

    result = build_agent(model, max_tool_rounds=2).answer("cookbooks")

    assert len(result.steps) == 3  # the call made after the limit is not executed
    assert [s.result_count for s in result.steps[:2]] == [3, 3]
    assert [r.book.title for r in result.recommendations] == ["Cookbook"]


def test_details_tool_reports_unknown_books_without_failing():
    model = FakeModel(
        turns=[tool_call("get_book_details", "c1", book_id=str(uuid4())),
               tool_call("search_books", "c2", query="cook"), AIMessage(content="ok")],
        finals=[final("Done", str(BOOKS[2].id))],
    )

    result = build_agent(model).answer("details please")

    assert result.steps[0].tool == "get_book_details" and result.steps[0].result_count == 0
    assert [r.book.title for r in result.recommendations] == ["Cookbook"]


def test_duplicate_recommendations_are_kept_once():
    war = str(BOOKS[0].id)
    model = FakeModel(turns=[tool_call("search_books", "c1", query="war"), AIMessage(content="ok")],
                      finals=[final("Twice", war, war)])

    result = build_agent(model).answer("war")

    assert len(result.recommendations) == 1


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        build_agent(FakeModel([], [])).answer("  ")
