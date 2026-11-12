"""Bounds of an assistant conversation and the plain-search reply (domain, no LLM)."""

from typing import List
from uuid import uuid4

import pytest

from app.domain.entities import Book, SearchResult
from app.domain.services import NOTICE_LLM_UNAVAILABLE, SearchService, plain_search_reply
from app.domain.value_objects import (
    MAX_CONVERSATION_MESSAGES,
    MAX_MESSAGE_CHARS,
    ChatMessage,
    bound_conversation,
    count_clarifications,
    last_user_message,
)


def user(text: str) -> ChatMessage:
    return ChatMessage(role="user", content=text)


def bot(text: str, route=None) -> ChatMessage:
    return ChatMessage(role="assistant", content=text, route=route)


def test_older_turns_are_dropped_and_long_messages_cut():
    messages = [m for i in range(10) for m in (user(f"u{i} " + "x" * 2000), bot(f"a{i}", "search"))] + [user("last")]

    bounded = bound_conversation(messages)

    assert len(bounded) == MAX_CONVERSATION_MESSAGES
    assert bounded[-1].content == "last"
    assert all(len(m.content) <= MAX_MESSAGE_CHARS for m in bounded)
    assert bounded[0].content.startswith("a5")  # the most recent ten are kept


def test_empty_messages_are_ignored_and_the_last_must_be_the_users():
    assert [m.content for m in bound_conversation([user("  "), user("hola ")])] == ["hola"]
    with pytest.raises(ValueError):
        bound_conversation([user("hola"), bot("¿Qué buscas?")])
    with pytest.raises(ValueError):
        bound_conversation([])


def test_clarifications_are_counted_from_routes_and_missing_routes_count():
    messages = [user("a"), bot("?", "clarify"), user("b"), bot("ok", "search"), user("c"), bot("?"), user("d")]

    assert count_clarifications(messages) == 2
    assert last_user_message(messages) == "d"


def test_invalid_roles_and_routes_are_rejected():
    with pytest.raises(ValueError):
        ChatMessage(role="system", content="x")
    with pytest.raises(ValueError):
        ChatMessage(role="assistant", content="x", route="chat")


class Repo:
    def __init__(self):
        self.book = Book(id=uuid4(), title="Moby Dick", authors=["Melville"])

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        return [SearchResult(book=self.book, final_score=1.0, rank=1, source="lexical", lexical_score=1.0)]

    def is_ready(self) -> bool:
        return True


class Embeddings:
    def generate_query_embedding(self, text: str) -> List[float]:
        return [0.1]

    def is_ready(self) -> bool:
        return True


def test_plain_search_reply_searches_the_last_user_message():
    service = SearchService(Repo(), Repo(), Embeddings())

    reply = plain_search_reply(service, [user("novela"), bot("?", "clarify"), user("ballenas")], 5,
                               message="fixed", notice=NOTICE_LLM_UNAVAILABLE)

    assert reply.route == "search" and reply.search_query == "ballenas"
    assert reply.message == "fixed" and reply.notices[0] == NOTICE_LLM_UNAVAILABLE
    assert reply.search_response.results[0].book.title == "Moby Dick"
    assert reply.clarifications == 1 and "retrieve" in reply.timings_ms
