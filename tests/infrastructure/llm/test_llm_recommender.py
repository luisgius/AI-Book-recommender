"""
Tests for the LLM recommender adapter (app/infrastructure/llm/llm_recommender.py),
its prompts (prompts_recommender.py) and schemas (schemas_recommender.py).

No LLM is called: the recommender gets a fake Completion (a plain function),
and LangChainCompletion gets a LangChain fake chat model.
"""

import json
from datetime import datetime
from typing import Any, List
from uuid import uuid4

import httpx
import openai
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.domain.entities import Book, SearchResult
from app.domain.ports import Reranker
from app.domain.services import SearchService
from app.domain.value_objects import SearchQuery
from app.infrastructure.llm.llm_recommender import (
    CLOSED_BOOK_OPERATION,
    LONG_CONTEXT_OPERATION,
    RERANK_OPERATION,
    LangChainCompletion,
    LLMListwiseReranker,
    LLMOutput,
    LLMRecommender,
    LLMRequest,
    candidate_ids,
    check_ids,
    complete_order,
    listwise_request,
    normalize_id,
)
from app.infrastructure.llm.prompts_recommender import (
    DESCRIPTION_MAX_CHARS,
    MISSING,
    build_closed_book_prompt,
    truncate_description,
)
from app.infrastructure.llm.schemas_recommender import (
    REASON_MAX_CHARS,
    BookRecommendationsLLM,
    CandidateRecommendationsLLM,
)
from tests.domain.test_reranking import FakeEmbeddings, RecordingRepo


def make_book(i: int, **overrides) -> Book:
    fields = dict(
        id=uuid4(),
        title=f"Book {i:02d}",
        authors=[f"Author {i}"],
        description=f"Description of book {i}.",
        language="en",
        categories=["Fiction"],
        published_date=datetime(2000 + i % 20, 1, 1),
        source="google_books",
        source_id=f"id{i:02d}",
    )
    fields.update(overrides)
    return Book(**fields)


def answer(ids: List[str]) -> LLMOutput:
    """A usable listwise answer with these ids."""
    return LLMOutput(parsed={"recommendations": [{"id": i, "reason": f"because {i}"} for i in ids]},
                     latency_s=0.5, input_tokens=100, output_tokens=20, cost_usd=0.0001)


class FakeCompletion:
    """Records the requests and returns a fixed output."""

    def __init__(self, output: LLMOutput) -> None:
        self.output = output
        self.requests: List[LLMRequest] = []

    def __call__(self, request: LLMRequest) -> LLMOutput:
        self.requests.append(request)
        return self.output


def results_for(books: List[Book]) -> List[SearchResult]:
    return [SearchResult(book=b, final_score=1.0 / (i + 1), rank=i + 1, source="hybrid",
                         lexical_score=2.0, vector_score=0.5) for i, b in enumerate(books)]


# =============================================================================
# Prompts
# =============================================================================


class TestPrompts:
    def test_listwise_prompt_shows_every_candidate_and_ends_with_the_query(self):
        books = [make_book(i) for i in range(30)]
        request = listwise_request("machine learning python", books, RERANK_OPERATION)
        for book_id, book in zip(candidate_ids(30), books):
            assert f"[{book_id}] title: {book.title};" in request.user_prompt
        assert request.user_prompt.index("[B30]") < request.user_prompt.index('Query: "machine learning python"')
        assert "Choose the 10 books" in request.user_prompt
        assert "only recommend books from the list" in request.system_prompt
        assert request.schema is CandidateRecommendationsLLM

    def test_entry_has_every_field(self):
        book = make_book(1, title="Clean  Code", authors=["Robert C. Martin", "X"], language="es",
                         categories=["Computers", "Software"], published_date=datetime(2008, 8, 1))
        prompt = listwise_request("q", [book], RERANK_OPERATION).user_prompt
        assert ("[B1] title: Clean Code; authors: Robert C. Martin, X; year: 2008; language: es; "
                "categories: Computers, Software\ndescription: Description of book 1.") in prompt

    def test_description_never_exceeds_the_limit(self):
        long_text = " ".join(["word"] * 400)  # 1999 characters
        book = make_book(1, description=long_text)
        prompt = listwise_request("q", [book], RERANK_OPERATION).user_prompt
        line = next(line for line in prompt.splitlines() if line.startswith("description: "))
        shown = line[len("description: "):]
        assert len(shown) <= DESCRIPTION_MAX_CHARS and shown.endswith("...")
        assert long_text not in prompt

    def test_truncation_cuts_at_a_word_boundary(self):
        text = "alpha beta gamma delta"
        assert truncate_description(text, limit=15) == "alpha beta..."
        assert truncate_description("  short   text ") == "short text"

    @pytest.mark.parametrize("description", [None, "", "   "])
    def test_missing_description_is_named(self, description):
        assert truncate_description(description) == MISSING

    def test_missing_fields_are_named(self):
        book = make_book(1, description=None, categories=[], published_date=None, language=None)
        prompt = listwise_request("q", [book], RERANK_OPERATION).user_prompt
        assert "year: unknown; language: unknown; categories: (none)\ndescription: (none)" in prompt

    def test_closed_book_prompt_has_the_query_and_no_catalog(self):
        system, user = build_closed_book_prompt(" novela guerra civil ", 10)
        assert 'Query: "novela guerra civil"' in user and "10 real, published books" in user
        assert "[B1]" not in user and "list" not in system


# =============================================================================
# Id validation and completion
# =============================================================================


class TestIdRules:
    def test_normalize_id(self):
        assert normalize_id(" [b7] ") == "B7"
        assert normalize_id("B12") == "B12"

    def test_check_ids_splits_valid_unknown_and_duplicates(self):
        valid, unknown, duplicates = check_ids(["B2", "b2", "B9", "X1", "[B1]", "B31"], candidate_ids(30)[:5])
        assert valid == ["B2", "B1"]
        assert unknown == ["B9", "X1", "B31"]
        assert duplicates == ["b2"]

    def test_complete_order_puts_picks_first_then_the_shown_order(self):
        order, filled = complete_order([4, 1], 6, top_n=4)
        assert order == [4, 1, 0, 2, 3, 5]
        assert filled == 2

    def test_no_filling_with_enough_picks(self):
        order, filled = complete_order(list(range(9, -1, -1)), 12, top_n=10)
        assert order[:10] == list(range(9, -1, -1)) and filled == 0

    def test_fewer_candidates_than_top_n(self):
        assert complete_order([2], 3, top_n=10) == ([2, 0, 1], 2)


class TestRankBooks:
    def test_valid_ids_come_first_and_the_list_is_completed(self):
        books = [make_book(i) for i in range(30)]
        recommender = LLMRecommender(FakeCompletion(answer(["B5", "B99", "B5", "B3"])), "m")
        outcome = recommender.rank_books("q", books)
        assert outcome.valid_ids == ["B5", "B3"]
        assert outcome.unknown_ids == ["B99"] and outcome.duplicate_ids == ["B5"]
        assert outcome.filled == 8 and outcome.completed and not outcome.failed
        assert [b.title for b in outcome.ranked_books[:4]] == ["Book 04", "Book 02", "Book 00", "Book 01"]
        assert sorted(outcome.order) == list(range(30))
        assert outcome.reasons == {"B5": "because B5", "B3": "because B3"}

    def test_more_than_ten_ids_are_all_kept_in_order(self):
        books = [make_book(i) for i in range(30)]
        ids = [f"B{i}" for i in range(12, 0, -1)]
        outcome = LLMRecommender(FakeCompletion(answer(ids)), "m").rank_books("q", books)
        assert outcome.valid_ids == ids and outcome.filled == 0
        assert outcome.order[:12] == [11 - k for k in range(12)]

    def test_unusable_answer_keeps_the_shown_order(self):
        books = [make_book(i) for i in range(12)]
        failed = LLMOutput(parsed=None, error="ValidationError: bad json")
        outcome = LLMRecommender(FakeCompletion(failed), "m").rank_books("q", books, LONG_CONTEXT_OPERATION)
        assert outcome.failed and outcome.order == list(range(12)) and outcome.filled == 10
        assert outcome.returned_ids == []

    def test_operation_label_is_sent(self):
        completion = FakeCompletion(answer(["B1"]))
        LLMRecommender(completion, "m").rank_books("q", [make_book(1)], LONG_CONTEXT_OPERATION)
        assert completion.requests[0].operation == LONG_CONTEXT_OPERATION

    def test_closed_book(self):
        output = LLMOutput(parsed={"recommendations": [{"title": " Dune ", "author": "Frank  Herbert", "reason": "x"}]})
        completion = FakeCompletion(output)
        outcome = LLMRecommender(completion, "m").recommend_closed_book("science fiction")
        assert outcome.recommendations == [{"title": "Dune", "author": "Frank Herbert", "reason": "x"}]
        assert completion.requests[0].operation == CLOSED_BOOK_OPERATION
        assert completion.requests[0].schema is BookRecommendationsLLM


# =============================================================================
# Reranker contract
# =============================================================================


class TestReranker:
    def test_is_a_reranker(self):
        assert Reranker in LLMListwiseReranker.__mro__  # the port is not runtime-checkable

    def test_same_results_reordered_with_new_ranks(self):
        books = [make_book(i) for i in range(30)]
        results = results_for(books)
        before = [(r.book.id, r.rank, r.rerank_score) for r in results]
        seen = []
        reranker = LLMListwiseReranker(LLMRecommender(FakeCompletion(answer(["B3", "B1", "B30"])), "m"),
                                       observer=seen.append)
        reranked = reranker.rerank("q", results)

        assert {r.book.id for r in reranked} == {r.book.id for r in results} and len(reranked) == 30
        assert [r.book.title for r in reranked[:4]] == ["Book 02", "Book 00", "Book 29", "Book 01"]
        assert [r.rank for r in reranked] == list(range(1, 31))
        assert [r.rerank_score for r in reranked[:3]] == [1.0, 0.5, pytest.approx(1 / 3)]
        moved = reranked[0]
        assert (moved.final_score, moved.lexical_score, moved.vector_score) == (1 / 3, 2.0, 0.5)
        assert [(r.book.id, r.rank, r.rerank_score) for r in results] == before  # inputs untouched
        assert len(seen) == 1 and seen[0].valid_ids == ["B3", "B1", "B30"]

    def test_empty_input(self):
        completion = FakeCompletion(answer([]))
        assert LLMListwiseReranker(LLMRecommender(completion, "m")).rerank("q", []) == []
        assert completion.requests == []

    def test_failed_request_raises_runtime_error(self):
        def broken(request):
            raise ConnectionError("down")

        with pytest.raises(RuntimeError, match="LLM reranking failed"):
            LLMListwiseReranker(LLMRecommender(broken, "m")).rerank("q", results_for([make_book(1)]))

    def test_model_name_and_ready(self):
        reranker = LLMListwiseReranker(LLMRecommender(FakeCompletion(answer([])), "qwen/x"))
        assert reranker.get_model_name() == "qwen/x" and reranker.is_ready()

    def test_plugs_into_search_service(self):
        books = [make_book(i) for i in range(40)]
        completion = FakeCompletion(answer(["B20", "B2"]))
        service = SearchService(
            lexical_search=RecordingRepo(books, "lexical"), vector_search=RecordingRepo(books, "vector"),
            embeddings_store=FakeEmbeddings(books),
            reranker=LLMListwiseReranker(LLMRecommender(completion, "m")), rerank_depth=30,
        )
        response = service.search_with_fallback(SearchQuery(text="q", max_results=10, use_explanations=False))
        assert [r.book.title for r in response.results[:3]] == ["Book 19", "Book 01", "Book 00"]
        assert len(response.results) == 10 and response.results[0].rerank_score == 1.0
        assert completion.requests[0].user_prompt.count("[B") == 30  # the top 30 fused candidates


# =============================================================================
# LangChainCompletion
# =============================================================================


class StructuredFake(GenericFakeChatModel):
    """Fake chat model whose structured output parses the JSON content (include_raw supported)."""

    model: str = "fake-model"

    def with_structured_output(self, schema: Any, **kwargs: Any):
        def parse(message: AIMessage):
            parsed = schema.model_validate_json(message.content)
            return {"raw": message, "parsed": parsed, "parsing_error": None} if kwargs.get("include_raw") else parsed

        return self | RunnableLambda(parse)


class RequestFailingFake(StructuredFake):
    def _generate(self, *args: Any, **kwargs: Any):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))


def request(schema=CandidateRecommendationsLLM) -> LLMRequest:
    return LLMRequest(RERANK_OPERATION, "q", "system", "user", schema)


class TestLangChainCompletion:
    def test_parsed_answer_with_usage(self):
        content = json.dumps({"recommendations": [{"id": "B1", "reason": "r"}]})
        message = AIMessage(content=content, usage_metadata={
            "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
            "input_token_details": {"cache_read": 100}})
        output = LangChainCompletion(StructuredFake(messages=iter([message])))(request())
        assert output.error is None and output.parsed == {"recommendations": [{"id": "B1", "reason": "r"}]}
        assert (output.input_tokens, output.output_tokens, output.cached_input_tokens) == (120, 30, 100)
        assert output.content == content and output.latency_s >= 0

    def test_invalid_answer_is_recorded_as_an_error(self):
        output = LangChainCompletion(StructuredFake(messages=iter([AIMessage(content="not json")])))(request())
        assert output.parsed is None and output.error.startswith("ValidationError")

    def test_request_errors_propagate(self):
        with pytest.raises(openai.APIConnectionError):
            LangChainCompletion(RequestFailingFake(messages=iter([])))(request())

    def test_output_round_trips_through_a_dict(self):
        output = LLMOutput(parsed={"a": 1}, content="{}", latency_s=1.5, input_tokens=3, cost_usd=0.1)
        assert LLMOutput.from_dict(json.loads(json.dumps(output.to_dict()))) == output


# =============================================================================
# Schemas
# =============================================================================


class TestSchemas:
    def test_long_reason_is_cut_not_rejected(self):
        parsed = CandidateRecommendationsLLM.model_validate(
            {"recommendations": [{"id": "B1", "reason": "x" * (REASON_MAX_CHARS + 50)}]})
        assert len(parsed.recommendations[0].reason) == REASON_MAX_CHARS

    def test_schemas_have_no_example_keyword(self):
        for schema in (CandidateRecommendationsLLM, BookRecommendationsLLM):
            assert '"example"' not in json.dumps(schema.model_json_schema())  # no JSON Schema keyword

    def test_book_recommendation_fields(self):
        parsed = BookRecommendationsLLM.model_validate(
            {"recommendations": [{"title": "Dune", "author": "Frank Herbert", "reason": "classic"}]})
        assert parsed.recommendations[0].author == "Frank Herbert"
