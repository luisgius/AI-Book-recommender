"""
Tests for the LLM usage recorder (app/infrastructure/llm/usage.py) and the
price settings it uses (TokenPrices in config/settings.py).

The chat models are LangChain fakes, so the real callback machinery runs
without any network call.
"""

import json
from typing import Any, TypedDict
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

import app.infrastructure.llm.langchain_llm_client as llm_client_module
from app.infrastructure.config.settings import TokenPrices
from app.infrastructure.llm.usage import (
    TOTAL,
    UNLABELLED,
    LLMCallRecord,
    LLMUsageRecorder,
    operation_config,
)


class FakeChatModel(GenericFakeChatModel):
    """Fake chat model with a model id, as real chat models have."""

    model: str = "fake-model"


class StructuredFakeChatModel(FakeChatModel):
    """Fake chat model whose structured output parses the JSON content."""

    def with_structured_output(self, schema: Any, **kwargs: Any):
        return self | RunnableLambda(lambda message: schema.model_validate_json(message.content))


class FailingChatModel(FakeChatModel):
    def _generate(self, *args: Any, **kwargs: Any):
        raise ConnectionError("server down")


def answer(content: str = "ok", input_tokens: int | None = None, output_tokens: int | None = None) -> AIMessage:
    """AI message with or without provider-reported token usage."""
    if input_tokens is None:
        return AIMessage(content=content)
    return AIMessage(content=content, usage_metadata={
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    })


def fake_model(recorder: LLMUsageRecorder, *answers: AIMessage, cls=FakeChatModel):
    return cls(messages=iter(answers), callbacks=[recorder])


# =============================================================================
# Recording
# =============================================================================


class TestRecording:
    def test_records_operation_model_latency_and_tokens(self):
        recorder = LLMUsageRecorder()
        model = fake_model(recorder, answer(input_tokens=120, output_tokens=30))

        model.invoke("hello", config=operation_config("explanation"))

        [record] = recorder.records
        assert (record.operation, record.model) == ("explanation", "fake-model")
        assert (record.input_tokens, record.output_tokens) == (120, 30)
        assert record.latency_s >= 0 and record.step is None and record.error is None

    def test_missing_usage_is_recorded_as_none(self):
        recorder = LLMUsageRecorder()
        fake_model(recorder, answer()).invoke("hello")
        [record] = recorder.records
        assert (record.input_tokens, record.output_tokens) == (None, None)

    def test_unlabelled_calls_get_the_default_operation(self):
        recorder = LLMUsageRecorder()
        fake_model(recorder, answer()).invoke("hello")
        judge_recorder = LLMUsageRecorder(default_operation="relevance_judge")
        fake_model(judge_recorder, answer()).invoke("hello")
        assert recorder.records[0].operation == UNLABELLED
        assert judge_recorder.records[0].operation == "relevance_judge"

    def test_failed_call_is_recorded(self):
        recorder = LLMUsageRecorder()
        with pytest.raises(ConnectionError):
            fake_model(recorder, cls=FailingChatModel).invoke("hello", config=operation_config("agent"))
        [record] = recorder.records
        assert (record.operation, record.error) == ("agent", "ConnectionError")
        assert record.input_tokens is None

    def test_label_reaches_calls_inside_a_langgraph_node(self):
        recorder = LLMUsageRecorder()
        model = fake_model(recorder, answer(input_tokens=5, output_tokens=2))

        class State(TypedDict):
            question: str
            reply: str

        def ask(state: State) -> dict:
            # The node does not pass any config: LangChain propagates it.
            return {"reply": model.invoke(state["question"]).content}

        graph = StateGraph(State)
        graph.add_node("ask", ask)
        graph.add_edge(START, "ask")
        graph.add_edge("ask", END)
        graph.compile().invoke({"question": "hi", "reply": ""}, config=operation_config("agent"))

        [record] = recorder.records
        assert (record.operation, record.step) == ("agent", "ask")

    def test_clear(self):
        recorder = LLMUsageRecorder()
        fake_model(recorder, answer()).invoke("hello")
        recorder.clear()
        assert recorder.records == []

    def test_records_returns_a_copy(self):
        recorder = LLMUsageRecorder()
        fake_model(recorder, answer()).invoke("hello")
        recorder.records.clear()
        assert len(recorder.records) == 1


# =============================================================================
# Summary and cost
# =============================================================================


def recorder_with(*records: LLMCallRecord) -> LLMUsageRecorder:
    recorder = LLMUsageRecorder()
    recorder._records.extend(records)
    return recorder


def record(operation: str, latency: float, tokens_in: int | None, tokens_out: int | None,
           error: str | None = None) -> LLMCallRecord:
    return LLMCallRecord(operation=operation, model="m", latency_s=latency,
                         input_tokens=tokens_in, output_tokens=tokens_out, error=error)


class TestSummary:
    def test_aggregates_by_operation_with_a_total(self):
        recorder = recorder_with(
            record("query_understanding", 1.0, 900, 40),
            record("query_understanding", 3.0, 800, 60),
            record("query_understanding", 2.0, 850, 50),
            record("explanation", 4.0, 1000, 300),
        )
        summary = recorder.summary()

        assert set(summary) == {"query_understanding", "explanation", TOTAL}
        understanding = summary["query_understanding"]
        assert (understanding["calls"], understanding["input_tokens"], understanding["output_tokens"]) == (3, 2550, 150)
        assert (understanding["median_latency_s"], understanding["mean_latency_s"]) == (2.0, 2.0)
        assert understanding["total_latency_s"] == 6.0
        assert understanding["estimated_cost_usd"] is None  # no prices given
        assert (summary[TOTAL]["calls"], summary[TOTAL]["input_tokens"], summary[TOTAL]["output_tokens"]) == (4, 3550, 450)

    def test_cost_estimate_from_prices(self):
        recorder = recorder_with(record("explanation", 1.0, 1_000_000, 200_000))
        summary = recorder.summary(TokenPrices(input_per_mtok=0.15, output_per_mtok=0.47))
        assert summary["explanation"]["estimated_cost_usd"] == pytest.approx(0.15 + 0.2 * 0.47)

    def test_calls_without_usage_are_counted_not_guessed(self):
        recorder = recorder_with(
            record("explanation", 1.0, 100, 10),
            record("explanation", 1.0, None, None),
            record("explanation", 0.5, None, None, error="TimeoutError"),
        )
        row = recorder.summary(TokenPrices(1.0, 1.0))["explanation"]
        assert (row["calls"], row["calls_without_usage"], row["failed_calls"]) == (3, 2, 1)
        assert (row["input_tokens"], row["output_tokens"]) == (100, 10)
        assert row["estimated_cost_usd"] == pytest.approx(110 / 1_000_000)

    def test_no_reported_tokens_gives_none_not_zero(self):
        row = recorder_with(record("agent", 1.0, None, None)).summary(TokenPrices(1.0, 1.0))["agent"]
        assert (row["input_tokens"], row["output_tokens"], row["estimated_cost_usd"]) == (None, None, None)

    def test_empty_recorder(self):
        summary = LLMUsageRecorder().summary()
        assert summary == {TOTAL: {
            "calls": 0, "failed_calls": 0, "calls_without_usage": 0, "input_tokens": None,
            "output_tokens": None, "total_latency_s": 0, "mean_latency_s": None,
            "median_latency_s": None, "estimated_cost_usd": None,
        }}


class TestTokenPrices:
    def test_read_from_the_environment(self):
        prices = TokenPrices.from_env({"LLM_PRICE_INPUT_PER_MTOK": "0.15", "LLM_PRICE_OUTPUT_PER_MTOK": "0.47"})
        assert (prices.input_per_mtok, prices.output_per_mtok) == (0.15, 0.47)
        assert prices.cost(2_000_000, 1_000_000) == pytest.approx(0.77)

    def test_none_when_not_configured(self):
        assert TokenPrices.from_env({}) is None

    def test_both_variables_are_needed(self):
        with pytest.raises(ValueError, match="LLM_PRICE_OUTPUT_PER_MTOK"):
            TokenPrices.from_env({"LLM_PRICE_INPUT_PER_MTOK": "0.15"})

    def test_negative_price_is_rejected(self):
        with pytest.raises(ValueError):
            TokenPrices(input_per_mtok=-1.0, output_per_mtok=0.5)


# =============================================================================
# Through the LLM client (real graph and chain, fake chat model)
# =============================================================================


def build_client_with(model_answers: list) -> llm_client_module.LangChainLLMClient:
    """LangChainLLMClient whose factory returns a fake model with the client's recorder."""
    def factory(**kwargs):
        return StructuredFakeChatModel(messages=iter(model_answers), callbacks=kwargs.get("callbacks"))

    with patch.object(llm_client_module, "load_dotenv"), \
         patch.object(llm_client_module, "init_chat_model", side_effect=factory):
        return llm_client_module.LangChainLLMClient(provider="ollama", model_name="fake-model")


def test_query_understanding_calls_are_recorded_per_graph_node():
    client = build_client_with([
        answer(json.dumps({"intent_type": "exploratory", "confidence": 0.8,
                           "reasoning": "A topic search for mystery novels."}), 700, 40),
        answer(json.dumps({"language": None, "category": None, "min_year": None,
                           "max_year": None, "author_hint": None}), 900, 30),
        answer(json.dumps({"optimized_query": "victorian mystery novels", "keywords": ["victorian", "mystery"],
                           "removed_noise": []}), 850, 35),
    ])

    intent = client.extract_query_intent("victorian mystery novels")

    assert intent.intent_type == "exploratory"
    records = client.usage_recorder.records
    assert [r.operation for r in records] == ["query_understanding"] * 3
    assert [r.step for r in records] == ["parse_intent", "extract_filters", "reformulate_query"]
    row = client.usage_recorder.summary()["query_understanding"]
    assert (row["calls"], row["input_tokens"], row["output_tokens"]) == (3, 2450, 105)
