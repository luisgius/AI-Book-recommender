"""
Tests for the V16 evaluation (app/evaluation/api_models_evaluation.py) and the
speed-quality rule (app/evaluation/llm_model_selection.py). No network and no
model: a fake transport plays OpenRouter.
"""

import base64
import json
from typing import Any, Dict, List

import httpx
import numpy as np
import pytest

import app.evaluation.api_models_evaluation as v16
from app.evaluation.llm_model_selection import (
    select_by_speed_quality,
    select_judge_speed_quality,
    select_system_llm_speed_quality,
)
from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder
from app.infrastructure.search.openrouter_http import OpenRouterHttp


class FakeOpenRouter:
    """Embeddings and rerank answers; `fail_first` statuses are returned first."""

    def __init__(self, fail_first: List[int] = ()) -> None:
        self.fail_first = list(fail_first)
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.fail_first:
            return httpx.Response(self.fail_first.pop(0), json={"error": {"message": "busy"}})
        body = json.loads(request.content)
        if request.url.path.endswith("/rerank"):
            results = [{"index": i, "relevance_score": float(i), "document": {"text": d}}
                       for i, d in enumerate(body["documents"])][::-1]
            return httpx.Response(200, json={"results": results, "provider": "P",
                                             "usage": {"total_tokens": 9, "cost": 2e-6}})
        dimension = body.get("dimensions", 1024)
        data = [{"index": i, "embedding": base64.b64encode(np.ones(dimension, dtype="<f4").tobytes()).decode()}
                for i, _ in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data, "provider": "P",
                                         "usage": {"prompt_tokens": 4, "total_tokens": 4, "cost": 1e-7}})


def encoder_through(transport: v16.CachingTransport) -> OpenRouterEmbeddingEncoder:
    http = OpenRouterHttp(api_key="k", transport=transport, max_retries=2, sleep=lambda s: None)
    return OpenRouterEmbeddingEncoder("baai/bge-m3", upstream_provider="deepinfra", http=http)


class TestCachingTransport:
    def test_new_calls_are_cached_per_run_and_replayed_offline(self, tmp_path):
        cache = v16.ResponseCache(tmp_path / "responses.jsonl")
        inner = FakeOpenRouter()
        live = v16.CachingTransport(cache, inner)
        encoder = encoder_through(live)

        live.run = "run1"
        first = encoder.generate_query_embedding("q")
        encoder.generate_query_embedding("q")  # same run and body: a cache hit
        live.run = "run2"
        encoder.generate_query_embedding("q")  # another run: a new call
        assert inner.calls == 2 and live.calls_made == 2
        assert live.spent_usd == pytest.approx(2e-7)

        offline = v16.CachingTransport(v16.ResponseCache(tmp_path / "responses.jsonl"), None)
        replay = encoder_through(offline)
        offline.run = "run1"
        assert replay.generate_query_embedding("q") == first
        entry = offline.history[-1]
        assert entry["run"] == "run1" and entry["attempts"] == 1 and entry["provider_pin"] == ["deepinfra"]

    def test_offline_miss_fails_loudly(self, tmp_path):
        offline = v16.CachingTransport(v16.ResponseCache(tmp_path / "none.jsonl"), None)
        offline.run = "run1"
        with pytest.raises(v16.MissingApiResponseError, match="without --offline"):
            encoder_through(offline).generate_query_embedding("q")
        with pytest.raises(v16.MissingApiResponseError):
            offline.check()

    def test_retried_call_records_attempts_and_failures(self, tmp_path):
        cache = v16.ResponseCache(tmp_path / "responses.jsonl")
        transport = v16.CachingTransport(cache, FakeOpenRouter(fail_first=[503]))
        transport.run = "run1"
        encoder_through(transport).generate_query_embedding("q")
        entry = transport.history[-1]
        assert entry["attempts"] == 2 and entry["call_latency_s"] >= entry["latency_s"]
        reloaded = v16.ResponseCache(tmp_path / "responses.jsonl")
        assert len(reloaded) == 1 and [f["error"] for f in reloaded.failures] == ["HTTP 503"]

    def test_spend_limit_stops_the_run(self, tmp_path):
        transport = v16.CachingTransport(v16.ResponseCache(tmp_path / "r.jsonl"), FakeOpenRouter(),
                                         spend_limit_usd=1.5e-7)
        transport.run = "run1"
        encoder = encoder_through(transport)
        encoder.generate_query_embedding("a")
        with pytest.raises(v16.SpendLimitError):
            encoder.generate_query_embedding("b")

    def test_rerank_responses_are_stored_without_document_texts(self):
        stored = v16.compact_response("rerank", {"results": [{"index": 0, "relevance_score": 1.0,
                                                               "document": {"text": "long"}}]})
        assert stored["results"] == [{"index": 0, "relevance_score": 1.0}]
        embeddings = {"data": [{"index": 0, "embedding": "AAAA"}]}
        assert v16.compact_response("embeddings", embeddings) is embeddings

    def test_request_key_ignores_key_order_but_not_the_run(self):
        a = v16.request_key("run1", "embeddings", {"model": "m", "input": ["x"]})
        assert a == v16.request_key("run1", "embeddings", {"input": ["x"], "model": "m"})
        assert a != v16.request_key("run2", "embeddings", {"model": "m", "input": ["x"]})


class TestCostArithmetic:
    def test_storage_per_million_works(self):
        assert v16.storage_gb(1024) == {"float32_gb": pytest.approx(4.096), "halfvec_gb": pytest.approx(2.048)}

    def test_document_extrapolation(self):
        calls = [{"latency_s": 1.0, "cost_usd": 1e-4, "tokens": 5000}, {"latency_s": 0.77, "cost_usd": 7.7e-5,
                                                                         "tokens": 3850}]
        summary = v16.document_summary(calls, 177)
        assert summary["cost_usd"] == pytest.approx(1.77e-4)
        assert summary["extrapolated"]["200000"]["cost_usd"] == pytest.approx(0.2)
        assert summary["extrapolated"]["1000000"]["sequential_hours"] == pytest.approx(1.77 / 177 * 1e6 / 3600)
        assert summary["tokens_per_book"] == pytest.approx(50.0)

    def test_call_summary_counts_failures_and_retries_apart(self):
        calls = [{"ok": True, "latency_s": 0.1, "attempts": 1, "tokens": 10, "cost_usd": 1e-6, "provider": "A"},
                 {"ok": True, "latency_s": 0.3, "attempts": 2, "tokens": 20, "cost_usd": 3e-6, "provider": "A"},
                 {"ok": False}]
        summary = v16.call_summary(calls)
        assert (summary["calls"], summary["failed"], summary["retried"]) == (3, 1, 1)
        assert summary["median_ms"] == pytest.approx(200.0)
        assert summary["tokens_mean"] == 15 and summary["cost_mean_usd"] == pytest.approx(2e-6)
        assert summary["providers"] == ["A"]

    def test_max_cosine_difference(self):
        a = {"q1": np.array([1.0, 0.0]), "q2": np.array([0.0, 1.0])}
        b = {"q1": np.array([1.0, 0.0]), "q2": np.array([0.1, 1.0])}
        assert v16.max_cosine_difference(a, b) == pytest.approx(1 - 1 / np.sqrt(1.01))


def row(quality, latency, cost=1e-6, failed=0, reason=None) -> Dict[str, Any]:
    return {"quality": quality, "median_latency_s": latency, "cost_usd": cost, "failed_calls": failed,
            "excluded_reason": reason}


class TestSpeedQualityRule:
    def test_clear_quality_winner(self):
        decision = select_by_speed_quality({"a": row(0.70, 2.0), "b": row(0.60, 0.1)})
        assert decision["winner"] == "a" and decision["tied"] == ["a"]
        assert "highest quality" in decision["decided_by"]

    def test_tie_on_quality_goes_to_the_fastest(self):
        decision = select_by_speed_quality({"a": row(0.70, 2.0), "b": row(0.685, 0.5), "c": row(0.60, 0.1)})
        assert decision["winner"] == "b" and decision["tied"] == ["b", "a"]
        assert "lowest median latency" in decision["decided_by"]

    def test_within_ten_percent_the_cheaper_wins(self):
        decision = select_by_speed_quality({"a": row(0.70, 1.00, cost=3e-6), "b": row(0.69, 1.08, cost=1e-6)})
        assert decision["winner"] == "b" and decision["within_latency_band"] == ["a", "b"]
        assert "lower cost" in decision["decided_by"]

    def test_failed_calls_exclude(self):
        decision = select_by_speed_quality({"a": row(0.90, 0.1, failed=1), "b": row(0.50, 1.0)})
        assert decision["winner"] == "b" and "failed call" in decision["excluded"]["a"]

    def test_system_llm_needs_the_query_language(self):
        b = {"fast": {"parsed": 60, "n": 60, "parsed_share": 1.0, "citation_precision": 0.97,
                      "in_query_language_share": 0.60, "median_latency_s": 1.0, "cost_per_explanation_usd": 1e-4},
             "slow": {"parsed": 60, "n": 60, "parsed_share": 1.0, "citation_precision": 0.96,
                      "in_query_language_share": 1.0, "median_latency_s": 6.0, "cost_per_explanation_usd": 1e-4}}
        c = {m: {"parsed": 44, "n": 44, "parsed_share": 1.0} for m in b}
        decision = select_system_llm_speed_quality(b, c)
        assert decision["winner"] == "slow" and "query language" in decision["excluded"]["fast"]

    def test_judge_ignores_speed_and_takes_the_cheaper_in_a_tie(self):
        rows = {"x/a": {"complete": True, "kappa": 0.76, "cost_per_judgment_usd": 7e-5, "failures": 0},
                "y/b": {"complete": True, "kappa": 0.75, "cost_per_judgment_usd": 6e-5, "failures": 0},
                "z/c": {"complete": True, "kappa": 0.80, "cost_per_judgment_usd": 1e-5, "failures": 0}}
        decision = select_judge_speed_quality(rows, "w/system")
        assert decision["winner"] == "z/c"
        decision = select_judge_speed_quality(rows, "z/other")
        assert decision["winner"] == "y/b" and "same vendor" in decision["excluded"]["z/c"]


class TestFailedCalls:
    def test_a_failed_document_call_counts_once_and_is_replayed_offline(self, tmp_path):
        from app.domain.entities import Book
        from app.evaluation.types import TestQuery

        books = [Book.create_new(title=f"Libro {i}", authors=["A"], description=f"Texto {i}") for i in range(5)]
        queries = [TestQuery("q01", "novela", language="es"), TestQuery("q02", "poetry", language="en")]
        arm = v16.EmbeddingArm("vec_api_test", "baai/bge-m3", 1024, True)
        logs: List[str] = []

        live = v16.CachingTransport(v16.ResponseCache(tmp_path / "r.jsonl"), FakeOpenRouter(fail_first=[404]))
        http = OpenRouterHttp(api_key="k", transport=live, max_retries=0, sleep=lambda s: None)
        record = v16.measure_embedding_arm(arm, http, live, books, queries, logs.append)

        assert record["failed_calls"] == 1 and record["failures"][0]["item"] == "documents"
        assert "404" in record["failures"][0]["error"]
        assert set(record["runs"]["run1"]) == {"q01", "q02"}  # the second attempt gave the arm its runs
        assert record["stability"]["identical_vectors"] == 2

        offline = v16.CachingTransport(v16.ResponseCache(tmp_path / "r.jsonl"), None)
        replay_http = OpenRouterHttp(api_key="k", transport=offline, max_retries=0, sleep=lambda s: None)
        replay = v16.measure_embedding_arm(arm, replay_http, offline, books, queries, logs.append)
        assert replay["failed_calls"] == 1
        assert replay["runs"] == record["runs"]
        assert len(v16.ResponseCache(tmp_path / "r.jsonl").call_failures("vec_api_test")) == 1
