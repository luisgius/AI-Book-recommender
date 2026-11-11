"""Pure parts of the V13 live check of the cloud profile (no API, no judge)."""

import json
from pathlib import Path

import httpx
import pytest

from app.evaluation import cloud_live_check
from app.evaluation.cloud_live_check import _api_totals, run_queries

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_api_totals_read_the_usage_rows_and_tolerate_missing_operations():
    usage = {"api": {"embedding": {"total_latency_s": 0.5, "calls": 2, "reported_cost_usd": 1e-7},
                     "total": {"calls": 2}}}
    assert _api_totals(usage) == {"embedding_s": 0.5, "embedding_calls": 2.0, "embedding_cost": 1e-7,
                                  "rerank_s": 0.0, "rerank_calls": 0.0, "rerank_cost": 0.0}


def test_run_queries_splits_each_search_with_the_usage_endpoint():
    """A fake API: each search makes one embedding call (0.2 s) and one rerank call (0.3 s)."""
    state = {"searches": 0}

    def usage():
        n = state["searches"]
        return {"api": {"embedding": {"total_latency_s": 0.2 * n, "calls": n, "reported_cost_usd": 1e-7 * n},
                        "rerank": {"total_latency_s": 0.3 * n, "calls": n, "reported_cost_usd": 1e-3 * n}}}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/usage":
            return httpx.Response(200, json=usage())
        state["searches"] += 1
        return httpx.Response(200, json={"latency_ms": 650.0, "degraded": False,
                                         "results": [{"book": {"id": "b1"}}, {"book": {"id": "b2"}}]})

    with httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)) as client:
        out = run_queries(client, {"q1": "a", "q2": "b"}, "warm", log=lambda m: None)

    q1 = out["per_query"]["q1"]
    assert q1["ids"] == ["b1", "b2"] and q1["embedding_calls"] == 1 and q1["rerank_calls"] == 1
    assert q1["embedding_ms"] == pytest.approx(200) and q1["rerank_ms"] == pytest.approx(300)
    assert q1["database_ms"] == pytest.approx(150) and q1["cost_usd"] == pytest.approx(1e-3 + 1e-7)
    assert state["searches"] == 3  # warm-up + two queries


def test_published_live_check_is_complete():
    path = REPO_ROOT / cloud_live_check.DEFAULT_OUTPUT_JSON
    if not path.is_file():
        pytest.skip("results_cloud_live_check.json not generated yet")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["runs"][cloud_live_check.CLOUD]) == 44
    assert saved["analysis"]["degraded_searches"] == 0
    assert saved["analysis"]["systems"][cloud_live_check.CLOUD]["list_lengths"]["judged_share_of_returned"] == 1.0
    endpoints = saved["endpoints"]
    assert endpoints["health"]["status"] == "ok" and endpoints["ui"] == endpoints["book"] == 200
    assert endpoints["mcp_initialize"] == 200 and endpoints["similar_mmr"]["status"] == 200
