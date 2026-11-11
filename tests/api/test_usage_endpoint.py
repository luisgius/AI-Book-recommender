"""GET /api/v1/usage: off by default, a read-only summary of the API calls when enabled (V13)."""

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import dependencies as deps
from app.infrastructure.config.settings import UsageEndpointSettings
from app.main import app


@pytest.fixture
def client(monkeypatch):
    deps.reset_dependencies()
    monkeypatch.setattr(deps, "load_dotenv", lambda *args, **kwargs: None)
    yield TestClient(app)
    deps.reset_dependencies()


def test_usage_endpoint_is_off_by_default(client, monkeypatch):
    monkeypatch.delenv("USAGE_ENDPOINT_ENABLED", raising=False)
    assert client.get("/api/v1/usage").status_code == 404


def test_usage_endpoint_summarizes_the_recorded_calls_by_operation(client, monkeypatch):
    monkeypatch.setenv("USAGE_ENDPOINT_ENABLED", "true")
    recorder = deps.get_usage_recorder()
    recorder.record_call("embedding", "qwen/qwen3-embedding-8b", 0.4, 20, 0, cost_usd=2e-7)
    recorder.record_call("rerank", "qwen/qwen3-reranker-8b", 0.6, 7000, 0, cost_usd=1.4e-3)
    recorder.record_call("rerank", "qwen/qwen3-reranker-8b", 0.5, None, 0, error="OpenRouterError")

    body = client.get("/api/v1/usage").json()

    assert body["since"] is not None and body["llm"] is None
    assert body["api"]["embedding"]["calls"] == 1 and body["api"]["embedding"]["input_tokens"] == 20
    assert body["api"]["rerank"]["calls"] == 2 and body["api"]["rerank"]["failed_calls"] == 1
    assert body["api"]["total"]["reported_cost_usd"] == pytest.approx(2e-7 + 1.4e-3)
    assert body["api"]["rerank"]["median_latency_s"] == pytest.approx(0.55)


@pytest.mark.parametrize("raw, enabled", [("", False), ("false", False), ("true", True), ("1", True)])
def test_usage_setting(raw, enabled):
    assert UsageEndpointSettings.from_env({"USAGE_ENDPOINT_ENABLED": raw}).enabled is enabled
    with pytest.raises(ValueError):
        UsageEndpointSettings.from_env({"USAGE_ENDPOINT_ENABLED": "maybe"})


def test_api_encoder_and_reranker_share_the_usage_recorder(monkeypatch):
    deps.reset_dependencies()
    monkeypatch.setattr(deps, "load_dotenv", lambda *args, **kwargs: None)
    for name in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "EMBEDDING_OPENROUTER_PROVIDER",
                 "RERANKER_PROVIDER", "RERANKER_MODEL", "RERANKER_OPENROUTER_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SEARCH_PROFILE", "cloud")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    try:
        encoder = deps.build_query_encoder(deps.get_embedding_settings(), usage_recorder=deps.get_usage_recorder())
        reranker = deps.get_reranker()
        assert encoder._recorder is deps.get_usage_recorder()
        assert reranker._recorder is deps.get_usage_recorder()
    finally:
        deps.reset_dependencies()
