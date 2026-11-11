"""
Defaults of the cloud profile (V13): the API models chosen in
data/evaluation/API_MODELS.md, each overridable; the local defaults do not change.
"""

import pytest

from app.infrastructure.config.settings import (
    CLOUD_EMBEDDING_MODEL,
    CLOUD_RERANKER_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingSettings,
    RerankerSettings,
)

CLOUD = {"SEARCH_PROFILE": "cloud"}


def test_cloud_embedding_defaults_are_the_measured_choice():
    s = EmbeddingSettings.from_env(CLOUD)
    assert (s.provider, s.model, s.dimensions, s.openrouter_provider) == (
        "openrouter", "qwen/qwen3-embedding-8b", 1024, "deepinfra")
    assert CLOUD_EMBEDDING_MODEL == "qwen/qwen3-embedding-8b"


def test_cloud_reranker_defaults_are_the_measured_choice():
    s = RerankerSettings.from_env(CLOUD)
    assert (s.provider, s.model, s.openrouter_provider, s.depth) == (
        "openrouter", "qwen/qwen3-reranker-8b", "primeintellect", 30)
    assert s.enabled and CLOUD_RERANKER_MODEL == "qwen/qwen3-reranker-8b"


@pytest.mark.parametrize("variable, value, field, expected", [
    ("EMBEDDING_DIMENSIONS", "512", "dimensions", 512),
    ("EMBEDDING_OPENROUTER_PROVIDER", "nebius", "openrouter_provider", "nebius"),
    ("EMBEDDING_MODEL", "baai/bge-m3", "model", "baai/bge-m3"),
])
def test_each_cloud_embedding_default_is_overridable(variable, value, field, expected):
    assert getattr(EmbeddingSettings.from_env({**CLOUD, variable: value}), field) == expected


def test_another_cloud_embedding_model_does_not_inherit_the_dimension_or_provider_pin():
    s = EmbeddingSettings.from_env({**CLOUD, "EMBEDDING_MODEL": "baai/bge-m3"})
    assert s.dimensions is None and s.openrouter_provider is None


def test_cloud_profile_with_a_local_embedding_model_keeps_the_local_default():
    s = EmbeddingSettings.from_env({**CLOUD, "EMBEDDING_PROVIDER": "local"})
    assert (s.provider, s.model, s.dimensions) == ("local", DEFAULT_EMBEDDING_MODEL, None)


@pytest.mark.parametrize("value", ["", "none", "NONE"])
def test_cloud_reranker_can_be_switched_off(value):
    assert RerankerSettings.from_env({**CLOUD, "RERANKER_MODEL": value}).enabled is False


def test_cloud_reranker_model_and_pin_are_overridable():
    s = RerankerSettings.from_env({**CLOUD, "RERANKER_MODEL": "qwen/qwen3-reranker-0.6b"})
    assert s.model == "qwen/qwen3-reranker-0.6b" and s.openrouter_provider is None
    s = RerankerSettings.from_env({**CLOUD, "RERANKER_OPENROUTER_PROVIDER": "fireworks"})
    assert s.openrouter_provider == "fireworks"
    s = RerankerSettings.from_env({**CLOUD, "RERANKER_PROVIDER": "local"})
    assert (s.provider, s.model) == ("local", None)


@pytest.mark.parametrize("env", [{}, {"SEARCH_PROFILE": "local"}])
def test_local_defaults_do_not_change(env):
    e = EmbeddingSettings.from_env(env)
    assert (e.provider, e.model, e.dimensions, e.openrouter_provider) == ("local", DEFAULT_EMBEDDING_MODEL, None, None)
    r = RerankerSettings.from_env(env)
    assert (r.provider, r.model, r.openrouter_provider, r.enabled) == ("local", None, None, False)
    with pytest.raises(ValueError):  # an API model must still be named outside the cloud profile
        EmbeddingSettings.from_env({**env, "EMBEDDING_PROVIDER": "openrouter"})
