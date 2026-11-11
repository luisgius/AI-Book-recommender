"""
Tests for the reranker configuration: RerankerSettings parsing and wiring
into the API dependencies.

No model is loaded: the reranker class is replaced by a recorder.
"""

from typing import Any, Dict, List, Optional

import pytest

import app.api.v1.dependencies as dependencies
from app.domain.services import DEFAULT_RERANK_DEPTH as DOMAIN_DEFAULT_DEPTH
from app.infrastructure.config.settings import DEFAULT_RERANK_DEPTH, RerankerSettings


# =============================================================================
# RerankerSettings
# =============================================================================


class TestRerankerSettings:
    def test_disabled_by_default(self):
        settings = RerankerSettings.from_env({})
        assert settings.enabled is False
        assert settings.model is None
        assert settings.depth == DEFAULT_RERANK_DEPTH == 30
        assert settings.device is None

    def test_default_depth_matches_the_domain_default(self):
        assert DEFAULT_RERANK_DEPTH == DOMAIN_DEFAULT_DEPTH

    def test_reads_model_depth_and_device(self):
        settings = RerankerSettings.from_env({
            "RERANKER_MODEL": " BAAI/bge-reranker-v2-m3 ",
            "RERANK_DEPTH": "50",
            "RERANKER_DEVICE": "MPS",
        })
        assert settings.enabled is True
        assert settings.model == "BAAI/bge-reranker-v2-m3"
        assert settings.depth == 50
        assert settings.device == "mps"

    @pytest.mark.parametrize("value", ["", "   ", "none", "None"])
    def test_blank_or_none_model_disables_reranking(self, value):
        assert RerankerSettings.from_env({"RERANKER_MODEL": value}).enabled is False

    def test_blank_depth_and_device_fall_back_to_defaults(self):
        settings = RerankerSettings.from_env({"RERANK_DEPTH": " ", "RERANKER_DEVICE": ""})
        assert settings.depth == DEFAULT_RERANK_DEPTH
        assert settings.device is None

    @pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:1"])
    def test_accepts_known_devices(self, device):
        assert RerankerSettings.from_env({"RERANKER_DEVICE": device}).device == device

    def test_rejects_unknown_device(self):
        with pytest.raises(ValueError, match="RERANKER_DEVICE"):
            RerankerSettings.from_env({"RERANKER_DEVICE": "gpu"})

    @pytest.mark.parametrize("depth", ["0", "-5"])
    def test_rejects_non_positive_depth(self, depth):
        with pytest.raises(ValueError, match="RERANK_DEPTH"):
            RerankerSettings.from_env({"RERANK_DEPTH": depth})

    def test_rejects_non_integer_depth(self):
        with pytest.raises(ValueError):
            RerankerSettings.from_env({"RERANK_DEPTH": "thirty"})

    def test_rejects_blank_model_given_directly(self):
        with pytest.raises(ValueError, match="RERANKER_MODEL"):
            RerankerSettings(model=" ")


# =============================================================================
# Wiring into the API dependencies
# =============================================================================


class RecordingReranker:
    """Replaces CrossEncoderReranker and records how it was built."""

    created: List[Dict[str, Any]] = []

    def __init__(self, model_name: str, device: Optional[str] = None):
        RecordingReranker.created.append({"model_name": model_name, "device": device})
        self._model_name = model_name

    def rerank(self, query_text, results):
        return results

    def get_model_name(self) -> str:
        return self._model_name

    def is_ready(self) -> bool:
        return True


class Ready:
    """Minimal repository / store stand-in for building the service."""

    def is_ready(self) -> bool:
        return True


@pytest.fixture
def clean_dependencies(monkeypatch):
    dependencies.reset_dependencies()
    RecordingReranker.created = []
    monkeypatch.setattr(dependencies, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(dependencies, "CrossEncoderReranker", RecordingReranker)
    yield
    dependencies.reset_dependencies()


def test_reranker_is_disabled_without_a_model(monkeypatch, clean_dependencies):
    monkeypatch.delenv("RERANKER_MODEL", raising=False)

    assert dependencies.get_reranker() is None
    assert RecordingReranker.created == []


def test_reranker_is_built_from_settings_once(monkeypatch, clean_dependencies):
    monkeypatch.setenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    monkeypatch.setenv("RERANKER_DEVICE", "cpu")

    reranker = dependencies.get_reranker()

    assert isinstance(reranker, RecordingReranker)
    assert RecordingReranker.created == [
        {"model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "device": "cpu"}
    ]
    assert dependencies.get_reranker() is reranker


def test_search_service_gets_the_reranker_and_depth(monkeypatch, clean_dependencies):
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setenv("RERANK_DEPTH", "40")
    for getter in ("get_bm25_repository", "get_vector_repository", "get_embeddings_store"):
        monkeypatch.setattr(dependencies, getter, lambda: Ready())
    monkeypatch.setattr(dependencies, "get_llm_client", lambda: None)

    service = dependencies.get_search_service()

    assert service.has_reranker() is True
    assert service.get_reranker_model_name() == "BAAI/bge-reranker-v2-m3"
    assert service._retrieval_size(10) == 40


def test_search_service_without_reranker_by_default(monkeypatch, clean_dependencies):
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    monkeypatch.delenv("RERANK_DEPTH", raising=False)
    for getter in ("get_bm25_repository", "get_vector_repository", "get_embeddings_store"):
        monkeypatch.setattr(dependencies, getter, lambda: Ready())
    monkeypatch.setattr(dependencies, "get_llm_client", lambda: None)

    service = dependencies.get_search_service()

    assert service.has_reranker() is False
    assert service._retrieval_size(10) == 10
