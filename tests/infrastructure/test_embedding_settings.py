"""
Tests for the embedding configuration: EmbeddingSettings parsing, wiring into
the API dependencies, and the output folder of the index build script.

No embedding model is loaded: the store class is replaced by a recorder.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import app.api.v1.dependencies as dependencies
from app.infrastructure.config.settings import DEFAULT_EMBEDDING_MODEL, EmbeddingSettings


# =============================================================================
# EmbeddingSettings
# =============================================================================


class TestEmbeddingSettings:
    def test_defaults_without_environment(self):
        settings = EmbeddingSettings.from_env({})
        assert settings.model == DEFAULT_EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert settings.device is None

    def test_reads_model_and_device(self):
        settings = EmbeddingSettings.from_env(
            {"EMBEDDING_MODEL": " intfloat/multilingual-e5-large-instruct ", "EMBEDDING_DEVICE": "MPS"}
        )
        assert settings.model == "intfloat/multilingual-e5-large-instruct"
        assert settings.device == "mps"

    def test_blank_values_fall_back_to_defaults(self):
        settings = EmbeddingSettings.from_env({"EMBEDDING_MODEL": "  ", "EMBEDDING_DEVICE": ""})
        assert settings.model == DEFAULT_EMBEDDING_MODEL
        assert settings.device is None

    @pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:1"])
    def test_accepts_known_devices(self, device):
        assert EmbeddingSettings.from_env({"EMBEDDING_DEVICE": device}).device == device

    def test_rejects_unknown_device(self):
        with pytest.raises(ValueError, match="EMBEDDING_DEVICE"):
            EmbeddingSettings.from_env({"EMBEDDING_DEVICE": "gpu"})

    def test_rejects_empty_model(self):
        with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
            EmbeddingSettings(model="")


# =============================================================================
# Wiring into the API dependencies
# =============================================================================


class RecordingStore:
    """Replaces EmbeddingsStoreFaiss and records how it was built."""

    created: List[Dict[str, Any]] = []

    def __init__(self, model_name: str, device: Optional[str] = None):
        RecordingStore.created.append({"model_name": model_name, "device": device})
        self.loaded_from: Optional[str] = None

    def load_index(self, path: str) -> None:
        self.loaded_from = path


@pytest.fixture
def clean_dependencies():
    dependencies.reset_dependencies()
    RecordingStore.created = []
    yield
    dependencies.reset_dependencies()


def test_embeddings_store_is_built_from_settings(monkeypatch, tmp_path: Path, clean_dependencies):
    monkeypatch.setenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr("app.infrastructure.search.embeddings_store_faiss.EmbeddingsStoreFaiss", RecordingStore)
    monkeypatch.setattr(dependencies, "INDEXES_DIR", tmp_path)
    (tmp_path / "faiss_index").mkdir()

    store = dependencies.get_embeddings_store()

    assert isinstance(store, RecordingStore)
    assert RecordingStore.created == [
        {"model_name": "intfloat/multilingual-e5-large-instruct", "device": "cpu"}
    ]
    assert store.loaded_from == str(tmp_path / "faiss_index")
    # Singleton: a second call reuses the store.
    assert dependencies.get_embeddings_store() is store


def test_default_wiring_keeps_minilm(monkeypatch, tmp_path: Path, clean_dependencies):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.setattr(dependencies, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.infrastructure.search.embeddings_store_faiss.EmbeddingsStoreFaiss", RecordingStore)
    monkeypatch.setattr(dependencies, "INDEXES_DIR", tmp_path)

    dependencies.get_embeddings_store()

    assert RecordingStore.created == [{"model_name": "all-MiniLM-L6-v2", "device": None}]


# =============================================================================
# Index build script
# =============================================================================


class TestIngestScriptOutputDir:
    def test_default_model_writes_to_the_shipped_folder(self):
        from scripts.ingest_and_build import default_out_dir

        assert default_out_dir("all-MiniLM-L6-v2") == "data/indexes"

    def test_other_models_get_their_own_ablation_folder(self):
        from scripts.ingest_and_build import default_out_dir

        assert default_out_dir("intfloat/multilingual-e5-large-instruct") == str(
            Path("data/indexes_ablation") / "intfloat_multilingual-e5-large-instruct"
        )
        assert default_out_dir("Qwen/Qwen3-Embedding-0.6B") == str(
            Path("data/indexes_ablation") / "qwen_qwen3-embedding-0.6b"
        )
