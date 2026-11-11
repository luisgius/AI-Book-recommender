"""
Tests for the CrossEncoderReranker adapter (Reranker port).

sentence_transformers.CrossEncoder is replaced by a stub, so no model is
downloaded or loaded.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

from app.domain.entities import Book, SearchResult
from app.infrastructure.search import cross_encoder_reranker as module
from app.infrastructure.search.cross_encoder_reranker import CrossEncoderReranker


class StubCrossEncoder:
    """Records how it is built and scores a pair by the book title it contains."""

    instances: List["StubCrossEncoder"] = []
    scores_by_title: Dict[str, float] = {}

    def __init__(self, model_name: str, device: Optional[str] = None, max_length: Optional[int] = None):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.predict_calls: List[Dict[str, Any]] = []
        StubCrossEncoder.instances.append(self)

    def predict(self, pairs, **kwargs):
        self.predict_calls.append({"pairs": list(pairs), **kwargs})
        scores = []
        for _query, document in pairs:
            title = next(t for t in self.scores_by_title if document.startswith(t))
            scores.append(self.scores_by_title[title])
        return np.array(scores, dtype=np.float32)


class FailingCrossEncoder:
    attempts = 0

    def __init__(self, *args, **kwargs):
        FailingCrossEncoder.attempts += 1
        raise OSError("model not found on the Hub")


@pytest.fixture
def stub_encoder(monkeypatch):
    StubCrossEncoder.instances = []
    StubCrossEncoder.scores_by_title = {}
    monkeypatch.setattr(module, "CrossEncoder", StubCrossEncoder)
    # conftest.py may switch the Hub client offline; these tests choose explicitly
    monkeypatch.setattr(module.hf_constants, "HF_HUB_OFFLINE", False)
    return StubCrossEncoder


def make_results(titles_and_scores: Dict[str, float]) -> List[SearchResult]:
    """Fused results in the given order; the dict values are the stub's scores."""
    results = []
    for i, title in enumerate(titles_and_scores, start=1):
        book = Book.create_new(title=title, authors=["Ann Author"], description=f"About {title}",
                               categories=["Fiction"])
        results.append(SearchResult(book=book, final_score=1.0 / (60 + i), rank=i, source="hybrid",
                                    lexical_score=float(10 - i), vector_score=0.5))
    return results


class TestCrossEncoderReranker:
    def test_model_is_loaded_lazily_and_once(self, stub_encoder):
        stub_encoder.scores_by_title = {"Alpha": 1.0}
        reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3", device="cpu", max_length=256)

        assert stub_encoder.instances == []
        assert reranker.is_ready() is True

        reranker.rerank("query", make_results({"Alpha": 1.0}))
        reranker.rerank("query", make_results({"Alpha": 1.0}))

        assert len(stub_encoder.instances) == 1
        encoder = stub_encoder.instances[0]
        assert (encoder.model_name, encoder.device, encoder.max_length) == (
            "BAAI/bge-reranker-v2-m3", "cpu", 256,
        )
        assert len(encoder.predict_calls) == 2

    def test_results_are_sorted_by_score_with_new_ranks(self, stub_encoder):
        scores = {"Alpha": -2.0, "Beta": 5.5, "Gamma": 0.25}
        stub_encoder.scores_by_title = scores
        fused = make_results(scores)

        reranked = CrossEncoderReranker("model").rerank("query", fused)

        assert [r.book.title for r in reranked] == ["Beta", "Gamma", "Alpha"]
        assert [r.rank for r in reranked] == [1, 2, 3]
        assert [r.rerank_score for r in reranked] == [5.5, 0.25, -2.0]

    def test_fusion_scores_are_kept_and_inputs_are_not_modified(self, stub_encoder):
        scores = {"Alpha": 0.0, "Beta": 1.0}
        stub_encoder.scores_by_title = scores
        fused = make_results(scores)

        reranked = CrossEncoderReranker("model").rerank("query", fused)

        beta_before, beta_after = fused[1], reranked[0]
        assert beta_after is not beta_before
        assert beta_after.book is beta_before.book
        assert (beta_after.final_score, beta_after.lexical_score, beta_after.vector_score) == (
            beta_before.final_score, beta_before.lexical_score, beta_before.vector_score,
        )
        assert beta_after.source == "hybrid"
        # The fused list is untouched
        assert [r.rank for r in fused] == [1, 2]
        assert all(r.rerank_score is None for r in fused)

    def test_ties_keep_the_fused_order(self, stub_encoder):
        scores = {"Alpha": 1.0, "Beta": 1.0, "Gamma": 1.0}
        stub_encoder.scores_by_title = scores

        reranked = CrossEncoderReranker("model").rerank("query", make_results(scores))

        assert [r.book.title for r in reranked] == ["Alpha", "Beta", "Gamma"]

    def test_pairs_use_the_query_and_the_searchable_text(self, stub_encoder):
        scores = {"Alpha": 1.0, "Beta": 0.0}
        stub_encoder.scores_by_title = scores
        fused = make_results(scores)

        CrossEncoderReranker("model", batch_size=8).rerank("novelas de misterio", fused)

        call = stub_encoder.instances[0].predict_calls[0]
        assert call["pairs"] == [("novelas de misterio", r.book.get_searchable_text()) for r in fused]
        assert call["batch_size"] == 8
        assert call["show_progress_bar"] is False
        # Raw score (no sigmoid), whatever the model's config says
        assert isinstance(call["activation_fct"], torch.nn.Identity)

    def test_empty_input_does_not_load_the_model(self, stub_encoder):
        assert CrossEncoderReranker("model").rerank("query", []) == []
        assert stub_encoder.instances == []

    def test_load_failure_is_reported_and_not_retried(self, monkeypatch):
        FailingCrossEncoder.attempts = 0
        monkeypatch.setattr(module, "CrossEncoder", FailingCrossEncoder)
        monkeypatch.setattr(module.hf_constants, "HF_HUB_OFFLINE", False)
        reranker = CrossEncoderReranker("no/such-model")
        results = make_results({"Alpha": 1.0})

        with pytest.raises(RuntimeError, match="no/such-model"):
            reranker.rerank("query", results)
        assert reranker.is_ready() is False

        with pytest.raises(RuntimeError, match="failed to load"):
            reranker.rerank("query", results)
        assert FailingCrossEncoder.attempts == 1

    def test_offline_loads_the_cached_snapshot_folder(self, stub_encoder, monkeypatch):
        requested = []

        def fake_snapshot_download(repo_id, local_files_only=False):
            requested.append((repo_id, local_files_only))
            return "/hf-cache/models--BAAI--bge-reranker-v2-m3/snapshots/abc"

        monkeypatch.setattr(module.hf_constants, "HF_HUB_OFFLINE", True)
        monkeypatch.setattr(module, "snapshot_download", fake_snapshot_download)
        stub_encoder.scores_by_title = {"Alpha": 1.0}
        reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")

        reranker.rerank("query", make_results({"Alpha": 1.0}))

        assert requested == [("BAAI/bge-reranker-v2-m3", True)]
        assert stub_encoder.instances[0].model_name.endswith("snapshots/abc")
        assert reranker.get_model_name() == "BAAI/bge-reranker-v2-m3"

    def test_offline_without_cache_passes_the_model_id(self, stub_encoder, monkeypatch):
        def missing(repo_id, local_files_only=False):
            raise FileNotFoundError("not in the local cache")

        monkeypatch.setattr(module.hf_constants, "HF_HUB_OFFLINE", True)
        monkeypatch.setattr(module, "snapshot_download", missing)
        stub_encoder.scores_by_title = {"Alpha": 1.0}

        CrossEncoderReranker("some/model").rerank("query", make_results({"Alpha": 1.0}))

        assert stub_encoder.instances[0].model_name == "some/model"

    def test_online_passes_the_model_id_without_checking_the_cache(self, stub_encoder, monkeypatch):
        def should_not_run(*args, **kwargs):
            raise AssertionError("the cache is only resolved offline")

        monkeypatch.setattr(module, "snapshot_download", should_not_run)
        stub_encoder.scores_by_title = {"Alpha": 1.0}

        CrossEncoderReranker("some/model").rerank("query", make_results({"Alpha": 1.0}))

        assert stub_encoder.instances[0].model_name == "some/model"

    def test_get_model_name(self):
        assert CrossEncoderReranker("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1").get_model_name() == (
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"model_name": " "}, "model_name"),
            ({"model_name": "m", "batch_size": 0}, "batch_size"),
            ({"model_name": "m", "max_length": 0}, "max_length"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            CrossEncoderReranker(**kwargs)
