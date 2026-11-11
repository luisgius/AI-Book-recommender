"""
Tests for ApiReranker (OpenRouter's rerank API) and the settings and wiring
of both API adapters. No network: a fake transport answers every request.

Covered:
- the Reranker contract: same results back, sorted by score, ties kept in
  the fused order, rank and rerank_score set, fusion scores kept, input untouched
- one request per search, document text = searchable text cut to its limit
- failures raise, and SearchService then keeps the fused order
- is_ready after configuration errors
- EMBEDDING_PROVIDER / RERANKER_PROVIDER settings and the API dependencies
"""

import json
from dataclasses import replace
from typing import Any, Dict, List, Optional

import httpx
import pytest

import app.api.v1.dependencies as dependencies
from app.domain.entities import Book, SearchResult
from app.domain.value_objects import SearchQuery
from app.infrastructure.config.settings import EmbeddingSettings, RerankerSettings
from app.infrastructure.llm.usage import RERANK_OPERATION, LLMUsageRecorder
from app.infrastructure.search.api_reranker import DEFAULT_MAX_DOCUMENT_CHARS, ApiReranker, read_scores
from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder
from app.infrastructure.search.openrouter_http import OpenRouterError, OpenRouterHttp
from tests.domain.test_reranking import build_service, make_books, titles


# =============================================================================
# Fake rerank endpoint
# =============================================================================


class FakeRerank:
    """
    Answers like OpenRouter's rerank endpoint, best first.

    score_of(document_text) gives each document's score (default: the number
    in "Book NN", so later books score higher). `script` items are HTTP
    statuses or callables(request) used before the normal answer.
    """

    def __init__(self, score_of=None, script: Optional[List[Any]] = None) -> None:
        self.score_of = score_of or (lambda text: float(text.split()[1]))
        self.script = list(script or [])
        self.requests: List[httpx.Request] = []

    @property
    def bodies(self) -> List[Dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, int):
                return httpx.Response(step, json={"error": {"code": step, "message": "boom"}})
            return step(request)
        body = json.loads(request.content)
        scored = [(i, self.score_of(text)) for i, text in enumerate(body["documents"])]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return httpx.Response(200, json={
            "id": "gen-rerank-1", "model": body["model"], "provider": "FakeProvider",
            "results": [{"index": i, "relevance_score": s, "document": {"text": body["documents"][i]}}
                        for i, s in scored],
            "usage": {"total_tokens": 10 * len(body["documents"]), "cost": 2e-6 * len(body["documents"])},
        })


def make_reranker(fake: Optional[FakeRerank] = None, *, recorder=None, max_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
                  upstream: Optional[str] = None, max_retries: int = 2) -> ApiReranker:
    http = OpenRouterHttp(api_key="test-key", transport=httpx.MockTransport(fake or FakeRerank()),
                          max_retries=max_retries, sleep=lambda seconds: None)
    return ApiReranker("qwen/qwen3-reranker-8b", upstream_provider=upstream, max_document_chars=max_chars,
                       usage_recorder=recorder, http=http)


def fused(books: List[Book]) -> List[SearchResult]:
    return [SearchResult(book=b, final_score=1.0 / (60 + i), rank=i, source="hybrid",
                         lexical_score=0.5, vector_score=0.25)
            for i, b in enumerate(books, start=1)]


# =============================================================================
# Contract
# =============================================================================


class TestContract:
    def test_reorders_by_score_and_keeps_fusion_scores(self):
        results = fused(make_books(5))
        snapshot = [replace(r) for r in results]

        reranked = make_reranker().rerank("query", results)

        assert titles(reranked) == ["Book 04", "Book 03", "Book 02", "Book 01", "Book 00"]
        assert [r.rank for r in reranked] == [1, 2, 3, 4, 5]
        assert [r.rerank_score for r in reranked] == [4.0, 3.0, 2.0, 1.0, 0.0]
        assert reranked[0].final_score == results[4].final_score
        assert reranked[0].lexical_score == 0.5 and reranked[0].vector_score == 0.25
        assert results == snapshot  # the input list is not modified
        assert {r.book.id for r in reranked} == {r.book.id for r in results}

    def test_equal_scores_keep_the_fused_order(self):
        reranked = make_reranker(FakeRerank(score_of=lambda text: 1.0)).rerank("q", fused(make_books(4)))
        assert titles(reranked) == ["Book 00", "Book 01", "Book 02", "Book 03"]

    def test_empty_input_makes_no_request(self):
        fake = FakeRerank()
        assert make_reranker(fake).rerank("q", []) == []
        assert fake.requests == []

    def test_one_request_with_query_and_cut_texts(self):
        fake = FakeRerank(score_of=lambda text: 0.0)
        books = make_books(3)
        books[1] = replace(books[1], description="x" * 5000)
        make_reranker(fake, max_chars=100, upstream="PrimeIntellect").rerank("novela negra", fused(books))

        assert len(fake.requests) == 1
        assert str(fake.requests[0].url) == "https://openrouter.ai/api/v1/rerank"
        body = fake.bodies[0]
        assert body["model"] == "qwen/qwen3-reranker-8b" and body["query"] == "novela negra"
        assert body["documents"] == [b.get_searchable_text()[:100] for b in books]
        assert len(body["documents"][1]) == 100
        assert body["provider"] == {"order": ["primeintellect"], "allow_fallbacks": False}

    def test_default_cut_is_2000_characters(self):
        assert DEFAULT_MAX_DOCUMENT_CHARS == 2000
        book = replace(make_books(1)[0], description="y" * 3000)
        assert len(make_reranker().document_text(book)) == 2000

    def test_usage_is_recorded_as_a_rerank_call(self):
        recorder = LLMUsageRecorder()
        make_reranker(recorder=recorder).rerank("q", fused(make_books(3)))
        (record,) = recorder.records
        assert record.operation == RERANK_OPERATION == "rerank"
        assert record.input_tokens == 30 and record.output_tokens == 0
        assert record.cost_usd == pytest.approx(6e-6)
        assert record.model == "qwen/qwen3-reranker-8b"


# =============================================================================
# Failures
# =============================================================================


class TestFailures:
    def test_transient_errors_are_retried(self):
        fake = FakeRerank(script=[503, 429])
        reranked = make_reranker(fake).rerank("q", fused(make_books(3)))
        assert len(fake.requests) == 3 and titles(reranked)[0] == "Book 02"

    def test_failure_after_retries_raises(self):
        with pytest.raises(OpenRouterError, match="HTTP 503"):
            make_reranker(FakeRerank(script=[503, 503, 503])).rerank("q", fused(make_books(3)))

    @pytest.mark.parametrize("results", [
        [{"index": 0, "relevance_score": 1.0}],  # a document without a score
        [{"index": 0, "relevance_score": 1.0}, {"index": 0, "relevance_score": 0.5}],  # repeated
        [{"index": 0, "relevance_score": 1.0}, {"index": 7, "relevance_score": 0.5}],  # unknown index
    ])
    def test_incomplete_response_raises(self, results):
        with pytest.raises(ValueError):
            read_scores({"results": results}, 2)

    def test_search_service_keeps_the_fused_order_when_the_api_fails(self, caplog):
        fused_service, _, _ = build_service()
        expected = titles(fused_service.search(SearchQuery(text="anything", max_results=10)))

        reranker = make_reranker(FakeRerank(script=[500, 500, 500]))
        service, _, _ = build_service(reranker=reranker)
        with caplog.at_level("WARNING"):
            results = service.search(SearchQuery(text="anything", max_results=10))

        assert titles(results) == expected
        assert all(r.rerank_score is None for r in results)
        assert "keeping the fused order" in caplog.text

    def test_search_service_uses_the_api_scores_when_it_works(self):
        service, _, _ = build_service(reranker=make_reranker(), rerank_depth=5)
        results = service.search(SearchQuery(text="anything", max_results=5))
        assert titles(results) == ["Book 04", "Book 03", "Book 02", "Book 01", "Book 00"]
        assert results[0].rerank_score == 4.0

    def test_is_ready_after_a_configuration_error_until_a_success(self):
        fake = FakeRerank(script=[401])
        reranker = make_reranker(fake)
        assert reranker.is_ready()
        with pytest.raises(OpenRouterError):
            reranker.rerank("q", fused(make_books(2)))
        assert reranker.is_ready() is False
        reranker.rerank("q", fused(make_books(2)))
        assert reranker.is_ready() is True

    def test_a_transient_failure_does_not_make_it_unready(self):
        reranker = make_reranker(FakeRerank(script=[503, 503, 503]))
        with pytest.raises(OpenRouterError):
            reranker.rerank("q", fused(make_books(2)))
        assert reranker.is_ready() is True


# =============================================================================
# Settings
# =============================================================================


class TestSettings:
    def test_local_is_the_default_for_both(self):
        embedding = EmbeddingSettings.from_env({})
        reranker = RerankerSettings.from_env({})
        assert (embedding.provider, embedding.model, embedding.dimensions, embedding.openrouter_provider) == (
            "local", "all-MiniLM-L6-v2", None, None)
        assert (reranker.provider, reranker.model, reranker.openrouter_provider) == ("local", None, None)

    def test_openrouter_embedding_settings(self):
        settings = EmbeddingSettings.from_env({
            "EMBEDDING_PROVIDER": " OpenRouter ", "EMBEDDING_MODEL": "qwen/qwen3-embedding-8b",
            "EMBEDDING_DIMENSIONS": "1024", "EMBEDDING_OPENROUTER_PROVIDER": "DeepInfra",
        })
        assert (settings.provider, settings.model, settings.dimensions, settings.openrouter_provider) == (
            "openrouter", "qwen/qwen3-embedding-8b", 1024, "deepinfra")

    def test_openrouter_embedding_needs_a_model(self):
        with pytest.raises(ValueError, match="OpenRouter embedding model"):
            EmbeddingSettings.from_env({"EMBEDDING_PROVIDER": "openrouter"})

    def test_unknown_provider_is_refused(self):
        with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
            EmbeddingSettings.from_env({"EMBEDDING_PROVIDER": "cohere"})
        with pytest.raises(ValueError, match="RERANKER_PROVIDER"):
            RerankerSettings.from_env({"RERANKER_PROVIDER": "voyage"})

    def test_api_options_are_refused_for_the_local_provider(self):
        with pytest.raises(ValueError, match="only to EMBEDDING_PROVIDER=openrouter"):
            EmbeddingSettings.from_env({"EMBEDDING_DIMENSIONS": "512"})
        with pytest.raises(ValueError, match="only to RERANKER_PROVIDER=openrouter"):
            RerankerSettings.from_env({"RERANKER_OPENROUTER_PROVIDER": "fireworks"})

    @pytest.mark.parametrize("value", ["abc", "0", "-4"])
    def test_bad_dimensions_are_refused(self, value):
        with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
            EmbeddingSettings.from_env({"EMBEDDING_PROVIDER": "openrouter", "EMBEDDING_MODEL": "baai/bge-m3",
                                        "EMBEDDING_DIMENSIONS": value})

    def test_openrouter_reranker_settings(self):
        settings = RerankerSettings.from_env({
            "RERANKER_PROVIDER": "openrouter", "RERANKER_MODEL": "qwen/qwen3-reranker-0.6b",
            "RERANKER_OPENROUTER_PROVIDER": "PrimeIntellect", "RERANK_DEPTH": "20",
        })
        assert (settings.provider, settings.model, settings.depth, settings.openrouter_provider, settings.enabled) == (
            "openrouter", "qwen/qwen3-reranker-0.6b", 20, "primeintellect", True)


# =============================================================================
# Wiring
# =============================================================================


@pytest.fixture
def clean_dependencies(monkeypatch):
    dependencies.reset_dependencies()
    monkeypatch.setattr(dependencies, "load_dotenv", lambda *args, **kwargs: None)
    for name in ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "EMBEDDING_OPENROUTER_PROVIDER",
                 "EMBEDDING_DEVICE", "RERANKER_PROVIDER", "RERANKER_MODEL", "RERANKER_OPENROUTER_PROVIDER",
                 "RERANK_DEPTH", "SEARCH_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    yield
    dependencies.reset_dependencies()


def test_openrouter_reranker_is_wired(monkeypatch, clean_dependencies):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("RERANKER_PROVIDER", "openrouter")
    monkeypatch.setenv("RERANKER_MODEL", "qwen/qwen3-reranker-8b")
    monkeypatch.setenv("RERANKER_OPENROUTER_PROVIDER", "primeintellect")

    reranker = dependencies.get_reranker()

    assert isinstance(reranker, ApiReranker)
    assert reranker.get_model_name() == "qwen/qwen3-reranker-8b"
    assert reranker.upstream_provider == "primeintellect"
    assert dependencies.get_reranker() is reranker


def test_openrouter_reranker_without_key_disables_reranking(monkeypatch, clean_dependencies):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("RERANKER_PROVIDER", "openrouter")
    monkeypatch.setenv("RERANKER_MODEL", "qwen/qwen3-reranker-8b")
    assert dependencies.get_reranker() is None


def test_openrouter_embeddings_are_wired_into_the_faiss_store(monkeypatch, clean_dependencies, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    monkeypatch.setenv("EMBEDDING_OPENROUTER_PROVIDER", "deepinfra")
    monkeypatch.setattr(dependencies, "INDEXES_DIR", tmp_path)  # no index there

    store = dependencies.get_embeddings_store()

    assert isinstance(store, EmbeddingsStoreFaiss)
    assert isinstance(store.get_encoder(), OpenRouterEmbeddingEncoder)
    assert store.get_dimension() == 512
    assert store.get_model_name() == "openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=512"


def test_local_embeddings_stay_the_default(monkeypatch, clean_dependencies):
    created = []

    class RecordingStore:
        def __init__(self, model_name="all-MiniLM-L6-v2", device=None, encoder=None):
            created.append((model_name, device, encoder))

    monkeypatch.setattr(dependencies, "EmbeddingsStoreFaiss", RecordingStore)
    dependencies.build_embeddings_store(EmbeddingSettings.from_env({"EMBEDDING_DEVICE": "cpu"}))
    assert created == [("all-MiniLM-L6-v2", "cpu", None)]
