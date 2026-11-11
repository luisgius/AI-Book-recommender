"""
Tests for the OpenRouter embedding encoder, its HTTP client and its use in
the FAISS store. No network: a fake transport (httpx.MockTransport) answers
every request.

Covered:
- request body (model, base64, dimensions only for Matryoshka models, provider pin)
- batching of documents, order of the vectors, base64 and JSON-number vectors
- query/document conventions per model (E5 prefixes, Qwen3 instruction, BGE-M3 none)
- retries on transient errors, no retry on other errors, timeouts, Retry-After
- usage recording (operation "embedding": tokens, cost, latency, errors)
- identity in the index metadata: local vs API, provider and dimension
- importing the API path loads no model library (subprocess)
"""

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import numpy as np
import pytest

import app.infrastructure.search.embeddings_store_faiss as store_module
from app.infrastructure.llm.usage import EMBEDDING_OPERATION, LLMUsageRecorder
from app.infrastructure.search.embeddings_store_faiss import (
    EmbeddingsStoreFaiss,
    IndexModelMismatchError,
    read_index_metadata,
)
from app.infrastructure.search.openrouter_embedding_encoder import (
    OpenRouterEmbeddingEncoder,
    api_model_identity,
    decode_embedding,
)
from app.infrastructure.search.openrouter_http import OpenRouterError, OpenRouterHttp
from app.infrastructure.search.text_encoding import QWEN3_QUERY_INSTRUCTION, TextEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Fake OpenRouter
# =============================================================================


def fake_vector(text: str, dimension: int) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    vector = np.random.default_rng(seed).normal(size=dimension)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


class FakeOpenRouter:
    """
    Records every request and answers like OpenRouter's embeddings endpoint.

    `script` is a list of responses or exceptions used before the normal
    answer: each item is an int (HTTP status with an error body), an
    exception instance (raised by the transport) or a callable(request).
    """

    def __init__(self, dimension: int, *, as_list: bool = False, script: Optional[List[Any]] = None,
                 headers: Optional[Dict[str, str]] = None) -> None:
        self.dimension = dimension
        self.as_list = as_list
        self.script = list(script or [])
        self.headers = headers or {}
        self.requests: List[httpx.Request] = []

    @property
    def bodies(self) -> List[Dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, int):
                return httpx.Response(step, json={"error": {"code": step, "message": "boom"}},
                                      headers=self.headers)
            return step(request)
        body = json.loads(request.content)
        dimension = body.get("dimensions", self.dimension)
        data = []
        for index, text in enumerate(body["input"]):
            vector = fake_vector(text, dimension)
            raw = vector.tolist() if self.as_list else base64.b64encode(vector.astype("<f4").tobytes()).decode()
            data.append({"object": "embedding", "embedding": raw, "index": index})
        data.reverse()  # the client must order by index, not by position
        return httpx.Response(200, json={
            "object": "list", "data": data, "model": body["model"], "provider": "FakeProvider",
            "usage": {"prompt_tokens": 3 * len(body["input"]), "total_tokens": 3 * len(body["input"]),
                      "cost": 1e-6 * len(body["input"])},
        })


def make_encoder(model: str = "baai/bge-m3", fake: Optional[FakeOpenRouter] = None, *,
                 dimensions: Optional[int] = None, upstream: Optional[str] = None, batch_size: int = 32,
                 recorder: Optional[LLMUsageRecorder] = None, max_retries: int = 3,
                 sleeps: Optional[List[float]] = None) -> OpenRouterEmbeddingEncoder:
    fake = fake or FakeOpenRouter(1024)
    waits = sleeps if sleeps is not None else []
    http = OpenRouterHttp(api_key="test-key", transport=httpx.MockTransport(fake), max_retries=max_retries,
                          sleep=waits.append)
    return OpenRouterEmbeddingEncoder(model, dimensions=dimensions, upstream_provider=upstream,
                                      batch_size=batch_size, usage_recorder=recorder, http=http)


# =============================================================================
# Request body and conventions
# =============================================================================


class TestRequest:
    def test_body_url_and_headers(self):
        fake = FakeOpenRouter(1024)
        make_encoder("baai/bge-m3", fake).generate_query_embedding("novela negra")

        request = fake.requests[0]
        assert str(request.url) == "https://openrouter.ai/api/v1/embeddings"
        assert request.headers["authorization"] == "Bearer test-key"
        # No identifying header (OpenRouter's optional attribution headers)
        assert "http-referer" not in request.headers and "x-title" not in request.headers
        assert fake.bodies[0] == {"model": "baai/bge-m3", "input": ["novela negra"], "encoding_format": "base64"}

    def test_provider_pin_and_dimensions_for_a_matryoshka_model(self):
        fake = FakeOpenRouter(4096)
        encoder = make_encoder("qwen/qwen3-embedding-8b", fake, dimensions=1024, upstream="DeepInfra")
        vector = encoder.generate_embedding("Un libro")

        assert len(vector) == 1024 == encoder.get_dimension()
        assert fake.bodies[0]["dimensions"] == 1024
        assert fake.bodies[0]["provider"] == {"order": ["deepinfra"], "allow_fallbacks": False}

    def test_native_dimension_sends_no_dimensions_parameter(self):
        fake = FakeOpenRouter(4096)
        encoder = make_encoder("qwen/qwen3-embedding-8b", fake, dimensions=4096)
        encoder.generate_embedding("Un libro")
        assert "dimensions" not in fake.bodies[0]
        assert encoder.get_dimension() == 4096

    def test_dimensions_are_refused_for_a_model_not_trained_for_truncation(self):
        with pytest.raises(ValueError, match="not trained to be truncated"):
            make_encoder("baai/bge-m3", dimensions=512)
        with pytest.raises(ValueError, match="not trained to be truncated"):
            make_encoder("intfloat/multilingual-e5-large", dimensions=256)

    @pytest.mark.parametrize("dimensions", [16, 5000])
    def test_dimensions_outside_the_model_range_are_refused(self, dimensions):
        with pytest.raises(ValueError, match="between 32 and 4096"):
            make_encoder("qwen/qwen3-embedding-8b", dimensions=dimensions)

    def test_unknown_model_is_refused(self):
        with pytest.raises(ValueError, match="Unknown OpenRouter embedding model"):
            make_encoder("some-org/some-embedding")

    def test_model_ids_are_compared_canonically(self):
        fake = FakeOpenRouter(1024)
        make_encoder("BAAI/bge-m3", fake).generate_embedding("x")
        assert fake.bodies[0]["model"] == "baai/bge-m3"


class TestConventions:
    def test_e5_uses_query_and_passage_prefixes(self):
        fake = FakeOpenRouter(1024)
        encoder = make_encoder("intfloat/multilingual-e5-large", fake)
        encoder.generate_query_embedding("dragones")
        encoder.generate_embedding("Un libro de dragones")
        encoder.generate_embeddings_batch(["uno", "dos"])
        assert [b["input"] for b in fake.bodies] == [
            ["query: dragones"], ["passage: Un libro de dragones"], ["passage: uno", "passage: dos"],
        ]

    @pytest.mark.parametrize("model", ["qwen/qwen3-embedding-8b", "qwen/qwen3-embedding-4b"])
    def test_qwen3_instruction_on_queries_only(self, model):
        fake = FakeOpenRouter(4096)
        encoder = make_encoder(model, fake, dimensions=1024)
        encoder.generate_query_embedding("dragones")
        encoder.generate_embeddings_batch(["Un libro de dragones"])
        assert fake.bodies[0]["input"] == [
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:dragones"
        ]
        assert fake.bodies[0]["input"][0] == QWEN3_QUERY_INSTRUCTION + "dragones"
        assert fake.bodies[1]["input"] == ["Un libro de dragones"]

    def test_bge_m3_is_symmetric(self):
        fake = FakeOpenRouter(1024)
        encoder = make_encoder("baai/bge-m3", fake)
        query = encoder.generate_query_embedding("dragones")
        document = encoder.generate_embedding("dragones")
        assert fake.bodies[0]["input"] == fake.bodies[1]["input"] == ["dragones"]
        assert query == document

    def test_empty_texts_are_refused_before_any_request(self):
        fake = FakeOpenRouter(1024)
        encoder = make_encoder("baai/bge-m3", fake)
        with pytest.raises(ValueError, match="empty"):
            encoder.generate_query_embedding("  ")
        with pytest.raises(ValueError, match="index 1"):
            encoder.generate_embeddings_batch(["ok", ""])
        assert fake.requests == []


# =============================================================================
# Batching and decoding
# =============================================================================


class TestBatching:
    def test_documents_are_sent_in_batches_and_kept_in_order(self):
        fake = FakeOpenRouter(1024)
        texts = [f"book {i}" for i in range(70)]
        vectors = make_encoder("baai/bge-m3", fake, batch_size=32).generate_embeddings_batch(texts)

        assert [len(b["input"]) for b in fake.bodies] == [32, 32, 6]
        assert len(vectors) == 70
        for text, vector in zip(texts, vectors):
            np.testing.assert_allclose(vector, fake_vector(text, 1024), rtol=0, atol=0)

    def test_requests_are_also_capped_by_characters(self):
        fake = FakeOpenRouter(1024)
        http = OpenRouterHttp(api_key="k", transport=httpx.MockTransport(fake))
        encoder = OpenRouterEmbeddingEncoder("baai/bge-m3", batch_size=32, max_batch_chars=100, http=http)
        texts = ["a" * 40, "b" * 40, "c" * 40, "d" * 150, "e" * 10]
        vectors = encoder.generate_embeddings_batch(texts)

        # 40+40 fit, a third 40 would pass 100; the 150-character text goes alone
        assert [b["input"] for b in fake.bodies] == [[texts[0], texts[1]], [texts[2]], [texts[3]], [texts[4]]]
        assert len(vectors) == 5
        assert encoder.batches(["x" * 30] * 3) == [["x" * 30] * 3]

    def test_default_character_cap(self):
        from app.infrastructure.search.openrouter_embedding_encoder import DEFAULT_MAX_BATCH_CHARS

        assert DEFAULT_MAX_BATCH_CHARS == 20_000
        encoder = make_encoder("baai/bge-m3")
        assert [len(b) for b in encoder.batches(["y" * 1000] * 50)] == [20, 20, 10]

    def test_empty_batch_makes_no_request(self):
        fake = FakeOpenRouter(1024)
        assert make_encoder("baai/bge-m3", fake).generate_embeddings_batch([]) == []
        assert fake.requests == []

    def test_json_number_vectors_are_read_too(self):
        fake = FakeOpenRouter(1024, as_list=True)
        vector = make_encoder("baai/bge-m3", fake).generate_embedding("x")
        np.testing.assert_allclose(vector, fake_vector("x", 1024), rtol=1e-7)

    def test_base64_decoding_is_exact_float32(self):
        values = np.array([0.1, -2.5, 3.0e-7], dtype=np.float32)
        raw = base64.b64encode(values.astype("<f4").tobytes()).decode()
        assert decode_embedding(raw) == [float(v) for v in values]

    def test_wrong_count_or_dimension_is_an_error(self):
        def two_vectors(request):
            vector = base64.b64encode(fake_vector("x", 1024).tobytes()).decode()
            return httpx.Response(200, json={"data": [{"embedding": vector, "index": 0}, {"embedding": vector, "index": 1}]})

        def short_vector(request):
            vector = base64.b64encode(fake_vector("x", 512).tobytes()).decode()
            return httpx.Response(200, json={"data": [{"embedding": vector, "index": 0}]})

        with pytest.raises(ValueError, match="2 vectors for 1 inputs"):
            make_encoder("baai/bge-m3", FakeOpenRouter(1024, script=[two_vectors])).generate_embedding("x")
        with pytest.raises(ValueError, match="512-dimensional"):
            make_encoder("baai/bge-m3", FakeOpenRouter(1024, script=[short_vector])).generate_embedding("x")


# =============================================================================
# Retries, timeouts and errors
# =============================================================================


class TestRetries:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
    def test_transient_status_is_retried(self, status):
        fake = FakeOpenRouter(1024, script=[status, status])
        sleeps: List[float] = []
        vector = make_encoder("baai/bge-m3", fake, sleeps=sleeps).generate_embedding("x")
        assert len(vector) == 1024
        assert len(fake.requests) == 3
        assert sleeps == [1.0, 2.0]  # exponential backoff

    def test_timeout_and_connection_errors_are_retried(self):
        fake = FakeOpenRouter(1024, script=[httpx.ReadTimeout("slow"), httpx.ConnectError("down")])
        assert len(make_encoder("baai/bge-m3", fake).generate_embedding("x")) == 1024
        assert len(fake.requests) == 3

    def test_retry_after_header_is_honoured(self):
        fake = FakeOpenRouter(1024, script=[429], headers={"Retry-After": "7"})
        sleeps: List[float] = []
        make_encoder("baai/bge-m3", fake, sleeps=sleeps).generate_embedding("x")
        assert sleeps == [7.0]

    def test_gives_up_after_the_retries(self):
        fake = FakeOpenRouter(1024, script=[503, 503, 503])
        with pytest.raises(OpenRouterError, match="HTTP 503.*after 3 attempts") as error:
            make_encoder("baai/bge-m3", fake, max_retries=2).generate_embedding("x")
        assert error.value.transient is True
        assert len(fake.requests) == 3

    @pytest.mark.parametrize("status", [400, 401, 402, 404])
    def test_other_errors_are_not_retried(self, status):
        fake = FakeOpenRouter(1024, script=[status])
        with pytest.raises(OpenRouterError) as error:
            make_encoder("baai/bge-m3", fake).generate_embedding("x")
        assert error.value.status == status and error.value.transient is False
        assert len(fake.requests) == 1
        assert "test-key" not in str(error.value)

    def test_error_inside_a_200_body(self):
        overloaded = lambda request: httpx.Response(200, json={"error": {"code": 529, "message": "overloaded"}})
        fake = FakeOpenRouter(1024, script=[overloaded])
        assert len(make_encoder("baai/bge-m3", fake).generate_embedding("x")) == 1024
        assert len(fake.requests) == 2

        bad = lambda request: httpx.Response(200, json={"error": {"code": 400, "message": "bad input"}})
        with pytest.raises(OpenRouterError, match="bad input"):
            make_encoder("baai/bge-m3", FakeOpenRouter(1024, script=[bad])).generate_embedding("x")

    def test_timeout_is_set_on_the_client(self):
        http = OpenRouterHttp(api_key="k", timeout_s=12.5, transport=httpx.MockTransport(FakeOpenRouter(8)))
        assert http._client.timeout.read == 12.5

    def test_missing_key_names_the_variable(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterEmbeddingEncoder("baai/bge-m3")


# =============================================================================
# Usage recording
# =============================================================================


class TestUsage:
    def test_each_request_is_recorded_as_an_embedding_call(self):
        recorder = LLMUsageRecorder()
        make_encoder("baai/bge-m3", recorder=recorder, batch_size=2).generate_embeddings_batch(["a", "b", "c"])

        records = recorder.records
        assert [r.operation for r in records] == [EMBEDDING_OPERATION] * 2
        assert [r.input_tokens for r in records] == [6, 3]
        assert [r.output_tokens for r in records] == [0, 0]
        assert [r.cost_usd for r in records] == pytest.approx([2e-6, 1e-6])
        assert all(r.latency_s >= 0 and r.error is None and r.model == "baai/bge-m3" for r in records)
        total = recorder.summary()["total"]
        assert total["calls"] == 2 and total["calls_without_usage"] == 0
        assert total["reported_cost_usd"] == pytest.approx(3e-6)

    def test_a_failed_request_is_recorded_with_its_error(self):
        recorder = LLMUsageRecorder()
        with pytest.raises(OpenRouterError):
            make_encoder("baai/bge-m3", FakeOpenRouter(1024, script=[401]), recorder=recorder).generate_embedding("x")
        (record,) = recorder.records
        assert record.error == "OpenRouterError" and record.input_tokens is None

    def test_record_call_directly(self):
        recorder = LLMUsageRecorder()
        recorder.record_call("rerank", "m", 0.5, 120, cost_usd=2.4e-5)
        summary = recorder.summary()["rerank"]
        assert summary["calls"] == 1 and summary["input_tokens"] == 120 and summary["output_tokens"] == 0
        assert summary["reported_cost_usd"] == pytest.approx(2.4e-5)


# =============================================================================
# Identity and the FAISS store
# =============================================================================


class FakeSentenceTransformer:
    """Local model stand-in (1024 dimensions, like e5-large)."""

    def __init__(self, model_name_or_path: str, device: Optional[str] = None, **kwargs: Any):
        self.device = device or "cpu"

    def get_sentence_embedding_dimension(self) -> int:
        return 1024

    def encode(self, sentences, convert_to_numpy: bool = True, **kwargs):
        return np.array([fake_vector(s, 1024) for s in sentences])


def build_and_save(encoder: TextEncoder, path: Path, n: int = 4) -> EmbeddingsStoreFaiss:
    from uuid import uuid4

    store = EmbeddingsStoreFaiss(encoder=encoder)
    texts = [f"book {i}" for i in range(n)]
    store.store_embeddings_batch([uuid4() for _ in texts], store.generate_embeddings_batch(texts))
    store.build_index()
    store.save_index(str(path))
    return store


class TestIdentityAndStore:
    def test_identity_names_api_model_provider_and_dimension(self):
        assert api_model_identity("Qwen/Qwen3-Embedding-8B", 1024, "DeepInfra") == (
            "openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=1024"
        )
        assert make_encoder("baai/bge-m3").get_model_name() == "openrouter:baai/bge-m3?provider=any&dimensions=1024"
        assert make_encoder("baai/bge-m3").get_device() == "api"

    def test_store_with_an_api_encoder_records_the_identity(self, tmp_path):
        encoder = make_encoder("qwen/qwen3-embedding-8b", FakeOpenRouter(4096), dimensions=1024, upstream="deepinfra")
        store = build_and_save(encoder, tmp_path)

        metadata = read_index_metadata(str(tmp_path))
        assert metadata.model_name == encoder.get_model_name()
        assert metadata.dimension == 1024 and metadata.vector_count == 4
        assert store.get_device() == "api" and store.get_encoder() is encoder
        assert store.is_ready()

        # Same identity: loads and answers queries with the API encoder
        same = EmbeddingsStoreFaiss(encoder=make_encoder("qwen/qwen3-embedding-8b", FakeOpenRouter(4096),
                                                         dimensions=1024, upstream="deepinfra"))
        same.load_index(str(tmp_path))
        assert len(same.generate_query_embedding("dragones")) == 1024

    @pytest.mark.parametrize("other", [
        dict(model="qwen/qwen3-embedding-8b", dimensions=1024, upstream="nebius"),   # another provider
        dict(model="qwen/qwen3-embedding-8b", dimensions=1024, upstream=None),       # unpinned
        dict(model="qwen/qwen3-embedding-8b", dimensions=512, upstream="deepinfra"),  # another dimension
        dict(model="qwen/qwen3-embedding-4b", dimensions=1024, upstream="deepinfra"),  # another model
    ])
    def test_another_api_identity_cannot_load_the_index(self, tmp_path, other):
        build_and_save(make_encoder("qwen/qwen3-embedding-8b", FakeOpenRouter(4096), dimensions=1024,
                                    upstream="deepinfra"), tmp_path)
        encoder = make_encoder(other["model"], FakeOpenRouter(4096), dimensions=other["dimensions"],
                               upstream=other["upstream"])
        with pytest.raises(IndexModelMismatchError, match="Rebuild the index"):
            EmbeddingsStoreFaiss(encoder=encoder).load_index(str(tmp_path))

    def test_local_and_api_versions_of_one_model_do_not_mix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store_module, "SentenceTransformer", FakeSentenceTransformer)
        local_dir, api_dir = tmp_path / "local", tmp_path / "api"
        local = EmbeddingsStoreFaiss(model_name="intfloat/multilingual-e5-large")
        from uuid import uuid4
        local.store_embedding(uuid4(), local.generate_embedding("book"))
        local.build_index()
        local.save_index(str(local_dir))
        build_and_save(make_encoder("intfloat/multilingual-e5-large"), api_dir)

        with pytest.raises(IndexModelMismatchError):
            EmbeddingsStoreFaiss(encoder=make_encoder("intfloat/multilingual-e5-large")).load_index(str(local_dir))
        with pytest.raises(IndexModelMismatchError):
            EmbeddingsStoreFaiss(model_name="intfloat/multilingual-e5-large").load_index(str(api_dir))

    def test_encoder_and_model_name_together_are_refused(self):
        with pytest.raises(ValueError, match="not both"):
            EmbeddingsStoreFaiss(model_name="intfloat/multilingual-e5-large", encoder=make_encoder())
        with pytest.raises(ValueError, match="not both"):
            EmbeddingsStoreFaiss(device="cpu", encoder=make_encoder())

    def test_api_encoder_fits_the_postgres_store_contract(self):
        """The Postgres store takes any object with the encoding half of EmbeddingsStore."""
        encoder = make_encoder()
        for method in ("generate_embedding", "generate_query_embedding", "generate_embeddings_batch",
                       "get_dimension", "get_model_name"):
            assert callable(getattr(encoder, method))


# =============================================================================
# No model library on the API path
# =============================================================================


def test_api_path_imports_no_model_library():
    """The cloud backend must not need torch: build both API adapters from settings."""
    code = (
        "import sys\n"
        "from app.infrastructure.config.settings import EmbeddingSettings, RerankerSettings\n"
        "from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder\n"
        "from app.infrastructure.search.api_reranker import ApiReranker\n"
        "OpenRouterEmbeddingEncoder.from_settings(EmbeddingSettings.from_env())\n"
        "ApiReranker.from_settings(RerankerSettings.from_env())\n"
        "loaded = [m for m in ('torch', 'sentence_transformers', 'transformers', 'faiss') if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    env = {"PATH": "/usr/bin:/bin", "OPENROUTER_API_KEY": "test-key", "EMBEDDING_PROVIDER": "openrouter",
           "EMBEDDING_MODEL": "qwen/qwen3-embedding-8b", "EMBEDDING_DIMENSIONS": "1024",
           "RERANKER_PROVIDER": "openrouter", "RERANKER_MODEL": "qwen/qwen3-reranker-8b"}
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"model libraries imported: {result.stdout.strip()}"
