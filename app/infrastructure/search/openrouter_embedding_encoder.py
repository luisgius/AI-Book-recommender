"""
TextEncoder that computes embeddings through OpenRouter's embeddings API.

This is the encoder of the cloud profile: the process loads no model (no
torch, no sentence-transformers), so the backend fits a small serverless
function. It can be injected into EmbeddingsStoreFaiss (local FAISS index,
vectors from the API) or used by any store that takes an encoder.

Request (POST https://openrouter.ai/api/v1/embeddings, checked against
OpenRouter's API reference on 2026-11-11):
    {"model": ..., "input": [texts], "encoding_format": "base64",
     "dimensions": N (optional), "provider": {"order": [p], "allow_fallbacks": false} (optional)}
Response: {"data": [{"embedding": <base64 float32 or list>, "index": i}, ...],
           "model": ..., "provider": ..., "usage": {"prompt_tokens", "total_tokens", "cost"}}

Design decisions:
- Query/document conventions come from PROMPT_CONVENTIONS (text_encoding.py),
  the same table as the local encoder: "query: "/"passage: " for E5, the
  Qwen3 instruction on queries only, nothing for BGE-M3. An API has no stored
  prompts, so a model whose convention needs one is rejected.
- Output dimension: `dimensions` is sent only for models trained to be
  truncated (Matryoshka representation learning: the Qwen3 embeddings).
  Some providers truncate any model when asked (DeepInfra returned 512
  values for baai/bge-m3, which is not trained for it), so the check is made
  here, from a table of known models, not left to the provider.
- Provider pinning: OpenRouter may route the same model to providers whose
  vectors differ (Qwen3-Embedding-8B: Nebius and DeepInfra gave cosine 0.997
  for the same text on 2026-11-11). Pinning one provider (no fallbacks) keeps
  queries and documents in one vector space; the provider is part of the
  identity recorded in the index metadata.
- Identity (get_model_name): "openrouter:<model>?provider=<p or any>&dimensions=<d>".
  An index built with one identity cannot be loaded by a store whose encoder
  has another (IndexModelMismatchError), so a local model, another API
  provider or another dimension is never mixed with it.
- Documents are sent in requests of at most `batch_size` texts and at most
  `max_batch_chars` characters (greedy, in input order; a longer text goes
  alone). The character cap exists because OpenRouter compares the tokens of
  the whole request with the endpoint's context and answers HTTP 404 "No
  endpoints found" when they exceed it: 32 book texts of 43,098 characters
  (about 9,150 tokens) failed for baai/bge-m3, whose DeepInfra endpoint
  lists 8,192 tokens, while each half succeeded (2026-11-11). 20,000
  characters stay under 8,192 tokens down to about 2.5 characters per token.
  Each request is retried on transient errors with a timeout (openrouter_http.py). Vectors are requested
  as base64 float32 (about 2.4 times smaller than JSON numbers, and exact);
  a provider that answers with JSON numbers is read as well.
- Every request is recorded in an optional LLMUsageRecorder under the
  operation "embedding" (latency, input tokens, reported cost, errors).
- Vectors are returned as the API sends them (unit length for the supported
  models); the store normalizes before indexing, as with the local encoder.
"""

import base64
import logging
from typing import Any, Dict, List, Mapping, Optional

import httpx
import numpy as np

from app.infrastructure.config.settings import DEFAULT_OPENROUTER_URL, EmbeddingSettings
from app.infrastructure.llm.usage import EMBEDDING_OPERATION, LLMUsageRecorder
from app.infrastructure.search.openrouter_http import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_S,
    OpenRouterError,
    OpenRouterHttp,
)
from app.infrastructure.search.text_encoding import (
    TextEncoder,
    canonical_model_name,
    prompt_convention_for,
)

logger = logging.getLogger(__name__)

# Native output dimension of the embedding models this encoder knows
# (OpenRouter ids; from each model card). Another model needs an entry here.
NATIVE_DIMENSIONS: Dict[str, int] = {
    "baai/bge-m3": 1024,
    "intfloat/multilingual-e5-large": 1024,
    "qwen/qwen3-embedding-4b": 2560,
    "qwen/qwen3-embedding-8b": 4096,
}

# Models trained with Matryoshka representation learning: their vectors can
# be cut to a shorter prefix (the model cards allow 32 up to the native size).
MRL_MODELS = frozenset({"qwen/qwen3-embedding-4b", "qwen/qwen3-embedding-8b"})
MIN_MRL_DIMENSIONS = 32

DEFAULT_BATCH_SIZE = 32  # documents per request
DEFAULT_MAX_BATCH_CHARS = 20_000  # characters per request (see the module docstring)
API_NAME = "openrouter"
ANY_PROVIDER = "any"  # identity of an unpinned encoder (OpenRouter's routing)


def api_model_identity(model: str, dimension: int, upstream_provider: Optional[str] = None) -> str:
    """
    Identity of API vectors, as recorded in an index's metadata.

    Example: "openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=1024"
    """
    provider = (upstream_provider or ANY_PROVIDER).lower()
    return f"{API_NAME}:{canonical_model_name(model)}?provider={provider}&dimensions={dimension}"


def decode_embedding(raw: Any) -> List[float]:
    """A vector from the API: base64 little-endian float32, or a list of numbers."""
    if isinstance(raw, str):
        return np.frombuffer(base64.b64decode(raw), dtype="<f4").astype(float).tolist()
    if isinstance(raw, list):
        return [float(value) for value in raw]
    raise ValueError(f"Unexpected embedding format: {type(raw).__name__}")


class OpenRouterEmbeddingEncoder(TextEncoder):
    """
    Embeddings of documents and queries computed by OpenRouter (see module docstring).

    Example:
        >>> encoder = OpenRouterEmbeddingEncoder("qwen/qwen3-embedding-8b", dimensions=1024,
        ...                                      upstream_provider="deepinfra")
        >>> store = EmbeddingsStoreFaiss(encoder=encoder)
    """

    def __init__(
        self,
        model: str,
        *,
        dimensions: Optional[int] = None,
        upstream_provider: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
        usage_recorder: Optional[LLMUsageRecorder] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: Optional[httpx.BaseTransport] = None,
        http: Optional[OpenRouterHttp] = None,
    ) -> None:
        """
        Args:
            model: OpenRouter id of an embedding model (a key of NATIVE_DIMENSIONS)
            dimensions: Output dimension for a Matryoshka model; None keeps the
                native dimension (and sends no `dimensions` parameter)
            upstream_provider: Provider slug to pin (e.g. "deepinfra"); None
                lets OpenRouter route each request
            batch_size: Documents per request
            max_batch_chars: Characters per request (a longer single text goes alone)
            usage_recorder: Records every request under the operation "embedding"
            api_key, base_url, timeout_s, max_retries, transport: HTTP options
                (see OpenRouterHttp); ignored when `http` is given
            http: An already built client (tests and evaluation)

        Raises:
            ValueError: Unknown model, a dimension the model does not support,
                a convention that needs a stored prompt, or batch_size < 1
        """
        canonical = canonical_model_name(model)
        if canonical not in NATIVE_DIMENSIONS:
            raise ValueError(
                f"Unknown OpenRouter embedding model '{model}': its native dimension and whether it "
                f"can be truncated are not known. Known models: {sorted(NATIVE_DIMENSIONS)}"
            )
        native = NATIVE_DIMENSIONS[canonical]
        if dimensions is not None and dimensions != native:
            if canonical not in MRL_MODELS:
                raise ValueError(
                    f"'{model}' is not trained to be truncated (no Matryoshka training): it only "
                    f"produces {native}-dimensional vectors, got dimensions={dimensions}"
                )
            if not MIN_MRL_DIMENSIONS <= dimensions <= native:
                raise ValueError(
                    f"dimensions for '{model}' must be between {MIN_MRL_DIMENSIONS} and {native}, got {dimensions}"
                )
        convention = prompt_convention_for(model)
        if convention.query_prompt_name:
            raise ValueError(
                f"'{model}' needs the stored prompt '{convention.query_prompt_name}', which only a local "
                f"model can apply"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_batch_chars < 1:
            raise ValueError(f"max_batch_chars must be >= 1, got {max_batch_chars}")

        self._model = canonical
        self._dimension = dimensions if dimensions is not None else native
        # Send `dimensions` only when it changes the output (truncation).
        self._dimensions_parameter = self._dimension if self._dimension != native else None
        self._upstream = upstream_provider.lower() if upstream_provider else None
        self._convention = convention
        self._batch_size = batch_size
        self._max_batch_chars = max_batch_chars
        self._recorder = usage_recorder
        self._http = http or OpenRouterHttp(
            api_key=api_key, base_url=base_url, timeout_s=timeout_s,
            max_retries=max_retries, transport=transport,
        )

    @classmethod
    def from_settings(
        cls, settings: EmbeddingSettings, usage_recorder: Optional[LLMUsageRecorder] = None
    ) -> "OpenRouterEmbeddingEncoder":
        """Build the encoder of EMBEDDING_PROVIDER=openrouter (reads OPENROUTER_API_KEY)."""
        if settings.provider != "openrouter":
            raise ValueError(f"EMBEDDING_PROVIDER is '{settings.provider}', not 'openrouter'")
        return cls(
            settings.model,
            dimensions=settings.dimensions,
            upstream_provider=settings.openrouter_provider,
            usage_recorder=usage_recorder,
        )

    # -- the request -------------------------------------------------------------

    def request_body(self, inputs: List[str]) -> Dict[str, Any]:
        """JSON body of one embeddings request for already prefixed inputs."""
        body: Dict[str, Any] = {"model": self._model, "input": list(inputs), "encoding_format": "base64"}
        if self._dimensions_parameter is not None:
            body["dimensions"] = self._dimensions_parameter
        if self._upstream:
            body["provider"] = {"order": [self._upstream], "allow_fallbacks": False}
        return body

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        """One request: vectors of `inputs` (already prefixed), in input order."""
        try:
            response = self._http.post("embeddings", self.request_body(inputs))
            vectors = self._read_vectors(response.body, len(inputs))
        except (OpenRouterError, ValueError, KeyError, TypeError) as error:
            latency = getattr(error, "latency_s", None)
            self._record(latency or 0.0, None, None, type(error).__name__)
            raise
        usage = response.body.get("usage") or {}
        self._record(response.latency_s, usage.get("prompt_tokens"), usage.get("cost"), None)
        return vectors

    def _read_vectors(self, body: Mapping[str, Any], expected: int) -> List[List[float]]:
        """Decode, order and check the vectors of a response."""
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            got = len(data) if isinstance(data, list) else "no"
            raise ValueError(f"Embeddings response has {got} vectors for {expected} inputs")
        items = sorted(data, key=lambda item: int(item["index"]))
        if [int(item["index"]) for item in items] != list(range(expected)):
            raise ValueError("Embeddings response indexes are not 0..n-1")
        vectors = [decode_embedding(item["embedding"]) for item in items]
        for vector in vectors:
            if len(vector) != self._dimension:
                raise ValueError(
                    f"Embeddings response has a {len(vector)}-dimensional vector; "
                    f"{self.get_model_name()} must give {self._dimension}"
                )
        return vectors

    def _record(self, latency_s: float, tokens: Optional[int], cost: Optional[float], error: Optional[str]) -> None:
        if self._recorder is not None:
            self._recorder.record_call(
                EMBEDDING_OPERATION, self._model, latency_s, tokens, 0,
                cost_usd=float(cost) if isinstance(cost, (int, float)) else None, error=error,
            )

    @staticmethod
    def _check_text(text: str, position: Optional[int] = None) -> None:
        if not text or not text.strip():
            where = "" if position is None else f" at index {position}"
            raise ValueError(f"Cannot generate embedding for empty text{where}")

    # -- TextEncoder -------------------------------------------------------------

    def generate_embedding(self, text: str) -> List[float]:
        """Vector of one document (the model's document prefix is added)."""
        self._check_text(text)
        return self._embed([self._convention.document_prefix + text])[0]

    def generate_query_embedding(self, text: str) -> List[float]:
        """Vector of one query (the model's query prefix or instruction is added)."""
        self._check_text(text)
        return self._embed([self._convention.query_prefix + text])[0]

    def batches(self, inputs: List[str]) -> List[List[str]]:
        """Split inputs into requests: at most batch_size texts and max_batch_chars characters each."""
        batches: List[List[str]] = []
        current: List[str] = []
        size = 0
        for text in inputs:
            if current and (len(current) == self._batch_size or size + len(text) > self._max_batch_chars):
                batches.append(current)
                current, size = [], 0
            current.append(text)
            size += len(text)
        if current:
            batches.append(current)
        return batches

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Vectors of several documents, in input order (see batches() for the requests)."""
        for position, text in enumerate(texts):
            self._check_text(text, position)
        prefixed = [self._convention.document_prefix + text for text in texts]
        vectors: List[List[float]] = []
        for batch in self.batches(prefixed):
            vectors.extend(self._embed(batch))
        return vectors

    def get_dimension(self) -> int:
        """Dimension of the returned vectors (requested or native)."""
        return self._dimension

    def get_model_name(self) -> str:
        """Identity of the vectors: API, model, pinned provider and dimension."""
        return api_model_identity(self._model, self._dimension, self._upstream)

    def get_device(self) -> str:
        """Encoding runs remotely."""
        return "api"

    @property
    def model(self) -> str:
        """OpenRouter id of the model."""
        return self._model

    @property
    def upstream_provider(self) -> Optional[str]:
        """Pinned provider slug, or None (OpenRouter's routing)."""
        return self._upstream
