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
- Every request is recorded in an optional UsageRecorder under the
  operation "embedding" (latency, input tokens, reported cost, errors).
- Query embeddings are computed while a user waits, so they can have their
  own short timeout, retries and deadline (query_call; the cloud profile
  sets them from ApiQuerySettings, V17); document batches keep the client's
  long defaults.
- Vectors are returned as the API sends them (unit length for the supported
  models); the store normalizes before indexing, as with the local encoder.
- Data policy (optional `zdr`): with zdr=True every request carries
  provider.zdr=true, so OpenRouter routes it only to Zero Data Retention
  endpoints (no storage of prompts or responses, hence no training on them).
  Checked on 2026-11-12 against https://openrouter.ai/api/v1/endpoints/zdr:
  the DeepInfra, Nebius and SiliconFlow endpoints of qwen/qwen3-embedding-8b
  are listed, none as "safety retention". Off by default here, so the request
  bodies of the published evaluation runs (whose response cache is keyed on
  the body) do not change; the application turns it on (EmbeddingSettings).
- Hedged query embeddings (optional QueryHedge): the query is sent to the
  pinned provider and, if it has not answered after `after_s`, to a second
  provider as well; the first answer wins and the other request is cancelled
  (openrouter_hedge.py). Query vectors of the second provider are usable
  against documents embedded by the first: measured for Nebius against the
  stored DeepInfra documents, nDCG@10 0.689 / 0.694 (two judges) against
  0.682 / 0.690 with DeepInfra's own query vectors, top-10 overlap 0.95
  (FINDINGS.md, question 2, evidence 4). So the vectors' identity
  (get_model_name) stays the pinned provider's: the documents are its.
- Query cache (optional, query_cache_size > 0): the vectors of the last
  queries, by exact text and model (not by provider, which only adds noise of
  cosine 0.996-0.9998). A repeated search in the same process then gets the
  same vector, so the same results, instead of one of DeepInfra's two
  variants. It is per process: a new serverless instance, or another one
  serving in parallel, starts empty and may still reorder the top 10.
"""

import base64
import logging
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx

from app.infrastructure.config.settings import ApiQuerySettings, DEFAULT_OPENROUTER_URL, EmbeddingSettings
from app.infrastructure.usage_recording import EMBEDDING_OPERATION, UsageRecorder
from app.infrastructure.search.openrouter_hedge import (
    HedgedPoster,
    HedgeFailed,
    HedgeLeg,
    LegOutcome,
    CANCELLED,
    NOT_SENT,
)
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

DEFAULT_HEDGE_DEADLINE_S = 5.0  # a hedged query call without an ApiQuerySettings deadline
DEFAULT_BATCH_SIZE = 32  # documents per request
DEFAULT_MAX_BATCH_CHARS = 20_000  # characters per request (see the module docstring)
API_NAME = "openrouter"
ANY_PROVIDER = "any"  # identity of an unpinned encoder (OpenRouter's routing)


@dataclass(frozen=True)
class QueryHedge:
    """
    How a query embedding is hedged (FINDINGS.md, question 2, recommendation).

    second_provider: provider slug of the second request (e.g. "nebius")
    after_s: seconds without an answer before the second request is sent
        (it is sent at once if the first one fails before that)
    primary_again_s: seconds before one more request to the pinned provider;
        None sends none
    deadline_s: most seconds for the whole call
    """

    second_provider: str
    after_s: float = 1.0
    primary_again_s: Optional[float] = 2.5
    deadline_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.second_provider.strip():
            raise ValueError("the hedge needs a second provider")
        if self.after_s < 0:
            raise ValueError(f"after_s must be >= 0, got {self.after_s}")
        if self.deadline_s <= 0:
            raise ValueError(f"deadline_s must be > 0, got {self.deadline_s}")
        for name, value in (("after_s", self.after_s), ("primary_again_s", self.primary_again_s)):
            if value is not None and value >= self.deadline_s:
                raise ValueError(f"{name} ({value}) must be shorter than the deadline ({self.deadline_s})")


class QueryVectorCache:
    """A bounded, thread-safe LRU of query vectors keyed by (model, dimension, exact text)."""

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._max = max_entries
        self._entries: "OrderedDict[Tuple[str, int, str], List[float]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple[str, int, str]) -> Optional[List[float]]:
        with self._lock:
            vector = self._entries.get(key)
            if vector is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return list(vector)  # a copy: callers may change their list

    def put(self, key: Tuple[str, int, str], vector: List[float]) -> None:
        with self._lock:
            self._entries[key] = list(vector)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def api_model_identity(model: str, dimension: int, upstream_provider: Optional[str] = None) -> str:
    """
    Identity of API vectors, as recorded in an index's metadata.

    Example: "openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=1024"
    """
    provider = (upstream_provider or ANY_PROVIDER).lower()
    return f"{API_NAME}:{canonical_model_name(model)}?provider={provider}&dimensions={dimension}"


def decode_embedding(raw: Any) -> List[float]:
    """
    A vector from the API: base64 little-endian float32, or a list of numbers.

    Decoded with the standard library (struct, "<f": little-endian float32),
    so the cloud profile needs no numpy; each value is the exact float32.
    """
    if isinstance(raw, str):
        data = base64.b64decode(raw)
        if len(data) % 4:
            raise ValueError(f"Base64 embedding of {len(data)} bytes is not a float32 vector")
        return list(struct.unpack(f"<{len(data) // 4}f", data))
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
        usage_recorder: Optional[UsageRecorder] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: Optional[httpx.BaseTransport] = None,
        http: Optional[OpenRouterHttp] = None,
        query_call: Optional[Mapping[str, Any]] = None,
        zdr: bool = False,
        query_hedge: Optional[QueryHedge] = None,
        hedged_poster: Optional[HedgedPoster] = None,
        query_cache_size: int = 0,
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
            query_call: Options of OpenRouterHttp.post for query embeddings
                (timeout_s, max_retries, deadline_s); None keeps the client's.
                Not used for queries while a hedge is configured.
            zdr: Restrict routing to Zero Data Retention endpoints (module docstring)
            query_hedge: Hedge query embeddings (needs upstream_provider, the
                first leg's provider); None sends one request as before
            hedged_poster: The client of the hedged calls (tests pass one
                with a fake transport); None builds one from api_key and base_url
            query_cache_size: Query vectors kept in memory; 0 keeps none

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
        if query_hedge is not None:
            if not upstream_provider:
                raise ValueError("a query hedge needs a pinned upstream_provider for its first request")
            if query_hedge.second_provider.lower() == upstream_provider.lower():
                raise ValueError("the hedge's second provider must differ from the pinned provider")
        if query_cache_size < 0:
            raise ValueError(f"query_cache_size must be >= 0, got {query_cache_size}")

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
        self._query_call: Dict[str, Any] = dict(query_call or {})
        self._zdr = zdr
        self._hedge = query_hedge
        self._hedged = hedged_poster if hedged_poster is not None or query_hedge is None else HedgedPoster(
            api_key=api_key, base_url=base_url,
        )
        self._cache = QueryVectorCache(query_cache_size) if query_cache_size > 0 else None

    @classmethod
    def from_settings(
        cls,
        settings: EmbeddingSettings,
        usage_recorder: Optional[UsageRecorder] = None,
        query_settings: Optional[ApiQuerySettings] = None,
    ) -> "OpenRouterEmbeddingEncoder":
        """
        Build the encoder of EMBEDDING_PROVIDER=openrouter (reads OPENROUTER_API_KEY).

        query_settings: deadlines of query embeddings (ApiQuerySettings); None
        keeps the long document deadlines for queries too. Its deadline is
        also the hedge's (DEFAULT_HEDGE_DEADLINE_S when it has none).
        """
        if settings.provider != "openrouter":
            raise ValueError(f"EMBEDDING_PROVIDER is '{settings.provider}', not 'openrouter'")
        hedge = None
        if settings.hedge_provider:
            deadline = query_settings.deadline_s if query_settings is not None else None
            hedge = QueryHedge(
                second_provider=settings.hedge_provider,
                after_s=settings.hedge_after_s,
                primary_again_s=settings.hedge_primary_again_s,
                deadline_s=deadline or DEFAULT_HEDGE_DEADLINE_S,
            )
        return cls(
            settings.model,
            dimensions=settings.dimensions,
            upstream_provider=settings.openrouter_provider,
            usage_recorder=usage_recorder,
            query_call=query_settings.call_options() if query_settings is not None else None,
            zdr=settings.openrouter_zdr,
            query_hedge=hedge,
            query_cache_size=settings.query_cache_size,
        )

    # -- the request -------------------------------------------------------------

    def request_body(self, inputs: List[str], provider: Optional[str] = None) -> Dict[str, Any]:
        """
        JSON body of one embeddings request for already prefixed inputs.

        provider: slug to pin instead of the encoder's (a hedge's second leg).
        """
        body: Dict[str, Any] = {"model": self._model, "input": list(inputs), "encoding_format": "base64"}
        if self._dimensions_parameter is not None:
            body["dimensions"] = self._dimensions_parameter
        pinned = provider or self._upstream
        routing: Dict[str, Any] = {"order": [pinned], "allow_fallbacks": False} if pinned else {}
        if self._zdr:
            routing["zdr"] = True
        if routing:
            body["provider"] = routing
        return body

    def _embed(self, inputs: List[str], call: Optional[Mapping[str, Any]] = None) -> List[List[float]]:
        """One request: vectors of `inputs` (already prefixed), in input order."""
        try:
            response = self._http.post("embeddings", self.request_body(inputs), **dict(call or {}))
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
        """
        Vector of one query (the model's query prefix or instruction is added).

        Served from the query cache when the same text was embedded before in
        this process; otherwise one request, or a hedged call when a hedge is
        configured (module docstring).
        """
        self._check_text(text)
        prefixed = self._convention.query_prefix + text
        key = (self._model, self._dimension, prefixed)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        if self._hedge is None:
            vector = self._embed([prefixed], self._query_call)[0]
        else:
            vector = self._embed_hedged(prefixed)
        if self._cache is not None:
            self._cache.put(key, vector)
        return vector

    def hedge_legs(self, prefixed: str) -> List[HedgeLeg]:
        """The requests of a hedged query call: pinned provider, second provider, pinned again."""
        assert self._hedge is not None and self._upstream is not None
        legs = [
            HedgeLeg(0.0, self.request_body([prefixed]), self._upstream),
            HedgeLeg(self._hedge.after_s, self.request_body([prefixed], self._hedge.second_provider),
                     self._hedge.second_provider.lower()),
        ]
        if self._hedge.primary_again_s is not None:
            legs.append(HedgeLeg(self._hedge.primary_again_s, self.request_body([prefixed]), self._upstream))
        return legs

    def _embed_hedged(self, prefixed: str) -> List[float]:
        """One hedged query call; every request sent is recorded (cost when OpenRouter reports one)."""
        assert self._hedge is not None and self._hedged is not None
        try:
            response = self._hedged.post_first("embeddings", self.hedge_legs(prefixed),
                                               deadline_s=self._hedge.deadline_s)
        except HedgeFailed as error:
            self._record_legs(error.legs)
            raise
        try:
            vector = self._read_vectors(response.body, 1)[0]
        except (ValueError, KeyError, TypeError) as error:
            self._record_legs(response.legs, failed_winner=type(error).__name__)
            raise
        self._record_legs(response.legs)
        return vector

    def _record_legs(self, legs: List[LegOutcome], failed_winner: Optional[str] = None) -> None:
        """
        One usage record per request sent. The winner carries OpenRouter's
        tokens and cost; a failed or cancelled request has no usage in any
        answer, so it is recorded as an error without cost ("cancelled" for
        the loser of a race).
        """
        for leg in legs:
            if not leg.sent or leg.error == NOT_SENT:
                continue
            if leg.error is None and failed_winner is None:
                usage = (leg.body or {}).get("usage") or {}
                self._record(leg.latency_s or 0.0, usage.get("prompt_tokens"), usage.get("cost"), None)
            elif leg.error is None:
                self._record(leg.latency_s or 0.0, None, None, failed_winner)
            else:
                self._record(leg.latency_s or 0.0, None, None,
                             CANCELLED if leg.error == CANCELLED else "OpenRouterError")

    @property
    def query_cache(self) -> Optional[QueryVectorCache]:
        """The query vector cache, or None without one."""
        return self._cache

    @property
    def query_hedge(self) -> Optional[QueryHedge]:
        """The hedge of query embeddings, or None."""
        return self._hedge

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
