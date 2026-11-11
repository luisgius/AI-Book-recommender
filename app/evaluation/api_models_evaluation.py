"""
Embeddings and re-ranking by API, measured against the local models (task V16).

The cloud profile cannot load a model (its backend must fit a small
serverless function), so embeddings and re-ranking come from OpenRouter's
API there. This module measures the available API models (cheap Chinese
models only: BAAI and Alibaba/Qwen) against the local models of the evaluated
profile, on this project's collection, and chooses with a rule that counts
speed as well as quality.

Design (fixed before any result was seen):
- Collection: the 177 books and 44 test queries, 10 results per query, the
  harness of the V06 ablation (retrieval_ablation.py: ModelStack,
  make_searcher, the system definitions) and comparison.py for the metrics.
- Embedding arms (vector side only, exact FAISS index over L2-normalized
  vectors, document text Book.get_searchable_text()): local references
  vec_e5 (intfloat/multilingual-e5-large-instruct) and vec_qwen3
  (Qwen/Qwen3-Embedding-0.6B) of V06; primary API arms baai/bge-m3,
  intfloat/multilingual-e5-large and qwen/qwen3-embedding-8b at 1024
  dimensions; descriptive API arms qwen/qwen3-embedding-8b at 512, 256 and
  its native 4096 dimensions, and qwen/qwen3-embedding-4b at 1024.
- Reranker arms, all re-ranking the same top 30 of hybrid_norm_e5: local
  reference hybrid_norm_e5_bge (BAAI/bge-reranker-v2-m3); API arms
  qwen/qwen3-reranker-0.6b and qwen/qwen3-reranker-8b (document text cut to
  2,000 characters, ApiReranker's default).
- Providers pinned (no fallbacks), so the vectors of one arm come from one
  endpoint and the latency of the arms is not confounded by routing: every
  embedding arm on DeepInfra (the only provider of e5-large and
  qwen3-embedding-4b), both rerankers on PrimeIntellect (the only provider
  of qwen3-reranker-0.6b, bf16).
- Judgments: start from the V10 file
  relevance_judgments_extended_z-ai-glm-5.3-flash.json (read only); the
  unjudged top-10 pairs are graded by the configured judge (JUDGE_*, which
  must be the file's judge) through its cache, and the merged judgments are
  written to relevance_judgments_extended_api_models.json. Judged@10 = 1.0
  is asserted for every system reported; every system is scored on that
  final file.
- Pre-declared contrasts (A1-A5): each primary API embedding arm against
  vec_e5, each API reranker against hybrid_norm_e5_bge. Two-sided paired
  randomisation test on per-query nDCG@10, Holm across exactly these five,
  paired bootstrap 95% intervals (10,000 resamples, seed 0).
- Speed and cost: per query-side call (one embedding request per query, one
  rerank request per search), 44 queries, sequential, after one warm-up
  call: median and p95 latency (hosted latency includes OpenRouter's
  routing), tokens and reported cost per call; time and cost to embed the
  177 documents (requests of at most 32 texts and 20,000 characters,
  sequential) and the one-off cost
  extrapolated to 200,000 and 1,000,000 works. Local references: the same
  calls timed in this process on this machine's GPU.
- Stability: the 44 queries are embedded twice by every API arm (maximum
  cosine difference between the two vectors of a query, and how many top-10
  lists change); each API reranker runs twice (identical top-10 lists).
- Selection rule (select_by_speed_quality in llm_model_selection.py): the
  candidates within 0.02 of the best quality are tied; among them the lowest
  median latency wins; if another tied candidate is within 10% of that
  latency, the cheaper wins; a candidate with a failed call is excluded.
  Cloud embedding model: quality = nDCG@10 of the vector side, candidates =
  the three primary API arms. Cloud reranker: quality = nDCG@10 after
  re-ranking, candidates = the two API rerankers and the LLM listwise
  recommender R1 of V07 (its run-1 lists, LLM-call latency and cost, from
  results_llm_recommender.json and its cache; not re-run).
- End-to-end check (descriptive): the real SearchService with BM25
  (normalizing tokenizer, local), FAISS over the selected API embedding
  model's vectors, the API query embedding and the selected API reranker.

Every API response is cached on disk (data/evaluation/api_models/responses.jsonl,
keyed by the request body and a run label: documents, warmup, run1, run2,
e2e), so the tables are recomputed without any network call. Rerank
responses are stored without the echoed document texts (they are the
request). Local timings and the end-to-end wall clock are saved next to
them (local_timings.json), because they cannot be replayed.

CLI (from the repository root):

    python -m app.evaluation.api_models_evaluation            # call what is not cached, judge the holes
    python -m app.evaluation.api_models_evaluation --offline  # caches only; fail loudly if anything is missing
    python -m app.evaluation.api_models_evaluation --render   # re-render the Markdown from the saved JSON

Outputs: data/evaluation/results_api_models.json, data/evaluation/API_MODELS.md
and app/evaluation/relevance_judgments_extended_api_models.json.
"""

import argparse
import json
import logging
import os
import shlex
import statistics
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

import httpx
import numpy as np

from app.domain.entities import Book, SearchResult
from app.domain.ports import Reranker
from app.evaluation.comparison import PRIMARY_METRIC, _git, _sha256, compare_pairs, compare_systems
from app.evaluation.evaluation_service import BookId
from app.evaluation.llm_model_selection import (
    LATENCY_BAND,
    TIE_MARGIN,
    select_by_speed_quality,
)
from app.evaluation.relevance_judge import GRADES, find_unjudged
from app.evaluation.retrieval_ablation import (
    DEPTH,
    NORMALIZED,
    RERANK_DEPTH,
    SYSTEMS,
    Contrast,
    ModelStack,
    RerankerFailedError,
    SystemSpec,
    UnjudgedResultsError,
    _fmt,
    _fmt_ci,
    _fmt_p,
    _table,
    make_searcher,
    require_judged,
    resolve_grades,
)
from app.evaluation.types import TestQuery
from app.infrastructure.search.api_reranker import DEFAULT_MAX_DOCUMENT_CHARS, ApiReranker
from app.infrastructure.search.openrouter_embedding_encoder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_BATCH_CHARS,
    NATIVE_DIMENSIONS,
    OpenRouterEmbeddingEncoder,
)
from app.infrastructure.search.openrouter_http import OpenRouterError, OpenRouterHttp

logger = logging.getLogger(__name__)

# =============================================================================
# The design (fixed in advance)
# =============================================================================

EMBEDDING_PIN = "deepinfra"
RERANK_PIN = "primeintellect"
API_TIMEOUT_S = 30.0
API_RETRIES = 3
SPEND_LIMIT_USD = 1.0
EXTRAPOLATION_SIZES = (200_000, 1_000_000)
STORAGE_WORKS = 1_000_000
RUNS = ("run1", "run2")

LOCAL_SPECS = {spec.name: spec for spec in SYSTEMS}
VEC_E5, VEC_QWEN3 = "vec_e5", "vec_qwen3"
CANDIDATES = "hybrid_norm_e5"  # candidate generator of every reranker arm
BGE_REFERENCE = "hybrid_norm_e5_bge"
R1 = "rag_llm_recommender"
E2E = "e2e_api_stack"


@dataclass(frozen=True)
class EmbeddingArm:
    """An API embedding arm: one model at one output dimension, pinned to EMBEDDING_PIN."""

    name: str
    model: str
    dimension: int
    primary: bool

    def build(self, http: OpenRouterHttp) -> OpenRouterEmbeddingEncoder:
        return OpenRouterEmbeddingEncoder(self.model, dimensions=self.dimension, upstream_provider=EMBEDDING_PIN,
                                          batch_size=DEFAULT_BATCH_SIZE, http=http)

    @property
    def description(self) -> str:
        native = NATIVE_DIMENSIONS[self.model]
        size = "native" if self.dimension == native else f"cut from {native} (Matryoshka)"
        return f"FAISS, `{self.model}` by API at {self.dimension} dimensions ({size}), pinned to {EMBEDDING_PIN}"


@dataclass(frozen=True)
class RerankArm:
    """An API reranker re-ranking the top 30 of hybrid_norm_e5, pinned to RERANK_PIN."""

    name: str
    model: str

    def build(self, http: OpenRouterHttp) -> ApiReranker:
        return ApiReranker(self.model, upstream_provider=RERANK_PIN, http=http)

    @property
    def description(self) -> str:
        return (f"{CANDIDATES} + `{self.model}` by API over the top {RERANK_DEPTH} "
                f"(texts cut to {DEFAULT_MAX_DOCUMENT_CHARS} characters), pinned to {RERANK_PIN}")


EMBEDDING_ARMS: Tuple[EmbeddingArm, ...] = (
    EmbeddingArm("vec_api_bge_m3", "baai/bge-m3", 1024, True),
    EmbeddingArm("vec_api_e5_large", "intfloat/multilingual-e5-large", 1024, True),
    EmbeddingArm("vec_api_qwen3_8b_1024", "qwen/qwen3-embedding-8b", 1024, True),
    EmbeddingArm("vec_api_qwen3_8b_512", "qwen/qwen3-embedding-8b", 512, False),
    EmbeddingArm("vec_api_qwen3_8b_256", "qwen/qwen3-embedding-8b", 256, False),
    EmbeddingArm("vec_api_qwen3_8b_4096", "qwen/qwen3-embedding-8b", 4096, False),
    EmbeddingArm("vec_api_qwen3_4b_1024", "qwen/qwen3-embedding-4b", 1024, False),
)

RERANK_ARMS: Tuple[RerankArm, ...] = (
    RerankArm("hybrid_norm_e5_api_qwen3_rerank_0.6b", "qwen/qwen3-reranker-0.6b"),
    RerankArm("hybrid_norm_e5_api_qwen3_rerank_8b", "qwen/qwen3-reranker-8b"),
)

CONTRASTS: Tuple[Contrast, ...] = (
    Contrast("A1", "vec_api_bge_m3", VEC_E5, "baai/bge-m3 by API vs the local e5 (vector side)"),
    Contrast("A2", "vec_api_e5_large", VEC_E5, "intfloat/multilingual-e5-large by API vs the local e5 (vector side)"),
    Contrast("A3", "vec_api_qwen3_8b_1024", VEC_E5, "qwen/qwen3-embedding-8b at 1024 by API vs the local e5 (vector side)"),
    Contrast("A4", "hybrid_norm_e5_api_qwen3_rerank_0.6b", BGE_REFERENCE,
             "qwen/qwen3-reranker-0.6b by API vs the local bge reranker"),
    Contrast("A5", "hybrid_norm_e5_api_qwen3_rerank_8b", BGE_REFERENCE,
             "qwen/qwen3-reranker-8b by API vs the local bge reranker"),
)

DEFAULT_DB_PATH = "data/catalog.db"
DEFAULT_INDEXES_DIR = "data/indexes"
DEFAULT_ABLATION_DIR = "data/indexes_ablation"
DEFAULT_QUERIES_PATH = "app/evaluation/test_queries.json"
DEFAULT_JUDGMENTS_PATH = "app/evaluation/relevance_judgments.json"
DEFAULT_START_JUDGMENTS = "app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json"
DEFAULT_OUTPUT_JUDGMENTS = "app/evaluation/relevance_judgments_extended_api_models.json"
DEFAULT_JUDGE_CACHE_DIR = "data/evaluation/llm_judgments"
DEFAULT_CACHE_DIR = "data/evaluation/api_models"
DEFAULT_OUTPUT_JSON = "data/evaluation/results_api_models.json"
DEFAULT_OUTPUT_MD = "data/evaluation/API_MODELS.md"
DEFAULT_V07_RESULTS = "data/evaluation/results_llm_recommender.json"
DEFAULT_V06_RESULTS = "data/evaluation/results_retrieval_ablation_z-ai-glm-5.3-flash.json"
RESPONSES_FILE = "responses.jsonl"
JUDGING_RUNS_FILE = "judging_runs.jsonl"
TIMINGS_FILE = "local_timings.json"
MODEL_INFO_FILE = "openrouter_models.json"
OPENROUTER_API = "https://openrouter.ai/api/v1"
DEFAULT_WORKERS = 4


# =============================================================================
# Cache of API responses
# =============================================================================


class MissingApiResponseError(RuntimeError):
    """--offline: a response the report needs is not in the cache."""


def request_key(run: str, path: str, body: Mapping[str, Any]) -> str:
    """Hash of a request: run label, endpoint and the canonical JSON body."""
    import hashlib

    payload = json.dumps({"run": run, "path": path, "body": body}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compact_response(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """The response as stored: rerank results lose the echoed document text."""
    if path != "rerank":
        return payload
    stored = dict(payload)
    stored["results"] = [{k: v for k, v in item.items() if k != "document"} for item in payload.get("results", [])]
    return stored


class ResponseCache:
    """
    Append-only JSON Lines file of API responses (ok entries) and failed attempts.

    Each line is written, flushed and synced as soon as the response arrives,
    so an interrupted run loses nothing that was paid for. Thread-safe.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._entries: Dict[str, Dict[str, Any]] = {}
        self.failures: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path.exists():
            for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping unreadable line %d of %s", number, self.path)
                    continue
                if entry.get("ok", True):
                    self._entries[entry["key"]] = entry
                else:
                    self.failures.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def call_failures(self, arm: str) -> List[Dict[str, Any]]:
        """Calls of an arm that failed after the adapter's retries (recorded by the measurement)."""
        with self._lock:
            return [f for f in self.failures if f.get("kind") == "call_failure" and f.get("arm") == arm]

    def entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entries.values())

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._entries.get(key)

    def add(self, entry: Mapping[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if entry.get("ok", True):
                self._entries[entry["key"]] = dict(entry)
            else:
                self.failures.append(dict(entry))


class SpendLimitError(RuntimeError):
    """The cost of the API calls made by this run passed SPEND_LIMIT_USD."""


class CachingTransport(httpx.BaseTransport):
    """
    httpx transport that answers from the ResponseCache and calls `inner` only on a miss.

    The cache key is the run label (set by the caller before each phase), the
    endpoint and the request body. A new 200 response is stored with the
    latency of its attempt and of the whole call (first attempt to success,
    retry waits included) and the number of attempts; a failed attempt is
    stored as a failure record. With inner=None (--offline) a miss raises
    MissingApiResponseError. Errors that SearchService would swallow (a
    missing response, the spend limit) are also kept in `fatal`, which the
    measurement checks after every search.
    """

    def __init__(self, cache: ResponseCache, inner: Optional[httpx.BaseTransport],
                 spend_limit_usd: float = SPEND_LIMIT_USD) -> None:
        self.cache = cache
        self.inner = inner
        self.spend_limit_usd = spend_limit_usd
        self.spent_usd = 0.0
        self.run = "unset"
        self.history: List[Dict[str, Any]] = []  # entries returned, hits included, in order
        self.calls_made = 0
        self.fatal: Optional[BaseException] = None
        self._pending: Dict[str, Tuple[float, int]] = {}

    def _fail(self, key: str, path: str, body: Mapping[str, Any], error: str, latency_s: float) -> None:
        self.cache.add({"ok": False, "kind": "attempt", "key": key, "run": self.run, "path": path,
                        "model": body.get("model"), "error": error, "latency_s": latency_s,
                        "date": datetime.now(UTC).isoformat(timespec="seconds")})

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.rstrip("/").rsplit("/", 1)[-1]
        body = json.loads(request.content.decode("utf-8"))
        key = request_key(self.run, path, body)
        hit = self.cache.get(key)
        if hit is not None:
            self.history.append(hit)
            return httpx.Response(200, json=hit["response"], request=request)
        if self.inner is None:
            self.fatal = MissingApiResponseError(
                f"No cached {path} response for run '{self.run}' (model {body.get('model')}, "
                f"{len(body.get('input') or body.get('documents') or [])} inputs) in {self.cache.path}. Run "
                f"`python -m app.evaluation.api_models_evaluation` without --offline, or check that the catalog, "
                f"indexes and local models are the ones the published run used."
            )
            raise self.fatal
        first, attempts = self._pending.get(key, (time.perf_counter(), 0))
        attempts += 1
        started = time.perf_counter()
        try:
            response = self.inner.handle_request(request)
            content = response.read()
        except Exception as error:
            self._pending[key] = (first, attempts)
            self._fail(key, path, body, type(error).__name__, time.perf_counter() - started)
            raise
        ended = time.perf_counter()
        try:
            payload = json.loads(content)
        except ValueError:
            payload = None
        if response.status_code != 200 or not isinstance(payload, dict) or "error" in payload:
            self._pending[key] = (first, attempts)
            self._fail(key, path, body, f"HTTP {response.status_code}", ended - started)
            headers = {k: v for k, v in response.headers.items() if k.lower() in ("retry-after", "content-type")}
            return httpx.Response(response.status_code, headers=headers, content=content, request=request)
        self._pending.pop(key, None)
        entry = {
            "ok": True, "key": key, "run": self.run, "path": path, "model": body.get("model"),
            "dimensions": body.get("dimensions"), "provider_pin": (body.get("provider") or {}).get("order"),
            "inputs": len(body.get("input") or body.get("documents") or []),
            "query": body.get("query"),
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "latency_s": ended - started, "call_latency_s": ended - first, "attempts": attempts,
            "response": compact_response(path, payload),
        }
        self.cache.add(entry)
        self.history.append(entry)
        self.calls_made += 1
        self.spent_usd += float((payload.get("usage") or {}).get("cost") or 0.0)
        if self.spent_usd > self.spend_limit_usd:
            self.fatal = SpendLimitError(
                f"API spend of this run is {self.spent_usd:.4f} USD, over the {self.spend_limit_usd} USD limit")
            raise self.fatal
        return httpx.Response(200, json=entry["response"], request=request)

    def check(self) -> None:
        """Re-raise an error that a caller may have swallowed (missing response, spend limit)."""
        if self.fatal is not None:
            raise self.fatal


def call_record(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """What the report keeps of one cached call."""
    usage = entry["response"].get("usage") or {}
    tokens = usage.get("prompt_tokens", usage.get("total_tokens"))
    return {"ok": True, "latency_s": entry["call_latency_s"], "attempts": entry["attempts"],
            "tokens": tokens, "cost_usd": usage.get("cost"), "provider": entry["response"].get("provider"),
            "inputs": entry["inputs"], "date": entry["date"]}


FAILED_CALL = {"ok": False}


def record_call_failure(transport: CachingTransport, arm: str, item: str, error: BaseException) -> None:
    """
    Store a call that failed after the adapter's retries.

    The failure counts against the arm in the selection rule (a candidate
    with a failed call is excluded), also when the measurement's second
    attempt at the same call succeeds. It is stored in the cache, so an
    --offline run counts it although it replays the successful attempt.
    """
    if transport.inner is None:  # offline: the live run recorded it already
        return
    transport.cache.add({"ok": False, "kind": "call_failure", "arm": arm, "run": transport.run, "item": item,
                         "error": f"{type(error).__name__}: {error}"[:300],
                         "date": datetime.now(UTC).isoformat(timespec="seconds")})


def failure_summary(transport: CachingTransport, arm: str) -> Tuple[int, List[Dict[str, Any]]]:
    """Failed calls of an arm (all runs, from the cache) and their records."""
    failures = transport.cache.call_failures(arm)
    return len(failures), [{k: f.get(k) for k in ("run", "item", "error", "date")} for f in failures]


# =============================================================================
# Local models, timed
# =============================================================================


class CallClock:
    """Seconds spent in timed calls since the last take(), per kind."""

    def __init__(self) -> None:
        self._seconds: Dict[str, float] = defaultdict(float)

    def add(self, kind: str, seconds: float) -> None:
        self._seconds[kind] += seconds

    def take(self) -> Dict[str, float]:
        taken = dict(self._seconds)
        self._seconds.clear()
        return taken


class TimedEmbeddings:
    """An EmbeddingsStore whose query encoding is timed (everything else is delegated)."""

    def __init__(self, inner: Any, clock: CallClock) -> None:
        self._inner = inner
        self._clock = clock

    def generate_query_embedding(self, text: str) -> List[float]:
        started = time.perf_counter()
        vector = self._inner.generate_query_embedding(text)
        self._clock.add("encode", time.perf_counter() - started)
        return vector

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TimedReranker(Reranker):
    """A Reranker whose rerank call is timed."""

    def __init__(self, inner: Reranker, clock: CallClock) -> None:
        self._inner = inner
        self._clock = clock

    def rerank(self, query_text: str, results: List[SearchResult]) -> List[SearchResult]:
        started = time.perf_counter()
        reranked = self._inner.rerank(query_text, results)
        self._clock.add("rerank", time.perf_counter() - started)
        return reranked

    def get_model_name(self) -> str:
        return self._inner.get_model_name()

    def is_ready(self) -> bool:
        return self._inner.is_ready()


class TimingStack(ModelStack):
    """ModelStack whose query encoder and reranker are timed separately from the search."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.clock = CallClock()

    def searcher_for(self, spec: SystemSpec, reranker: Optional[Reranker] = None):
        self._reranker = None  # one reranker at a time
        lexical = self._lexical(spec.lexical) if spec.lexical else None
        if spec.embedding:
            self._load_embedding_model(spec.embedding)
        if reranker is None and spec.reranker:
            reranker = self._load_reranker(spec.reranker)
        embeddings = TimedEmbeddings(self._embeddings, self.clock) if spec.embedding else None
        timed = TimedReranker(reranker, self.clock) if reranker is not None else None
        return make_searcher(spec, lexical=lexical, vector=self._vector, embeddings=embeddings, reranker=timed)

    def lexical(self, tokenizer: str):
        """The in-memory BM25 repository of a tokenizer (built once)."""
        return self._lexical(tokenizer)


def run_local(search: Callable[[str], Sequence[SearchResult]], queries: Sequence[TestQuery],
              clock: CallClock) -> Tuple[Dict[str, List[BookId]], Dict[str, Dict[str, float]], Dict[str, float]]:
    """Warm-up search, then every query: ranked ids and seconds (total, encode, rerank) per query."""
    clock.take()
    started = time.perf_counter()
    search(queries[0].text)
    warmup = {"total_s": time.perf_counter() - started, **{f"{k}_s": v for k, v in clock.take().items()}}
    runs: Dict[str, List[BookId]] = {}
    timings: Dict[str, Dict[str, float]] = {}
    for query in queries:
        clock.take()
        started = time.perf_counter()
        results = search(query.text)
        total = time.perf_counter() - started
        timings[query.query_id] = {"total_s": total, **{f"{k}_s": v for k, v in clock.take().items()}}
        runs[query.query_id] = [result.book.id for result in list(results)[:DEPTH]]
    return runs, timings, warmup


# =============================================================================
# API arms
# =============================================================================


def _new_calls(transport: CachingTransport, start: int, path: str) -> List[Dict[str, Any]]:
    return [entry for entry in transport.history[start:] if entry["path"] == path]


def embed_documents(encoder: OpenRouterEmbeddingEncoder, transport: CachingTransport, books: Sequence[Book]):
    """Embed the catalog (run label "documents"); return the FAISS store, vector repository and calls."""
    from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
    from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository

    store = EmbeddingsStoreFaiss(encoder=encoder)
    transport.run = "documents"
    start = len(transport.history)
    vectors = store.generate_embeddings_batch([book.get_searchable_text() for book in books])
    transport.check()
    calls = [call_record(entry) for entry in _new_calls(transport, start, "embeddings")]
    store.store_embeddings_batch([book.id for book in books], vectors)
    store.build_index()
    return store, FaissVectorSearchRepository(store, list(books)), calls


def max_cosine_difference(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> Optional[float]:
    """Largest 1 - cos(a, b) over the queries embedded in both runs."""
    values = []
    for query_id, a in first.items():
        b = second.get(query_id)
        if b is not None:
            values.append(1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    return max(values) if values else None


def measure_embedding_arm(arm: EmbeddingArm, http: OpenRouterHttp, transport: CachingTransport,
                          books: Sequence[Book], queries: Sequence[TestQuery],
                          log: Callable[[str], None]) -> Dict[str, Any]:
    """Embed the documents, then the queries twice (after a warm-up call), searching FAISS each time."""
    encoder = arm.build(http)
    record: Dict[str, Any] = {
        "name": arm.name, "kind": "api_embedding", "model": arm.model, "dimension": arm.dimension,
        "primary": arm.primary, "identity": encoder.get_model_name(), "provider_pin": EMBEDDING_PIN,
        "description": arm.description, "runs": {}, "query_calls": {}, "failed_calls": 0,
    }
    documents = None
    for attempt in (1, 2):  # a failed call counts against the arm; the second attempt gives its quality
        try:
            documents = embed_documents(encoder, transport, books)
            break
        except (OpenRouterError, ValueError) as error:
            transport.check()
            log(f"  {arm.name}: document embedding failed (attempt {attempt}): {error}")
            record_call_failure(transport, arm.name, "documents", error)
    if documents is None:
        record.update(document_calls=[], runs={run: {q.query_id: [] for q in queries} for run in RUNS},
                      query_calls={run: {q.query_id: dict(FAILED_CALL) for q in queries} for run in RUNS})
        record["failed_calls"], record["failures"] = failure_summary(transport, arm.name)
        return record
    store, vector, record["document_calls"] = documents

    def embed_query(text: str, item: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        for _ in (1, 2):
            start = len(transport.history)
            try:
                values = store.generate_query_embedding(text)
            except (OpenRouterError, ValueError) as error:
                transport.check()
                record_call_failure(transport, arm.name, item, error)
                continue
            transport.check()
            (entry,) = _new_calls(transport, start, "embeddings")
            return values, call_record(entry)
        return None

    transport.run = "warmup"
    warm = embed_query(queries[0].text, "warmup")
    record["warmup_call"] = warm[1] if warm else dict(FAILED_CALL)
    vectors: Dict[str, Dict[str, np.ndarray]] = {}
    for run in RUNS:
        transport.run = run
        record["runs"][run], record["query_calls"][run], vectors[run] = {}, {}, {}
        for query in queries:
            answer = embed_query(query.text, query.query_id)
            if answer is None:
                record["query_calls"][run][query.query_id] = dict(FAILED_CALL)
                record["runs"][run][query.query_id] = []
                continue
            vector_values, record["query_calls"][run][query.query_id] = answer
            vectors[run][query.query_id] = np.asarray(vector_values, dtype=float)
            results = vector.search(vector_values, max_results=DEPTH)
            record["runs"][run][query.query_id] = [result.book.id for result in results[:DEPTH]]
    first, second = record["runs"]["run1"], record["runs"]["run2"]
    record["stability"] = {
        "max_cosine_difference": max_cosine_difference(vectors["run1"], vectors["run2"]),
        "changed_top10_lists": sum(1 for q in first if first[q] != second.get(q)),
        "identical_vectors": sum(1 for q, a in vectors["run1"].items()
                                 if q in vectors["run2"] and np.array_equal(a, vectors["run2"][q])),
        "queries": len(first),
    }
    record["failed_calls"], record["failures"] = failure_summary(transport, arm.name)
    log(f"  {arm.name}: {len(record['document_calls'])} document calls, "
        f"{sum(1 for c in record['query_calls']['run1'].values() if c['ok'])} query calls in run 1, "
        f"{record['failed_calls']} failed")
    return record


def measure_rerank_arm(arm: RerankArm, http: OpenRouterHttp, transport: CachingTransport, stack: TimingStack,
                       queries: Sequence[TestQuery], fused: Mapping[str, Sequence[BookId]],
                       log: Callable[[str], None]) -> Dict[str, Any]:
    """Re-rank the top 30 of hybrid_norm_e5 with the API reranker, twice, after a warm-up search."""
    reranker = arm.build(http)
    search = stack.searcher_for(LOCAL_SPECS[CANDIDATES], reranker=reranker)
    record: Dict[str, Any] = {
        "name": arm.name, "kind": "api_rerank", "model": arm.model, "provider_pin": RERANK_PIN,
        "description": arm.description, "document_chars": DEFAULT_MAX_DOCUMENT_CHARS,
        "runs": {}, "query_calls": {}, "failed_calls": 0,
    }

    def one(text: str, item: str) -> Tuple[Optional[List[BookId]], Optional[Dict[str, Any]]]:
        for _ in (1, 2):  # a failed call counts against the arm; the second attempt gives its quality
            start = len(transport.history)
            try:
                results = search(text)
            except (RerankerFailedError, OpenRouterError) as error:
                transport.check()
                record_call_failure(transport, arm.name, item, error)
                continue
            transport.check()
            (entry,) = _new_calls(transport, start, "rerank")
            return [result.book.id for result in list(results)[:DEPTH]], call_record(entry)
        return None, None

    transport.run = "warmup"
    _, record["warmup_call"] = one(queries[0].text, "warmup")
    for run in RUNS:
        transport.run = run
        record["runs"][run], record["query_calls"][run] = {}, {}
        for query in queries:
            ranked, call = one(query.text, query.query_id)
            if call is None:  # SearchService kept the fused order
                ranked, call = list(fused[query.query_id]), dict(FAILED_CALL)
            record["runs"][run][query.query_id] = ranked
            record["query_calls"][run][query.query_id] = call
    first, second = record["runs"]["run1"], record["runs"]["run2"]
    record["stability"] = {"identical_top10_lists": sum(1 for q in first if first[q] == second.get(q)),
                           "queries": len(first)}
    record["failed_calls"], record["failures"] = failure_summary(transport, arm.name)
    log(f"  {arm.name}: {record['failed_calls']} failed calls; "
        f"{record['stability']['identical_top10_lists']}/{len(first)} identical lists between runs")
    return record


# =============================================================================
# R1 of V07 (not re-run)
# =============================================================================


def load_r1(path: str, queries: Sequence[TestQuery]) -> Dict[str, Any]:
    """
    R1 (LLM listwise recommender) of V07, run 1: top-10 lists from the results
    file, and the LLM call of each query (latency, tokens, reported cost) from
    the output cache that file names.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = {q: [UUID(b) for b in ids] for q, ids in document["runs"][R1].items()}
    by_text = {query.text: query.query_id for query in queries}
    calls: Dict[str, Dict[str, Any]] = {}
    cache_file = document["run"]["cache_file"]
    for line in Path(cache_file).read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("system") != R1 or entry.get("run") != 1 or entry.get("query_text") not in by_text:
            continue
        output = entry["output"]
        calls[by_text[entry["query_text"]]] = {
            "ok": output.get("error") is None, "latency_s": output.get("latency_s"),
            "tokens": output.get("input_tokens"), "output_tokens": output.get("output_tokens"),
            "cost_usd": output.get("cost_usd"), "provider": None, "attempts": 1, "date": entry.get("date"),
        }
    outputs = document.get("outputs", {}).get(R1, {}).get("run1", {})
    model = document["models"]["system_llm"]["model"]
    published = document["results"]["systems"][R1]["metrics"][PRIMARY_METRIC]["mean"]
    return {
        "name": R1, "kind": "llm_rerank", "model": model, "provider_pin": None,
        "description": (f"V07 R1: top {RERANK_DEPTH} of {CANDIDATES} re-ordered by one listwise call of "
                        f"`{model}` (run 1, not re-run)"),
        "runs": {"run1": runs}, "query_calls": {"run1": calls},
        "failed_calls": sum(1 for call in calls.values() if not call["ok"])
        + sum(1 for out in outputs.values() if out.get("error")),
        "missing_calls": sorted(set(runs) - set(calls)),
        "published_ndcg_at_10": published, "source": {"results": path, "cache": cache_file},
    }


# =============================================================================
# End-to-end check (descriptive)
# =============================================================================


def measure_e2e(embedding_arm: EmbeddingArm, rerank_arm: RerankArm, http: OpenRouterHttp,
                transport: CachingTransport, stack: TimingStack, books: Sequence[Book],
                queries: Sequence[TestQuery], log: Callable[[str], None]) -> Dict[str, Any]:
    """
    The real SearchService with local BM25 and FAISS, the API query embedding
    and the API reranker. Per query: wall clock, embedding call, rerank call;
    retrieval = wall clock - both calls (BM25, FAISS, fusion, filters).
    """
    store, vector, _ = embed_documents(embedding_arm.build(http), transport, books)  # cached document calls
    reranker = rerank_arm.build(http)
    spec = SystemSpec(E2E, lexical=NORMALIZED, embedding=store.get_model_name())
    search = make_searcher(spec, lexical=stack.lexical(NORMALIZED), vector=vector, embeddings=store,
                           reranker=reranker)
    record: Dict[str, Any] = {
        "name": E2E, "kind": "e2e", "embedding_arm": embedding_arm.name, "rerank_arm": rerank_arm.name,
        "model": f"{embedding_arm.model} @ {embedding_arm.dimension} + {rerank_arm.model}",
        "description": (f"SearchService: BM25 (normalizing tokenizer, local) + FAISS over `{embedding_arm.model}` "
                        f"API vectors ({embedding_arm.dimension} d) + RRF + `{rerank_arm.model}` by API over the "
                        f"top {RERANK_DEPTH}"),
        "runs": {"run1": {}}, "query_calls": {"run1": {}}, "timings": {}, "failed_calls": 0,
    }
    transport.run = "e2e-warmup"
    try:
        search(queries[0].text)
    except (RerankerFailedError, OpenRouterError) as error:
        transport.check()
        record_call_failure(transport, E2E, "warmup", error)
    transport.run = "e2e"
    for query in queries:
        start = len(transport.history)
        started = time.perf_counter()
        try:
            results = search(query.text)
        except (RerankerFailedError, OpenRouterError) as error:
            transport.check()
            record_call_failure(transport, E2E, query.query_id, error)
            record["runs"]["run1"][query.query_id] = []
            continue
        total = time.perf_counter() - started
        transport.check()
        (embedding,) = _new_calls(transport, start, "embeddings")
        (rerank,) = _new_calls(transport, start, "rerank")
        record["runs"]["run1"][query.query_id] = [result.book.id for result in list(results)[:DEPTH]]
        record["query_calls"]["run1"][query.query_id] = {"embedding": call_record(embedding),
                                                         "rerank": call_record(rerank)}
        record["timings"][query.query_id] = {"total_s": total}
    record["failed_calls"], record["failures"] = failure_summary(transport, E2E)
    log(f"  {E2E}: {record['failed_calls']} failed calls")
    return record


# =============================================================================
# Judgments
# =============================================================================


def complete_judgments(runs: Mapping[str, Mapping[str, Sequence[BookId]]], queries: Sequence[TestQuery],
                       books: Sequence[Book], original_raw: Mapping[str, Any], start_raw: Mapping[str, Any],
                       *, live: bool, workers: int, judge_cache_dir: str, output_path: str,
                       judging_log: Path, log: Callable[[str], None]
                       ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Grade the top-10 holes of `runs` (live) or require that there are none (--offline).

    The judge is the configured one (JUDGE_*), which must be the start file's
    judge; the merged judgments are written to `output_path` only, and the
    facts of the judging pass are appended to `judging_log` at once.
    """
    holes = find_unjudged(runs, resolve_grades(start_raw, books), DEPTH)
    if not holes:
        return dict(start_raw), None
    if not live:
        raise UnjudgedResultsError(
            f"{len(holes)} top-10 pairs have no judgment in the judgments file; run without --offline to grade them")
    from app.evaluation.llm_recommender_evaluation import fill_judgments
    from app.evaluation.relevance_judge import JudgmentCache, _build_judge
    from app.infrastructure.llm.relevance_judge import LangChainRelevanceJudge

    settings, judge = _build_judge()
    judges = [judge] + [LangChainRelevanceJudge.from_settings(settings) for _ in range(workers - 1)]
    cache = JudgmentCache.for_judge(judge_cache_dir, judge.model_name, judge.prompt_version)
    merged, judging = fill_judgments(runs, queries, books, original_raw, start_raw, judges, cache, log=log)
    final_raw = {entry["query_id"]: entry["judgments"] for entry in merged}
    Path(output_path).write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    judging["cache_file"] = str(cache.path)
    judging_log.parent.mkdir(parents=True, exist_ok=True)
    with judging_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(judging, ensure_ascii=False, default=str) + "\n")
    log(f"  judged {judging['judged_in_this_run']} new pairs ({judging['reported_cost_usd'] or 0:.4f} USD); "
        f"wrote {output_path}")
    return final_raw, judging


def write_judgments_file(raw: Mapping[str, Sequence[Mapping[str, Any]]], path: str) -> None:
    """Write judgments in the file format (list of {query_id, judgments})."""
    merged = [{"query_id": query_id, "judgments": list(items)} for query_id, items in raw.items()]
    Path(path).write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# =============================================================================
# Analysis (no network from here on)
# =============================================================================


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def _sum(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def call_summary(calls: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Latency (ms), tokens and cost of a set of calls (failed calls counted apart)."""
    ok = [call for call in calls if call.get("ok")]
    latencies = [call["latency_s"] * 1000 for call in ok if call.get("latency_s") is not None]
    return {
        "calls": len(calls),
        "failed": len(calls) - len(ok),
        "retried": sum(1 for call in ok if (call.get("attempts") or 1) > 1),
        "median_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": _percentile(latencies, 95),
        "tokens_mean": _mean([call.get("tokens") for call in ok]),
        "cost_mean_usd": _mean([call.get("cost_usd") for call in ok]),
        "cost_total_usd": _sum([call.get("cost_usd") for call in ok]),
        "providers": sorted({str(call["provider"]) for call in ok if call.get("provider")}),
    }


def document_summary(calls: Sequence[Mapping[str, Any]], num_books: int) -> Dict[str, Any]:
    """Time, tokens and cost of embedding the catalog, and the extrapolated one-off cost."""
    seconds = _sum([call.get("latency_s") for call in calls])
    cost = _sum([call.get("cost_usd") for call in calls])
    tokens = _sum([call.get("tokens") for call in calls])
    per_book_cost = cost / num_books if cost is not None else None
    per_book_s = seconds / num_books if seconds is not None else None
    return {
        "calls": len(calls), "seconds": seconds, "tokens": tokens, "cost_usd": cost,
        "tokens_per_book": tokens / num_books if tokens is not None else None,
        "extrapolated": {
            str(n): {"cost_usd": per_book_cost * n if per_book_cost is not None else None,
                     "sequential_hours": per_book_s * n / 3600 if per_book_s is not None else None}
            for n in EXTRAPOLATION_SIZES
        },
    }


def storage_gb(dimension: int, works: int = STORAGE_WORKS) -> Dict[str, float]:
    """Vector data per `works` works, decimal GB: float32 (4 bytes) and pgvector halfvec (2 bytes)."""
    return {"float32_gb": dimension * 4 * works / 1e9, "halfvec_gb": dimension * 2 * works / 1e9}


def speed_and_cost(arm: Mapping[str, Any], num_books: int) -> Dict[str, Any]:
    """The speed and cost row of an arm."""
    kind = arm["kind"]
    if kind in ("api_embedding", "api_rerank", "llm_rerank"):
        row: Dict[str, Any] = {"query_call": call_summary(list(arm["query_calls"]["run1"].values()))}
        if kind == "api_embedding":
            row["documents"] = document_summary(arm.get("document_calls", []), num_books)
            row["storage"] = storage_gb(arm["dimension"])
        if kind != "llm_rerank":
            row["cost_per_1000_searches_usd"] = (row["query_call"]["cost_mean_usd"] * 1000
                                                 if row["query_call"]["cost_mean_usd"] is not None else None)
        return row
    if kind == "e2e":
        timings = arm["timings"]
        parts = {"embedding": [], "rerank": [], "retrieval": [], "total": []}
        for query_id, calls in arm["query_calls"]["run1"].items():
            total = timings[query_id]["total_s"]
            embedding, rerank = calls["embedding"]["latency_s"], calls["rerank"]["latency_s"]
            parts["embedding"].append(embedding * 1000)
            parts["rerank"].append(rerank * 1000)
            parts["total"].append(total * 1000)
            parts["retrieval"].append((total - embedding - rerank) * 1000)
        costs = [(c["embedding"].get("cost_usd") or 0.0) + (c["rerank"].get("cost_usd") or 0.0)
                 for c in arm["query_calls"]["run1"].values()]
        return {f"{name}_median_ms": statistics.median(values) if values else None for name, values in parts.items()} | {
            f"{name}_p95_ms": _percentile(values, 95) for name, values in parts.items()} | {
            "cost_mean_usd": _mean(costs)}
    # local references: the timed call of the stage
    stage = "encode_s" if kind == "local_embedding" else "rerank_s"
    latencies = [t[stage] * 1000 for t in arm["timings"].values() if stage in t]
    row = {"query_call": {"calls": len(latencies), "failed": 0, "retried": 0,
                          "median_ms": statistics.median(latencies) if latencies else None,
                          "p95_ms": _percentile(latencies, 95), "tokens_mean": None, "cost_mean_usd": 0.0,
                          "providers": [f"local ({arm.get('device')})"]},
           "total_median_ms": statistics.median([t["total_s"] * 1000 for t in arm["timings"].values()])}
    if kind == "local_rerank":
        row["cost_per_1000_searches_usd"] = 0.0
    if kind == "local_embedding":
        seconds = arm.get("index_build_seconds")
        row["documents"] = {"seconds": seconds, "cost_usd": 0.0, "source": arm.get("index_source"),
                            "extrapolated": {str(n): {"cost_usd": 0.0,
                                                      "sequential_hours": seconds / num_books * n / 3600
                                                      if seconds else None} for n in EXTRAPOLATION_SIZES}}
        row["storage"] = storage_gb(arm["dimension"])
    return row


def selection_rows(arms: Mapping[str, Mapping[str, Any]], names: Sequence[str], metrics: Mapping[str, Any],
                   speed: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Input rows of the speed-quality rule: nDCG@10, median query-call latency, cost, failures."""
    rows = {}
    for name in names:
        call = speed[name]["query_call"]
        rows[name] = {
            "quality": metrics[name]["metrics"][PRIMARY_METRIC]["mean"],
            "median_latency_s": call["median_ms"] / 1000 if call["median_ms"] is not None else None,
            "cost_usd": call["cost_mean_usd"],
            "failed_calls": arms[name].get("failed_calls", 0),
        }
    return rows


def select_roles(arms: Mapping[str, Mapping[str, Any]], metrics: Mapping[str, Any],
                 speed: Mapping[str, Any]) -> Dict[str, Any]:
    """The cloud embedding model and the cloud reranker under the speed-quality rule."""
    embedding_names = [arm.name for arm in EMBEDDING_ARMS if arm.primary]
    rerank_names = [arm.name for arm in RERANK_ARMS] + [R1]
    embedding_rows = selection_rows(arms, embedding_names, metrics, speed)
    rerank_rows = selection_rows(arms, rerank_names, metrics, speed)
    embedding = select_by_speed_quality(embedding_rows)
    reranker = select_by_speed_quality(rerank_rows)
    api_only = select_by_speed_quality({n: rerank_rows[n] for n in rerank_names if n != R1})
    return {
        "embedding": {**embedding, "rows": embedding_rows},
        "reranker": {**reranker, "rows": rerank_rows},
        "api_reranker_for_e2e": api_only["winner"],
        "rule": {"tie_margin": TIE_MARGIN, "latency_band": LATENCY_BAND},
    }


def analyse(arms: Mapping[str, Mapping[str, Any]], final_raw: Mapping[str, Any], books: Sequence[Book],
            queries: Sequence[TestQuery], *, n_resamples: int, seed: int) -> Dict[str, Any]:
    """Metrics, contrasts, speed and cost, stability and the decisions, on the final judgments."""
    final = resolve_grades(final_raw, books)
    runs = {name: arm["runs"]["run1"] for name, arm in arms.items()}
    require_judged(runs, final, DEPTH)
    report = compare_systems(runs, final, queries, None, n_resamples=n_resamples, seed=seed)
    not_judged = {name: data["metrics"]["judged_at_10"]["mean"] for name, data in report["systems"].items()
                  if data["metrics"]["judged_at_10"]["mean"] != 1.0}
    if not_judged:
        raise UnjudgedResultsError(f"Judged@10 is not 1.0: {not_judged}")
    contrasts = compare_pairs(report, [(c.name, c.system, c.reference) for c in CONTRASTS],
                              n_resamples=n_resamples, seed=seed)
    speed = {name: speed_and_cost(arm, len(books)) for name, arm in arms.items()}
    systems = {name: {"metrics": data["metrics"], "by_language": data["by_language"]}
               for name, data in report["systems"].items()}
    return {
        "num_queries": report["num_queries"], "languages": report["languages"],
        "systems": systems, "per_query": {n: d["per_query"] for n, d in report["systems"].items()},
        "contrasts": contrasts, "speed": speed,
        "stability": {name: arm["stability"] for name, arm in arms.items() if arm.get("stability")},
        "decisions": select_roles(arms, report["systems"], speed),
    }


def llm_roles(path: str) -> Optional[Dict[str, Any]]:
    """The system LLM and judge under the speed-quality rule, from the saved V10 results."""
    from app.evaluation.llm_model_selection import compute_tables

    try:
        results = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    current = compute_tables(results)["decision_speed_quality"]
    return {"source": path,
            "system_llm": {k: current["system_llm"][k] for k in ("winner", "tied", "excluded", "decided_by",
                                                                 "best_quality")},
            "judge": {k: current["judge"][k] for k in ("winner", "tied", "excluded", "decided_by", "best_quality")},
            "judge_if_failures_anywhere_exclude": current["judge_if_failures_anywhere_exclude"]}


def reproduction_check(arms: Mapping[str, Mapping[str, Any]], path: str = DEFAULT_V06_RESULTS) -> Dict[str, Any]:
    """The local references' top 10 against the V06/V10 run of the same systems."""
    try:
        stored = json.loads(Path(path).read_text(encoding="utf-8")).get("runs", {})
    except FileNotFoundError:
        return {"file": path, "systems": {}}
    out = {}
    for name in (VEC_E5, VEC_QWEN3, CANDIDATES, BGE_REFERENCE):
        if name in stored and name in arms:
            run = {q: [str(b) for b in ids] for q, ids in arms[name]["runs"]["run1"].items()}
            out[name] = {"queries": len(run), "identical": sum(1 for q in run if run[q] == stored[name].get(q))}
    return {"file": path, "systems": out}


# =============================================================================
# Report
# =============================================================================


def _ms(value: Optional[float]) -> str:
    return "n/a" if value is None else (f"{value:,.0f}" if value >= 100 else f"{value:.1f}")


def _usd(value: Optional[float], digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _num(value: Optional[float], digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _metric(systems: Mapping[str, Any], name: str, metric: str = PRIMARY_METRIC) -> str:
    stat = systems[name]["metrics"][metric]
    return _fmt_ci(stat["mean"], stat["ci_low"], stat["ci_high"])


def _decision_text(block: Mapping[str, Any], label: str) -> str:
    if not block.get("winner"):
        return f"No {label} candidate is eligible."
    excluded = "; ".join(f"`{n}`: {r}" for n, r in block["excluded"].items()) or "none"
    return (f"**`{block['winner']}`**. Best quality {_fmt(block['best_quality'])}; tied within {TIE_MARGIN}: "
            f"{', '.join(f'`{n}`' for n in block['tied'])}; within {LATENCY_BAND:.0%} of the fastest tied: "
            f"{', '.join(f'`{n}`' for n in block['within_latency_band'])}. Decided by: {block['decided_by']}. "
            f"Excluded: {excluded}.")


def build_markdown(document: Mapping[str, Any]) -> str:
    """API_MODELS.md from the results document (no network)."""
    arms, a = document["arms"], document["analysis"]
    systems, speed, info = a["systems"], a["speed"], document.get("model_info") or {}
    run = document["run"]
    prices = document.get("prices", {})

    def model_cell(name: str) -> str:
        arm = arms[name]
        return f"`{arm['model']}`"

    emb_names = [VEC_E5, VEC_QWEN3] + [arm.name for arm in EMBEDDING_ARMS]
    emb_quality = _table(
        ["System", "Model", "Dim.", "nDCG@10 [95% CI]", "Recall@10", "MRR", "en", "es",
         "Median ms / query call", "p95 ms", "Failed calls", "Served by"],
        [[f"`{n}`" + ("" if n in (VEC_E5, VEC_QWEN3) or arms[n].get("primary") else " (descriptive)"),
          model_cell(n), str(arms[n]["dimension"]), _metric(systems, n),
          _fmt(systems[n]["metrics"]["recall_at_10"]["mean"]), _fmt(systems[n]["metrics"]["mrr"]["mean"]),
          _fmt(systems[n]["by_language"]["en"][PRIMARY_METRIC]["mean"]),
          _fmt(systems[n]["by_language"]["es"][PRIMARY_METRIC]["mean"]),
          _ms(speed[n]["query_call"]["median_ms"]), _ms(speed[n]["query_call"]["p95_ms"]),
          str(arms[n].get("failed_calls", 0)), ", ".join(speed[n]["query_call"]["providers"])]
         for n in emb_names],
    )
    emb_cost = _table(
        ["System", "Tokens / query", "Cost / query (USD)", "177 docs: s", "177 docs: USD", "Tokens / doc",
         "200,000 works: USD", "1,000,000 works: USD", "1,000,000 works: hours (sequential)",
         "Storage / 1M works, float32 (GB)", "halfvec (GB)"],
        [[f"`{n}`", _num(speed[n]["query_call"]["tokens_mean"], 1), _usd(speed[n]["query_call"]["cost_mean_usd"], 8),
          _num(speed[n]["documents"]["seconds"], 2), _usd(speed[n]["documents"]["cost_usd"]),
          _num(speed[n]["documents"].get("tokens_per_book"), 1),
          _usd(speed[n]["documents"]["extrapolated"]["200000"]["cost_usd"], 3),
          _usd(speed[n]["documents"]["extrapolated"]["1000000"]["cost_usd"], 3),
          _num(speed[n]["documents"]["extrapolated"]["1000000"]["sequential_hours"], 1),
          _num(speed[n]["storage"]["float32_gb"], 2), _num(speed[n]["storage"]["halfvec_gb"], 2)]
         for n in emb_names],
    )
    rr_names = [CANDIDATES, BGE_REFERENCE] + [arm.name for arm in RERANK_ARMS] + [R1]
    rr_rows = []
    for n in rr_names:
        call = speed[n].get("query_call") if n != CANDIDATES else None
        rr_rows.append([
            f"`{n}`" + (" (candidates, no reranker)" if n == CANDIDATES else ""),
            "-" if n == CANDIDATES else model_cell(n), _metric(systems, n),
            _fmt(systems[n]["metrics"]["recall_at_10"]["mean"]), _fmt(systems[n]["metrics"]["mrr"]["mean"]),
            _fmt(systems[n]["by_language"]["en"][PRIMARY_METRIC]["mean"]),
            _fmt(systems[n]["by_language"]["es"][PRIMARY_METRIC]["mean"]),
            _ms(call["median_ms"]) if call else "-", _ms(call["p95_ms"]) if call else "-",
            _num(call["tokens_mean"], 0) if call else "-", _usd(call["cost_mean_usd"]) if call else "-",
            _usd(speed[n].get("cost_per_1000_searches_usd"), 3) if call and n != R1 else
            (_usd(call["cost_mean_usd"] * 1000, 3) if call and call["cost_mean_usd"] is not None else "-"),
            "-" if n == CANDIDATES else str(arms[n].get("failed_calls", 0)),
            ", ".join(call["providers"]) if call and call["providers"] else ("-" if not call else "OpenRouter routing"),
        ])
    rr_table = _table(
        ["System", "Model", "nDCG@10 [95% CI]", "Recall@10", "MRR", "en", "es", "Median ms / rerank call",
         "p95 ms", "Tokens / call", "Cost / call (USD)", "Cost / 1,000 searches (USD)", "Failed calls",
         "Served by"], rr_rows)

    contrast_rows = [[name, f"`{c['system']}`", f"`{c['reference']}`", _fmt(c["mean_system"]),
                      _fmt(c["mean_reference"]), _fmt_ci(c["mean_difference"], c["ci_low"], c["ci_high"], signed=True),
                      _fmt_p(c["p_value"]), _fmt_p(c["p_value_holm"])]
                     for name, c in a["contrasts"].items()]
    contrast_table = _table(["Contrast", "System", "Reference", "nDCG@10 system", "nDCG@10 reference",
                             "Diff. [95% CI]", "p", "p (Holm)"], contrast_rows)

    stability_rows = []
    for arm in EMBEDDING_ARMS:
        s = a["stability"].get(arm.name, {})
        stability_rows.append([f"`{arm.name}`", "two embeddings of each query",
                               f"{s.get('max_cosine_difference', 0):.2e}" if s.get("max_cosine_difference") is not None else "n/a",
                               f"{s.get('identical_vectors', 'n/a')} / {s.get('queries', 'n/a')}",
                               f"{s.get('changed_top10_lists', 'n/a')} / {s.get('queries', 'n/a')}"])
    for arm in RERANK_ARMS:
        s = a["stability"].get(arm.name, {})
        stability_rows.append([f"`{arm.name}`", "two re-rankings of each query", "-", "-",
                               f"{s['queries'] - s['identical_top10_lists']} / {s['queries']}" if s else "n/a"])
    stability_table = _table(["System", "Repeat", "Max 1 - cos between runs", "Identical vectors",
                              "Top-10 lists that changed"], stability_rows)

    failure_lines = "\n".join(
        f"- `{name}`, {f.get('run')} / {f.get('item')} ({(f.get('date') or '')[:16]}): {f.get('error')}"
        for name, arm in arms.items() for f in arm.get("failures", [])
    ) or "- None."
    d = a["decisions"]
    llm = document.get("llm_roles") or {}
    e2e = document.get("e2e") or {}
    e2e_text = "Not run."
    if e2e:
        es = speed.get(E2E, {})
        e2e_text = (
            f"{arms[E2E]['description']}.\n\n"
            + _table(["System", "nDCG@10 [95% CI]", "Recall@10", "MRR", "Median total ms", "p95 total ms",
                      "Embedding call ms", "Retrieval ms (BM25, FAISS, fusion)", "Rerank call ms", "Cost / search (USD)"],
                     [[f"`{E2E}`", _metric(systems, E2E), _fmt(systems[E2E]["metrics"]["recall_at_10"]["mean"]),
                       _fmt(systems[E2E]["metrics"]["mrr"]["mean"]), _ms(es.get("total_median_ms")),
                       _ms(es.get("total_p95_ms")), _ms(es.get("embedding_median_ms")),
                       _ms(es.get("retrieval_median_ms")), _ms(es.get("rerank_median_ms")),
                       _usd(es.get("cost_mean_usd"))],
                      [f"`{BGE_REFERENCE}` (local)", _metric(systems, BGE_REFERENCE),
                       _fmt(systems[BGE_REFERENCE]["metrics"]["recall_at_10"]["mean"]),
                       _fmt(systems[BGE_REFERENCE]["metrics"]["mrr"]["mean"]),
                       _ms(speed[BGE_REFERENCE].get("total_median_ms")), "-",
                       _ms(speed[VEC_E5]["query_call"]["median_ms"]), "-",
                       _ms(speed[BGE_REFERENCE]["query_call"]["median_ms"]), "0"]])
            + f"\n\nMedians of each column are taken separately (they do not add up). Embedding models of the "
              f"stack: `{e2e.get('embedding_arm')}`; reranker: `{e2e.get('rerank_arm')}`"
            + (f" ({e2e['note']})" if e2e.get("note") else "") + ". Local reference row: median of its whole "
              f"search, its e5 query encoding and its bge rerank call, on this machine's GPU."
        )

    verified_rows = []
    for model, entry in (info.get("models") or {}).items():
        endpoints = entry.get("endpoints") or []
        verified_rows.append([
            f"`{model}`", entry.get("kind", ""),
            "; ".join(f"{e.get('provider_name')} ({e.get('quantization') or 'unknown'})" for e in endpoints) or "n/a",
            _usd(entry.get("list_price_per_mtok"), 3) if entry.get("list_price_per_mtok") is not None else "not listed",
            _usd(prices.get(model, {}).get("reported_usd_per_mtok"), 3),
        ])
    verified_table = _table(["Model", "Kind", "Endpoints on OpenRouter (stated precision)",
                             "List price (USD / M tokens)", "Reported cost / tokens (USD / M tokens)"],
                            verified_rows) if verified_rows else "No model information saved."

    judgments = document["judgments"]
    judging_lines = "\n".join(
        f"- Judging run of {r.get('date')}: {r.get('holes')} holes, {r.get('judged_in_this_run')} judged, "
        f"{r.get('failed_in_this_run')} failures, {_num(r.get('seconds_per_judgment'), 2)} s per judgment, "
        f"{_usd(r.get('reported_cost_usd'), 4)} USD; grades 0/1/2/3: "
        f"{' / '.join(str(x) for x in r.get('grade_distribution', []))}."
        + (f" ({r['note']}.)" if r.get("note") else "")
        for r in judgments.get("judging_runs", [])) or "- No pair needed a new grade."
    spend = document["spend"]
    repro = document.get("reproduction_check", {}).get("systems", {})
    repro_text = ", ".join(f"`{n}` {v['identical']}/{v['queries']}" for n, v in repro.items()) or "not available"
    llm_text = ""
    if llm:
        alt = llm.get("judge_if_failures_anywhere_exclude") or {}
        llm_text = (
            f"- System LLM (re-applied to the saved V10 measurements, `{llm['source']}`): "
            f"**`{llm['system_llm']['winner']}`**; tied on citation precision: "
            f"{', '.join(f'`{m}`' for m in llm['system_llm']['tied'])}; decided by: {llm['system_llm']['decided_by']}.\n"
            f"- Judge: **`{llm['judge']['winner']}`**; tied on kappa: "
            f"{', '.join(f'`{m}`' for m in llm['judge']['tied'])}; decided by: {llm['judge']['decided_by']}."
            + (f" (If a failed call in any V10 task excluded a model from every role, the judge would be "
               f"`{alt.get('winner')}`.)" if alt.get("winner") and alt.get("winner") != llm["judge"]["winner"] else "")
            + "\n  Details: `data/evaluation/LLM_MODEL_SELECTION.md`."
        )

    return f"""# Embeddings and re-ranking by API versus the local models (V16)

Generated by `{run['command']}` on {run['date']} (git `{run.get('git_commit') or 'unknown'}`{' with uncommitted changes' if run.get('git_worktree_dirty') else ''}).
{document['num_queries']} test queries ({', '.join(f'{k} {v}' for k, v in a['languages'].items())}), catalog of
{run['num_books']} books, 10 results per query. API responses made by the run(s) of
{', '.join(sorted(set(run.get('api_call_dates', [])))) or 'n/a'} and read from `{run['cache_file']}`;
local timings measured on {document.get('local_timings_date', 'n/a')} ({run['device']}, {run['hardware']}).

## Why

The cloud profile (`docs/notes/2026-11-11-propuesta-nube-y-mcp.md`, sections 2 and 3) must not load any
model, so its embeddings and re-ranking come from an API. The author's rules: models through OpenRouter,
cheap Chinese models, no GPT; models balanced between speed and quality; measure and test. This report
measures the API models available on OpenRouter against the local models of the evaluated profile and
chooses with a rule that counts speed as well as quality.

## Design (fixed before any result was seen)

- Systems (vector side only for embeddings; exact FAISS index over L2-normalized vectors, document text
  `Book.get_searchable_text()`):
{chr(10).join(f"  - `{n}`: {arms[n]['description']}" for n in emb_names + rr_names if n in arms)}
- Providers pinned with `provider.order` and `allow_fallbacks: false`: embeddings on `{EMBEDDING_PIN}`,
  rerankers on `{RERANK_PIN}`, so the vectors of one arm come from one endpoint and the arms' latencies
  are not confounded by routing.
- Metric: nDCG@10 (gain 2^grade - 1) on judgments with every top-10 result judged (Judged@10 = 1.000 for
  every system in this report).
- Pre-declared contrasts: {'; '.join(f"{c.name} `{c.system}` - `{c.reference}`" for c in CONTRASTS)}. Two-sided
  paired randomisation test (sign flips) on per-query nDCG@10, {document['design']['n_resamples']} resamples,
  seed {document['design']['seed']}; Holm correction across exactly these five; paired percentile bootstrap
  95% CI over queries.
- Speed and cost: one embedding request per query or one rerank request per search, the 44 queries in
  sequence after one warm-up call; latency is the wall clock of the HTTP call (retries included; hosted
  latency includes OpenRouter's routing); tokens and cost are those each response reports. Documents:
  177 texts in requests of at most {DEFAULT_BATCH_SIZE} texts and {DEFAULT_MAX_BATCH_CHARS:,} characters, sequential;
  the one-off cost for 200,000 and 1,000,000 works
  is the cost per document times the number of works (it assumes works of the same text length). Local
  references: the same calls timed in this process ({run['device']}).
- Stability: the 44 queries embedded twice by every API arm; every API reranker run twice.
- Selection rule (`select_by_speed_quality` in `app/evaluation/llm_model_selection.py`): the candidates
  within {TIE_MARGIN} of the best quality are tied; among them the lowest median latency wins; if another
  tied candidate is within {LATENCY_BAND:.0%} of that latency, the cheaper one (cost per call) wins; a
  candidate that failed any call is excluded. Cloud embedding model: nDCG@10 of the vector side, the three
  primary API arms. Cloud reranker: nDCG@10 after re-ranking, the two API rerankers and the LLM listwise
  recommender R1 of V07 (run 1; its latency is the LLM call, its cost the reported cost; not re-run).
- Everything else is descriptive (the descriptive arms, Recall@10, MRR, the en/es columns, stability,
  the end-to-end check).

## What was verified about the APIs (2026-11-11)

- Embeddings: `POST {OPENROUTER_API}/embeddings` with `model`, `input` (list of non-empty strings; an empty
  string is rejected with HTTP 400), `encoding_format` (`base64` used: little-endian float32, decoded
  exactly equal to the JSON numbers), optional `dimensions` and `provider`. The response has
  `data[].embedding` and `data[].index`, `model`, `provider` (the endpoint that served it) and
  `usage.prompt_tokens`, `usage.total_tokens`, `usage.cost` (USD).
- `dimensions`: DeepInfra cuts any model when asked (it returned 512 values for `baai/bge-m3`, which is not
  trained for it), so the encoder sends it only for the Matryoshka models (Qwen3 embeddings). On DeepInfra,
  `qwen/qwen3-embedding-8b` with `dimensions=1024` equals the first 1,024 values of the native vector
  re-normalized (cosine 1.000000), and repeated calls returned identical vectors. Nebius and DeepInfra gave
  different vectors for the same text (cosine 0.997), which is why the provider is pinned and is part of
  the index identity.
- Query/document conventions (model cards): `intfloat/multilingual-e5-large` takes `query: ` / `passage: `
  prefixes (an API cannot add them for the client: the request has no query/document field that the
  providers use); the Qwen3 embeddings take `Instruct: Given a web search query, retrieve relevant passages
  that answer the query\\nQuery:` before queries only (the prompt stored in Qwen3-Embedding-0.6B, used by the
  local `vec_qwen3`); `baai/bge-m3` takes no instruction. `intfloat/multilingual-e5-large` on DeepInfra
  silently truncates inputs to 512 tokens (`usage.prompt_tokens` = 512 for a longer text), as the local
  model does.
- Rerank: `POST {OPENROUTER_API}/rerank` with `model`, `query`, `documents` (strings) and optional `top_n`
  and `provider`; the response has `results[]` with `index` and `relevance_score` sorted best first (each
  echoes its document), `provider`, and `usage.total_tokens` and `usage.cost`. The rerank models list no
  price in the models API; the reported cost was 0.20 USD per million tokens for both. No instruction can be
  passed; whatever template the provider applies to Qwen3-Reranker (the model card's default instruction is
  "Given a web search query, retrieve relevant passages that answer the query") was not verifiable.

{verified_table}

List prices from `{OPENROUTER_API}/embeddings/models` and the endpoints of each model, read
{info.get('fetched_at', 'n/a')}.

## Embedding models (vector side)

{emb_quality}

{emb_cost}

Latency of the local references: their query encoding in this process ({run['device']}); their document
time is the index build time recorded when V06 built their index. Storage: vector data only, without the
index structure (pgvector `halfvec` HNSW indexes up to 4,000 dimensions, so the native 4,096 dimensions of
`qwen/qwen3-embedding-8b` cannot use it).

## Rerankers (all re-rank the same top {RERANK_DEPTH} of `{CANDIDATES}`)

{rr_table}

R1 is taken from V07 (`{arms[R1]['source']['results']}`, run 1, not re-run) and scored on this report's
judgments; V07 published nDCG@10 {_fmt(arms[R1]['published_ndcg_at_10'])} on its own judgments. Its latency is
the LLM call of each query (the call that replaces the rerank call), recorded when V07 made it.

Failed calls (after the adapter's retries; each counts against its arm in the selection rule even when the
measurement's second attempt at the same call succeeded, which is what gives the arm its quality numbers):

{failure_lines}

Failed HTTP attempts recorded, retried ones included: {document['spend'].get('failed_attempts', 0)}.

## Deviations from the design

- The first measurement sent the documents in requests of 32 texts. For `baai/bge-m3` a request of 32 book
  texts (43,098 characters, about 9,150 tokens) was answered HTTP 404 "No endpoints found", because
  OpenRouter compares the tokens of the whole request with the 8,192-token context of the pinned DeepInfra
  endpoint; each half of that request succeeded. The adapter now also caps a request at
  {DEFAULT_MAX_BATCH_CHARS:,} characters, and the documents of every API arm were embedded again with it
  (the query calls, one text each, did not change). The failed calls stay recorded and, applied mechanically,
  exclude `baai/bge-m3` from the choice; they were caused by the client's request size, not by the model or
  the provider being unavailable.
- R1 (V07) is scored on this task's final judgments, like every other system, so its nDCG@10 differs from
  the value V07 published on its own judgments.

## Pre-declared contrasts

{contrast_table}

## Stability (descriptive)

{stability_table}

## Decisions (mechanical application of the rule)

- Cloud embedding model: {_decision_text(d['embedding'], 'embedding')}
- Cloud reranker: {_decision_text(d['reranker'], 'reranker')}
{llm_text}

## End-to-end check (descriptive)

{e2e_text}

## Judgments

`{document['judgments']['path']}`: {judgments['num_final']} judgments ({judgments['num_original']} original,
{judgments['num_llm']} by {', '.join(f'`{m}`' for m in judgments['llm_judge_models'])}); this task added
{judgments['added_by_this_task']} LLM grades (0/1/2/3: {' / '.join(str(x) for x in judgments['added_grade_distribution'])})
to the {judgments['num_start']} of `{judgments['start_path']}` (read only). The judge saw the stored query text.
Grades of pairs that were already in the judge cache (graded earlier by the same judge and prompt) are reused,
not repeated; they are the difference between the grades added and the pairs judged below.

{judging_lines}

Reproduction of the local references against `{document.get('reproduction_check', {}).get('file')}` (identical
top-10 lists): {repro_text}.

## Spend

- API calls (embeddings and rerank, cost reported per response, all runs in the cache): {_usd(spend['api_usd'], 4)} USD
  ({spend['api_calls']} calls; embeddings {_usd(spend['embedding_usd'], 4)}, rerank {_usd(spend['rerank_usd'], 4)}).
- Judge: {_usd(spend['judge_usd'], 4)} USD.
- Total: {_usd(spend['total_usd'], 4)} USD.{(' API key usage counter over the last live run: ' + _usd(spend['key_usage_delta_usd'], 4) + ' USD.') if spend.get('key_usage_delta_usd') is not None else ''}

## Limits

- 44 queries and 177 books: nDCG@10 differences below about 0.08 are hard to detect, and the catalog of the
  cloud profile is much larger.
- The added grades come from an LLM judge (`z-ai/glm-5.3-flash`), not from people.
- Hosted latency depends on the provider's load and on OpenRouter's routing at the time of the run; each
  arm was measured once (plus the stability repeat), at different minutes.
- Local latencies were measured on a laptop GPU shared with other work; other agents were running on the
  same machine.
- Prices are those of 2026-11-11 and change often.

## Reproduce

    python -m app.evaluation.api_models_evaluation
    python -m app.evaluation.api_models_evaluation --offline

The first command calls the API and the judge only for what is not cached; the second calls nothing and
fails if any response or judgment is missing. Both load the local models (e5, Qwen3-0.6B, bge) to rebuild
the candidates and the local runs; the local timings come from `{run['timings_file']}`.
"""


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Embeddings and re-ranking by API versus the local models (V16)")
    parser.add_argument("--offline", action="store_true",
                        help="Call no API and no judge: recompute everything from the caches; fail if anything is missing")
    parser.add_argument("--render", action="store_true",
                        help="Only re-render the Markdown report from the saved results JSON (no model, no network)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Judge requests in flight")
    parser.add_argument("--device", default=None, help="Torch device of the local models (default: cuda, mps or cpu)")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--indexes-dir", default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--ablation-dir", default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--queries-path", default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--judgments-path", default=DEFAULT_JUDGMENTS_PATH, help="Original judgments (read only)")
    parser.add_argument("--start-judgments-path", default=DEFAULT_START_JUDGMENTS,
                        help="Extended judgments to start from (read only)")
    parser.add_argument("--output-judgments-path", default=DEFAULT_OUTPUT_JUDGMENTS,
                        help="Merged judgments of this task (read if it exists, written)")
    parser.add_argument("--judge-cache-dir", default=DEFAULT_JUDGE_CACHE_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="API responses and local timings")
    parser.add_argument("--v07-results", default=DEFAULT_V07_RESULTS)
    parser.add_argument("--llm-selection-results", default="data/evaluation/results_llm_model_selection.json")
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    return parser


def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _load_json(path: Path | str) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def fetch_model_info() -> Dict[str, Any]:
    """Endpoints, stated precision and list prices of the measured models (public API, no key)."""
    models: Dict[str, Any] = {}
    listed = {entry["id"]: entry for entry in
              httpx.get(f"{OPENROUTER_API}/embeddings/models", timeout=60).json()["data"]}
    for model, kind in [(m, "embedding") for m in sorted({arm.model for arm in EMBEDDING_ARMS})] + \
                       [(arm.model, "rerank") for arm in RERANK_ARMS]:
        endpoints = httpx.get(f"{OPENROUTER_API}/models/{model}/endpoints", timeout=60).json()["data"]["endpoints"]
        price = (listed.get(model) or {}).get("pricing", {}).get("prompt")
        models[model] = {
            "kind": kind,
            "list_price_per_mtok": float(price) * 1e6 if price is not None and float(price) > 0 else None,
            "endpoints": [{k: e.get(k) for k in ("provider_name", "tag", "quantization", "context_length", "pricing")}
                          for e in endpoints],
        }
    return {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds"), "models": models}


def reported_prices(cache: ResponseCache) -> Dict[str, Dict[str, Optional[float]]]:
    """USD per million tokens actually charged, per model: reported cost / reported tokens."""
    totals: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    for entry in cache.entries():
        usage = entry["response"].get("usage") or {}
        tokens = usage.get("prompt_tokens", usage.get("total_tokens"))
        if usage.get("cost") is not None and tokens:
            totals[entry["model"]][0] += float(usage["cost"])
            totals[entry["model"]][1] += float(tokens)
    return {model: {"reported_usd_per_mtok": cost / tokens * 1e6 if tokens else None, "tokens": tokens}
            for model, (cost, tokens) in totals.items()}


def read_judging_runs(path: Path) -> List[Dict[str, Any]]:
    """Every judging pass of this task, in order (judging_runs.jsonl)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def judgments_block(final_raw: Mapping[str, Sequence[Mapping[str, Any]]], start_raw: Mapping[str, Any],
                    judging_runs: Sequence[Mapping[str, Any]], path: str, start_path: str) -> Dict[str, Any]:
    """Counts of the final judgments, the LLM grades this task added and its judging passes."""
    def llm_keys(raw: Mapping[str, Any]) -> set:
        return {(q, i["source"], i["source_id"]) for q, items in raw.items() for i in items if i.get("judge") == "llm"}

    added = llm_keys(final_raw) - llm_keys(start_raw)
    grade = {(q, i["source"], i["source_id"]): i["relevance"] for q, items in final_raw.items() for i in items}
    runs = [dict(r) for r in judging_runs]
    return {
        "path": path, "start_path": start_path,
        "num_final": sum(len(items) for items in final_raw.values()),
        "num_start": sum(len(items) for items in start_raw.values()),
        "num_original": sum(1 for items in final_raw.values() for i in items if i.get("judge") != "llm"),
        "num_llm": len(llm_keys(final_raw)),
        "llm_judge_models": sorted({str(i.get("model")) for items in final_raw.values() for i in items
                                    if i.get("judge") == "llm"}),
        "added_by_this_task": len(added),
        "added_grade_distribution": [sum(1 for key in added if grade[key] == g) for g in GRADES],
        "judging_runs": runs,
        "judge_cost_usd": _sum([r.get("reported_cost_usd") for r in runs]),
    }


def serializable_arm(arm: Mapping[str, Any]) -> Dict[str, Any]:
    """An arm with book ids as strings."""
    out = dict(arm)
    out["runs"] = {run: {q: [str(b) for b in ids] for q, ids in lists.items()} for run, lists in arm["runs"].items()}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Measure every arm, complete the judgments, select, check end to end and write the reports."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.render:
        document = _load_json(args.output_json)
        if not document:
            print(f"No saved results in {args.output_json}")
            return 1
        _write(args.output_md, build_markdown(document))
        print(f"Saved {args.output_md}")
        return 0
    live = not args.offline

    from app.evaluation.retrieval_ablation import _hardware, _versions, resolve_device

    device = args.device or resolve_device()  # imports torch before faiss (macOS)
    from dotenv import load_dotenv

    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

    def log(message: str) -> None:
        print(message, flush=True)

    load_dotenv(Path(".env"))
    started = datetime.now(UTC)
    books = SqliteBookCatalogRepository(Path(args.db_path)).get_all()
    queries = load_test_queries(args.queries_path)
    original_raw = load_relevance_judgments(args.judgments_path)
    output_exists = Path(args.output_judgments_path).exists()
    start_raw = load_relevance_judgments(args.start_judgments_path)
    current_raw = load_relevance_judgments(args.output_judgments_path) if output_exists else start_raw
    previous = _load_json(args.output_json)
    cache_dir = Path(args.cache_dir)
    cache = ResponseCache(cache_dir / RESPONSES_FILE)
    timings_path = cache_dir / TIMINGS_FILE
    saved_timings = _load_json(timings_path)
    if not live and not saved_timings:
        log(f"Stopped: {timings_path} is missing; run once without --offline")
        return 1
    if not live and not output_exists:
        log(f"Stopped: {args.output_judgments_path} is missing; run once without --offline")
        return 1

    key_before = None
    if live:
        from app.evaluation.llm_model_selection import key_usage_usd
        from app.infrastructure.config.settings import openrouter_api_key

        api_key = openrouter_api_key()
        key_before = key_usage_usd()
        info = fetch_model_info()
        _write(str(cache_dir / MODEL_INFO_FILE), json.dumps(info, indent=2) + "\n")
    else:
        api_key = "offline-no-network"
        info = _load_json(cache_dir / MODEL_INFO_FILE)
    transport = CachingTransport(cache, httpx.HTTPTransport() if live else None)
    http = OpenRouterHttp(api_key=api_key, transport=transport, timeout_s=API_TIMEOUT_S, max_retries=API_RETRIES)

    stack = TimingStack(books, indexes_dir=args.indexes_dir, ablation_dir=args.ablation_dir, device=device, log=log)
    arms: Dict[str, Dict[str, Any]] = {}
    local_timings: Dict[str, Any] = {"date": started.isoformat(timespec="seconds"), "device": device,
                                     "hardware": _hardware(), "systems": {}, "warmup": {}}
    try:
        # 1. Local references (e5 group first, so the API rerankers reuse the loaded e5)
        for name in (VEC_E5, CANDIDATES, BGE_REFERENCE, None, VEC_QWEN3):
            if name is None:
                # 2. API rerankers over the top 30 of hybrid_norm_e5 (e5 still loaded)
                log("API rerankers")
                for arm in RERANK_ARMS:
                    arms[arm.name] = measure_rerank_arm(arm, http, transport, stack, queries,
                                                        arms[CANDIDATES]["runs"]["run1"], log)
                continue
            spec = LOCAL_SPECS[name]
            log(f"Local reference {name}")
            runs, timings, warmup = run_local(stack.searcher_for(spec), queries, stack.clock)
            local_timings["systems"][name], local_timings["warmup"][name] = timings, warmup
            model = spec.reranker or spec.embedding
            cost = stack.costs.get(spec.embedding, {}) if spec.embedding else {}
            arms[name] = {
                "name": name, "model": model, "description": spec.description, "device": device,
                "kind": "local_embedding" if spec.kind == "vector" else ("local_rerank" if spec.reranker else "local_hybrid"),
                "dimension": cost.get("dimension"), "index_build_seconds": cost.get("index_build_seconds"),
                "index_source": cost.get("index_source"), "runs": {"run1": runs},
                "timings": timings if live else saved_timings["systems"][name], "failed_calls": 0,
            }
        # 3. API embedding arms
        log("API embedding arms")
        for arm in EMBEDDING_ARMS:
            arms[arm.name] = measure_embedding_arm(arm, http, transport, books, queries, log)
        # 4. R1 of V07
        arms[R1] = load_r1(args.v07_results, queries)
    except (MissingApiResponseError, SpendLimitError) as error:
        log(f"Stopped: {error}")
        return 1
    stack._release()  # give the last local embedding model's memory back

    # 5. Judgments, analysis and selection; then the end-to-end check with the selected models
    final_raw = current_raw
    e2e_info: Dict[str, Any] = {}
    analysis: Dict[str, Any] = {}
    max_iterations = 4
    for iteration in range(max_iterations):
        runs = {name: arm["runs"]["run1"] for name, arm in arms.items()}
        try:
            final_raw, _ = complete_judgments(
                runs, queries, books, original_raw, final_raw, live=live, workers=args.workers,
                judge_cache_dir=args.judge_cache_dir, output_path=args.output_judgments_path,
                judging_log=cache_dir / JUDGING_RUNS_FILE, log=log)
        except UnjudgedResultsError as error:
            log(f"Stopped: {error}")
            return 1
        if live and not Path(args.output_judgments_path).exists():
            write_judgments_file(final_raw, args.output_judgments_path)
        analysis = analyse(arms, final_raw, books, queries, n_resamples=args.n_resamples, seed=args.seed)
        decisions = analysis["decisions"]
        embedding_name = decisions["embedding"]["winner"]
        rerank_name = decisions["reranker"]["winner"]
        api_rerank_name = rerank_name if rerank_name != R1 else decisions["api_reranker_for_e2e"]
        wanted = (embedding_name, api_rerank_name)
        if not all(wanted):
            log("No eligible API embedding model or reranker: the end-to-end check is skipped")
            break
        if E2E in arms and (arms[E2E]["embedding_arm"], arms[E2E]["rerank_arm"]) == wanted:
            break  # judgments and decisions are stable with the end-to-end run included
        if iteration == max_iterations - 1:
            log("The selection did not settle; the end-to-end check reports the last stack it ran")
            break
        log(f"End-to-end check with {embedding_name} and {api_rerank_name} (iteration {iteration + 1})")
        embedding_arm = next(a for a in EMBEDDING_ARMS if a.name == embedding_name)
        rerank_arm = next(a for a in RERANK_ARMS if a.name == api_rerank_name)
        e2e_stack = TimingStack(books, indexes_dir=args.indexes_dir, ablation_dir=args.ablation_dir,
                                device=device, log=log)
        try:
            arms[E2E] = measure_e2e(embedding_arm, rerank_arm, http, transport, e2e_stack, books, queries, log)
        except (MissingApiResponseError, SpendLimitError) as error:
            log(f"Stopped: {error}")
            return 1
        if live:
            local_timings["e2e"] = {"embedding_arm": embedding_name, "rerank_arm": api_rerank_name,
                                    "timings": arms[E2E]["timings"]}
        else:
            saved = saved_timings.get("e2e") or {}
            if (saved.get("embedding_arm"), saved.get("rerank_arm")) != wanted:
                log(f"Stopped: {timings_path} has no end-to-end timings for {wanted}; run without --offline")
                return 1
            arms[E2E]["timings"] = saved["timings"]
        e2e_info = {"embedding_arm": embedding_name, "rerank_arm": api_rerank_name,
                    "note": "R1 won the reranker role; the stack uses the API reranker the rule ranks first among "
                            "the two API rerankers" if rerank_name == R1 else None}

    if live:
        _write(str(timings_path), json.dumps(local_timings, indent=2) + "\n")
        saved_timings = local_timings
    key_after = key_usage_usd() if live else None

    entries = cache.entries()
    api_usd = _sum([(e["response"].get("usage") or {}).get("cost") for e in entries]) or 0.0
    judgments = judgments_block(final_raw, start_raw, read_judging_runs(cache_dir / JUDGING_RUNS_FILE),
                                args.output_judgments_path, args.start_judgments_path)
    judge_usd = judgments["judge_cost_usd"] or 0.0
    key_delta = (key_after - key_before) if key_before is not None and key_after is not None else \
        previous.get("spend", {}).get("key_usage_delta_usd")
    command = " ".join(["python -m app.evaluation.api_models_evaluation", *map(shlex.quote, raw_args)]).strip()
    document = {
        "design": {
            "embedding_arms": [{"name": a.name, "model": a.model, "dimension": a.dimension, "primary": a.primary}
                               for a in EMBEDDING_ARMS],
            "rerank_arms": [{"name": a.name, "model": a.model} for a in RERANK_ARMS],
            "local_references": [VEC_E5, VEC_QWEN3, CANDIDATES, BGE_REFERENCE],
            "llm_reranker": R1,
            "contrasts": [{"name": c.name, "system": c.system, "reference": c.reference, "question": c.question}
                          for c in CONTRASTS],
            "pins": {"embedding": EMBEDDING_PIN, "rerank": RERANK_PIN},
            "document_batch_size": DEFAULT_BATCH_SIZE, "rerank_document_chars": DEFAULT_MAX_DOCUMENT_CHARS,
            "rerank_depth": RERANK_DEPTH, "depth": DEPTH, "n_resamples": args.n_resamples, "seed": args.seed,
            "rule": {"tie_margin": TIE_MARGIN, "latency_band": LATENCY_BAND},
        },
        "num_queries": analysis["num_queries"],
        "arms": {name: serializable_arm(arm) for name, arm in arms.items()},
        "analysis": {k: v for k, v in analysis.items() if k != "per_query"},
        "per_query_ndcg_at_10": {name: {q: s[PRIMARY_METRIC] for q, s in pq.items()}
                                 for name, pq in analysis["per_query"].items()},
        "llm_roles": llm_roles(args.llm_selection_results),
        "e2e": e2e_info,
        "judgments": judgments,
        "reproduction_check": reproduction_check(arms),
        "model_info": info,
        "prices": reported_prices(cache),
        "local_timings_date": saved_timings.get("date"),
        "spend": {
            "api_usd": api_usd, "api_calls": len(entries),
            "embedding_usd": _sum([(e["response"].get("usage") or {}).get("cost") for e in entries
                                   if e["path"] == "embeddings"]) or 0.0,
            "rerank_usd": _sum([(e["response"].get("usage") or {}).get("cost") for e in entries
                                if e["path"] == "rerank"]) or 0.0,
            "judge_usd": judge_usd, "total_usd": api_usd + judge_usd,
            "failed_attempts": sum(1 for f in cache.failures if f.get("kind") != "call_failure"),
            "key_usage_delta_usd": key_delta,
            "calls_made_this_run": transport.calls_made, "this_run_api_usd": transport.spent_usd,
        },
        "run": {
            "date": started.isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_worktree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "command": command, "mode": "live" if live else "offline",
            "device": device, "hardware": _hardware(), "library_versions": _versions(),
            "num_books": len(books), "cache_file": str(cache.path), "timings_file": str(timings_path),
            "api_call_dates": sorted({e["date"][:10] for e in entries}),
            "queries_sha256": _sha256(args.queries_path),
            "judgments_sha256": _sha256(args.output_judgments_path),
            "responses_sha256": _sha256(str(cache.path)),
        },
    }
    _write(args.output_json, json.dumps(document, ensure_ascii=False, indent=2, default=str) + "\n")
    _write(args.output_md, build_markdown(document))

    d = analysis["decisions"]
    log(f"Cloud embedding model: {d['embedding']['winner']} ({d['embedding']['decided_by']})")
    log(f"Cloud reranker: {d['reranker']['winner']} ({d['reranker']['decided_by']})")
    for name, c in analysis["contrasts"].items():
        log(f"{name} {c['system']} - {c['reference']}: {c['mean_difference']:+.3f} "
            f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] p={c['p_value']:.3f} p_holm={c['p_value_holm']:.3f}")
    log(f"API calls made in this run: {transport.calls_made} ({transport.spent_usd:.4f} USD). "
        f"Total spend {document['spend']['total_usd']:.4f} USD. Saved {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
