"""
The system under evaluation, built like the cloud profile's composition root
(app/api/v1/dependencies.py) but without the HTTP layer:

- settings from the environment with SEARCH_PROFILE=cloud and
  DATABASE_SCHEMA=catalog_deploy, as the deployment reads them;
- the Neon catalog through a read-only wrapper: every statement runs with
  transaction_read_only=on and any write method raises;
- the application's query encoder (OpenRouterEmbeddingEncoder.from_settings
  with the cloud deadlines, the hedge to a second provider, Zero Data
  Retention and the in-process query cache), behind a persistent cache of
  query vectors by text, so a re-run embeds nothing twice and the same
  text always gets the same vector;
- the API reranker with the evaluation's long deadlines (as V14);
- the LLM client, the agent and the assistant exactly as get_llm_client,
  get_search_agent and get_assistant build them;
- SearchService without a usage budget, so the live site's daily counters
  (a table in the same database) are neither read nor written.

Every paid call goes to the spend ledger with the cost OpenRouter reported.
The raw outputs of the assistant's routing and answer calls are captured
(the reply carries only the verified recommendations).
"""

import hashlib
import contextvars
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.evaluation.assistant_evaluation import design as D
from app.evaluation.catalog_evaluation.ledger import LedgerLLMRecorder, LedgerRecorder, SpendLedger
from app.evaluation.catalog_evaluation.runs import JsonlStore, decode_vector, encode_vector, model_slug

# The raw outputs captured during one assistant request: {"router": [...], "answer": [...]}.
CAPTURE: contextvars.ContextVar[Optional[Dict[str, List[Any]]]] = contextvars.ContextVar("assistant_capture",
                                                                                         default=None)


class ReadOnlyDatabase:
    """A PostgresDatabase whose statements run read-only; write methods raise."""

    READ_ONLY = {"transaction_read_only": "on"}

    def __init__(self, db: Any) -> None:
        self._db = db

    def query(self, sql: str, params: Any = None, *, settings: Optional[Dict[str, object]] = None):
        return self._db.query(sql, params, settings={**self.READ_ONLY, **(settings or {})})

    def execute(self, *args: Any, **kwargs: Any):
        raise PermissionError("the evaluation's database connection is read-only")

    def execute_outside_transaction(self, *args: Any, **kwargs: Any):
        raise PermissionError("the evaluation's database connection is read-only")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)


class PersistentQueryEncoder:
    """
    The application's query encoder behind a cache of vectors by text, on disk.

    A text seen before (in this run or an earlier one) gets its stored
    vector; a new one is encoded by the inner encoder (with its own deadlines
    and hedge), up to D.EMBED_ATTEMPTS attempts, and stored before it is
    returned. Every other attribute is the inner encoder's.
    """

    def __init__(self, inner: Any, path: Path, check: Callable[[], None] = lambda: None) -> None:
        self._inner = inner
        self._store = JsonlStore(path, key="text")
        self._check = check
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self.failures: List[Dict[str, Any]] = []

    def _lock(self, text: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(text, threading.Lock())

    def generate_query_embedding(self, text: str) -> List[float]:
        with self._lock(text):
            record = self._store.get(text)
            if record is not None:
                return [float(x) for x in decode_vector(record["vector"])]
            errors = []
            for attempt in range(1, D.EMBED_ATTEMPTS + 1):
                self._check()
                started = time.perf_counter()
                try:
                    vector = self._inner.generate_query_embedding(text)
                except Exception as error:  # noqa: BLE001 - retried, then raised
                    errors.append(type(error).__name__)
                    self.failures.append({"text": text, "attempt": attempt, "error": type(error).__name__})
                    continue
                self._store.add({"text": text, "model": self._inner.get_model_name(), "dimension": len(vector),
                                 "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                                 "attempts": attempt, "failed_attempts": errors,
                                 "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "vector": encode_vector(vector)})
                return list(vector)
            raise RuntimeError(f"query embedding failed {D.EMBED_ATTEMPTS} times: {errors}")

    def __len__(self) -> int:
        return len(self._store.records)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _call_dict(record: Any) -> Dict[str, Any]:
    return {"operation": record.operation, "step": record.step, "model": record.model,
            "latency_s": round(record.latency_s, 3), "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens, "cost_usd": record.cost_usd, "error": record.error}


class ContextLedgerRecorder(LedgerRecorder):
    """Ledger recorder of the API encoder and reranker that also files each call under CAPTURE["api_calls"]."""

    def _append(self, record: Any) -> None:  # type: ignore[override]
        super()._append(record)
        store = CAPTURE.get()
        if store is not None:
            store.setdefault("api_calls", []).append(_call_dict(record))


class ContextLedgerLLMRecorder(LedgerLLMRecorder):
    """Ledger recorder of the chat model that also files each call under CAPTURE["llm_calls"]."""

    run_inline = True

    def _append(self, record: Any) -> None:  # type: ignore[override]
        super()._append(record)
        store = CAPTURE.get()
        if store is not None:
            store.setdefault("llm_calls", []).append(_call_dict(record))


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return repr(value)


class CapturingRunnable:
    """Wraps a runnable (structured-output chain): each output, or error, is added to CAPTURE[key]."""

    def __init__(self, inner: Any, key: str) -> None:
        self._inner = inner
        self._key = key

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        store = CAPTURE.get()
        try:
            output = self._inner.invoke(*args, **kwargs)
        except Exception as error:
            if store is not None:
                store.setdefault(self._key, []).append({"error": f"{type(error).__name__}: {error}"[:300]})
            raise
        if store is not None:
            store.setdefault(self._key, []).append({"output": _dump(output)})
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def openrouter_options_without_key(reasoning: Any) -> Dict[str, Any]:
    """The OpenRouter request fields the client sends (its extra_body), without the key or the URL."""
    from app.infrastructure.llm.langchain_llm_client import openrouter_options

    return openrouter_options(None, reasoning)["extra_body"]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


class CloudSystem:
    """The cloud profile's search service, agent and assistant, built once per stage."""

    def __init__(self, data_dir: Path, ledger: SpendLedger, stage: str, log: Callable[[str], None] = print) -> None:
        from app.domain.services import SearchService
        from app.infrastructure.config.settings import (
            ApiQuerySettings,
            EmbeddingSettings,
            LLMSettings,
            RerankerSettings,
            SearchProfileSettings,
            database_url,
        )
        from app.infrastructure.db.postgres_book_catalog_repository import PostgresBookCatalogRepository
        from app.infrastructure.db.postgres_connection import PostgresDatabase
        from app.infrastructure.llm.agents.search_agent import LangGraphSearchAgent
        from app.infrastructure.llm.graphs.assistant import LangGraphAssistant
        from app.infrastructure.llm.langchain_llm_client import LangChainLLMClient
        from app.infrastructure.search.api_reranker import ApiReranker
        from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder
        from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository
        from app.infrastructure.search.postgres_embeddings_store import PostgresEmbeddingsStore
        from app.infrastructure.search.postgres_lexical_search_repository import PostgresLexicalSearchRepository

        self.data_dir = data_dir
        self.ledger = ledger
        self.stage = stage
        self.log = log
        env = {**os.environ, **D.CLOUD_ENV}
        self.profile = SearchProfileSettings.from_env(env)
        self.embedding_settings = EmbeddingSettings.from_env(env)
        self.query_settings = ApiQuerySettings.from_env(env)
        self.reranker_settings = RerankerSettings.from_env(env)
        self.llm_settings = LLMSettings.from_env(env)
        rerank_settings = ApiQuerySettings(timeout_s=D.RERANK_TIMEOUT_S, max_retries=D.RERANK_RETRIES, deadline_s=None)

        self._raw_db = PostgresDatabase(database_url(env), pool_max_size=D.DB_POOL)
        self.db = ReadOnlyDatabase(self._raw_db)
        recorder = ContextLedgerRecorder(ledger, stage)
        inner = OpenRouterEmbeddingEncoder.from_settings(self.embedding_settings, usage_recorder=recorder,
                                                         query_settings=self.query_settings)
        self.encoder = PersistentQueryEncoder(
            inner, data_dir / "query_vectors" / f"{model_slug(self.embedding_settings.model)}.jsonl", ledger.check)
        self.embeddings = PostgresEmbeddingsStore(self.encoder, self.db, self.profile.schema)
        self.catalog = PostgresBookCatalogRepository(self.db, self.profile.schema)
        lexical = PostgresLexicalSearchRepository(self.db, self.profile.schema, ranking=self.profile.lexical_ranking)
        self.vector = PgVectorSearchRepository(self.db, self.profile.schema, mode=self.profile.vector_search,
                                               expected_identity=self.embeddings.get_model_name())
        self.reranker = ApiReranker.from_settings(self.reranker_settings, usage_recorder=recorder,
                                                  query_settings=rerank_settings)
        s = self.llm_settings
        self.llm_client = LangChainLLMClient(
            provider=s.provider, model_name=s.model, temperature=s.temperature, max_retries=s.max_retries,
            base_url=s.base_url, num_ctx=s.num_ctx, reasoning=s.reasoning,
            known_categories=self.catalog.get_categories(), apply_extracted_category=s.apply_extracted_category,
            max_tokens=s.max_tokens, timeout_s=s.timeout_s, client_max_retries=s.client_max_retries,
        )
        # Every call of the chat model (assistant and agent) also goes to the ledger.
        self.llm_client.model.callbacks.append(ContextLedgerLLMRecorder(ledger, stage, "assistant"))
        self.search = SearchService(
            lexical_search=lexical, vector_search=self.vector, embeddings_store=self.embeddings,
            llm_client=self.llm_client, max_explanations=s.max_explanations, reranker=self.reranker,
            rerank_depth=self.reranker_settings.depth, usage_budget=None, candidates=self.profile.candidates,
        )
        self.agent = LangGraphSearchAgent(model=self.llm_client.model, search_service=self.search,
                                          catalog_repo=self.catalog, model_name=s.model)
        self.assistant = LangGraphAssistant(model=self.llm_client.model, search_service=self.search,
                                            agent=self.agent, model_name=s.model)
        self.assistant._router = CapturingRunnable(self.assistant._router, "router")
        self.assistant._answerer = CapturingRunnable(self.assistant._answerer, "answer")
        self.commit = git_commit()

    def close(self) -> None:
        self._raw_db.close()

    def describe(self) -> Dict[str, Any]:
        """The configuration evaluated (no secret)."""
        from app.infrastructure.llm.graphs import assistant as G
        from app.infrastructure.llm import prompts_assistant as P
        from app.infrastructure.llm.agents import search_agent as A
        from app.domain import value_objects as V

        e, r, s, q = self.embedding_settings, self.reranker_settings, self.llm_settings, self.query_settings
        return {
            "commit": self.commit,
            "profile": self.profile.describe(),
            "embedding": {"model": e.model, "dimensions": e.dimensions, "provider": e.openrouter_provider,
                          "zdr": e.openrouter_zdr, "hedge_provider": e.hedge_provider, "hedge_after_s": e.hedge_after_s,
                          "hedge_primary_again_s": e.hedge_primary_again_s, "query_cache_size": e.query_cache_size,
                          "query_deadlines": q.call_options(), "evaluation_attempts": D.EMBED_ATTEMPTS},
            "reranker": {"model": r.model, "provider": r.openrouter_provider, "depth": r.depth,
                         "evaluation_deadlines": {"timeout_s": D.RERANK_TIMEOUT_S, "max_retries": D.RERANK_RETRIES}},
            "llm": {"provider": s.provider, "model": s.model, "temperature": s.temperature, "reasoning": s.reasoning,
                    "max_tokens": s.max_tokens, "timeout_s": s.timeout_s, "max_retries": s.max_retries,
                    "client_max_retries": s.client_max_retries},
            "assistant": {"router_max_tokens": G.ROUTER_MAX_TOKENS, "answer_context_books": G.ANSWER_CONTEXT_BOOKS,
                          "max_clarifying_turns": V.MAX_CLARIFYING_TURNS, "max_results": D.ASSISTANT_MAX_RESULTS,
                          "prompts_version": P.ASSISTANT_PROMPTS_VERSION,
                          "router_prompt_sha256": hashlib.sha256(P.ROUTER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]},
            "llm_openrouter_extra_body": openrouter_options_without_key(s.reasoning),
            "agent": {"max_tool_rounds": A.MAX_TOOL_ROUNDS, "max_reflections": A.MAX_REFLECTIONS,
                      "max_recommendations": A.MAX_RECOMMENDATIONS, "tool_results": A.MAX_TOOL_RESULTS},
            "usage_budget": None,
        }
