"""
FastAPI dependencies for dependency injection (the composition root).

This module provides singleton instances of repositories and services
for use with FastAPI's Depends() system.

Note: We use module-level singletons instead of @lru_cache with Depends()
parameters, which is an antipattern that can cause unexpected behavior.

Adapters are imported inside the provider that builds them, for the profile
in use (V13). The local profile imports SQLite, rank_bm25, FAISS and the
sentence-transformers models when it builds them. The cloud profile with API
embeddings and an API reranker (SEARCH_PROFILE=cloud, the defaults of that
profile) imports none of torch, sentence-transformers, transformers, faiss
or rank_bm25, so the backend fits a small serverless function
(tests/api/test_cloud_profile_imports.py checks it in a fresh process).
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from app.domain.ports import (
    BookCatalogRepository,
    ConversationalAssistant,
    LexicalSearchRepository,
    VectorSearchRepository,
    EmbeddingsStore,
    LLMClient,
    Reranker,
    SearchAgent,
    UsageBudget,
)
from app.domain.services import SearchService
from app.infrastructure.config.settings import (
    ApiQuerySettings,
    EmbeddingSettings,
    EvaluateEndpointSettings,
    LLMSettings,
    RateLimitSettings,
    RerankerSettings,
    SearchProfileSettings,
    UsageBudgetSettings,
    UsageEndpointSettings,
    database_url,
)
from app.infrastructure.usage_recording import UsageRecorder

logger = logging.getLogger(__name__)

# Configuration from environment
DB_PATH = Path(os.getenv("DB_PATH", "data/catalog.db"))
INDEXES_DIR = Path(os.getenv("INDEXES_DIR", "data/indexes"))

# Module-level singletons (initialized lazily)
_profile_settings: Optional[SearchProfileSettings] = None
_postgres_database = None  # PostgresDatabase of the cloud profile (imported lazily)
_catalog_repository: Optional[BookCatalogRepository] = None
_bm25_repository: Optional[LexicalSearchRepository] = None
_embeddings_store: Optional[EmbeddingsStore] = None
_vector_repository: Optional[VectorSearchRepository] = None
_search_service: Optional[SearchService] = None
_llm_settings: Optional[LLMSettings] = None
_embedding_settings: Optional[EmbeddingSettings] = None
_llm_client: Optional[LLMClient] = None
_llm_initialized: bool = False
_search_agent: Optional[SearchAgent] = None
_agent_initialized: bool = False
_assistant: Optional[ConversationalAssistant] = None
_assistant_initialized: bool = False
_reranker_settings: Optional[RerankerSettings] = None
_reranker: Optional[Reranker] = None
_reranker_initialized: bool = False
_usage_recorder: Optional[UsageRecorder] = None
_usage_since: Optional[str] = None
_usage_settings: Optional[UsageEndpointSettings] = None
_api_query_settings: Optional[ApiQuerySettings] = None
_budget_settings: Optional[UsageBudgetSettings] = None
_usage_budget: Optional[UsageBudget] = None
_budget_initialized: bool = False
_rate_limit_settings: Optional[RateLimitSettings] = None
_rest_rate_limits = None  # RestRateLimits (app/api/rate_limiting.py), built on first use
_evaluate_settings: Optional[EvaluateEndpointSettings] = None


def get_search_profile_settings() -> SearchProfileSettings:
    """
    Provide the search profile, read once from the environment (and .env).

    SEARCH_PROFILE=local (default) keeps the evaluated system: SQLite
    catalog, BM25 and FAISS indexes on disk. SEARCH_PROFILE=cloud serves the
    catalog, lexical and vector search from Postgres (DATABASE_SCHEMA), with
    the query embedding and the reranker from OpenRouter's API by default
    (EmbeddingSettings and RerankerSettings). The LLM wiring is the same in
    both profiles.
    """
    global _profile_settings
    if _profile_settings is None:
        load_dotenv()
        _profile_settings = SearchProfileSettings.from_env()
    return _profile_settings


def get_postgres_database():
    """
    Provide the Postgres database of the cloud profile (pooled connection string).

    Imported lazily so the local profile does not need a database driver.

    Raises:
        ValueError: If DATABASE_URL is not set
    """
    global _postgres_database
    if _postgres_database is None:
        from app.infrastructure.db.postgres_connection import PostgresDatabase

        load_dotenv()
        _postgres_database = PostgresDatabase(database_url())
    return _postgres_database


def get_catalog_repository() -> BookCatalogRepository:
    """Provide a singleton instance of the catalog repository."""
    global _catalog_repository
    if _catalog_repository is None:
        profile = get_search_profile_settings()
        if profile.is_cloud:
            from app.infrastructure.db.postgres_book_catalog_repository import PostgresBookCatalogRepository

            _catalog_repository = PostgresBookCatalogRepository(get_postgres_database(), profile.schema)
        else:
            from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

            _catalog_repository = SqliteBookCatalogRepository(DB_PATH)
    return _catalog_repository


def get_bm25_repository() -> LexicalSearchRepository:
    """
    Provide a singleton instance of the lexical (BM25) repository.

    The local BM25 adapter returns only books that contain a query term
    (V13); its legacy padding is only for reproducing evaluation runs.
    """
    global _bm25_repository
    if _bm25_repository is None:
        profile = get_search_profile_settings()
        if profile.is_cloud:
            from app.infrastructure.search.postgres_lexical_search_repository import (
                PostgresLexicalSearchRepository,
            )

            _bm25_repository = PostgresLexicalSearchRepository(
                get_postgres_database(), profile.schema, ranking=profile.lexical_ranking
            )
            return _bm25_repository
        from app.infrastructure.search.bm25_search_repository import BM25SearchRepository

        bm25_path = INDEXES_DIR / "bm25_index.pkl"
        repo = BM25SearchRepository()
        if bm25_path.exists():
            repo.load_index(str(bm25_path))
        _bm25_repository = repo
    return _bm25_repository


def get_embedding_settings() -> EmbeddingSettings:
    """Provide the embedding settings, read once from the environment (and .env)."""
    global _embedding_settings
    if _embedding_settings is None:
        load_dotenv()
        _embedding_settings = EmbeddingSettings.from_env()
    return _embedding_settings


def get_usage_recorder() -> UsageRecorder:
    """
    Provide the recorder of the API calls of the embedding encoder and the reranker.

    One recorder for both (operations "embedding" and "rerank"), created
    with the process; /api/v1/usage summarizes it (USAGE_ENDPOINT_ENABLED).
    """
    global _usage_recorder, _usage_since
    if _usage_recorder is None:
        _usage_recorder = UsageRecorder()
        _usage_since = datetime.now(UTC).isoformat(timespec="seconds")
    return _usage_recorder


def get_api_query_settings() -> ApiQuerySettings:
    """Deadlines of the model API calls a user waits for (cloud defaults: 3 s, 1 retry, 5 s per call)."""
    global _api_query_settings
    if _api_query_settings is None:
        load_dotenv()
        _api_query_settings = ApiQuerySettings.from_env()
    return _api_query_settings


def build_query_encoder(
    settings: EmbeddingSettings,
    usage_recorder: Optional[UsageRecorder] = None,
    query_settings: Optional[ApiQuerySettings] = None,
):
    """
    The encoder the cloud profile's embeddings store uses (no FAISS index).

    "openrouter" gives OpenRouterEmbeddingEncoder (no model in this process;
    its requests are recorded in usage_recorder; query embeddings use the
    deadlines of query_settings). "local" gives the local
    sentence-transformers model through EmbeddingsStoreFaiss, used only to
    encode (this one imports torch).
    """
    if settings.provider == "openrouter":
        from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder

        return OpenRouterEmbeddingEncoder.from_settings(
            settings, usage_recorder=usage_recorder, query_settings=query_settings
        )
    from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss

    return EmbeddingsStoreFaiss(model_name=settings.model, device=settings.device)


def build_embeddings_store(settings: EmbeddingSettings, usage_recorder: Optional[UsageRecorder] = None):
    """
    The FAISS store (local profile) with the encoder that EMBEDDING_PROVIDER selects.

    "local" (default) loads the sentence-transformers model EMBEDDING_MODEL in
    this process, as before. "openrouter" encodes through OpenRouter's
    embeddings API (OpenRouterEmbeddingEncoder: EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS, EMBEDDING_OPENROUTER_PROVIDER) and loads no model;
    the index must have been built with the same API model, provider and
    dimension, or loading it raises IndexModelMismatchError.
    """
    from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss

    if settings.provider == "openrouter":
        from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder

        return EmbeddingsStoreFaiss(
            encoder=OpenRouterEmbeddingEncoder.from_settings(settings, usage_recorder=usage_recorder)
        )
    return EmbeddingsStoreFaiss(model_name=settings.model, device=settings.device)


def get_embeddings_store() -> EmbeddingsStore:
    """
    Provide a singleton instance of the embeddings store.

    Local profile: the FAISS store, its index loaded from INDEXES_DIR (an
    index built with another model raises an error that asks to rebuild it).
    Cloud profile: PostgresEmbeddingsStore over the query encoder; the
    document vectors live in Postgres.
    """
    global _embeddings_store
    if _embeddings_store is None:
        settings = get_embedding_settings()
        profile = get_search_profile_settings()
        if profile.is_cloud:
            from app.infrastructure.search.postgres_embeddings_store import PostgresEmbeddingsStore

            encoder = build_query_encoder(
                settings, usage_recorder=get_usage_recorder(), query_settings=get_api_query_settings()
            )
            _embeddings_store = PostgresEmbeddingsStore(encoder, get_postgres_database(), profile.schema)
            return _embeddings_store
        store = build_embeddings_store(settings, usage_recorder=get_usage_recorder())
        faiss_path = INDEXES_DIR / "faiss_index"
        if faiss_path.exists():
            store.load_index(str(faiss_path))
        _embeddings_store = store
    return _embeddings_store


def get_vector_repository() -> VectorSearchRepository:
    """
    Provide a singleton instance of the vector search repository.

    Cloud profile: pgvector, which refuses to search when the vectors of the
    schema were built with another encoder identity (model, provider,
    dimension) than the query encoder's.
    """
    global _vector_repository
    if _vector_repository is None:
        profile = get_search_profile_settings()
        if profile.is_cloud:
            from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository

            _vector_repository = PgVectorSearchRepository(
                get_postgres_database(),
                profile.schema,
                mode=profile.vector_search,
                expected_identity=get_embeddings_store().get_model_name(),
            )
            return _vector_repository
        from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository

        catalog_repo = get_catalog_repository()
        embeddings_store = get_embeddings_store()
        books = catalog_repo.get_all()  # Option A: book_id -> Book map in memory
        _vector_repository = FaissVectorSearchRepository(embeddings_store, books)
    return _vector_repository


def get_llm_settings() -> LLMSettings:
    """Provide the LLM settings, read once from the environment (and .env)."""
    global _llm_settings
    if _llm_settings is None:
        load_dotenv()
        _llm_settings = LLMSettings.from_env()
    return _llm_settings


def get_llm_client() -> Optional[LLMClient]:
    """
    Provide the LLM client, or None when the LLM is disabled or cannot be built.

    The LLM is optional: if it is off (LLM_PROVIDER=none, or LLM_PROVIDER
    unset and no OPENROUTER_API_KEY) or the client fails to initialise, the
    API keeps serving plain hybrid search and logs why; requests that ask
    for understanding or explanations get the notice "llm_unavailable".
    The catalog categories are passed so filters extracted from queries are
    validated against real categories; they come from one aggregate query
    of the catalog (get_categories), not from reading every book.
    """
    global _llm_client, _llm_initialized
    if _llm_initialized:
        return _llm_client
    _llm_initialized = True

    settings = get_llm_settings()
    if not settings.enabled:
        logger.info(
            "LLM off (LLM_PROVIDER=none, or unset without OPENROUTER_API_KEY): search works; "
            "query understanding, explanations and the agent are unavailable"
        )
        return None

    try:
        # Imported here so the API can start without LangChain when the LLM is off
        from app.infrastructure.llm.langchain_llm_client import LangChainLLMClient

        categories = get_catalog_repository().get_categories()
        _llm_client = LangChainLLMClient(
            provider=settings.provider,
            model_name=settings.model,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
            base_url=settings.base_url,
            num_ctx=settings.num_ctx,
            reasoning=settings.reasoning,
            known_categories=categories,
            apply_extracted_category=settings.apply_extracted_category,
            max_tokens=settings.max_tokens,
            timeout_s=settings.timeout_s,
            client_max_retries=settings.client_max_retries,
        )
    except Exception as e:
        logger.warning(f"LLM client unavailable ({settings.provider}/{settings.model}): {e}")
        _llm_client = None
    return _llm_client


def get_search_agent() -> Optional[SearchAgent]:
    """
    Provide the agentic search (LangGraph ReAct agent), or None without an LLM.

    The agent reuses the chat model of the LLM client and the same search
    service and catalog as the rest of the API.
    """
    global _search_agent, _agent_initialized
    if _agent_initialized:
        return _search_agent
    _agent_initialized = True

    model = getattr(get_llm_client(), "model", None)
    if model is None:
        return None
    try:
        from app.infrastructure.llm.agents.search_agent import LangGraphSearchAgent

        _search_agent = LangGraphSearchAgent(
            model=model,
            search_service=get_search_service(),
            catalog_repo=get_catalog_repository(),
            model_name=get_llm_settings().model,
        )
    except Exception as e:
        logger.warning(f"Search agent unavailable: {e}")
        _search_agent = None
    return _search_agent


def get_assistant() -> Optional[ConversationalAssistant]:
    """
    Provide the conversational assistant (LangGraph router graph), or None without an LLM.

    Same wiring in both profiles: the chat model of the LLM client, the
    search service of the profile in use and the search agent for multi-step
    requests (when the agent cannot be built, those requests are searched).
    """
    global _assistant, _assistant_initialized
    if _assistant_initialized:
        return _assistant
    _assistant_initialized = True

    model = getattr(get_llm_client(), "model", None)
    if model is None:
        return None
    try:
        from app.infrastructure.llm.graphs.assistant import LangGraphAssistant

        _assistant = LangGraphAssistant(
            model=model,
            search_service=get_search_service(),
            agent=get_search_agent(),
            model_name=get_llm_settings().model,
        )
    except Exception as e:
        logger.warning(f"Assistant unavailable: {e}")
        _assistant = None
    return _assistant


def get_reranker_settings() -> RerankerSettings:
    """Provide the reranker settings, read once from the environment (and .env)."""
    global _reranker_settings
    if _reranker_settings is None:
        load_dotenv()
        _reranker_settings = RerankerSettings.from_env()
    return _reranker_settings


def get_reranker() -> Optional[Reranker]:
    """
    Provide the reranker, or None when re-ranking is disabled.

    Local profile: re-ranking is off unless RERANKER_MODEL is set.
    RERANKER_PROVIDER=local (default there) gives the cross-encoder, whose
    model is loaded on the first search, so a wrong model name shows up then
    (logged by the search service, which keeps the fused order) and in
    /health. RERANKER_PROVIDER=openrouter (default in the cloud profile,
    with qwen/qwen3-reranker-8b) gives ApiReranker (OpenRouter's rerank API,
    RERANKER_OPENROUTER_PROVIDER to pin a provider), whose calls are
    recorded in the usage recorder; a missing OPENROUTER_API_KEY disables
    re-ranking with a warning.
    """
    global _reranker, _reranker_initialized
    if _reranker_initialized:
        return _reranker
    _reranker_initialized = True

    settings = get_reranker_settings()
    if not settings.enabled:
        return None
    try:
        if settings.provider == "openrouter":
            # OpenRouter's rerank API: no model in this process
            from app.infrastructure.search.api_reranker import ApiReranker

            _reranker = ApiReranker.from_settings(
                settings, usage_recorder=get_usage_recorder(), query_settings=get_api_query_settings()
            )
            return _reranker
        from app.infrastructure.search.cross_encoder_reranker import CrossEncoderReranker

        _reranker = CrossEncoderReranker(model_name=settings.model, device=settings.device)
    except Exception as e:
        logger.warning(f"Reranker unavailable ({settings.model}): {e}")
        _reranker = None
    return _reranker


def get_budget_settings() -> UsageBudgetSettings:
    """Daily allowance settings (DAILY_SEARCH_LIMIT, DAILY_LLM_LIMIT, BUDGET_SCHEMA)."""
    global _budget_settings
    if _budget_settings is None:
        load_dotenv()
        _budget_settings = UsageBudgetSettings.from_env()
    return _budget_settings


def get_usage_budget() -> Optional[UsageBudget]:
    """
    Provide the daily budget of paid model calls, or None without limits (V17).

    Cloud profile: PostgresUsageBudget, a counter table in the catalog's
    database shared by every serverless instance. Local profile: an
    in-memory counter (one process). Unset limits (the default): None.
    """
    global _usage_budget, _budget_initialized
    if _budget_initialized:
        return _usage_budget
    _budget_initialized = True
    settings = get_budget_settings()
    if not settings.enabled:
        return None
    if get_search_profile_settings().is_cloud:
        from app.infrastructure.db.postgres_usage_budget import PostgresUsageBudget

        _usage_budget = PostgresUsageBudget(get_postgres_database(), settings.schema, settings.limits())
    else:
        from app.infrastructure.usage_budget import InMemoryUsageBudget

        _usage_budget = InMemoryUsageBudget(settings.limits())
    return _usage_budget


def get_rate_limit_settings() -> RateLimitSettings:
    """Per-client limits of the REST endpoints and the trusted client-address header."""
    global _rate_limit_settings
    if _rate_limit_settings is None:
        load_dotenv()
        _rate_limit_settings = RateLimitSettings.from_env()
    return _rate_limit_settings


def get_rest_rate_limits():
    """The per-client sliding-window limiters of the REST endpoints (one set per process)."""
    global _rest_rate_limits
    if _rest_rate_limits is None:
        from app.api.rate_limiting import RestRateLimits

        _rest_rate_limits = RestRateLimits(get_rate_limit_settings())
    return _rest_rate_limits


def get_evaluate_settings() -> EvaluateEndpointSettings:
    """Switch of POST /evaluate (EVALUATE_ENDPOINT_ENABLED)."""
    global _evaluate_settings
    if _evaluate_settings is None:
        load_dotenv()
        _evaluate_settings = EvaluateEndpointSettings.from_env()
    return _evaluate_settings


def get_search_service() -> SearchService:
    """
    Provide the Search Service with all dependencies wired.

    The candidate mode comes from SEARCH_CANDIDATES (SearchProfileSettings):
    "vector" by default in both profiles, "hybrid" to fuse BM25 and vector
    search with RRF as in the earlier experiments. The REST search, the MCP
    server, the search agent's tools and the assistant all use this service.
    """
    global _search_service
    if _search_service is None:
        _search_service = SearchService(
            lexical_search=get_bm25_repository(),
            vector_search=get_vector_repository(),
            embeddings_store=get_embeddings_store(),
            llm_client=get_llm_client(),
            max_explanations=get_llm_settings().max_explanations,
            reranker=get_reranker(),
            rerank_depth=get_reranker_settings().depth,
            usage_budget=get_usage_budget(),
            candidates=get_search_profile_settings().candidates,
        )
    return _search_service


def database_health() -> Optional[Dict[str, Any]]:
    """
    Whether the cloud profile's database answers (one SELECT 1), or None in the local profile.

    Returns {"reachable": bool, "latency_ms": float}; never raises.
    """
    if not get_search_profile_settings().is_cloud:
        return None
    import time

    started = time.perf_counter()
    try:
        reachable = get_postgres_database().ping()
    except Exception as e:  # noqa: BLE001 - health must answer
        logger.warning(f"Database health check failed: {e}")
        reachable = False
    return {"reachable": reachable, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def get_usage_settings() -> UsageEndpointSettings:
    """Provide the settings of the usage endpoint, read once from the environment (and .env)."""
    global _usage_settings
    if _usage_settings is None:
        load_dotenv()
        _usage_settings = UsageEndpointSettings.from_env()
    return _usage_settings


def usage_summary() -> Dict[str, Any]:
    """
    What the model APIs cost since the process started, by operation.

    "api": the embedding encoder and the reranker (operations "embedding" and
    "rerank"); "llm": the chat model of the LLM client, if one was built.
    Reported costs are the providers' own (OpenRouter's usage.cost, USD).
    Nothing is built here: components not used yet report no calls.
    """
    recorder = get_usage_recorder()
    llm_recorder = getattr(_llm_client, "usage_recorder", None) if _llm_initialized else None
    return {
        "since": _usage_since,
        "api": recorder.summary(),
        "llm": llm_recorder.summary() if llm_recorder is not None else None,
    }


def reset_dependencies() -> None:
    """
    Reset all singletons. Useful for testing.

    This allows tests to inject mock dependencies by resetting
    the module state between test cases.
    """
    global _catalog_repository, _bm25_repository, _embeddings_store
    global _vector_repository, _search_service
    global _llm_settings, _llm_client, _llm_initialized
    global _search_agent, _agent_initialized, _embedding_settings
    global _assistant, _assistant_initialized
    global _reranker_settings, _reranker, _reranker_initialized
    global _profile_settings, _postgres_database
    global _usage_recorder, _usage_since, _usage_settings
    global _api_query_settings, _budget_settings, _usage_budget, _budget_initialized
    global _rate_limit_settings, _rest_rate_limits, _evaluate_settings

    if _postgres_database is not None:
        _postgres_database.close()
    _profile_settings = None
    _postgres_database = None
    _catalog_repository = None
    _bm25_repository = None
    _embeddings_store = None
    _vector_repository = None
    _search_service = None
    _llm_settings = None
    _embedding_settings = None
    _llm_client = None
    _llm_initialized = False
    _search_agent = None
    _agent_initialized = False
    _assistant = None
    _assistant_initialized = False
    _reranker_settings = None
    _reranker = None
    _reranker_initialized = False
    _usage_recorder = None
    _usage_since = None
    _usage_settings = None
    _api_query_settings = None
    _budget_settings = None
    _usage_budget = None
    _budget_initialized = False
    _rate_limit_settings = None
    _rest_rate_limits = None
    _evaluate_settings = None
