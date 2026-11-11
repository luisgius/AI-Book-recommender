"""
FastAPI dependencies for dependency injection.

This module provides singleton instances of repositories and services
for use with FastAPI's Depends() system.

Note: We use module-level singletons instead of @lru_cache with Depends()
parameters, which is an antipattern that can cause unexpected behavior.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from app.domain.ports import (
    BookCatalogRepository,
    LexicalSearchRepository,
    VectorSearchRepository,
    EmbeddingsStore,
    LLMClient,
    Reranker,
    SearchAgent,
)
from app.domain.services import SearchService
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
from app.infrastructure.search.bm25_search_repository import BM25SearchRepository
from app.infrastructure.search.cross_encoder_reranker import CrossEncoderReranker
from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository
from app.infrastructure.config.settings import (
    EmbeddingSettings,
    LLMSettings,
    RerankerSettings,
    SearchProfileSettings,
    database_url,
)

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
_reranker_settings: Optional[RerankerSettings] = None
_reranker: Optional[Reranker] = None
_reranker_initialized: bool = False


def get_search_profile_settings() -> SearchProfileSettings:
    """
    Provide the search profile, read once from the environment (and .env).

    SEARCH_PROFILE=local (default) keeps the evaluated system: SQLite
    catalog, BM25 and FAISS indexes on disk. SEARCH_PROFILE=cloud serves the
    catalog, lexical and vector search from Postgres (DATABASE_SCHEMA). In
    both profiles the query embedding comes from the configured embedding
    model and the reranker and LLM wiring are the same.
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
            _catalog_repository = SqliteBookCatalogRepository(DB_PATH)
    return _catalog_repository


def get_bm25_repository() -> LexicalSearchRepository:
    """Provide a singleton instance of the lexical (BM25) repository."""
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


def build_embeddings_store(settings: EmbeddingSettings) -> EmbeddingsStoreFaiss:
    """
    The FAISS store with the encoder that EMBEDDING_PROVIDER selects.

    "local" (default) loads the sentence-transformers model EMBEDDING_MODEL in
    this process, as before. "openrouter" encodes through OpenRouter's
    embeddings API (OpenRouterEmbeddingEncoder: EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS, EMBEDDING_OPENROUTER_PROVIDER) and loads no model;
    the index must have been built with the same API model, provider and
    dimension, or loading it raises IndexModelMismatchError.
    """
    if settings.provider == "openrouter":
        # Imported here so the local profile never needs the API adapter
        from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder

        return EmbeddingsStoreFaiss(encoder=OpenRouterEmbeddingEncoder.from_settings(settings))
    return EmbeddingsStoreFaiss(model_name=settings.model, device=settings.device)


def get_embeddings_store() -> EmbeddingsStore:
    """
    Provide a singleton instance of the embeddings store.

    The model comes from EMBEDDING_MODEL. Loading an index built with a
    different model raises an error that asks to rebuild the index.
    """
    global _embeddings_store
    if _embeddings_store is None:
        settings = get_embedding_settings()
        store = build_embeddings_store(settings)
        profile = get_search_profile_settings()
        if profile.is_cloud:
            # The model only encodes; document vectors live in Postgres.
            from app.infrastructure.search.postgres_embeddings_store import PostgresEmbeddingsStore

            _embeddings_store = PostgresEmbeddingsStore(store, get_postgres_database(), profile.schema)
            return _embeddings_store
        faiss_path = INDEXES_DIR / "faiss_index"
        if faiss_path.exists():
            store.load_index(str(faiss_path))
        _embeddings_store = store
    return _embeddings_store


def get_vector_repository() -> VectorSearchRepository:
    """Provide a singleton instance of the vector search repository."""
    global _vector_repository
    if _vector_repository is None:
        profile = get_search_profile_settings()
        if profile.is_cloud:
            from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository

            _vector_repository = PgVectorSearchRepository(
                get_postgres_database(),
                profile.schema,
                mode=profile.vector_search,
                expected_model=get_embedding_settings().model,
            )
            return _vector_repository
        catalog_repo = get_catalog_repository()
        embeddings_store = get_embeddings_store()
        books = catalog_repo.get_all()
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

    The LLM is optional: if LLM_PROVIDER=none or the client fails to
    initialise, the API keeps serving plain hybrid search and logs why.
    The catalog categories are passed so filters extracted from queries are
    validated against real categories.
    """
    global _llm_client, _llm_initialized
    if _llm_initialized:
        return _llm_client
    _llm_initialized = True

    settings = get_llm_settings()
    if not settings.enabled:
        logger.info("LLM disabled (LLM_PROVIDER=none): understanding and explanations off")
        return None

    try:
        # Imported here so the API can start without LangChain when the LLM is off
        from app.infrastructure.llm.langchain_llm_client import LangChainLLMClient

        categories = sorted(
            {category for book in get_catalog_repository().get_all() for category in book.categories}
        )
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

    Re-ranking is off unless RERANKER_MODEL is set. RERANKER_PROVIDER=local
    (default) gives the cross-encoder, whose model is loaded on the first
    search, so a wrong model name shows up then (logged by the search
    service, which keeps the fused order) and in /health.
    RERANKER_PROVIDER=openrouter gives ApiReranker (OpenRouter's rerank API,
    RERANKER_OPENROUTER_PROVIDER to pin a provider); a missing
    OPENROUTER_API_KEY disables re-ranking with a warning.
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

            _reranker = ApiReranker.from_settings(settings)
            return _reranker
        _reranker = CrossEncoderReranker(model_name=settings.model, device=settings.device)
    except Exception as e:
        logger.warning(f"Reranker unavailable ({settings.model}): {e}")
        _reranker = None
    return _reranker


def get_search_service() -> SearchService:
    """Provide the Search Service with all dependencies wired."""
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
        )
    return _search_service


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
    global _reranker_settings, _reranker, _reranker_initialized
    global _profile_settings, _postgres_database

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
    _reranker_settings = None
    _reranker = None
    _reranker_initialized = False
