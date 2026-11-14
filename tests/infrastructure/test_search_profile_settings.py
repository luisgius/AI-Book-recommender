"""SearchProfileSettings, database_url() and the cloud wiring of the API dependencies (no network)."""

import inspect

import pytest

from app.api.v1 import dependencies as deps
from app.domain.services import SearchService
from app.domain.value_objects import CANDIDATES_HYBRID, DEFAULT_CANDIDATES
from app.infrastructure.config.settings import SearchProfileSettings, database_url

POOLED = "postgresql://user:secret@ep-x-pooler.example/db"
DIRECT = "postgresql://user:secret@ep-x.example/db"


def test_defaults_keep_the_local_profile():
    settings = SearchProfileSettings.from_env({})
    assert settings == SearchProfileSettings("local", "catalog", "bm25", "approximate", "hybrid")
    assert settings.is_cloud is False
    assert settings.describe() == {"name": "local", "candidates": "hybrid"}


def test_cloud_profile_from_env():
    settings = SearchProfileSettings.from_env({
        "SEARCH_PROFILE": "Cloud", "DATABASE_SCHEMA": "eval177",
        "PG_LEXICAL_RANKING": "TS_RANK_CD", "PG_VECTOR_SEARCH": "exact", "DATABASE_URL": POOLED,
    })
    assert settings.is_cloud
    assert settings.describe() == {"name": "cloud", "candidates": "hybrid", "schema": "eval177",
                                   "lexical_ranking": "ts_rank_cd", "vector_search": "exact"}
    assert "secret" not in repr(settings)  # the connection string is never stored


@pytest.mark.parametrize("env", [
    {"SEARCH_PROFILE": "remote"},
    {"DATABASE_SCHEMA": "Eval-177"},
    {"DATABASE_SCHEMA": "x; DROP SCHEMA public"},
    {"PG_LEXICAL_RANKING": "tfidf"},
    {"PG_VECTOR_SEARCH": "fast"},
    {"SEARCH_CANDIDATES": "bm25"},
])
def test_invalid_values_are_rejected(env):
    with pytest.raises(ValueError):
        SearchProfileSettings.from_env(env)


@pytest.mark.parametrize("profile", ["local", "cloud"])
def test_search_candidates_default_to_hybrid_in_both_profiles_and_vector_is_selectable(profile):
    assert SearchProfileSettings.from_env({"SEARCH_PROFILE": profile}).candidates == "hybrid"
    vector = SearchProfileSettings.from_env({"SEARCH_PROFILE": profile, "SEARCH_CANDIDATES": " Vector "})
    assert vector.candidates == "vector"
    assert vector.describe()["candidates"] == "vector"


def test_database_url_pooled_direct_and_fallback():
    assert database_url({"DATABASE_URL": POOLED}) == POOLED
    assert database_url({"DATABASE_URL": POOLED, "DATABASE_URL_UNPOOLED": DIRECT}, direct=True) == DIRECT
    assert database_url({"DATABASE_URL": POOLED}, direct=True) == POOLED  # no unpooled: fall back


def test_missing_database_url_is_a_clear_error():
    with pytest.raises(ValueError, match="DATABASE_URL is not set"):
        database_url({})
    with pytest.raises(ValueError, match="DATABASE_URL is not set"):
        database_url({"DATABASE_URL_UNPOOLED": DIRECT})  # the application needs the pooled one


class FakeEncoderStore:
    """Stands in for EmbeddingsStoreFaiss: records whether an index was loaded."""

    def __init__(self, model_name="m", device=None, **_):
        self.model_name = model_name
        self.loaded = False

    def load_index(self, path):
        self.loaded = True

    def get_dimension(self):
        return 3


@pytest.fixture
def cloud(monkeypatch):
    deps.reset_dependencies()
    monkeypatch.setattr(deps, "_profile_settings", SearchProfileSettings("cloud", "eval177", "ts_rank_cd", "exact"))
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")
    monkeypatch.setattr("app.infrastructure.search.embeddings_store_faiss.EmbeddingsStoreFaiss", FakeEncoderStore)
    yield
    deps.reset_dependencies()


def test_cloud_profile_wires_the_postgres_adapters_without_connecting(cloud):
    from app.infrastructure.db.postgres_book_catalog_repository import PostgresBookCatalogRepository
    from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository
    from app.infrastructure.search.postgres_embeddings_store import PostgresEmbeddingsStore
    from app.infrastructure.search.postgres_lexical_search_repository import PostgresLexicalSearchRepository

    lexical = deps.get_bm25_repository()
    vector = deps.get_vector_repository()
    store = deps.get_embeddings_store()
    catalog = deps.get_catalog_repository()

    assert isinstance(lexical, PostgresLexicalSearchRepository) and lexical.ranking == "ts_rank_cd"
    assert isinstance(vector, PgVectorSearchRepository) and vector.mode == "exact"
    assert isinstance(store, PostgresEmbeddingsStore) and store.get_dimension() == 3
    assert isinstance(catalog, PostgresBookCatalogRepository)
    assert store._encoder.loaded is False  # no FAISS index in the cloud profile
    # one pool shared by every adapter
    assert lexical._db is vector._db is catalog._db is deps.get_postgres_database()


def test_cloud_profile_without_a_database_url_fails_clearly(monkeypatch):
    deps.reset_dependencies()
    monkeypatch.setattr(deps, "_profile_settings", SearchProfileSettings("cloud"))
    monkeypatch.setattr(deps, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    try:
        with pytest.raises(ValueError, match="DATABASE_URL is not set"):
            deps.get_bm25_repository()
    finally:
        deps.reset_dependencies()


def test_health_reports_the_profile(cloud, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.v1 import search_endpoints
    from app.domain.services import SearchService
    from app.main import app

    class Ready:
        def is_ready(self):
            return True

    # The database check (SELECT 1, V17) would wait for the unreachable test address
    monkeypatch.setattr(search_endpoints, "database_health", lambda: {"reachable": True, "latency_ms": 1.0})
    app.dependency_overrides[deps.get_search_service] = lambda: SearchService(Ready(), Ready(), Ready())
    try:
        body = TestClient(app).get("/api/v1/health").json()
    finally:
        app.dependency_overrides.clear()
    assert body["profile"] == {"name": "cloud", "candidates": "hybrid", "schema": "eval177",
                               "lexical_ranking": "ts_rank_cd", "vector_search": "exact"}
    assert body["database"] == {"reachable": True, "latency_ms": 1.0} and body["status"] == "ok"


@pytest.mark.parametrize("candidates", ["vector", "hybrid"])
def test_the_search_service_takes_the_candidate_mode_of_the_settings(cloud, monkeypatch, candidates):
    monkeypatch.setattr(deps, "_profile_settings",
                        SearchProfileSettings("cloud", "eval177", "ts_rank_cd", "exact", candidates))
    monkeypatch.setattr(deps, "get_llm_client", lambda: None)
    monkeypatch.setattr(deps, "get_reranker", lambda: None)
    monkeypatch.setattr(deps, "get_usage_budget", lambda: None)

    assert deps.get_search_service().candidate_mode == candidates


def test_the_domain_and_the_application_share_one_default_candidate_mode(cloud, monkeypatch):
    """
    One default candidate mode (V30): SearchService's own default, the settings
    without SEARCH_CANDIDATES (both profiles) and the service the application
    wires must agree, whoever builds the service. Fails if any of them gets a
    default of its own.
    """
    domain_default = inspect.signature(SearchService.__init__).parameters["candidates"].default
    assert domain_default == DEFAULT_CANDIDATES == CANDIDATES_HYBRID

    assert SearchProfileSettings().candidates == domain_default
    for profile in ("local", "cloud"):
        assert SearchProfileSettings.from_env({"SEARCH_PROFILE": profile}).candidates == domain_default

    # The application's wiring, with settings read from an environment without SEARCH_CANDIDATES
    monkeypatch.setattr(deps, "_profile_settings", SearchProfileSettings.from_env(
        {"SEARCH_PROFILE": "cloud", "DATABASE_SCHEMA": "eval177"}))
    monkeypatch.setattr(deps, "get_llm_client", lambda: None)
    monkeypatch.setattr(deps, "get_reranker", lambda: None)
    monkeypatch.setattr(deps, "get_usage_budget", lambda: None)
    assert deps.get_search_service().candidate_mode == domain_default
