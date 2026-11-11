"""
Integration tests of the cloud profile against a real Postgres.

Skipped unless TEST_DATABASE_URL is set (pooled connection string; the
direct one, TEST_DATABASE_URL_UNPOOLED, is used for DDL when given). Use a
throwaway database or Neon branch: every run creates its own schema
(it_v12_<random>) and drops it at the end. The BM25 tests need Neon's
lakebase_text extension and are skipped elsewhere.

The loader runs end to end on a small SQLite catalog whose vectors come from
a FAISS index written by the test (4 dimensions, no model download).
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.domain.entities import Book
from app.domain.value_objects import BookDescription, BookMetadata, SearchFilters

TEST_URL = os.environ.get("TEST_DATABASE_URL", "").strip()
TEST_URL_DIRECT = os.environ.get("TEST_DATABASE_URL_UNPOOLED", "").strip() or TEST_URL

pytestmark = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set (no test Postgres)")

MODEL = "test/fake-embedding-model"
STAMP = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)


def _book(title, language, description, categories, year, source_id, metadata=None):
    return Book(id=uuid4(), title=title, authors=["Test Author"], description=description, language=language,
                categories=categories, published_date=datetime(year, 1, 1) if year else None, source="test",
                source_id=source_id, metadata=metadata, created_at=STAMP, updated_at=STAMP)


BOOKS = [
    _book("Programación en Python", "es", "Aprende programación con ejemplos prácticos para niños.",
          ["Computers"], 2020, "b1"),
    _book("Python Programming for Beginners", "en", "Learn programming with Python from scratch.",
          ["Computers"], 2018, "b2"),
    _book("Cien años de soledad", "es", "Novela del realismo mágico sobre la familia Buendía.",
          ["Fiction"], 1967, "b3"),
    _book("Don Quijote de la Mancha", "es", "Novela de caballerías y parodia de un hidalgo.", ["Fiction"], 1605, "b4",
          metadata=BookMetadata(descriptions=(
              BookDescription("Novela de caballerías y parodia de un hidalgo.", "es", "wikipedia"),
              BookDescription("A novel about a knight errant and his squire.", "en", "wikipedia")))),
    _book("Untitled poems", "en", None, ["Poetry"], None, "b5"),
]
VECTORS = {
    BOOKS[0].id: [0.9, 0.1, 0.0, 0.0],
    BOOKS[1].id: [1.0, 0.0, 0.0, 0.0],
    BOOKS[2].id: [0.0, 1.0, 0.0, 0.0],
    BOOKS[3].id: [0.0, 0.8, 0.2, 0.0],
    BOOKS[4].id: [0.0, 0.0, 0.0, 1.0],
}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from app.infrastructure.db.postgres_connection import PostgresDatabase
    from app.infrastructure.db.postgres_schema import drop_schema, installed_extensions
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
    from app.infrastructure.search.embeddings_store_faiss import (
        INDEX_FILE, MAPPING_FILE, METADATA_FILE, METRIC_INNER_PRODUCT, IndexMetadata)
    from app.infrastructure.search.faiss_runtime import faiss
    from scripts.load_postgres import load

    folder = tmp_path_factory.mktemp("cloud_profile")
    source = folder / "catalog.db"
    SqliteBookCatalogRepository(source).save_many(BOOKS)
    index_dir = folder / "faiss_index"
    index_dir.mkdir()
    ids = sorted(str(b.id) for b in BOOKS)
    matrix = np.array([VECTORS[next(b.id for b in BOOKS if str(b.id) == i)] for i in ids], dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(4)
    index.add(matrix)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    (index_dir / MAPPING_FILE).write_text(json.dumps(ids))
    (index_dir / METADATA_FILE).write_text(json.dumps(IndexMetadata(MODEL, 4, METRIC_INNER_PRODUCT, True, 5, None).to_dict()))

    direct = PostgresDatabase(TEST_URL_DIRECT, pool_max_size=1, statement_timeout_ms=None)
    pooled = PostgresDatabase(TEST_URL)
    schema = f"it_v12_{uuid4().hex[:8]}"
    vector_index = "lakebase_ann" if "lakebase_vector" in {r["name"] for r in direct.query(
        "SELECT name FROM pg_available_extensions")} else "hnsw"
    first = load(direct, source, schema, embedding_model=MODEL, device=None, vector_index=vector_index,
                 reuse_index=index_dir, log=lambda message: None)
    second = load(direct, source, schema, embedding_model=MODEL, device=None, vector_index=vector_index,
                  reuse_index=index_dir, log=lambda message: None)
    has_bm25 = "lakebase_text" in installed_extensions(direct)
    try:
        yield {"direct": direct, "db": pooled, "schema": schema, "first": first, "second": second,
               "has_bm25": has_bm25, "source": source, "index_dir": index_dir, "vector_index": vector_index}
    finally:
        drop_schema(direct, schema)
        direct.close()
        pooled.close()


def _titles(results):
    return [r.book.title for r in results]


# --- loader and catalog ------------------------------------------------------------


def test_loader_loads_everything_once(setup):
    first, second = setup["first"]["facts"], setup["second"]["facts"]
    assert first["works"] == 5 and first["works_with_embedding"] == 5
    assert first["work_texts"] == 6  # Don Quijote has a Spanish and an English text
    assert first["last_load"]["works_changed"] == 5 and first["last_load"]["vectors_reused"] == 5
    assert "last_load" not in second  # nothing changed on the second run


def test_catalog_round_trip(setup):
    from app.infrastructure.db.postgres_book_catalog_repository import PostgresBookCatalogRepository

    repo = PostgresBookCatalogRepository(setup["db"], setup["schema"])
    assert repo.count() == 5
    for book in BOOKS:
        stored = repo.get_by_id(book.id)
        for field in ("title", "authors", "description", "language", "categories", "published_date",
                      "source", "source_id", "metadata", "created_at"):
            assert getattr(stored, field) == getattr(book, field), field
    assert repo.get_by_source_id("test", "b3").id == BOOKS[2].id
    assert set(repo.get_many([BOOKS[0].id, BOOKS[1].id, uuid4()])) == {BOOKS[0].id, BOOKS[1].id}
    assert [b.id for b in repo.iter_all(batch_size=2)] == sorted(b.id for b in BOOKS)
    assert len(repo.get_all(limit=3)) == 3


def test_source_id_clash_raises_value_error(setup):
    from app.infrastructure.db.postgres_book_catalog_repository import PostgresBookCatalogRepository

    repo = PostgresBookCatalogRepository(setup["db"], setup["schema"])
    clash = _book("Other", "en", "x", [], 2000, "b1")
    with pytest.raises(ValueError):
        repo.save(clash)


# --- lexical -----------------------------------------------------------------------


@pytest.fixture
def bm25(setup):
    if not setup["has_bm25"]:
        pytest.skip("lakebase_text (BM25 index) is not available on this server")
    from app.infrastructure.search.postgres_lexical_search_repository import PostgresLexicalSearchRepository

    return PostgresLexicalSearchRepository(setup["db"], setup["schema"], "bm25")


def test_bm25_folds_accents_and_stems_per_language(bm25):
    assert _titles(bm25.search("programacion"))[0] == "Programación en Python"  # accent-insensitive
    assert "Python Programming for Beginners" in _titles(bm25.search("programs"))  # English stemmer
    assert _titles(bm25.search("novelas de caballeria"))[0] == "Don Quijote de la Mancha"  # Spanish stemmer


def test_bm25_ors_the_terms_and_drops_stopwords(bm25):
    results = bm25.search("python soledad")  # no work has both words
    assert {"Cien años de soledad", "Python Programming for Beginners", "Programación en Python"} <= set(_titles(results))
    assert bm25.search("de la the") == []  # only stopwords


def test_a_work_with_two_texts_is_returned_once(bm25):
    titles = _titles(bm25.search("knight novela caballerias"))
    assert titles.count("Don Quijote de la Mancha") == 1 and titles[0] == "Don Quijote de la Mancha"


def test_bm25_filters(bm25):
    assert _titles(bm25.search("python", filters=SearchFilters(language="en"))) == ["Python Programming for Beginners"]
    assert _titles(bm25.search("novela", filters=SearchFilters(category="fic", max_year=1900))) == [
        "Don Quijote de la Mancha"]
    assert bm25.search("poems", filters=SearchFilters(min_year=1900)) == []  # no year: never matches


def test_ts_rank_cd_ranking(setup):
    from app.infrastructure.search.postgres_lexical_search_repository import PostgresLexicalSearchRepository

    repo = PostgresLexicalSearchRepository(setup["db"], setup["schema"], "ts_rank_cd")
    results = repo.search("python programming")
    assert set(_titles(results)) == {"Programación en Python", "Python Programming for Beginners"}
    assert results[0].lexical_score >= results[1].lexical_score > 0


# --- vector ------------------------------------------------------------------------


def _brute_force(query, k):
    q = np.array(query) / np.linalg.norm(query)
    scored = []
    for book in BOOKS:
        v = np.array(VECTORS[book.id]) / np.linalg.norm(VECTORS[book.id])
        scored.append((-(float(q @ v)), str(book.id), book.id))
    return [book_id for _, _, book_id in sorted(scored)[:k]]


@pytest.mark.parametrize("mode", ["exact", "approximate"])
def test_vector_search_matches_brute_force(setup, mode):
    from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository

    repo = PgVectorSearchRepository(setup["db"], setup["schema"], mode, expected_identity=MODEL)
    query = [0.95, 0.05, 0.0, 0.0]
    results = repo.search(query, max_results=3)
    assert [r.book.id for r in results] == _brute_force(query, 3)
    assert results[0].vector_score == pytest.approx(1.0, abs=1e-2)
    filtered = repo.search(query, max_results=2, filters=SearchFilters(language="es"))
    assert [r.book.language for r in filtered] == ["es", "es"]


def test_exact_mode_settings_do_not_leak_through_the_pool(setup):
    from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository

    PgVectorSearchRepository(setup["db"], setup["schema"], "exact").search([1.0, 0.0, 0.0, 0.0], 2)
    assert setup["db"].query("SHOW enable_indexscan")[0]["enable_indexscan"] == "on"


def test_embedding_is_kept_on_metadata_change_and_reset_on_text_change(setup):
    from app.infrastructure.db.postgres_book_catalog_repository import PostgresBookCatalogRepository
    from app.infrastructure.search.postgres_embeddings_store import PostgresEmbeddingsStore

    class Dim:
        def get_dimension(self):
            return 4

    repo = PostgresBookCatalogRepository(setup["direct"], setup["schema"])
    store = PostgresEmbeddingsStore(Dim(), setup["direct"], setup["schema"])  # type: ignore[arg-type]
    poems = repo.get_by_id(BOOKS[4].id)
    poems.metadata = BookMetadata(isbn="123")
    assert repo.upsert_many([poems]) == 1
    assert store.get_embedding(poems.id) is not None
    poems.description = "Now with a description."
    assert repo.upsert_many([poems]) == 1
    assert store.get_embedding(poems.id) is None
    store.store_embeddings_batch([poems.id], [VECTORS[poems.id]])
    assert store.get_embedding(poems.id) == pytest.approx([0.0, 0.0, 0.0, 1.0])


# --- V13: encoder identity, batch lookups, catalog facts ----------------------------


API_IDENTITY = "openrouter:test/api-model?provider=testprovider&dimensions=4"


class FakeApiEncoder:
    """Stands in for OpenRouterEmbeddingEncoder: vectors by book text, counts the texts it encodes."""

    model = "test/api-model"
    upstream_provider = "testprovider"

    def __init__(self):
        self.encoded = 0
        self._by_text = {b.get_searchable_text(): VECTORS[b.id] for b in BOOKS}

    def generate_embeddings_batch(self, texts):
        self.encoded += len(texts)
        return [self._by_text.get(text, [0.5, 0.5, 0.5, 0.5]) for text in texts]

    def generate_query_embedding(self, text):
        return [0.95, 0.05, 0.0, 0.0]

    def get_dimension(self):
        return 4

    def get_model_name(self):
        return API_IDENTITY

    def get_device(self):
        return "api"


@pytest.fixture(scope="module")
def api_setup(setup):
    from app.infrastructure.db.postgres_schema import drop_schema
    from scripts.load_postgres import load

    schema = f"it_v13_{uuid4().hex[:8]}"
    encoder = FakeApiEncoder()
    first = load(setup["direct"], setup["source"], schema, encoder=encoder, vector_index=setup["vector_index"],
                 reuse_index=None, log=lambda message: None)
    encoded_first = encoder.encoded
    second = load(setup["direct"], setup["source"], schema, encoder=encoder, vector_index=setup["vector_index"],
                  reuse_index=None, log=lambda message: None)
    try:
        yield {**setup, "api_schema": schema, "encoder": encoder, "first": first, "second": second,
               "encoded_first": encoded_first, "encoded_second": encoder.encoded - encoded_first}
    finally:
        drop_schema(setup["direct"], schema)


def test_loader_embeds_with_the_api_encoder_and_records_its_identity(api_setup):
    facts = api_setup["first"]["facts"]
    assert api_setup["encoded_first"] == 5 and api_setup["encoded_second"] == 0  # stored vectors reused
    assert facts["embedding_model"] == API_IDENTITY and facts["embedding_dimension"] == 4
    assert facts["embedding_identity"] == {"identity": API_IDENTITY, "model": "test/api-model",
                                           "provider": "openrouter/testprovider", "dimension": 4}


def test_loader_refuses_a_schema_of_another_encoder_identity(setup):
    from scripts.load_postgres import load

    with pytest.raises(ValueError, match="holds vectors of 'test/fake-embedding-model'"):
        load(setup["direct"], setup["source"], setup["schema"], encoder=FakeApiEncoder(),
             vector_index=setup["vector_index"], reuse_index=None, log=lambda message: None)


def test_vector_search_refuses_another_identity_and_serves_its_own(api_setup):
    from app.infrastructure.search.pgvector_search_repository import (
        EmbeddingModelMismatchError,
        PgVectorSearchRepository,
    )

    own = PgVectorSearchRepository(api_setup["db"], api_setup["api_schema"], "exact", expected_identity=API_IDENTITY)
    assert [r.book.id for r in own.search([0.95, 0.05, 0.0, 0.0], 2)] == _brute_force([0.95, 0.05, 0.0, 0.0], 2)
    other = PgVectorSearchRepository(api_setup["db"], api_setup["api_schema"], "exact", expected_identity=MODEL)
    with pytest.raises(EmbeddingModelMismatchError, match=MODEL):
        other.search([1.0, 0.0, 0.0, 0.0], 2)


def test_batch_embeddings_and_categories_in_one_statement_each(setup):
    from app.infrastructure.db.postgres_book_catalog_repository import PostgresBookCatalogRepository
    from app.infrastructure.search.postgres_embeddings_store import PostgresEmbeddingsStore

    store = PostgresEmbeddingsStore(FakeApiEncoder(), setup["db"], setup["schema"])  # type: ignore[arg-type]
    found = store.get_embeddings_batch([b.id for b in BOOKS] + [uuid4()])
    assert set(found) == {b.id for b in BOOKS}
    for book in BOOKS:
        assert found[book.id] == pytest.approx(store.get_embedding(book.id), abs=1e-3)
    repo = PostgresBookCatalogRepository(setup["db"], setup["schema"])
    assert repo.get_categories() == ["Computers", "Fiction", "Poetry"]
