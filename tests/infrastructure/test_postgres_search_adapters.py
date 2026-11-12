"""
Unit tests of the Postgres adapters with a fake database (no network):
SQL and settings each search sends, fallbacks, row-to-result mapping,
readiness and the catalog repository's write path.
"""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest

from app.domain.entities import Book
from app.domain.value_objects import SearchFilters
from app.infrastructure.db.postgres_book_catalog_repository import PostgresBookCatalogRepository
from app.infrastructure.db.postgres_rows import book_to_work_params, content_hash
from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository
from app.infrastructure.search.postgres_embeddings_store import PostgresEmbeddingsStore
from app.infrastructure.search.postgres_lexical_search_repository import PostgresLexicalSearchRepository

INFO = {
    "text_configs": ["cat.english_unaccent", "cat.simple_unaccent", "cat.spanish_unaccent"],
    "max_texts_per_work": 2,
    "lexical_index": "lakebase_bm25",
    "vector_index": "hnsw",
    "embedding_model": "intfloat/multilingual-e5-large-instruct",
    "embedding_dimension": 3,
    "works_with_embedding": 5,
}


def make_book(title: str = "Python") -> Book:
    return Book(id=uuid4(), title=title, authors=["A"], description="About Python.", language="en",
                categories=["Computers"], published_date=datetime(2020, 1, 1), source="test",
                source_id=title, created_at=datetime(2025, 1, 1, tzinfo=UTC), updated_at=datetime(2025, 1, 1, tzinfo=UTC))


def work_row(book: Book, **extra: Any) -> Dict[str, Any]:
    params = book_to_work_params(book)
    row = {k: params[k] for k in ("id", "source", "source_id", "title", "authors", "description", "language",
                                  "categories", "published_date", "created_at", "updated_at")}
    row["metadata"] = None
    row.update(extra)
    return row


class FakeDatabase:
    """Records every call; answers catalog_info reads and queued search results."""

    def __init__(self, info: Optional[Dict[str, Any]] = None, results: Optional[List[List[dict]]] = None,
                 configs: Optional[List[str]] = None, stored: Optional[List[dict]] = None) -> None:
        self.info = INFO if info is None else info
        self.results = list(results or [])
        self.configs = configs if configs is not None else ["simple_unaccent", "english_unaccent"]
        self.stored = stored or []
        self.calls: List[Dict[str, Any]] = []

    def query(self, sql: str, params=None, *, settings=None):
        self.calls.append({"kind": "query", "sql": sql, "params": params, "settings": settings or {}})
        if "IS NOT NULL AS found" in sql:
            return [{"found": bool(self.info)}]
        if "FROM cat.catalog_info" in sql:
            return [{"key": k, "value": v} for k, v in self.info.items()]
        if "FROM pg_ts_config" in sql:
            return [{"cfgname": c} for c in self.configs]
        if "content_hash FROM" in sql:
            return self.stored
        return self.results.pop(0) if self.results else []

    def execute(self, sql: str, params=None, *, settings=None):
        self.calls.append({"kind": "execute", "sql": sql, "params": params, "settings": settings or {}})
        return 1

    def searches(self) -> List[Dict[str, Any]]:
        return [c for c in self.calls if "WITH" in c["sql"]]


# --- lexical -------------------------------------------------------------------


def test_bm25_search_sends_one_statement_and_maps_rows():
    a, b = make_book("Python"), make_book("Java")
    db = FakeDatabase(results=[[work_row(a, score=7.5), work_row(b, score=2.0)]])
    repo = PostgresLexicalSearchRepository(db, "cat", "bm25")

    results = repo.search("python programming", max_results=5)

    assert [(r.book.id, r.rank, r.source, r.final_score, r.lexical_score, r.vector_score) for r in results] == [
        (a.id, 1, "lexical", 7.5, 7.5, None), (b.id, 2, "lexical", 2.0, 2.0, None)]
    [call] = db.searches()
    assert "<@> to_bm25query" in call["sql"] and "cat.query_lexemes" in call["sql"]
    assert "JOIN cat.works w ON w.id = t.work_id AND" not in call["sql"]  # no filter join
    # documents per work (2) x results (5) are ranked before grouping by work
    assert call["params"]["n"] == 10 and call["params"]["k"] == 5
    assert call["settings"] == {"lakebase_bm25.default_limit": 10}
    assert call["params"]["query"] == "python programming"
    assert call["params"]["configs"][0] == "cat.simple_unaccent"
    assert call["params"]["bm25_index"] == "cat.work_texts_bm25"
    # a work with texts in several languages keeps its best (most negative) BM25 score
    assert "-min(score) AS score FROM ranked GROUP BY work_id" in call["sql"]


def test_bm25_with_filters_joins_works_and_falls_back_to_exact_scoring_when_starved():
    a = make_book()
    db = FakeDatabase(results=[[work_row(a, score=1.0)], [work_row(a, score=1.0), work_row(make_book("B"), score=0.5)]])
    repo = PostgresLexicalSearchRepository(db, "cat", "bm25")

    results = repo.search("python", max_results=2, filters=SearchFilters(language="en"))

    first, second = db.searches()
    assert "lower(w.language) = lower(%(f_language)s)" in first["sql"]
    assert first["settings"]["lakebase_bm25.default_limit"] == 1000
    assert "lakebase_bm25.enable_scan" not in first["settings"]
    assert second["settings"]["lakebase_bm25.enable_scan"] == "off"
    assert len(results) == 2


def test_ts_rank_cd_uses_the_or_tsquery_and_no_bm25_settings():
    db = FakeDatabase(results=[[]])
    repo = PostgresLexicalSearchRepository(db, "cat", "ts_rank_cd")

    assert repo.search("python", max_results=3) == []
    [call] = db.searches()
    assert "ts_rank_cd(t.tsv, (SELECT tsq FROM q), 1)" in call["sql"] and "@@" in call["sql"]
    assert call["settings"] == {}


def test_lexical_rejects_empty_queries_and_unknown_rankings():
    repo = PostgresLexicalSearchRepository(FakeDatabase(), "cat")
    with pytest.raises(ValueError):
        repo.search("   ")
    with pytest.raises(ValueError):
        PostgresLexicalSearchRepository(FakeDatabase(), "cat", "tf-idf")


def test_bm25_is_not_ready_without_a_bm25_index_and_reads_catalog_info_once():
    no_index = PostgresLexicalSearchRepository(FakeDatabase(info={**INFO, "lexical_index": None}), "cat", "bm25")
    assert no_index.is_ready() is False
    assert PostgresLexicalSearchRepository(FakeDatabase(info={**INFO, "lexical_index": None}), "cat",
                                           "ts_rank_cd").is_ready() is True
    db = FakeDatabase()
    repo = PostgresLexicalSearchRepository(db, "cat")
    assert repo.is_ready() and repo.is_ready()
    assert sum(1 for c in db.calls if "catalog_info" in c["sql"]) == 1
    with pytest.raises(NotImplementedError):
        repo.build_index([])


def test_empty_catalog_is_not_ready():
    assert PostgresLexicalSearchRepository(FakeDatabase(info={}), "cat").is_ready() is False


# --- vector ----------------------------------------------------------------------


def test_exact_vector_search_disables_index_scans_and_returns_cosine_similarity():
    a, b = make_book("A"), make_book("B")
    db = FakeDatabase(results=[[work_row(a, distance=0.1), work_row(b, distance=0.25)]])
    repo = PgVectorSearchRepository(db, "cat", "exact")

    results = repo.search([0.1, 0.2, 0.3], max_results=2)

    assert [(r.book.id, r.rank, r.source) for r in results] == [(a.id, 1, "vector"), (b.id, 2, "vector")]
    assert results[0].final_score == pytest.approx(0.9) and results[0].vector_score == pytest.approx(0.9)
    assert results[0].lexical_score is None
    [call] = db.searches()
    assert call["settings"]["enable_indexscan"] == "off" and call["settings"]["enable_bitmapscan"] == "off"
    assert call["params"]["vector"] == "[0.1,0.2,0.3]"
    assert "<=> %(vector)s::halfvec" in call["sql"] and "ORDER BY distance, id" in call["sql"]


def test_hnsw_settings_and_exact_fallback_for_starved_filtered_queries():
    db = FakeDatabase(results=[[work_row(make_book(), distance=0.1)],
                               [work_row(make_book(), distance=0.1), work_row(make_book("B"), distance=0.2)]])
    repo = PgVectorSearchRepository(db, "cat", "approximate")

    results = repo.search([1.0, 0.0, 0.0], max_results=2, filters=SearchFilters(category="comp"))

    first, second = db.searches()
    assert first["settings"] == {"hnsw.ef_search": 40, "hnsw.iterative_scan": "relaxed_order"}
    assert second["settings"]["enable_indexscan"] == "off"
    assert len(results) == 2


def test_lakebase_ann_prefilters_only_with_filters():
    info = {**INFO, "vector_index": "lakebase_ann"}
    db = FakeDatabase(info=info, results=[[], []])
    repo = PgVectorSearchRepository(db, "cat", "approximate")
    repo.search([1.0, 0.0, 0.0], max_results=1)
    repo.search([1.0, 0.0, 0.0], max_results=1, filters=SearchFilters(language="es"))
    unfiltered, filtered = db.searches()[:2]
    assert unfiltered["settings"] == {}
    assert filtered["settings"] == {"lakebase_ann.prefilter": "on"}


def test_vector_dimension_and_model_checks():
    repo = PgVectorSearchRepository(FakeDatabase(), "cat", "exact")
    with pytest.raises(ValueError):
        repo.search([1.0, 2.0], max_results=3)
    mismatch = PgVectorSearchRepository(FakeDatabase(), "cat", expected_identity="all-MiniLM-L6-v2")
    assert mismatch.is_ready() is False
    same = PgVectorSearchRepository(FakeDatabase(), "cat",
                                    expected_identity="intfloat/multilingual-e5-large-instruct")
    assert same.is_ready() is True
    with pytest.raises(ValueError):
        PgVectorSearchRepository(FakeDatabase(), "cat", mode="fuzzy")


API_IDENTITY = "openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=1024"
API_INFO = {**INFO, "embedding_model": API_IDENTITY, "embedding_dimension": 3,
            "embedding_identity": {"model": "qwen/qwen3-embedding-8b", "provider": "openrouter/deepinfra",
                                   "dimension": 1024, "identity": API_IDENTITY}}


@pytest.mark.parametrize("query_identity", [
    "openrouter:qwen/qwen3-embedding-8b?provider=nebius&dimensions=1024",  # another provider
    "openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=512",  # another dimension
    "qwen/qwen3-embedding-8b",  # the same model, run locally
])
def test_vector_search_refuses_vectors_of_another_encoder_identity(query_identity):
    from app.infrastructure.search.pgvector_search_repository import EmbeddingModelMismatchError

    db = FakeDatabase(info=API_INFO)
    repo = PgVectorSearchRepository(db, "cat", "exact", expected_identity=query_identity)

    with pytest.raises(EmbeddingModelMismatchError) as error:
        repo.search([1.0, 0.0, 0.0], max_results=3)
    assert API_IDENTITY in str(error.value) and query_identity in str(error.value)
    assert "provider openrouter/deepinfra" in str(error.value)
    assert not db.searches() and repo.is_ready() is False


def test_vector_search_accepts_the_same_identity_and_refuses_an_unrecorded_one():
    same = PgVectorSearchRepository(FakeDatabase(info=API_INFO), "cat", "exact", expected_identity=API_IDENTITY)
    assert same.is_ready() is True and same.stored_identity() == API_IDENTITY
    unrecorded = {k: v for k, v in INFO.items() if k != "embedding_model"}
    assert PgVectorSearchRepository(FakeDatabase(info=unrecorded), "cat",
                                    expected_identity=API_IDENTITY).is_ready() is False
    assert PgVectorSearchRepository(FakeDatabase(info=unrecorded), "cat").is_ready() is True  # no check asked


# --- embeddings store ------------------------------------------------------------


class Encoder:
    def generate_query_embedding(self, text):
        return [3.0, 4.0, 0.0]

    def get_dimension(self):
        return 3


def test_embeddings_store_encodes_with_the_encoder_and_stores_unit_vectors():
    db = FakeDatabase()
    store = PostgresEmbeddingsStore(Encoder(), db, "cat")
    assert store.generate_query_embedding("q") == [3.0, 4.0, 0.0]
    book_id = uuid4()
    store.store_embeddings_batch([book_id], [[3.0, 4.0, 0.0]])
    [call] = [c for c in db.calls if c["kind"] == "execute"]
    assert "UPDATE cat.works w SET embedding" in call["sql"]
    assert call["params"]["v0"] == "[0.6,0.8,0]"
    with pytest.raises(ValueError):
        store.store_embeddings_batch([book_id], [[1.0, 2.0]])
    assert store.is_ready() is True
    with pytest.raises(NotImplementedError):
        store.save_index("x")



def test_embeddings_batch_is_one_statement_for_all_ids():
    a, b = uuid4(), uuid4()
    db = FakeDatabase(results=[[{"id": a, "e": "[0.6,0.8,0]"}, {"id": b, "e": "[1,0,0]"}]])
    store = PostgresEmbeddingsStore(Encoder(), db, "cat")

    found = store.get_embeddings_batch([a, b, a, uuid4()])

    assert found == {a: [0.6, 0.8, 0.0], b: [1.0, 0.0, 0.0]}
    [call] = db.calls
    assert "WHERE id = ANY(%(ids)s::uuid[]) AND embedding IS NOT NULL" in call["sql"]
    assert len(call["params"]["ids"]) == 3  # duplicates removed
    assert store.get_embeddings_batch([]) == {} and len(db.calls) == 1  # no statement for no ids


def test_port_default_batch_lookup_uses_the_single_item_method():
    from app.domain.ports import EmbeddingsStore

    class OnlySingle(EmbeddingsStore):
        def __init__(self):
            self.asked = []

        def get_embedding(self, book_id):
            self.asked.append(book_id)
            return None if book_id == missing else [1.0]

    a, missing = uuid4(), uuid4()
    store = OnlySingle()

    assert store.get_embeddings_batch([a, missing, a]) == {a: [1.0]}
    assert store.asked == [a, missing]


# --- catalog repository ------------------------------------------------------------


def test_upsert_skips_unchanged_books():
    book = make_book()
    db = FakeDatabase(stored=[{"id": book.id, "content_hash": content_hash(book)}])
    repo = PostgresBookCatalogRepository(db, "cat")
    assert repo.upsert_many([book]) == 0
    assert not [c for c in db.calls if c["kind"] == "execute"]


def test_upsert_writes_works_texts_tsvectors_and_catalog_facts_in_one_transaction():
    books = [make_book("Python"), make_book("Java")]
    db = FakeDatabase()
    repo = PostgresBookCatalogRepository(db, "cat")

    assert repo.upsert_many(books) == 2

    [write] = [c for c in db.calls if c["kind"] == "execute"]
    sql = write["sql"]
    assert sql.index("INSERT INTO cat.works") < sql.index("DELETE FROM cat.work_texts") < sql.index(
        "INSERT INTO cat.work_texts") < sql.index("INSERT INTO cat.catalog_info")
    assert "ON CONFLICT (id) DO UPDATE" in sql and "WHEN cur.text_hash = EXCLUDED.text_hash" in sql
    assert write["params"]["tc_0_0"] == "cat.english_unaccent"
    assert write["params"]["ids"] == [b.id for b in books]
    assert "cat.simple_unaccent" in write["params"]["configs_used"]


def test_upsert_writes_each_row_once_in_its_final_form():
    """V18: no insert-then-update (no dead rows): tsvectors and vectors go in with the rows."""
    books = [make_book("Python"), make_book("Java")]
    db = FakeDatabase()
    repo = PostgresBookCatalogRepository(db, "cat", store_text_bodies=False)

    assert repo.upsert_many(books, embeddings={books[0].id: [3.0, 4.0]}) == 2

    [write] = [c for c in db.calls if c["kind"] == "execute"]
    sql, params = write["sql"], write["params"]
    assert "UPDATE cat.work_texts" not in sql and "UPDATE cat.works" not in sql
    assert "INSERT INTO cat.work_texts (work_id, language, ts_config, body, tsv)" in sql and "to_tsvector(" in sql
    assert "SELECT v.work_id, v.language, v.ts_config, NULL," in sql  # no stored description bodies
    assert "embedding)" in sql.split("VALUES")[0]
    assert params["embedding_0"] == "[0.6,0.8]" and params["embedding_1"] is None  # unit length; none given
    assert "coalesce(cur.embedding, EXCLUDED.embedding)" in sql


def test_repository_requires_an_applied_schema():
    repo = PostgresBookCatalogRepository(FakeDatabase(configs=[]), "cat")
    with pytest.raises(RuntimeError):
        repo.upsert_many([make_book()])


def test_get_many_without_ids_makes_no_query():
    db = FakeDatabase()
    assert PostgresBookCatalogRepository(db, "cat").get_many([]) == {}
    assert db.calls == []
