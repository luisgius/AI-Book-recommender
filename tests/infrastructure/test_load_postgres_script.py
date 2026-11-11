"""Helpers of scripts/load_postgres.py that need no database."""

import json
import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

from app.domain.entities import Book
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
from scripts.load_postgres import ReadOnlySqliteCatalog, faiss_vectors, find_reusable_index


def _books(n=3):
    return [Book(id=uuid4(), title=f"Book {i}", authors=["A"], source="test", source_id=str(i),
                 created_at=datetime(2025, 1, 1, tzinfo=UTC), updated_at=datetime(2025, 1, 1, tzinfo=UTC))
            for i in range(n)]


def _write_index(folder, ids, model, dim=3):
    from app.infrastructure.search.embeddings_store_faiss import (
        INDEX_FILE, MAPPING_FILE, METADATA_FILE, METRIC_INNER_PRODUCT, IndexMetadata)
    from app.infrastructure.search.faiss_runtime import faiss

    folder.mkdir(parents=True)
    matrix = np.eye(len(ids), dim, dtype=np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, str(folder / INDEX_FILE))
    (folder / MAPPING_FILE).write_text(json.dumps([str(i) for i in ids]))
    (folder / METADATA_FILE).write_text(
        json.dumps(IndexMetadata(model, dim, METRIC_INNER_PRODUCT, True, len(ids), None).to_dict()))
    return folder


def test_source_catalog_is_opened_read_only(tmp_path):
    path = tmp_path / "catalog.db"
    books = _books()
    SqliteBookCatalogRepository(path).save_many(books)
    before = path.stat().st_mtime_ns
    catalog = ReadOnlySqliteCatalog(path)
    assert sorted(catalog.all_ids()) == sorted(b.id for b in books)
    assert {b.id for b in catalog.iter_all(batch_size=2)} == {b.id for b in books}
    with pytest.raises(sqlite3.OperationalError):
        with catalog._get_connection() as conn:
            conn.execute("DELETE FROM books")
    assert path.stat().st_mtime_ns == before
    with pytest.raises(FileNotFoundError):
        ReadOnlySqliteCatalog(tmp_path / "missing.db")


def test_reusable_index_needs_the_same_model_and_catalog(tmp_path):
    ids = [b.id for b in _books()]
    same = _write_index(tmp_path / "same", ids, "org/model")
    other_model = _write_index(tmp_path / "other_model", ids, "org/another")
    other_catalog = _write_index(tmp_path / "other_catalog", ids[:2], "org/model")

    assert find_reusable_index("org/model", ids, [other_model, other_catalog, same], None) == same
    assert find_reusable_index("org/model", ids, [other_model, other_catalog], None) is None
    assert find_reusable_index("org/model", ids, [], other_catalog) == other_catalog  # explicit: model only
    with pytest.raises(ValueError):
        find_reusable_index("org/model", ids, [], other_model)

    vectors = faiss_vectors(same)
    assert set(vectors) == set(ids) and len(next(iter(vectors.values()))) == 3


# --- encoder identity (V13) ----------------------------------------------------------


def test_identity_record_of_a_local_model_and_of_the_api_encoder():
    from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder
    from scripts.load_postgres import identity_record

    assert identity_record("intfloat/multilingual-e5-large-instruct", 1024) == {
        "identity": "intfloat/multilingual-e5-large-instruct", "model": "intfloat/multilingual-e5-large-instruct",
        "provider": "local", "dimension": 1024}
    api = OpenRouterEmbeddingEncoder("qwen/qwen3-embedding-8b", dimensions=1024, upstream_provider="deepinfra",
                                     api_key="test-key")
    assert identity_record(api.get_model_name(), api.get_dimension(), api) == {
        "identity": "openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=1024",
        "model": "qwen/qwen3-embedding-8b", "provider": "openrouter/deepinfra", "dimension": 1024}


def test_loader_encoder_follows_the_api_settings(monkeypatch):
    from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder
    from scripts.load_postgres import build_document_encoder, embedding_settings

    cloud = embedding_settings({"SEARCH_PROFILE": "cloud", "OPENROUTER_API_KEY": "k"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    encoder = build_document_encoder(cloud)
    assert isinstance(encoder, OpenRouterEmbeddingEncoder)
    assert encoder.get_model_name() == "openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=1024"
    local = embedding_settings({}, model_override="intfloat/multilingual-e5-large-instruct")
    assert local.provider == "local" and local.model == "intfloat/multilingual-e5-large-instruct"
    assert build_document_encoder(local) is None  # load() builds the local model only if needed


def test_load_needs_exactly_one_encoder_source(tmp_path):
    from scripts.load_postgres import load

    with pytest.raises(ValueError, match="either embedding_model"):
        load(None, tmp_path / "x.db", "s", vector_index="hnsw", reuse_index=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="either embedding_model"):
        load(None, tmp_path / "x.db", "s", embedding_model="m", encoder=object(),  # type: ignore[arg-type]
             vector_index="hnsw", reuse_index=None)
