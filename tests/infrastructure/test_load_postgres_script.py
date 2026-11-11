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
