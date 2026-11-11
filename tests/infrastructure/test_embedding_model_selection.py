"""
Tests for a swappable embedding model in EmbeddingsStoreFaiss.

Covered:
- query vs document prompt conventions per model
- cosine index (normalized vectors + IndexFlatIP) ranks like cosine similarity
- index metadata round trip, and the error on a model mismatch
- legacy indexes without metadata (L2, all-MiniLM-L6-v2) keep loading

A fake SentenceTransformer replaces the real one, so no model is downloaded.
Only the last test class loads the shipped index with the real MiniLM model.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np
import pytest

import app.infrastructure.search.embeddings_store_faiss as store_module
from app.domain.entities import Book
from app.infrastructure.search.embeddings_store_faiss import (
    E5_INSTRUCT_TASK,
    METADATA_FILE,
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    EmbeddingsStoreFaiss,
    IndexMetadata,
    IndexModelMismatchError,
    model_slug,
    prompt_convention_for,
    read_index_metadata,
)
from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository
from app.infrastructure.search.faiss_runtime import faiss  # torch before faiss (macOS)

MINILM = "all-MiniLM-L6-v2"
E5_INSTRUCT = "intfloat/multilingual-e5-large-instruct"
E5_BASE = "intfloat/multilingual-e5-base"
QWEN3 = "Qwen/Qwen3-Embedding-0.6B"

DIM = 8


# -----------------------------------------------------------------------------
# Fake model
# -----------------------------------------------------------------------------


class FakeSentenceTransformer:
    """
    Stand-in for sentence_transformers.SentenceTransformer.

    Returns deterministic vectors that are NOT unit length (so normalization
    is visible) and records every encode() call.
    """

    instances: List["FakeSentenceTransformer"] = []

    def __init__(self, model_name_or_path: str, device: Optional[str] = None, **kwargs: Any):
        self.model_name = model_name_or_path
        self.device = device or "cpu"
        self.calls: List[Dict[str, Any]] = []
        FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return DIM

    def encode(self, sentences, convert_to_numpy: bool = True, prompt_name: Optional[str] = None, **kwargs):
        sentences = [sentences] if isinstance(sentences, str) else list(sentences)
        self.calls.append({"sentences": sentences, "prompt_name": prompt_name})
        # The stored prompt changes the vector, as in the real model.
        return np.array([self._vector((prompt_name or "") + s) for s in sentences], dtype=np.float32)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=DIM) * 3.0


@pytest.fixture
def fake_model(monkeypatch):
    """Patch the adapter to use the fake model; return a store factory."""
    FakeSentenceTransformer.instances = []
    monkeypatch.setattr(store_module, "SentenceTransformer", FakeSentenceTransformer)

    def make_store(model_name: str = MINILM, device: Optional[str] = None) -> EmbeddingsStoreFaiss:
        return EmbeddingsStoreFaiss(model_name=model_name, device=device)

    return make_store


def last_call(store: EmbeddingsStoreFaiss) -> Dict[str, Any]:
    return store._model.calls[-1]  # type: ignore[attr-defined]


def make_book(title: str, book_id: Optional[UUID] = None) -> Book:
    return Book(id=book_id or uuid4(), title=title, authors=["A"], source="test", source_id=title)


def build_store_with(store: EmbeddingsStoreFaiss, vectors: List[np.ndarray]) -> List[Book]:
    """Store the given vectors as document embeddings and build the index."""
    books = [make_book(f"Book {i}") for i in range(len(vectors))]
    for book, vector in zip(books, vectors):
        store.store_embedding(book.id, [float(x) for x in vector])
    store.build_index()
    return books


# -----------------------------------------------------------------------------
# Prompt conventions
# -----------------------------------------------------------------------------


class TestPromptConventions:
    def test_minilm_uses_no_prompt(self, fake_model):
        store = fake_model(MINILM)

        query = store.generate_query_embedding("novela de misterio")
        assert last_call(store) == {"sentences": ["novela de misterio"], "prompt_name": None}

        document = store.generate_embedding("novela de misterio")
        assert last_call(store) == {"sentences": ["novela de misterio"], "prompt_name": None}

        # Symmetric model: the query vector is the document vector.
        assert query == document

    def test_e5_instruct_adds_the_instruction_to_queries_only(self, fake_model):
        store = fake_model(E5_INSTRUCT)

        query = store.generate_query_embedding("novela de misterio")
        assert last_call(store)["sentences"] == [
            "Instruct: Given a book search query, retrieve relevant books\n"
            "Query: novela de misterio"
        ]
        assert E5_INSTRUCT_TASK == "Given a book search query, retrieve relevant books"

        document = store.generate_embedding("novela de misterio")
        assert last_call(store)["sentences"] == ["novela de misterio"]

        store.generate_embeddings_batch(["uno", "dos"])
        assert last_call(store)["sentences"] == ["uno", "dos"]

        assert query != document

    def test_e5_uses_query_and_passage_prefixes(self, fake_model):
        store = fake_model(E5_BASE)

        store.generate_query_embedding("dragones")
        assert last_call(store)["sentences"] == ["query: dragones"]

        store.generate_embedding("Un libro de dragones")
        assert last_call(store)["sentences"] == ["passage: Un libro de dragones"]

        store.generate_embeddings_batch(["uno", "dos"])
        assert last_call(store)["sentences"] == ["passage: uno", "passage: dos"]

    @pytest.mark.parametrize("name", ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large"])
    def test_other_e5_sizes_share_the_prefixes(self, name):
        convention = prompt_convention_for(name)
        assert (convention.query_prefix, convention.document_prefix) == ("query: ", "passage: ")

    def test_qwen3_uses_the_stored_query_prompt(self, fake_model):
        store = fake_model(QWEN3)

        store.generate_query_embedding("dragones")
        assert last_call(store) == {"sentences": ["dragones"], "prompt_name": "query"}

        store.generate_embedding("Un libro de dragones")
        assert last_call(store) == {"sentences": ["Un libro de dragones"], "prompt_name": None}

    def test_unknown_model_uses_no_prompt(self, fake_model):
        store = fake_model("some-org/some-model")

        store.generate_query_embedding("dragones")
        assert last_call(store) == {"sentences": ["dragones"], "prompt_name": None}

    def test_model_names_are_compared_canonically(self):
        assert prompt_convention_for("sentence-transformers/all-MiniLM-L6-v2") == prompt_convention_for(MINILM)
        assert prompt_convention_for("qwen/qwen3-embedding-0.6b") == prompt_convention_for(QWEN3)

    def test_empty_query_raises(self, fake_model):
        store = fake_model(E5_INSTRUCT)
        with pytest.raises(ValueError, match="empty"):
            store.generate_query_embedding("   ")

    def test_device_is_passed_to_the_model(self, fake_model):
        store = fake_model(MINILM, device="cpu")
        assert FakeSentenceTransformer.instances[-1].device == "cpu"
        assert store.get_device() == "cpu"

    def test_model_slug_is_file_system_friendly(self):
        assert model_slug(E5_INSTRUCT) == "intfloat_multilingual-e5-large-instruct"
        assert model_slug(QWEN3) == "qwen_qwen3-embedding-0.6b"
        assert model_slug(MINILM) == "all-minilm-l6-v2"


# -----------------------------------------------------------------------------
# Cosine index
# -----------------------------------------------------------------------------


class TestCosineIndex:
    def test_new_index_is_inner_product_over_unit_vectors(self, fake_model):
        store = fake_model(E5_INSTRUCT)
        vectors = [FakeSentenceTransformer._vector(f"doc {i}") for i in range(5)]
        build_store_with(store, vectors)

        index = store.get_index()
        assert index.metric_type == faiss.METRIC_INNER_PRODUCT
        stored = index.reconstruct_n(0, index.ntotal)
        np.testing.assert_allclose(np.linalg.norm(stored, axis=1), 1.0, rtol=1e-5)

    def test_ranking_and_scores_equal_cosine_similarity(self, fake_model):
        store = fake_model(E5_INSTRUCT)
        query = np.zeros(DIM)
        query[0] = 2.0  # not unit length on purpose

        # A long vector far in angle and a short vector close in angle:
        # a raw dot product prefers the long one, cosine the short one.
        long_far = np.zeros(DIM)
        long_far[[0, 1]] = [10 * np.cos(np.pi / 3), 10 * np.sin(np.pi / 3)]
        short_close = np.zeros(DIM)
        short_close[[0, 1]] = [np.cos(0.1), np.sin(0.1)]
        others = [FakeSentenceTransformer._vector(f"doc {i}") for i in range(6)]
        vectors = [long_far, short_close] + others

        books = build_store_with(store, vectors)
        repo = FaissVectorSearchRepository(store, books)
        results = repo.search([float(x) for x in query], max_results=len(vectors))

        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

        expected = sorted(
            ((book.id, cosine(query, v)) for book, v in zip(books, vectors)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        assert [r.book.id for r in results] == [book_id for book_id, _ in expected]
        np.testing.assert_allclose(
            [r.final_score for r in results], [score for _, score in expected], atol=1e-5
        )
        # The raw dot product would have put the long vector first.
        assert float(query @ long_far) > float(query @ short_close)
        assert results[0].book.id == books[1].id

    def test_find_similar_style_query_with_a_stored_vector(self, fake_model):
        """A stored (raw) document vector used as query finds itself with cosine 1."""
        store = fake_model(MINILM)
        vectors = [FakeSentenceTransformer._vector(f"doc {i}") for i in range(4)]
        books = build_store_with(store, vectors)
        repo = FaissVectorSearchRepository(store, books)

        results = repo.search(store.get_embedding(books[2].id), max_results=1)

        assert results[0].book.id == books[2].id
        assert results[0].final_score == pytest.approx(1.0, abs=1e-5)


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------


class TestIndexMetadata:
    def test_metadata_round_trip(self, fake_model, tmp_path: Path):
        store = fake_model(E5_INSTRUCT)
        build_store_with(store, [FakeSentenceTransformer._vector(f"doc {i}") for i in range(3)])
        store.save_index(str(tmp_path))

        raw = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
        assert raw["model_name"] == E5_INSTRUCT
        assert raw["dimension"] == DIM
        assert raw["metric"] == METRIC_INNER_PRODUCT
        assert raw["normalized"] is True
        assert raw["vector_count"] == 3
        assert datetime.fromisoformat(raw["created_at"]).tzinfo is not None

        assert read_index_metadata(str(tmp_path)) == IndexMetadata.from_dict(raw)

        loaded = fake_model(E5_INSTRUCT)
        loaded.load_index(str(tmp_path))
        assert loaded.get_index_metadata() == store.get_index_metadata()
        assert loaded.get_index().metric_type == faiss.METRIC_INNER_PRODUCT

    def test_model_mismatch_raises_a_clear_error(self, fake_model, tmp_path: Path):
        store = fake_model(E5_INSTRUCT)
        build_store_with(store, [FakeSentenceTransformer._vector("doc")])
        store.save_index(str(tmp_path))

        other = fake_model(QWEN3)
        with pytest.raises(IndexModelMismatchError) as error:
            other.load_index(str(tmp_path))

        message = str(error.value)
        assert E5_INSTRUCT in message and QWEN3 in message
        assert "Rebuild the index" in message
        assert isinstance(error.value, ValueError)
        # Nothing was loaded.
        assert other.get_index() is None
        assert other.get_index_metadata() is None

    def test_same_model_with_another_spelling_loads(self, fake_model, tmp_path: Path):
        store = fake_model(MINILM)
        build_store_with(store, [FakeSentenceTransformer._vector("doc")])
        store.save_index(str(tmp_path))

        fake_model("sentence-transformers/all-MiniLM-L6-v2").load_index(str(tmp_path))

    def test_read_metadata_of_legacy_index_is_none(self, tmp_path: Path):
        assert read_index_metadata(str(tmp_path)) is None


# -----------------------------------------------------------------------------
# Legacy indexes (no metadata)
# -----------------------------------------------------------------------------


def write_legacy_index(path: Path, vectors: np.ndarray, ids: List[str]) -> None:
    """Write an index the way the code did before metadata existed."""
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype(np.float32))  # type: ignore[call-arg]  (SWIG signature)
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path / "faiss_index.bin"))
    (path / "faiss_id_mapping.json").write_text(json.dumps(ids), encoding="utf-8")
    embeddings = {book_id: v.tolist() for book_id, v in zip(ids, vectors)}
    (path / "faiss_embeddings.json").write_text(json.dumps(embeddings), encoding="utf-8")


class TestLegacyIndex:
    def test_legacy_index_loads_as_minilm_l2(self, fake_model, tmp_path: Path):
        vectors = np.array([FakeSentenceTransformer._vector(f"doc {i}") for i in range(5)])
        books = [make_book(f"Book {i}") for i in range(5)]
        write_legacy_index(tmp_path, vectors, [str(b.id) for b in books])
        files_before = sorted(p.name for p in tmp_path.iterdir())

        store = fake_model(MINILM)
        store.load_index(str(tmp_path))

        metadata = store.get_index_metadata()
        assert metadata.model_name == MINILM
        assert metadata.metric == METRIC_L2
        assert metadata.normalized is False
        assert metadata.created_at is None
        assert store.get_index().metric_type == faiss.METRIC_L2
        # Loading never writes a metadata file next to a legacy index.
        assert sorted(p.name for p in tmp_path.iterdir()) == files_before

        # Scores keep the legacy formula: 1 / (1 + squared L2 distance).
        query = FakeSentenceTransformer._vector("query")
        results = FaissVectorSearchRepository(store, books).search(query.tolist(), max_results=5)
        squared = ((vectors - query) ** 2).sum(axis=1)
        order = np.argsort(squared)
        assert [r.book.id for r in results] == [books[i].id for i in order]
        np.testing.assert_allclose(
            [r.final_score for r in results], 1.0 / (1.0 + squared[order]), rtol=1e-4
        )

    def test_legacy_index_with_another_model_raises(self, fake_model, tmp_path: Path):
        vectors = np.array([FakeSentenceTransformer._vector("doc")])
        write_legacy_index(tmp_path, vectors, [str(uuid4())])

        with pytest.raises(IndexModelMismatchError, match="legacy index without metadata"):
            fake_model(E5_INSTRUCT).load_index(str(tmp_path))


SHIPPED_INDEX = Path(__file__).resolve().parents[2] / "data" / "indexes" / "faiss_index"


@pytest.mark.skipif(not SHIPPED_INDEX.exists(), reason="shipped index not available")
class TestShippedIndex:
    """The shipped index (no metadata) loads with the real default model."""

    def test_shipped_index_loads_unchanged(self):
        files_before = {p.name: p.stat().st_mtime for p in SHIPPED_INDEX.iterdir()}

        store = EmbeddingsStoreFaiss()
        store.load_index(str(SHIPPED_INDEX))

        metadata, index = store.get_index_metadata(), store.get_index()
        assert metadata is not None and index is not None
        assert metadata.metric == METRIC_L2
        assert index.metric_type == faiss.METRIC_L2
        assert {p.name: p.stat().st_mtime for p in SHIPPED_INDEX.iterdir()} == files_before

        query = store.generate_query_embedding("novelas de misterio ambientadas en la época victoriana")
        assert query == store.generate_embedding("novelas de misterio ambientadas en la época victoriana")
        distances, indices = index.search(np.array([query], dtype=np.float32), 5)  # type: ignore[call-arg]
        mapping = store.get_id_mapping()

        # One stand-in Book per indexed id, so the repository can hydrate results.
        repo_books = [make_book(book_id, book_id=UUID(book_id)) for book_id in mapping]
        results = FaissVectorSearchRepository(store, repo_books).search(query, max_results=5)

        assert [str(r.book.id) for r in results] == [mapping[i] for i in indices[0]]
        np.testing.assert_allclose(
            [r.final_score for r in results], 1.0 / (1.0 + distances[0]), rtol=1e-6
        )
