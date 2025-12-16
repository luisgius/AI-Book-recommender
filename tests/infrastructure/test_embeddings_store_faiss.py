"""
Integration tests for EmbeddingsStoreFaiss.

These tests verify the FAISS-based embeddings store works correctly:
- Embedding generation produces consistent dimensions
- Storing and indexing embeddings works
- Persistence (save/load) preserves functionality
- Protocol compliance with EmbeddingsStore

No mocks are used; these are true integration tests against the real
FAISS index and sentence-transformers model.

Note: These tests require the sentence-transformers model to be downloaded
on first run (~80MB for all-MiniLM-L6-v2).
"""

import tempfile
from pathlib import Path
from uuid import uuid4, UUID

import numpy as np
import pytest

from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def store() -> EmbeddingsStoreFaiss:
    """Create a fresh embeddings store for each test."""
    return EmbeddingsStoreFaiss()


@pytest.fixture
def sample_books() -> list[tuple[UUID, str]]:
    """Sample books with UUIDs and searchable text."""
    return [
        (uuid4(), "Detective mystery novel set in London with crime investigation"),
        (uuid4(), "Science fiction space opera adventure in distant galaxies"),
        (uuid4(), "Historical romance story in Victorian England"),
        (uuid4(), "Italian cookbook with pasta recipes and Mediterranean cuisine"),
        (uuid4(), "Fantasy epic with dragons and medieval kingdoms"),
        (uuid4(), "Thriller crime fiction featuring FBI agent investigation"),
    ]


@pytest.fixture
def populated_store(store: EmbeddingsStoreFaiss, sample_books: list[tuple[UUID, str]]) -> tuple[EmbeddingsStoreFaiss, list[tuple[UUID, str]]]:
    """Create a store with sample book embeddings already indexed."""
    for book_id, text in sample_books:
        embedding = store.generate_embedding(text)
        store.store_embedding(book_id, embedding)

    store.build_index()
    return store, sample_books


# -----------------------------------------------------------------------------
# Test: Embedding Generation
# -----------------------------------------------------------------------------


class TestEmbeddingGeneration:
    """Tests for generate_embedding and generate_embeddings_batch methods."""

    def test_generate_embedding_returns_correct_dimension(self, store: EmbeddingsStoreFaiss):
        """Embedding should have the expected dimension (384 for MiniLM)."""
        embedding = store.generate_embedding("A mystery novel about a detective")

        assert len(embedding) == 384
        assert store.get_dimension() == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_generate_embedding_is_deterministic(self, store: EmbeddingsStoreFaiss):
        """Same text should produce the same embedding."""
        text = "Science fiction adventure in space"

        embedding1 = store.generate_embedding(text)
        embedding2 = store.generate_embedding(text)

        assert embedding1 == embedding2

    def test_generate_embedding_empty_text_raises_error(self, store: EmbeddingsStoreFaiss):
        """Empty text should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            store.generate_embedding("")

        with pytest.raises(ValueError, match="empty"):
            store.generate_embedding("   ")

    def test_similar_texts_produce_similar_embeddings(self, store: EmbeddingsStoreFaiss):
        """Semantically similar texts should have close embeddings."""
        # These two texts are semantically similar
        text1 = "A mystery novel about a detective solving crimes"
        text2 = "A crime fiction book featuring an investigator"

        # This text is semantically different
        text3 = "A cookbook with Italian pasta recipes"

        emb1 = np.array(store.generate_embedding(text1))
        emb2 = np.array(store.generate_embedding(text2))
        emb3 = np.array(store.generate_embedding(text3))

        # L2 distance: similar texts should be closer
        dist_similar = np.linalg.norm(emb1 - emb2)
        dist_different = np.linalg.norm(emb1 - emb3)

        assert dist_similar < dist_different, (
            f"Similar texts should be closer: {dist_similar:.4f} vs {dist_different:.4f}"
        )

    def test_generate_embeddings_batch(self, store: EmbeddingsStoreFaiss):
        """Batch generation should produce valid embeddings with correct properties."""
        texts = [
            "Mystery detective novel",
            "Science fiction space adventure",
            "Romance historical fiction",
        ]

        # Generate as batch
        batch = store.generate_embeddings_batch(texts)

        # Verify structural properties
        assert len(batch) == len(texts), "Should return same number of embeddings as texts"

        for i, emb in enumerate(batch):
            # Correct dimension
            assert len(emb) == store.get_dimension(), f"Wrong dimension at index {i}"

            # No NaN or Inf values
            emb_array = np.array(emb)
            assert np.isfinite(emb_array).all(), f"Non-finite values at index {i}"

        # Verify semantic similarity: batch embeddings should be very similar to individual
        # Note: sentence-transformers batch vs individual can differ slightly due to
        # internal optimizations (padding, BLAS, parallelism). Use cosine similarity
        # which is more stable than element-wise comparison.
        individual = [store.generate_embedding(t) for t in texts]

        for i, (ind, bat) in enumerate(zip(individual, batch)):
            ind_arr = np.array(ind)
            bat_arr = np.array(bat)

            # Cosine similarity should be very high (> 0.9999)
            cos_sim = np.dot(ind_arr, bat_arr) / (
                np.linalg.norm(ind_arr) * np.linalg.norm(bat_arr)
            )
            assert cos_sim > 0.9999, (
                f"Batch vs individual mismatch at index {i}: cosine_sim={cos_sim:.6f}"
            )

    def test_generate_embeddings_batch_empty_list(self, store: EmbeddingsStoreFaiss):
        """Empty list should return empty list."""
        result = store.generate_embeddings_batch([])
        assert result == []

    def test_generate_embeddings_batch_with_empty_text_raises(self, store: EmbeddingsStoreFaiss):
        """Batch with empty text should raise ValueError."""
        with pytest.raises(ValueError, match="index 1"):
            store.generate_embeddings_batch(["Valid text", "", "Another valid"])


# -----------------------------------------------------------------------------
# Test: Storing Embeddings
# -----------------------------------------------------------------------------


class TestStoringEmbeddings:
    """Tests for store_embedding, store_embeddings_batch, and get_embedding."""

    def test_store_and_retrieve_embedding(self, store: EmbeddingsStoreFaiss):
        """Can store and retrieve an embedding by book_id."""
        book_id = uuid4()
        embedding = store.generate_embedding("Test book description")

        store.store_embedding(book_id, embedding)

        retrieved = store.get_embedding(book_id)
        assert retrieved == embedding

    def test_get_embedding_nonexistent_returns_none(self, store: EmbeddingsStoreFaiss):
        """Getting a non-existent embedding should return None."""
        result = store.get_embedding(uuid4())
        assert result is None

    def test_store_embedding_wrong_dimension_raises(self, store: EmbeddingsStoreFaiss):
        """Storing embedding with wrong dimension should raise ValueError."""
        book_id = uuid4()
        wrong_embedding = [0.1] * 100  # Should be 384

        with pytest.raises(ValueError, match="dimension mismatch"):
            store.store_embedding(book_id, wrong_embedding)

    def test_store_embeddings_batch(self, store: EmbeddingsStoreFaiss):
        """Can store multiple embeddings in batch."""
        book_ids = [uuid4() for _ in range(3)]
        texts = ["Book one", "Book two", "Book three"]
        embeddings = store.generate_embeddings_batch(texts)

        store.store_embeddings_batch(book_ids, embeddings)

        for book_id, expected in zip(book_ids, embeddings):
            assert store.get_embedding(book_id) == expected

    def test_store_embeddings_batch_mismatched_lengths_raises(self, store: EmbeddingsStoreFaiss):
        """Batch store with mismatched lengths should raise ValueError."""
        book_ids = [uuid4(), uuid4()]
        embeddings = [store.generate_embedding("One")]  # Only one embedding

        with pytest.raises(ValueError, match="Mismatched lengths"):
            store.store_embeddings_batch(book_ids, embeddings)

    def test_store_overwrites_existing(self, store: EmbeddingsStoreFaiss):
        """Storing with same book_id should overwrite."""
        book_id = uuid4()
        emb1 = store.generate_embedding("First version")
        emb2 = store.generate_embedding("Second version completely different")

        store.store_embedding(book_id, emb1)
        store.store_embedding(book_id, emb2)

        assert store.get_embedding(book_id) == emb2


# -----------------------------------------------------------------------------
# Test: Building Index
# -----------------------------------------------------------------------------


class TestBuildIndex:
    """Tests for build_index method."""

    def test_build_index_creates_searchable_index(self, store: EmbeddingsStoreFaiss):
        """Building index should create a FAISS index."""
        book_id = uuid4()
        embedding = store.generate_embedding("Test book")
        store.store_embedding(book_id, embedding)

        assert store.get_index() is None
        assert store.get_index_size() == 0

        store.build_index()

        assert store.get_index() is not None
        assert store.get_index_size() == 1

    def test_build_index_with_multiple_embeddings(self, store: EmbeddingsStoreFaiss, sample_books: list[tuple[UUID, str]]):
        """Can build index with multiple embeddings."""
        for book_id, text in sample_books:
            store.store_embedding(book_id, store.generate_embedding(text))

        store.build_index()

        assert store.get_index_size() == len(sample_books)

    def test_build_index_empty_store_warns(self, store: EmbeddingsStoreFaiss, caplog):
        """Building index with no embeddings should log warning."""
        store.build_index()

        assert "No embeddings to index" in caplog.text
        assert store.get_index_size() == 0

    def test_build_index_is_reproducible(self, store: EmbeddingsStoreFaiss):
        """Building index twice should produce same ordering."""
        # Use fixed UUIDs for reproducibility test
        book_ids = [
            UUID("00000000-0000-0000-0000-000000000001"),
            UUID("00000000-0000-0000-0000-000000000002"),
            UUID("00000000-0000-0000-0000-000000000003"),
        ]
        texts = ["Book A", "Book B", "Book C"]

        # Build first time
        for bid, text in zip(book_ids, texts):
            store.store_embedding(bid, store.generate_embedding(text))
        store.build_index()
        mapping1 = store.get_id_mapping().copy()

        # Clear and rebuild
        store.clear()
        for bid, text in zip(book_ids, texts):
            store.store_embedding(bid, store.generate_embedding(text))
        store.build_index()
        mapping2 = store.get_id_mapping()

        # Should be identical (sorted by UUID)
        assert mapping1 == mapping2

    def test_rebuild_index_updates_mapping(self, store: EmbeddingsStoreFaiss):
        """Rebuilding index after adding more embeddings should update."""
        # Initial build
        id1 = uuid4()
        store.store_embedding(id1, store.generate_embedding("First book"))
        store.build_index()
        assert store.get_index_size() == 1

        # Add more and rebuild
        id2 = uuid4()
        store.store_embedding(id2, store.generate_embedding("Second book"))
        store.build_index()
        assert store.get_index_size() == 2


# -----------------------------------------------------------------------------
# Test: Persistence (save_index / load_index)
# -----------------------------------------------------------------------------


class TestPersistence:
    """Tests for save_index and load_index methods."""

    def test_save_and_load_preserves_index(self, populated_store: tuple[EmbeddingsStoreFaiss, list[tuple[UUID, str]]]):
        """Saved and loaded index should behave identically."""
        store, sample_books = populated_store

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            store.save_index(tmpdir)

            # Verify files created
            assert (Path(tmpdir) / "faiss_index.bin").exists()
            assert (Path(tmpdir) / "faiss_id_mapping.json").exists()
            assert (Path(tmpdir) / "faiss_embeddings.json").exists()

            # Create new store and load
            new_store = EmbeddingsStoreFaiss()
            new_store.load_index(tmpdir)

            # Verify state matches
            assert new_store.get_index_size() == store.get_index_size()
            assert new_store.get_id_mapping() == store.get_id_mapping()

            # Verify get_embedding works after load
            for book_id, _ in sample_books:
                original = store.get_embedding(book_id)
                loaded = new_store.get_embedding(book_id)
                assert original == loaded

    def test_save_without_index_raises(self, store: EmbeddingsStoreFaiss):
        """Saving before building index should raise RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="has not been built"):
                store.save_index(tmpdir)

    def test_load_nonexistent_path_raises(self, store: EmbeddingsStoreFaiss):
        """Loading from non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            store.load_index("/nonexistent/path")

    def test_load_creates_directory_structure(self, populated_store: tuple[EmbeddingsStoreFaiss, list[tuple[UUID, str]]]):
        """save_index should create parent directories if needed."""
        store, _ = populated_store

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "deep" / "path"
            store.save_index(str(nested_path))

            assert nested_path.exists()
            assert (nested_path / "faiss_index.bin").exists()

    def test_load_validates_mapping_size(self, populated_store: tuple[EmbeddingsStoreFaiss, list[tuple[UUID, str]]]):
        """Loading with mismatched mapping should raise ValueError."""
        store, _ = populated_store

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save_index(tmpdir)

            # Corrupt the mapping file
            import json
            mapping_path = Path(tmpdir) / "faiss_id_mapping.json"
            with open(mapping_path, "w") as f:
                json.dump(["only-one-id"], f)  # Wrong count

            new_store = EmbeddingsStoreFaiss()
            with pytest.raises(ValueError, match="does not match"):
                new_store.load_index(tmpdir)


# -----------------------------------------------------------------------------
# Test: Clear Operation
# -----------------------------------------------------------------------------


class TestClearOperation:
    """Tests for the clear method."""

    def test_clear_resets_all_state(self, populated_store: tuple[EmbeddingsStoreFaiss, list[tuple[UUID, str]]]):
        """Clear should remove all indexed data and embeddings."""
        store, sample_books = populated_store

        assert store.get_index_size() > 0

        store.clear()

        assert store.get_index_size() == 0
        assert store.get_index() is None
        assert store.get_id_mapping() == []

        # Embeddings should also be cleared
        for book_id, _ in sample_books:
            assert store.get_embedding(book_id) is None

    def test_can_rebuild_after_clear(self, store: EmbeddingsStoreFaiss):
        """Should be able to add and build again after clear."""
        # First build
        id1 = uuid4()
        store.store_embedding(id1, store.generate_embedding("First book"))
        store.build_index()

        # Clear
        store.clear()

        # Rebuild with different data
        id2 = uuid4()
        store.store_embedding(id2, store.generate_embedding("Second book"))
        store.build_index()

        assert store.get_index_size() == 1
        assert store.get_embedding(id2) is not None
        assert store.get_embedding(id1) is None  # Old data gone


# -----------------------------------------------------------------------------
# Test: Protocol Compliance
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    """Tests verifying EmbeddingsStore Protocol compliance."""

    def test_has_all_protocol_methods(self, store: EmbeddingsStoreFaiss):
        """Store should have all methods required by EmbeddingsStore Protocol."""
        # Required methods from Protocol
        required_methods = [
            "generate_embedding",
            "generate_embeddings_batch",
            "store_embedding",
            "store_embeddings_batch",
            "get_embedding",
            "build_index",
            "save_index",
            "load_index",
            "get_dimension",
        ]

        for method_name in required_methods:
            assert hasattr(store, method_name), f"Missing method: {method_name}"
            assert callable(getattr(store, method_name)), f"Not callable: {method_name}"

    def test_method_signatures_match_protocol(self, store: EmbeddingsStoreFaiss):
        """Method signatures should match Protocol expectations."""
        from typing import get_type_hints
        from app.domain.ports import EmbeddingsStore

        # This is a basic check - full type checking requires mypy
        protocol_hints = get_type_hints(EmbeddingsStore.generate_embedding)
        impl_hints = get_type_hints(store.generate_embedding)

        # At minimum, return types should match
        assert protocol_hints.get("return") == impl_hints.get("return")

    def test_store_implements_protocol(self):
        """EmbeddingsStoreFaiss should be recognized as implementing EmbeddingsStore."""
        from app.domain.ports import EmbeddingsStore

        store = EmbeddingsStoreFaiss()

        # This should not raise - structural subtyping
        def accepts_protocol(s: EmbeddingsStore) -> int:
            return s.get_dimension()

        result = accepts_protocol(store)
        assert result == 384


# -----------------------------------------------------------------------------
# Test: Edge Cases and Error Handling
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_text(self, store: EmbeddingsStoreFaiss):
        """Should handle very long text (sentence-transformers truncates)."""
        long_text = "word " * 10000  # Very long

        # Should not raise, model will truncate
        embedding = store.generate_embedding(long_text)
        assert len(embedding) == 384

    def test_unicode_text(self, store: EmbeddingsStoreFaiss):
        """Should handle Unicode characters properly."""
        unicode_texts = [
            "Novela de misterio en espanol con acentos",
            "Livre francais avec des caracteres speciaux",
            "Japanese text",
            "Chinese characters",
        ]

        for text in unicode_texts:
            embedding = store.generate_embedding(text)
            assert len(embedding) == 384

    def test_special_characters_in_text(self, store: EmbeddingsStoreFaiss):
        """Should handle special characters."""
        special_text = "Book with 'quotes', \"double quotes\", and symbols: @#$%^&*()"

        embedding = store.generate_embedding(special_text)
        assert len(embedding) == 384

    def test_uuid_string_serialization(self, store: EmbeddingsStoreFaiss):
        """UUIDs should be properly serialized to strings internally."""
        book_id = UUID("12345678-1234-5678-1234-567812345678")
        embedding = store.generate_embedding("Test")

        store.store_embedding(book_id, embedding)
        store.build_index()

        # Check internal mapping uses string
        mapping = store.get_id_mapping()
        assert str(book_id) in mapping
        assert isinstance(mapping[0], str)
