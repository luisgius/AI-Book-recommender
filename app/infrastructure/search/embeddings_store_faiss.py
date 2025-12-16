"""
FAISS-based implementation of the EmbeddingsStore port.

This adapter uses:
- sentence-transformers: To generate dense vector embeddings from text
- FAISS: To index embeddings for fast nearest-neighbor search

The implementation follows Hexagonal Architecture principles:
- Implements the EmbeddingsStore Protocol from the domain layer
- Encapsulates all FAISS and sentence-transformers specifics
- Can be swapped for another implementation without domain changes

IMPORTANT: This store handles embedding generation, storage, and index building.
It does NOT perform search operations - that is the responsibility of
VectorSearchRepository, which uses the index built by this store.

Design decisions (see ADR 003 for details):
- MVP uses IndexFlatL2 (exact L2 distance, no normalization)
- Alternative: normalize embeddings + IndexFlatIP for cosine similarity
- book_id uses UUID (serialized to string for JSON persistence)
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
from uuid import UUID

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.domain.ports import EmbeddingsStore

logger = logging.getLogger(__name__)


class EmbeddingsStoreFaiss(EmbeddingsStore):
    """
    FAISS-based embeddings store for semantic search.

    This implementation uses:
    - IndexFlatL2 for exact nearest-neighbor search (MVP, simplest)
    - A mapping from FAISS internal indices to book UUIDs
    - Persistence to disk for both the index and the mapping

    For larger datasets (>100k books), consider switching to approximate
    nearest neighbor (ANN) indices like HNSW or IVF for better performance.
    See the ADR at docs/arch_decisions/003_faiss_vector_index.md for details.

    Distance metric choice (MVP vs production):
    - MVP: IndexFlatL2 with raw embeddings (L2/Euclidean distance)
    - Production alternative: Normalize embeddings + IndexFlatIP (cosine similarity)
      Cosine is often better for semantic similarity but adds complexity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the FAISS embeddings store.

        Args:
            model_name: Name of the sentence-transformers model to use.
                        "all-MiniLM-L6-v2" is a good balance of speed and quality,
                        producing 384-dimensional embeddings.
        """
        # Load the sentence-transformers model for generating embeddings.
        # This model converts text into dense vectors of fixed dimension.
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

        # Get the embedding dimension from the model.
        # All embeddings must have the same dimension to be indexed together.
        self._dimension: int = self._model.get_sentence_embedding_dimension()

        # FAISS index - initialized as None, created when build_index() is called.
        # We use IndexFlatL2 which performs exact L2 (Euclidean) distance search.
        # This is the simplest index type: no training required, exact results.
        self._index: Optional[faiss.IndexFlatL2] = None

        # Mapping from FAISS internal index (0, 1, 2, ...) to book UUID.
        # FAISS uses contiguous integer indices internally, but our books have
        # UUIDs. This list maintains the correspondence:
        # _id_mapping[faiss_idx] = book_uuid (as string for JSON serialization)
        self._id_mapping: List[str] = []

        # Storage for embeddings keyed by book_id (UUID as string).
        # We accumulate embeddings here, then batch-add them to FAISS when
        # build_index() is called.
        self._embeddings: Dict[str, List[float]] = {}

        logger.info(
            f"Initialized EmbeddingsStoreFaiss with model '{model_name}' "
            f"(dimension={self._dimension})"
        )

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.

        This method uses the sentence-transformers model to encode text
        into a dense vector representation. The resulting embedding captures
        the semantic meaning of the text.

        Args:
            text: The text to embed (e.g., book title + description).

        Returns:
            A list of floats representing the embedding vector.
            The length equals self._dimension (384 for all-MiniLM-L6-v2).

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        # The encode() method handles tokenization, padding, and inference.
        # We get back a numpy array of shape (dimension,).
        embedding = self._model.encode(text, convert_to_numpy=True)

        # Convert to Python list for serialization compatibility.
        # The domain layer works with List[float], not numpy arrays.
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a batch.

        This is more efficient than calling generate_embedding repeatedly
        because sentence-transformers can batch the inference.

        Args:
            texts: List of texts to encode.

        Returns:
            List of embedding vectors, in the same order as input texts.

        Raises:
            ValueError: If any text is empty.
        """
        if not texts:
            return []

        # Validate all texts are non-empty
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")

        # Batch encode is much faster than individual calls.
        # The model handles batching internally for GPU/CPU efficiency.
        embeddings = self._model.encode(texts, convert_to_numpy=True)

        # Convert numpy array to list of lists
        return [emb.tolist() for emb in embeddings]

    def store_embedding(self, book_id: UUID, embedding: List[float]) -> None:
        """
        Store an embedding associated with a book UUID.

        The embedding is stored in memory and will be added to the FAISS
        index when build_index() is called.

        Args:
            book_id: The book's unique identifier (UUID).
            embedding: The embedding vector for this book.

        Raises:
            ValueError: If embedding dimension is incorrect.
        """
        # Validate dimension matches what the model produces
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"got {len(embedding)}"
            )

        # Store with UUID serialized to string for JSON compatibility
        book_id_str = str(book_id)
        self._embeddings[book_id_str] = embedding
        logger.debug(f"Stored embedding for book_id={book_id_str}")

    def store_embeddings_batch(
        self, book_ids: List[UUID], embeddings: List[List[float]]
    ) -> None:
        """
        Store multiple embeddings in a batch (FULL REFRESH - clears existing).

        WARNING: This method performs a FULL REFRESH, not an incremental update.
        All previously stored embeddings are cleared before the new batch is stored.

        This semantic is intentional for use with CatalogIngestionService, which
        rebuilds from the complete SQLite catalog (source of truth). Without clearing,
        embeddings from deleted books would remain as stale data in the FAISS index.

        For incremental updates (adding one book at a time), use store_embedding().

        Args:
            book_ids: List of book UUIDs.
            embeddings: List of corresponding embedding vectors.

        Raises:
            ValueError: If lists have different lengths or invalid data.
        """
        if len(book_ids) != len(embeddings):
            raise ValueError(
                f"Mismatched lengths: {len(book_ids)} book_ids vs "
                f"{len(embeddings)} embeddings"
            )

        # Log before clearing to make the full-refresh behavior explicit
        if self._embeddings:
            logger.info(
                f"store_embeddings_batch: clearing {len(self._embeddings)} existing "
                f"embeddings (full refresh)"
            )

        self._embeddings.clear()

        for book_id, embedding in zip(book_ids, embeddings):
            self.store_embedding(book_id, embedding)

        logger.info(f"Stored {len(book_ids)} embeddings")

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        """
        Retrieve the stored embedding for a book.

        Args:
            book_id: The book's unique identifier (UUID).

        Returns:
            The embedding vector if found, None otherwise.
        """
        book_id_str = str(book_id)
        return self._embeddings.get(book_id_str)


    def build_index(self) -> None:
        """
        Build or rebuild the FAISS index from all stored embeddings.

        This method:
        1. Creates a new IndexFlatL2 index
        2. Converts all stored embeddings to a numpy matrix
        3. Adds them to the index in a single batch operation
        4. Updates the id_mapping to track FAISS index -> book_id correspondence

        After calling this method, the index is ready for search operations
        (via VectorSearchRepository).

        Note on index types:
        - IndexFlatL2: Exact search, O(n) per query. Best for small datasets (<10k).
        - IndexIVFFlat: Approximate, uses clustering. Better for medium datasets.
        - IndexHNSWFlat: Approximate, uses graph. Best quality/speed trade-off.

        For this MVP, we use IndexFlatL2. See _build_index_hnsw() for an ANN option.

        Raises:
            RuntimeError: If no embeddings have been stored.
        """
        if not self._embeddings:
            logger.warning("No embeddings to index")
            return

        # Create a fresh index with the correct dimension.
        # IndexFlatL2 computes exact L2 (Euclidean) distances.
        # L2 distance: d(x,y) = sqrt(sum((x_i - y_i)^2))
        #
        # Alternative for cosine similarity:
        #   1. Normalize all embeddings: emb = emb / np.linalg.norm(emb)
        #   2. Use IndexFlatIP (inner product) instead of IndexFlatL2
        #   With normalized vectors, inner product == cosine similarity.
        self._index = faiss.IndexFlatL2(self._dimension)

        # Build the id mapping and embeddings matrix.
        # The order in _id_mapping corresponds to FAISS internal indices.
        # Sort by UUID string for reproducibility across runs (helps with evaluation).
        self._id_mapping = sorted(self._embeddings.keys())
        embeddings_list = [self._embeddings[book_id] for book_id in self._id_mapping]

        # Convert to a contiguous numpy matrix.
        # FAISS expects shape (n_vectors, dimension) with dtype float32.
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

        # Add all embeddings to the index in one batch.
        # This is much faster than adding one by one.
        # After this call, self._index.ntotal == len(self._embeddings).
        self._index.add(embeddings_matrix)

        logger.info(
            f"Built FAISS index with {len(self._embeddings)} embeddings "
            f"(index.ntotal={self._index.ntotal})"
        )
    

    def save_index(self, path: str) -> None:
        """
        Persist the FAISS index and id mapping to disk.

        This saves two files in the specified directory:
        - faiss_index.bin: The FAISS index binary file
        - faiss_id_mapping.json: The book_id mapping

        Args:
            path: Directory path where index files should be saved.

        Raises:
            IOError: If saving fails.
            RuntimeError: If index has not been built.
        """
        if self._index is None:
            raise RuntimeError("Cannot save: index has not been built")

        # Create Path objects for the two files
        dir_path = Path(path)
        index_path = dir_path / "faiss_index.bin"
        mapping_path = dir_path / "faiss_id_mapping.json"
        embeddings_path = dir_path / "faiss_embeddings.json"

        # Ensure directory exists
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save the FAISS index to disk.
        # faiss.write_index() handles the binary serialization.
        faiss.write_index(self._index, str(index_path))

        # Save the id mapping as JSON.
        # This is a simple list: FAISS index position -> book_id (UUID string).
        # ensure_ascii=False allows proper Unicode (e.g., Spanish characters).
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self._id_mapping, f, ensure_ascii=False)

        # Save the embeddings dict so we can restore get_embedding() functionality.
        with open(embeddings_path, "w", encoding="utf-8") as f:
            json.dump(self._embeddings, f, ensure_ascii=False)

        logger.info(
            f"Saved FAISS index ({self._index.ntotal} vectors) to {path}"
        )

    def load_index(self, path: str) -> None:
        """
        Load a previously saved FAISS index and id mapping from disk.

        This restores the index to a searchable state without needing
        to regenerate embeddings.

        Args:
            path: Directory path to the saved index files.

        Raises:
            IOError: If loading fails.
            ValueError: If index format is invalid.
        """
        dir_path = Path(path)
        index_path = dir_path / "faiss_index.bin"
        mapping_path = dir_path / "faiss_id_mapping.json"
        embeddings_path = dir_path / "faiss_embeddings.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

        # Load the FAISS index from disk.
        self._index = faiss.read_index(str(index_path))

        # Load the id mapping from JSON.
        with open(mapping_path, "r", encoding="utf-8") as f:
            self._id_mapping = json.load(f)

        # Load the embeddings dict if it exists (for get_embedding support).
        if embeddings_path.exists():
            with open(embeddings_path, "r", encoding="utf-8") as f:
                self._embeddings = json.load(f)
        else:
            # Backwards compatibility: reconstruct from mapping if embeddings not saved
            logger.warning(
                "Embeddings file not found, get_embedding() will return None "
                "for all book_ids until embeddings are re-stored"
            )
            self._embeddings = {}

        # Validate that the mapping matches the index size.
        if len(self._id_mapping) != self._index.ntotal:
            raise ValueError(
                f"Mapping size ({len(self._id_mapping)}) does not match "
                f"index size ({self._index.ntotal})"
            )

        logger.info(
            f"Loaded FAISS index ({self._index.ntotal} vectors) from {path}"
        )

    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Returns:
            The embedding dimension (384 for all-MiniLM-L6-v2).
        """
        return self._dimension

    # -------------------------------------------------------------------------
    # Additional helper methods (not part of Protocol, but useful internally)
    # -------------------------------------------------------------------------

    def get_index(self) -> Optional[faiss.IndexFlatL2]:
        """
        Get the underlying FAISS index for use by VectorSearchRepository.

        This is an internal method, not part of the EmbeddingsStore Protocol.
        VectorSearchRepository uses this to perform searches.

        Returns:
            The FAISS index, or None if not built.
        """
        return self._index

    def get_id_mapping(self) -> List[str]:
        """
        Get the FAISS index position -> book_id mapping.

        This is an internal method, not part of the EmbeddingsStore Protocol.
        VectorSearchRepository uses this to convert FAISS indices to book UUIDs.

        Returns:
            List where index i contains the book_id at FAISS position i.
        """
        return self._id_mapping

    def get_index_size(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            Number of indexed embeddings.
        """
        if self._index is None:
            return 0
        return self._index.ntotal

    def clear(self) -> None:
        """Clear the index and all stored embeddings."""
        self._index = None
        self._id_mapping = []
        self._embeddings = {}
        logger.info("Cleared FAISS index and stored embeddings")


# -----------------------------------------------------------------------------
# Alternative Index Types (documented for future use)
# -----------------------------------------------------------------------------


def _build_index_hnsw(
    embeddings: np.ndarray,
    dimension: int,
    M: int = 32,
    ef_construction: int = 200,
) -> faiss.IndexHNSWFlat:
    """
    Build an HNSW index for approximate nearest-neighbor search.

    HNSW (Hierarchical Navigable Small World) provides excellent query
    performance with high recall. It's recommended for datasets > 10k vectors.

    Trade-offs vs IndexFlatL2:
    - Faster queries: O(log n) vs O(n)
    - Uses more memory (stores graph structure)
    - Approximate results (configurable recall)
    - Requires more time to build

    Args:
        embeddings: Matrix of shape (n_vectors, dimension)
        dimension: Embedding dimension
        M: Number of connections per layer (higher = better recall, more memory)
        ef_construction: Search depth during construction (higher = better quality)

    Returns:
        A built HNSW index ready for search.

    Example usage (not used in MVP):
        index = _build_index_hnsw(embeddings_matrix, 384)
        index.hnsw.efSearch = 64  # Set search depth for queries
        distances, indices = index.search(query, k=10)
    """
    # Create the HNSW index with L2 distance.
    index = faiss.IndexHNSWFlat(dimension, M)

    # ef_construction controls build quality vs speed.
    index.hnsw.efConstruction = ef_construction

    # Add all embeddings (HNSW builds the graph incrementally).
    index.add(embeddings)

    return index


# -----------------------------------------------------------------------------
# Normalization helper (for cosine similarity alternative)
# -----------------------------------------------------------------------------


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length for cosine similarity.

    When using normalized embeddings with IndexFlatIP (inner product),
    the result is equivalent to cosine similarity:
        cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
    With ||a|| = ||b|| = 1, this simplifies to dot(a, b).

    Args:
        embeddings: Matrix of shape (n_vectors, dimension)

    Returns:
        Normalized embeddings with unit L2 norm per row.

    Example:
        normalized = _normalize_embeddings(embeddings_matrix)
        index = faiss.IndexFlatIP(dimension)  # Inner product
        index.add(normalized)
        # Now search returns cosine similarity scores
    """
    # Compute L2 norm for each embedding
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Avoid division by zero
    norms = np.maximum(norms, 1e-10)

    return embeddings / norms