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
- The embedding model is configurable (constructor argument, wired from
  EMBEDDING_MODEL). Asymmetric models format queries and documents
  differently; PROMPT_CONVENTIONS (text_encoding.py) lists the convention of
  each known model.
- Encoding is delegated to a TextEncoder (text_encoding.py). By default the
  store builds the local sentence-transformers encoder below, exactly as
  before; a caller can inject another one instead, such as the OpenRouter
  API encoder of the cloud profile, which loads no model. The store itself
  only stores, normalizes, indexes and persists vectors.
- New indexes store L2-normalized vectors in an IndexFlatIP, so the inner
  product is the cosine similarity.
- Every saved index carries a sidecar metadata file (faiss_metadata.json) with
  the model that produced its vectors. Loading an index with a different model
  raises IndexModelMismatchError instead of silently mixing vector spaces.
- Legacy indexes saved before the metadata file existed (data/indexes) are
  still loaded as before: all-MiniLM-L6-v2 vectors in an IndexFlatL2.
- book_id uses UUID (serialized to string for JSON persistence)
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
from sentence_transformers import SentenceTransformer

from app.domain.ports import EmbeddingsStore
from app.infrastructure.search.faiss_runtime import faiss  # torch before faiss (macOS)

# The conventions live in a module without model libraries (so the API
# encoder can use them without torch); they are re-exported here because
# callers and tests import them from this module.
from app.infrastructure.search.text_encoding import (  # noqa: F401  (re-exports)
    E5_INSTRUCT_TASK,
    PROMPT_CONVENTIONS,
    PromptConvention,
    TextEncoder,
    canonical_model_name,
    model_slug,
    prompt_convention_for,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Model conventions
# -----------------------------------------------------------------------------

# Model used when the store is built without a model name.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Model of every index saved before faiss_metadata.json existed. This is a
# fact about old files and must not follow future changes of the default.
LEGACY_INDEX_MODEL = "all-MiniLM-L6-v2"

# Metric names written to the metadata file.
METRIC_INNER_PRODUCT = "inner_product"
METRIC_L2 = "l2"

# File names inside an index directory.
INDEX_FILE = "faiss_index.bin"
MAPPING_FILE = "faiss_id_mapping.json"
EMBEDDINGS_FILE = "faiss_embeddings.json"
METADATA_FILE = "faiss_metadata.json"


# -----------------------------------------------------------------------------
# Index metadata (sidecar file)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexMetadata:
    """
    Description of a saved FAISS index, persisted as faiss_metadata.json.

    It records which model produced the vectors, so a store never answers
    queries embedded by one model against documents embedded by another.
    """

    model_name: str
    dimension: int
    metric: str  # METRIC_INNER_PRODUCT or METRIC_L2
    normalized: bool  # True if vectors were L2-normalized before indexing
    vector_count: int
    created_at: Optional[str]  # ISO 8601 (UTC); None for legacy indexes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexMetadata":
        """Build from the dict stored in faiss_metadata.json."""
        return cls(
            model_name=str(data["model_name"]),
            dimension=int(data["dimension"]),
            metric=str(data["metric"]),
            normalized=bool(data["normalized"]),
            vector_count=int(data["vector_count"]),
            created_at=data.get("created_at"),
        )


def legacy_index_metadata(index: Any) -> IndexMetadata:
    """
    Metadata assumed for an index saved before faiss_metadata.json existed.

    Those indexes (e.g. the shipped data/indexes/faiss_index) were always
    built with all-MiniLM-L6-v2 vectors in an IndexFlatL2, without extra
    normalization.
    """
    return IndexMetadata(
        model_name=LEGACY_INDEX_MODEL,
        dimension=index.d,
        metric=METRIC_L2,
        normalized=False,
        vector_count=index.ntotal,
        created_at=None,
    )


def read_index_metadata(path: str) -> Optional[IndexMetadata]:
    """
    Read the metadata of a saved index without loading any model.

    Useful to find out which embedding model an index needs before building
    a store for it.

    Args:
        path: Directory that contains the index files.

    Returns:
        The stored metadata, or None for a legacy index without metadata.
    """
    metadata_path = Path(path) / METADATA_FILE
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return IndexMetadata.from_dict(json.load(f))


class IndexModelMismatchError(ValueError):
    """Raised when an index was built with a different model than the store's."""


# -----------------------------------------------------------------------------
# Local encoder (sentence-transformers)
# -----------------------------------------------------------------------------


class SentenceTransformerEncoder(TextEncoder):
    """
    TextEncoder backed by a sentence-transformers model running in this process.

    It applies the model's prompt convention (PROMPT_CONVENTIONS): a prefix
    or instruction for queries and documents, or the prompt stored in the
    model under ``query_prompt_name``. This is the encoding code the store
    always had; it only moved out of the store class.
    """

    def __init__(self, model: Any, model_name: str):
        """
        Args:
            model: A loaded sentence_transformers.SentenceTransformer
            model_name: The name it was loaded with (recorded in index metadata)
        """
        self._model = model
        self._model_name = model_name
        self._prompts = prompt_convention_for(model_name)
        self._dimension: int = model.get_sentence_embedding_dimension()

    def _encode(
        self,
        texts: List[str],
        prefix: str = "",
        prompt_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode texts with the model, applying a prefix or a stored prompt.

        Returns:
            Matrix of shape (len(texts), dimension).
        """
        inputs = [prefix + text for text in texts]
        # Pass prompt_name only when the convention defines one, so symmetric
        # models are called exactly as before.
        extra: Dict[str, Any] = {"prompt_name": prompt_name} if prompt_name else {}
        return self._model.encode(inputs, convert_to_numpy=True, **extra)

    def generate_embedding(self, text: str) -> List[float]:
        """Document vector (the model's document convention is applied)."""
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        embedding = self._encode([text], prefix=self._prompts.document_prefix)[0]
        # Convert to Python list for serialization compatibility.
        # The domain layer works with List[float], not numpy arrays.
        return embedding.tolist()

    def generate_query_embedding(self, text: str) -> List[float]:
        """Query vector (prefix, instruction or stored prompt of the model)."""
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        embedding = self._encode(
            [text],
            prefix=self._prompts.query_prefix,
            prompt_name=self._prompts.query_prompt_name,
        )[0]
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Document vectors in one batched model call (input order kept)."""
        if not texts:
            return []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")
        embeddings = self._encode(texts, prefix=self._prompts.document_prefix)
        return [emb.tolist() for emb in embeddings]

    def get_dimension(self) -> int:
        """Output dimension of the model."""
        return self._dimension

    def get_model_name(self) -> str:
        """The model name, recorded unchanged in the index metadata."""
        return self._model_name

    def get_device(self) -> str:
        """Torch device the model runs on (e.g. "cpu", "mps")."""
        return str(getattr(self._model, "device", "unknown"))


# -----------------------------------------------------------------------------
# Store
# -----------------------------------------------------------------------------


class EmbeddingsStoreFaiss(EmbeddingsStore):
    """
    FAISS-based embeddings store for semantic search.

    This implementation uses:
    - A TextEncoder for queries and documents: by default a configurable
      sentence-transformers model with its query/document prompt convention
      (see PROMPT_CONVENTIONS); an injected encoder otherwise (for example the
      OpenRouter API encoder)
    - IndexFlatIP over L2-normalized vectors (exact cosine search) for new
      indexes; legacy IndexFlatL2 indexes are still loaded and searched
    - A mapping from FAISS internal indices to book UUIDs
    - Persistence to disk for the index, the mapping and its metadata

    For larger datasets (>100k books), consider switching to approximate
    nearest neighbor (ANN) indices like HNSW or IVF for better performance.
    See the ADR at docs/arch_decisions/003_faiss_vector_index.md for details.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        encoder: Optional[TextEncoder] = None,
    ):
        """
        Initialize the FAISS embeddings store.

        Args:
            model_name: Name of the sentence-transformers model to use.
                        "all-MiniLM-L6-v2" (default) is small and fast and
                        produces 384-dimensional embeddings, but it is
                        English-only. Not used when `encoder` is given.
            device: Torch device for the model ("cpu", "mps", "cuda").
                    None lets sentence-transformers pick the best available.
            encoder: An already built TextEncoder to use instead of loading a
                     sentence-transformers model (e.g. an API encoder). Its
                     get_model_name() is the identity recorded in the index.

        Raises:
            ValueError: If both a non-default model_name or a device and an
                encoder are given (the encoder decides the model).
        """
        if encoder is None:
            # Load the sentence-transformers model for generating embeddings.
            # This model converts text into dense vectors of fixed dimension.
            self._model = SentenceTransformer(model_name, device=device)
            encoder = SentenceTransformerEncoder(self._model, model_name)
        else:
            if model_name != DEFAULT_EMBEDDING_MODEL or device is not None:
                raise ValueError(
                    "Give either model_name/device (local model) or encoder, not both: "
                    f"the encoder '{encoder.get_model_name()}' decides the model"
                )
            self._model = None  # no model in this process
        self._encoder: TextEncoder = encoder
        self._model_name = encoder.get_model_name()

        # Get the embedding dimension from the encoder.
        # All embeddings must have the same dimension to be indexed together.
        self._dimension: int = encoder.get_dimension()

        # FAISS index - None until build_index() or load_index() is called.
        self._index: Optional[faiss.Index] = None

        # Metadata of the current index (model, metric, ...). None until built
        # or loaded.
        self._index_metadata: Optional[IndexMetadata] = None

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
            f"Initialized EmbeddingsStoreFaiss with model '{self._model_name}' "
            f"(dimension={self._dimension}, device={self.get_device()})"
        )

    # -------------------------------------------------------------------------
    # Embedding generation (delegated to the encoder)
    # -------------------------------------------------------------------------

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate the embedding of a document (book text).

        The model's document convention is applied (e.g. "passage: " for E5).
        Queries must use generate_query_embedding instead.

        Args:
            text: The text to embed (e.g., book title + description).

        Returns:
            A list of floats representing the embedding vector.
            The length equals self._dimension (384 for all-MiniLM-L6-v2).

        Raises:
            ValueError: If text is empty.
        """
        return self._encoder.generate_embedding(text)

    def generate_query_embedding(self, text: str) -> List[float]:
        """
        Generate the embedding of a search query.

        The model's query convention is applied: a prefix such as "query: "
        or an instruction, or the prompt stored in the model under
        ``query_prompt_name``. For symmetric models this returns the same
        vector as generate_embedding.

        Args:
            text: The query text.

        Returns:
            The query embedding vector.

        Raises:
            ValueError: If text is empty.
        """
        return self._encoder.generate_query_embedding(text)

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate document embeddings for multiple texts in a batch.

        This is more efficient than calling generate_embedding repeatedly
        because the encoder can batch the inference (or the API request).

        Args:
            texts: List of texts to encode.

        Returns:
            List of embedding vectors, in the same order as input texts.

        Raises:
            ValueError: If any text is empty.
        """
        return self._encoder.generate_embeddings_batch(texts)

    # -------------------------------------------------------------------------
    # Embedding storage
    # -------------------------------------------------------------------------

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
        Store multiple embeddings in a batch.

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

    # -------------------------------------------------------------------------
    # Index building and persistence
    # -------------------------------------------------------------------------

    def build_index(self) -> None:
        """
        Build or rebuild the FAISS index from all stored embeddings.

        This method:
        1. Converts all stored embeddings to a numpy matrix
        2. L2-normalizes every row (unit length)
        3. Adds them to a new IndexFlatIP in a single batch operation
        4. Updates the id_mapping to track FAISS index -> book_id correspondence

        With unit-length vectors, the inner product equals the cosine
        similarity: cos(a, b) = a.b / (|a| |b|) = a.b. The vector search
        repository normalizes each query the same way before searching.

        Note on index types:
        - IndexFlatIP / IndexFlatL2: Exact search, O(n) per query. Best for
          small datasets (<10k), like this catalog.
        - IndexIVFFlat: Approximate, uses clustering. Better for medium datasets.
        - IndexHNSWFlat: Approximate, uses graph. Best quality/speed trade-off.
        See _build_index_hnsw() for an ANN option.
        """
        if not self._embeddings:
            logger.warning("No embeddings to index")
            return

        # Build the id mapping and embeddings matrix.
        # The order in _id_mapping corresponds to FAISS internal indices.
        # Sort by UUID string for reproducibility across runs (helps with evaluation).
        self._id_mapping = sorted(self._embeddings.keys())
        embeddings_list = [self._embeddings[book_id] for book_id in self._id_mapping]

        # Convert to a contiguous numpy matrix.
        # FAISS expects shape (n_vectors, dimension) with dtype float32.
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

        # Normalize rows in place to unit length, then index by inner product.
        faiss.normalize_L2(embeddings_matrix)
        self._index = faiss.IndexFlatIP(self._dimension)

        # Add all embeddings to the index in one batch.
        # After this call, self._index.ntotal == len(self._embeddings).
        self._index.add(embeddings_matrix)

        self._index_metadata = IndexMetadata(
            model_name=self._model_name,
            dimension=self._dimension,
            metric=METRIC_INNER_PRODUCT,
            normalized=True,
            vector_count=self._index.ntotal,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        logger.info(
            f"Built FAISS index with {len(self._embeddings)} embeddings "
            f"(index.ntotal={self._index.ntotal}, metric=cosine)"
        )

    def save_index(self, path: str) -> None:
        """
        Persist the FAISS index, its id mapping and its metadata to disk.

        This saves four files in the specified directory:
        - faiss_index.bin: The FAISS index binary file
        - faiss_id_mapping.json: The book_id mapping
        - faiss_embeddings.json: The stored embeddings (for get_embedding)
        - faiss_metadata.json: Model, dimension, metric and creation date

        Args:
            path: Directory path where index files should be saved.

        Raises:
            IOError: If saving fails.
            RuntimeError: If index has not been built.
        """
        if self._index is None:
            raise RuntimeError("Cannot save: index has not been built")

        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save the FAISS index to disk.
        # faiss.write_index() handles the binary serialization.
        faiss.write_index(self._index, str(dir_path / INDEX_FILE))

        # Save the id mapping as JSON.
        # This is a simple list: FAISS index position -> book_id (UUID string).
        # ensure_ascii=False allows proper Unicode (e.g., Spanish characters).
        with open(dir_path / MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(self._id_mapping, f, ensure_ascii=False)

        # Save the embeddings dict so we can restore get_embedding() functionality.
        with open(dir_path / EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(self._embeddings, f, ensure_ascii=False)

        # Save the metadata sidecar (which model produced these vectors).
        metadata = self._index_metadata or legacy_index_metadata(self._index)
        with open(dir_path / METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(
            f"Saved FAISS index ({self._index.ntotal} vectors, "
            f"model={metadata.model_name}, metric={metadata.metric}) to {path}"
        )

    def load_index(self, path: str) -> None:
        """
        Load a previously saved FAISS index and id mapping from disk.

        This restores the index to a searchable state without needing
        to regenerate embeddings.

        If the directory has a metadata file, the model recorded there must be
        the model of this store. A legacy index without metadata is assumed to
        hold all-MiniLM-L6-v2 vectors in an IndexFlatL2 and keeps working as
        before.

        Args:
            path: Directory path to the saved index files.

        Raises:
            FileNotFoundError: If the index or the mapping file is missing.
            IndexModelMismatchError: If the index was built with another model.
            ValueError: If the index format is invalid (size or dimension).
        """
        dir_path = Path(path)
        index_path = dir_path / INDEX_FILE
        mapping_path = dir_path / MAPPING_FILE
        embeddings_path = dir_path / EMBEDDINGS_FILE

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

        # Read everything into locals first; the store state only changes
        # once all checks pass.
        index = faiss.read_index(str(index_path))
        with open(mapping_path, "r", encoding="utf-8") as f:
            id_mapping = json.load(f)

        stored_metadata = read_index_metadata(path)
        metadata = stored_metadata or legacy_index_metadata(index)

        # Check 1: the index must come from the same model as the queries.
        if canonical_model_name(metadata.model_name) != canonical_model_name(self._model_name):
            origin = (
                "recorded in its metadata"
                if stored_metadata
                else "assumed: legacy index without metadata"
            )
            raise IndexModelMismatchError(
                f"The FAISS index at '{path}' was built with embedding model "
                f"'{metadata.model_name}' ({origin}), but this store uses "
                f"'{self._model_name}'. Query and document vectors would come "
                f"from different models. Rebuild the index with "
                f"`python -m scripts.ingest_and_build --rebuild "
                f"--embedding-model {self._model_name} --out-dir <dir>`, "
                f"or set EMBEDDING_MODEL={metadata.model_name}."
            )

        # Check 2: vector dimension must match the model output.
        if index.d != self._dimension:
            raise ValueError(
                f"Index dimension ({index.d}) does not match the dimension of "
                f"model '{self._model_name}' ({self._dimension}). Rebuild the index."
            )

        # Check 3: the mapping must match the index size.
        if len(id_mapping) != index.ntotal:
            raise ValueError(
                f"Mapping size ({len(id_mapping)}) does not match "
                f"index size ({index.ntotal})"
            )

        self._index = index
        self._id_mapping = id_mapping
        self._index_metadata = metadata

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

        if stored_metadata is None:
            logger.info(
                f"Loaded legacy FAISS index without metadata from {path} "
                f"(assumed model={LEGACY_INDEX_MODEL}, metric=l2)"
            )
        logger.info(
            f"Loaded FAISS index ({index.ntotal} vectors, "
            f"metric={metadata.metric}) from {path}"
        )

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Returns:
            The embedding dimension (384 for all-MiniLM-L6-v2).
        """
        return self._dimension

    def get_model_name(self) -> str:
        """Identity of the encoder's vectors (the model name for a local model)."""
        return self._model_name

    def get_device(self) -> str:
        """Where the encoder runs: torch device of a local model, "api" for an API."""
        return self._encoder.get_device()

    def get_encoder(self) -> TextEncoder:
        """The encoder that produces this store's vectors."""
        return self._encoder

    def get_index_metadata(self) -> Optional[IndexMetadata]:
        """Metadata of the built or loaded index, or None if there is none."""
        return self._index_metadata

    # -------------------------------------------------------------------------
    # Additional helper methods (not part of Protocol, but useful internally)
    # -------------------------------------------------------------------------

    def get_index(self) -> Optional[faiss.Index]:
        """
        Get the underlying FAISS index for use by VectorSearchRepository.

        This is an internal method, not part of the EmbeddingsStore Protocol.
        VectorSearchRepository uses this to perform searches. The index's
        metric_type tells the repository how to turn distances into scores.

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

    def is_ready(self) -> bool:
        """
        Check if the embeddings store and FAISS index are ready for operations.

        Used for health checks and graceful degradation (RNF-08).

        Returns:
            True if embeddings can be generated and index is searchable, False otherwise
        """
        try:
            return (
                self._encoder is not None
                and self._index is not None
                and self._index.ntotal > 0
            )
        except Exception:
            return False

    def clear(self) -> None:
        """Clear the index and all stored embeddings."""
        self._index = None
        self._index_metadata = None
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

    Trade-offs vs the flat (exact) indexes:
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
    # Create the HNSW index with L2 distance. On unit-length vectors,
    # L2 ranks neighbours exactly like cosine similarity.
    index = faiss.IndexHNSWFlat(dimension, M)

    # ef_construction controls build quality vs speed.
    index.hnsw.efConstruction = ef_construction

    # Add all embeddings (HNSW builds the graph incrementally).
    index.add(embeddings)

    return index
