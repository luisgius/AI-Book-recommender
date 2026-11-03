"""
FAISS-based implementation of the VectorSearchRepository port.

This adapter performs vector similarity search using a FAISS index built and
maintained by EmbeddingsStoreFaiss. It converts FAISS search results into
domain SearchResult entities with fully hydrated Book objects.

Design decisions:
- Uses Option A from CLAUDE.md: keeps Book objects in-memory for fast hydration
- Applies filters post-retrieval (FAISS only knows about vectors, not metadata)
- Converts L2 distances to similarity scores: similarity = 1 / (1 + distance)
- Over-fetches candidates to compensate for filtering
- Explicitly sorts results after filtering to guarantee descending similarity order

Infrastructure coupling (documented decision):
- This adapter depends on FaissIndexProvider (infrastructure-level Protocol)
- FaissIndexProvider exposes FAISS-specific methods (get_index, get_id_mapping)
- This coupling is acceptable at the infrastructure layer; the domain remains clean
- See app/infrastructure/search/__init__.py for the Protocol definition

This adapter does NOT generate embeddings or build indices - that is the
responsibility of EmbeddingsStore. It only performs search operations.
"""

import logging
from typing import List, Optional, Dict

import numpy as np

from app.domain.entities import Book, SearchResult
from app.domain.value_objects import SearchFilters
from app.domain.ports import VectorSearchRepository
from app.infrastructure.search import FaissIndexProvider

logger = logging.getLogger(__name__)

# Factor by which to over-fetch results to compensate for post-retrieval filtering.
# If user requests 10 results and filters remove 50%, we need ~20 candidates.
# 3x provides good margin while keeping FAISS search fast.
OVER_FETCH_FACTOR = 3


class FaissVectorSearchRepository(VectorSearchRepository):
    """
    FAISS-based vector search repository.

    This implementation:
    - Queries the FAISS index via FaissIndexProvider protocol
    - Maintains an in-memory book_id -> Book mapping for fast hydration
    - Applies filters in-memory after FAISS search
    - Converts L2 distances to similarity scores
    - Validates index/mapping consistency before search

    The repository is designed to be used alongside BM25SearchRepository
    for hybrid search with RRF fusion.

    Usage:
        embeddings_store = EmbeddingsStoreFaiss()
        embeddings_store.load_index("data/faiss_index")

        books = catalog_repo.get_all()
        vector_repo = FaissVectorSearchRepository(embeddings_store, books)

        query_embedding = embeddings_store.generate_embedding("science fiction")
        results = vector_repo.search(query_embedding, max_results=10)
    """

    def __init__(
        self,
        index_provider: FaissIndexProvider,
        books: List[Book],
    ) -> None:
        """
        Initialize the FAISS vector search repository.

        Args:
            index_provider: Provider of FAISS index access (typically EmbeddingsStoreFaiss).
                           Must satisfy the FaissIndexProvider protocol.
                           The index should already be built (via build_index or load_index).
            books: List of all books in the catalog. Used to build the book_id -> Book
                   mapping for hydrating search results. Should match the books
                   that were indexed in the embeddings store.
        """
        self._index_provider = index_provider

        # Build in-memory mapping from book_id (UUID) to Book entity.
        # This allows O(1) lookup when converting FAISS results to SearchResult.
        # The mapping uses string keys to match FaissIndexProvider's id_mapping format.
        self._book_map: Dict[str, Book] = {str(book.id): book for book in books}

        logger.info(
            f"Initialized FaissVectorSearchRepository with {len(self._book_map)} books"
        )

    def search(
        self,
        query_embedding: List[float],
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """
        Perform a vector similarity search using FAISS.

        Search process:
        1. Validate query embedding dimension
        2. Validate index/mapping consistency
        3. Query FAISS index for k nearest neighbors (over-fetched)
        4. Convert FAISS indices to book IDs using id_mapping
        5. Hydrate Book entities from in-memory mapping
        6. Convert L2 distances to similarity scores
        7. Apply filters (language, category, year range)
        8. Sort by similarity descending (explicit, post-filter guarantee)
        9. Truncate to max_results and assign ranks

        Args:
            query_embedding: The embedding vector for the query. Must have the
                            same dimensionality as the indexed embeddings.
            max_results: Maximum number of results to return.
            filters: Optional filters to apply post-search.

        Returns:
            List of SearchResult entities, ranked by similarity score (descending).
            Each result has:
            - book: The matched book entity (fully hydrated)
            - final_score: Similarity score (1 / (1 + L2_distance))
            - vector_score: Same as final_score
            - source: "vector"
            - rank: 1-indexed position
            - lexical_score: None

        Raises:
            ValueError: If query_embedding has wrong dimensionality.
            RuntimeError: If search execution fails or index is out of sync.
        """
        # -----------------------------------------------------------------------
        # Step 1: Validate query embedding dimension
        # -----------------------------------------------------------------------
        expected_dim = self._index_provider.get_dimension()
        if len(query_embedding) != expected_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {expected_dim}, "
                f"got {len(query_embedding)}"
            )

        # -----------------------------------------------------------------------
        # Step 2: Get FAISS index and check if it's ready
        # -----------------------------------------------------------------------
        faiss_index = self._index_provider.get_index()
        if faiss_index is None or faiss_index.ntotal == 0:
            # Index not built or empty - return empty results
            logger.warning("FAISS index is empty or not built, returning no results")
            return []

        # Get the ID mapping (FAISS index position -> book_id string)
        id_mapping = self._index_provider.get_id_mapping()

        # -----------------------------------------------------------------------
        # Step 3: Validate index/mapping consistency
        # -----------------------------------------------------------------------
        # This catches silent corruption where the mapping and index are out of sync
        if len(id_mapping) != faiss_index.ntotal:
            raise RuntimeError(
                f"Index out of sync: id_mapping has {len(id_mapping)} entries "
                f"but FAISS index has {faiss_index.ntotal} vectors. "
                "Rebuild the index to fix this inconsistency."
            )

        # -----------------------------------------------------------------------
        # Step 4: Shape query vector for FAISS
        # -----------------------------------------------------------------------
        # FAISS expects a 2D numpy array of shape (n_queries, dimension).
        # We have a single query, so shape is (1, dimension).
        # dtype must be float32 (FAISS uses single precision).
        query_vector = np.array([query_embedding], dtype=np.float32)

        # -----------------------------------------------------------------------
        # Step 5: Determine how many candidates to fetch
        # -----------------------------------------------------------------------
        # Over-fetch to compensate for post-retrieval filtering.
        # We can't fetch more than the index contains.
        k_fetch = min(max_results * OVER_FETCH_FACTOR, faiss_index.ntotal)

        # -----------------------------------------------------------------------
        # Step 6: Execute FAISS search
        # -----------------------------------------------------------------------
        # faiss_index.search returns:
        #   distances: shape (n_queries, k) - L2 distances to nearest neighbors
        #   indices: shape (n_queries, k) - internal FAISS indices of neighbors
        #
        # For IndexFlatL2:
        #   - distances are squared L2 distances: sum((a_i - b_i)^2)
        #   - Lower distance = more similar
        #   - Results are sorted by distance ascending (most similar first)
        try:
            distances, indices = faiss_index.search(query_vector, k_fetch)
        except Exception as e:
            raise RuntimeError(f"FAISS search failed: {e}") from e

        # Extract results for the single query (index 0)
        distances = distances[0]  # Shape: (k_fetch,)
        indices = indices[0]      # Shape: (k_fetch,)

        # -----------------------------------------------------------------------
        # Step 7: Convert FAISS results to (Book, similarity_score) pairs
        # -----------------------------------------------------------------------
        candidates: List[tuple[Book, float]] = []

        for faiss_idx, distance in zip(indices, distances):
            # FAISS returns -1 for indices when fewer than k results exist
            if faiss_idx == -1:
                continue

            # Map FAISS internal index to book_id (UUID string)
            book_id_str = id_mapping[faiss_idx]

            # Look up the Book entity in our in-memory mapping
            book = self._book_map.get(book_id_str)
            if book is None:
                # Book exists in index but not in our mapping - data inconsistency
                logger.warning(
                    f"Book {book_id_str} found in FAISS index but not in book_map. "
                    "Index may be out of sync with catalog."
                )
                continue

            # Convert L2 distance to similarity score.
            # Formula: similarity = 1 / (1 + distance)
            # Properties:
            #   - distance = 0 -> similarity = 1 (identical vectors)
            #   - distance -> inf -> similarity -> 0 (very different)
            #   - Monotonically decreasing (preserves ranking)
            #   - Always positive, bounded in (0, 1]
            similarity_score = 1.0 / (1.0 + float(distance))

            candidates.append((book, similarity_score))

        # -----------------------------------------------------------------------
        # Step 8: Apply filters
        # -----------------------------------------------------------------------
        # Filters are applied post-retrieval because FAISS only knows about
        # vector distances, not book metadata.
        if filters is not None and not filters.is_empty():
            candidates = [
                (book, score)
                for book, score in candidates
                if self._matches_filters(book, filters)
            ]

        # -----------------------------------------------------------------------
        # Step 9: Explicit sort by similarity descending
        # -----------------------------------------------------------------------
        # FAISS returns results sorted by distance (ascending), which means
        # similarity is descending. After filtering, the relative order is preserved,
        # but we sort explicitly here to:
        # 1. Guarantee the invariant regardless of implementation details
        # 2. Make the behavior clear and testable
        # 3. Handle any edge cases (e.g., if filtering somehow reordered)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # -----------------------------------------------------------------------
        # Step 10: Truncate to max_results and build SearchResult entities
        # -----------------------------------------------------------------------
        candidates = candidates[:max_results]

        results: List[SearchResult] = []
        for rank, (book, similarity_score) in enumerate(candidates, start=1):
            result = SearchResult(
                book=book,
                final_score=similarity_score,
                rank=rank,
                source="vector",
                lexical_score=None,
                vector_score=similarity_score,
            )
            results.append(result)

        logger.debug(
            f"Vector search returned {len(results)} results "
            f"(fetched {k_fetch}, after filtering)"
        )

        return results

    @staticmethod
    def _matches_filters(book: Book, filters: SearchFilters) -> bool:
        """
        Check if a book matches the given filters.

        This method is identical to BM25SearchRepository._matches_filters
        to ensure consistent filtering behavior across search methods.

        Filters are applied as AND conditions:
        - language: exact match (case-insensitive)
        - category: case-insensitive substring match in any category
        - min_year/max_year: inclusive range check

        Args:
            book: The book to check
            filters: The filters to apply

        Returns:
            True if book matches all non-None filters, False otherwise
        """
        # Language filter: exact match, case-insensitive
        if filters.language is not None:
            if book.language is None:
                return False
            if book.language.lower() != filters.language.lower():
                return False

        # Category filter: substring match in any category, case-insensitive
        if filters.category is not None:
            if not book.categories:
                return False
            category_lower = filters.category.lower()
            if not any(category_lower in cat.lower() for cat in book.categories):
                return False

        # Year range filters: inclusive bounds
        book_year = book.get_published_year()

        if filters.min_year is not None:
            if book_year is None or book_year < filters.min_year:
                return False

        if filters.max_year is not None:
            if book_year is None or book_year > filters.max_year:
                return False

        return True

    # -------------------------------------------------------------------------
    # Helper methods (not part of VectorSearchRepository port)
    # -------------------------------------------------------------------------

    def update_books(self, books: List[Book]) -> None:
        """
        Update the in-memory book mapping.

        NOTE: This method is NOT part of the VectorSearchRepository port.
        It is an infrastructure helper used by ingestion/reindex workflows
        to keep the book_map synchronized with the FAISS index.

        Call this method after re-indexing to ensure the book_map stays
        synchronized with the FAISS index.

        Args:
            books: The new list of books (should match indexed books)
        """
        self._book_map = {str(book.id): book for book in books}
        logger.info(f"Updated book mapping with {len(self._book_map)} books")

    def is_ready(self) -> bool:
        """
        Check if the FAISS index is loaded and ready for search operations.

        Used for health checks and graceful degradation (RNF-08).

        Returns:
            True if the index is loaded and searchable, False otherwise
        """
        try:
            index = self._index_provider.get_index()
            id_mapping = self._index_provider.get_id_mapping()
            return (
                index is not None
                and id_mapping is not None
                and index.ntotal > 0
                and len(self._book_map) > 0
            )
        except Exception:
            return False
