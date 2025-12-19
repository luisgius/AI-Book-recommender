"""
Port interfaces (protocols) for the domain layer.

Ports define the contracts between the domain and infrastructure layers.
They are implemented by adapters in the infrastructure layer, allowing
the domain to remain independent of technical details.

Following Hexagonal Architecture principles, the domain layer depends only
on these abstract protocols, never on concrete implementations.
"""

from typing import Protocol, List, Optional, Dict, Any
from uuid import UUID

from .entities import Book, SearchResult, Explanation
from .value_objects import SearchQuery, SearchFilters

class BookCatalogRepository(Protocol):
    """
    Port for persisting and retrieving books from the catalog.

    This repository is responsible for CRUD operations on Book entities.
    It abstracts away the persistence mechanism (SQLite, PostgreSQL, etc.).

    Implementations should handle:
    - Unique constraint on (source, source_id) to avoid duplicates
    - Proper error handling for database constraints
    - Efficient batch operations for bulk ingestion
    """

    def save(self, book: Book) -> None:
        """
        Save a book to the catalog.

        If a book with the same ID exists, it should be updated.
        If a book with the same (source, source_id) exists but different ID,
        the implementation should decide whether to update or raise an error.

        Args:
            book: The book entity to persist

        Raises:
            ValueError: If book data violates catalog constraints
            RuntimeError: If a database error occurs
        """
        ...


    
    def save_many(self, books: List[Book]) -> None:
        """
        Save multiple books in a single transaction for efficiency.

        This is useful during bulk ingestion from external sources.

        Args:
            books: List of book entities to persist

        Raises:
            ValueError: If any book violates catalog constraints
            RuntimeError: If a database error occurs
        """
        ...

    def get_by_id(self, book_id: UUID) -> Optional[Book]:
        """
        Retrieve a book by its internal UUID.

        Args:
            book_id: The unique identifier of the book

        Returns:
            The Book entity if found, None otherwise
        """
        ...

    def get_by_source_id(self, source: str, source_id: str) -> Optional[Book]:
        """
        Retrieve a book by its external source identifier.

        This is useful to check if a book from an external API
        already exists in our catalog.

        Args:
            source: Source system name (e.g., 'google_books')
            source_id: ID in the external system

        Returns:
            The Book entity if found, None otherwise
        """
        ...

    def get_all(self, limit: Optional[int] = None) -> List[Book]:
        """
        Retrieve all books from the catalog.

        Args:
            limit: Optional maximum number of books to return

        Returns:
            List of all books, up to the specified limit
        """
        ...

    def count(self) -> int:
        """
        Get the total number of books in the catalog.

        Returns:
            Total book count
        """
        ...

    def delete(self, book_id: UUID) -> bool:
        """
        Delete a book from the catalog.

        Args:
            book_id: ID of the book to delete

        Returns:
            True if the book was deleted, False if not found
        """
        ...


class LexicalSearchRepository(Protocol):
    """
    Port for lexical (BM25) search over book content.

    This repository is responsible for:
    - Building and maintaining a BM25 index over book text
    - Executing keyword-based searches
    - Returning scored results

    The index should be built from Book.get_searchable_text() and
    kept in sync with the catalog.
    """

    def build_index(self, books: List[Book]) -> None:
        """
        Build or rebuild the BM25 index from a list of books.

        This should be called during initial ingestion and whenever
        the catalog is significantly updated.

        Args:
            books: List of books to index

        Raises:
            RuntimeError: If index building fails
        """
        ...

    def add_to_index(self, book: Book) -> None:
        """
        Add a single book to the existing index.

        This is more efficient than rebuilding the entire index
        when adding new books incrementally.

        Args:
            book: The book to add to the index

        Raises:
            RuntimeError: If adding to index fails
        """
        ...

    def search(
        self,
        query_text: str,
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """
        Perform a BM25 search over the indexed books.

        Args:
            query_text: The search query string
            max_results: Maximum number of results to return
            filters: Optional filters to apply (language, category, year range).
                    If None, return all results without filtering.
                    If provided, use filters.is_empty() to check if any filters are set.

        Returns:
            List of SearchResult entities, ranked by BM25 score (descending).

            IMPORTANT: Each result MUST have:
            - book: The matched book entity
            - final_score: set to the BM25 score (higher = better match)
            - lexical_score: set to the same BM25 score
            - source: set to "lexical"
            - rank: 1-indexed position (1, 2, 3, ...) based on BM25 score ordering
            - vector_score: None (not applicable for lexical search)

            The rank field is critical for RRF fusion in SearchService.

        Raises:
            ValueError: If query_text is empty or invalid
            RuntimeError: If search execution fails
        """
        ...

    def save_index(self, path: str) -> None:
        """
        Persist the index to disk for later reuse.

        Args:
            path: File path where the index should be saved

        Raises:
            IOError: If saving fails
        """
        ...

    def load_index(self, path: str) -> None:
        """
        Load a previously saved index from disk.

        Args:
            path: File path to the saved index

        Raises:
            IOError: If loading fails
            ValueError: If the index format is invalid
        """
        ...

    def is_ready(self) -> bool:
        """
        Check if the index is loaded and ready for search operations.

        Used for health checks and graceful degradation (RNF-08).

        Returns:
            True if the index is loaded and searchable, False otherwise
        """
        ...


class VectorSearchRepository(Protocol):
    """
    Port for semantic vector search using embeddings.

    This repository is responsible for:
    - Querying the vector index (ANN search) given a query embedding
    - Returning the most similar books based on cosine similarity
    - Applying post-search filters

    This protocol performs search operations over the index built and maintained
    by EmbeddingsStore. It does NOT handle embedding generation or index building.
    """

    def search(
        self,
        query_embedding: List[float],
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """
        Perform a vector similarity search.

        Given a query embedding (typically generated from the user's query text),
        find the most similar book embeddings in the index using approximate
        nearest neighbor search.

        Args:
            query_embedding: The embedding vector for the query (must match
                            the dimensionality of indexed book embeddings)
            max_results: Maximum number of results to return
            filters: Optional filters to apply post-search (language, category, year).
                    If None, return all results without filtering.
                    If provided, use filters.is_empty() to check if any filters are set.

        Returns:
            List of SearchResult entities, ranked by similarity score (descending).

            IMPORTANT: Each result MUST have:
            - book: The matched book entity
            - final_score: set to the cosine similarity score (higher = better match)
            - vector_score: set to the same cosine similarity score
            - source: set to "vector"
            - rank: 1-indexed position (1, 2, 3, ...) based on similarity ordering
            - lexical_score: None (not applicable for vector search)

            The rank field is critical for RRF fusion in SearchService.

        Raises:
            ValueError: If query_embedding has wrong dimensionality
            RuntimeError: If search execution fails
        """
        ...

    def is_ready(self) -> bool:
        """
        Check if the vector index is loaded and ready for search operations.

        Used for health checks and graceful degradation (RNF-08).

        Returns:
            True if the index is loaded and searchable, False otherwise
        """
        ...


class EmbeddingsStore(Protocol):
    """
    Port for managing and storing vector embeddings.

    This store is responsible for:
    - Generating embeddings for text (book content, queries)
    - Storing embeddings with book associations (UUID -> embedding vector)
    - Building and maintaining the ANN index (e.g., FAISS)
    - Persisting and loading the index to/from disk

    This store does NOT perform search operations. Search is the responsibility
    of VectorSearchRepository, which uses the index built by this store.

    The embeddings model (e.g., sentence-transformers) is an implementation
    detail hidden behind this port.
    """

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.

        This is typically used for:
        - Encoding book searchable text during ingestion
        - Encoding user query text during search

        Args:
            text: The text to encode

        Returns:
            Embedding vector (list of floats)

        Raises:
            ValueError: If text is empty or too long
            RuntimeError: If embedding generation fails
        """
        ...

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a batch.

        This is more efficient than calling generate_embedding repeatedly.
        Used during bulk ingestion.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors, in the same order as input texts

        Raises:
            ValueError: If any text is invalid
            RuntimeError: If batch generation fails
        """
        ...

    def store_embedding(self, book_id: UUID, embedding: List[float]) -> None:
        """
        Store an embedding associated with a book ID.

        This updates the internal mapping and should trigger an index rebuild
        or incremental update.

        Args:
            book_id: The book's unique identifier
            embedding: The embedding vector for this book

        Raises:
            ValueError: If embedding dimension is incorrect
            RuntimeError: If storage fails
        """
        ...

    def store_embeddings_batch(
        self, book_ids: List[UUID], embeddings: List[List[float]]
    ) -> None:
        """
        Store multiple embeddings in a batch.

        Args:
            book_ids: List of book IDs
            embeddings: List of corresponding embedding vectors

        Raises:
            ValueError: If lists have different lengths or invalid data
            RuntimeError: If batch storage fails
        """
        ...

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        """
        Retrieve the stored embedding for a book.

        Args:
            book_id: The book's unique identifier

        Returns:
            The embedding vector if found, None otherwise
        """
        ...

    def build_index(self) -> None:
        """
        Build or rebuild the ANN index from all stored embeddings.

        This should be called after bulk ingestion or significant updates.

        Raises:
            RuntimeError: If index building fails
        """
        ...

    def save_index(self, path: str) -> None:
        """
        Persist the ANN index to disk.

        Args:
            path: Directory path where index files should be saved

        Raises:
            IOError: If saving fails
        """
        ...

    def load_index(self, path: str) -> None:
        """
        Load a previously saved ANN index from disk.

        Args:
            path: Directory path to the saved index files

        Raises:
            IOError: If loading fails
            ValueError: If index format is invalid
        """
        ...

    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Returns:
            The embedding dimension (e.g., 384 for MiniLM, 768 for BERT)
        """
        ...

    def is_ready(self) -> bool:
        """
        Check if the embeddings store and index are ready for operations.

        Used for health checks and graceful degradation (RNF-08).

        Returns:
            True if embeddings can be generated and index is searchable, False otherwise
        """
        ...


class ExternalBooksProvider(Protocol):
    """
    Port for fetching book data from external APIs.

    This provider abstracts away the details of external book APIs
    (Google Books, Open Library, etc.) and normalizes their responses
    into our domain Book entities.

    Implementations should handle:
    - API authentication and rate limiting
    - Response parsing and normalization
    - Error handling for network issues
    """

    def search_books(
        self, query: str, max_results: int = 10, language: Optional[str] = None
    ) -> List[Book]:
        """
        Search for books in the external API.

        Args:
            query: Search query string
            max_results: Maximum number of results to fetch
            language: Optional language filter (ISO 639-1 code)

        Returns:
            List of Book entities with data from the external source

        Raises:
            ValueError: If query is invalid
            RuntimeError: If API request fails
        """
        ...

    def get_book_by_id(self, external_id: str) -> Optional[Book]:
        """
        Fetch a specific book by its external ID.

        Args:
            external_id: The book's ID in the external system

        Returns:
            A Book entity if found, None otherwise

        Raises:
            RuntimeError: If API request fails
        """
        ...

    def get_source_name(self) -> str:
        """
        Get the name of this external source.

        Returns:
            Source identifier (e.g., 'google_books', 'open_library')
        """
        ...


class LLMClient(Protocol):
    """
    Port for interacting with Large Language Models.

    This client is used for:
    - Query understanding and intent extraction
    - Generating natural language explanations for search results (RAG pattern)

    The implementation should support RAG-style flows:
    1. Retrieve relevant context (books, metadata)
    2. Construct a prompt with context
    3. Generate a response conditioned on that context

    The actual LLM (OpenAI, Anthropic, local model, etc.) and orchestration
    framework (LangChain, LangGraph) are implementation details.
    """

    def generate_explanation(
        self, query: SearchQuery, book: Book, context: Optional[Dict[str, Any]] = None
    ) -> Explanation:
        """
        Generate a natural language explanation for why a book is relevant.

        This implements the "generation" step of the RAG pattern:
        - Context (book data) has already been retrieved
        - The LLM generates an explanation conditioned on this context

        Args:
            query: The user's search query
            book: The book to explain
            context: Optional additional context (e.g., other results, metadata)

        Returns:
            An Explanation entity with the generated text

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If LLM generation fails
        """
        ...

    def extract_query_intent(
        self, query_text: str
    ) -> Dict[str, Any]:
        """
        Analyze a user query to extract intent and structured information.

        This can be used to:
        - Identify whether the query is asking for recommendations, facts, etc.
        - Extract implicit filters (e.g., "recent books" -> max_year filter)
        - Detect the query language

        Args:
            query_text: The raw user query

        Returns:
            A dictionary with extracted information, e.g.:
            {
                "intent": "recommendation",
                "filters": {"language": "en", "min_year": 2020},
                "reformulated_query": "science fiction novels"
            }

        Raises:
            ValueError: If query_text is empty
            RuntimeError: If LLM analysis fails
        """
        ...

    def get_model_name(self) -> str:
        """
        Get the identifier of the LLM model being used.

        Returns:
            Model name (e.g., 'gpt-4', 'claude-2', 'llama-2-7b')
        """
        ...

