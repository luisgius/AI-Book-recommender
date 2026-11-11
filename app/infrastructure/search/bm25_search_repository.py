"""
BM25-based implementation of lexical search repository.

This adapter implements the LexicalSearchRepository port using the BM25
algorithm for keyword-based search over book content.

BM25 (Best Match 25) is a probabilistic ranking function that scores documents
based on query term frequency, inverse document frequency, and document length
normalization. It is widely used for lexical search and forms one half of our
hybrid search approach.

Tokenization is pluggable by name (see text_normalization.py): "normalized"
for new indexes, "whitespace" for the legacy behaviour. The tokenizer name is
saved with the index so documents and queries are always tokenized the same
way.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Dict
from uuid import UUID

from rank_bm25 import BM25Okapi

from app.domain.entities import Book, SearchResult
from app.domain.value_objects import SearchFilters
from app.domain.ports import LexicalSearchRepository
from app.infrastructure.search.text_normalization import (
    NORMALIZED_TOKENIZER,
    WHITESPACE_TOKENIZER,
    Tokenizer,
    get_tokenizer,
)


class BM25SearchRepository(LexicalSearchRepository):
    """
    BM25-based lexical search repository.

    This implementation:
    - Builds an in-memory BM25 index from Book.get_searchable_text()
    - Tokenizes documents and queries with the same named tokenizer
    - Maintains a book_id -> Book mapping for duplicate detection
    - Returns SearchResult with complete Book entities (no DB lookups needed)
    - Supports index persistence via pickle (controlled environment only)
    - Applies filters in-memory after scoring

    Design decision: We keep books in memory (Option A from CLAUDE.md) to
    avoid coupling this repository to BookCatalogRepository and to keep
    search operations fast.
    """

    def __init__(self, tokenizer: str = NORMALIZED_TOKENIZER) -> None:
        """
        Create an empty repository.

        Args:
            tokenizer: Name of the tokenizer used to build new indexes and to
                tokenize queries ("normalized" or "whitespace"). load_index
                replaces it with the name stored in the index file.

        Raises:
            ValueError: If the tokenizer name is unknown.
        """
        self._tokenizer_name: str = tokenizer
        self._tokenizer: Tokenizer = get_tokenizer(tokenizer)
        self._index: Optional[BM25Okapi] = None
        self._books: List[Book] = []
        self._book_map: Dict[UUID, Book] = {}
        self._tokenized_corpus: List[List[str]] = []

    def build_index(self, books: List[Book]) -> None:
        """
        Build or rebuild the BM25 index from a list of books.

        This tokenizes each book's searchable text and creates a BM25 index.
        The index is built over title + authors + description + categories.

        Args:
            books: List of books to index

        Raises:
            RuntimeError: If index building fails
        """
        if not books:
            # Empty index is valid
            self._index = None
            self._books = []
            self._book_map = {}
            self._tokenized_corpus = []
            return

        try:
            # Store books and build lookup map (defensive copy)
            self._books = list(books)
            self._book_map = {book.id: book for book in self._books}

            # Tokenize corpus with the configured tokenizer (same one used for queries)
            self._tokenized_corpus = [
                self._tokenize(book.get_searchable_text()) for book in books
            ]

            # Build BM25 index
            self._index = BM25Okapi(self._tokenized_corpus)

        except Exception as e:
            raise RuntimeError(f"Failed to build BM25 index: {e}") from e

    def add_to_index(self, book: Book) -> None:
        """
        Add a single book to the existing index.

        Note: BM25Okapi does not natively support incremental updates.
        This implementation rebuilds the entire index, which is acceptable
        for small-to-medium catalogs but may be inefficient for very large ones.

        For production with frequent updates, consider:
        - Batch updates (accumulate changes, rebuild periodically)
        - Alternative BM25 implementations with incremental support
        - Elasticsearch or similar systems

        Args:
            book: The book to add to the index

        Raises:
            RuntimeError: If adding to index fails
        """
        try:
            # Avoid duplicates: check if book already indexed
            if book.id in self._book_map:
                # Book already exists; for simplicity, we skip it
                # In production, you might want to update/replace it
                return

            # Add book to collection
            self._books.append(book)
            self._book_map[book.id] = book

            # rebuild index (necessary with BM25Okapi)
            self._tokenized_corpus.append(self._tokenize(book.get_searchable_text()))
            self._index = BM25Okapi(self._tokenized_corpus)

        except Exception as e:
            raise RuntimeError(f"Failed to add book to BM25 index: {e}") from e

    
    def search(
        self,
        query_text: str,
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """
        Perform a BM25 search over the indexed books.

        Search process:
        1. Tokenize query text
        2. Compute BM25 scores for all documents
        3. Rank results by score (descending)
        4. Apply filters if provided
        5. Take top max_results
        6. Construct SearchResult entities with complete Book objects

        Args:
            query_text: The search query string
            max_results: Maximum number of results to return
            filters: Optional filters (language, category, year range)

        Returns:
            List of SearchResult entities, ranked by BM25 score (descending).
            Each result has:
            - book: The matched book entity
            - final_score: BM25 score
            - lexical_score: same as final_score
            - source: "lexical"
            - rank: 1-indexed position
            - vector_score: None

        Raises:
            ValueError: If query_text is empty or invalid
            RuntimeError: If search execution fails
        """
        # Validate input
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")

        # Handle empty index
        if not self._index or not self._books:
            return []

        try:
            # Tokenize query
            tokenized_query = self._tokenize(query_text)

            # Get BM25 scores for all docs
            scores = self._index.get_scores(tokenized_query)

            # Pair books with scores
            book_scores = list(zip(self._books, scores))

            # Sort by score descending
            book_scores.sort(key=lambda x: x[1], reverse=True)

            # Apply filters if provided
            if filters and not filters.is_empty():
                book_scores = [
                    (book, score)
                    for book, score in book_scores
                    if self._matches_filters(book, filters)
                ]

            # Take top max_results
            book_scores = book_scores[:max_results]

            # Construct SearchResult entities
            results = []
            for rank, (book, score) in enumerate(book_scores, start=1):
                result = SearchResult(
                    book=book,
                    final_score=float(score),
                    rank=rank,
                    source="lexical",
                    lexical_score=float(score),
                    vector_score=None,
                )
                results.append(result)

            return results
        except Exception as e:
            raise RuntimeError(f"BM25 search failed: {e}") from e

        
    def save_index(self, path: str) -> None:
        """
        Persist the index to disk using pickle.

        This saves:
        - The BM25 index object
        - The list of books
        - The book_id -> Book mapping
        - The tokenized corpus
        - The tokenizer name (so queries are tokenized like the documents)

        SECURITY WARNING: This uses pickle serialization, which is only safe
        in controlled environments. Never load pickle files from untrusted sources.
        For production, consider alternative serialization formats (JSON, msgpack, etc.).

        Args:
            path: File path where the index should be saved

        Raises:
            IOError: If saving fails
        """
        try:
            index_data = {
                "index": self._index,
                "books": self._books,
                "book_map": self._book_map,
                "tokenized_corpus": self._tokenized_corpus,
                "tokenizer": self._tokenizer_name,
            }

            # Ensure parent directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, "wb") as f:
                pickle.dump(index_data, f)

        except Exception as e:
            raise IOError(f"Failed to save BM25 index to {path}: {e}") from e        


    def load_index(self, path: str) -> None:
        """
        Load a previously saved index from disk.

        The tokenizer stored with the index replaces the one given to the
        constructor. Indexes saved before the tokenizer name was persisted
        were built with the whitespace tokenizer, so a file without that
        entry loads as "whitespace".

        SECURITY WARNING: This uses pickle deserialization. Only load files
        from trusted sources in controlled environments.

        Args:
            path: File path to the saved index

        Raises:
            IOError: If loading fails
            ValueError: If the index format is invalid
        """
        try:
            with open(path, "rb") as f:
                index_data = pickle.load(f)
            
            # validate structure
            required_keys = {"index", "books", "book_map", "tokenized_corpus"}
            if not all(key in index_data for key in required_keys):
                raise ValueError(
                    f"Invalid index format: missing required keys. "
                    f"Expected {required_keys}, got {set(index_data.keys())}"
                )
            
            # Resolve the tokenizer before changing any state
            tokenizer_name = index_data.get("tokenizer", WHITESPACE_TOKENIZER)
            tokenizer = get_tokenizer(tokenizer_name)

            self._tokenizer_name = tokenizer_name
            self._tokenizer = tokenizer
            self._index = index_data["index"]
            self._books = index_data["books"]
            self._book_map = index_data["book_map"]
            self._tokenized_corpus = index_data["tokenized_corpus"]

        except pickle.UnpicklingError as e:
            raise ValueError(f"Invalid index format at {path}: {e}") from e
        except FileNotFoundError as e:
            raise IOError(f"Index file not found at {path}: {e}") from e
        except Exception as e:
            raise IOError(f"Failed to load BM25 index from {path}: {e}") from e

    @property
    def tokenizer_name(self) -> str:
        """Name of the active tokenizer ("normalized" or "whitespace")."""
        return self._tokenizer_name

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing and search.

        Delegates to the active named tokenizer, so documents and queries
        always go through the same function.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        return self._tokenizer(text)

    @staticmethod
    def _matches_filters(book: Book, filters: SearchFilters) -> bool:
        """
        Check if a book matches the given filters.

        Delegates to Book.matches_filters, the single definition shared by
        the search service and every repository.
        """
        return book.matches_filters(filters)

    def is_ready(self) -> bool:
        """
        Check if the BM25 index is loaded and ready for search operations.

        Used for health checks and graceful degradation (RNF-08).

        Returns:
            True if the index is built and searchable, False otherwise
        """
        return self._index is not None and len(self._books) > 0