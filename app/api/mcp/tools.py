"""
What the three read-only MCP tools do, written against the domain ports.

server.py publishes these methods over MCP (names, descriptions, input
schemas, transport); this module holds their behaviour, so it can be read and
tested without the protocol. Every tool:

- uses the same SearchService and BookCatalogRepository as the REST API;
- never calls the LLM: search_books runs the plain search (no query
  understanding, explanations off) and similar_books is pure vector search,
  so a public server cannot spend the owner's LLM budget;
- never writes anything;
- reports a problem the caller can fix (empty query, unknown id, bad
  filters) as a ToolError, whose message the model reads and can act on;
- keeps its JSON payload under a byte limit, by dropping trailing results
  or shortening descriptions.
"""

import logging
from typing import Callable, Optional, TypeVar
from uuid import UUID

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from app.api.mcp.schemas import (
    AppliedFilters,
    BookRecord,
    DescriptionEntry,
    ProvenanceEntry,
    SearchBooksResult,
    SearchHit,
    SimilarBooksResult,
    SimilarHit,
)
from app.domain.entities import Book, SearchResult
from app.domain.ports import BookCatalogRepository
from app.domain.services import SearchService
from app.domain.value_objects import SearchFilters, SearchQuery

logger = logging.getLogger(__name__)

# Bounds and lengths stated in the tool descriptions (server.py)
MAX_RESULTS_LIMIT = 20
DEFAULT_MAX_RESULTS = 10
MAX_QUERY_CHARS = 500
DESCRIPTION_PREVIEW_CHARS = 300  # description length in search and similarity results
FULL_DESCRIPTION_MAX_CHARS = 8_000  # description length in get_book
MIN_DESCRIPTION_CHARS = 200  # get_book never shortens descriptions below this to fit the size cap

ELLIPSIS = "..."


# The two tool results that hold a list of books (and a "truncated" flag)
ListResultT = TypeVar("ListResultT", SearchBooksResult, SimilarBooksResult)


def truncate_text(text: Optional[str], max_chars: int) -> tuple[Optional[str], bool]:
    """
    Cut a text to at most max_chars characters, marking the cut with "...".

    Returns:
        The (possibly shortened) text and whether it was cut.
    """
    if text is None or len(text) <= max_chars:
        return text, False
    return text[: max_chars - len(ELLIPSIS)].rstrip() + ELLIPSIS, True


def payload_size(result: BaseModel) -> int:
    """Size in bytes of a tool result as compact UTF-8 JSON."""
    return len(result.model_dump_json().encode("utf-8"))


def _score(value: Optional[float]) -> Optional[float]:
    """Plain rounded float (FAISS returns numpy floats)."""
    return None if value is None else round(float(value), 6)


def _summary_fields(book: Book) -> dict:
    """Fields shared by search and similarity results, description shortened."""
    description, cut = truncate_text(book.description, DESCRIPTION_PREVIEW_CHARS)
    return {
        "book_id": book.id,
        "title": book.title,
        "authors": list(book.authors),
        "year": book.get_published_year(),
        "language": book.language,
        "categories": list(book.categories),
        "description": description,
        "description_truncated": cut,
        "source": book.source,
        "source_id": book.source_id,
    }


def to_search_hit(result: SearchResult) -> SearchHit:
    """Map a domain SearchResult to the search_books output item."""
    return SearchHit(
        **_summary_fields(result.book),
        rank=result.rank,
        score=_score(result.final_score),
        keyword_score=_score(result.lexical_score),
        semantic_score=_score(result.vector_score),
        rerank_score=_score(result.rerank_score),
        retrieved_by=result.source,
    )


def to_similar_hit(result: SearchResult) -> SimilarHit:
    """Map a domain SearchResult (vector search) to the similar_books output item."""
    return SimilarHit(**_summary_fields(result.book), rank=result.rank, similarity=_score(result.vector_score))


def to_book_record(book: Book, max_description_chars: int = FULL_DESCRIPTION_MAX_CHARS) -> BookRecord:
    """
    Map a domain Book to the full record returned by get_book.

    The multi-source metadata fields (descriptions with licence, provenance)
    are read with getattr, so the mapping works both with the original
    Google Books metadata and with the multi-source catalog.
    """
    description, cut = truncate_text(book.description, max_description_chars)
    metadata = book.metadata
    published = book.published_date.date().isoformat() if book.published_date else None

    descriptions = [
        DescriptionEntry(
            text=truncate_text(entry.text, max_description_chars)[0],
            language=entry.language,
            source=entry.source,
            license=entry.license,
            attribution_url=entry.attribution_url,
        )
        for entry in getattr(metadata, "descriptions", ())
    ]
    provenance = [
        ProvenanceEntry(source=record.source, record_id=record.record_id, license=record.license, url=record.url)
        for record in getattr(metadata, "provenance", ())
    ]

    return BookRecord(
        book_id=book.id,
        title=book.title,
        authors=list(book.authors),
        year=book.get_published_year(),
        published_date=published,
        language=book.language,
        categories=list(book.categories),
        description=description,
        description_truncated=cut,
        source=book.source,
        source_id=book.source_id,
        isbn=getattr(metadata, "isbn", None),
        isbn13=getattr(metadata, "isbn13", None),
        publisher=getattr(metadata, "publisher", None),
        page_count=getattr(metadata, "page_count", None),
        average_rating=getattr(metadata, "average_rating", None),
        ratings_count=getattr(metadata, "ratings_count", None),
        thumbnail_url=getattr(metadata, "thumbnail_url", None),
        preview_link=getattr(metadata, "preview_link", None),
        descriptions=descriptions,
        provenance=provenance,
    )


class CatalogTools:
    """
    The read-only catalog tools, independent of the MCP protocol.

    Dependencies are passed as providers (callables) and resolved on each
    call, so the search indexes and models load on the first tool call, as
    with the REST endpoints, and tests can pass fakes.
    """

    def __init__(
        self,
        search_service: Callable[[], SearchService],
        catalog: Callable[[], BookCatalogRepository],
        max_response_bytes: int,
    ) -> None:
        """
        Args:
            search_service: Provider of the SearchService (hybrid and vector search)
            catalog: Provider of the catalog repository (get a book by id)
            max_response_bytes: Largest JSON payload a tool may return
        """
        self._search_service = search_service
        self._catalog = catalog
        self._max_bytes = max_response_bytes

    # ------------------------------------------------------------------ tools

    def search_books(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        language: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        category: Optional[str] = None,
    ) -> SearchBooksResult:
        """The server's search (SearchService.search_with_fallback, optional re-ranking); no LLM call."""
        text = query.strip()
        if not text:
            raise ToolError(
                "The query is empty. Describe the books you are looking for, "
                "for example 'novels about the Spanish civil war'."
            )
        if len(text) > MAX_QUERY_CHARS:
            raise ToolError(f"The query is too long ({len(text)} characters); use at most {MAX_QUERY_CHARS}.")
        self._check_max_results(max_results)

        try:
            filters = SearchFilters(
                language=language.strip().lower() if language else None,
                category=category.strip() if category and category.strip() else None,
                min_year=min_year,
                max_year=max_year,
            )
        except ValueError as e:
            raise ToolError(f"Invalid filters: {e}.") from e

        # use_explanations stays False and search_with_understanding is never
        # used: this path cannot reach the LLM client.
        search_query = SearchQuery(text=text, filters=filters, max_results=max_results, use_explanations=False)
        try:
            response = self._search_service().search_with_fallback(search_query)
        except RuntimeError as e:
            logger.error(f"MCP search_books failed: {e}")
            raise ToolError("Search is temporarily unavailable on this server. Try again later.") from e

        result = SearchBooksResult(
            query=text,
            filters=AppliedFilters(
                language=filters.language,
                category=filters.category,
                min_year=filters.min_year,
                max_year=filters.max_year,
            ),
            search_mode=response.search_mode,
            reranked=any(r.rerank_score is not None for r in response.results),
            results=[to_search_hit(r) for r in response.results],
        )
        return self._fit_list(result)

    def get_book(self, book_id: UUID) -> BookRecord:
        """Full record of one book; the description is shortened only to respect the size cap."""
        book = self._require_book(book_id)

        max_chars = FULL_DESCRIPTION_MAX_CHARS
        record = to_book_record(book, max_chars)
        while payload_size(record) > self._max_bytes and max_chars > MIN_DESCRIPTION_CHARS:
            max_chars = max(MIN_DESCRIPTION_CHARS, max_chars // 2)
            record = to_book_record(book, max_chars)
        if payload_size(record) > self._max_bytes:
            raise ToolError(f"The record of book {book_id} is too large for this server's response limit.")
        return record

    def similar_books(self, book_id: UUID, max_results: int = DEFAULT_MAX_RESULTS) -> SimilarBooksResult:
        """Books closest to the given one in embedding space (item-to-item); no LLM call."""
        self._check_max_results(max_results)
        book = self._require_book(book_id)
        try:
            results = self._search_service().find_similar_books(book_id=book_id, max_results=max_results)
        except ValueError as e:
            raise ToolError(
                f"Book {book_id} has no vector in the similarity index, so similar books "
                "cannot be computed for it. Try search_books with its title or topic instead."
            ) from e

        result = SimilarBooksResult(
            book_id=book.id,
            title=book.title,
            results=[to_similar_hit(r) for r in results],
        )
        return self._fit_list(result)

    # ---------------------------------------------------------------- helpers

    def _require_book(self, book_id: UUID) -> Book:
        """Return the book or raise a ToolError a model can act on."""
        book = self._catalog().get_by_id(book_id)
        if book is None:
            raise ToolError(
                f"No book with id {book_id} in the catalog. "
                "Use search_books to find books and their ids."
            )
        return book

    @staticmethod
    def _check_max_results(max_results: int) -> None:
        """Same bound as the input schema, for callers that bypass the SDK."""
        if not 1 <= max_results <= MAX_RESULTS_LIMIT:
            raise ToolError(f"max_results must be between 1 and {MAX_RESULTS_LIMIT}, got {max_results}.")

    def _fit_list(self, result: ListResultT) -> ListResultT:
        """Drop results from the end until the payload fits the size cap."""
        while result.results and payload_size(result) > self._max_bytes:
            result = result.model_copy(update={"results": result.results[:-1], "truncated": True})
        return result
