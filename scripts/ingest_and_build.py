#!/usr/bin/env python3
"""
Book Ingestion and Index Building Script (End-to-End MVP).

This script orchestrates the complete ingestion and indexing pipeline:
1. Fetch books from Google Books API
2. Save them to the catalog (SQLite)
3. Build and persist BM25 index (lexical search)
4. Generate embeddings and build FAISS index (vector search)
5. Run a smoke test with SearchService to verify the system works

Usage:
    python -m scripts.ingest_and_build --query "science fiction" --max-items 50

Args:
    --query: Search query for Google Books API (required unless --rebuild)
    --max-items: Maximum number of books to fetch (default: 100)
    --db-path: Path to SQLite database (default: data/catalog.db)
    --out-dir: Directory for storing indices (default: data/indexes)
    --rebuild: Rebuild indices from existing catalog without fetching new books
    --smoke-test-query: Query for smoke test (default: uses --query; if omitted in --rebuild, uses 'programming')
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
from app.infrastructure.external.google_books_provider import GoogleBooksProvider
from app.infrastructure.search.bm25_search_repository import BM25SearchRepository
from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository
from app.ingestion.ingestion_service import IngestionService
from app.domain.services import SearchService
from app.domain.value_objects import SearchQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = "data/catalog.db"
DEFAULT_OUT_DIR = "data/indexes"


def create_directories(db_path: str, out_dir: str) -> None:
    """Ensure all required directories exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directories: db={Path(db_path).parent}, indexes={out_dir}")


def ingest_books(
    query: str,
    max_items: int,
    catalog_repo: SqliteBookCatalogRepository,
    language: Optional[str] = None,
) -> int:
    """
    Step 1: Ingest books from Google Books API.

    Args:
        query: Search query
        max_items: Maximum number of books to fetch
        catalog_repo: Catalog repository
        language: Optional language filter

    Returns:
        Number of books inserted into catalog
    """
    logger.info("=" * 70)
    logger.info("STEP 1: INGESTING BOOKS FROM GOOGLE BOOKS API")
    logger.info("=" * 70)

    catalog_before = catalog_repo.count()

    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
    provider = GoogleBooksProvider(api_key=api_key if api_key else None)
    ingestion_service = IngestionService(
        books_provider=provider,
        catalog_repo=catalog_repo,
    )

    summary = ingestion_service.ingest_books(query, max_items, language)

    catalog_after = catalog_repo.count()

    logger.info(f"Ingestion summary:")
    logger.info(f"  - Fetched: {summary.n_fetched}")
    logger.info(f"  - Inserted: {summary.n_inserted}")
    logger.info(f"  - Skipped (duplicates): {summary.n_skipped}")
    logger.info(f"  - Errors: {summary.n_errors}")
    logger.info(f"  - Catalog size before: {catalog_before}")
    logger.info(f"  - Catalog size after: {catalog_after}")

    if summary.errors:
        logger.warning(f"Errors encountered during ingestion:")
        for error in summary.errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")

    return summary.n_inserted


def build_bm25_index(
    catalog_repo: SqliteBookCatalogRepository,
    out_dir: str,
) -> BM25SearchRepository:
    """
    Step 2: Build and persist BM25 index.

    Args:
        catalog_repo: Catalog repository
        out_dir: Output directory for index

    Returns:
        BM25 search repository with built index
    """
    logger.info("=" * 70)
    logger.info("STEP 2: BUILDING BM25 INDEX")
    logger.info("=" * 70)

    bm25_repo = BM25SearchRepository()
    books = catalog_repo.get_all()

    if not books:
        logger.warning("No books in catalog. Skipping BM25 index build.")
        return bm25_repo

    logger.info(f"Building BM25 index for {len(books)} books...")
    bm25_repo.build_index(books)

    bm25_index_path = Path(out_dir) / "bm25_index.pkl"
    bm25_repo.save_index(str(bm25_index_path))
    logger.info(f"BM25 index saved to: {bm25_index_path}")

    return bm25_repo


def build_faiss_index(
    catalog_repo: SqliteBookCatalogRepository,
    out_dir: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[EmbeddingsStoreFaiss, FaissVectorSearchRepository]:
    """
    Step 3: Generate embeddings and build FAISS index.

    Args:
        catalog_repo: Catalog repository
        out_dir: Output directory for index
        model_name: Sentence transformer model name

    Returns:
        Tuple of (embeddings_store, vector_search_repo)
    """
    logger.info("=" * 70)
    logger.info("STEP 3: BUILDING FAISS INDEX (EMBEDDINGS)")
    logger.info("=" * 70)

    books = catalog_repo.get_all()

    if not books:
        logger.warning("No books in catalog. Skipping FAISS index build.")
        embeddings_store = EmbeddingsStoreFaiss(model_name=model_name)
        vector_repo = FaissVectorSearchRepository(embeddings_store, [])
        return embeddings_store, vector_repo

    logger.info(f"Initializing embeddings model: {model_name}")
    embeddings_store = EmbeddingsStoreFaiss(model_name=model_name)

    logger.info(f"Generating embeddings for {len(books)} books...")
    texts = [book.get_searchable_text() for book in books]
    book_ids = [book.id for book in books]

    embeddings = embeddings_store.generate_embeddings_batch(texts)
    embeddings_store.store_embeddings_batch(book_ids, embeddings)

    logger.info("Building FAISS index from embeddings...")
    embeddings_store.build_index()

    faiss_index_path = Path(out_dir) / "faiss_index"
    embeddings_store.save_index(str(faiss_index_path))
    logger.info(f"FAISS index saved to: {faiss_index_path}")

    # Create vector search repo with book map for hydration
    vector_repo = FaissVectorSearchRepository(embeddings_store, books)
    logger.info(f"Vector search repository initialized with {len(books)} books")

    return embeddings_store, vector_repo


def run_smoke_test(
    query: str,
    lexical_repo: BM25SearchRepository,
    vector_repo: FaissVectorSearchRepository,
    embeddings_store: EmbeddingsStoreFaiss,
    max_results: int = 5,
) -> None:
    """
    Step 4: Run smoke test using SearchService.

    Args:
        query: Search query for smoke test
        lexical_repo: BM25 search repository
        vector_repo: Vector search repository
        max_results: Number of results to display
    """
    logger.info("=" * 70)
    logger.info("STEP 4: SMOKE TEST WITH SEARCHSERVICE")
    logger.info("=" * 70)

    search_service = SearchService(
        lexical_search=lexical_repo,
        vector_search=vector_repo,
        embeddings_store=embeddings_store,
    )

    search_query = SearchQuery(text=query, max_results=max_results)

    logger.info(f"Running search for: '{query}'")
    response = search_service.search_with_fallback(search_query)

    if response.degraded:
        logger.warning(f"Search degraded: {response.degradation_reason}")
        logger.warning(f"Search mode: {response.search_mode}")

    if not response.results:
        logger.warning("No results found!")
        return

    def _fmt(score: Optional[float]) -> str:
        return f"{score:.4f}" if score is not None else "n/a"

    logger.info(f"\nTop {len(response.results)} results:")
    logger.info("-" * 70)

    for i, result in enumerate(response.results, 1):
        logger.info(f"\n{i}. {result.book.title}")
        logger.info(f"   Authors: {', '.join(result.book.authors)}")
        logger.info(
            f"   Score: {result.final_score:.4f} "
            f"(lexical={_fmt(result.lexical_score)}, vector={_fmt(result.vector_score)})"
        )
        logger.info(f"   Source: {result.source}")
        if result.book.categories:
            logger.info(f"   Categories: {', '.join(result.book.categories)}")

    logger.info("-" * 70)
    logger.info(f"Smoke test completed successfully! Found {len(response.results)} results.")


def rebuild_indices_only(
    db_path: str,
    out_dir: str,
) -> tuple[BM25SearchRepository, EmbeddingsStoreFaiss, FaissVectorSearchRepository]:
    """
    Rebuild indices from existing catalog without fetching new books.

    Args:
        db_path: Path to SQLite database
        out_dir: Output directory for indices

    Returns:
        Tuple of (bm25_repo, vector_repo)
    """
    logger.info("=" * 70)
    logger.info("REBUILD MODE: Rebuilding indices from existing catalog")
    logger.info("=" * 70)

    catalog_repo = SqliteBookCatalogRepository(Path(db_path))
    books = catalog_repo.get_all()
    logger.info(f"Found {len(books)} books in catalog")

    bm25_repo = build_bm25_index(catalog_repo, out_dir)
    embeddings_store, vector_repo = build_faiss_index(catalog_repo, out_dir)

    return bm25_repo, embeddings_store, vector_repo


def main(
    query: Optional[str],
    max_items: int = 100,
    db_path: str = DEFAULT_DB_PATH,
    out_dir: str = DEFAULT_OUT_DIR,
    language: Optional[str] = None,
    rebuild: bool = False,
    smoke_test_query: Optional[str] = None,
) -> None:
    """
    Main entry point for the ingestion and build script.

    Args:
        query: Search query for Google Books API
        max_items: Maximum number of books to fetch
        db_path: Path to SQLite database
        out_dir: Directory for storing indices
        language: Optional language filter (e.g., 'es', 'en')
        rebuild: If True, rebuild indices from existing catalog
        smoke_test_query: Query for smoke test (defaults to query)
    """
    logger.info("Starting Book Ingestion and Index Building Pipeline")
    logger.info(f"Configuration:")
    logger.info(f"  - Query: '{query}'" if query else "  - Query: n/a")
    logger.info(f"  - Max items: {max_items}")
    logger.info(f"  - DB path: {db_path}")
    logger.info(f"  - Output dir: {out_dir}")
    logger.info(f"  - Language: {language or 'all'}")
    logger.info(f"  - Rebuild mode: {rebuild}")
    logger.info("")

    if not rebuild and (query is None or not query.strip()):
        raise ValueError("--query is required unless --rebuild is set")

    # Create directories
    create_directories(db_path, out_dir)

    if rebuild:
        # Rebuild mode: skip ingestion
        bm25_repo, embeddings_store, vector_repo = rebuild_indices_only(db_path, out_dir)
    else:
        # Normal mode: ingest + build indices
        catalog_repo = SqliteBookCatalogRepository(Path(db_path))

        # Step 1: Ingest books
        if query is None:
            raise ValueError("--query is required unless --rebuild is set")
        n_inserted = ingest_books(query, max_items, catalog_repo, language)

        if n_inserted == 0 and not catalog_repo.get_all():
            logger.error("No books in catalog after ingestion. Cannot build indices.")
            sys.exit(1)

        # Step 2: Build BM25 index
        bm25_repo = build_bm25_index(catalog_repo, out_dir)

        # Step 3: Build FAISS index
        embeddings_store, vector_repo = build_faiss_index(catalog_repo, out_dir)

    # Step 4: Smoke test
    test_query = smoke_test_query or query or "programming"
    try:
        run_smoke_test(test_query, bm25_repo, vector_repo, embeddings_store)
    except Exception as e:
        logger.error(f"Smoke test failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info("Next steps:")
    if (project_root / "app" / "main.py").exists():
        logger.info("  1. Start the API: python -m app.main")
    logger.info("  2. Test search: POST /search with your query")
    logger.info("  3. Run evaluation: python -m scripts.evaluate_search")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest books and build search indices (end-to-end MVP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        required=False,
        default=None,
        help="Search query for Google Books API",
    )
    parser.add_argument(
        "--max-items", "-n",
        type=int,
        default=100,
        help="Maximum number of books to fetch (default: 100)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for storing indices (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Language filter (ISO 639-1 code, e.g., 'es', 'en')",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild indices from existing catalog without fetching new books",
    )
    parser.add_argument(
        "--smoke-test-query",
        type=str,
        default=None,
        help="Custom query for smoke test (defaults to --query)",
    )

    args = parser.parse_args()

    if args.query is not None:
        args.query = args.query.strip()
    if args.smoke_test_query is not None:
        args.smoke_test_query = args.smoke_test_query.strip()

    if not args.rebuild and (args.query is None or not args.query):
        parser.error("--query is required unless --rebuild is set")

    try:
        main(
            query=args.query,
            max_items=args.max_items,
            db_path=args.db_path,
            out_dir=args.out_dir,
            language=args.language,
            rebuild=args.rebuild,
            smoke_test_query=args.smoke_test_query,
        )
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
