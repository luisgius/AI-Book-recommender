#!/usr/bin/env python3
"""
Book Ingestion Script.

This script fetches books from Google Books API, saves them to the catalog,
and builds search indices (BM25 + FAISS).

Usage:
    python -m scripts.ingest_books --query "science fiction" --max-results 50
"""

import argparse
import logging
import sys
from pathlib import Path

from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
from scripts.ingest_and_build import main as ingest_and_build_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths (deprecated - use scripts/ingest_and_build.py instead)
DEFAULT_DB_PATH = Path("data/catalog.db")
DEFAULT_BM25_INDEX_PATH = "data/indexes/bm25_index.pkl"
DEFAULT_FAISS_INDEX_PATH = "data/indexes/faiss_index"


def create_directories(db_path: Path, bm25_path: str, faiss_path: str) -> None:
    """Ensure all required directories exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    Path(bm25_path).parent.mkdir(parents=True, exist_ok=True)
    Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)


def main(query: str, max_results: int = 100, language: str = None) -> int:
    """
    Main entry point for the ingestion script.
    
    Args:
        query: Search query for Google Books API
        max_results: Maximum number of books to fetch
        language: Optional language filter (e.g., 'es', 'en')
        
    Returns:
        Number of books ingested
    """
    logger.info(f"Starting ingestion: query='{query}', max_results={max_results}")
    
    # Create directories if they don't exist
    create_directories(DEFAULT_DB_PATH, DEFAULT_BM25_INDEX_PATH, DEFAULT_FAISS_INDEX_PATH)
    
    try:
        ingest_and_build_main(
            query=query,
            max_items=max_results,
            db_path=str(DEFAULT_DB_PATH),
            out_dir=str(Path(DEFAULT_BM25_INDEX_PATH).parent),
            language=language,
            rebuild=False,
            smoke_test_query=None,
        )
        catalog_repo = SqliteBookCatalogRepository(DEFAULT_DB_PATH)
        return catalog_repo.count()
    except RuntimeError as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest books from Google Books API")
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Search query for Google Books API"
    )
    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=100,
        help="Maximum number of books to fetch (default: 100)"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Language filter (ISO 639-1 code, e.g., 'es', 'en')"
    )
    
    args = parser.parse_args()
    main(args.query, args.max_results, args.language)
