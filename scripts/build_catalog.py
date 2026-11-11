#!/usr/bin/env python3
"""
Build the multi-source book catalog (Wikidata/Wikipedia, Gutenberg, DOAB, Open Library).

Steps, each resumable and runnable on its own:

    python -m scripts.build_catalog download [--sources ...] [--languages ...] [--limit N]
    python -m scripts.build_catalog ingest   [--sources ...] [--languages ...] [--limit N] [--out DB]
    python -m scripts.build_catalog report   [--out DB]
    python -m scripts.build_catalog select   --size 5000 [--from DB] --out data/catalog_small/catalog.db

- download: fetches raw data into data/downloads/<source>/ (cached; a re-run
  only fetches what is missing).
- ingest: reads the downloads, normalises, filters, de-duplicates and merges
  them into the catalog (default data/catalog_v2/catalog.db). Sources are
  always ingested in the order wikidata, gutenberg, doab, open_library;
  finished sources are skipped on a re-run (use --restart to rebuild).
- report: writes CATALOG_REPORT.md and catalog_report.json next to the DB.
- select: copies a deterministic, language-stratified selection of popular
  works into a small catalog.

The original catalog (data/catalog.db) is never written by this script.
"""

import argparse
from dataclasses import asdict
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.domain.ports import BulkBooksSource
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
from app.infrastructure.db.sqlite_catalog_build_state import SqliteCatalogBuildState, SqliteDuplicateKeyIndex
from app.infrastructure.external import doab_source, gutenberg_source, openlibrary_dump_source
from app.infrastructure.external.bulk_download import USER_AGENT, PoliteHttpClient, download_file
from app.infrastructure.external.doab_source import DoabSource
from app.infrastructure.external.gutenberg_source import GutenbergSource
from app.infrastructure.external.openlibrary_dump_source import OpenLibraryDumpSource
from app.infrastructure.external.wikidata_wikipedia_source import DEFAULT_LANGUAGES, WikidataWikipediaSource
from app.ingestion.catalog_builder import CatalogBuildService, CatalogRules
from app.ingestion.catalog_report import compute_catalog_report, render_markdown
from app.ingestion.catalog_selection import describe_rule, language_mix, language_quotas, select_books, to_candidate

logger = logging.getLogger("build_catalog")

SOURCE_ORDER = ("wikidata", "gutenberg", "doab", "open_library")
DOWNLOADS_DIR = Path("data/downloads")
DEFAULT_OUT = Path("data/catalog_v2/catalog.db")
PROTECTED_DB = Path("data/catalog.db")
OPENLIBRARY_DUMP_DATE = "2026-10-31"
OPENLIBRARY_DUMP_URL = "https://archive.org/download/ol_dump_{date}/{name}"

# Wikimedia: 10 requests/minute for clients whose User-Agent has no contact
# details; 200/minute with a policy-compliant User-Agent (contact URL/e-mail).
ANONYMOUS_WIKI_RATE = 9 / 60
IDENTIFIED_WIKI_RATE = 2.5  # 150 requests/minute, a margin under the 200 limit


def make_source(name: str, downloads: Path, languages: List[str]) -> BulkBooksSource:
    """Instantiate the adapter of a source over its download directory."""
    if name == "wikidata":
        return WikidataWikipediaSource(downloads / "wikidata", languages=languages)
    if name == "gutenberg":
        return GutenbergSource(downloads / "gutenberg")
    if name == "doab":
        return DoabSource(downloads / "doab")
    if name == "open_library":
        return OpenLibraryDumpSource(downloads / "openlibrary")
    raise ValueError(f"Unknown source {name!r}; choose from {SOURCE_ORDER}")


def is_downloaded(name: str, downloads: Path) -> bool:
    """True when the files a source parses are present."""
    if name == "wikidata":
        return (downloads / "wikidata" / "sparql" / "articles.csv").exists()
    if name == "gutenberg":
        return (downloads / "gutenberg" / gutenberg_source.RDF_ARCHIVE_NAME).exists()
    if name == "doab":
        return (downloads / "doab" / doab_source.EXPORT_NAME).exists()
    if name == "open_library":
        # The raw works dump or its cached candidates file is enough; the source
        # itself fails loudly if the authors dump (needed for names) is missing.
        return OpenLibraryDumpSource(downloads / "openlibrary").is_available()
    return False


# --------------------------------------------------------------------- steps

def step_download(args: argparse.Namespace) -> int:
    """Fetch the raw data of every selected source (cached, resumable)."""
    downloads = Path(args.downloads)
    timings: Dict[str, float] = {}
    for name in args.sources:
        start = time.time()
        logger.info("Downloading %s ...", name)
        if name == "wikidata":
            source = WikidataWikipediaSource(downloads / "wikidata", languages=args.languages)
            identified = args.wiki_user_agent is not None
            headers_client = None
            if identified:
                import httpx

                headers_client = httpx.Client(
                    headers={"User-Agent": args.wiki_user_agent, "Accept-Encoding": "gzip, deflate"},
                    timeout=90.0, follow_redirects=True,
                )
            wiki = PoliteHttpClient(
                rate_per_second=IDENTIFIED_WIKI_RATE if identified else ANONYMOUS_WIKI_RATE,
                client=headers_client,
            )
            sparql = PoliteHttpClient(rate_per_second=0.5, timeout_s=300)
            missing = source.download(
                sparql, wiki,
                extract_languages=args.languages,
                limit=args.limit,
                workers=3 if identified else 1,
                time_budget_s=args.wiki_minutes * 60 if args.wiki_minutes else None,
            )
            logger.info("Wikipedia intros still missing per language: %s", json.dumps(missing))
            logger.info("Wikipedia API retries %d, rate-limit answers %d", wiki.retry_events, wiki.rate_limit_events)
        elif name == "gutenberg":
            download_file(gutenberg_source.RDF_ARCHIVE_URL, downloads / "gutenberg" / gutenberg_source.RDF_ARCHIVE_NAME)
        elif name == "doab":
            download_file(doab_source.EXPORT_URL, downloads / "doab" / doab_source.EXPORT_NAME)
        elif name == "open_library":
            for file_name in openlibrary_dump_source.dump_file_names(args.openlibrary_date):
                url = OPENLIBRARY_DUMP_URL.format(date=args.openlibrary_date, name=file_name)
                download_file(url, downloads / "openlibrary" / file_name, segments=args.segments)
        timings[name] = round(time.time() - start, 1)
    logger.info("Download times (s): %s", timings)
    return 0


def _guard_out(path: Path) -> None:
    if path.resolve() == PROTECTED_DB.resolve():
        raise SystemExit(f"Refusing to write {PROTECTED_DB}: it is the original catalog")


def step_ingest(args: argparse.Namespace) -> int:
    """Normalise, filter, de-duplicate and store the selected sources."""
    out = Path(args.out)
    _guard_out(out)
    if args.restart:
        for path in (out, out.with_name("match_keys.db")):
            for suffix in ("", "-wal", "-shm"):
                Path(str(path) + suffix).unlink(missing_ok=True)
    catalog = SqliteBookCatalogRepository(out, compact_json=True)
    state = SqliteCatalogBuildState(out)
    index = SqliteDuplicateKeyIndex(out.with_name("match_keys.db"))
    service = CatalogBuildService(catalog, index, CatalogRules(languages=tuple(args.languages)))

    downloads = Path(args.downloads)
    for position, name in enumerate(SOURCE_ORDER):
        if name not in args.sources:
            continue
        if state.status(name) == "done":
            logger.info("%s already ingested: skipped (use --restart to rebuild)", name)
            continue
        if not is_downloaded(name, downloads):
            logger.warning("%s is not downloaded: skipped", name)
            continue
        source = make_source(name, downloads, args.languages)
        start = time.time()
        state.mark_started(name, position)
        stats = service.ingest_source(source, limit=args.limit)
        payload = stats.to_dict()
        payload["seconds"] = round(time.time() - start, 1)
        state.mark_done(name, payload, asdict(source.get_license()))
        logger.info("%s ingested in %.0f s; catalog has %d works", name, payload["seconds"], catalog.count())
    index.close()
    return 0


def step_report(args: argparse.Namespace) -> int:
    """Write CATALOG_REPORT.md and catalog_report.json next to the catalog."""
    out = Path(args.out)
    if not out.exists():
        raise SystemExit(f"{out} does not exist: run the ingest step first")
    catalog = SqliteBookCatalogRepository(out, compact_json=True)
    state = SqliteCatalogBuildState(out)
    report = compute_catalog_report(catalog.iter_all(batch_size=2000), state.all_sources(), args.languages)
    report["database"] = {"path": str(out), "size_bytes": out.stat().st_size}
    (out.parent / "catalog_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    extra = [
        f"Database: `{out}` ({out.stat().st_size / 1e6:.0f} MB). Generated by "
        "`python -m scripts.build_catalog report` from the database and the per-source counters stored in it.",
        "Quality filter: title, author, and a description of at least 200 characters (after markup is "
        "stripped, truncated to about 1,200) in one of: " + ", ".join(args.languages) + ".",
    ]
    (out.parent / "CATALOG_REPORT.md").write_text(render_markdown(report, extra=extra), encoding="utf-8")
    logger.info("Report written to %s", out.parent / "CATALOG_REPORT.md")
    return 0


def step_select(args: argparse.Namespace) -> int:
    """Copy a deterministic, language-stratified selection into a small catalog."""
    source_db = Path(getattr(args, "from"))
    out = Path(args.out)
    _guard_out(out)
    if not source_db.exists():
        raise SystemExit(f"{source_db} does not exist: run the ingest step first")
    large = SqliteBookCatalogRepository(source_db, compact_json=True)
    candidates = [to_candidate(book) for book in large.iter_all(batch_size=2000)]
    selected = select_books(candidates, args.size, args.languages)

    for suffix in ("", "-wal", "-shm"):
        Path(str(out) + suffix).unlink(missing_ok=True)
    small = SqliteBookCatalogRepository(out, compact_json=True)
    ids = [c.book_id for c in selected]
    for start in range(0, len(ids), 1000):
        chunk = ids[start:start + 1000]
        books = large.get_many(chunk)
        small.save_many([books[i] for i in chunk if i in books])

    mix = language_mix(selected)
    summary = {
        "size": len(selected),
        "rule": describe_rule(args.size, args.languages),
        "quotas": language_quotas(args.size, args.languages),
        "language_mix": mix,
        "source_catalog": str(source_db),
        "database_bytes": out.stat().st_size,
    }
    # catalog.db keeps the historical name selection.json; any other file name
    # gets its own summary so that a side-by-side selection never overwrites it.
    summary_name = "selection.json" if out.stem == "catalog" else f"{out.stem}.selection.json"
    (out.parent / summary_name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Selected %d works into %s; language mix: %s", len(selected), out, mix)
    return 0


# ----------------------------------------------------------------------- cli

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["download", "ingest", "report", "select"])
    parser.add_argument("--sources", nargs="+", default=list(SOURCE_ORDER), choices=SOURCE_ORDER)
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES),
                        help="Description languages in priority order (ISO 639-1)")
    parser.add_argument("--limit", type=int, default=None, help="Max records per source (quick trials)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Catalog database to build or read")
    parser.add_argument("--from", default=str(DEFAULT_OUT), help="select: large catalog to select from")
    parser.add_argument("--size", type=int, default=5000, help="select: number of works")
    parser.add_argument("--downloads", default=str(DOWNLOADS_DIR), help="Directory of raw downloads")
    parser.add_argument("--restart", action="store_true", help="ingest: delete the catalog and rebuild it")
    parser.add_argument("--wiki-minutes", type=float, default=None,
                        help="download: stop Wikipedia API requests after this many minutes")
    parser.add_argument("--wiki-user-agent", default=None,
                        help="download: policy-compliant User-Agent with contact details; raises the "
                             f"Wikimedia limit from 10 to 200 requests/minute (default: {USER_AGENT!r})")
    parser.add_argument("--openlibrary-date", default=OPENLIBRARY_DUMP_DATE, help="Open Library dump date")
    parser.add_argument("--segments", type=int, default=8, help="Parallel byte ranges for the Open Library dumps")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args(argv)
    steps = {"download": step_download, "ingest": step_ingest, "report": step_report, "select": step_select}
    return steps[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
