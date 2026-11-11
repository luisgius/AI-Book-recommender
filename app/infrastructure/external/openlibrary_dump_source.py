"""
Open Library monthly dump source (implements the BulkBooksSource port).

Open Library publishes monthly dumps at https://openlibrary.org/developers/dumps
(a mirror on archive.org). Its API terms ask not to bulk-download through
the API; the dumps are the supported route. Each dump is a gzip file with
one record per line: ``type<TAB>key<TAB>revision<TAB>last_modified<TAB>json``.

The works dump has about 41 million works, but only about 2.5% carry a
description of 200 characters or more, so the adapter works in three
streaming passes and never holds the dump in memory:

1. Works dump: keep works whose description (string or ``{"value": ...}``)
   reaches ``min_description_chars``; write them, reduced to the fields used,
   to a gzip JSONL file of candidates and collect their author keys.
2. Authors dump (if present): read the names of those authors only.
3. Candidates file: yield one Book per candidate with resolved author names.

Works records have no language (languages belong to editions), so the
description language is detected by the ingestion pipeline. Candidates are
cached next to the dump (``works_candidates_<min>.jsonl.gz``) and reused by
later runs.
"""

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from uuid import NAMESPACE_URL, uuid5

from app.domain.book_normalization import parse_year, strip_markup
from app.domain.entities import Book
from app.domain.ports import BulkBooksSource
from app.domain.value_objects import BookDescription, BookMetadata, SourceLicense, SourceRecord

logger = logging.getLogger(__name__)

SOURCE_NAME = "open_library"
DUMP_BASE_URL = "https://openlibrary.org/data/"
METADATA_LICENSE = "Open Library dump (no explicit licence)"
COVER_URL = "https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"


def description_text(value: object) -> Optional[str]:
    """Open Library stores text fields as a string or as {"type": "/type/text", "value": ...}."""
    if isinstance(value, dict):
        value = value.get("value")
    return value if isinstance(value, str) and value.strip() else None


def parse_dump_line(line: str) -> Optional[Dict]:
    """Return the JSON record of one dump line (None if malformed)."""
    parts = line.rstrip("\n").split("\t", 4)
    if len(parts) != 5:
        return None
    try:
        return json.loads(parts[4])
    except ValueError:
        return None


def work_to_candidate(work: Dict, min_description_chars: int) -> Optional[Dict]:
    """
    Reduce a work record to the fields used, if its description is long enough.

    The length is measured after markup is stripped, like the quality filter.
    """
    text = description_text(work.get("description"))
    if not text or len(strip_markup(text)) < min_description_chars:
        return None
    authors = []
    for entry in work.get("authors", []) or []:
        author = entry.get("author") if isinstance(entry, dict) else None
        key = author.get("key") if isinstance(author, dict) else (author if isinstance(author, str) else None)
        if key and key not in authors:
            authors.append(key)
    covers = [c for c in work.get("covers", []) or [] if isinstance(c, int) and c > 0]
    return {
        "key": work.get("key", ""),
        "title": work.get("title") or "",
        "subtitle": work.get("subtitle") or "",
        "description": text,
        "authors": authors,
        "subjects": [s for s in work.get("subjects", []) or [] if isinstance(s, str)],
        "first_publish_date": work.get("first_publish_date"),
        "cover": covers[0] if covers else None,
    }


class OpenLibraryDumpSource(BulkBooksSource):
    """BulkBooksSource over the Open Library works and authors dumps."""

    def __init__(self, download_dir: Path, min_description_chars: int = 200) -> None:
        """
        Args:
            download_dir: Directory with ol_dump_works_*.txt.gz (and authors)
            min_description_chars: Works with a shorter description are not
                kept as candidates (same threshold as the quality filter)
        """
        self._dir = Path(download_dir)
        self._min_chars = min_description_chars
        self._rejected: Dict[str, int] = {}

    def get_source_name(self) -> str:
        """Source identifier."""
        return SOURCE_NAME

    def get_license(self) -> SourceLicense:
        """Open Library states no licence for its data and asks for dumps instead of API crawling."""
        return SourceLicense(
            source=SOURCE_NAME,
            name="Open Library (Internet Archive)",
            license="No explicit licence; Open Library 'does not assert any new copyright'",
            terms_url="https://openlibrary.org/developers/dumps ; https://openlibrary.org/developers/licensing",
            attribution="Open Library, https://openlibrary.org/works/<id>",
            notes="Bulk use must go through the monthly dumps, not the API.",
        )

    def get_rejected_counts(self) -> Dict[str, int]:
        """Works skipped by iter_books(), by reason."""
        return dict(self._rejected)

    def _latest(self, kind: str) -> Optional[Path]:
        files = sorted(self._dir.glob(f"ol_dump_{kind}_*.txt.gz"))
        return files[-1] if files else None

    def works_dump(self) -> Path:
        """Latest complete works dump in the download directory."""
        path = self._latest("works")
        if path is None:
            raise FileNotFoundError(f"No ol_dump_works_*.txt.gz in {self._dir}: run the download step")
        return path

    def iter_books(self, limit: Optional[int] = None) -> Iterator[Book]:
        """
        Yield one Book per work with a long enough description and an author.

        Raises:
            FileNotFoundError: If the works dump has not been downloaded
        """
        self._rejected = {}
        candidates_path = self._candidates_path()
        if not candidates_path.exists():
            self._extract_candidates(candidates_path)
        else:
            self._load_pass_one_counts(candidates_path)

        author_keys: Set[str] = set()
        for candidate in self._read_candidates(candidates_path):
            author_keys.update(candidate["authors"])
        names = self._author_names(author_keys)

        produced = 0
        for candidate in self._read_candidates(candidates_path):
            book = self.candidate_to_book(candidate, names)
            if book is None:
                continue
            yield book
            produced += 1
            if limit is not None and produced >= limit:
                return

    def candidate_to_book(self, candidate: Dict, names: Dict[str, str]) -> Optional[Book]:
        """Build the raw Book of a candidate (None, counted, if no title or author)."""
        title = candidate["title"].strip()
        if candidate.get("subtitle"):
            title = f"{title}: {candidate['subtitle'].strip()}"
        if not title:
            self._reject("no_title")
            return None
        authors = [names[k] for k in candidate["authors"] if names.get(k)]
        if not authors:
            self._reject("no_author")
            return None
        olid = candidate["key"].rsplit("/", 1)[-1]
        url = f"https://openlibrary.org/works/{olid}"
        year = parse_year(candidate.get("first_publish_date"))
        metadata = BookMetadata(
            descriptions=(
                BookDescription(
                    text=candidate["description"],
                    language=None,  # detected by the ingestion pipeline
                    source=SOURCE_NAME,
                    license=METADATA_LICENSE,
                    attribution_url=url,
                ),
            ),
            openlibrary_ids=(olid,),
            thumbnail_url=COVER_URL.format(cover_id=candidate["cover"]) if candidate.get("cover") else None,
            preview_link=url,
            provenance=(SourceRecord(SOURCE_NAME, olid, METADATA_LICENSE, url),),
        )
        return Book(
            id=uuid5(NAMESPACE_URL, f"{SOURCE_NAME}:{olid}"),
            title=title,
            authors=authors,
            categories=list(candidate.get("subjects", [])),
            published_date=datetime(year, 1, 1) if year and 1 <= year <= 9999 else None,
            source=SOURCE_NAME,
            source_id=olid,
            metadata=metadata,
        )

    # ---------------------------------------------------------------- passes

    def _candidates_path(self) -> Path:
        return self._dir / f"works_candidates_{self._min_chars}.jsonl.gz"

    def _extract_candidates(self, target: Path) -> None:
        """Pass 1: stream the works dump and keep works with long descriptions."""
        dump = self.works_dump()
        tmp = target.with_suffix(".tmp")
        read = kept = 0
        with gzip.open(dump, "rt", encoding="utf-8", errors="replace") as source, \
                gzip.open(tmp, "wt", encoding="utf-8") as out:
            for line in source:
                read += 1
                # Cheap pre-filter: most works have no description at all.
                if '"description"' not in line:
                    continue
                work = parse_dump_line(line)
                candidate = work_to_candidate(work, self._min_chars) if work else None
                if candidate is not None:
                    kept += 1
                    out.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                if read % 5_000_000 == 0:
                    logger.info("Open Library works: %d read, %d candidates", read, kept)
            out.write(json.dumps({"_summary": {"read": read, "kept": kept}}) + "\n")
        tmp.replace(target)
        self._rejected["no_or_short_description"] = read - kept
        logger.info("Open Library works: %d read, %d candidates", read, kept)

    def _load_pass_one_counts(self, path: Path) -> None:
        """Restore the pass-1 rejection count stored at the end of the candidates file."""
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            last = None
            for line in handle:
                last = line
        if last:
            summary = json.loads(last).get("_summary")
            if summary:
                self._rejected["no_or_short_description"] = summary["read"] - summary["kept"]

    @staticmethod
    def _read_candidates(path: Path) -> Iterator[Dict]:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "_summary" not in record:
                    yield record

    def _author_names(self, keys: Set[str]) -> Dict[str, str]:
        """Pass 2: names of the given author keys from the authors dump (if downloaded)."""
        dump = self._latest("authors")
        if dump is None:
            logger.warning("No authors dump in %s: works will have no author names", self._dir)
            return {}
        names: Dict[str, str] = {}
        with gzip.open(dump, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split("\t", 2)
                if len(parts) < 2 or parts[1] not in keys:
                    continue
                record = parse_dump_line(line)
                name = (record or {}).get("name") or (record or {}).get("personal_name")
                if isinstance(name, str) and name.strip():
                    names[parts[1]] = name.strip()
        logger.info("Open Library authors resolved: %d of %d", len(names), len(keys))
        return names

    def _reject(self, reason: str) -> None:
        self._rejected[reason] = self._rejected.get(reason, 0) + 1


def dump_file_names(date: str) -> List[str]:
    """File names of the works and authors dumps of a given date (YYYY-MM-DD)."""
    return [f"ol_dump_authors_{date}.txt.gz", f"ol_dump_works_{date}.txt.gz"]
