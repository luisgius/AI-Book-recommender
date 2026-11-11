"""
Directory of Open Access Books (DOAB) bulk source (implements BulkBooksSource).

Reads the full DOAB metadata export in CSV format
(https://directory.doabooks.org/download-export?format=csv). DOAB states on
https://doabooks.org/en/librarians/doab-metadata that "all metadata is
openly available under a CC0 license"; the licences of the books themselves
vary per title (most are Creative Commons) and are kept per record.

Columns used (multi-valued cells are separated by "||"):

- ``dc.title``, ``dc.contributor.author`` (editors when there is no author),
  ``dc.date.issued``, ``dc.language`` ("English[eng]"),
  ``dc.description.abstract``;
- ``dc.subject.classification`` (Thema/BIC paths, the last level is kept)
  and ``dc.subject.other`` (free keywords, only multi-word or capitalised
  ones);
- ``oapen.relation.isbn`` and ``BITSTREAM ISBN``, ``oapen.identifier.doi``,
  ``oapen.relation.isPublishedBy_publisher.name``, ``oapen.pages``,
  ``BITSTREAM License``, ``dc.identifier.uri`` (the DOAB record page).

Only records of type "book" are read (chapters are skipped). The abstract's
language is left for the ingestion pipeline to detect, because DOAB records
the language of the book, and abstracts are often written in English.
"""

import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from uuid import NAMESPACE_URL, uuid5

from app.domain.book_normalization import clean_isbn13, parse_year, to_iso639_1
from app.domain.entities import Book
from app.domain.ports import BulkBooksSource
from app.domain.value_objects import BookDescription, BookMetadata, SourceLicense, SourceRecord

logger = logging.getLogger(__name__)

SOURCE_NAME = "doab"
EXPORT_URL = "https://directory.doabooks.org/download-export?format=csv"
EXPORT_NAME = "doab_export.csv"
METADATA_LICENSE = "CC0 1.0"

_LANGUAGE_CELL = re.compile(r"\[([a-z]{2,3})\]")
_THEMA_CODE = re.compile(r"^[A-Z0-9]{1,8}\s+")


def split_cell(value: Optional[str]) -> List[str]:
    """Split a multi-valued DOAB cell ("a||b") into stripped non-empty values."""
    if not value:
        return []
    return [part.strip() for part in value.split("||") if part.strip()]


def parse_languages(value: Optional[str]) -> List[str]:
    """'Italian[ita]||English[eng]' -> ['it', 'en'] (unknown codes dropped)."""
    codes = []
    for part in split_cell(value):
        match = _LANGUAGE_CELL.search(part)
        code = to_iso639_1(match.group(1) if match else part)
        if code and code not in codes:
            codes.append(code)
    return codes


def classification_labels(value: Optional[str]) -> List[str]:
    """
    Last level of each Thema/BIC path, without its code.

    'thema EDItEUR::J Society and Social Sciences::JN Education::JNM Higher
    education' -> 'Higher education'
    """
    labels = []
    for path in split_cell(value):
        last = path.split("::")[-1].strip()
        label = _THEMA_CODE.sub("", last).strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def useful_keywords(value: Optional[str]) -> List[str]:
    """Free keywords worth keeping: multi-word phrases or capitalised terms."""
    return [k for k in split_cell(value) if " " in k or k[:1].isupper()]


class DoabSource(BulkBooksSource):
    """BulkBooksSource over the DOAB CSV export."""

    def __init__(self, download_dir: Path) -> None:
        """
        Args:
            download_dir: Directory containing doab_export.csv
        """
        self._dir = Path(download_dir)
        self._rejected: Dict[str, int] = {}

    def get_source_name(self) -> str:
        """Source identifier."""
        return SOURCE_NAME

    def get_license(self) -> SourceLicense:
        """DOAB metadata is CC0 1.0; book licences vary per record."""
        return SourceLicense(
            source=SOURCE_NAME,
            name="Directory of Open Access Books",
            license="CC0 1.0 (metadata)",
            terms_url="https://doabooks.org/en/librarians/doab-metadata",
            attribution="Not required (CC0); each record links to its DOAB page",
            notes="Book contents have their own licences (mostly Creative Commons), stored per record.",
        )

    def get_rejected_counts(self) -> Dict[str, int]:
        """Records skipped by iter_books(), by reason."""
        return dict(self._rejected)

    def export_path(self) -> Path:
        """Location of the CSV export."""
        return self._dir / EXPORT_NAME

    def iter_books(self, limit: Optional[int] = None) -> Iterator[Book]:
        """
        Stream the CSV export and yield one Book per DOAB book record.

        Raises:
            FileNotFoundError: If the export has not been downloaded
        """
        self._rejected = {}
        path = self.export_path()
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}: run the download step for doab first")
        csv.field_size_limit(1 << 26)
        produced = 0
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                book = self.row_to_book(row)
                if book is None:
                    continue
                yield book
                produced += 1
                if limit is not None and produced >= limit:
                    return

    def row_to_book(self, row: Dict[str, str]) -> Optional[Book]:
        """Turn one CSV row into a raw Book (None, counted, if unusable)."""
        if (row.get("dc.type") or "book").strip().lower() != "book":
            self._reject("not_a_book")
            return None
        title = (row.get("dc.title") or "").strip()
        if not title:
            self._reject("no_title")
            return None
        authors = split_cell(row.get("dc.contributor.author")) or split_cell(row.get("dc.contributor.editor"))
        if not authors:
            self._reject("no_author")
            return None

        record_id = (row.get("id") or "").strip()
        url = (row.get("dc.identifier.uri") or "").strip() or None
        languages = parse_languages(row.get("dc.language"))
        isbns = []
        for value in split_cell(row.get("oapen.relation.isbn")) + split_cell(row.get("BITSTREAM ISBN")):
            isbn = clean_isbn13(value)
            if isbn and isbn not in isbns:
                isbns.append(isbn)
        book_license = (split_cell(row.get("BITSTREAM License")) or [None])[0]
        abstract = (row.get("dc.description.abstract") or "").strip()
        descriptions = ()
        if abstract:
            descriptions = (
                BookDescription(
                    text=abstract,
                    language=None,  # detected by the ingestion pipeline
                    source=SOURCE_NAME,
                    license=METADATA_LICENSE,
                    attribution_url=url,
                ),
            )
        pages = (row.get("oapen.pages") or "").strip()
        doi = (row.get("oapen.identifier.doi") or "").strip().lower() or None
        year = parse_year(row.get("dc.date.issued"))
        metadata = BookMetadata(
            isbn13=isbns[0] if isbns else None,
            isbn13s=tuple(isbns),
            publisher=(row.get("oapen.relation.isPublishedBy_publisher.name") or "").strip() or None,
            page_count=int(pages) if pages.isdigit() else None,
            preview_link=url,
            work_language=languages[0] if len(languages) == 1 else None,
            descriptions=descriptions,
            doi=doi,
            provenance=(SourceRecord(SOURCE_NAME, record_id, f"metadata {METADATA_LICENSE}; book {book_license}", url),),
        )
        categories = classification_labels(row.get("dc.subject.classification")) + useful_keywords(
            row.get("dc.subject.other")
        )
        return Book(
            id=uuid5(NAMESPACE_URL, f"{SOURCE_NAME}:{record_id}"),
            title=title,
            authors=authors,
            language=languages[0] if len(languages) == 1 else None,
            categories=categories,
            published_date=datetime(year, 1, 1) if year and 1 <= year <= 9999 else None,
            source=SOURCE_NAME,
            source_id=record_id,
            metadata=metadata,
        )

    def _reject(self, reason: str) -> None:
        self._rejected[reason] = self._rejected.get(reason, 0) + 1
