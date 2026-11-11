"""
Project Gutenberg bulk source (implements the BulkBooksSource port).

Reads the offline RDF catalog ``rdf-files.tar.bz2`` published at
https://www.gutenberg.org/cache/epub/feeds/ (listed on
https://www.gutenberg.org/ebooks/offline_catalogs.html). Project Gutenberg
asks not to crawl its website; the offline catalogs are the supported route
for bulk metadata, so this adapter never contacts gutenberg.org except to
download that one archive.

Each ``pgN.rdf`` file describes one e-book. The adapter keeps text e-books
and reads, with the standard library XML parser:

- title, authors (``dcterms:creator`` agents), language(s);
- subjects: Library of Congress Subject Headings and bookshelves;
- ``pgterms:downloads``: downloads in the last 30 days (popularity signal);
- ``pgterms:marc520``: the summary. Gutenberg marks its generated summaries
  with "(This is an automatically generated summary.)"; that marker is
  removed and the description is flagged ``auto_generated``.

The ``Issued`` date is when the e-text was released, not when the work was
published, so Gutenberg records carry no publication year.
"""

import logging
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from uuid import NAMESPACE_URL, uuid5

from app.domain.book_normalization import to_iso639_1
from app.domain.entities import Book
from app.domain.ports import BulkBooksSource
from app.domain.value_objects import BookDescription, BookMetadata, SourceLicense, SourceRecord

logger = logging.getLogger(__name__)

SOURCE_NAME = "gutenberg"
RDF_ARCHIVE_URL = "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"
RDF_ARCHIVE_NAME = "rdf-files.tar.bz2"

AUTO_SUMMARY_MARKER = re.compile(r"\s*\(This is an automatically generated summary\.\)\s*", re.IGNORECASE)
METADATA_LICENSE = "Project Gutenberg catalog (no explicit licence)"

_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dcterms": "http://purl.org/dc/terms/",
    "pgterms": "http://www.gutenberg.org/2009/pgterms/",
    "dcam": "http://purl.org/dc/dcam/",
}
_RDF_ABOUT = f"{{{_NS['rdf']}}}about"
_RDF_RESOURCE = f"{{{_NS['rdf']}}}resource"


def parse_gutenberg_rdf(xml_bytes: bytes) -> Optional[Dict]:
    """
    Parse one Gutenberg RDF/XML record into a plain dict.

    Returns:
        A dict with id, title, authors, languages, subjects, bookshelves,
        downloads, summary, auto_generated, type and rights; or None if the
        file holds no e-book
    """
    root = ET.fromstring(xml_bytes)
    ebook = root.find("pgterms:ebook", _NS)
    if ebook is None:
        return None
    about = ebook.get(_RDF_ABOUT, "")
    ebook_id = about.rsplit("/", 1)[-1]

    def text(path: str) -> Optional[str]:
        node = ebook.find(path, _NS)
        return node.text.strip() if node is not None and node.text else None

    def values(path: str) -> List[str]:
        return [n.text.strip() for n in ebook.findall(path, _NS) if n.text and n.text.strip()]

    subjects = []
    for description in ebook.findall("dcterms:subject/rdf:Description", _NS):
        member = description.find("dcam:memberOf", _NS)
        value = description.find("rdf:value", _NS)
        if member is not None and value is not None and value.text and member.get(_RDF_RESOURCE, "").endswith("LCSH"):
            subjects.append(value.text.strip())

    summary = text("pgterms:marc520")
    auto_generated = False
    if summary and AUTO_SUMMARY_MARKER.search(summary):
        auto_generated = True
        summary = AUTO_SUMMARY_MARKER.sub(" ", summary).strip()

    downloads = text("pgterms:downloads")
    return {
        "id": ebook_id,
        "title": text("dcterms:title"),
        "authors": values("dcterms:creator/pgterms:agent/pgterms:name"),
        "languages": values("dcterms:language/rdf:Description/rdf:value"),
        "subjects": subjects,
        "bookshelves": values("pgterms:bookshelf/rdf:Description/rdf:value"),
        "downloads": int(downloads) if downloads and downloads.isdigit() else None,
        "summary": summary,
        "auto_generated": auto_generated,
        "type": text("dcterms:type/rdf:Description/rdf:value"),
        "rights": text("dcterms:rights"),
    }


def _clean_title(title: str) -> str:
    """Gutenberg separates title and subtitle with a line break: join them with ': '."""
    lines = [line.strip() for line in title.replace("\r", "\n").split("\n") if line.strip()]
    if not lines:
        return ""
    head = lines[0].rstrip(":;,")
    return f"{head}: {' '.join(lines[1:])}" if len(lines) > 1 else head


class GutenbergSource(BulkBooksSource):
    """BulkBooksSource over Project Gutenberg's offline RDF catalog."""

    def __init__(self, download_dir: Path) -> None:
        """
        Args:
            download_dir: Directory containing rdf-files.tar.bz2
        """
        self._dir = Path(download_dir)
        self._rejected: Dict[str, int] = {}

    def get_source_name(self) -> str:
        """Source identifier."""
        return SOURCE_NAME

    def get_license(self) -> SourceLicense:
        """Gutenberg catalog metadata is offered free of restrictions."""
        return SourceLicense(
            source=SOURCE_NAME,
            name="Project Gutenberg",
            license="No explicit licence for the catalog metadata ('freely available')",
            terms_url="https://www.gutenberg.org/ebooks/offline_catalogs.html",
            attribution="Project Gutenberg, https://www.gutenberg.org/ebooks/<number>",
            notes=(
                "The offline catalogs page asks to use these files instead of crawling the "
                "website and calls the MARC records 'freely available'; it states no licence. "
                "Summaries are generated automatically by Project Gutenberg and flagged as such."
            ),
        )

    def get_rejected_counts(self) -> Dict[str, int]:
        """Records skipped by iter_books(), by reason."""
        return dict(self._rejected)

    def archive_path(self) -> Path:
        """Location of the RDF archive."""
        return self._dir / RDF_ARCHIVE_NAME

    def iter_books(self, limit: Optional[int] = None) -> Iterator[Book]:
        """
        Stream the RDF archive and yield one Book per text e-book.

        Raises:
            FileNotFoundError: If the archive has not been downloaded
        """
        self._rejected = {}
        path = self.archive_path()
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}: run the download step for gutenberg first")
        produced = 0
        with tarfile.open(path, "r|bz2") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".rdf"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                try:
                    record = parse_gutenberg_rdf(handle.read())
                except ET.ParseError:
                    self._reject("unparseable")
                    continue
                if record is None:
                    continue
                book = self.record_to_book(record)
                if book is None:
                    continue
                yield book
                produced += 1
                if limit is not None and produced >= limit:
                    return

    def record_to_book(self, record: Dict) -> Optional[Book]:
        """Turn a parsed RDF record into a raw Book (None, counted, if unusable)."""
        if record.get("type") and record["type"] != "Text":
            self._reject("not_text")
            return None
        title = _clean_title(record.get("title") or "")
        if not title:
            self._reject("no_title")
            return None
        authors = [a for a in record.get("authors", []) if a]
        if not authors:
            self._reject("no_author")
            return None

        ebook_id = record["id"]
        url = f"https://www.gutenberg.org/ebooks/{ebook_id}"
        languages = [code for code in (to_iso639_1(v) for v in record.get("languages", [])) if code]
        descriptions = ()
        if record.get("summary"):
            descriptions = (
                BookDescription(
                    text=record["summary"],
                    language=None,  # detected by the ingestion pipeline
                    source=SOURCE_NAME,
                    license=METADATA_LICENSE,
                    attribution_url=url,
                    auto_generated=bool(record.get("auto_generated")),
                ),
            )
        metadata = BookMetadata(
            work_language=languages[0] if len(languages) == 1 else None,
            descriptions=descriptions,
            gutenberg_ids=(ebook_id,),
            gutenberg_downloads=record.get("downloads"),
            preview_link=url,
            provenance=(SourceRecord(SOURCE_NAME, ebook_id, METADATA_LICENSE, url),),
        )
        return Book(
            id=uuid5(NAMESPACE_URL, f"{SOURCE_NAME}:{ebook_id}"),
            title=title,
            authors=authors,
            language=languages[0] if len(languages) == 1 else None,
            categories=[*record.get("subjects", []), *record.get("bookshelves", [])],
            source=SOURCE_NAME,
            source_id=ebook_id,
            metadata=metadata,
        )

    def _reject(self, reason: str) -> None:
        self._rejected[reason] = self._rejected.get(reason, 0) + 1
