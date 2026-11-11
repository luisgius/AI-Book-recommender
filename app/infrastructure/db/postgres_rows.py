"""
Mapping between domain Books and the rows of the cloud-profile schema, and
the SQL pieces the three Postgres adapters share.

- works: one row per Book. Scalar fields are columns; BookMetadata is stored
  as JSONB in exactly the format of the SQLite catalog (the same
  serialisation functions are reused), so both profiles hold the same data.
- work_texts: one row per description language (see text_rows()).
- filter_condition(): SearchFilters as SQL, with the semantics of
  Book.matches_filters (the single definition of filter matching).
- vector_literal(): an embedding as a halfvec literal. Vectors travel as
  text and are cast on the server, so no client-side type registration (one
  extra round trip per new connection) is needed.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from psycopg.types.json import Jsonb

from app.domain.entities import Book
from app.domain.value_objects import SearchFilters
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

# The metadata JSON format of the catalog is defined once, in the SQLite
# adapter (pure static functions); the Postgres profile reuses it so a work
# reads back identically from both catalogs.
metadata_to_dict = SqliteBookCatalogRepository._metadata_to_dict
metadata_from_dict = SqliteBookCatalogRepository._metadata_from_dict

# Columns of works that hydrate a Book, with the table alias "w".
WORK_COLUMNS = (
    "w.id, w.source, w.source_id, w.title, w.authors, w.description, w.language, "
    "w.categories, w.published_date, w.metadata, w.created_at, w.updated_at"
)

# Columns written by an upsert, in order.
WORK_INSERT_COLUMNS = (
    "id", "source", "source_id", "title", "authors", "description", "language",
    "categories", "published_date", "metadata", "created_at", "updated_at",
    "content_hash", "text_hash",
)

SIMPLE_CONFIG = "simple_unaccent"

# ISO 639-1 code -> name of the Snowball configuration Postgres ships for it.
# Only languages whose configuration exists in the schema are used (the
# schema builds them from pg_ts_config); others fall back to simple_unaccent.
LANGUAGE_CONFIGS: Dict[str, str] = {
    "ar": "arabic", "hy": "armenian", "eu": "basque", "ca": "catalan", "da": "danish",
    "nl": "dutch", "en": "english", "et": "estonian", "fi": "finnish", "fr": "french",
    "de": "german", "el": "greek", "hi": "hindi", "hu": "hungarian", "id": "indonesian",
    "ga": "irish", "it": "italian", "lt": "lithuanian", "ne": "nepali", "no": "norwegian",
    "nb": "norwegian", "nn": "norwegian", "pt": "portuguese", "ro": "romanian",
    "ru": "russian", "sr": "serbian", "es": "spanish", "sv": "swedish", "ta": "tamil",
    "tr": "turkish", "yi": "yiddish",
}


# =============================================================================
# Books <-> works rows
# =============================================================================


def _naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    """published_date is a timestamp without time zone: aware values go to UTC."""
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _aware_utc(value: datetime) -> datetime:
    """created_at / updated_at: naive values are taken as UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def text_hash(book: Book) -> str:
    """Hash of the text the document embedding is computed from."""
    return hashlib.sha256(book.get_searchable_text().encode("utf-8")).hexdigest()


def content_hash(book: Book) -> str:
    """Hash of everything stored for a work (detects changes on reload)."""
    payload = {
        "source": book.source,
        "source_id": book.source_id,
        "title": book.title,
        "authors": book.authors,
        "description": book.description,
        "language": book.language,
        "categories": book.categories,
        "published_date": book.published_date.isoformat() if book.published_date else None,
        "metadata": metadata_to_dict(book),
        "created_at": book.created_at.isoformat(),
        "updated_at": book.updated_at.isoformat(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def book_to_work_params(book: Book) -> Dict[str, Any]:
    """Parameters of a works upsert (keys are WORK_INSERT_COLUMNS)."""
    metadata = metadata_to_dict(book)
    return {
        "id": book.id,
        "source": book.source,
        "source_id": book.source_id,
        "title": book.title,
        "authors": list(book.authors),
        "description": book.description,
        "language": book.language,
        "categories": list(book.categories or []),
        "published_date": _naive_utc(book.published_date),
        "metadata": Jsonb(metadata) if metadata is not None else None,
        "created_at": _aware_utc(book.created_at),
        "updated_at": _aware_utc(book.updated_at),
        "content_hash": content_hash(book),
        "text_hash": text_hash(book),
    }


def row_to_book(row: Mapping[str, Any]) -> Book:
    """Hydrate a Book from a row holding the WORK_COLUMNS."""
    metadata_raw = row["metadata"]
    if isinstance(metadata_raw, str):  # defensive: jsonb normally arrives as a dict
        metadata_raw = json.loads(metadata_raw)
    return Book(
        id=row["id"],
        title=row["title"],
        authors=list(row["authors"]),
        description=row["description"],
        language=row["language"],
        categories=list(row["categories"] or []),
        published_date=row["published_date"],
        source=row["source"],
        source_id=row["source_id"],
        metadata=metadata_from_dict(metadata_raw, row["description"]) if metadata_raw is not None else None,
        created_at=_aware_utc(row["created_at"]),
        updated_at=_aware_utc(row["updated_at"]),
    )


# =============================================================================
# Text rows (lexical documents)
# =============================================================================


@dataclass(frozen=True)
class TextRow:
    """One work_texts row: a description in one language and its configuration."""

    language: str  # ISO 639-1 code, '' when unknown or when the work has no description
    ts_config: str  # schema-qualified text-search configuration
    body: Optional[str]


def ts_config_for(language: Optional[str], schema: str, available: Set[str]) -> str:
    """
    Text-search configuration of a description language.

    Args:
        language: ISO 639-1 code (None or '' when unknown)
        schema: Catalog schema holding the configurations
        available: Configuration names that exist in the schema

    Returns:
        "<schema>.<language>_unaccent" when Postgres has a stemmer for the
        language, "<schema>.simple_unaccent" otherwise
    """
    name = LANGUAGE_CONFIGS.get((language or "").lower())
    if name and f"{name}_unaccent" in available:
        return f"{schema}.{name}_unaccent"
    return f"{schema}.{SIMPLE_CONFIG}"


def text_rows(book: Book, schema: str, available: Set[str]) -> List[TextRow]:
    """
    The lexical documents of a work: one per description language.

    Descriptions come from metadata.descriptions (multi-source catalog) or,
    for the original catalog, from Book.description in Book.language. A
    description without a language takes the work's language. Two
    descriptions in the same language keep the first (the primary). A work
    without any description still gets one row, so its title and authors
    are searchable.
    """
    meta = book.metadata
    if meta is not None and meta.descriptions:
        found = [(d.language or book.language, d.text) for d in meta.descriptions]
    elif book.has_description():
        found = [(book.language, book.description)]
    else:
        found = []

    rows: List[TextRow] = []
    seen: Set[str] = set()
    for language, body in found:
        key = (language or "").lower()
        if key in seen or not body or not body.strip():
            continue
        seen.add(key)
        rows.append(TextRow(key, ts_config_for(key, schema, available), body))
    if not rows:
        rows.append(TextRow("", f"{schema}.{SIMPLE_CONFIG}", None))
    return rows


def lexical_header_sql(alias: str = "w") -> str:
    """Text indexed unstemmed in every lexical document: title, authors, categories."""
    return (
        f"coalesce({alias}.title, '') || ' ' || array_to_string({alias}.authors, ' ') || ' ' || "
        f"array_to_string({alias}.categories, ' ')"
    )


# =============================================================================
# Filters and vectors
# =============================================================================


def filter_condition(filters: Optional[SearchFilters], alias: str = "w") -> Tuple[str, Dict[str, Any]]:
    """
    SearchFilters as a SQL condition on works, with Book.matches_filters semantics.

    - language: exact ISO code, case-insensitive
    - category: case-insensitive substring of any category (strpos, so '%'
      and '_' in the filter are plain characters)
    - min_year / max_year: inclusive; a work without a year never matches

    Returns:
        (condition, parameters); condition is "TRUE" without filters
    """
    if filters is None or filters.is_empty():
        return "TRUE", {}
    conditions: List[str] = []
    params: Dict[str, Any] = {}
    if filters.language is not None:
        conditions.append(f"lower({alias}.language) = lower(%(f_language)s)")
        params["f_language"] = filters.language
    if filters.category is not None:
        conditions.append(
            f"EXISTS (SELECT 1 FROM unnest({alias}.categories) AS c(name) "
            f"WHERE strpos(lower(c.name), lower(%(f_category)s)) > 0)"
        )
        params["f_category"] = filters.category
    if filters.min_year is not None:
        conditions.append(f"{alias}.year >= %(f_min_year)s")
        params["f_min_year"] = filters.min_year
    if filters.max_year is not None:
        conditions.append(f"{alias}.year <= %(f_max_year)s")
        params["f_max_year"] = filters.max_year
    return " AND ".join(conditions), params


def vector_literal(vector: Sequence[float]) -> str:
    """
    An embedding as a pgvector literal ('[0.1,0.2,...]'), cast to halfvec in SQL.

    Raises:
        ValueError: If the vector is empty or has a non-finite value
    """
    values = [float(x) for x in vector]
    if not values:
        raise ValueError("Empty embedding")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Embedding has a non-finite value")
    return "[" + ",".join(format(x, ".7g") for x in values) + "]"


def parse_vector(text: str) -> List[float]:
    """Parse a pgvector text value ('[0.1,0.2]') into floats."""
    inner = text.strip()[1:-1].strip()
    return [float(x) for x in inner.split(",")] if inner else []


def normalize(vector: Sequence[float]) -> List[float]:
    """Unit-length copy of a vector (cosine distance does not need it, halfvec range does)."""
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm == 0.0:
        raise ValueError("Cannot normalise a zero vector")
    return [float(x) / norm for x in vector]
