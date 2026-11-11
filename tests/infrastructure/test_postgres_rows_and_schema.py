"""
Unit tests of the cloud profile's building blocks (no database):
row mapping, lexical documents, filters as SQL, vector literals, the
connection helpers and the versioned schema file.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.domain.entities import Book
from app.domain.value_objects import BookDescription, BookMetadata, SearchFilters, SourceRecord
from app.infrastructure.db.postgres_connection import PostgresDatabase, redact, settings_prefix
from app.infrastructure.db.postgres_rows import (
    book_to_work_params,
    content_hash,
    filter_condition,
    normalize,
    parse_vector,
    row_to_book,
    text_hash,
    text_rows,
    ts_config_for,
    vector_literal,
)
from app.infrastructure.db.postgres_schema import (
    SCHEMA_FILE,
    load_sections,
    render,
    validate_schema_name,
)

AVAILABLE = {"simple_unaccent", "english_unaccent", "spanish_unaccent", "french_unaccent"}


def make_book(**overrides) -> Book:
    values = dict(
        id=uuid4(),
        title="Cien años de soledad",
        authors=["Gabriel García Márquez"],
        description="Novela de la familia Buendía en Macondo.",
        language="es",
        categories=["Fiction", "Magic realism"],
        published_date=datetime(1967, 5, 30),
        source="google_books",
        source_id="abc123",
        metadata=BookMetadata(isbn="123", page_count=471),
        created_at=datetime(2025, 12, 18, 16, 45, 50, 816638, tzinfo=UTC),
        updated_at=datetime(2025, 12, 18, 16, 45, 50, 816648, tzinfo=UTC),
    )
    values.update(overrides)
    return Book(**values)


def as_db_row(book: Book) -> dict:
    """What Postgres returns for the WORK_COLUMNS of a stored book."""
    params = book_to_work_params(book)
    return {
        "id": params["id"], "source": params["source"], "source_id": params["source_id"],
        "title": params["title"], "authors": params["authors"], "description": params["description"],
        "language": params["language"], "categories": params["categories"],
        "published_date": params["published_date"],
        "metadata": params["metadata"].obj if params["metadata"] is not None else None,
        "created_at": params["created_at"], "updated_at": params["updated_at"],
    }


# --- rows ---------------------------------------------------------------------


def test_book_round_trips_through_a_works_row():
    book = make_book()
    back = row_to_book(as_db_row(book))
    for field in ("id", "title", "authors", "description", "language", "categories", "published_date",
                  "source", "source_id", "metadata", "created_at", "updated_at"):
        assert getattr(back, field) == getattr(book, field), field


def test_multisource_metadata_round_trips():
    book = make_book(metadata=BookMetadata(
        work_language="es",
        descriptions=(BookDescription("Novela de la familia Buendía en Macondo.", "es", "wikipedia", "CC BY-SA 4.0"),
                      BookDescription("Novel about the Buendía family.", "en", "wikipedia")),
        titles=(("en", "One Hundred Years of Solitude"),),
        wikidata_id="Q131156",
        provenance=(SourceRecord("wikidata", "Q131156", "CC0 1.0"),),
    ))
    assert row_to_book(as_db_row(book)).metadata == book.metadata


def test_naive_created_at_is_read_as_utc_and_aware_published_date_is_stored_naive():
    book = make_book(created_at=datetime(2025, 1, 1, 12, 0), published_date=datetime(2000, 1, 1, tzinfo=UTC))
    params = book_to_work_params(book)
    assert params["created_at"].tzinfo is not None
    assert params["published_date"].tzinfo is None and params["published_date"].year == 2000


def test_text_hash_follows_the_embedded_text_only():
    book = make_book()
    metadata_only = make_book(id=book.id, metadata=BookMetadata(isbn="999"))
    new_text = make_book(id=book.id, description="Otra descripción.")
    assert text_hash(book) == text_hash(metadata_only)
    assert content_hash(book) != content_hash(metadata_only)
    assert text_hash(book) != text_hash(new_text)


# --- lexical documents --------------------------------------------------------


def test_ts_config_for_known_unknown_and_unavailable_languages():
    assert ts_config_for("es", "cat", AVAILABLE) == "cat.spanish_unaccent"
    assert ts_config_for("EN", "cat", AVAILABLE) == "cat.english_unaccent"
    assert ts_config_for("zh", "cat", AVAILABLE) == "cat.simple_unaccent"  # no stemmer
    assert ts_config_for("de", "cat", AVAILABLE) == "cat.simple_unaccent"  # not created in this schema
    assert ts_config_for(None, "cat", AVAILABLE) == "cat.simple_unaccent"


def test_original_catalog_book_has_one_text_in_its_language():
    rows = text_rows(make_book(), "cat", AVAILABLE)
    assert [(r.language, r.ts_config) for r in rows] == [("es", "cat.spanish_unaccent")]
    assert rows[0].body.startswith("Novela")


def test_book_without_description_still_has_a_title_document():
    rows = text_rows(make_book(description=None), "cat", AVAILABLE)
    assert [(r.language, r.ts_config, r.body) for r in rows] == [("", "cat.simple_unaccent", None)]


def test_one_text_per_description_language_primary_first_duplicates_dropped():
    book = make_book(metadata=BookMetadata(descriptions=(
        BookDescription("Primera.", "es"),
        BookDescription("First.", "en"),
        BookDescription("Segunda en español.", "es"),
        BookDescription("Sin idioma: toma el de la obra.", None),
        BookDescription("中文简介", "zh"),
    )))
    rows = text_rows(book, "cat", AVAILABLE)
    assert [(r.language, r.ts_config) for r in rows] == [
        ("es", "cat.spanish_unaccent"), ("en", "cat.english_unaccent"), ("zh", "cat.simple_unaccent")]
    assert rows[0].body == "Primera."


# --- filters and vectors -------------------------------------------------------


def test_no_filters_is_true():
    assert filter_condition(None) == ("TRUE", {})
    assert filter_condition(SearchFilters()) == ("TRUE", {})


def test_filters_match_book_matches_filters_semantics():
    condition, params = filter_condition(SearchFilters(language="ES", category="fic%", min_year=1900, max_year=2000))
    assert "lower(w.language) = lower(%(f_language)s)" in condition
    assert "strpos(lower(c.name), lower(%(f_category)s)) > 0" in condition  # substring, '%' is literal
    assert "w.year >= %(f_min_year)s" in condition and "w.year <= %(f_max_year)s" in condition
    assert params == {"f_language": "ES", "f_category": "fic%", "f_min_year": 1900, "f_max_year": 2000}
    assert " AND " in condition


def test_vector_literal_round_trip_and_validation():
    literal = vector_literal([0.5, -0.25, 1e-3])
    assert literal == "[0.5,-0.25,0.001]"
    assert parse_vector(literal) == [0.5, -0.25, 0.001]
    with pytest.raises(ValueError):
        vector_literal([])
    with pytest.raises(ValueError):
        vector_literal([float("nan")])


def test_normalize_gives_unit_length():
    assert normalize([3.0, 4.0]) == [0.6, 0.8]
    with pytest.raises(ValueError):
        normalize([0.0, 0.0])


# --- connection helpers -------------------------------------------------------


def test_settings_prefix_quotes_values_and_rejects_bad_names():
    assert settings_prefix({}) == ""
    assert settings_prefix({"enable_indexscan": "off", "hnsw.ef_search": 40}) == (
        "SET LOCAL enable_indexscan = 'off'; SET LOCAL hnsw.ef_search = '40'; ")
    assert settings_prefix({"x": "a'b"}) == "SET LOCAL x = 'a''b'; "
    with pytest.raises(ValueError):
        settings_prefix({"x; DROP TABLE works": "1"})


def test_redact_hides_connection_strings_and_credentials():
    message = "could not connect to postgresql://user:secret@host-pooler.neon.tech/db?sslmode=require (password=secret)"
    cleaned = redact(message)
    assert "secret" not in cleaned and "neon.tech" not in cleaned


def test_database_needs_a_url_and_does_not_connect_when_built():
    with pytest.raises(ValueError):
        PostgresDatabase("")
    db = PostgresDatabase("postgresql://u:p@127.0.0.1:1/none", statement_timeout_ms=1500)
    assert db._script("SELECT 1", {"enable_seqscan": "off"}) == (
        "SET LOCAL statement_timeout = '1500ms'; SET LOCAL enable_seqscan = 'off'; SELECT 1")
    db.close()
    no_timeout = PostgresDatabase("postgresql://u:p@127.0.0.1:1/none", statement_timeout_ms=None)
    assert no_timeout._script("VACUUM x", {}) == "VACUUM x"


def test_connections_detect_a_dead_network_path_with_tcp_keepalives():
    db = PostgresDatabase("postgresql://u:p@127.0.0.1:1/none")
    options = db._pool.kwargs
    assert options["keepalives"] == 1
    assert options["keepalives_idle"] + options["keepalives_interval"] * options["keepalives_count"] <= 30
    assert options["tcp_user_timeout"] <= 30000


# --- schema file ---------------------------------------------------------------


def test_schema_file_has_the_expected_sections():
    assert list(load_sections()) == [
        "extensions", "schema", "lexical_index_gin", "lexical_index_bm25",
        "vector_index_lakebase_ann", "vector_index_hnsw"]


def test_every_create_in_the_schema_file_is_idempotent():
    text = SCHEMA_FILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("CREATE ") and not stripped.startswith("CREATE OR REPLACE"):
            assert "IF NOT EXISTS" in stripped, line


def test_render_fills_every_placeholder():
    sections = load_sections()
    sql = render(sections["schema"] + sections["lexical_index_bm25"], "eval177", dim=1024, k1=1.5, b=0.75)
    assert "__" not in sql.replace("__init__", "")
    assert "halfvec(1024)" in sql and "k1 = 1.5, b = 0.75" in sql and "eval177.works" in sql


@pytest.mark.parametrize("name", ["", "Eval", "a-b", "x; DROP", "1abc", "pg_temp", "public", "a" * 64])
def test_invalid_schema_names_are_rejected(name):
    with pytest.raises(ValueError):
        validate_schema_name(name)
