"""
Idempotent migration of the cloud-profile schema (postgres_schema.sql).

The SQL file is the versioned definition; this module only fills its
placeholders and runs its sections:

- apply_schema(): extensions, the catalog schema, its tables, the
  text-search configurations and the query functions. Safe to run again.
- build_lexical_indexes() / build_vector_index(): the indexes, built after
  the bulk load (BM25 corpus statistics are taken at build time).
- catalog_info read/write, relation sizes and server versions for reports.

Every catalog lives in its own schema (DATABASE_SCHEMA), so the 177-book
evaluation collection and a larger catalog can share one database.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from app.infrastructure.config.settings import SCHEMA_NAME_PATTERN
from app.infrastructure.db.postgres_connection import PostgresDatabase

SCHEMA_VERSION = 1
SCHEMA_FILE = Path(__file__).with_name("postgres_schema.sql")

# BM25 parameters of the local implementation (rank_bm25.BM25Okapi defaults),
# so both profiles score with the same k1 and b. lakebase_bm25 accepts
# k1 in [1.2, 2.0] and b in [0.0, 1.0].
BM25_K1 = 1.5
BM25_B = 0.75

VECTOR_INDEX_TYPES = ("lakebase_ann", "hnsw")
BM25_INDEX = "work_texts_bm25"
GIN_INDEX = "work_texts_tsv_gin"
VECTOR_INDEX = "works_embedding_ann"

_SECTION_MARKER = re.compile(r"^-- @section (\w+)\s*$", re.MULTILINE)


def validate_schema_name(schema: str) -> str:
    """
    Return the schema name if it is a plain lower-case identifier.

    Raises:
        ValueError: Otherwise (it is written into SQL unquoted)
    """
    if not isinstance(schema, str) or not re.match(SCHEMA_NAME_PATTERN, schema):
        raise ValueError(f"Invalid schema name {schema!r}: use lower-case letters, digits and _")
    if schema.startswith("pg_") or schema in ("public", "information_schema"):
        raise ValueError(f"Schema name {schema!r} is reserved")
    return schema


def load_sections(path: Path = SCHEMA_FILE) -> Dict[str, str]:
    """Sections of the schema file, by name, in file order."""
    text = path.read_text(encoding="utf-8")
    markers = list(_SECTION_MARKER.finditer(text))
    sections: Dict[str, str] = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        sections[marker.group(1)] = text[marker.end():end].strip()
    return sections


def render(sql: str, schema: str, *, dim: int = 0, k1: float = BM25_K1, b: float = BM25_B) -> str:
    """Fill the placeholders of a section."""
    validate_schema_name(schema)
    return (
        sql.replace("__SCHEMA__", schema)
        .replace("__DIM__", str(int(dim)))
        .replace("__BM25_K1__", repr(float(k1)))
        .replace("__BM25_B__", repr(float(b)))
    )


def _section(name: str, schema: str, **values: Any) -> str:
    return render(load_sections()[name], schema, **values)


# =============================================================================
# Migration
# =============================================================================


def schema_exists(db: PostgresDatabase, schema: str) -> bool:
    rows = db.query("SELECT 1 AS found FROM pg_namespace WHERE nspname = %(s)s", {"s": validate_schema_name(schema)})
    return bool(rows)


def embedding_dimension_in_db(db: PostgresDatabase, schema: str) -> Optional[int]:
    """Dimension of works.embedding if the table exists (halfvec(n) -> n)."""
    rows = db.query(
        """SELECT format_type(a.atttypid, a.atttypmod) AS type
           FROM pg_attribute a
           WHERE a.attrelid = to_regclass(%(table)s) AND a.attname = 'embedding' AND NOT a.attisdropped""",
        {"table": f"{validate_schema_name(schema)}.works"},
    )
    if not rows:
        return None
    match = re.search(r"\((\d+)\)", rows[0]["type"])
    return int(match.group(1)) if match else None


def apply_schema(db: PostgresDatabase, schema: str, dim: int) -> None:
    """
    Create (or keep) the extensions, the schema, its tables, configurations
    and functions. Idempotent.

    Raises:
        ValueError: If the schema already holds embeddings of another dimension
    """
    if dim < 1:
        raise ValueError(f"Embedding dimension must be >= 1, got {dim}")
    existing = embedding_dimension_in_db(db, schema)
    if existing is not None and existing != dim:
        raise ValueError(
            f"Schema '{schema}' stores {existing}-dimensional embeddings but the embedding model gives {dim}. "
            f"Load into another schema or drop this one (python -m scripts.load_postgres --schema {schema} --drop)."
        )
    db.execute(_section("extensions", schema))
    db.execute(_section("schema", schema, dim=dim))
    write_catalog_info(db, schema, {"schema_version": SCHEMA_VERSION, "embedding_dimension": dim})


def drop_schema(db: PostgresDatabase, schema: str) -> None:
    """Drop a catalog schema and everything in it (extensions are kept)."""
    db.execute(f"DROP SCHEMA IF EXISTS {validate_schema_name(schema)} CASCADE")


def installed_extensions(db: PostgresDatabase) -> Dict[str, str]:
    """Installed extension -> version."""
    return {row["extname"]: row["extversion"] for row in db.query("SELECT extname, extversion FROM pg_extension")}


def text_config_names(db: PostgresDatabase, schema: str) -> Set[str]:
    """Names of the text-search configurations created in the schema."""
    rows = db.query(
        "SELECT cfgname FROM pg_ts_config WHERE cfgnamespace = to_regnamespace(%(s)s)",
        {"s": validate_schema_name(schema)},
    )
    return {row["cfgname"] for row in rows}


def index_exists(db: PostgresDatabase, schema: str, index: str) -> bool:
    rows = db.query("SELECT to_regclass(%(name)s) IS NOT NULL AS found", {"name": f"{validate_schema_name(schema)}.{index}"})
    return bool(rows and rows[0]["found"])


def vector_index_type(db: PostgresDatabase, schema: str) -> Optional[str]:
    """Access method of the ANN index on works.embedding (lakebase_ann, hnsw) or None."""
    rows = db.query(
        """SELECT am.amname FROM pg_class c JOIN pg_am am ON am.oid = c.relam
           WHERE c.oid = to_regclass(%(name)s)""",
        {"name": f"{validate_schema_name(schema)}.{VECTOR_INDEX}"},
    )
    return rows[0]["amname"] if rows else None


# =============================================================================
# Indexes (after the bulk load)
# =============================================================================


def build_lexical_indexes(
    db: PostgresDatabase, schema: str, *, k1: float = BM25_K1, b: float = BM25_B, gin: bool = True
) -> Dict[str, float]:
    """
    (Re)build the GIN index and, when lakebase_text is installed, the BM25 index.

    The BM25 index is always rebuilt (its corpus statistics are frozen at
    build time). The GIN index serves only the ts_rank_cd comparison ranking
    (matching with @@), not BM25, so a deployed catalog can leave it out
    (gin=False drops an existing one); ts_rank_cd still works without it,
    by scanning every text row. Returns seconds per index.
    """
    seconds: Dict[str, float] = {}
    schema = validate_schema_name(schema)
    if not gin:
        db.execute(f"DROP INDEX IF EXISTS {schema}.{GIN_INDEX}")
    elif not index_exists(db, schema, GIN_INDEX):
        started = time.perf_counter()
        db.execute(_section("lexical_index_gin", schema))
        seconds[GIN_INDEX] = time.perf_counter() - started
    if "lakebase_text" in installed_extensions(db):
        db.execute(f"DROP INDEX IF EXISTS {schema}.{BM25_INDEX}")
        started = time.perf_counter()
        db.execute(_section("lexical_index_bm25", schema, k1=k1, b=b))
        seconds[BM25_INDEX] = time.perf_counter() - started
    return seconds


def build_vector_index(db: PostgresDatabase, schema: str, index_type: str) -> float:
    """
    (Re)build the ANN index on works.embedding with the given access method.

    Returns:
        Seconds taken

    Raises:
        ValueError: If the index type is unknown or its extension is missing
    """
    if index_type not in VECTOR_INDEX_TYPES:
        raise ValueError(f"Vector index must be one of {VECTOR_INDEX_TYPES}, got '{index_type}'")
    if index_type == "lakebase_ann" and "lakebase_vector" not in installed_extensions(db):
        raise ValueError("lakebase_ann needs the lakebase_vector extension (Neon); use --vector-index hnsw")
    schema = validate_schema_name(schema)
    db.execute(f"DROP INDEX IF EXISTS {schema}.{VECTOR_INDEX}")
    started = time.perf_counter()
    db.execute(_section(f"vector_index_{index_type}", schema))
    return time.perf_counter() - started


def vacuum_analyze(db: PostgresDatabase, schema: str) -> float:
    """VACUUM ANALYZE both tables (refreshes planner and BM25 statistics). Seconds taken."""
    schema = validate_schema_name(schema)
    started = time.perf_counter()
    for table in ("works", "work_texts"):
        db.execute_outside_transaction(f"VACUUM (ANALYZE) {schema}.{table}")
    return time.perf_counter() - started


# =============================================================================
# catalog_info and reports
# =============================================================================


def read_catalog_info(db: PostgresDatabase, schema: str) -> Dict[str, Any]:
    """Every catalog_info entry (empty if the table does not exist)."""
    schema = validate_schema_name(schema)
    exists = db.query("SELECT to_regclass(%(t)s) IS NOT NULL AS found", {"t": f"{schema}.catalog_info"})
    if not exists or not exists[0]["found"]:
        return {}
    return {row["key"]: row["value"] for row in db.query(f"SELECT key, value FROM {schema}.catalog_info")}


def write_catalog_info(db: PostgresDatabase, schema: str, entries: Mapping[str, Any]) -> None:
    """Insert or replace catalog_info entries (values are stored as JSON)."""
    if not entries:
        return
    schema = validate_schema_name(schema)
    values = ", ".join(f"(%(k{i})s, %(v{i})s::jsonb)" for i in range(len(entries)))
    params: Dict[str, Any] = {}
    for i, (key, value) in enumerate(entries.items()):
        params[f"k{i}"] = key
        params[f"v{i}"] = json.dumps(value, ensure_ascii=False)
    db.execute(
        f"INSERT INTO {schema}.catalog_info (key, value) VALUES {values} "
        f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        params,
    )


def relation_sizes(db: PostgresDatabase, schema: str) -> List[Dict[str, Any]]:
    """
    Tables and indexes of the schema with their row counts and sizes in bytes.

    Tables: rows (exact count) and total_bytes (pg_total_relation_size:
    heap, TOAST and indexes). Indexes: bytes (pg_relation_size) and the
    access method.
    """
    schema = validate_schema_name(schema)
    out: List[Dict[str, Any]] = []
    for table in ("works", "work_texts", "catalog_info"):
        qualified = f"{schema}.{table}"
        rows = db.query(
            f"SELECT count(*) AS n, pg_total_relation_size(%(t)s) AS total, pg_relation_size(%(t)s) AS heap, "
            f"pg_table_size(%(t)s) AS table_bytes FROM {qualified}",
            {"t": qualified},
        )
        out.append({
            "relation": table, "kind": "table", "rows": rows[0]["n"], "total_bytes": rows[0]["total"],
            "table_bytes": rows[0]["table_bytes"], "heap_bytes": rows[0]["heap"],
        })
    for row in db.query(
        """SELECT c.relname AS name, t.relname AS on_table, am.amname AS method, pg_relation_size(c.oid) AS bytes
           FROM pg_index i
           JOIN pg_class c ON c.oid = i.indexrelid
           JOIN pg_class t ON t.oid = i.indrelid
           JOIN pg_am am ON am.oid = c.relam
           WHERE c.relnamespace = to_regnamespace(%(s)s)
           ORDER BY t.relname, c.relname""",
        {"s": schema},
    ):
        out.append({
            "relation": row["name"], "kind": "index", "table": row["on_table"], "method": row["method"],
            "bytes": row["bytes"],
        })
    return out


def schema_total_bytes(db: PostgresDatabase, schema: str) -> int:
    """Total size of every table (with TOAST and indexes) in the schema."""
    rows = db.query(
        """SELECT coalesce(sum(pg_total_relation_size(c.oid)), 0)::bigint AS total
           FROM pg_class c WHERE c.relnamespace = to_regnamespace(%(s)s) AND c.relkind = 'r'""",
        {"s": validate_schema_name(schema)},
    )
    return int(rows[0]["total"])


def server_versions(db: PostgresDatabase) -> Dict[str, Any]:
    """Postgres version, database size and the versions of the extensions used."""
    version = db.query("SELECT current_setting('server_version') AS v, pg_database_size(current_database()) AS size")
    extensions = installed_extensions(db)
    return {
        "postgres": version[0]["v"],
        "database_bytes": int(version[0]["size"]),
        "extensions": {name: extensions.get(name) for name in ("vector", "lakebase_vector", "lakebase_text", "unaccent")},
    }
