"""
Load a SQLite catalog of this project into a Postgres schema (cloud profile, V12).

    python -m scripts.load_postgres --from data/catalog.db --schema eval177 \\
        --embedding-model intfloat/multilingual-e5-large-instruct
    python -m scripts.load_postgres --from data/catalog_small/catalog.db --schema catalog_small
    SEARCH_PROFILE=cloud python -m scripts.load_postgres --from data/catalog.db --schema eval177_api
        # documents embedded by the cloud profile's API encoder (qwen/qwen3-embedding-8b, 1024 d, deepinfra)
    python -m scripts.load_postgres --schema eval177 --migrate-only   # schema only
    python -m scripts.load_postgres --schema eval177 --report         # sizes only
    python -m scripts.load_postgres --schema eval177 --drop           # delete the schema

Uses the direct connection string (DATABASE_URL_UNPOOLED, falling back to
DATABASE_URL), as Neon recommends for migrations, bulk loads and index
builds. Steps, each resumable (a second run with an unchanged source writes
nothing, re-embeds nothing and rebuilds nothing):

1. Apply the schema (app/infrastructure/db/postgres_schema.sql) with the
   embedding dimension of the model.
2. Upsert works and their texts in batches through
   PostgresBookCatalogRepository (unchanged works are skipped).
3. Delete works that are no longer in the source catalog.
4. Fill missing embeddings. Vectors of a local FAISS index built with the
   same encoder identity are reused (an index under data/indexes_ablation/<slug>/
   or data/indexes/ is used automatically when it holds exactly the catalog's
   books; --reuse-index names one explicitly); the rest are encoded with
   the configured encoder (EmbeddingSettings, as the API: EMBEDDING_PROVIDER
   "local" loads the sentence-transformers model EMBEDDING_MODEL, or
   --embedding-model; "openrouter" uses OpenRouter's embeddings API with
   EMBEDDING_MODEL, EMBEDDING_DIMENSIONS and EMBEDDING_OPENROUTER_PROVIDER,
   whose defaults under SEARCH_PROFILE=cloud are the cloud choice). The
   halfvec dimension is the encoder's. A stored vector is kept while its
   text is unchanged (works.text_hash) and the encoder identity is the one
   recorded in catalog_info; a schema loaded with another identity (model,
   provider or dimension) is refused: use another schema or --drop it.
5. Build the indexes (GIN, BM25 if lakebase_text is installed, ANN) when
   anything changed or an index is missing, then VACUUM ANALYZE.
6. Record the facts the search adapters need in catalog_info (among them
   the encoder identity: "embedding_model" is the identity string the API
   compares with its query encoder, "embedding_identity" holds model,
   provider and dimension) and print row counts and sizes per table and index.

The source SQLite file is opened read-only.
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from dotenv import load_dotenv

from app.infrastructure.config.settings import (
    DEFAULT_DATABASE_SCHEMA,
    EmbeddingSettings,
    database_url,
)
from app.infrastructure.search.text_encoding import model_slug
from app.infrastructure.usage_recording import UsageRecorder
from app.infrastructure.db.postgres_book_catalog_repository import PostgresBookCatalogRepository
from app.infrastructure.db.postgres_connection import PostgresDatabase, redact
from app.infrastructure.db.postgres_schema import (
    BM25_B,
    BM25_INDEX,
    BM25_K1,
    GIN_INDEX,
    VECTOR_INDEX,
    VECTOR_INDEX_TYPES,
    apply_schema,
    build_lexical_indexes,
    build_vector_index,
    drop_schema,
    embedding_dimension_in_db,
    index_exists,
    installed_extensions,
    read_catalog_info,
    relation_sizes,
    schema_total_bytes,
    server_versions,
    vacuum_analyze,
    validate_schema_name,
    write_catalog_info,
)
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

UPSERT_BATCH = 500
EMBED_BATCH = 64


class ReadOnlySqliteCatalog(SqliteBookCatalogRepository):
    """A SQLite catalog opened read-only: the loader never writes to its source."""

    def __init__(self, db_path: Path) -> None:  # no schema creation, unlike the parent
        if not db_path.is_file():
            raise FileNotFoundError(f"SQLite catalog not found: {db_path}")
        self._db_path = db_path
        self._compact_json = False

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path.resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def all_ids(self) -> List[UUID]:
        with self._get_connection() as conn:
            return [UUID(row["id"]) for row in conn.execute("SELECT id FROM books ORDER BY id")]


# =============================================================================
# Reusing the vectors of a local FAISS index
# =============================================================================


def _canonical(model: str) -> str:
    name = model.strip().lower()
    return name[len("sentence-transformers/"):] if name.startswith("sentence-transformers/") else name


def faiss_index_model(index_dir: Path) -> Optional[str]:
    """Model recorded with a FAISS index (legacy indexes are MiniLM), or None if there is no index."""
    if not index_dir.is_dir():
        return None  # nothing to read: avoid importing the FAISS store (torch) for nothing
    from app.infrastructure.search.embeddings_store_faiss import (
        INDEX_FILE,
        LEGACY_INDEX_MODEL,
        read_index_metadata,
    )

    if not (index_dir / INDEX_FILE).is_file():
        return None
    metadata = read_index_metadata(str(index_dir))
    return metadata.model_name if metadata else LEGACY_INDEX_MODEL


def faiss_vectors(index_dir: Path) -> Dict[UUID, List[float]]:
    """Every vector of a saved FAISS index, by book id (as stored: unit length for cosine indexes)."""
    from app.infrastructure.search.embeddings_store_faiss import INDEX_FILE, MAPPING_FILE
    from app.infrastructure.search.faiss_runtime import faiss  # torch before faiss (macOS)

    index = faiss.read_index(str(index_dir / INDEX_FILE))
    mapping = json.loads((index_dir / MAPPING_FILE).read_text(encoding="utf-8"))
    matrix = index.reconstruct_n(0, index.ntotal)
    return {UUID(book_id): matrix[i].tolist() for i, book_id in enumerate(mapping)}


def find_reusable_index(
    model: str, catalog_ids: Sequence[UUID], candidates: Sequence[Path], explicit: Optional[Path]
) -> Optional[Path]:
    """
    The FAISS index whose vectors can be reused, or None.

    An explicit index only needs the same model (its vectors are used for
    the ids it has). A candidate found automatically must also hold exactly
    the catalog's ids, i.e. it was built from this catalog.

    Raises:
        ValueError: If the explicit index was built with another model
    """
    if explicit is not None:
        found = faiss_index_model(explicit)
        if found is None:
            raise ValueError(f"No FAISS index in {explicit}")
        if _canonical(found) != _canonical(model):
            raise ValueError(f"{explicit} holds vectors of '{found}', not of '{model}'")
        return explicit
    wanted = set(catalog_ids)
    for candidate in candidates:
        found = faiss_index_model(candidate)
        if found is None or _canonical(found) != _canonical(model):
            continue
        from app.infrastructure.search.embeddings_store_faiss import MAPPING_FILE

        ids = {UUID(i) for i in json.loads((candidate / MAPPING_FILE).read_text(encoding="utf-8"))}
        if ids == wanted:
            return candidate
    return None


# =============================================================================
# Encoder identity
# =============================================================================


def identity_record(identity: str, dimension: int, encoder: Any = None) -> Dict[str, Any]:
    """
    What catalog_info records about the encoder of the stored vectors.

    identity: the encoder's get_model_name() (a local model's name, or
    "openrouter:<model>?provider=<p>&dimensions=<d>" for the API encoder),
    compared by PgVectorSearchRepository with the query encoder's.
    """
    is_api = encoder is not None and getattr(encoder, "get_device", lambda: "")() == "api"
    return {
        "identity": identity,
        "model": (getattr(encoder, "model", None) or identity) if is_api else identity,
        "provider": f"openrouter/{getattr(encoder, 'upstream_provider', None) or 'any'}" if is_api else "local",
        "dimension": int(dimension),
    }


# =============================================================================
# Load
# =============================================================================


def _log(message: str) -> None:
    print(message, flush=True)


def _fmt_bytes(value: Optional[int]) -> str:
    return "n/a" if value is None else f"{value / 1024 / 1024:.2f} MiB"


def print_report(db: PostgresDatabase, schema: str, log: Callable[[str], None] = _log) -> Dict[str, Any]:
    """Row counts and sizes per table and index; returns them."""
    sizes = relation_sizes(db, schema)
    total = schema_total_bytes(db, schema)
    versions = server_versions(db)
    log(f"Schema {schema}: {_fmt_bytes(total)} in total; database {_fmt_bytes(versions['database_bytes'])}")
    for item in sizes:
        if item["kind"] == "table":
            log(f"  table {item['relation']:<14} rows {item['rows']:>8}  total {_fmt_bytes(item['total_bytes'])}"
                f"  (table+TOAST {_fmt_bytes(item['table_bytes'])})")
        else:
            log(f"  index {item['relation']:<24} on {item['table']:<11} {item['method']:<13} {_fmt_bytes(item['bytes'])}")
    return {"relations": sizes, "schema_total_bytes": total, **versions}


def load(
    db: PostgresDatabase,
    source: Path,
    schema: str,
    *,
    embedding_model: Optional[str] = None,
    encoder: Any = None,
    device: Optional[str] = None,
    vector_index: str,
    reuse_index: Optional[Path],
    rebuild_indexes: bool = False,
    upsert_batch: int = UPSERT_BATCH,
    embed_batch: int = EMBED_BATCH,
    log: Callable[[str], None] = _log,
) -> Dict[str, Any]:
    """
    Run the load steps (see the module docstring); returns what was done.

    Give exactly one of:
        embedding_model: a local sentence-transformers model, loaded only if
            a vector must be computed (reused vectors need no model)
        encoder: an already built encoder (e.g. OpenRouterEmbeddingEncoder);
            its get_model_name() is the identity, its get_dimension() the
            halfvec dimension

    Raises:
        ValueError: If the schema holds vectors of another encoder identity
            or dimension, or both or neither of embedding_model and encoder are given
    """
    if (embedding_model is None) == (encoder is None):
        raise ValueError("Give either embedding_model (local model) or encoder")
    started = time.perf_counter()
    catalog = ReadOnlySqliteCatalog(source)
    source_ids = catalog.all_ids()
    log(f"Source {source}: {len(source_ids)} books")

    # 1. Encoder identity, embedding dimension (from the encoder, a reusable
    #    index, the schema or the local model) and schema.
    identity = encoder.get_model_name() if encoder is not None else embedding_model
    candidates = [Path("data/indexes_ablation") / model_slug(identity) / "faiss_index",
                  Path("data/indexes/faiss_index")]
    reusable = find_reusable_index(identity, source_ids, candidates, reuse_index)

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss

            log(f"Loading embedding model {embedding_model}")
            encoder = EmbeddingsStoreFaiss(model_name=embedding_model, device=device)
        return encoder

    reused_vectors: Dict[UUID, List[float]] = {}
    if reusable is not None:
        reused_vectors = faiss_vectors(reusable)
        log(f"Reusing {len(reused_vectors)} vectors of {identity} from {reusable}")
    if encoder is not None:
        dim = encoder.get_dimension()
    elif reused_vectors:
        dim = len(next(iter(reused_vectors.values())))
    else:
        dim = embedding_dimension_in_db(db, schema) or get_encoder().get_dimension()
    if reused_vectors and len(next(iter(reused_vectors.values()))) != dim:
        raise ValueError(f"{reusable} holds vectors of another dimension than the encoder's {dim}")
    apply_schema(db, schema, dim)
    info = read_catalog_info(db, schema)
    stored = info.get("embedding_model")
    if stored and _canonical(stored) != _canonical(identity):
        raise ValueError(
            f"Schema '{schema}' holds vectors of '{stored}', not of '{identity}' (another model, provider or "
            f"dimension): use another schema or --drop it first"
        )
    write_catalog_info(db, schema, {"embedding_model": identity,
                                    "embedding_identity": identity_record(identity, dim, encoder)})

    # 2. Works and texts.
    repo = PostgresBookCatalogRepository(db, schema)
    t0 = time.perf_counter()
    changed = 0
    batch = []
    for book in catalog.iter_all(batch_size=upsert_batch):
        batch.append(book)
        if len(batch) >= upsert_batch:
            changed += repo.upsert_many(batch)
            batch = []
            log(f"  upserted up to {book.id} ({changed} new or changed so far)")
    changed += repo.upsert_many(batch)
    upsert_seconds = time.perf_counter() - t0
    log(f"Works: {changed} new or changed ({upsert_seconds:.1f} s)")

    # 3. Prune works that left the source.
    extra = sorted(set(repo.all_ids()) - set(source_ids))
    for start in range(0, len(extra), 1000):
        repo.delete_many(extra[start:start + 1000])
    if extra:
        log(f"Deleted {len(extra)} works that are no longer in the source")
    if changed or extra:
        write_catalog_info(db, schema, {"indexes_stale": True})

    # 4. Embeddings.
    from app.infrastructure.search.postgres_embeddings_store import PostgresEmbeddingsStore

    missing = [row["id"] for row in db.query(f"SELECT id FROM {schema}.works WHERE embedding IS NULL ORDER BY id")]
    t0 = time.perf_counter()
    reused_now = [book_id for book_id in missing if book_id in reused_vectors]
    to_encode = [book_id for book_id in missing if book_id not in reused_vectors]

    class _DimOnly:  # stores reused vectors without loading the model
        def get_dimension(self) -> int:
            return dim

    store = PostgresEmbeddingsStore(_DimOnly() if not to_encode else get_encoder(), db, schema)  # type: ignore[arg-type]
    if reused_now:
        store.store_embeddings_batch(reused_now, [reused_vectors[i] for i in reused_now])
        log(f"Stored {len(reused_now)} reused vectors")
    encode_seconds = 0.0
    for start in range(0, len(to_encode), embed_batch * 8):
        ids = to_encode[start:start + embed_batch * 8]
        books = catalog.get_many(ids)
        texts = [books[i].get_searchable_text() for i in ids]
        e0 = time.perf_counter()
        vectors = []
        for j in range(0, len(texts), embed_batch):
            vectors.extend(get_encoder().generate_embeddings_batch(texts[j:j + embed_batch]))
        encode_seconds += time.perf_counter() - e0
        store.store_embeddings_batch(ids, vectors)
        log(f"  encoded {min(start + len(ids), len(to_encode))}/{len(to_encode)}")
    embed_seconds = time.perf_counter() - t0
    if missing:
        write_catalog_info(db, schema, {"indexes_stale": True})
    log(f"Embeddings: {len(reused_now)} reused, {len(to_encode)} encoded ({embed_seconds:.1f} s)")

    # 5. Indexes and statistics.
    info = read_catalog_info(db, schema)
    has_bm25 = "lakebase_text" in installed_extensions(db)
    current_vector = info.get("vector_index")
    needs_build = (
        rebuild_indexes
        or info.get("indexes_stale", True)
        or not index_exists(db, schema, GIN_INDEX)
        or (has_bm25 and not index_exists(db, schema, BM25_INDEX))
        or not index_exists(db, schema, VECTOR_INDEX)
        or current_vector != vector_index
    )
    build_seconds: Dict[str, float] = dict(info.get("build_seconds") or {})
    vacuum_seconds = None
    if needs_build:
        log("Building indexes")
        db.execute(f"DROP INDEX IF EXISTS {schema}.{GIN_INDEX}")
        build_seconds.update(build_lexical_indexes(db, schema, k1=BM25_K1, b=BM25_B))
        build_seconds[f"{VECTOR_INDEX} ({vector_index})"] = build_vector_index(db, schema, vector_index)
        vacuum_seconds = vacuum_analyze(db, schema)
        for name, seconds in build_seconds.items():
            log(f"  {name}: {seconds:.2f} s")
        log(f"  VACUUM ANALYZE: {vacuum_seconds:.2f} s")
    else:
        log("Indexes are up to date")

    # 6. catalog_info and report.
    counts = db.query(
        f"SELECT (SELECT count(*) FROM {schema}.works) AS works, "
        f"(SELECT count(*) FROM {schema}.works WHERE embedding IS NOT NULL) AS with_embedding, "
        f"(SELECT count(*) FROM {schema}.work_texts) AS texts"
    )[0]
    facts = {
        "embedding_model": identity,
        "embedding_identity": identity_record(identity, dim, encoder),
        "embedding_dimension": dim,
        "vector_index": vector_index,
        "lexical_index": "lakebase_bm25" if has_bm25 else None,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "works": counts["works"],
        "works_with_embedding": counts["with_embedding"],
        "work_texts": counts["texts"],
        "indexes_stale": False,
        "build_seconds": build_seconds,
        "source_catalog": str(source),
        "loaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if needs_build:
        facts["vacuum_seconds"] = vacuum_seconds
    if changed or missing:
        facts["last_load"] = {
            "upsert_seconds": upsert_seconds,
            "works_changed": changed,
            "works_deleted": len(extra),
            "vectors_reused": len(reused_now),
            "vectors_encoded": len(to_encode),
            "vectors_reused_from": str(reusable) if reusable else None,
            "encode_seconds": encode_seconds,
            "embedding_seconds": embed_seconds,
        }
    write_catalog_info(db, schema, facts)
    report = print_report(db, schema, log)
    log(f"Done in {time.perf_counter() - started:.1f} s")
    return {"facts": facts, "report": report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, help="SQLite catalog of this project")
    parser.add_argument("--schema", default=None, help=f"target schema (default DATABASE_SCHEMA or {DEFAULT_DATABASE_SCHEMA})")
    parser.add_argument("--embedding-model", default=None,
                        help="default: EMBEDDING_MODEL (the configured encoder: EMBEDDING_PROVIDER local or openrouter)")
    parser.add_argument("--device", default=None, help="cpu, mps or cuda (default: EMBEDDING_DEVICE or automatic)")
    parser.add_argument("--vector-index", choices=VECTOR_INDEX_TYPES, default=None,
                        help="ANN index (default lakebase_ann when available, else hnsw)")
    parser.add_argument("--reuse-index", type=Path, default=None, help="FAISS index directory whose vectors to reuse")
    parser.add_argument("--rebuild-indexes", action="store_true", help="rebuild the indexes even if up to date")
    parser.add_argument("--migrate-only", action="store_true", help="apply the schema and stop (needs --dim)")
    parser.add_argument("--dim", type=int, default=None, help="embedding dimension for --migrate-only")
    parser.add_argument("--report", action="store_true", help="print sizes of the schema and stop")
    parser.add_argument("--drop", action="store_true", help="drop the schema and everything in it")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(Path(".env"))
    import os

    schema = validate_schema_name(args.schema or os.environ.get("DATABASE_SCHEMA", "").strip() or DEFAULT_DATABASE_SCHEMA)
    db = PostgresDatabase(database_url(direct=True), pool_max_size=1, statement_timeout_ms=None)
    try:
        if args.drop:
            drop_schema(db, schema)
            _log(f"Dropped schema {schema}")
            return 0
        if args.report:
            print_report(db, schema)
            return 0
        if args.migrate_only:
            if not args.dim:
                _log("--migrate-only needs --dim (the embedding dimension)")
                return 2
            apply_schema(db, schema, args.dim)
            _log(f"Schema {schema} is up to date")
            return 0
        if args.source is None:
            _log("--from is required to load a catalog")
            return 2
        settings = embedding_settings(os.environ, args.embedding_model)
        vector_index = args.vector_index or (
            "lakebase_ann" if "lakebase_vector" in _available(db) else "hnsw"
        )
        recorder = UsageRecorder()
        encoder = build_document_encoder(settings, recorder)
        load(
            db,
            args.source,
            schema,
            embedding_model=None if encoder is not None else settings.model,
            encoder=encoder,
            device=args.device or settings.device,
            vector_index=vector_index,
            reuse_index=args.reuse_index,
            rebuild_indexes=args.rebuild_indexes,
        )
        total = recorder.summary()["total"]
        if total["calls"]:
            _log(f"Embedding API: {total['calls']} requests, {total['input_tokens']} tokens, "
                 f"reported cost {total['reported_cost_usd']} USD")
        return 0
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        _log(f"Error: {redact(error)}")
        return 1
    finally:
        db.close()


def embedding_settings(env: Mapping[str, str], model_override: Optional[str] = None) -> EmbeddingSettings:
    """
    The encoder settings, resolved like the API's (EmbeddingSettings.from_env, with
    the cloud defaults when SEARCH_PROFILE=cloud); --embedding-model replaces EMBEDDING_MODEL.
    """
    if model_override:
        env = {**env, "EMBEDDING_MODEL": model_override}
    return EmbeddingSettings.from_env(env)


def build_document_encoder(settings: EmbeddingSettings, recorder: Optional[UsageRecorder] = None) -> Any:
    """
    The API encoder when EMBEDDING_PROVIDER is openrouter (its requests are
    recorded in `recorder`), or None for a local model, which load() builds
    only if some vector must be computed.
    """
    if settings.provider != "openrouter":
        return None
    from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder

    return OpenRouterEmbeddingEncoder.from_settings(settings, usage_recorder=recorder)


def _available(db: PostgresDatabase) -> set:
    return {row["name"] for row in db.query("SELECT name FROM pg_available_extensions")}


if __name__ == "__main__":
    sys.exit(main())
