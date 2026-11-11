"""
pgvector implementation of the VectorSearchRepository port (cloud profile).

Document embeddings live in <schema>.works.embedding as halfvec (16-bit
floats, half the size of float32; pgvector indexes halfvec up to 4,000
dimensions). Similarity is the cosine: score = 1 - (embedding <=> query).

Two modes:
- "approximate" (default for serving): the ANN index of the schema answers
  ORDER BY distance LIMIT k. The index type is read from catalog_info:
    * lakebase_ann (Neon's index, IVF with full-precision re-ranking; flat
      exact search while the table is small): probes and epsilon stay at
      'auto'; with filters, lakebase_ann.prefilter is switched on.
    * hnsw (pgvector): hnsw.ef_search = max(40, 2k) and iterative scan in
      relaxed order, so a filtered query keeps walking the graph until it has
      k rows; the k rows are re-sorted by exact distance afterwards.
  If a filtered query still returns fewer than k rows, it is repeated once
  in exact mode.
- "exact" (evaluation): index scans are disabled inside the statement's
  transaction (SET LOCAL enable_indexscan / enable_bitmapscan = off), so
  Postgres scans every row and sorts by distance: the same ranking as the
  local flat FAISS index up to ties and half-precision rounding.

One statement per search returns fully hydrated rows.

Identity check (V13): the loader records which encoder produced the stored
vectors (catalog_info "embedding_model": the encoder's identity, e.g.
"intfloat/multilingual-e5-large-instruct" for a local model or
"openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=1024" for
the API encoder, plus "embedding_identity" with model, provider and
dimension). Given the query encoder's identity, the repository refuses to
search vectors of another model, provider or dimension, as the FAISS index
metadata check does in the local profile.
"""

import logging
import threading
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.entities import SearchResult
from app.domain.ports import VectorSearchRepository
from app.domain.value_objects import SearchFilters
from app.infrastructure.db.postgres_connection import PostgresDatabase
from app.infrastructure.db.postgres_rows import WORK_COLUMNS, filter_condition, parse_vector, row_to_book, vector_literal
from app.infrastructure.db.postgres_schema import read_catalog_info, validate_schema_name

logger = logging.getLogger(__name__)

MODES = ("approximate", "exact")
EXACT_SETTINGS: Dict[str, object] = {
    "enable_indexscan": "off",
    "enable_bitmapscan": "off",
    "enable_indexonlyscan": "off",
}
MIN_EF_SEARCH = 40  # pgvector's default


class EmbeddingModelMismatchError(RuntimeError):
    """The stored vectors were computed with another encoder (model, provider or dimension) than the query's."""


class PgVectorSearchRepository(VectorSearchRepository):
    """
    Vector search served by Postgres (pgvector / lakebase_vector).

    Args:
        db: Database (the pooled connection string in the API)
        schema: Catalog schema
        mode: "approximate" (ANN index) or "exact" (sequential scan)
        expected_identity: Identity of the query encoder (its
            get_model_name(): the model name of a local model, or
            "openrouter:<model>?provider=<p>&dimensions=<d>" for the API
            encoder). When given, the identity recorded by the loader must
            be the same, or every search raises EmbeddingModelMismatchError
            (and is_ready() is False, so the API degrades to lexical search).
    """

    def __init__(
        self,
        db: PostgresDatabase,
        schema: str,
        mode: str = "approximate",
        expected_identity: Optional[str] = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
        self._db = db
        self._schema = validate_schema_name(schema)
        self._mode = mode
        self._expected_identity = expected_identity
        self._info: Optional[Dict[str, Any]] = None
        self._info_lock = threading.Lock()

    @property
    def mode(self) -> str:
        return self._mode

    def _catalog_info(self) -> Dict[str, Any]:
        if self._info is None:
            with self._info_lock:
                if self._info is None:
                    info = read_catalog_info(self._db, self._schema)
                    if not info.get("works_with_embedding"):
                        raise RuntimeError(
                            f"Catalog schema '{self._schema}' has no embeddings: load it with "
                            f"python -m scripts.load_postgres"
                        )
                    self._check_identity(info)
                    self._info = info
        return self._info

    def _check_identity(self, info: Dict[str, Any]) -> None:
        """
        Raise if the stored vectors come from another encoder than the queries.

        Raises:
            EmbeddingModelMismatchError: Naming both identities
        """
        if not self._expected_identity:
            return
        stored = info.get("embedding_model")
        if stored and _canonical(stored) == _canonical(self._expected_identity):
            return
        details = info.get("embedding_identity") or {}
        described = (
            f" (model {details.get('model')}, provider {details.get('provider')}, "
            f"dimension {details.get('dimension')})" if details else
            f" (dimension {info.get('embedding_dimension')})" if stored else ""
        )
        raise EmbeddingModelMismatchError(
            f"Schema '{self._schema}' holds vectors of '{stored or 'an unrecorded encoder'}'{described}, but queries "
            f"are embedded with '{self._expected_identity}'. Query and document vectors must come from the same "
            f"model, provider and dimension: load the schema with this encoder (python -m scripts.load_postgres) "
            f"or configure the encoder it was loaded with (EMBEDDING_PROVIDER, EMBEDDING_MODEL, "
            f"EMBEDDING_DIMENSIONS, EMBEDDING_OPENROUTER_PROVIDER)."
        )

    def dimension(self) -> int:
        return int(self._catalog_info()["embedding_dimension"])

    def stored_identity(self) -> Optional[str]:
        """Identity of the encoder the stored vectors were built with (catalog_info)."""
        return self._catalog_info().get("embedding_model")

    def index_type(self) -> Optional[str]:
        """ANN index of the schema (lakebase_ann, hnsw) or None."""
        return self._catalog_info().get("vector_index")

    # ------------------------------------------------------------------
    # SQL
    # ------------------------------------------------------------------

    def build_sql(
        self, max_results: int, filters: Optional[SearchFilters], *, exact: bool
    ) -> tuple[str, Dict[str, Any], Dict[str, object]]:
        """The search statement, its parameters (except the vector) and its settings."""
        s = self._schema
        condition, params = filter_condition(filters, "w")
        filtered = condition != "TRUE"
        params["k"] = max_results
        if exact:
            settings = dict(EXACT_SETTINGS)
        else:
            settings = {}
            index = self.index_type()
            if index == "hnsw":
                settings["hnsw.ef_search"] = max(MIN_EF_SEARCH, 2 * max_results)
                settings["hnsw.iterative_scan"] = "relaxed_order"
            elif index == "lakebase_ann" and filtered:
                settings["lakebase_ann.prefilter"] = "on"
        # The inner ORDER BY ... LIMIT is the shape the ANN index serves; the
        # outer sort restores exact distance order (relaxed iterative scans
        # may return rows slightly out of order) and breaks ties by id.
        sql = f"""
WITH nearest AS MATERIALIZED (
  SELECT {WORK_COLUMNS}, w.embedding <=> %(vector)s::halfvec AS distance
  FROM {s}.works w
  WHERE w.embedding IS NOT NULL AND {condition}
  ORDER BY distance
  LIMIT %(k)s
)
SELECT * FROM nearest ORDER BY distance, id"""
        return sql, params, settings

    # ------------------------------------------------------------------
    # Port
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """
        Nearest works by cosine similarity.

        Returns:
            SearchResult list (source "vector", 1-based rank, final_score =
            vector_score = cosine similarity)

        Raises:
            ValueError: If the embedding has the wrong dimension
            RuntimeError: If the database is unavailable or the catalog is not loaded
        """
        expected = self.dimension()
        if len(query_embedding) != expected:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {expected}, got {len(query_embedding)}"
            )
        if max_results < 1:
            return []
        exact = self._mode == "exact"
        rows = self._run(query_embedding, max_results, filters, exact=exact)
        if not exact and filters is not None and not filters.is_empty() and len(rows) < max_results:
            rows = self._run(query_embedding, max_results, filters, exact=True)
        results = []
        for rank, row in enumerate(rows, start=1):
            similarity = 1.0 - float(row["distance"])
            results.append(
                SearchResult(
                    book=row_to_book(row),
                    final_score=similarity,
                    rank=rank,
                    source="vector",
                    lexical_score=None,
                    vector_score=similarity,
                )
            )
        return results

    def _run(self, vector: List[float], max_results: int, filters: Optional[SearchFilters], *, exact: bool):
        sql, params, settings = self.build_sql(max_results, filters, exact=exact)
        params["vector"] = vector_literal(vector)
        return self._db.query(sql, params, settings=settings)

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        """Stored embedding of a work (infrastructure helper for item-to-item search)."""
        rows = self._db.query(
            f"SELECT embedding::text AS e FROM {self._schema}.works WHERE id = %(id)s AND embedding IS NOT NULL",
            {"id": book_id},
        )
        return parse_vector(rows[0]["e"]) if rows else None

    def is_ready(self) -> bool:
        """True once the catalog facts were read and match the query model."""
        try:
            self._catalog_info()
            return True
        except Exception as error:  # noqa: BLE001 - health check must not raise
            logger.warning("pgvector search not ready: %s", error)
            return False


def _canonical(model_name: str) -> str:
    name = model_name.strip().lower()
    prefix = "sentence-transformers/"
    return name[len(prefix):] if name.startswith(prefix) else name
