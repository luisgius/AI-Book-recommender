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
    """The stored vectors were computed with another model than the query's."""


class PgVectorSearchRepository(VectorSearchRepository):
    """
    Vector search served by Postgres (pgvector / lakebase_vector).

    Args:
        db: Database (the pooled connection string in the API)
        schema: Catalog schema
        mode: "approximate" (ANN index) or "exact" (sequential scan)
        expected_model: Embedding model of the queries; when given, the
            model recorded by the loader must be the same (like the FAISS
            index metadata check of the local profile)
    """

    def __init__(
        self,
        db: PostgresDatabase,
        schema: str,
        mode: str = "approximate",
        expected_model: Optional[str] = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
        self._db = db
        self._schema = validate_schema_name(schema)
        self._mode = mode
        self._expected_model = expected_model
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
                    model = info.get("embedding_model")
                    if self._expected_model and model and _canonical(model) != _canonical(self._expected_model):
                        raise EmbeddingModelMismatchError(
                            f"Schema '{self._schema}' holds embeddings of '{model}' but queries are embedded "
                            f"with '{self._expected_model}'. Reload the schema with that model or set "
                            f"EMBEDDING_MODEL={model}."
                        )
                    self._info = info
        return self._info

    def dimension(self) -> int:
        return int(self._catalog_info()["embedding_dimension"])

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
