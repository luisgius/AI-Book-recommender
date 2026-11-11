"""
Postgres implementation of the LexicalSearchRepository port (cloud profile).

Two rankings over the same documents (<schema>.work_texts.tsv: title,
authors and categories unstemmed, plus the description stemmed in its own
language, accents folded everywhere):

- "bm25" (default): real BM25 from the lakebase_text index (Neon), with the
  k1 and b of the local BM25 (1.5, 0.75). The index computes the exact top K
  (block-max WAND) from corpus statistics taken when it was built.
- "ts_rank_cd": Postgres' own full-text ranking (cover density, divided by
  1 + log(document length) to approximate BM25's length normalisation). It
  has no corpus statistics (no IDF), so it is kept as a comparison.

Query construction (both rankings), designed for recall:
  The query text is turned into lexemes under every text-search
  configuration in use (SQL function <schema>.query_lexemes): unstemmed
  (simple_unaccent, matches titles and names) and stemmed in each catalog
  language (matches descriptions), with a word dropped when it is a stopword
  in any of those languages. The lexemes are then combined with OR, like
  BM25 over a bag of words. The default websearch_to_tsquery would AND every
  term, and long natural-language queries would then match nothing.

A work with descriptions in several languages has several documents; it is
ranked by its best one. Each search is one SQL statement (one round trip)
that returns fully hydrated rows. Filters are applied in SQL; if the BM25
index scan cannot supply enough filtered results, the search is repeated
once with the index scan off (exact scoring over every filtered row).

Index maintenance belongs to scripts.load_postgres, so build_index,
add_to_index, save_index and load_index are not supported here.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from app.domain.entities import SearchResult
from app.domain.ports import LexicalSearchRepository
from app.domain.value_objects import SearchFilters
from app.infrastructure.db.postgres_connection import PostgresDatabase
from app.infrastructure.db.postgres_rows import SIMPLE_CONFIG, WORK_COLUMNS, filter_condition, row_to_book
from app.infrastructure.db.postgres_schema import BM25_INDEX, read_catalog_info, validate_schema_name

logger = logging.getLogger(__name__)

RANKINGS = ("bm25", "ts_rank_cd")

# ts_rank_cd normalisation flag 1: divide the rank by 1 + log(document length).
TS_RANK_NORMALIZATION = 1

# Rows the BM25 index hands back before filters when filters are set.
FILTERED_BM25_LIMIT = 1000


class PostgresLexicalSearchRepository(LexicalSearchRepository):
    """
    Lexical search served by Postgres.

    Args:
        db: Database (the pooled connection string in the API)
        schema: Catalog schema
        ranking: "bm25" or "ts_rank_cd"
    """

    def __init__(self, db: PostgresDatabase, schema: str, ranking: str = "bm25") -> None:
        if ranking not in RANKINGS:
            raise ValueError(f"ranking must be one of {RANKINGS}, got '{ranking}'")
        self._db = db
        self._schema = validate_schema_name(schema)
        self._ranking = ranking
        self._info: Optional[Dict[str, Any]] = None
        self._info_lock = threading.Lock()

    @property
    def ranking(self) -> str:
        return self._ranking

    # ------------------------------------------------------------------
    # Catalog facts (read once)
    # ------------------------------------------------------------------

    def _catalog_info(self) -> Dict[str, Any]:
        if self._info is None:
            with self._info_lock:
                if self._info is None:
                    info = read_catalog_info(self._db, self._schema)
                    if not info.get("text_configs"):
                        raise RuntimeError(
                            f"Catalog schema '{self._schema}' is empty or missing: "
                            f"load it with python -m scripts.load_postgres"
                        )
                    if self._ranking == "bm25" and info.get("lexical_index") != "lakebase_bm25":
                        raise RuntimeError(
                            f"Schema '{self._schema}' has no BM25 index (lakebase_text); "
                            f"use PG_LEXICAL_RANKING=ts_rank_cd or rebuild the indexes"
                        )
                    self._info = info
        return self._info

    def query_configs(self) -> List[str]:
        """Text-search configurations the query is parsed with (simple first)."""
        configs = list(self._catalog_info()["text_configs"])
        simple = f"{self._schema}.{SIMPLE_CONFIG}"
        return [simple] + sorted(c for c in configs if c != simple)

    def _texts_per_work(self) -> int:
        return max(1, int(self._catalog_info().get("max_texts_per_work", 1)))

    # ------------------------------------------------------------------
    # SQL
    # ------------------------------------------------------------------

    def build_sql(
        self, max_results: int, filters: Optional[SearchFilters], *, exact_bm25: bool = False
    ) -> tuple[str, Dict[str, Any], Dict[str, object]]:
        """
        The search statement, its parameters (except the query text) and its
        transaction-local settings.

        The documents of a work are ranked together and the work keeps its
        best one, so up to max_results * max_texts_per_work documents are
        taken before grouping (enough for max_results distinct works).
        """
        s = self._schema
        condition, params = filter_condition(filters, "w")
        filtered = condition != "TRUE"
        candidates = max_results * self._texts_per_work()
        params.update({"configs": self.query_configs(), "k": max_results, "n": candidates})
        settings: Dict[str, object] = {}
        join = f"JOIN {s}.works w ON w.id = t.work_id AND {condition}" if filtered else ""

        if self._ranking == "bm25":
            params["bm25_index"] = f"{s}.{BM25_INDEX}"
            # The index returns its top `default_limit` rows before SQL filters.
            settings["lakebase_bm25.default_limit"] = max(candidates, FILTERED_BM25_LIMIT) if filtered else candidates
            if exact_bm25:
                settings["lakebase_bm25.enable_scan"] = "off"
            hits = f"""
  SELECT t.work_id, t.tsv <@> to_bm25query(
           array_to_tsvector((SELECT lexemes FROM q)), %(bm25_index)s::regclass) AS score
  FROM {s}.work_texts t {join}"""
            # BM25 scores from <@> are negative: lower is better, 0 = no match.
            # A work keeps its best document: the lowest (most negative) score.
            ranked = f"""
ranked AS (
  SELECT work_id, score FROM ({hits}) AS scored
  WHERE score < 0
  ORDER BY score, work_id
  LIMIT %(n)s
),
best AS (
  SELECT work_id, -min(score) AS score FROM ranked GROUP BY work_id
)"""
        else:
            ranked = f"""
best AS (
  SELECT t.work_id, max(ts_rank_cd(t.tsv, (SELECT tsq FROM q), {TS_RANK_NORMALIZATION})) AS score
  FROM {s}.work_texts t {join}
  WHERE t.tsv @@ (SELECT tsq FROM q)
  GROUP BY t.work_id
)"""
        sql = f"""
WITH q AS MATERIALIZED (
  SELECT lexemes, {s}.or_tsquery(lexemes) AS tsq
  FROM (SELECT {s}.query_lexemes(%(query)s, %(configs)s::regconfig[]) AS lexemes) AS l
),
{ranked}
SELECT {WORK_COLUMNS}, best.score AS score
FROM best JOIN {s}.works w ON w.id = best.work_id
ORDER BY best.score DESC, w.id
LIMIT %(k)s"""
        return sql, params, settings

    # ------------------------------------------------------------------
    # Port
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """
        Rank works for the query (see the module docstring).

        Returns:
            SearchResult list (source "lexical", 1-based rank, final_score =
            lexical_score = BM25 score or ts_rank_cd value; higher is better)

        Raises:
            ValueError: If query_text is empty
            RuntimeError: If the database is unavailable or the catalog is not loaded
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")
        if max_results < 1:
            return []
        rows = self._run(query_text, max_results, filters, exact_bm25=False)
        if (
            self._ranking == "bm25"
            and filters is not None
            and not filters.is_empty()
            and len(rows) < max_results
        ):
            # The index scan returned too few rows that pass the filters:
            # score every filtered document instead (exact, slower).
            rows = self._run(query_text, max_results, filters, exact_bm25=True)
        return [
            SearchResult(
                book=row_to_book(row),
                final_score=float(row["score"]),
                rank=rank,
                source="lexical",
                lexical_score=float(row["score"]),
                vector_score=None,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def _run(self, query_text: str, max_results: int, filters: Optional[SearchFilters], *, exact_bm25: bool):
        sql, params, settings = self.build_sql(max_results, filters, exact_bm25=exact_bm25)
        params["query"] = query_text
        return self._db.query(sql, params, settings=settings)

    def is_ready(self) -> bool:
        """True once the catalog facts were read (no query per search after that)."""
        try:
            self._catalog_info()
            return True
        except Exception as error:  # noqa: BLE001 - health check must not raise
            logger.warning("Postgres lexical search not ready: %s", error)
            return False

    def build_index(self, books) -> None:
        raise NotImplementedError(
            "The cloud profile's lexical index lives in Postgres: build it with python -m scripts.load_postgres"
        )

    def add_to_index(self, book) -> None:
        raise NotImplementedError(
            "Add books with PostgresBookCatalogRepository.save(); rebuild the BM25 index with scripts.load_postgres"
        )

    def save_index(self, path: str) -> None:
        raise NotImplementedError("The Postgres lexical index is stored in the database, not in a file")

    def load_index(self, path: str) -> None:
        raise NotImplementedError("The Postgres lexical index is stored in the database, not in a file")
