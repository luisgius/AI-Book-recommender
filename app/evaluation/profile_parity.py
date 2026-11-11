"""
Profile parity (task V12): what changes when search is served from Postgres.

The local profile (SQLite, BM25 with the normalizing tokenizer, exact FAISS)
is the evaluated system. The cloud profile serves the same catalog from
Postgres (Neon): BM25 from the lakebase_text index (or ts_rank_cd), pgvector
halfvec with an exact mode and an ANN index. This module measures how the
rankings differ when everything else is identical: the same 177 books, the
same document vectors (copied from the local multilingual-e5 FAISS index),
the same query vectors (each query is encoded once and given to both
profiles) and the same fusion (RRF in the unchanged SearchService).

Design (fixed before any result was seen):
- Queries: the 44 test queries, depth 10 (as in production and V06).
- Local systems (retrieval_ablation definitions): bm25_normalized, vec_e5,
  hybrid_norm_e5.
- Cloud systems: pg_bm25 (lakebase_text BM25, k1 1.5, b 0.75),
  pg_fts_tsrank (ts_rank_cd), pg_vec_exact (index scans off), pg_vec_ann
  (the schema's ANN index), optionally pg_vec_hnsw (pgvector HNSW in a
  second schema), and pg_hybrid = RRF(pg_bm25, pg_vec_exact) through
  SearchService.
- Three pre-declared contrasts on per-query nDCG@10 (CONTRASTS): lexical
  pg_bm25 vs bm25_normalized, vector pg_vec_exact vs vec_e5, hybrid
  pg_hybrid vs hybrid_norm_e5. Two-sided paired randomisation test, Holm
  across exactly these three, paired bootstrap 95% interval (10,000
  resamples, seed 0), as in V06.
- Descriptive only: pg_fts_tsrank and the ANN systems, top-10 overlaps,
  identical lists, recall@10 of the ANN indexes against exact search,
  AND versus OR tsquery coverage, latencies, sizes and build times.
- Judgments: start from the GLM-5.3 Flash extended judgments; the
  (query, book) pairs of any top 10 still unjudged are graded by the same
  judge (cached), and the merged file is written to
  app/evaluation/relevance_judgments_extended_cloud.json. Judged@10 must be
  1.0 for every system.

CLI (from the repository root):

    python -m app.evaluation.profile_parity run --schema eval177 --hnsw-schema eval177_hnsw
    python -m app.evaluation.profile_parity run --schema eval177 --no-judge
    python -m app.evaluation.profile_parity report      # tables from the saved rankings, no database
    python -m app.evaluation.profile_parity scale --schema catalog_small   # sizes, builds, ANN recall
    python -m app.evaluation.profile_parity cold-start --schema eval177 --note "..."

`run` needs the database, the embedding model and (unless --no-judge) the
judge; `report` needs none of them. Outputs:
data/evaluation/results_profile_parity.json and data/evaluation/PROFILE_PARITY.md.
"""

import argparse
import json
import statistics as stats_module
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from app.domain.entities import Book
from app.evaluation.comparison import PRIMARY_METRIC, _git, _sha256, compare_pairs, compare_systems
from app.evaluation.evaluation_service import BookId
from app.evaluation.retrieval_ablation import (
    DEPTH,
    E5,
    SYSTEMS as ABLATION_SYSTEMS,
    Searcher,
    SystemSpec,
    collect_run,
    latency_summary,
    make_searcher,
    require_judged,
    resolve_grades,
)
from app.evaluation.types import TestQuery

DEFAULT_DB_PATH = "data/catalog.db"
DEFAULT_QUERIES_PATH = "app/evaluation/test_queries.json"
DEFAULT_BASE_JUDGMENTS = "app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json"
DEFAULT_CLOUD_JUDGMENTS = "app/evaluation/relevance_judgments_extended_cloud.json"
DEFAULT_CACHE_DIR = "data/evaluation/llm_judgments"
DEFAULT_E5_INDEX = "data/indexes_ablation/intfloat_multilingual-e5-large-instruct/faiss_index"
DEFAULT_OUTPUT_JSON = "data/evaluation/results_profile_parity.json"
DEFAULT_OUTPUT_MD = "data/evaluation/PROFILE_PARITY.md"

# Free plan of the Neon project (neon.com/docs/introduction/plans, checked
# 2026-11-11; the project reports branch_logical_size_limit_bytes = 1 GiB).
FREE_PLAN_STORAGE_BYTES = 1024 ** 3

LOCAL_SYSTEMS = ("bm25_normalized", "vec_e5", "hybrid_norm_e5")


# =============================================================================
# The design
# =============================================================================


PG_SYSTEMS: Tuple[SystemSpec, ...] = (
    SystemSpec("pg_bm25", lexical="lakebase_bm25",
               description="Postgres BM25 (lakebase_text index, k1 1.5, b 0.75), unaccent + per-language stemming"),
    SystemSpec("pg_fts_tsrank", lexical="ts_rank_cd",
               description="Postgres full-text search ranked with ts_rank_cd (no IDF), same documents and query"),
    SystemSpec("pg_vec_exact", embedding=E5,
               description="pgvector halfvec, cosine, index scans disabled (exact)"),
    SystemSpec("pg_vec_ann", embedding=E5,
               description="pgvector halfvec, cosine, the schema's ANN index (approximate)"),
    SystemSpec("pg_vec_hnsw", embedding=E5,
               description="pgvector halfvec, cosine, pgvector HNSW index (approximate; second schema)"),
    SystemSpec("pg_hybrid", lexical="lakebase_bm25", embedding=E5,
               description="RRF(pg_bm25, pg_vec_exact) through the unchanged SearchService"),
)

# (name, system, reference): the pre-declared family (Holm across these three).
CONTRASTS: Tuple[Tuple[str, str, str], ...] = (
    ("lexical", "pg_bm25", "bm25_normalized"),
    ("vector_exact", "pg_vec_exact", "vec_e5"),
    ("hybrid", "pg_hybrid", "hybrid_norm_e5"),
)

# Derived run (no search of its own): the local BM25 list without its
# zero-score tail. rank_bm25 scores every book, so the local list always has
# 10 entries, padded with books that share no term with the query; Postgres
# returns only matching works. This run separates the two effects.
MATCHES_ONLY = "bm25_normalized_matches"

# Local hybrid whose BM25 leg drops zero-score books before RRF (descriptive):
# separates the effect of the padding on the fusion from the engines.
HYBRID_MATCHES_ONLY = SystemSpec(
    "hybrid_norm_e5_matches", lexical="normalized", embedding=E5,
    description="descriptive: RRF(bm25_normalized without zero-score books, vec_e5) through SearchService",
)

# Systems whose scores are kept per query (score differences, MATCHES_ONLY).
SCORED_SYSTEMS = ("bm25_normalized", "pg_bm25", "vec_e5", "pg_vec_exact")

# Descriptive comparisons (no test).
DESCRIPTIVE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("pg_bm25", MATCHES_ONLY),
    ("pg_fts_tsrank", "bm25_normalized"),
    ("pg_vec_ann", "vec_e5"),
    ("pg_vec_hnsw", "vec_e5"),
    ("pg_hybrid", HYBRID_MATCHES_ONLY.name),
    (HYBRID_MATCHES_ONLY.name, "hybrid_norm_e5"),
)

# Top-10 overlap pairs: (system, reference).
OVERLAP_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("pg_bm25", "bm25_normalized"),
    ("pg_bm25", MATCHES_ONLY),
    ("pg_fts_tsrank", "bm25_normalized"),
    ("pg_bm25", "pg_fts_tsrank"),
    ("pg_vec_exact", "vec_e5"),
    ("pg_vec_ann", "pg_vec_exact"),
    ("pg_vec_hnsw", "pg_vec_exact"),
    ("pg_hybrid", "hybrid_norm_e5"),
    ("pg_hybrid", HYBRID_MATCHES_ONLY.name),
)


def local_specs() -> List[SystemSpec]:
    """The local systems, exactly as declared in retrieval_ablation.SYSTEMS."""
    by_name = {spec.name: spec for spec in ABLATION_SYSTEMS}
    return [by_name[name] for name in LOCAL_SYSTEMS]


# =============================================================================
# Pure analysis (shared by `run` and `report`)
# =============================================================================


def overlap(a: Sequence[Any], b: Sequence[Any], k: int = DEPTH) -> float:
    """|top-k(a) ∩ top-k(b)| / max(|top-k(a)|, |top-k(b)|); two empty lists overlap fully."""
    top_a, top_b = list(a)[:k], list(b)[:k]
    size = max(len(top_a), len(top_b))
    return 1.0 if size == 0 else len(set(top_a) & set(top_b)) / size


def recall_against(approx: Sequence[Any], exact: Sequence[Any], k: int = DEPTH) -> float:
    """Share of the exact top k that the approximate top k also returns."""
    truth = list(exact)[:k]
    return 1.0 if not truth else len(set(list(approx)[:k]) & set(truth)) / len(truth)


def overlap_report(
    runs: Mapping[str, Mapping[str, Sequence[Any]]], pairs: Sequence[Tuple[str, str]], k: int = DEPTH
) -> Dict[str, Dict[str, Any]]:
    """Per pair present in runs: mean overlap, recall against the reference, identical lists and sets."""
    out: Dict[str, Dict[str, Any]] = {}
    for system, reference in pairs:
        if system not in runs or reference not in runs:
            continue
        qids = [q for q in runs[reference] if q in runs[system]]
        overlaps = [overlap(runs[system][q], runs[reference][q], k) for q in qids]
        recalls = [recall_against(runs[system][q], runs[reference][q], k) for q in qids]
        out[f"{system} vs {reference}"] = {
            "system": system,
            "reference": reference,
            "num_queries": len(qids),
            "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
            "mean_recall_of_reference": sum(recalls) / len(recalls) if recalls else None,
            "identical_lists": sum(1 for q in qids if list(runs[system][q])[:k] == list(runs[reference][q])[:k]),
            "identical_sets": sum(1 for q in qids if set(list(runs[system][q])[:k]) == set(list(runs[reference][q])[:k])),
            "per_query_overlap": dict(zip(qids, overlaps)),
        }
    return out


def score_differences(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]], system: str, reference: str
) -> Optional[Dict[str, float]]:
    """Largest and mean absolute score difference over the books both return (same query)."""
    if system not in scores or reference not in scores:
        return None
    diffs = []
    for qid, by_book in scores[reference].items():
        other = scores[system].get(qid, {})
        diffs.extend(abs(other[book] - value) for book, value in by_book.items() if book in other)
    if not diffs:
        return None
    return {"pairs": len(diffs), "max_abs_difference": max(diffs), "mean_abs_difference": sum(diffs) / len(diffs)}


def matches_only_run(
    runs: Mapping[str, Mapping[str, Sequence[str]]], scores: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> Optional[Dict[str, List[str]]]:
    """bm25_normalized without its zero-score results (see MATCHES_ONLY), or None without scores."""
    if "bm25_normalized" not in runs or "bm25_normalized" not in scores:
        return None
    by_query = scores["bm25_normalized"]
    return {
        qid: [book for book in ranked if by_query.get(qid, {}).get(str(book), 0.0) > 0.0]
        for qid, ranked in runs["bm25_normalized"].items()
    }


def analyse(
    runs: Mapping[str, Mapping[str, Sequence[BookId]]],
    queries: Sequence[TestQuery],
    judgments: Mapping[str, Mapping[BookId, int]],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Metrics, the three contrasts, descriptive differences and overlaps.

    Every returned top-10 result must have a judgment. Judged@10 divides by
    10, so a system that returns fewer than 10 works for a query (Postgres
    full-text search returns only matching works) scores below 1.0 there
    with no unjudged result; list lengths are reported next to it.

    Raises:
        UnjudgedResultsError: If a returned top-10 result has no judgment
    """
    require_judged(runs, judgments, DEPTH)
    report = compare_systems(runs, judgments, queries, None, n_resamples=n_resamples, seed=seed)
    query_ids = report["query_ids"]
    for name, data in report["systems"].items():
        lengths = [min(len(runs[name][q]), DEPTH) for q in query_ids]
        returned = sum(lengths)
        judged = sum(1 for q in query_ids for b in list(runs[name][q])[:DEPTH] if b in judgments.get(q, {}))
        data["list_lengths"] = {
            "mean": sum(lengths) / len(lengths),
            "queries_with_fewer_than_10": sum(1 for n in lengths if n < DEPTH),
            "queries_with_no_result": sum(1 for n in lengths if n == 0),
            "judged_share_of_returned": judged / returned if returned else 1.0,
        }
    contrasts = compare_pairs(report, [c for c in CONTRASTS if c[1] in runs and c[2] in runs],
                              n_resamples=n_resamples, seed=seed)
    descriptive = {}
    for system, reference in DESCRIPTIVE_PAIRS:
        if system in runs and reference in runs:
            one = compare_pairs(report, [("d", system, reference)], n_resamples=n_resamples, seed=seed)["d"]
            one.pop("p_value_holm", None)
            descriptive[f"{system} vs {reference}"] = one
    return {
        "num_queries": report["num_queries"],
        "languages": report["languages"],
        "queries_without_relevant": report["queries_without_relevant"],
        "systems": report["systems"],
        "contrasts": contrasts,
        "descriptive": descriptive,
        "overlaps": overlap_report(runs, OVERLAP_PAIRS),
    }


# =============================================================================
# Running the systems (database and models)
# =============================================================================


class CachedQueryEmbeddings:
    """
    EmbeddingsStore wrapper that encodes each query text once.

    Both profiles then receive the very same query vector, and search
    latencies exclude the model (which is the same in both profiles).
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._cache: Dict[str, List[float]] = {}

    def generate_query_embedding(self, text: str) -> List[float]:
        if text not in self._cache:
            self._cache[text] = self._store.generate_query_embedding(text)
        return self._cache[text]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


class MatchesOnlyLexical:
    """A lexical repository without its zero-score results (ranks renumbered)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def search(self, query_text: str, max_results: int = 10, filters: Any = None) -> List[Any]:
        kept = [r for r in self._inner.search(query_text, max_results=max_results, filters=filters)
                if (r.lexical_score or 0.0) > 0.0]
        for rank, result in enumerate(kept, start=1):
            result.rank = rank
        return kept

    def is_ready(self) -> bool:
        return self._inner.is_ready()


def with_scores(search: Searcher, scores: Dict[str, Dict[str, float]], texts: Mapping[str, str]) -> Searcher:
    """Wrap a searcher so the final_score of every result is kept per query id."""
    by_text = {text: qid for qid, text in texts.items()}

    def wrapped(text: str):
        results = search(text)
        if text in by_text:
            scores[by_text[text]] = {str(r.book.id): float(r.final_score) for r in results}
        return results

    return wrapped


def tsquery_coverage(db: Any, schema: str, configs: Sequence[str], texts: Mapping[str, str]) -> Dict[str, Any]:
    """
    How many works each query matches with an AND query versus the OR query.

    AND: websearch_to_tsquery (every term required) under each configuration,
    a work matching in any of them. OR: the adapter's query (any lexeme).
    """
    and_query = " || ".join(f"websearch_to_tsquery(%(c{i})s::regconfig, %(q)s)" for i in range(len(configs)))
    params_base = {f"c{i}": c for i, c in enumerate(configs)}
    counts: Dict[str, Dict[str, int]] = {}
    for qid, text in texts.items():
        row = db.query(
            f"""SELECT
                  (SELECT count(DISTINCT work_id) FROM {schema}.work_texts WHERE tsv @@ ({and_query})) AS and_matches,
                  (SELECT count(DISTINCT work_id) FROM {schema}.work_texts
                   WHERE tsv @@ {schema}.or_tsquery({schema}.query_lexemes(%(q)s, %(configs)s::regconfig[]))) AS or_matches""",
            {**params_base, "q": text, "configs": list(configs)},
        )[0]
        counts[qid] = {"and": int(row["and_matches"]), "or": int(row["or_matches"])}

    def summary(key: str) -> Dict[str, Any]:
        values = [c[key] for c in counts.values()]
        return {
            "queries_with_no_match": sum(1 for v in values if v == 0),
            "queries_with_fewer_than_10": sum(1 for v in values if v < DEPTH),
            "median_matches": stats_module.median(values) if values else None,
        }

    return {"and": summary("and"), "or": summary("or"), "per_query": counts}


def _database():
    from dotenv import load_dotenv

    from app.infrastructure.config.settings import database_url
    from app.infrastructure.db.postgres_connection import PostgresDatabase

    load_dotenv(Path(".env"))
    return PostgresDatabase(database_url())


def _load_e5(device: Optional[str], index_dir: str, log: Callable[[str], None]):
    from app.evaluation.retrieval_ablation import resolve_device

    device = device or resolve_device()  # imports torch before faiss (macOS)
    from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss

    log(f"Loading {E5} on {device}")
    store = EmbeddingsStoreFaiss(model_name=E5, device=device)
    if index_dir:
        store.load_index(index_dir)
    return store


def schema_facts(db: Any, schema: str) -> Dict[str, Any]:
    """catalog_info, relation sizes, schema size and stored bytes per column of a loaded schema."""
    from app.infrastructure.db.postgres_schema import read_catalog_info, relation_sizes, schema_total_bytes

    columns = db.query(
        f"""SELECT
              (SELECT sum(pg_column_size(embedding)) FROM {schema}.works) AS works_embedding,
              (SELECT sum(pg_column_size(metadata)) FROM {schema}.works) AS works_metadata,
              (SELECT sum(pg_column_size(description)) FROM {schema}.works) AS works_description,
              (SELECT sum(pg_column_size(body)) FROM {schema}.work_texts) AS work_texts_body,
              (SELECT sum(pg_column_size(tsv)) FROM {schema}.work_texts) AS work_texts_tsv"""
    )[0]
    return {
        "catalog_info": read_catalog_info(db, schema),
        "relations": relation_sizes(db, schema),
        "schema_total_bytes": schema_total_bytes(db, schema),
        "column_bytes": {key: int(value or 0) for key, value in columns.items()},
    }


def run_measurement(args: argparse.Namespace, log: Callable[[str], None]) -> Dict[str, Any]:
    """`run`: every system on the 44 queries, judging, statistics, latencies, sizes."""
    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.infrastructure.db.postgres_schema import server_versions
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

    started = datetime.now(UTC)
    store = _load_e5(args.device, args.e5_index, log)  # torch before faiss
    from app.infrastructure.search.bm25_search_repository import BM25SearchRepository
    from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository
    from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository
    from app.infrastructure.search.postgres_lexical_search_repository import PostgresLexicalSearchRepository

    books = SqliteBookCatalogRepository(Path(args.db_path)).get_all()
    queries = load_test_queries(args.queries_path)
    texts = {q.query_id: q.text for q in queries}
    base_raw = load_relevance_judgments(args.base_judgments)
    base_sha = _sha256(args.base_judgments)
    embeddings = CachedQueryEmbeddings(store)
    for text in [queries[0].text, *texts.values()]:
        embeddings.generate_query_embedding(text)  # encode once, outside the timed searches

    db = _database()
    try:
        bm25_local = BM25SearchRepository(tokenizer="normalized")
        bm25_local.build_index(books)
        faiss_vector = FaissVectorSearchRepository(store, books)
        pg_lexical = {"pg_bm25": PostgresLexicalSearchRepository(db, args.schema, "bm25"),
                      "pg_fts_tsrank": PostgresLexicalSearchRepository(db, args.schema, "ts_rank_cd")}
        pg_vector = {"pg_vec_exact": PgVectorSearchRepository(db, args.schema, "exact", expected_model=E5),
                     "pg_vec_ann": PgVectorSearchRepository(db, args.schema, "approximate", expected_model=E5)}
        if args.hnsw_schema:
            pg_vector["pg_vec_hnsw"] = PgVectorSearchRepository(db, args.hnsw_schema, "approximate", expected_model=E5)

        searchers: Dict[str, Searcher] = {}
        for spec in local_specs():
            searchers[spec.name] = make_searcher(
                spec, lexical=bm25_local, vector=faiss_vector, embeddings=embeddings)
        searchers[HYBRID_MATCHES_ONLY.name] = make_searcher(
            HYBRID_MATCHES_ONLY, lexical=MatchesOnlyLexical(bm25_local), vector=faiss_vector, embeddings=embeddings)
        for spec in PG_SYSTEMS:
            if spec.name in pg_lexical:
                searchers[spec.name] = make_searcher(spec, lexical=pg_lexical[spec.name])
            elif spec.name in pg_vector:
                searchers[spec.name] = make_searcher(spec, vector=pg_vector[spec.name], embeddings=embeddings)
            elif spec.name == "pg_hybrid":
                searchers[spec.name] = make_searcher(
                    spec, lexical=pg_lexical["pg_bm25"], vector=pg_vector["pg_vec_exact"], embeddings=embeddings)

        runs: Dict[str, Dict[str, List[BookId]]] = {}
        latency: Dict[str, Dict[str, Any]] = {}
        scores: Dict[str, Dict[str, Dict[str, float]]] = {}
        for name, search in searchers.items():
            log(f"Running {name}")
            if name in SCORED_SYSTEMS:
                scores[name] = {}
                search = with_scores(search, scores[name], texts)
            runs[name], per_query_ms, warmup_ms = collect_run(search, texts, warmup_text=queries[0].text)
            latency[name] = {**latency_summary(list(per_query_ms.values())), "warmup_ms": warmup_ms}

        round_trips = []
        for _ in range(20):  # baseline: one trivial statement through the same pool
            t0 = time.perf_counter()
            db.query("SELECT 1 AS one")
            round_trips.append((time.perf_counter() - t0) * 1000)
        latency["SELECT 1 (network round trip)"] = {**latency_summary(round_trips), "warmup_ms": None}
        coverage = tsquery_coverage(db, args.schema, pg_lexical["pg_bm25"].query_configs(), texts)
        facts = {"schema": args.schema, **schema_facts(db, args.schema)}
        hnsw_facts = {"schema": args.hnsw_schema, **schema_facts(db, args.hnsw_schema)} if args.hnsw_schema else None
        versions = server_versions(db)
    finally:
        db.close()

    # Judgments: the base file, plus the judge's grades of the remaining holes.
    from app.evaluation.relevance_judge import JudgeTask, JudgmentCache, find_unjudged, merge_judgments

    current = resolve_grades(base_raw, books)
    holes = find_unjudged(runs, current, DEPTH)
    books_by_id = {book.id: book for book in books}
    judging: Dict[str, Any] = {"holes_at_start": len(holes)}
    if holes and args.no_judge:
        raise SystemExit(f"{len(holes)} top-10 pairs have no judgment; run without --no-judge to grade them")
    new_judgments = []
    if holes:
        from app.evaluation.judge_sensitivity import judge_in_parallel
        from app.evaluation.relevance_judge import _build_judge
        from app.infrastructure.llm.relevance_judge import LangChainRelevanceJudge

        settings, judge = _build_judge()
        base_models = {item.get("model") for items in base_raw.values() for item in items if item.get("judge") == "llm"}
        if base_models and judge.model_name not in base_models:
            raise SystemExit(f"The base judgments were graded by {sorted(base_models)}, the judge is {judge.model_name}")
        judges = [judge] + [LangChainRelevanceJudge.from_settings(settings) for _ in range(args.workers - 1)]
        cache = JudgmentCache.for_judge(args.cache_dir, judge.model_name, judge.prompt_version)
        tasks = [JudgeTask(qid, texts[qid], books_by_id[book_id]) for qid, book_id in holes]
        result = judge_in_parallel(tasks, judges, cache)
        if result.remaining:
            raise SystemExit(f"{result.remaining} pairs still without a judgment; run again")
        new_judgments = [cache.get(task.key) for task in tasks]
        judging.update({
            "judge_model": judge.model_name, "prompt_version": judge.prompt_version,
            "judged_in_this_run": result.judged, "already_cached": result.cached,
            "seconds_per_judgment": result.seconds_per_judgment, "cache_file": str(cache.path),
            "grade_distribution": [sum(1 for j in new_judgments if j.grade == g) for g in range(4)],
        })
    merged = merge_judgments(base_raw, new_judgments)
    Path(args.cloud_judgments).write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote {args.cloud_judgments} ({sum(len(e['judgments']) for e in merged)} judgments, {len(new_judgments)} new)")

    return {
        "date": started.isoformat(timespec="seconds"),
        "command": "python -m app.evaluation.profile_parity " + " ".join(sys.argv[1:]),
        "git_commit": _git("rev-parse", "HEAD"),
        "schema": args.schema,
        "hnsw_schema": args.hnsw_schema,
        "database": versions,
        "judgments": {
            "base_file": args.base_judgments, "base_sha256": base_sha,
            "cloud_file": args.cloud_judgments, **judging,
        },
        "runs": {s: {q: [str(b) for b in run[q]] for q in run} for s, run in runs.items()},
        "scores": scores,
        "latency": latency,
        "tsquery_coverage": coverage,
        "schema_facts": facts,
        "hnsw_schema_facts": hnsw_facts,
    }


def ann_recall(args: argparse.Namespace, log: Callable[[str], None]) -> Dict[str, Any]:
    """`scale`: sizes, build times and ANN recall@10 against exact search on a (larger) schema."""
    from app.evaluation.evaluation_job import load_test_queries
    from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository
    from app.infrastructure.search.postgres_lexical_search_repository import PostgresLexicalSearchRepository

    store = _load_e5(args.device, "", log)
    queries = load_test_queries(args.queries_path)
    vectors = {q.query_id: store.generate_query_embedding(q.text) for q in queries}
    db = _database()
    try:
        exact = PgVectorSearchRepository(db, args.schema, "exact", expected_model=E5)
        approx = PgVectorSearchRepository(db, args.schema, "approximate", expected_model=E5)
        bm25 = PostgresLexicalSearchRepository(db, args.schema, "bm25")
        index_type = approx.index_type()
        timings: Dict[str, List[float]] = {"exact": [], "approximate": [], "bm25": []}
        runs: Dict[str, Dict[str, List[str]]] = {"exact": {}, "approximate": {}}
        exact.search(vectors[queries[0].query_id], DEPTH)  # warm-up
        approx.search(vectors[queries[0].query_id], DEPTH)
        bm25.search(queries[0].text, DEPTH)
        for q in queries:
            for name, repo in (("exact", exact), ("approximate", approx)):
                t0 = time.perf_counter()
                results = repo.search(vectors[q.query_id], DEPTH)
                timings[name].append((time.perf_counter() - t0) * 1000)
                runs[name][q.query_id] = [str(r.book.id) for r in results]
            t0 = time.perf_counter()
            bm25.search(q.text, DEPTH)
            timings["bm25"].append((time.perf_counter() - t0) * 1000)
        recalls = [recall_against(runs["approximate"][q], runs["exact"][q]) for q in runs["exact"]]
        facts = schema_facts(db, args.schema)
    finally:
        db.close()
    source = Path(str(facts["catalog_info"].get("source_catalog", "")))
    return {
        "source_catalog_sha256": _sha256(str(source)) if source.is_file() else None,
        "schema": args.schema,
        "vector_index": index_type,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "recall_at_10": sum(recalls) / len(recalls),
        "queries_with_full_recall": sum(1 for r in recalls if r == 1.0),
        "num_queries": len(recalls),
        "latency": {name: latency_summary(values) for name, values in timings.items()},
        **facts,
    }


def cold_start(args: argparse.Namespace, log: Callable[[str], None]) -> Dict[str, Any]:
    """
    `cold-start`: time one hybrid search on a new connection, then again warm.

    Run it after the compute has been idle long enough to be suspended
    (Neon free plan: 5 minutes); the note records how that was ensured.
    """
    from app.domain.services import SearchService
    from app.domain.value_objects import SearchQuery
    from app.evaluation.evaluation_job import load_test_queries
    from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository
    from app.infrastructure.search.postgres_lexical_search_repository import PostgresLexicalSearchRepository

    from app.infrastructure.search.postgres_embeddings_store import PostgresEmbeddingsStore

    encoder = _load_e5(args.device, "", log)
    text = load_test_queries(args.queries_path)[0].text
    vector = encoder.generate_query_embedding(text)

    class PrecomputedEncoder:  # the query vector is computed before the clock starts
        def generate_query_embedding(self, query: str):
            return vector

        def get_dimension(self) -> int:
            return len(vector)

    def make_service(db) -> SearchService:
        """The cloud profile's adapters as the API wires them (fresh: no catalog facts read yet)."""
        return SearchService(
            lexical_search=PostgresLexicalSearchRepository(db, args.schema, "bm25"),
            vector_search=PgVectorSearchRepository(db, args.schema, "exact", expected_model=E5),
            embeddings_store=PostgresEmbeddingsStore(PrecomputedEncoder(), db, args.schema),  # type: ignore[arg-type]
        )

    def search(service: SearchService) -> float:
        t0 = time.perf_counter()
        response = service.search_with_fallback(SearchQuery(text=text, max_results=DEPTH))
        elapsed = (time.perf_counter() - t0) * 1000
        if response.degraded:
            raise RuntimeError(f"search degraded: {response.degradation_reason}")
        return elapsed

    def first_result(db) -> Tuple[float, float]:
        """(open a connection and run SELECT 1, then the first search with fresh adapters), in ms."""
        t0 = time.perf_counter()
        db.query("SELECT 1 AS one")
        connect_ms = (time.perf_counter() - t0) * 1000
        return connect_ms, search(make_service(db))

    log("First statement on a new connection (cold if the compute is suspended)")
    first_db = _database()
    try:
        cold_connect_ms, cold_search_ms = first_result(first_db)
        service = make_service(first_db)
        search(service)
        warm_reused_ms = [search(service) for _ in range(5)]
    finally:
        first_db.close()
    second_db = _database()
    try:
        warm_connect_ms, warm_search_ms = first_result(second_db)
    finally:
        second_db.close()
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema": args.schema,
        "note": args.note,
        "cold_connect_and_select1_ms": cold_connect_ms,
        "cold_first_search_ms": cold_search_ms,
        "cold_time_to_first_result_ms": cold_connect_ms + cold_search_ms,
        "warm_connect_and_select1_ms": warm_connect_ms,
        "warm_first_search_ms": warm_search_ms,
        "warm_time_to_first_result_ms": warm_connect_ms + warm_search_ms,
        "warm_search_reused_adapters_median_ms": stats_module.median(warm_reused_ms),
        "what_is_timed": "pg_hybrid as the API wires it (BM25 + exact vector, RRF), one query, query vector "
                        "precomputed. 'first search' builds fresh adapters, so it includes reading the catalog "
                        "facts once (three small statements) besides the two search statements.",
    }


# =============================================================================
# Report
# =============================================================================


def build_report(saved: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Recompute every table from the saved rankings and the cloud judgments (no database)."""
    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

    books: List[Book] = SqliteBookCatalogRepository(Path(args.db_path)).get_all()
    queries = load_test_queries(args.queries_path)
    judgments = resolve_grades(load_relevance_judgments(args.cloud_judgments), books)
    runs = {s: {q: [UUID(b) for b in ids] for q, ids in run.items()} for s, run in saved["runs"].items()}
    matches_only = matches_only_run(saved["runs"], saved.get("scores", {}))
    if matches_only is not None:
        runs[MATCHES_ONLY] = {q: [UUID(b) for b in ids] for q, ids in matches_only.items()}
    analysis = analyse(runs, queries, judgments, n_resamples=args.n_resamples, seed=args.seed)
    analysis["score_differences"] = {
        "pg_vec_exact vs vec_e5": score_differences(saved.get("scores", {}), "pg_vec_exact", "vec_e5"),
    }
    return {**saved, "analysis": analysis, "judgments_cloud_sha256": _sha256(args.cloud_judgments)}


def _f(value: Optional[float], signed: bool = False, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(stat: Mapping[str, float]) -> str:
    return f"{_f(stat['mean'])} [{_f(stat['ci_low'])}, {_f(stat['ci_high'])}]"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _mib(value: Optional[int]) -> str:
    return "n/a" if value is None else f"{value / 1024 / 1024:.2f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _sizes_table(facts: Mapping[str, Any]) -> str:
    rows = []
    for item in facts.get("relations", []):
        if item["kind"] == "table":
            rows.append([f"table `{item['relation']}`", str(item["rows"]), _mib(item["total_bytes"]),
                         "heap + TOAST + indexes"])
        else:
            rows.append([f"index `{item['relation']}`", "", _mib(item["bytes"]), item["method"]])
    rows.append(["schema total", "", _mib(facts.get("schema_total_bytes")), ""])
    return _table(["Relation", "Rows", "MiB", "Note"], rows)


def capacity(scale: Mapping[str, Any], database_bytes: int, limit: int = FREE_PLAN_STORAGE_BYTES) -> Dict[str, Any]:
    """Works that fit in the free plan, extrapolated linearly from measured bytes per work."""
    works = scale["catalog_info"].get("works") or 0
    total = scale["schema_total_bytes"]
    ann = sum(i["bytes"] for i in scale["relations"] if i["kind"] == "index" and i["relation"] == "works_embedding_ann")
    free = limit - (database_bytes - total)  # what other schemas and the system catalogs already use stays
    per_work = total / works if works else None
    per_work_no_ann = (total - ann) / works if works else None
    return {
        "works": works,
        "bytes_per_work": per_work,
        "bytes_per_work_without_ann_index": per_work_no_ann,
        "storage_limit_bytes": limit,
        "available_bytes": free,
        "works_that_fit": int(free / per_work) if per_work else None,
        "works_that_fit_without_ann_index": int(free / per_work_no_ann) if per_work_no_ann else None,
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    a = report["analysis"]
    systems = a["systems"]
    out: List[str] = []
    w = out.append
    w("# Profile parity: local profile versus Postgres (V12)\n")
    w(f"- Date of the measurement: {report['date']} (git {str(report.get('git_commit'))[:7]})")
    db = report["database"]
    ext = db["extensions"]
    w(f"- Database: Neon (aws-eu-central-1, free plan), PostgreSQL {db['postgres']}; extensions: vector "
      f"{ext.get('vector')}, lakebase_text {ext.get('lakebase_text')}, lakebase_vector {ext.get('lakebase_vector')}, "
      f"unaccent {ext.get('unaccent')}")
    w(f"- Catalog: the 177-book evaluation collection in schema `{report['schema']}`"
      + (f" (and `{report['hnsw_schema']}` with an HNSW index)" if report.get("hnsw_schema") else ""))
    w(f"- Queries: {a['num_queries']} test queries ({', '.join(f'{k} {v}' for k, v in a['languages'].items())}), depth {DEPTH}")
    j = report["judgments"]
    w(f"- Judgments: `{j['cloud_file']}` = `{j['base_file']}` (sha256 {j['base_sha256'][:12]}...) plus "
      f"{j.get('judged_in_this_run', 0) + j.get('already_cached', 0)} pairs graded by {j.get('judge_model', 'the same judge')} "
      f"({j['holes_at_start']} holes at the start); every returned top-10 result of every system has a judgment\n")

    w("## Design (fixed before the run)\n")
    w("Same documents (the 177 books of `data/catalog.db`), same document vectors (copied from the local "
      "`multilingual-e5-large-instruct` FAISS index of the V06 `vec_e5` arm), same query vectors (each query "
      "encoded once and passed to both profiles), same fusion (RRF, k = 60, 20 candidates per leg, in the unchanged "
      "`SearchService`). Only the storage and the retrieval engines differ.\n")
    rows = [[f"`{s.name}`", s.description] for s in local_specs()] + [
        [f"`{s.name}`", s.description] for s in PG_SYSTEMS if s.name in systems]
    if MATCHES_ONLY in systems:
        rows.append([f"`{MATCHES_ONLY}`", "derived, descriptive: `bm25_normalized` without its zero-score results "
                                          "(rank_bm25 fills the list with books that share no term with the query)"])
    if HYBRID_MATCHES_ONLY.name in systems:
        rows.append([f"`{HYBRID_MATCHES_ONLY.name}`", HYBRID_MATCHES_ONLY.description])
    w(_table(["System", "Description"], rows) + "\n")
    w("Pre-declared contrasts (nDCG@10, system minus reference): " + "; ".join(
        f"{n}: `{s}` vs `{r}`" for n, s, r in CONTRASTS) + ". Two-sided paired randomisation test (sign flips), "
      "Holm correction across exactly these three; paired percentile bootstrap 95% interval (10,000 resamples, "
      "seed 0). Everything else is descriptive.\n")

    w("## Results\n")
    w("### Pre-declared contrasts\n")
    rows = []
    for name, c in a["contrasts"].items():
        rows.append([name, f"`{c['system']}`", f"`{c['reference']}`", _ci(systems[c["system"]]["metrics"][PRIMARY_METRIC]),
                     _ci(systems[c["reference"]]["metrics"][PRIMARY_METRIC]),
                     f"{_f(c['mean_difference'], True)} [{_f(c['ci_low'], True)}, {_f(c['ci_high'], True)}]",
                     _p(c["p_value"]), _p(c["p_value_holm"])])
    w(_table(["Contrast", "System", "Reference", "nDCG@10 system [95% CI]", "nDCG@10 reference [95% CI]",
              "Difference [95% CI]", "p", "p (Holm)"], rows) + "\n")

    w("### All systems (descriptive)\n")
    rows = []
    for name, data in systems.items():
        m = data["metrics"]
        langs = data["by_language"]
        lengths = data["list_lengths"]
        rows.append([f"`{name}`", _ci(m["ndcg_at_10"]), _f(m["recall_at_10"]["mean"]), _f(m["precision_at_5"]["mean"]),
                     _f(m["mrr"]["mean"]), _f(m["judged_at_10"]["mean"]), _f(lengths["judged_share_of_returned"]),
                     f"{lengths['mean']:.1f} ({lengths['queries_with_fewer_than_10']})",
                     *[_f(langs[lang][PRIMARY_METRIC]["mean"]) for lang in sorted(langs)]])
    langs = sorted(next(iter(systems.values()))["by_language"])
    w(_table(["System", "nDCG@10 [95% CI]", "Recall@10", "P@5", "MRR", "Judged@10", "Judged share of returned",
              "Results per query (queries < 10)", *[f"nDCG@10 {lang}" for lang in langs]], rows) + "\n")
    w("Judged@10 divides by 10 (as in ir-measures). Every returned result has a judgment (judged share of "
      "returned = 1.000); Judged@10 is below 1.000 only where a system returned fewer than 10 works. Postgres "
      "full-text search returns only works that contain a query lexeme; rank_bm25 scores every book, so the "
      "local list is always 10 long.\n")
    if a["descriptive"]:
        rows = [[k, _f(d["mean_system"]), _f(d["mean_reference"]),
                 f"{_f(d['mean_difference'], True)} [{_f(d['ci_low'], True)}, {_f(d['ci_high'], True)}]", _p(d["p_value"])]
                for k, d in a["descriptive"].items()]
        w("Descriptive differences (no correction, not part of the tested family):\n")
        w(_table(["Pair", "nDCG@10 system", "nDCG@10 reference", "Difference [95% CI]", "p (unadjusted)"], rows) + "\n")

    w("### Agreement of the rankings (top 10)\n")
    rows = [[k, _f(o["mean_overlap"]), _f(o["mean_recall_of_reference"]), f"{o['identical_lists']} of {o['num_queries']}",
             f"{o['identical_sets']} of {o['num_queries']}"] for k, o in a["overlaps"].items()]
    w(_table(["Pair", "Mean overlap", "Recall of the reference's top 10", "Identical lists", "Identical sets"], rows) + "\n")
    diff = a.get("score_differences", {}).get("pg_vec_exact vs vec_e5")
    if diff:
        w(f"Cosine similarity of the same (query, book) pairs, Postgres halfvec versus FAISS float32: mean absolute "
          f"difference {diff['mean_abs_difference']:.1e}, largest {diff['max_abs_difference']:.1e} over "
          f"{diff['pairs']} pairs.\n")

    cov = report.get("tsquery_coverage")
    if cov:
        w("### Turning the query into a tsquery\n")
        w(_table(["Query form", "Queries with no match", "Queries with fewer than 10 matches", "Median matching works"], [
            ["AND (websearch_to_tsquery in each configuration)", str(cov["and"]["queries_with_no_match"]),
             str(cov["and"]["queries_with_fewer_than_10"]), str(cov["and"]["median_matches"])],
            ["OR of the query lexemes (shipped)", str(cov["or"]["queries_with_no_match"]),
             str(cov["or"]["queries_with_fewer_than_10"]), str(cov["or"]["median_matches"])],
        ]) + "\n")

    w("### Latency from the measuring machine (warm, per query)\n")
    w("Postgres searches go through the pooled connection string over the Internet, from the measuring machine "
      "to aws-eu-central-1 (Frankfurt), one round trip per search leg; local searches run in process. The query "
      "vector is precomputed in both, so the model is not timed. The last row is a trivial statement through the "
      "same pool: the network floor.\n")
    rows = [[f"`{name}`" if not name.startswith("SELECT") else name, _f(lat["median_ms"], digits=2),
             _f(lat["p95_ms"], digits=2), _f(lat.get("warmup_ms"), digits=2)]
            for name, lat in report["latency"].items()]
    w(_table(["System", "Median ms", "p95 ms", "Warm-up query ms"], rows) + "\n")
    cold = [c for c in report.get("cold_start", []) if "cold_connect_and_select1_ms" in c]
    if cold:
        w("### Cold start\n")
        w(cold[0]["what_is_timed"] + "\n")
        rows = [[c["date"], c["note"],
                 f"{c['cold_connect_and_select1_ms']:.0f} + {c['cold_first_search_ms']:.0f} = {c['cold_time_to_first_result_ms']:.0f}",
                 f"{c['warm_connect_and_select1_ms']:.0f} + {c['warm_first_search_ms']:.0f} = {c['warm_time_to_first_result_ms']:.0f}",
                 f"{c['warm_search_reused_adapters_median_ms']:.0f}"]
                for c in cold]
        w(_table(["When (UTC)", "How the suspension was ensured",
                  "Suspended compute: connect + first search = ms", "Awake compute: connect + first search = ms",
                  "Awake, connection and adapters reused: search ms (median of 5)"], rows) + "\n")

    w("### Build times and sizes\n")
    for key, title in (("schema_facts", report["schema"]), ("hnsw_schema_facts", report.get("hnsw_schema"))):
        facts = report.get(key)
        if not facts:
            continue
        info = facts["catalog_info"]
        builds = info.get("build_seconds", {})
        w(f"Schema `{title}` ({info.get('works')} works, {info.get('work_texts')} texts; vector index "
          f"{info.get('vector_index')}). Index builds: " + ", ".join(f"{k} {v:.2f} s" for k, v in builds.items())
          + (f"; VACUUM ANALYZE {info['vacuum_seconds']:.2f} s" if info.get("vacuum_seconds") else "") + ".\n")
        w(_sizes_table(facts) + "\n")
    w(f"Database size (every schema and the system catalogs): {_mib(db['database_bytes'])} MiB at the time of the run.\n")

    for scale in report.get("scale", []):
        info = scale["catalog_info"]
        w(f"### Larger catalog: `{scale['schema']}` with {scale['vector_index']} ({scale['date']})\n")
        last = info.get("last_load") or {}
        if scale.get("source_catalog_sha256"):
            w(f"Source `{info.get('source_catalog')}` (sha256 {scale['source_catalog_sha256'][:12]}...).\n")
        w(f"{info.get('works')} works, {info.get('work_texts')} texts, text-search configurations in use: "
          f"{len(info.get('text_configs', []))}. Load: upsert {_f(last.get('upsert_seconds'), digits=1)} s, "
          f"{last.get('vectors_encoded', 0)} vectors encoded in {_f(last.get('encode_seconds'), digits=1)} s. "
          "Index builds: " + ", ".join(f"{k} {v:.2f} s" for k, v in info.get("build_seconds", {}).items()) + ".\n")
        w(f"ANN recall@10 against exact search ({scale['num_queries']} test queries): "
          f"{scale['recall_at_10']:.3f} ({scale['queries_with_full_recall']} of {scale['num_queries']} queries at 1.000). "
          "Latency (median / p95 ms): " + "; ".join(
              f"{k} {v['median_ms']:.1f} / {v['p95_ms']:.1f}" for k, v in scale["latency"].items()) + ".\n")
        w(_sizes_table(scale) + "\n")
        if scale.get("column_bytes"):
            w("Stored bytes by column (pg_column_size, after compression): " + ", ".join(
                f"{k} {_mib(v)} MiB" for k, v in scale["column_bytes"].items()) + ".\n")
        cap = scale.get("capacity")
        if cap:
            w(f"Free-plan capacity (1 GiB per project, linear extrapolation from {cap['works']} works): "
              f"{cap['bytes_per_work'] / 1024:.1f} KiB per work with the ANN index, "
              f"{cap['bytes_per_work_without_ann_index'] / 1024:.1f} KiB without it; about "
              f"{cap['works_that_fit']:,} works fit with the ANN index and {cap['works_that_fit_without_ann_index']:,} "
              f"without it (after what the database already holds).\n")

    w("## Reproduce\n")
    w("```\npython -m scripts.load_postgres --from data/catalog.db --schema eval177 "
      "--embedding-model intfloat/multilingual-e5-large-instruct\n"
      "python -m scripts.load_postgres --from data/catalog.db --schema eval177_hnsw "
      "--embedding-model intfloat/multilingual-e5-large-instruct --vector-index hnsw\n"
      "python -m app.evaluation.profile_parity run --schema eval177 --hnsw-schema eval177_hnsw\n"
      "python -m app.evaluation.profile_parity report   # no database, no model, no judge\n"
      "python -m scripts.load_postgres --from data/catalog_small/catalog.db --schema catalog_small "
      "--embedding-model intfloat/multilingual-e5-large-instruct\n"
      "python -m app.evaluation.profile_parity scale --schema catalog_small\n```\n")
    return "\n".join(out)


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db-path", default=DEFAULT_DB_PATH)
        p.add_argument("--queries-path", default=DEFAULT_QUERIES_PATH)
        p.add_argument("--cloud-judgments", default=DEFAULT_CLOUD_JUDGMENTS)
        p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
        p.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
        p.add_argument("--n-resamples", type=int, default=10000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default=None)

    run = sub.add_parser("run", help="run every system against the database and write the report")
    common(run)
    run.add_argument("--schema", required=True)
    run.add_argument("--hnsw-schema", default=None)
    run.add_argument("--base-judgments", default=DEFAULT_BASE_JUDGMENTS)
    run.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    run.add_argument("--e5-index", default=DEFAULT_E5_INDEX)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--no-judge", action="store_true")
    report = sub.add_parser("report", help="recompute the tables from the saved rankings (no database)")
    common(report)
    scale = sub.add_parser("scale", help="sizes, build times and ANN recall of another schema")
    common(scale)
    scale.add_argument("--schema", required=True)
    cold = sub.add_parser("cold-start", help="time a search on a suspended compute")
    common(cold)
    cold.add_argument("--schema", required=True)
    cold.add_argument("--note", required=True, help="how the suspension was ensured")
    return parser


def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    output = Path(args.output_json)
    saved: Dict[str, Any] = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    if args.command == "run":
        measured = run_measurement(args, log)
        saved = {**{k: saved[k] for k in ("scale", "cold_start") if k in saved}, **measured}
    elif args.command == "scale":
        from app.infrastructure.db.postgres_schema import server_versions

        result = ann_recall(args, log)
        db = _database()
        try:
            database_bytes = server_versions(db)["database_bytes"]
        finally:
            db.close()
        result["capacity"] = capacity(result, database_bytes)
        saved["scale"] = [s for s in saved.get("scale", [])
                          if (s["schema"], s["vector_index"]) != (result["schema"], result["vector_index"])] + [result]
    elif args.command == "cold-start":
        saved["cold_start"] = saved.get("cold_start", []) + [cold_start(args, log)]
    saved.pop("analysis", None)
    # Keep the measurements even if the analysis below fails.
    _write(args.output_json, json.dumps(saved, ensure_ascii=False, indent=2, default=str) + "\n")
    if "runs" not in saved:
        log("No saved rankings yet: run `profile_parity run` first")
        return 1
    report = build_report(saved, args)
    _write(args.output_json, json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    _write(args.output_md, build_markdown(report))
    log(f"Wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
