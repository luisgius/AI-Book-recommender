"""
Comparison of retrieval systems on the test collection, with uncertainty.

One shared way to turn rankings into a comparison table that can be defended.
Given, per system, the ranked book ids for every test query (a "run"), the
graded judgments and the query metadata, compare_systems returns a plain,
JSON-serializable dict with, per system:

- nDCG@10 (primary metric), Recall@10, P@5 and MRR: mean over queries and a
  95% percentile-bootstrap confidence interval (queries are resampled);
- Judged@10: share of the top 10 that has a judgment (pool coverage);
- condensed nDCG@10: nDCG@10 after removing unjudged books from the ranking,
  a robustness check against pool bias;
- nDCG@10 broken down by query language (es / en);
- against a named baseline: mean nDCG@10 difference with a paired bootstrap
  CI, paired permutation p-value and Holm-adjusted p-value across all the
  systems compared with the baseline.

The metric definitions are the ones in evaluation_service.py (gain
2^grade - 1, relevant when grade >= 1, unjudged = not relevant), so the means
match the evaluation job. Queries without any judgment are left out, as in
the evaluation job; queries whose judgments are all grade 0 stay in and score
0. render_markdown turns the dict into Markdown tables with three decimals.

The CLI runs the four non-LLM systems (BM25, FAISS, hybrid RRF and hybrid RRF
+ MMR) at depth 10 with hybrid RRF as the baseline and writes
data/evaluation/results_comparison_baseline.json and
data/evaluation/COMPARISON_BASELINE.md:

    python -m app.evaluation.comparison
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.evaluation.evaluation_service import (
    RELEVANT_GRADE,
    BookId,
    condensed_ranking,
    judged_fraction_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.statistics import (
    bootstrap_ci,
    holm_correction,
    paired_bootstrap_ci,
    paired_permutation_test,
)
from app.evaluation.types import TestQuery

PRIMARY_METRIC = "ndcg_at_10"
METRICS = (
    "ndcg_at_10",
    "recall_at_10",
    "precision_at_5",
    "mrr",
    "judged_at_10",
    "condensed_ndcg_at_10",
)
UNKNOWN_LANGUAGE = "unknown"

BASELINE_SYSTEMS = ("lexical_only", "vector_only", "hybrid_rrf", "hybrid_rrf_mmr")
DEFAULT_BASELINE = "hybrid_rrf"
DEFAULT_DEPTH = 10
DEFAULT_MMR_LAMBDA = 0.6
DEFAULT_MMR_TOP_K = 20
DEFAULT_OUTPUT_JSON = "data/evaluation/results_comparison_baseline.json"
DEFAULT_OUTPUT_MD = "data/evaluation/COMPARISON_BASELINE.md"

SYSTEM_DESCRIPTIONS = {
    "lexical_only": "BM25 only (rank-bm25 index in data/indexes/bm25_index.pkl)",
    "vector_only": "FAISS only (dense embeddings, index in data/indexes/faiss_index)",
    "hybrid_rrf": "BM25 + FAISS fused with Reciprocal Rank Fusion (SearchService)",
    "hybrid_rrf_mmr": "hybrid_rrf re-ranked with MMR (lambda 0.6 over the top 20)",
}


# =============================================================================
# Comparison
# =============================================================================


def score_query(ranked_ids: Sequence[BookId], grades: Mapping[BookId, int]) -> Dict[str, float]:
    """
    Every metric of the comparison for one query.

    Args:
        ranked_ids: Book ids in ranked order (as returned by the system)
        grades: Graded judgments of the query (book id -> 0-3)

    Returns:
        Metric name -> value (keys are METRICS)
    """
    ranked = list(ranked_ids)
    return {
        "ndcg_at_10": ndcg_at_k(ranked, grades, k=10),
        "recall_at_10": recall_at_k(ranked, grades, k=10),
        "precision_at_5": precision_at_k(ranked, grades, k=5),
        "mrr": reciprocal_rank(ranked, grades),
        "judged_at_10": judged_fraction_at_k(ranked, grades, k=10),
        "condensed_ndcg_at_10": ndcg_at_k(condensed_ranking(ranked, grades), grades, k=10),
    }


def compare_systems(
    runs: Mapping[str, Mapping[str, Sequence[BookId]]],
    judgments: Mapping[str, Mapping[BookId, int]],
    queries: Sequence[TestQuery],
    baseline: str,
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Compare several systems on the same queries against a baseline.

    Args:
        runs: System name -> query id -> ranked book ids
        judgments: Query id -> book id -> grade (0-3)
        queries: Test queries (order, ids and language labels)
        baseline: Name of the system the others are compared with
        n_resamples: Bootstrap and permutation resamples
        confidence: Confidence level of the intervals
        seed: Random seed for the bootstrap and the permutation test

    Returns:
        JSON-serializable dict with keys config, num_queries, query_ids,
        excluded_query_ids, queries_without_relevant, languages, systems and
        vs_baseline (see the module docstring).

    Raises:
        ValueError: If the baseline is not among the runs, no query has
            judgments, or a run lacks a ranking for an evaluated query
    """
    if baseline not in runs:
        raise ValueError(f"Baseline '{baseline}' is not among the systems {list(runs)}")

    evaluated = [q for q in queries if judgments.get(q.query_id)]
    if not evaluated:
        raise ValueError("No query has judgments")
    for system, run in runs.items():
        missing = [q.query_id for q in evaluated if q.query_id not in run]
        if missing:
            raise ValueError(f"Run '{system}' has no ranking for queries {missing}")

    def with_ci(values: List[float]) -> Dict[str, float]:
        mean, low, high = bootstrap_ci(values, n_resamples=n_resamples, confidence=confidence, seed=seed)
        return {"mean": mean, "ci_low": low, "ci_high": high}

    def language_of(query: TestQuery) -> str:
        return query.language or UNKNOWN_LANGUAGE

    languages = sorted({language_of(q) for q in evaluated})

    systems: Dict[str, Any] = {}
    for system, run in runs.items():
        per_query = {q.query_id: score_query(run[q.query_id], judgments[q.query_id]) for q in evaluated}
        metrics = {name: with_ci([per_query[q.query_id][name] for q in evaluated]) for name in METRICS}
        by_language = {}
        for language in languages:
            ids = [q.query_id for q in evaluated if language_of(q) == language]
            by_language[language] = {
                "num_queries": len(ids),
                PRIMARY_METRIC: with_ci([per_query[i][PRIMARY_METRIC] for i in ids]),
            }
        systems[system] = {"metrics": metrics, "by_language": by_language, "per_query": per_query}

    def primary_scores(system: str) -> List[float]:
        return [systems[system]["per_query"][q.query_id][PRIMARY_METRIC] for q in evaluated]

    vs_baseline: Dict[str, Dict[str, Any]] = {}
    raw_p: Dict[str, float] = {}
    for system in runs:
        if system == baseline:
            continue
        diff, low, high = paired_bootstrap_ci(
            primary_scores(system), primary_scores(baseline),
            n_resamples=n_resamples, confidence=confidence, seed=seed,
        )
        _, raw_p[system] = paired_permutation_test(
            primary_scores(system), primary_scores(baseline), n_resamples=n_resamples, seed=seed
        )
        vs_baseline[system] = {
            "metric": PRIMARY_METRIC,
            "mean_difference": diff,
            "ci_low": low,
            "ci_high": high,
            "p_value": raw_p[system],
        }
    for system, adjusted in holm_correction(raw_p).items():
        vs_baseline[system]["p_value_holm"] = adjusted

    return {
        "config": {
            "baseline": baseline,
            "primary_metric": PRIMARY_METRIC,
            "metrics": list(METRICS),
            "gain": "2^grade - 1",
            "relevant_if_grade_at_least": RELEVANT_GRADE,
            "unjudged_books": "count as not relevant; condensed_ndcg_at_10 removes them before scoring",
            "confidence_interval": (
                f"percentile bootstrap over queries, {n_resamples} resamples, "
                f"confidence {confidence}, seed {seed}"
            ),
            "significance": (
                f"two-sided paired permutation test (sign flips) on per-query {PRIMARY_METRIC}, "
                f"{n_resamples} resamples, seed {seed}; Holm correction across all systems "
                f"compared with the baseline"
            ),
            "n_resamples": n_resamples,
            "confidence": confidence,
            "seed": seed,
        },
        "num_queries": len(evaluated),
        "query_ids": [q.query_id for q in evaluated],
        "excluded_query_ids": [q.query_id for q in queries if not judgments.get(q.query_id)],
        "queries_without_relevant": [
            q.query_id for q in evaluated
            if not any(grade >= RELEVANT_GRADE for grade in judgments[q.query_id].values())
        ],
        "languages": {
            language: sum(1 for q in evaluated if language_of(q) == language) for language in languages
        },
        "systems": systems,
        "vs_baseline": vs_baseline,
    }


def describe_judgments(
    raw_judgments: Mapping[str, Sequence[Mapping[str, Any]]],
    queries: Sequence[TestQuery],
    num_books: int,
) -> Dict[str, Any]:
    """
    Descriptive statistics of the relevance judgments (qrels datasheet).

    Args:
        raw_judgments: Query id -> list of {source, source_id, relevance, ...}
            as loaded from relevance_judgments.json (extra fields are ignored)
        queries: Test queries
        num_books: Number of books in the catalog

    Returns:
        JSON-serializable dict: counts, judgments per query (min / median /
        max), grade distribution, queries without a relevant book, share of
        the query x book matrix that is judged and per-language counts.
    """
    def is_relevant(item: Mapping[str, Any]) -> bool:
        return item["relevance"] >= RELEVANT_GRADE

    items_by_query = {q.query_id: list(raw_judgments.get(q.query_id, [])) for q in queries}
    counts = [len(items) for items in items_by_query.values()]
    grades = Counter(item["relevance"] for items in items_by_query.values() for item in items)
    pairs = {
        (query_id, item["source"], item["source_id"])
        for query_id, items in items_by_query.items() for item in items
    }
    extra_fields = {
        field for items in items_by_query.values() for item in items for field in item
    } - {"source", "source_id", "relevance"}
    without_relevant = [qid for qid, items in items_by_query.items() if not any(map(is_relevant, items))]

    by_language: Dict[str, Dict[str, Any]] = {}
    for language in sorted({q.language or UNKNOWN_LANGUAGE for q in queries}):
        ids = [q.query_id for q in queries if (q.language or UNKNOWN_LANGUAGE) == language]
        by_language[language] = {
            "num_queries": len(ids),
            "num_judgments": sum(len(items_by_query[i]) for i in ids),
            "num_relevant_judgments": sum(1 for i in ids for item in items_by_query[i] if is_relevant(item)),
            "queries_without_relevant": [i for i in ids if i in without_relevant],
        }

    matrix_size = len(queries) * num_books
    return {
        "num_queries": len(queries),
        "num_books": num_books,
        "num_judgments": sum(counts),
        "num_distinct_query_book_pairs": len(pairs),
        "judged_share_of_matrix": len(pairs) / matrix_size if matrix_size else 0.0,
        "judgments_per_query": {
            "min": min(counts) if counts else 0,
            "median": median(counts) if counts else 0,
            "max": max(counts) if counts else 0,
        },
        "grade_distribution": {str(grade): grades.get(grade, 0) for grade in range(4)},
        "num_relevant_judgments": sum(grades.get(grade, 0) for grade in range(RELEVANT_GRADE, 4)),
        "queries_without_judgments": [qid for qid, items in items_by_query.items() if not items],
        "queries_without_relevant": without_relevant,
        "by_language": by_language,
        "extra_judgment_fields": sorted(extra_fields),
    }


# =============================================================================
# Markdown rendering
# =============================================================================


def _fmt(value: float, signed: bool = False) -> str:
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _fmt_ci(value: float, low: float, high: float, signed: bool = False) -> str:
    return f"{_fmt(value, signed)} [{_fmt(low, signed)}, {_fmt(high, signed)}]"


def _fmt_stat(stat: Mapping[str, float]) -> str:
    """A {mean, ci_low, ci_high} entry as 'mean [low, high]'."""
    return _fmt_ci(stat["mean"], stat["ci_low"], stat["ci_high"])


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """
    Render a compare_systems report as two Markdown tables (three decimals).

    The first table has the aggregate metrics and the comparison with the
    baseline; the second, nDCG@10 by query language.
    """
    baseline = report["config"]["baseline"]
    header = [
        "System", "nDCG@10 [95% CI]", "Recall@10", "P@5", "MRR", "Judged@10", "Condensed nDCG@10",
        f"Diff. nDCG@10 vs {baseline} [95% CI]", "p", "p (Holm)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for system, data in report["systems"].items():
        m = data["metrics"]
        row = [
            f"{system} (baseline)" if system == baseline else system,
            _fmt_stat(m["ndcg_at_10"]),
            _fmt(m["recall_at_10"]["mean"]),
            _fmt(m["precision_at_5"]["mean"]),
            _fmt(m["mrr"]["mean"]),
            _fmt(m["judged_at_10"]["mean"]),
            _fmt(m["condensed_ndcg_at_10"]["mean"]),
        ]
        if system == baseline:
            row += ["", "", ""]
        else:
            c = report["vs_baseline"][system]
            row += [
                _fmt_ci(c["mean_difference"], c["ci_low"], c["ci_high"], signed=True),
                _fmt_p(c["p_value"]),
                _fmt_p(c["p_value_holm"]),
            ]
        lines.append("| " + " | ".join(row) + " |")

    languages = report["languages"]
    header = ["System"] + [f"nDCG@10 {lang} (n={n}) [95% CI]" for lang, n in languages.items()]
    lines += ["", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for system, data in report["systems"].items():
        row = [system] + [_fmt_stat(data["by_language"][lang][PRIMARY_METRIC]) for lang in languages]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI: the four non-LLM systems on the bundled test collection
# =============================================================================


def run_system(
    system: str,
    query_text: str,
    *,
    search_service,
    lexical_repo,
    vector_repo,
    embeddings_store,
    depth: int = DEFAULT_DEPTH,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    mmr_top_k: int = DEFAULT_MMR_TOP_K,
) -> list:
    """
    Run one of BASELINE_SYSTEMS for a query, exactly as the evaluation job does.

    Returns:
        Ranked SearchResult list of at most `depth` results

    Raises:
        ValueError: If the system name is unknown
    """
    # Imported here so the comparison functions above can be used (and
    # tested) without loading the search infrastructure.
    from app.evaluation.evaluation_job import run_hybrid_search, run_lexical_search, run_vector_search

    if system == "lexical_only":
        return run_lexical_search(query_text, lexical_repo, max_results=depth)
    if system == "vector_only":
        return run_vector_search(query_text, vector_repo, embeddings_store, max_results=depth)
    if system in ("hybrid_rrf", "hybrid_rrf_mmr"):
        return run_hybrid_search(
            query_text, search_service, embeddings_store, max_results=depth,
            use_diversification=(system == "hybrid_rrf_mmr"),
            diversity_lambda=mmr_lambda, mmr_top_k=mmr_top_k,
        )
    raise ValueError(f"Unknown system '{system}'; choose from {list(BASELINE_SYSTEMS)}")


def _git(*args: str) -> Optional[str]:
    """Output of a read-only git command, or None if git is unavailable."""
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", *args], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_markdown_document(report: Mapping[str, Any]) -> str:
    """COMPARISON_BASELINE.md: provenance, method notes and the tables."""
    run = report["run"]
    qrels = report["qrels"]
    config = report["config"]
    languages = ", ".join(f"{n} {lang}" for lang, n in report["languages"].items())
    commit = run["git_commit"] or "unknown"
    dirty = " (with uncommitted changes in tracked files)" if run["git_worktree_dirty"] else ""
    systems = "\n".join(f"- `{name}`: {SYSTEM_DESCRIPTIONS[name]}" for name in report["systems"])
    others = len(report["vs_baseline"])
    return f"""# Baseline comparison of the non-LLM retrieval systems

- Date: {run["date"]}
- Git commit: `{commit}`{dirty}
- Reproduce: `{run["command"]}`
- Collection: {report["num_queries"]} queries ({languages}), {qrels["num_judgments"]} graded judgments (0-3),
  {qrels["num_books"]}-book catalog; see `QRELS_DATASHEET.md`. Judgments file SHA-256
  `{run["judgments_sha256"][:12]}`, queries file SHA-256 `{run["queries_sha256"][:12]}`.
- Depth: every system returns at most {run["depth"]} results per query (`max_results`).

Systems:

{systems}

Method:

- Metrics as in `app/evaluation/evaluation_service.py`: gain 2^grade - 1, a book is relevant
  when its grade is >= 1, a book without a judgment counts as not relevant. MRR is computed
  over the returned list, so at depth {run["depth"]} it is MRR@{run["depth"]}.
- Primary metric: nDCG@10. 95% CI: percentile bootstrap over queries
  ({config["n_resamples"]} resamples, seed {config["seed"]}).
- Diff.: mean per-query nDCG@10 difference against `{config["baseline"]}`, with a paired bootstrap
  CI; p: two-sided paired permutation test ({config["n_resamples"]} sign-flip resamples); p (Holm):
  Holm correction across the {others} comparisons with the baseline.
- Judged@10: share of the top 10 that has a judgment (denominator 10).
- Condensed nDCG@10: nDCG@10 after removing unjudged books from the ranking. It is computed
  from the same depth-{run["depth"]} lists, so it can only move judged books up within them.
- Queries whose judgments are all grade 0 ({", ".join(report["queries_without_relevant"]) or "none"})
  are kept and score 0 on every metric.

{render_markdown(report)}"""


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the four non-LLM systems, compare them with the baseline and save the report."""
    from app.domain.services import SearchService
    from app.evaluation.evaluation_job import (
        DEFAULT_DB_PATH,
        DEFAULT_INDEXES_DIR,
        DEFAULT_JUDGMENTS_PATH,
        DEFAULT_QUERIES_PATH,
        load_relevance_judgments,
        load_test_queries,
        resolve_judgments_to_uuids,
    )
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
    from app.infrastructure.search.bm25_search_repository import BM25SearchRepository
    from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
    from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository

    parser = argparse.ArgumentParser(description="Compare the non-LLM retrieval systems with a baseline")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--indexes-dir", default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--queries-path", default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--judgments-path", default=DEFAULT_JUDGMENTS_PATH)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Results per query (max_results)")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, choices=BASELINE_SYSTEMS)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)
    raw_args = list(sys.argv[1:] if argv is None else argv)

    started = datetime.now(UTC)
    catalog = SqliteBookCatalogRepository(Path(args.db_path))
    books = catalog.get_all()
    bm25 = BM25SearchRepository()
    bm25.load_index(str(Path(args.indexes_dir) / "bm25_index.pkl"))
    embeddings = EmbeddingsStoreFaiss()
    embeddings.load_index(str(Path(args.indexes_dir) / "faiss_index"))
    vector = FaissVectorSearchRepository(embeddings, books)
    search_service = SearchService(lexical_search=bm25, vector_search=vector, embeddings_store=embeddings)

    queries = load_test_queries(args.queries_path)
    raw_judgments = load_relevance_judgments(args.judgments_path)
    resolved = resolve_judgments_to_uuids(raw_judgments, catalog)
    grades = {query_id: dict(j.judgments) for query_id, j in resolved.items()}

    runs = {
        system: {
            q.query_id: [
                r.book.id for r in run_system(
                    system, q.text, search_service=search_service, lexical_repo=bm25,
                    vector_repo=vector, embeddings_store=embeddings, depth=args.depth,
                )
            ]
            for q in queries
        }
        for system in BASELINE_SYSTEMS
    }

    report = compare_systems(
        runs, grades, queries, baseline=args.baseline, n_resamples=args.n_resamples, seed=args.seed
    )
    report["qrels"] = describe_judgments(raw_judgments, queries, num_books=len(books))
    report["qrels"]["num_judgments_resolved_to_catalog"] = sum(len(g) for g in grades.values())
    report["run"] = {
        "date": started.isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_worktree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "command": " ".join(["python -m app.evaluation.comparison", *map(shlex.quote, raw_args)]).strip(),
        "depth": args.depth,
        "mmr_lambda": DEFAULT_MMR_LAMBDA,
        "mmr_top_k": DEFAULT_MMR_TOP_K,
        "db_path": args.db_path,
        "indexes_dir": args.indexes_dir,
        "queries_path": args.queries_path,
        "judgments_path": args.judgments_path,
        "queries_sha256": _sha256(args.queries_path),
        "judgments_sha256": _sha256(args.judgments_path),
        "num_books": len(books),
        "systems": {name: SYSTEM_DESCRIPTIONS[name] for name in BASELINE_SYSTEMS},
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(build_markdown_document(report), encoding="utf-8")

    print(render_markdown(report))
    print(f"Saved {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
