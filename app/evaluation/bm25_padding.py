"""
BM25 padding (task V13): what the zero-score books of the local BM25 cost the hybrid.

Until V13 the local BM25 adapter (rank_bm25) always returned max_results
books, filling the list with books that contain no query term (score 0).
In a hybrid those books enter the RRF fusion as if the lexical side had
retrieved them. V12 saw it descriptively (a local hybrid without them scored
0.622 nDCG@10 against 0.592). V13 fixed the adapter (it returns only books
with a query term) and kept the old behaviour behind
BM25SearchRepository(zero_score_padding=True) to reproduce published runs.
This module measures the fix with a design fixed before any result was seen.

Design:
- 177-book collection (data/catalog.db), the 44 test queries (24 en, 20 es),
  depth 10, the shared harness (retrieval_ablation: ModelStack,
  make_searcher, collect_run; SearchService unchanged, RRF k = 60, 20
  candidates per side, 60 with the reranker over the top 30).
- Systems: the published arms bm25_normalized, vec_e5, hybrid_norm_e5 and
  hybrid_norm_e5_bge exactly as declared in retrieval_ablation.SYSTEMS
  (legacy padding), and three new arms with the fixed adapter:
  bm25_normalized_nopad, hybrid_norm_e5_nopad, hybrid_norm_e5_bge_nopad.
- Three pre-declared contrasts on per-query nDCG@10 (CONTRASTS), Holm
  across exactly these three; two-sided paired randomisation test (sign
  flips) and paired percentile bootstrap 95% interval, 10,000 resamples,
  seed 0:
    F1 hybrid_norm_e5_nopad - hybrid_norm_e5 (does the fix change the hybrid?)
    F2 hybrid_norm_e5_nopad - vec_e5 (is the hybrid at least as good as its
       vector side once the defect is removed? The V06 question)
    F3 hybrid_norm_e5_bge_nopad - hybrid_norm_e5_bge (does the fix still
       matter after the reranker?)
- Descriptive only: BM25 alone with and without padding, list lengths, the
  en/es breakdown, how many padding books entered the fusion, the vector
  side's scores at rank 10, reproduction of the published runs.
- Judgments: start from relevance_judgments_extended_cloud.json (V12); the
  (query, book) pairs of any top 10 still unjudged are graded by the
  configured judge (JUDGE_*, cached in data/evaluation/llm_judgments); the
  merged file is written to relevance_judgments_extended_v13.json. Every
  returned top-10 result of every system must be judged. Judged@10 divides
  by 10, so a system that returns fewer than 10 books scores below 1.0
  without any hole; the judged share of the returned results (1.000) and
  the list lengths are reported next to it, as in V12.

CLI (from the repository root):

    python -m app.evaluation.bm25_padding run            # models + judge for the holes
    python -m app.evaluation.bm25_padding run --no-judge # fail on a hole
    python -m app.evaluation.bm25_padding report         # tables from the saved rankings only

`run` loads multilingual-e5-large-instruct (its FAISS index under
data/indexes_ablation/) and BAAI/bge-reranker-v2-m3; `report` needs no model,
judge or network. Outputs: data/evaluation/results_bm25_padding.json and
data/evaluation/BM25_PADDING.md.
"""

import argparse
import json
import statistics as stats_module
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from app.domain.entities import Book
from app.evaluation.comparison import PRIMARY_METRIC, _git, _sha256, compare_pairs, compare_systems
from app.evaluation.evaluation_service import BookId
from app.evaluation.retrieval_ablation import (
    BGE,
    DEPTH,
    E5,
    NORMALIZED,
    RERANK_DEPTH,
    SYSTEMS as ABLATION_SYSTEMS,
    SystemSpec,
    collect_run,
    execution_order,
    latency_summary,
    require_judged,
    resolve_grades,
)
from app.evaluation.types import TestQuery

DEFAULT_DB_PATH = "data/catalog.db"
DEFAULT_QUERIES_PATH = "app/evaluation/test_queries.json"
DEFAULT_BASE_JUDGMENTS = "app/evaluation/relevance_judgments_extended_cloud.json"
DEFAULT_V13_JUDGMENTS = "app/evaluation/relevance_judgments_extended_v13.json"
DEFAULT_CACHE_DIR = "data/evaluation/llm_judgments"
DEFAULT_INDEXES_DIR = "data/indexes"
DEFAULT_ABLATION_DIR = "data/indexes_ablation"
DEFAULT_OUTPUT_JSON = "data/evaluation/results_bm25_padding.json"
DEFAULT_OUTPUT_MD = "data/evaluation/BM25_PADDING.md"

# Published runs the legacy arms must reproduce (identical top-10 lists).
PROFILE_PARITY_JSON = "data/evaluation/results_profile_parity.json"
ABLATION_GLM_JSON = "data/evaluation/results_retrieval_ablation_z-ai-glm-5.3-flash.json"

# Candidates per side that SearchService asks for at depth 10: 2 x 10
# without a reranker, 2 x max(10, rerank depth 30) with one.
FUSION_CANDIDATES = 2 * DEPTH
RERANK_CANDIDATES = 2 * max(DEPTH, RERANK_DEPTH)

# =============================================================================
# The design (fixed before the run)
# =============================================================================

_PUBLISHED = {spec.name: spec for spec in ABLATION_SYSTEMS}

SYSTEMS: Tuple[SystemSpec, ...] = (
    _PUBLISHED["bm25_normalized"],
    SystemSpec("bm25_normalized_nopad", lexical=NORMALIZED,
               description="BM25, normalizing tokenizer, fixed adapter: only books with a query term"),
    _PUBLISHED["vec_e5"],
    _PUBLISHED["hybrid_norm_e5"],
    SystemSpec("hybrid_norm_e5_nopad", lexical=NORMALIZED, embedding=E5,
               description="RRF(bm25_normalized_nopad, vec_e5)"),
    _PUBLISHED["hybrid_norm_e5_bge"],
    SystemSpec("hybrid_norm_e5_bge_nopad", lexical=NORMALIZED, embedding=E5, reranker=BGE,
               description=f"hybrid_norm_e5_nopad + {BGE} over the top {RERANK_DEPTH}"),
)

# (name, system, reference, question): one Holm family.
CONTRASTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("F1", "hybrid_norm_e5_nopad", "hybrid_norm_e5", "Does removing the padding change the hybrid?"),
    ("F2", "hybrid_norm_e5_nopad", "vec_e5", "Is the fixed hybrid at least as good as its vector side?"),
    ("F3", "hybrid_norm_e5_bge_nopad", "hybrid_norm_e5_bge", "Does the fix still matter after the reranker?"),
)

# Descriptive pairs (no test, not in the Holm family).
DESCRIPTIVE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("bm25_normalized_nopad", "bm25_normalized"),
    ("hybrid_norm_e5", "vec_e5"),
    ("hybrid_norm_e5_bge_nopad", "vec_e5"),
)

# Legacy arm -> (published file, run name) whose top-10 lists it must reproduce;
# fixed arm -> the descriptive V12 run it should equal.
REPRODUCTION = {
    "bm25_normalized": (PROFILE_PARITY_JSON, "bm25_normalized"),
    "vec_e5": (PROFILE_PARITY_JSON, "vec_e5"),
    "hybrid_norm_e5": (PROFILE_PARITY_JSON, "hybrid_norm_e5"),
    "hybrid_norm_e5_bge": (ABLATION_GLM_JSON, "hybrid_norm_e5_bge"),
    "hybrid_norm_e5_nopad": (PROFILE_PARITY_JSON, "hybrid_norm_e5_matches"),
}


# =============================================================================
# Pure analysis (shared by `run` and `report`)
# =============================================================================


def list_lengths(runs: Mapping[str, Mapping[str, Sequence[Any]]], judgments: Mapping[str, Mapping[Any, int]],
                 query_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Per system: mean results per query, queries with fewer than 10 or none, judged share of the returned."""
    out = {}
    for name, run in runs.items():
        lengths = [min(len(run.get(q, [])), DEPTH) for q in query_ids]
        returned = sum(lengths)
        judged = sum(1 for q in query_ids for b in list(run.get(q, []))[:DEPTH] if b in judgments.get(q, {}))
        out[name] = {
            "mean": sum(lengths) / len(lengths) if lengths else 0.0,
            "queries_with_fewer_than_10": sum(1 for n in lengths if n < DEPTH),
            "queries_with_no_result": sum(1 for n in lengths if n == 0),
            "judged_share_of_returned": judged / returned if returned else 1.0,
        }
    return out


def not_relevant_share(run: Mapping[str, Sequence[Any]], judgments: Mapping[str, Mapping[Any, int]]) -> float:
    """Share of the returned top-10 results judged 0 (not relevant)."""
    grades = [judgments.get(q, {}).get(b) for q, ranked in run.items() for b in list(ranked)[:DEPTH]]
    return sum(1 for g in grades if g == 0) / len(grades) if grades else 0.0


def reproduction(runs: Mapping[str, Mapping[str, Sequence[str]]]) -> Dict[str, Dict[str, Any]]:
    """Identical top-10 lists of each arm against the published run it must equal (files that exist only)."""
    out: Dict[str, Dict[str, Any]] = {}
    cache: Dict[str, Any] = {}
    for system, (path, name) in REPRODUCTION.items():
        if system not in runs or not Path(path).is_file():
            continue
        published = cache.setdefault(path, json.loads(Path(path).read_text(encoding="utf-8")))["runs"].get(name)
        if published is None:
            continue
        mine = runs[system]
        qids = [q for q in mine if q in published]
        out[system] = {
            "reference_file": path, "reference_run": name, "num_queries": len(qids),
            "identical_lists": sum(1 for q in qids if [str(b) for b in mine[q]][:DEPTH] == list(published[q])[:DEPTH]),
        }
    return out


def analyse(
    runs: Mapping[str, Mapping[str, Sequence[BookId]]],
    queries: Sequence[TestQuery],
    judgments: Mapping[str, Mapping[BookId, int]],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Metrics, the three contrasts (Holm), descriptive pairs and list lengths.

    Raises:
        UnjudgedResultsError: If a returned top-10 result has no judgment
    """
    require_judged(runs, judgments, DEPTH)
    report = compare_systems(runs, judgments, queries, None, n_resamples=n_resamples, seed=seed)
    query_ids = report["query_ids"]
    lengths = list_lengths(runs, judgments, query_ids)
    for name, data in report["systems"].items():
        data["list_lengths"] = lengths[name]
        data["not_relevant_share_of_returned"] = not_relevant_share(runs[name], judgments)
        if lengths[name]["judged_share_of_returned"] != 1.0:  # require_judged guarantees it
            raise AssertionError(f"{name}: a returned result has no judgment")
    contrasts = compare_pairs(report, [(n, s, r) for n, s, r, _ in CONTRASTS if s in runs and r in runs],
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
    }


# =============================================================================
# Running the systems (models)
# =============================================================================


def padding_facts(books: Sequence[Book], texts: Mapping[str, str]) -> Dict[str, Any]:
    """
    How many non-matching books the legacy BM25 sends into the fusion, and a
    check that the fixed list is the legacy list without them, on every query
    at both candidate depths.
    """
    from app.infrastructure.search.bm25_search_repository import BM25SearchRepository

    legacy = BM25SearchRepository(tokenizer=NORMALIZED, zero_score_padding=True)
    legacy.build_index(list(books))
    fixed = BM25SearchRepository(tokenizer=NORMALIZED, zero_score_padding=False)
    fixed.build_index(list(books))
    out: Dict[str, Any] = {}
    for depth in (DEPTH, FUSION_CANDIDATES, RERANK_CANDIDATES):
        padding, prefix_ok = [], True
        for text in texts.values():
            old = [r.book.id for r in legacy.search(text, max_results=depth)]
            new = [r.book.id for r in fixed.search(text, max_results=depth)]
            padding.append(len(old) - len(new))
            prefix_ok = prefix_ok and old[:len(new)] == new
        out[str(depth)] = {
            "mean_padding_books": sum(padding) / len(padding),
            "max_padding_books": max(padding),
            "queries_with_padding": sum(1 for n in padding if n > 0),
            "fixed_list_is_legacy_prefix": prefix_ok,
        }
    return out


def vector_score_facts(scores: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    """Cosine similarity of vec_e5's first and tenth result over the queries."""
    first = [s[0] for s in scores.values() if s]
    tenth = [s[DEPTH - 1] for s in scores.values() if len(s) >= DEPTH]
    return {
        "rank_1": {"median": stats_module.median(first), "min": min(first)},
        "rank_10": {"median": stats_module.median(tenth), "min": min(tenth), "max": max(tenth)},
        "queries_with_10_results": len(tenth),
    }


def run_measurement(args: argparse.Namespace, log: Callable[[str], None]) -> Dict[str, Any]:
    """`run`: every system on the 44 queries, the judge for the holes, the merged judgments file."""
    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.evaluation.retrieval_ablation import ModelStack, resolve_device
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

    started = datetime.now(UTC)
    device = args.device or resolve_device()  # imports torch before faiss (macOS)
    books = SqliteBookCatalogRepository(Path(args.db_path)).get_all()
    queries = load_test_queries(args.queries_path)
    texts = {q.query_id: q.text for q in queries}
    base_raw = load_relevance_judgments(args.base_judgments)

    stack = ModelStack(books, indexes_dir=args.indexes_dir, ablation_dir=args.ablation_dir, device=device, log=log)
    runs: Dict[str, Dict[str, List[BookId]]] = {}
    latency: Dict[str, Dict[str, Any]] = {}
    vector_scores: Dict[str, List[float]] = {}
    for spec in execution_order(SYSTEMS):
        log(f"Running {spec.name}")
        search = stack.searcher_for(spec)
        if spec.name == "vec_e5":
            inner = search

            def search(text: str, _inner=inner):  # keep the cosine of each returned book
                results = _inner(text)
                vector_scores[text] = [float(r.final_score) for r in results]
                return results
        runs[spec.name], per_query_ms, warmup_ms = collect_run(search, texts, warmup_text=queries[0].text)
        latency[spec.name] = {**latency_summary(list(per_query_ms.values())), "warmup_ms": warmup_ms}
        del search
    runs = {spec.name: runs[spec.name] for spec in SYSTEMS}  # declared order

    # Judgments: the V12 file plus the judge's grades of the remaining holes.
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
        cost = [j.usage_recorder.summary()["total"]["reported_cost_usd"] for j in judges if j.usage_recorder]
        judging.update({
            "judge_model": judge.model_name, "prompt_version": judge.prompt_version,
            "judged_in_this_run": result.judged, "already_cached": result.cached,
            "seconds_per_judgment": result.seconds_per_judgment, "cache_file": str(cache.path),
            "grade_distribution": [sum(1 for j in new_judgments if j.grade == g) for g in range(4)],
            "reported_cost_usd": sum(c for c in cost if c is not None),
        })
    merged = merge_judgments(base_raw, new_judgments)
    Path(args.v13_judgments).write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote {args.v13_judgments} ({sum(len(e['judgments']) for e in merged)} judgments, {len(new_judgments)} new)")

    return {
        "date": started.isoformat(timespec="seconds"),
        "command": "python -m app.evaluation.bm25_padding " + " ".join(sys.argv[1:]),
        "git_commit": _git("rev-parse", "HEAD"),
        "device": device,
        "judgments": {
            "base_file": args.base_judgments, "base_sha256": _sha256(args.base_judgments),
            "v13_file": args.v13_judgments, **judging,
        },
        "runs": {s: {q: [str(b) for b in run[q]] for q in run} for s, run in runs.items()},
        "latency": latency,
        "padding": padding_facts(books, texts),
        "vector_scores": vector_score_facts(vector_scores),
        "model_costs": stack.costs,
    }


# =============================================================================
# Report
# =============================================================================


def build_report(saved: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Recompute every table from the saved rankings and the V13 judgments (no model, no judge)."""
    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

    books: List[Book] = SqliteBookCatalogRepository(Path(args.db_path)).get_all()
    queries = load_test_queries(args.queries_path)
    judgments = resolve_grades(load_relevance_judgments(args.v13_judgments), books)
    runs = {s: {q: [UUID(b) for b in ids] for q, ids in run.items()} for s, run in saved["runs"].items()}
    analysis = analyse(runs, queries, judgments, n_resamples=args.n_resamples, seed=args.seed)
    analysis["reproduction"] = reproduction(saved["runs"])
    base_file = saved["judgments"]["base_file"]
    analysis["judgment_counts"] = {
        "base": _count_judgments(load_relevance_judgments(base_file)) if Path(base_file).is_file() else None,
        "v13": _count_judgments(load_relevance_judgments(args.v13_judgments)),
    }
    return {**saved, "analysis": analysis, "judgments_v13_sha256": _sha256(args.v13_judgments)}


def _count_judgments(raw: Mapping[str, Sequence[Any]]) -> int:
    return sum(len(items) for items in raw.values())


def _f(value: Optional[float], signed: bool = False, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(stat: Mapping[str, float]) -> str:
    return f"{_f(stat['mean'])} [{_f(stat['ci_low'])}, {_f(stat['ci_high'])}]"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build_markdown(report: Mapping[str, Any]) -> str:
    a = report["analysis"]
    systems = a["systems"]
    j = report["judgments"]
    out: List[str] = []
    w = out.append
    w("# BM25 padding: what the zero-score books cost the hybrid (V13)\n")
    w(f"Generated by `python -m app.evaluation.bm25_padding report` from the rankings of the run of "
      f"{report['date']} (git `{str(report.get('git_commit'))[:7]}`, device {report.get('device')}).")
    w(f"{a['num_queries']} test queries ({', '.join(f'{k} {v}' for k, v in a['languages'].items())}), "
      f"the 177-book collection, depth {DEPTH}.\n")

    w("## Why\n")
    w("Until V13 the local BM25 adapter returned `max_results` books for every query: rank_bm25 scores every "
      "book, and the list was cut without looking at the scores, so books that contain no query term (score 0) "
      "filled it. In a hybrid those books enter the RRF fusion as if the lexical side had retrieved them. V12 "
      "saw it descriptively (`PROFILE_PARITY.md`). V13 fixed the adapter: it returns only books that contain a "
      "query term. The old behaviour stays behind `BM25SearchRepository(zero_score_padding=True)`, used only by "
      "the evaluation paths that reproduce published numbers.\n")

    w("## Design (fixed before the run)\n")
    w("Shared harness of V06 (`retrieval_ablation`: `ModelStack`, `make_searcher`, `collect_run`); the "
      "`SearchService` is unchanged (RRF k = 60, 2 x 10 candidates per side, 2 x 30 with the reranker, which "
      "re-orders the top 30). The published arms run exactly as declared in `retrieval_ablation.SYSTEMS` "
      "(legacy padding); the `_nopad` arms are the same systems with the fixed adapter.\n")
    rows = [[f"`{s.name}`", "yes" if s.lexical and s.zero_score_padding else ("no" if s.lexical else "-"),
             s.description] for s in SYSTEMS]
    w(_table(["System", "BM25 padding", "Description"], rows) + "\n")
    w("Pre-declared contrasts (nDCG@10, system minus reference), one Holm family of three: "
      + "; ".join(f"{n} `{s}` vs `{r}` ({q})" for n, s, r, q in CONTRASTS)
      + ". Two-sided paired randomisation test (sign flips) and paired percentile bootstrap 95% interval, "
        "10,000 resamples, seed 0. Everything else is descriptive.\n")
    graded_here = j.get("judged_in_this_run", 0) + j.get("already_cached", 0)
    counts = a.get("judgment_counts") or {}
    later = (counts["v13"] - counts["base"] - graded_here) if counts.get("base") is not None else 0
    w(f"Judgments: `{j['v13_file']}` = `{j['base_file']}` (sha256 {j['base_sha256'][:12]}...) plus "
      f"{graded_here} pairs graded by {j.get('judge_model', 'the V12 judge')} for this measurement "
      f"({j['holes_at_start']} holes at the start"
      + (f"; grades 0/1/2/3: {' / '.join(str(g) for g in j['grade_distribution'])}" if j.get("grade_distribution") else "")
      + (f"; {_f(j['reported_cost_usd'], digits=4)} USD" if j.get("reported_cost_usd") is not None else "")
      + "). Every returned top-10 result of every system has a judgment."
      + (f" The file has since received {later} more grades by the same judge, for books only the cloud stack "
         "returned (`cloud_live_check.py`); they judge no book of these runs but can raise a query's ideal DCG, "
         "and the tables here are computed on the file as it is now"
         f" ({counts['v13']} judgments, sha256 {str(report.get('judgments_v13_sha256'))[:12]}...)." if later > 0 else "")
      + "\n")

    w("## Pre-declared contrasts\n")
    rows = []
    for name, c in a["contrasts"].items():
        rows.append([name, f"`{c['system']}`", f"`{c['reference']}`", _f(c["mean_system"]), _f(c["mean_reference"]),
                     f"{_f(c['mean_difference'], True)} [{_f(c['ci_low'], True)}, {_f(c['ci_high'], True)}]",
                     _p(c["p_value"]), _p(c["p_value_holm"])])
    w(_table(["Contrast", "System", "Reference", "nDCG@10 system", "nDCG@10 reference", "Difference [95% CI]",
              "p", "p (Holm)"], rows) + "\n")

    w("## All systems (descriptive)\n")
    langs = sorted(next(iter(systems.values()))["by_language"])
    rows = []
    for name, data in systems.items():
        m, lengths = data["metrics"], data["list_lengths"]
        rows.append([f"`{name}`", _ci(m[PRIMARY_METRIC]), _f(m["recall_at_10"]["mean"]), _f(m["precision_at_5"]["mean"]),
                     _f(m["mrr"]["mean"]), _f(m["judged_at_10"]["mean"]), _f(lengths["judged_share_of_returned"]),
                     f"{lengths['mean']:.1f} ({lengths['queries_with_fewer_than_10']}, {lengths['queries_with_no_result']})",
                     _f(data["not_relevant_share_of_returned"]),
                     *[_ci(data["by_language"][lang][PRIMARY_METRIC]) for lang in langs]])
    w(_table(["System", "nDCG@10 [95% CI]", "Recall@10", "P@5", "MRR", "Judged@10", "Judged share of returned",
              "Results per query (queries < 10, with none)", "Share of returned judged 0",
              *[f"nDCG@10 {lang} [95% CI]" for lang in langs]], rows) + "\n")
    w("Judged@10 divides by 10; it is below 1.000 only where a system returned fewer than 10 books (every "
      "returned result is judged: judged share of returned = 1.000).\n")
    if a["descriptive"]:
        rows = [[k, _f(d["mean_system"]), _f(d["mean_reference"]),
                 f"{_f(d['mean_difference'], True)} [{_f(d['ci_low'], True)}, {_f(d['ci_high'], True)}]", _p(d["p_value"])]
                for k, d in a["descriptive"].items()]
        w("Descriptive differences (no correction, not part of the tested family):\n")
        w(_table(["Pair", "nDCG@10 system", "nDCG@10 reference", "Difference [95% CI]", "p (unadjusted)"], rows) + "\n")

    pad = report.get("padding") or {}
    if pad:
        w("## How many padding books entered the fusion\n")
        w("Legacy BM25 list minus the fixed one, per query, at the depth each consumer asks for (10: BM25 alone; "
          f"{FUSION_CANDIDATES}: the hybrid's lexical candidates; {RERANK_CANDIDATES}: with the reranker).\n")
        rows = [[depth, _f(p["mean_padding_books"], digits=1), str(p["max_padding_books"]),
                 f"{p['queries_with_padding']} of {a['num_queries']}", "yes" if p["fixed_list_is_legacy_prefix"] else "NO"]
                for depth, p in pad.items()]
        w(_table(["Depth", "Mean padding books per query", "Max", "Queries with padding",
                  "Fixed list = legacy list without them (every query)"], rows) + "\n")

    vec = report.get("vector_scores")
    if vec:
        w("## The vector side\n")
        w("`FaissVectorSearchRepository` never pads: it returns at most `max_results` books, each once, fewer when "
          "the index or a filter leaves fewer (tests in `tests/infrastructure/test_faiss_vector_search_repository.py`). "
          "Dense retrieval has no \"no match\", though: every book has a cosine, so the nearest ten are returned "
          f"however far they are. `vec_e5` cosine at rank 1: median {_f(vec['rank_1']['median'])} (min "
          f"{_f(vec['rank_1']['min'])}); at rank 10: median {_f(vec['rank_10']['median'])} (min "
          f"{_f(vec['rank_10']['min'])}, max {_f(vec['rank_10']['max'])}), over {vec['queries_with_10_results']} "
          "queries with 10 results.\n")

    rep = a.get("reproduction") or {}
    if rep:
        w("## Reproduction of the published runs\n")
        rows = [[f"`{s}`", f"`{r['reference_run']}` in `{r['reference_file']}`", f"{r['identical_lists']} of {r['num_queries']}"]
                for s, r in rep.items()]
        w(_table(["System", "Published run", "Identical top-10 lists"], rows) + "\n")
        w("`hybrid_norm_e5_nopad` is compared with V12's descriptive `hybrid_norm_e5_matches` (the local BM25 "
          "without its zero-score results, fused with `vec_e5`).\n")

    lat = report.get("latency") or {}
    if lat:
        w("## Latency per query (descriptive, this machine)\n")
        rows = [[f"`{n}`", _f(v["median_ms"], digits=1), _f(v["p95_ms"], digits=1)] for n, v in lat.items()]
        w(_table(["System", "Median ms", "p95 ms"], rows) + "\n")

    w("## Limits\n")
    w("- 44 queries and 177 books: differences below about 0.08 nDCG@10 are hard to detect.")
    w("- The added grades come from an LLM judge, not from people.")
    w("- The fixed lexical lists are shorter, so Judged@10 is below 1.000 for the BM25-only arm although every "
      "returned book is judged.\n")

    w("## Reproduce\n")
    w("```\npython -m app.evaluation.bm25_padding run      # models; the judge only for missing pairs\n"
      "python -m app.evaluation.bm25_padding report   # recompute every table from the saved rankings\n```\n")
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
        p.add_argument("--v13-judgments", default=DEFAULT_V13_JUDGMENTS)
        p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
        p.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
        p.add_argument("--n-resamples", type=int, default=10000)
        p.add_argument("--seed", type=int, default=0)

    run = sub.add_parser("run", help="run every system, grade the holes and write the report")
    common(run)
    run.add_argument("--base-judgments", default=DEFAULT_BASE_JUDGMENTS)
    run.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    run.add_argument("--indexes-dir", default=DEFAULT_INDEXES_DIR)
    run.add_argument("--ablation-dir", default=DEFAULT_ABLATION_DIR)
    run.add_argument("--device", default=None)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--no-judge", action="store_true")
    report = sub.add_parser("report", help="recompute the tables from the saved rankings (no model, no judge)")
    common(report)
    return parser


def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    output = Path(args.output_json)
    if args.command == "run":
        saved = run_measurement(args, log)
        _write(args.output_json, json.dumps(saved, ensure_ascii=False, indent=2, default=str) + "\n")
    else:
        if not output.exists():
            log("No saved rankings yet: run `bm25_padding run` first")
            return 1
        saved = json.loads(output.read_text(encoding="utf-8"))
        saved.pop("analysis", None)
        saved.pop("judgments_v13_sha256", None)
    report = build_report(saved, args)
    _write(args.output_json, json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    _write(args.output_md, build_markdown(report))
    log(f"Wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
