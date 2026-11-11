"""
Retrieval ablation (task V06): which search components help on this catalog.

One controlled experiment on the 44 test queries and the 177-book catalog.
The design below was fixed before any result was seen; the module holds it
as data (SYSTEMS, CONTRASTS) so the report can print it as it was declared.

Design:
- Eleven systems (SYSTEMS), each returning 10 results (production depth):
  BM25 with the legacy whitespace tokenizer or the V01 normalizing one; FAISS
  with three embedding models (V02); hybrid RRF combinations of both; and
  the e5 hybrid with one of two cross-encoder rerankers (V04, depth 30).
  hybrid_legacy is the system in production today.
- Primary metric: nDCG@10 on judgments where every top-10 result of every
  system is judged. The holes are graded by the V05 LLM judge (with the
  stored query text) and merged into relevance_judgments_extended.json; the
  original relevance_judgments.json is not touched. Judged@10 = 1.0 is
  asserted for every system.
- Seven pre-declared contrasts (CONTRASTS): two-sided paired randomisation
  test on per-query nDCG@10, Holm correction across exactly these seven,
  and a paired bootstrap 95% interval for each difference.
- Descriptive only (no test): the full table of the eleven systems,
  Recall@10, P@5, MRR, the en/es breakdown, the same seven contrasts on the
  original judgments only (a sensitivity check: there new systems are
  penalised by unjudged results), the accent table (Spanish queries with
  and without diacritics) and the cost table (latency, model size, index
  build time).

Reproduction check: hybrid_legacy (and BM25 / MiniLM alone) must give the
V05 numbers published in data/evaluation/COMPARISON_EXTENDED_QRELS.md on the
judgments as they were before this task added any (the original ones plus
the LLM grades of the V05 pool, which is the top 10 of those three systems).

CLI (from the repository root):

    python -m app.evaluation.retrieval_ablation             # judge the holes, then report
    python -m app.evaluation.retrieval_ablation --no-judge  # no LLM: fail if a hole remains

Outputs: data/evaluation/results_retrieval_ablation.json and
data/evaluation/RETRIEVAL_ABLATION.md; in judge mode also
app/evaluation/relevance_judgments_extended.json. Indexes of the non-default
embedding models are built on first use under data/indexes_ablation/<slug>/
(git-ignored); the shipped data/indexes/ is only read. BM25 indexes are
built in memory.
"""

import argparse
import gc
import json
import logging
import platform
import shlex
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.domain.entities import Book, SearchResult
from app.domain.services import SearchService
from app.domain.value_objects import SearchQuery, SearchResponse
from app.evaluation.comparison import (
    PRIMARY_METRIC,
    _git,
    _sha256,
    compare_pairs,
    compare_systems,
)
from app.evaluation.evaluation_service import BookId, ndcg_at_k
from app.evaluation.relevance_judge import (
    GRADES,
    JudgeTask,
    JudgingStats,
    JudgmentCache,
    RelevanceJudge,
    find_unjudged,
    judge_pairs,
    merge_judgments,
)
from app.evaluation.statistics import paired_bootstrap_ci
from app.evaluation.types import TestQuery

logger = logging.getLogger(__name__)

# =============================================================================
# The experimental design (fixed in advance)
# =============================================================================

DEPTH = 10  # results per query, as in production
RERANK_DEPTH = 30  # top fused results the cross-encoder re-orders

WHITESPACE = "whitespace"  # legacy BM25 tokenizer (shipped index)
NORMALIZED = "normalized"  # V01 tokenizer: accents folded, stopwords removed

MINILM = "all-MiniLM-L6-v2"
E5 = "intfloat/multilingual-e5-large-instruct"
QWEN3 = "Qwen/Qwen3-Embedding-0.6B"
BGE = "BAAI/bge-reranker-v2-m3"
MMARCO = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@dataclass(frozen=True)
class SystemSpec:
    """
    One retrieval system of the ablation.

    Attributes:
        name: System name used in every table
        lexical: BM25 tokenizer name, or None without a lexical side
        embedding: Embedding model id, or None without a vector side
        reranker: Cross-encoder model id, or None without re-ranking
        description: One line for the report
    """

    name: str
    lexical: Optional[str] = None
    embedding: Optional[str] = None
    reranker: Optional[str] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.lexical is None and self.embedding is None:
            raise ValueError(f"System '{self.name}' needs a lexical or a vector side")
        if self.reranker is not None and self.kind != "hybrid":
            raise ValueError(f"System '{self.name}': a reranker needs a hybrid system")

    @property
    def kind(self) -> str:
        """'lexical', 'vector' or 'hybrid'."""
        if self.lexical and self.embedding:
            return "hybrid"
        return "lexical" if self.lexical else "vector"


SYSTEMS: Tuple[SystemSpec, ...] = (
    SystemSpec("bm25_whitespace", lexical=WHITESPACE,
               description="BM25, legacy whitespace tokenizer (as the shipped index)"),
    SystemSpec("bm25_normalized", lexical=NORMALIZED,
               description="BM25, normalizing tokenizer (accents folded, punctuation split, stopwords removed)"),
    SystemSpec("vec_minilm", embedding=MINILM,
               description=f"FAISS, {MINILM} (shipped index data/indexes/faiss_index)"),
    SystemSpec("vec_e5", embedding=E5, description=f"FAISS, {E5} (cosine index)"),
    SystemSpec("vec_qwen3", embedding=QWEN3, description=f"FAISS, {QWEN3} (cosine index)"),
    SystemSpec("hybrid_legacy", lexical=WHITESPACE, embedding=MINILM,
               description="RRF(bm25_whitespace, vec_minilm): the system in production today"),
    SystemSpec("hybrid_norm_minilm", lexical=NORMALIZED, embedding=MINILM,
               description="RRF(bm25_normalized, vec_minilm)"),
    SystemSpec("hybrid_norm_e5", lexical=NORMALIZED, embedding=E5,
               description="RRF(bm25_normalized, vec_e5)"),
    SystemSpec("hybrid_norm_qwen3", lexical=NORMALIZED, embedding=QWEN3,
               description="RRF(bm25_normalized, vec_qwen3)"),
    SystemSpec("hybrid_norm_e5_bge", lexical=NORMALIZED, embedding=E5, reranker=BGE,
               description=f"hybrid_norm_e5 + {BGE} over the top {RERANK_DEPTH}"),
    SystemSpec("hybrid_norm_e5_mmarco", lexical=NORMALIZED, embedding=E5, reranker=MMARCO,
               description=f"hybrid_norm_e5 + {MMARCO} over the top {RERANK_DEPTH}"),
)


@dataclass(frozen=True)
class Contrast:
    """A pre-declared comparison: system - reference on per-query nDCG@10."""

    name: str
    system: str
    reference: str
    question: str


CONTRASTS: Tuple[Contrast, ...] = (
    Contrast("C1", "bm25_normalized", "bm25_whitespace", "Does the normalizing tokenizer improve BM25?"),
    Contrast("C2", "vec_e5", "vec_minilm", "Does multilingual-e5-large-instruct improve the vector side?"),
    Contrast("C3", "vec_qwen3", "vec_minilm", "Does Qwen3-Embedding-0.6B improve the vector side?"),
    Contrast("C4a", "hybrid_norm_e5", "hybrid_legacy", "Do tokenizer + e5 together improve the production hybrid?"),
    Contrast("C4b", "hybrid_norm_qwen3", "hybrid_legacy",
             "Do tokenizer + Qwen3 together improve the production hybrid?"),
    Contrast("C5", "hybrid_norm_e5_bge", "hybrid_norm_e5", "Does the bge cross-encoder improve the e5 hybrid?"),
    Contrast("C6", "hybrid_norm_e5_mmarco", "hybrid_norm_e5", "Does the mMARCO cross-encoder improve the e5 hybrid?"),
)

# Systems of the V05 fill (lexical_only, vector_only, hybrid_rrf there; hybrid_rrf_mmr
# only re-orders hybrid_rrf's top 10, so it adds no book to the pool).
V05_SYSTEMS = {"bm25_whitespace": "lexical_only", "vec_minilm": "vector_only", "hybrid_legacy": "hybrid_rrf"}

# Numbers published in data/evaluation/COMPARISON_EXTENDED_QRELS.md (V05), three decimals.
V05_REFERENCE: Dict[str, Dict[str, Dict[str, float]]] = {
    "original": {
        "bm25_whitespace": {"ndcg_at_10": 0.460, "recall_at_10": 0.441, "precision_at_5": 0.423, "mrr": 0.571,
                            "judged_at_10": 0.507},
        "vec_minilm": {"ndcg_at_10": 0.530, "recall_at_10": 0.498, "precision_at_5": 0.500, "mrr": 0.712,
                       "judged_at_10": 0.532},
        "hybrid_legacy": {"ndcg_at_10": 0.531, "recall_at_10": 0.555, "precision_at_5": 0.486, "mrr": 0.678,
                          "judged_at_10": 0.550},
    },
    "extended_before_v06": {
        "bm25_whitespace": {"ndcg_at_10": 0.491, "recall_at_10": 0.415, "precision_at_5": 0.518, "mrr": 0.646,
                            "judged_at_10": 1.000},
        "vec_minilm": {"ndcg_at_10": 0.598, "recall_at_10": 0.572, "precision_at_5": 0.618, "mrr": 0.787,
                       "judged_at_10": 1.000},
        "hybrid_legacy": {"ndcg_at_10": 0.588, "recall_at_10": 0.549, "precision_at_5": 0.605, "mrr": 0.768,
                          "judged_at_10": 1.000},
    },
}

ACCENTED_SUFFIX = "@accented"  # run name of a system on the accented query texts

DEFAULT_DB_PATH = "data/catalog.db"
DEFAULT_INDEXES_DIR = "data/indexes"
DEFAULT_ABLATION_DIR = "data/indexes_ablation"
DEFAULT_QUERIES_PATH = "app/evaluation/test_queries.json"
DEFAULT_JUDGMENTS_PATH = "app/evaluation/relevance_judgments.json"
DEFAULT_EXTENDED_JUDGMENTS = "app/evaluation/relevance_judgments_extended.json"
DEFAULT_CACHE_DIR = "data/evaluation/llm_judgments"
DEFAULT_OUTPUT_JSON = "data/evaluation/results_retrieval_ablation.json"
DEFAULT_OUTPUT_MD = "data/evaluation/RETRIEVAL_ABLATION.md"
BUILD_INFO_FILE = "build_info.json"


# =============================================================================
# Running the systems
# =============================================================================

Searcher = Callable[[str], Sequence[SearchResult]]


class UnjudgedResultsError(RuntimeError):
    """A top-10 result has no judgment (raised by --no-judge instead of judging)."""


class IncompleteJudgingError(RuntimeError):
    """The judge left pairs without a grade (limit reached or failures)."""


class SearchDegradedError(RuntimeError):
    """A hybrid search fell back to BM25 only: the system did not run as declared."""


class RerankerFailedError(RuntimeError):
    """The reranker failed and SearchService silently kept the fused order."""


def check_hybrid_response(response: SearchResponse, *, system: str, expects_reranker: bool) -> None:
    """
    Make sure a hybrid search ran as declared; SearchService hides both failures.

    SearchService degrades to BM25 only when the vector side fails, and keeps
    the fused order with a warning when the reranker fails. Either would
    measure another system under this system's name.

    Raises:
        SearchDegradedError: If the response is degraded
        RerankerFailedError: If a reranker is expected and the top result has
            no rerank_score
    """
    if response.degraded:
        raise SearchDegradedError(f"{system}: search degraded ({response.degradation_reason})")
    if expects_reranker and response.results and response.results[0].rerank_score is None:
        raise RerankerFailedError(
            f"{system}: the top result has no rerank_score, so the reranker failed and the fused "
            f"order was returned (see the SearchService warning above)"
        )


def make_searcher(
    spec: SystemSpec,
    *,
    lexical=None,
    vector=None,
    embeddings=None,
    reranker=None,
    depth: int = DEPTH,
    rerank_depth: int = RERANK_DEPTH,
) -> Searcher:
    """
    The search function of a system: query text -> ranked results (at most depth).

    Lexical and vector systems make the same calls as run_lexical_search and
    run_vector_search in evaluation_job.py. Hybrid systems go through
    SearchService.search_with_fallback, like run_hybrid_search, with the
    reranker passed where the service is built, and every response is checked
    with check_hybrid_response.

    Args:
        spec: The system
        lexical: LexicalSearchRepository (lexical and hybrid systems)
        vector: VectorSearchRepository (vector and hybrid systems)
        embeddings: EmbeddingsStore of the vector side
        reranker: Reranker of the system, or None
        depth: Results per query
        rerank_depth: Top fused results the reranker scores
    """
    if spec.kind == "lexical":
        return lambda text: lexical.search(text, max_results=depth)
    if spec.kind == "vector":
        return lambda text: vector.search(embeddings.generate_query_embedding(text), max_results=depth)

    service = SearchService(
        lexical_search=lexical,
        vector_search=vector,
        embeddings_store=embeddings,
        reranker=reranker,
        rerank_depth=rerank_depth,
    )

    def search(text: str) -> Sequence[SearchResult]:
        query = SearchQuery(text=text, max_results=depth, use_explanations=False, use_diversification=False)
        response = service.search_with_fallback(query)
        check_hybrid_response(response, system=spec.name, expects_reranker=reranker is not None)
        return response.results

    return search


def collect_run(
    search: Searcher,
    texts: Mapping[str, str],
    *,
    depth: int = DEPTH,
    warmup_text: Optional[str] = None,
) -> Tuple[Dict[str, List[BookId]], Dict[str, float], Optional[float]]:
    """
    Run a system on every query and time each search.

    Args:
        search: The system's search function
        texts: Query id -> text to search
        depth: Results kept per query
        warmup_text: If given, searched once before the timed queries (it
            loads lazy models and warms caches); its time is returned apart

    Returns:
        (query id -> ranked book ids, query id -> latency in ms, warm-up ms or None)
    """
    warmup_ms = None
    if warmup_text is not None:
        started = time.perf_counter()
        search(warmup_text)
        warmup_ms = (time.perf_counter() - started) * 1000
    run: Dict[str, List[BookId]] = {}
    latency: Dict[str, float] = {}
    for query_id, text in texts.items():
        started = time.perf_counter()
        results = search(text)
        latency[query_id] = (time.perf_counter() - started) * 1000
        run[query_id] = [result.book.id for result in list(results)[:depth]]
    return run, latency, warmup_ms


def accented_texts(queries: Sequence[TestQuery]) -> Dict[str, str]:
    """Query id -> accented text, for the queries whose accented text differs from the stored one."""
    return {
        query.query_id: query.text_accented
        for query in queries
        if query.text_accented and query.text_accented != query.text
    }


def latency_summary(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Median and 95th percentile (linear interpolation) of per-query latencies."""
    values = np.asarray(list(latencies_ms), dtype=float)
    return {
        "n": int(values.size),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def execution_order(systems: Sequence[SystemSpec]) -> List[SystemSpec]:
    """
    Systems grouped by embedding model, so each model is loaded once.

    Lexical-only systems first, then one group per embedding model in order
    of first appearance; within a group the declared order is kept.
    """
    models: List[Optional[str]] = [None]
    for spec in systems:
        if spec.embedding not in models:
            models.append(spec.embedding)
    return sorted(systems, key=lambda spec: models.index(spec.embedding))


# =============================================================================
# Judgments
# =============================================================================


def resolve_grades(
    raw: Mapping[str, Sequence[Mapping[str, Any]]], books: Sequence[Book]
) -> Dict[str, Dict[BookId, int]]:
    """
    Query id -> book id -> grade, resolving (source, source_id) against the catalog.

    Same result as resolve_judgments_to_uuids in evaluation_job.py, with an
    in-memory lookup; judgments of books outside the catalog are dropped.
    """
    by_key = {(book.source, book.source_id): book.id for book in books}
    return {
        query_id: {
            by_key[(item["source"], item["source_id"])]: int(item["relevance"])
            for item in items
            if (item["source"], item["source_id"]) in by_key
        }
        for query_id, items in raw.items()
    }


def require_judged(
    runs: Mapping[str, Mapping[str, Sequence[BookId]]],
    judgments: Mapping[str, Mapping[BookId, int]],
    k: int = DEPTH,
) -> None:
    """
    Fail loudly if any top-k result of any run has no judgment.

    Raises:
        UnjudgedResultsError: Listing how many pairs are missing and a few of them
    """
    holes = find_unjudged(runs, judgments, k)
    if holes:
        shown = ", ".join(f"({query_id}, {book_id})" for query_id, book_id in holes[:5])
        raise UnjudgedResultsError(
            f"{len(holes)} (query, book) pairs in the top {k} of the runs have no judgment, e.g. {shown}. "
            f"Run `python -m app.evaluation.retrieval_ablation` without --no-judge to grade them "
            f"with the LLM judge, or check that the indexes and models are the ones the published run used."
        )


def llm_judges(raw: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Distinct (model, prompt_version) of the LLM judgments in a judgments file, with counts."""
    counts = Counter(
        (item.get("model"), item.get("prompt_version"))
        for items in raw.values() for item in items if item.get("judge") == "llm"
    )
    return [
        {"model": model, "prompt_version": version, "num_judgments": count}
        for (model, version), count in sorted(counts.items(), key=lambda entry: str(entry[0]))
    ]


# =============================================================================
# Report pieces
# =============================================================================


def _round3(value: float) -> str:
    return f"{value:.3f}"


def reproduction_check(
    runs: Mapping[str, Mapping[str, Sequence[BookId]]],
    queries: Sequence[TestQuery],
    original: Mapping[str, Mapping[BookId, int]],
    before_v06: Mapping[str, Mapping[BookId, int]],
    *,
    n_resamples: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """
    The V05 numbers of the legacy systems, recomputed and compared at three decimals.

    Returns:
        None if the legacy systems are not among the runs; otherwise, per
        judgment set and system, every metric of V05_REFERENCE with this
        run's value, the published value and whether they match, plus
        all_match.
    """
    if not all(system in runs for system in V05_SYSTEMS):
        return None
    legacy_runs = {system: runs[system] for system in V05_SYSTEMS}
    check: Dict[str, Any] = {"all_match": True}
    for label, judgments in (("original", original), ("extended_before_v06", before_v06)):
        report = compare_systems(legacy_runs, judgments, queries, None, n_resamples=n_resamples, seed=seed)
        block: Dict[str, Any] = {}
        for system, reference in V05_REFERENCE[label].items():
            block[system] = {}
            for metric, published in reference.items():
                value = report["systems"][system]["metrics"][metric]["mean"]
                match = _round3(value) == _round3(published)
                check["all_match"] = check["all_match"] and match
                block[system][metric] = {"value": value, "published": published, "match": match}
        check[label] = block
    return check


def accent_report(
    runs: Mapping[str, Mapping[str, Sequence[BookId]]],
    accented_runs: Mapping[str, Mapping[str, Sequence[BookId]]],
    queries: Sequence[TestQuery],
    judgments: Mapping[str, Mapping[BookId, int]],
    *,
    n_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """
    nDCG@10 of each system on the Spanish queries with the stored and the accented text.

    A query without an accented variant keeps its stored ranking in the
    accented column (its text is the same). Differences are accented -
    stored with a paired bootstrap interval; descriptive only.
    """
    spanish = [q for q in queries if q.language == "es" and judgments.get(q.query_id)]
    with_variant = accented_texts(spanish)
    changed = [q for q in spanish if q.query_id in with_variant]
    result: Dict[str, Any] = {
        "num_spanish_queries": len(spanish),
        "num_changed_queries": len(changed),
        "changed_query_ids": [q.query_id for q in changed],
        "systems": {},
    }
    for system, run in runs.items():
        accented_run = accented_runs.get(system, {})

        def scores(subset: Sequence[TestQuery], use_accented: bool) -> List[float]:
            return [
                ndcg_at_k(
                    list(accented_run.get(q.query_id, run[q.query_id]) if use_accented else run[q.query_id]),
                    judgments[q.query_id], k=DEPTH,
                )
                for q in subset
            ]

        entry: Dict[str, Any] = {}
        for label, subset in (("spanish", spanish), ("changed", changed)):
            if not subset:
                continue
            stored, accented = scores(subset, False), scores(subset, True)
            diff, low, high = paired_bootstrap_ci(accented, stored, n_resamples=n_resamples, seed=seed)
            entry[label] = {
                "num_queries": len(subset),
                "ndcg_at_10_stored": float(np.mean(stored)),
                "ndcg_at_10_accented": float(np.mean(accented)),
                "difference": diff,
                "ci_low": low,
                "ci_high": high,
            }
        entry["identical_top10_lists"] = sum(
            1 for q in changed if list(accented_run.get(q.query_id, [])) == list(run[q.query_id])
        )
        result["systems"][system] = entry
    return result


def _metric_means(report: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """System -> metric -> mean (no CI), for compact JSON blocks."""
    return {
        system: {metric: values["mean"] for metric, values in data["metrics"].items()}
        for system, data in report["systems"].items()
    }


# =============================================================================
# The ablation
# =============================================================================


def run_ablation(
    searcher_for: Callable[[SystemSpec], Searcher],
    queries: Sequence[TestQuery],
    books: Sequence[Book],
    original_raw: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    systems: Sequence[SystemSpec] = SYSTEMS,
    contrasts: Sequence[Contrast] = CONTRASTS,
    extended_raw: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    judge: Optional[RelevanceJudge] = None,
    cache: Optional[JudgmentCache] = None,
    limit: Optional[int] = None,
    n_resamples: int = 10000,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    """
    Run every system, complete the judgments and compute the report.

    Two modes:
    - judge mode (judge and cache given): every (query, book) pair in the
      top 10 of any run (stored or accented query text) without a judgment
      is graded by the judge with the stored query text; the merged
      judgments (original + every cached LLM grade) are returned for writing.
    - --no-judge mode (extended_raw given): the judgments are taken as they
      are and a single unjudged top-10 result raises UnjudgedResultsError.

    Args:
        searcher_for: Builds the search function of a system; called once per
            system, in execution_order
        queries: Test queries (text_accented is used for the accent runs)
        books: The catalog (to resolve judgments and to show books to the judge)
        original_raw: Original judgments, as load_relevance_judgments returns
        systems: Systems to run (declared order is the table order)
        contrasts: Pre-declared contrasts (one Holm family)
        extended_raw: Final judgments for --no-judge mode
        judge: Relevance judge for judge mode
        cache: Judgment cache of that judge
        limit: Judge at most this many new pairs (judge mode)
        n_resamples: Bootstrap and permutation resamples
        seed: Random seed of the statistics
        log: Progress output

    Returns:
        (report dict, merged judgments to write or None in --no-judge mode)

    Raises:
        ValueError: If neither or both modes are configured
        UnjudgedResultsError: --no-judge mode with a hole in the judgments
        IncompleteJudgingError: Judge mode when pairs remain without a grade
    """
    if (extended_raw is None) == (judge is None or cache is None):
        raise ValueError("Give either extended_raw (--no-judge) or judge and cache (judge mode)")
    names = {spec.name for spec in systems}
    for contrast in contrasts:
        if contrast.system not in names or contrast.reference not in names:
            raise ValueError(f"Contrast {contrast.name} names a system that is not run")

    stored = {query.query_id: query.text for query in queries}
    accented = accented_texts(queries)
    books_by_id = {book.id: book for book in books}

    # 1. Run every system on the stored texts (timed) and on the accented ones.
    runs: Dict[str, Dict[str, List[BookId]]] = {}
    accented_runs: Dict[str, Dict[str, List[BookId]]] = {}
    latency: Dict[str, Dict[str, Any]] = {}
    for spec in execution_order(systems):
        log(f"Running {spec.name} ({len(stored)} queries + {len(accented)} accented variants)")
        search = searcher_for(spec)
        runs[spec.name], per_query_ms, warmup_ms = collect_run(search, stored, warmup_text=queries[0].text)
        accented_runs[spec.name], _, _ = collect_run(search, accented)
        latency[spec.name] = {**latency_summary(list(per_query_ms.values())), "warmup_ms": warmup_ms}
        del search  # lets the next system release this one's models
    runs ={spec.name: runs[spec.name] for spec in systems}  # declared order
    all_runs = {**runs, **{name + ACCENTED_SUFFIX: run for name, run in accented_runs.items()}}

    # 2. Complete the judgments (or check that they are complete).
    original = resolve_grades(original_raw, books)
    merged: Optional[List[Dict[str, Any]]] = None
    judging: Optional[Dict[str, Any]] = None
    if judge is None:
        final_raw = dict(extended_raw)
        final = resolve_grades(final_raw, books)
        require_judged(all_runs, final, DEPTH)
    else:
        current = resolve_grades(
            {entry["query_id"]: entry["judgments"] for entry in merge_judgments(original_raw, cache.judgments())},
            books,
        )
        holes = find_unjudged(all_runs, current, DEPTH)
        # The judge sees the stored query text, also for books that only the
        # accented variant retrieved: the information need is the same.
        tasks = [JudgeTask(query_id, stored[query_id], books_by_id[book_id]) for query_id, book_id in holes]
        stats: JudgingStats = judge_pairs(tasks, judge, cache, limit=limit, log=log)
        if stats.remaining:
            raise IncompleteJudgingError(
                f"{stats.remaining} of {len(holes)} pairs still have no LLM judgment "
                f"({stats.failed} judge failures in this run); run the same command again to finish. "
                f"Nothing was written."
            )
        merged = merge_judgments(original_raw, cache.judgments())
        final_raw = {entry["query_id"]: entry["judgments"] for entry in merged}
        final = resolve_grades(final_raw, books)
        require_judged(all_runs, final, DEPTH)
        judging = {
            "judge_model": judge.model_name,
            "prompt_version": judge.prompt_version,
            "holes_at_start": len(holes),
            "judged_in_this_run": stats.judged,
            "already_cached": stats.cached,
            "failed_in_this_run": stats.failed,
            "seconds_per_judgment": stats.seconds_per_judgment,
            "llm_grade_distribution_of_holes": [
                Counter(cache.get(task.key).grade for task in tasks).get(grade, 0) for grade in GRADES
            ],
        }

    # 3. The judgments as they were before this task (V05): original + the
    #    LLM grades of the V05 pool (top 10 of the legacy systems).
    v05_pool = set(find_unjudged({s: runs[s] for s in V05_SYSTEMS if s in runs}, original, DEPTH))
    before_v06 = {
        query_id: {
            book_id: grade for book_id, grade in grades.items()
            if book_id in original.get(query_id, {}) or (query_id, book_id) in v05_pool
        }
        for query_id, grades in final.items()
    }
    added_by_v06 = find_unjudged(all_runs, before_v06, DEPTH)

    # 4. Metrics on the final and on the original judgments.
    final_report = compare_systems(runs, final, queries, None, n_resamples=n_resamples, seed=seed)
    original_report = compare_systems(runs, original, queries, None, n_resamples=n_resamples, seed=seed)
    not_fully_judged = {
        system: data["metrics"]["judged_at_10"]["mean"]
        for system, data in final_report["systems"].items()
        if data["metrics"]["judged_at_10"]["mean"] != 1.0
    }
    if not_fully_judged:
        raise UnjudgedResultsError(f"Judged@10 is not 1.0 on the final judgments: {not_fully_judged}")

    pairs = [(c.name, c.system, c.reference) for c in contrasts]
    stats_args = dict(n_resamples=n_resamples, seed=seed)
    report: Dict[str, Any] = {
        "design": {
            "depth": DEPTH,
            "rerank_depth": RERANK_DEPTH,
            "primary_metric": PRIMARY_METRIC,
            "systems": [
                {"name": s.name, "kind": s.kind, "bm25_tokenizer": s.lexical, "embedding_model": s.embedding,
                 "reranker_model": s.reranker, "description": s.description}
                for s in systems
            ],
            "contrasts": [
                {"name": c.name, "system": c.system, "reference": c.reference, "question": c.question}
                for c in contrasts
            ],
            "statistics": (
                f"two-sided paired randomisation test (sign flips) on per-query {PRIMARY_METRIC}, "
                f"{n_resamples} resamples, seed {seed}; Holm correction across exactly the "
                f"{len(contrasts)} contrasts; paired percentile bootstrap 95% CI over queries "
                f"({n_resamples} resamples, seed {seed})"
            ),
            "descriptive_only": [
                "the full systems table", "Recall@10, P@5, MRR", "the en/es breakdown",
                "the contrasts on the original judgments", "the accent table", "the cost table",
            ],
        },
        "num_queries": final_report["num_queries"],
        "languages": final_report["languages"],
        "queries_without_relevant": final_report["queries_without_relevant"],
        "systems": final_report["systems"],
        "original_judgments": {
            "metrics": _metric_means(original_report),
            "queries_without_relevant": original_report["queries_without_relevant"],
        },
        "contrasts": {
            "final": compare_pairs(final_report, pairs, **stats_args),
            "original": compare_pairs(original_report, pairs, **stats_args),
        },
        "accent": accent_report(runs, accented_runs, queries, final, **stats_args),
        "latency": latency,
        "judgments": {
            "num_original": sum(len(grades) for grades in original.values()),
            "num_final": sum(len(grades) for grades in final.values()),
            "num_before_v06": sum(len(grades) for grades in before_v06.values()),
            "num_llm_in_final_file": sum(
                1 for items in final_raw.values() for item in items if item.get("judge") == "llm"
            ),
            "llm_judges": llm_judges(final_raw),
            "v05_pool_pairs": len(v05_pool),
            "added_by_v06": {
                "pairs": len(added_by_v06),
                "grade_distribution": [
                    sum(1 for query_id, book_id in added_by_v06 if final[query_id][book_id] == grade)
                    for grade in GRADES
                ],
                "from_accented_runs_only": len(
                    set(added_by_v06) - set(find_unjudged(runs, before_v06, DEPTH))
                ),
            },
            "judging_run": judging,
        },
        "reproduction_check": reproduction_check(
            runs, queries, original, before_v06, n_resamples=n_resamples, seed=seed
        ),
        "runs": {system: {qid: [str(b) for b in run[qid]] for qid in run} for system, run in runs.items()},
        "accented_runs": {
            system: {qid: [str(b) for b in run[qid]] for qid in run} for system, run in accented_runs.items()
        },
    }
    return report, merged


# =============================================================================
# Real models
# =============================================================================


def resolve_device() -> str:
    """The torch device sentence-transformers would pick: cuda, then mps, then cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_size_on_disk(model_id: str) -> Optional[int]:
    """
    Bytes of the model's files in the local Hugging Face cache (None if not cached).

    Counts every file of the cached snapshot, so it is what was downloaded,
    which can include more than one weight format.
    """
    from huggingface_hub import snapshot_download

    repo_id = model_id if "/" in model_id else f"sentence-transformers/{model_id}"
    try:
        folder = Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        return None
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


class ModelStack:
    """
    Builds the search components of each system from the real indexes and models.

    Keeps at most one embedding model and one reranker in memory: systems run
    grouped by embedding model (execution_order), and a model is released
    when the next system needs another one. Records the cost of each model.

    - BM25: built in memory from the catalog with the requested tokenizer.
    - all-MiniLM-L6-v2: the shipped index data/indexes/faiss_index is loaded
      (it is the production one, needed to reproduce hybrid_legacy); its
      build time is measured by re-encoding the catalog in a throwaway store.
    - Other embedding models: index loaded from
      <ablation_dir>/<slug>/faiss_index, or built there with the V02 code on
      first use (the build time is kept in build_info.json next to it).
    - Rerankers: CrossEncoderReranker, loaded by scoring one pair.
    """

    def __init__(self, books: Sequence[Book], *, indexes_dir: str, ablation_dir: str, device: str,
                 log: Callable[[str], None] = print) -> None:
        self._books = list(books)
        self._indexes_dir = Path(indexes_dir)
        self._ablation_dir = Path(ablation_dir)
        self._device = device
        self._log = log
        self._bm25: Dict[str, Any] = {}
        self._embedding_model: Optional[str] = None
        self._embeddings = None
        self._vector = None
        self._reranker = None
        self.costs: Dict[str, Dict[str, Any]] = {}

    def searcher_for(self, spec: SystemSpec) -> Searcher:
        """Build (or reuse) the components of a system and return its search function."""
        self._reranker = None  # one reranker at a time
        lexical = self._lexical(spec.lexical) if spec.lexical else None
        if spec.embedding:
            self._load_embedding_model(spec.embedding)
        reranker = self._load_reranker(spec.reranker) if spec.reranker else None
        return make_searcher(spec, lexical=lexical, vector=self._vector, embeddings=self._embeddings,
                             reranker=reranker)

    def _lexical(self, tokenizer: str):
        from app.infrastructure.search.bm25_search_repository import BM25SearchRepository

        if tokenizer not in self._bm25:
            started = time.perf_counter()
            repo = BM25SearchRepository(tokenizer=tokenizer)
            repo.build_index(self._books)
            self.costs[f"bm25:{tokenizer}"] = {
                "role": "lexical", "index_build_seconds": time.perf_counter() - started,
                "index_source": "built in memory from the catalog",
            }
            self._bm25[tokenizer] = repo
        return self._bm25[tokenizer]

    def _encode_catalog(self, store) -> float:
        """Encode every book and build the FAISS index in store (V02 code); seconds taken."""
        started = time.perf_counter()
        texts = [book.get_searchable_text() for book in self._books]
        store.store_embeddings_batch([book.id for book in self._books], store.generate_embeddings_batch(texts))
        store.build_index()
        return time.perf_counter() - started

    def _release(self) -> None:
        """Drop the current embedding model and give its memory back."""
        self._embeddings = self._vector = None
        self._embedding_model = None
        gc.collect()
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _load_embedding_model(self, model: str) -> None:
        from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss, model_slug
        from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository

        if model == self._embedding_model:
            return
        self._release()
        self._log(f"Loading embedding model {model} on {self._device}")
        started = time.perf_counter()
        store = EmbeddingsStoreFaiss(model_name=model, device=self._device)
        load_seconds = time.perf_counter() - started

        if model == MINILM:
            index_dir = self._indexes_dir / "faiss_index"
            store.load_index(str(index_dir))
            throwaway = EmbeddingsStoreFaiss(model_name=model, device=self._device)
            build_seconds = self._encode_catalog(throwaway)
            del throwaway
            source = f"shipped index {index_dir} (read only); build time measured by a throwaway rebuild"
        else:
            folder = self._ablation_dir / model_slug(model)
            index_dir = folder / "faiss_index"
            info_path = folder / BUILD_INFO_FILE
            if (index_dir / "faiss_index.bin").exists():
                store.load_index(str(index_dir))
                info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
                build_seconds = info.get("index_build_seconds")
                source = f"loaded from {index_dir} (built {info.get('built_at', 'at an unknown date')})"
            else:
                self._log(f"Building the {model} index in {index_dir} (first use)")
                build_seconds = self._encode_catalog(store)
                store.save_index(str(index_dir))
                info_path.write_text(json.dumps({
                    "model": model, "device": self._device, "index_build_seconds": build_seconds,
                    "num_books": len(self._books), "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }, indent=2) + "\n", encoding="utf-8")
                source = f"built by this run in {index_dir}"

        metadata = store.get_index_metadata()
        self.costs[model] = {
            "role": "embedding",
            "device": store.get_device(),
            "dimension": store.get_dimension(),
            "index_metric": metadata.metric if metadata else None,
            "size_on_disk_bytes": model_size_on_disk(model),
            "load_seconds": load_seconds,
            "index_build_seconds": build_seconds,
            "index_source": source,
        }
        self._embeddings = store
        self._vector = FaissVectorSearchRepository(store, self._books)
        self._embedding_model = model

    def _load_reranker(self, model: str):
        from app.infrastructure.search.cross_encoder_reranker import CrossEncoderReranker

        self._log(f"Loading reranker {model} on {self._device}")
        reranker = CrossEncoderReranker(model, device=self._device)
        probe = SearchResult(book=self._books[0], final_score=0.0, rank=1, source="hybrid")
        started = time.perf_counter()
        reranker.rerank("warm-up", [probe])  # loads the model; raises if it cannot
        self.costs[model] = {
            "role": "reranker",
            "device": self._device,
            "size_on_disk_bytes": model_size_on_disk(model),
            "load_seconds": time.perf_counter() - started,
            "index_build_seconds": None,
            "index_source": None,
        }
        self._reranker = reranker
        return reranker


# =============================================================================
# Markdown
# =============================================================================


def _fmt(value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _fmt_ci(value: float, low: float, high: float, signed: bool = False) -> str:
    return f"{_fmt(value, signed)} [{_fmt(low, signed)}, {_fmt(high, signed)}]"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _contrast_table(contrasts: Mapping[str, Mapping[str, Any]], design: Mapping[str, Any],
                    judged_original: Optional[Mapping[str, float]] = None) -> str:
    questions = {c["name"]: c["question"] for c in design["contrasts"]}
    header = ["Contrast", "System", "Reference", "nDCG@10 system", "nDCG@10 reference",
              "Diff. [95% CI]", "p", "p (Holm)"]
    if judged_original is not None:
        header += ["Judged@10 system / reference"]
    rows = []
    for name, c in contrasts.items():
        row = [
            f"{name}: {questions[name]}", f"`{c['system']}`", f"`{c['reference']}`",
            _fmt(c["mean_system"]), _fmt(c["mean_reference"]),
            _fmt_ci(c["mean_difference"], c["ci_low"], c["ci_high"], signed=True),
            _fmt_p(c["p_value"]), _fmt_p(c["p_value_holm"]),
        ]
        if judged_original is not None:
            row.append(f"{_fmt(judged_original[c['system']])} / {_fmt(judged_original[c['reference']])}")
        rows.append(row)
    return _table(header, rows)


def _mib(size: Optional[int]) -> str:
    return "n/a" if size is None else f"{size / 2**20:,.0f} MiB"


def _seconds(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def build_markdown(report: Mapping[str, Any]) -> str:
    """RETRIEVAL_ABLATION.md: design, results tables, costs and provenance."""
    run, design, judgments = report["run"], report["design"], report["judgments"]
    commit = run.get("git_commit") or "unknown"
    dirty = " (with uncommitted changes in tracked files)" if run.get("git_worktree_dirty") else ""
    languages = ", ".join(f"{n} {lang}" for lang, n in report["languages"].items())
    systems = "\n".join(f"- `{s['name']}`: {s['description']}" for s in design["systems"])
    contrasts = "\n".join(
        f"- {c['name']}: `{c['system']}` vs `{c['reference']}`. {c['question']}" for c in design["contrasts"]
    )
    judges = "; ".join(
        f"`{j['model']}`, prompt `{j['prompt_version']}` ({j['num_judgments']} judgments)"
        for j in judgments["llm_judges"]
    ) or "none"
    added = judgments["added_by_v06"]
    grades = " / ".join(str(n) for n in added["grade_distribution"])
    judging = judgments.get("judging_run")
    if judging:
        seconds = judging["seconds_per_judgment"]
        carried = judging.get("carried_from_run_of")
        who = f"The judging run of {carried}" if carried else "This run"
        judging_line = (
            f"{who} judged {judging['judged_in_this_run']} pairs ({judging['already_cached']} were already "
            f"in the cache, {judging['failed_in_this_run']} judge failures"
            + (f", {seconds:.2f} s per judgment, one request at a time" if seconds else "") + ")."
            + (" This run (`--no-judge`) judged nothing and found every top-10 result judged." if carried else "")
        )
    else:
        judging_line = "This run judged nothing (`--no-judge`): every top-10 result was already judged."

    # Reproduction check
    check = report.get("reproduction_check")
    if check:
        rows = []
        for label, title in (("extended_before_v06", "extended, before V06"), ("original", "original")):
            for system, metrics in check[label].items():
                for metric, entry in metrics.items():
                    rows.append([title, f"`{system}` ({V05_SYSTEMS[system]} in V05)", metric,
                                 _fmt(entry["value"]), _fmt(entry["published"]), "yes" if entry["match"] else "NO"])
        verdict = (
            "All values match: `hybrid_legacy` is the production system measured in V05."
            if check["all_match"] else "**Some values do not match; see the rows marked NO.**"
        )
        reproduction = (
            f"{verdict}\n\n"
            + _table(["Judgments", "System", "Metric", "This run", "V05 published", "Match"], rows)
        )
    else:
        reproduction = "Not run (the legacy systems are not part of this run)."

    # Systems table
    original_metrics = report["original_judgments"]["metrics"]
    rows = []
    for system, data in report["systems"].items():
        m = data["metrics"]
        rows.append([
            f"`{system}`", _fmt_ci(m["ndcg_at_10"]["mean"], m["ndcg_at_10"]["ci_low"], m["ndcg_at_10"]["ci_high"]),
            _fmt(m["recall_at_10"]["mean"]), _fmt(m["precision_at_5"]["mean"]), _fmt(m["mrr"]["mean"]),
            _fmt(m["judged_at_10"]["mean"]), _fmt(original_metrics[system]["ndcg_at_10"]),
            _fmt(original_metrics[system]["judged_at_10"]),
        ])
    systems_table = _table(
        ["System", "nDCG@10 [95% CI]", "Recall@10", "P@5", "MRR", "Judged@10",
         "nDCG@10 (original judgments)", "Judged@10 (original judgments)"],
        rows,
    )

    # Language table
    langs = list(report["languages"])
    rows = []
    for system, data in report["systems"].items():
        row = [f"`{system}`"]
        for lang in langs:
            stat = data["by_language"][lang][PRIMARY_METRIC]
            row.append(_fmt_ci(stat["mean"], stat["ci_low"], stat["ci_high"]))
        rows.append(row)
    language_table = _table(
        ["System"] + [f"nDCG@10 {lang} (n={report['languages'][lang]}) [95% CI]" for lang in langs], rows
    )

    # Accent table
    accent = report["accent"]
    rows = []
    for system, entry in accent["systems"].items():
        sp, ch = entry["spanish"], entry.get("changed")
        row = [f"`{system}`", _fmt(sp["ndcg_at_10_stored"]), _fmt(sp["ndcg_at_10_accented"]),
               _fmt_ci(sp["difference"], sp["ci_low"], sp["ci_high"], signed=True)]
        if ch:
            row += [_fmt(ch["ndcg_at_10_stored"]), _fmt(ch["ndcg_at_10_accented"]),
                    _fmt_ci(ch["difference"], ch["ci_low"], ch["ci_high"], signed=True),
                    f"{entry['identical_top10_lists']} of {ch['num_queries']}"]
        rows.append(row)
    n_es, n_ch = accent["num_spanish_queries"], accent["num_changed_queries"]
    accent_table = _table(
        ["System", f"es stored (n={n_es})", f"es accented (n={n_es})", "Diff. accented - stored [95% CI]",
         f"changed stored (n={n_ch})", f"changed accented (n={n_ch})", "Diff. [95% CI]",
         "Identical top 10"],
        rows,
    )

    # Cost tables
    latency = report["latency"]
    rows = [
        [f"`{system}`", f"{latency[system]['median_ms']:.1f}", f"{latency[system]['p95_ms']:.1f}",
         f"{latency[system]['warmup_ms']:.0f}" if latency[system]["warmup_ms"] is not None else "n/a"]
        for system in report["systems"]
    ]
    latency_table = _table(["System", "Median ms / query", "p95 ms / query", "Warm-up query ms"], rows)
    costs = report.get("costs", {})
    rows = [
        [f"`{name}`", c["role"], c.get("device") or "cpu", _mib(c.get("size_on_disk_bytes")),
         _seconds(c.get("load_seconds")), _seconds(c.get("index_build_seconds")),
         str(c.get("dimension") or ""), c.get("index_source") or ""]
        for name, c in costs.items()
    ]
    model_table = _table(
        ["Component", "Role", "Device", "Size on disk", "Load", "Index build (177 books)", "Dimension",
         "Index"], rows,
    )
    llm_note = run.get("llm_server_note", "")

    commands = "\n".join(f"    {command}" for command in run["reproduce"])
    return f"""# Retrieval ablation (V06)

- Date: {run["date"]}
- Git commit: `{commit}`{dirty}
- This run: `{run["command"]}`
- Collection: {report["num_queries"]} queries ({languages}), {run["num_books"]}-book catalog.
- Judgments: `{run["extended_judgments_path"]}`, {judgments["num_final"]} judgments ({judgments["num_original"]}
  original resolved to the catalog + {judgments["num_llm_in_final_file"]} LLM). Before this task it held
  {judgments["num_before_v06"]}. V06 added {added["pairs"]} LLM-graded pairs (grades 0 / 1 / 2 / 3: {grades};
  {added["from_accented_runs_only"]} of them retrieved only by the accented query variants). {judging_line}
- LLM judges in the file: {judges}. The original `relevance_judgments.json` is unchanged.

Reproduce (from the repository root):

{commands}

The first command judges any missing pair with the local LLM (see `.env.example`, `JUDGE_*`); the
second needs no LLM and fails if any top-10 result has no judgment.

## Design (fixed before the results were seen)

The design is held as data in `app/evaluation/retrieval_ablation.py` (`SYSTEMS`, `CONTRASTS`) and was
written before any system of this ablation was scored. Every system returns {design["depth"]} results
per query (production depth). Systems:

{systems}

Primary metric: nDCG@10 (gain 2^grade - 1, a book without a judgment counts as not relevant), on
judgments in which every top-10 result of every system is judged: the holes are graded by the V05
LLM judge with the stored query text and merged into the extended judgments file. Judged@10 is
asserted to be 1.000 for every system.

Pre-declared primary contrasts (system - reference):

{contrasts}

Statistics: {design["statistics"]}.

Descriptive only, no test: {", ".join(design["descriptive_only"])}. The contrasts on the original
judgments are a sensitivity check: there, books outside the original pool count as not relevant,
which penalises the new systems.

With {report["num_queries"]} queries, differences in nDCG@10 below about 0.08 are hard to detect.

## Reproduction check

`hybrid_legacy` must reproduce the `hybrid_rrf` numbers of `COMPARISON_EXTENDED_QRELS.md` on the
judgments as they were before this task (original + the LLM grades of the V05 pool, the top 10 of
the three legacy systems; {judgments["v05_pool_pairs"]} pairs).

{reproduction}

## Pre-declared contrasts (final judgments)

{_contrast_table(report["contrasts"]["final"], design)}

## Sensitivity: the same contrasts on the original judgments only (descriptive)

{_contrast_table(report["contrasts"]["original"], design,
                 {s: m["judged_at_10"] for s, m in original_metrics.items()})}

## All systems (final judgments; descriptive)

{systems_table}

## By query language (final judgments; descriptive)

{language_table}

## Accent robustness (Spanish queries; descriptive)

The Spanish test queries are stored without diacritics. {n_ch} of the {n_es} have an accented variant
(`text_accented` in `test_queries.json`) that differs from the stored text only in diacritics (the
stored capitalization is kept). Each system was run on both; a query without a variant keeps its
stored ranking in the accented column. Same judgments for both texts (same information need).

{accent_table}

## Cost (descriptive)

Device: `{run["device"]}` ({run["hardware"]}). Latency is per query over the {report["num_queries"]}
queries, after one warm-up query, measured in the same process right after each system was built.
{llm_note}

{latency_table}

{model_table}

Size on disk: files of the model's snapshot in the local Hugging Face cache. Load: model load
(rerankers: load plus scoring one pair). Index build: encode the 177 books and build the FAISS index.

## Notes

- `vec_minilm` (and the two hybrids built on it) search the shipped index `data/indexes/faiss_index`
  (IndexFlatL2 over MiniLM's unit-length vectors, which ranks as cosine), because that index is the
  production one that `hybrid_legacy` must reproduce. A cosine index rebuilt with the V02 code has the
  same vectors (largest difference 1.5e-7) and differs only in the order of exactly tied books.
- The reranked systems retrieve {design["rerank_depth"] * 2} candidates per retriever instead of 20
  (`SearchService` asks for at least the rerank depth), so C5 and C6 measure the re-ranking stage as it
  would run in production, larger candidate pool included.
- BM25 always returns 10 books; when fewer than 10 match, the rest have score 0 and come in catalog
  order. Those positions are judged like any other.
- LLM grades are not human grades (agreement with the original judgments: `JUDGE_AGREEMENT.md`).
- Run the commands online even with every model cached: with transformers 4.57.3, `HF_HUB_OFFLINE=1`
  makes loading the e5 and Qwen3 tokenizers fail (the library asks the Hub for model metadata; the
  same problem V04 worked around for the cross-encoders).
"""


# =============================================================================
# CLI
# =============================================================================


def _hardware() -> str:
    """CPU name on macOS (sysctl), else the platform string."""
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.platform()


def _resident_llm_models() -> Optional[List[str]]:
    """Models an Ollama server holds in memory (its /api/ps), or None if no server answers."""
    import os

    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=2) as response:
            return [model["name"] for model in json.load(response).get("models", [])]
    except Exception:
        return None


def _versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package in ("torch", "sentence-transformers", "transformers", "faiss-cpu", "rank-bm25"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Retrieval ablation with pre-declared contrasts (V06)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Do not call the LLM judge; fail if any top-10 result has no judgment")
    parser.add_argument("--limit", type=int, default=None,
                        help="Judge at most this many new pairs, then stop (cached for the next run)")
    parser.add_argument("--device", default=None, help="Torch device for every model (default: cuda, mps or cpu)")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--indexes-dir", default=DEFAULT_INDEXES_DIR, help="Shipped indexes (read only)")
    parser.add_argument("--ablation-dir", default=DEFAULT_ABLATION_DIR,
                        help="Where indexes of non-default embedding models are built on first use")
    parser.add_argument("--queries-path", default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--judgments-path", default=DEFAULT_JUDGMENTS_PATH, help="Original judgments (read only)")
    parser.add_argument("--extended-judgments-path", default=DEFAULT_EXTENDED_JUDGMENTS,
                        help="Extended judgments: read with --no-judge, written otherwise")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD,
                        help="Markdown report; an empty value writes only the JSON (e.g. for a judge-sensitivity run)")
    return parser


def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def previous_judging_run(previous_json: str, extended_judgments_sha256: str) -> Optional[Dict[str, Any]]:
    """
    The judging-run facts of an earlier report, if it wrote the same judgments file.

    A --no-judge run regenerates the report without judging; when the
    extended judgments are byte for byte the ones an earlier judging run
    wrote (same SHA-256), that run's facts (pairs judged, failures, time per
    judgment) still describe them, so they are carried over and dated.

    Returns:
        The earlier judging_run block with carried_from_run_of set, or None
    """
    try:
        previous = json.loads(Path(previous_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    judging = previous.get("judgments", {}).get("judging_run")
    run = previous.get("run", {})
    if not judging or run.get("extended_judgments_sha256") != extended_judgments_sha256:
        return None
    return {**judging, "carried_from_run_of": judging.get("carried_from_run_of") or run.get("date")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ablation and write the JSON and Markdown reports."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    device = args.device or resolve_device()  # imports torch before faiss (macOS)
    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.evaluation.relevance_judge import _build_judge, _judge_description
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

    started = datetime.now(UTC)
    books = SqliteBookCatalogRepository(Path(args.db_path)).get_all()
    queries = load_test_queries(args.queries_path)
    original_raw = load_relevance_judgments(args.judgments_path)

    judge = cache = extended_raw = None
    judge_info = None
    if args.no_judge:
        extended_raw = load_relevance_judgments(args.extended_judgments_path)
    else:
        settings, judge = _build_judge()
        cache = JudgmentCache.for_judge(args.cache_dir, judge.model_name, judge.prompt_version)
        judge_info = _judge_description(settings, judge)

    resident = _resident_llm_models()
    stack = ModelStack(books, indexes_dir=args.indexes_dir, ablation_dir=args.ablation_dir, device=device)
    try:
        report, merged = run_ablation(
            stack.searcher_for, queries, books, original_raw,
            extended_raw=extended_raw, judge=judge, cache=cache, limit=args.limit,
            n_resamples=args.n_resamples, seed=args.seed,
        )
    except IncompleteJudgingError as error:
        print(error)
        return 1

    if merged is not None:
        _write(args.extended_judgments_path, json.dumps(merged, ensure_ascii=False, indent=2) + "\n")

    command = " ".join(["python -m app.evaluation.retrieval_ablation", *map(shlex.quote, raw_args)]).strip()
    report["costs"] = stack.costs
    report["judge"] = judge_info
    report["run"] = {
        "date": started.isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_worktree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "command": command,
        "reproduce": [
            "python -m app.evaluation.retrieval_ablation",
            "python -m app.evaluation.retrieval_ablation --no-judge",
        ],
        "device": device,
        "hardware": _hardware(),
        "library_versions": _versions(),
        "llm_server_resident_models": resident,
        "llm_server_note": (
            f"A local LLM server held {', '.join(f'`{m}`' for m in resident)} in memory while the systems "
            f"ran (it shares the GPU and unified memory; no LLM request ran during the timings), so "
            f"the timings are indicative." if resident else
            "No local LLM server answered when the run started; a GPU shared with other work would "
            "still make the timings indicative only."
        ),
        "num_books": len(books),
        "db_path": args.db_path,
        "indexes_dir": args.indexes_dir,
        "ablation_dir": args.ablation_dir,
        "queries_path": args.queries_path,
        "judgments_path": args.judgments_path,
        "extended_judgments_path": args.extended_judgments_path,
        "queries_sha256": _sha256(args.queries_path),
        "judgments_sha256": _sha256(args.judgments_path),
        "extended_judgments_sha256": _sha256(args.extended_judgments_path),
    }
    if args.no_judge:
        report["judgments"]["judging_run"] = previous_judging_run(
            args.output_json, report["run"]["extended_judgments_sha256"]
        )
    _write(args.output_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.output_md:
        _write(args.output_md, build_markdown(report))

    contrasts = report["contrasts"]["final"]
    for name, c in contrasts.items():
        print(f"{name} {c['system']} - {c['reference']}: {c['mean_difference']:+.3f} "
              f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] p={c['p_value']:.3f} p_holm={c['p_value_holm']:.3f}")
    check = report["reproduction_check"]
    if check is not None:
        print(f"Reproduction check: {'all values match' if check['all_match'] else 'MISMATCH, see the report'}")
    print(f"Saved {args.output_json}" + (f" and {args.output_md}" if args.output_md else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
