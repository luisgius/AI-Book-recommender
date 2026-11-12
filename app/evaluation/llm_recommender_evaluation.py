"""
The LLM as a recommender, measured against the retrieval pipeline (task V07).

Two questions of 2026, answered with data on the 177-book catalog and the 44
test queries (10 results each):
1. A model with a 1M-token context can read the whole catalog: is retrieval
   still needed?
2. LLMs are used directly as recommenders, and without grounding they
   recommend books that are not in the catalog: how often, here?

Design (fixed in advance, before any system was run):
- R0 `pipeline`: hybrid_norm_e5_bge of the V06 ablation (BM25 with the
  normalizing tokenizer + multilingual-e5 + RRF + BAAI/bge-reranker-v2-m3
  over the top 30), built by retrieval_ablation.ModelStack, not re-implemented.
- R1 `rag_llm_recommender`: the top 30 of hybrid_norm_e5 (the same
  candidates the bge reranker of R0 sees), re-ordered by one listwise LLM
  call (LLMListwiseReranker, the domain Reranker port) inside SearchService.
- R2 `long_context_recommender`: no retrieval; one LLM call with all 177
  books (same fields and truncation as R1) returns 10 ids.
- R3 `closed_book_recommender`: no catalog; the LLM names 10 real books
  (title, author), matched to the catalog with match_book (rule below).
- Primary metric nDCG@10 (gain 2^grade - 1) on judgments with every top-10
  result judged: the V10 file relevance_judgments_extended_z-ai-glm-5.3-flash.json,
  whose new holes are graded by the JUDGE_* judge (z-ai/glm-5.3-flash, another
  vendor than the system LLM) and written back to that file. Judged@10 = 1.0
  is asserted for R0, R1 and R2.
- Two pre-declared contrasts, L1 = R1 - R0 and L2 = R2 - R0: paired
  randomisation test (sign flips) on per-query nDCG@10, Holm correction
  across exactly these two, paired bootstrap 95% intervals. Everything else
  is descriptive (Recall@10, P@5, MRR, en/es breakdown, R3, grounding, cost,
  scaling, determinism, the hybrid_norm_e5 reference row).
- Grounding: for R1 and R2, ids returned that were not in the list given, and
  lists completed with the remaining candidates in the order shown; for R3,
  the share of recommendations found in the catalog, overall and by language.
- Cost: per query latency (median, p95), tokens and the cost OpenRouter
  reports, from the usage recorder of each call. R2's prompt size and cost
  are extrapolated (not run) to larger catalogs.
- Determinism: R1 and R2 run twice; identical top-10 lists are counted.

R3 matching rule: a recommendation matches a catalog book when their titles
are equal after normalisation (lower case, accents removed, every run of
non-alphanumeric characters as one space), both cut before the first ':',
'(', '[' or ' - ' (subtitles and edition notes ignored), and the surname of
the recommended author (its last word, or the words before a comma; only
the first of several authors) equals the surname of one of the book's
authors. The first matching catalog book is taken.

Every LLM output is cached on disk (one JSON Lines file per model and prompt
version under data/evaluation/llm_recommender/), keyed by a hash of the
system, run, model, prompt version and the exact messages, so a changed
candidate list or prompt is a cache miss. CLI (from the repository root):

    python -m app.evaluation.llm_recommender_evaluation            # call what is not cached
    python -m app.evaluation.llm_recommender_evaluation --no-llm   # caches only; fail if anything is missing

Outputs: data/evaluation/results_llm_recommender.json and
data/evaluation/LLM_RECOMMENDER.md (and, without --no-llm, the extended
judgments file and the judge's cache).
"""

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shlex
import statistics
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.domain.entities import Book
from app.evaluation.comparison import PRIMARY_METRIC, _git, _sha256, compare_pairs, compare_systems
from app.evaluation.evaluation_service import BookId, ndcg_at_k
from app.evaluation.relevance_judge import GRADES, JudgeTask, JudgmentCache, find_unjudged, merge_judgments
from app.evaluation.retrieval_ablation import (
    DEPTH,
    RERANK_DEPTH,
    SYSTEMS,
    Contrast,
    IncompleteJudgingError,
    RerankerFailedError,
    Searcher,
    _fmt,
    _fmt_ci,
    _fmt_p,
    _table,
    collect_run,
    latency_summary,
    require_judged,
    resolve_grades,
)
from app.evaluation.types import TestQuery
from app.infrastructure.llm.llm_recommender import (
    LONG_CONTEXT_OPERATION,
    RECOMMENDER_MAX_TOKENS,
    TOP_N,
    ClosedBookOutcome,
    Completion,
    LLMListwiseReranker,
    LLMOutput,
    LLMRecommender,
    LLMRequest,
    ListwiseOutcome,
    book_entry,
    candidate_ids,
    listwise_request,
)
from app.infrastructure.llm.prompts_recommender import (
    DESCRIPTION_MAX_CHARS,
    RECOMMENDER_PROMPT_VERSION,
    prompt_texts,
)

logger = logging.getLogger(__name__)

# =============================================================================
# The design (fixed in advance)
# =============================================================================

R0 = "pipeline"
R1 = "rag_llm_recommender"
R2 = "long_context_recommender"
R3 = "closed_book_recommender"
REFERENCE = "hybrid_norm_e5"  # R1's candidate generator, descriptive only
RUN2_SUFFIX = "@run2"

R0_SPEC = next(spec for spec in SYSTEMS if spec.name == "hybrid_norm_e5_bge")
CANDIDATES_SPEC = next(spec for spec in SYSTEMS if spec.name == REFERENCE)

SYSTEM_DESCRIPTIONS = {
    R0: f"R0: `{R0_SPEC.name}` of V06 ({R0_SPEC.description}); no LLM in the ranking",
    R1: f"R1: top {RERANK_DEPTH} of `{REFERENCE}` re-ordered by one listwise LLM call (LLMListwiseReranker in SearchService)",
    R2: "R2: no retrieval; one LLM call with the whole catalog returns 10 ids",
    R3: "R3: no catalog; the LLM names 10 real books (title, author), matched to the catalog",
    REFERENCE: f"Reference (descriptive): `{REFERENCE}` of V06, the candidate generator of R1, without reranker",
}

CONTRASTS: Tuple[Contrast, ...] = (
    Contrast("L1", R1, R0, "Does the RAG LLM recommender (R1) improve on the pipeline (R0)?"),
    Contrast("L2", R2, R0, "Does the long-context LLM recommender, without retrieval (R2), improve on the pipeline (R0)?"),
)

NUM_RUNS = 2  # R1 and R2 run twice (determinism); R3 once
SCALING_SIZES = (5_000, 50_000, 200_000, 1_000_000)
SPEND_LIMIT_USD = 2.0  # stop the run beyond this reported spend
DEFAULT_WORKERS = 4  # LLM requests in flight (R2, R3 and the judge)

DEFAULT_DB_PATH = "data/catalog.db"
DEFAULT_INDEXES_DIR = "data/indexes"
DEFAULT_ABLATION_DIR = "data/indexes_ablation"
DEFAULT_QUERIES_PATH = "app/evaluation/test_queries.json"
DEFAULT_JUDGMENTS_PATH = "app/evaluation/relevance_judgments.json"
DEFAULT_EXTENDED_JUDGMENTS = "app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json"
DEFAULT_JUDGE_CACHE_DIR = "data/evaluation/llm_judgments"
DEFAULT_CACHE_DIR = "data/evaluation/llm_recommender"
DEFAULT_OUTPUT_JSON = "data/evaluation/results_llm_recommender.json"
DEFAULT_OUTPUT_MD = "data/evaluation/LLM_RECOMMENDER.md"
V06_RESULTS_JSON = "data/evaluation/results_retrieval_ablation_z-ai-glm-5.3-flash.json"
MODEL_INFO_FILE = "openrouter_model.json"


# =============================================================================
# Cache of LLM outputs
# =============================================================================


class MissingLLMOutputError(RuntimeError):
    """--no-llm: an output the report needs is not in the cache."""


class SpendLimitError(RuntimeError):
    """The reported spend of this run passed SPEND_LIMIT_USD."""


def cache_file_name(model: str, prompt_version: str) -> str:
    """File of the outputs of one model and prompt version."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
    return f"{slug}__prompt-{prompt_version}.jsonl"


def request_key(request: LLMRequest, *, system: str, run: int, model: str, prompt_version: str) -> str:
    """Hash of everything that determines an output: system, run, model, prompt version, messages."""
    payload = json.dumps(
        {"system": system, "run": run, "model": model, "prompt_version": prompt_version,
         "messages": request.messages()},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OutputCache:
    """
    Append-only JSON Lines file of LLM outputs, keyed by request_key.

    Each entry is written, flushed and synced as soon as the output arrives,
    so an interrupted run loses at most the calls in flight. Thread-safe.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping unreadable line %d of %s", number, self.path)
                    continue
                self._entries[entry["key"]] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._entries.get(key)

    def add(self, entry: Mapping[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._entries[entry["key"]] = dict(entry)


class SpendMeter:
    """Sums the cost reported by the calls made in this run; raises past the limit."""

    def __init__(self, limit_usd: float = SPEND_LIMIT_USD) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()

    def add(self, cost_usd: Optional[float]) -> None:
        with self._lock:
            self.spent_usd += cost_usd or 0.0
            if self.spent_usd > self.limit_usd:
                raise SpendLimitError(
                    f"Reported spend of this run is {self.spent_usd:.3f} USD, over the {self.limit_usd} USD limit"
                )


class CachedCompletion:
    """
    A Completion that answers from the cache and asks `inner` only on a miss.

    With inner=None (--no-llm) a miss raises MissingLLMOutputError. One
    instance serves one (system, run); elapsed_s sums the time spent inside
    calls, so a caller can subtract it from a timed search.
    """

    def __init__(
        self,
        cache: OutputCache,
        *,
        system: str,
        run: int,
        model: str,
        prompt_version: str = RECOMMENDER_PROMPT_VERSION,
        inner: Optional[Completion] = None,
        meter: Optional[SpendMeter] = None,
    ) -> None:
        self.cache = cache
        self.system = system
        self.run = run
        self.model = model
        self.prompt_version = prompt_version
        self.inner = inner
        self.meter = meter
        self.calls_made = 0
        self.elapsed_s = 0.0
        self.last_error: Optional[BaseException] = None
        self._lock = threading.Lock()

    def __call__(self, request: LLMRequest) -> LLMOutput:
        started = time.perf_counter()
        try:
            return self._answer(request)
        except BaseException as error:
            self.last_error = error
            raise
        finally:
            with self._lock:
                self.elapsed_s += time.perf_counter() - started

    def _answer(self, request: LLMRequest) -> LLMOutput:
        key = request_key(request, system=self.system, run=self.run, model=self.model,
                          prompt_version=self.prompt_version)
        hit = self.cache.get(key)
        if hit is not None:
            return LLMOutput.from_dict(hit["output"])
        if self.inner is None:
            raise MissingLLMOutputError(
                f"No cached output of {self.system} (run {self.run}, {self.model}, prompt "
                f"{self.prompt_version}) for the query {request.query_text!r} in {self.cache.path}. "
                f"Run `python -m app.evaluation.llm_recommender_evaluation` without --no-llm, or check "
                f"that the catalog, indexes and prompts are the ones the published run used."
            )
        output = self.inner(request)
        self.cache.add({
            "key": key, "system": self.system, "run": self.run, "model": self.model,
            "prompt_version": self.prompt_version, "operation": request.operation,
            "query_text": request.query_text, "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "prompt_chars": len(request.system_prompt) + len(request.user_prompt),
            "output": output.to_dict(),
        })
        with self._lock:
            self.calls_made += 1
        if self.meter is not None:
            self.meter.add(output.cost_usd)
        return output


# =============================================================================
# R3: matching recommendations to the catalog
# =============================================================================

_TITLE_CUT = re.compile(r":|\(|\[| - | – | — ")
_FIRST_AUTHOR = re.compile(r";|&|\band\b")


def normalize_text(text: str) -> str:
    """Lower case, accents removed, every run of non-alphanumeric characters as one space."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", plain).split())


def main_title(title: str) -> str:
    """The normalized title before any subtitle or note (':', '(', '[', ' - ')."""
    head = normalize_text(_TITLE_CUT.split(title, maxsplit=1)[0])
    return head or normalize_text(title)


def surname(name: str) -> str:
    """
    Surname of a person name: the words before a comma ("Galdos, Benito"),
    otherwise the last word; only the first of several authors ("A and B").
    """
    first = _FIRST_AUTHOR.split(name, maxsplit=1)[0]
    if "," in first:
        return normalize_text(first.split(",", 1)[0])
    words = normalize_text(first).split()
    return words[-1] if words else ""


def title_index(books: Sequence[Book]) -> Dict[str, List[Book]]:
    """Main title -> catalog books with that main title, in catalog order."""
    index: Dict[str, List[Book]] = {}
    for book in books:
        index.setdefault(main_title(book.title), []).append(book)
    return index


def _surname_words(name: str) -> List[str]:
    """Last word of the surname (a two-word surname "Perez Galdos" -> "galdos")."""
    words = surname(name).split()
    return words[-1:] if words else []


def match_book(title: str, author: str, index: Mapping[str, Sequence[Book]]) -> Optional[Book]:
    """
    The catalog book a closed-book recommendation names, or None.

    Titles must be equal after main_title; the last word of the recommended
    author's surname must be the last word of the surname of one of the
    book's authors (so "Perez Galdos, Benito" and "Benito Perez Galdos"
    agree). The first catalog book that matches is returned.
    """
    wanted = _surname_words(author)
    if not wanted:
        return None
    for book in index.get(main_title(title), []):
        if any(_surname_words(name) == wanted for name in book.authors):
            return book
    return None


# =============================================================================
# Running the systems
# =============================================================================


def catalog_order(books: Sequence[Book]) -> List[Book]:
    """The order R2 shows the catalog in: by (source, source_id), unrelated to any query."""
    return sorted(books, key=lambda book: (book.source, book.source_id or ""))


def run_rag(
    search: Searcher,
    outcomes: List[ListwiseOutcome],
    completion: CachedCompletion,
    texts: Mapping[str, str],
    *,
    log: Callable[[str], None] = print,
) -> Tuple[Dict[str, List[BookId]], Dict[str, ListwiseOutcome], Dict[str, float]]:
    """
    Run R1 query by query through SearchService (search) and its LLM reranker.

    Args:
        search: The searcher built with the LLMListwiseReranker
        outcomes: The list the reranker's observer appends to
        completion: The reranker's completion (to time the LLM part apart)
        texts: Query id -> text

    Returns:
        (query id -> top-10 book ids, query id -> outcome,
         query id -> ms spent outside the LLM call: retrieval, fusion, prompt)
    """
    run: Dict[str, List[BookId]] = {}
    by_query: Dict[str, ListwiseOutcome] = {}
    local_ms: Dict[str, float] = {}
    for number, (query_id, text) in enumerate(texts.items(), start=1):
        seen, llm_before = len(outcomes), completion.elapsed_s
        started = time.perf_counter()
        try:
            results = search(text)
        except RerankerFailedError as error:
            raise RerankerFailedError(f"{error}. Cause: {completion.last_error!r}") from completion.last_error
        wall_s = time.perf_counter() - started
        if len(outcomes) != seen + 1:
            raise RuntimeError(f"{completion.system}: the LLM reranker did not run for {query_id}")
        by_query[query_id] = outcomes[-1]
        local_ms[query_id] = (wall_s - (completion.elapsed_s - llm_before)) * 1000
        run[query_id] = [result.book.id for result in list(results)[:DEPTH]]
        if number % 11 == 0:
            log(f"  {completion.system} run {completion.run}: {number}/{len(texts)} queries")
    return run, by_query, local_ms


def _in_parallel(items: Sequence[Tuple[str, Any]], work: Callable[[Any], Any], workers: int) -> Dict[str, Any]:
    """id -> work(item) with at most `workers` calls in flight; the first error is raised."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {item_id: pool.submit(work, item) for item_id, item in items}
        return {item_id: future.result() for item_id, future in futures.items()}


def run_long_context(
    recommender: LLMRecommender, catalog: Sequence[Book], texts: Mapping[str, str], workers: int
) -> Tuple[Dict[str, List[BookId]], Dict[str, ListwiseOutcome]]:
    """Run R2: one call per query with the whole catalog (in catalog_order)."""
    outcomes = _in_parallel(
        list(texts.items()), lambda text: recommender.rank_books(text, catalog, LONG_CONTEXT_OPERATION), workers
    )
    run = {query_id: [book.id for book in outcome.ranked_books[:DEPTH]] for query_id, outcome in outcomes.items()}
    return run, outcomes


def run_closed_book(
    recommender: LLMRecommender, index: Mapping[str, Sequence[Book]], texts: Mapping[str, str], workers: int
) -> Tuple[Dict[str, List[BookId]], Dict[str, ClosedBookOutcome], Dict[str, List[Optional[Book]]]]:
    """
    Run R3: one call per query without a catalog; match each recommendation.

    Returns:
        (query id -> matched book ids in the model's order, without repeats,
         query id -> outcome, query id -> matched book or None per recommendation)
    """
    outcomes = _in_parallel(list(texts.items()), recommender.recommend_closed_book, workers)
    matches: Dict[str, List[Optional[Book]]] = {}
    run: Dict[str, List[BookId]] = {}
    for query_id, outcome in outcomes.items():
        matches[query_id] = [match_book(rec["title"], rec["author"], index) for rec in outcome.recommendations]
        ranked: List[BookId] = []
        for book in matches[query_id]:
            if book is not None and book.id not in ranked:
                ranked.append(book.id)
        run[query_id] = ranked[:DEPTH]
    return run, outcomes, matches


# =============================================================================
# Summaries (pure functions)
# =============================================================================


def _share(count: int, total: int) -> Optional[float]:
    return count / total if total else None


def listwise_grounding(outcomes: Sequence[ListwiseOutcome]) -> Dict[str, Any]:
    """Ids outside the list, repeated ids, completed lists and failed calls of a set of outcomes."""
    returned = sum(len(o.returned_ids) for o in outcomes)
    unknown = sum(len(o.unknown_ids) for o in outcomes)
    duplicates = sum(len(o.duplicate_ids) for o in outcomes)
    completed = sum(1 for o in outcomes if o.completed)
    return {
        "lists": len(outcomes),
        "ids_returned": returned,
        "unknown_ids": unknown,
        "unknown_share": _share(unknown, returned),
        "duplicate_ids": duplicates,
        "duplicate_share": _share(duplicates, returned),
        "lists_completed": completed,
        "lists_completed_share": _share(completed, len(outcomes)),
        "positions_filled": sum(o.filled for o in outcomes),
        "failed_calls": sum(1 for o in outcomes if o.failed),
        "empty_answers": sum(1 for o in outcomes if not o.failed and not o.returned_ids),
        "lists_with_more_than_10_ids": sum(1 for o in outcomes if len(o.returned_ids) > TOP_N),
    }


def closed_book_grounding(
    matches: Mapping[str, Sequence[Optional[Book]]],
    outcomes: Mapping[str, ClosedBookOutcome],
    language_of: Mapping[str, Optional[str]],
    index: Optional[Mapping[str, Sequence[Book]]] = None,
) -> Dict[str, Any]:
    """
    Share of closed-book recommendations found in the catalog, overall and by
    query language. With the title index, also the recommendations whose
    main title is in the catalog but whose author matches none of those
    books (a check that the rule does not miss obvious matches).
    """

    def block(query_ids: Sequence[str]) -> Dict[str, Any]:
        total = sum(len(matches[q]) for q in query_ids)
        found = sum(1 for q in query_ids for book in matches[q] if book is not None)
        return {
            "queries": len(query_ids),
            "recommendations": total,
            "in_catalog": found,
            "in_catalog_share": _share(found, total),
            "queries_with_a_match": sum(1 for q in query_ids if any(book is not None for book in matches[q])),
        }

    query_ids = list(matches)
    by_language = {
        language: block([q for q in query_ids if language_of.get(q) == language])
        for language in sorted({language_of.get(q) or "unknown" for q in query_ids})
    }
    title_only = None
    if index is not None:
        title_only = [
            {"query_id": q, "title": rec["title"], "author": rec["author"],
             "catalog_authors": [", ".join(b.authors) for b in index[main_title(rec["title"])]][:3]}
            for q in query_ids
            for rec, book in zip(outcomes[q].recommendations, matches[q])
            if book is None and main_title(rec["title"]) in index
        ]
    return {
        **block(query_ids),
        "failed_calls": sum(1 for o in outcomes.values() if o.failed),
        "by_language": by_language,
        "same_title_other_author": title_only,
    }


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def _sum(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return float(sum(present)) if present else None


def llm_cost(outputs: Sequence[LLMOutput], local_ms: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """
    Per-query cost of an LLM system from its call records.

    Args:
        outputs: One output per query (all runs pooled)
        local_ms: Same order: time outside the LLM call (retrieval), if any

    Returns:
        calls, failed calls, latency median and p95 in seconds (end to end:
        local part + LLM call), LLM-call latency median, mean input and output
        tokens, mean and total reported cost, total input tokens read from
        the provider's prompt cache
    """
    llm_s = [o.latency_s or 0.0 for o in outputs]
    local_s = [ms / 1000 for ms in local_ms] if local_ms is not None else [0.0] * len(outputs)
    end_to_end = [a + b for a, b in zip(llm_s, local_s)]
    return {
        "calls": len(outputs),
        "failed_calls": sum(1 for o in outputs if o.error is not None),
        "latency_median_s": _percentile(end_to_end, 50),
        "latency_p95_s": _percentile(end_to_end, 95),
        "llm_latency_median_s": _percentile(llm_s, 50),
        "local_median_ms": _percentile([s * 1000 for s in local_s], 50) if local_ms is not None else None,
        "input_tokens_mean": _mean([o.input_tokens for o in outputs]),
        "output_tokens_mean": _mean([o.output_tokens for o in outputs]),
        "cached_input_tokens_total": _sum([o.cached_input_tokens for o in outputs]),
        "calls_without_cost": sum(1 for o in outputs if o.cost_usd is None),
        "cost_mean_usd": _mean([o.cost_usd for o in outputs]),
        "cost_total_usd": _sum([o.cost_usd for o in outputs]),
    }


def tokens_per_book(
    input_tokens: Sequence[int], prompt_chars: Sequence[int], catalog_chars: int, num_books: int
) -> Tuple[float, float]:
    """
    Prompt tokens per book and fixed prompt tokens of R2, from reported counts.

    The provider reports one token count per prompt. The catalog block's
    share of the tokens is taken equal to its share of the characters, so
    per call: catalog tokens = tokens x catalog_chars / prompt_chars, and
    tokens per book = catalog tokens / num_books; the rest (instructions,
    query, chat template) is the fixed part. Medians over the calls.

    Returns:
        (tokens per book, fixed tokens per prompt)
    """
    if not input_tokens or len(input_tokens) != len(prompt_chars) or num_books < 1:
        raise ValueError("Need one prompt length per token count and at least one book")
    per_book = [t * catalog_chars / chars / num_books for t, chars in zip(input_tokens, prompt_chars)]
    fixed = [t - pb * num_books for t, pb in zip(input_tokens, per_book)]
    return float(statistics.median(per_book)), float(statistics.median(fixed))


def max_catalog_size(per_book: float, fixed: float, context_length: int, max_output_tokens: int) -> int:
    """Largest catalog whose prompt plus the output cap fits in the context window."""
    return max(0, math.floor((context_length - max_output_tokens - fixed) / per_book))


def scaling_rows(
    per_book: float,
    fixed: float,
    output_tokens: float,
    sizes: Sequence[int],
    *,
    context_length: int,
    max_output_tokens: int,
    price_input: float,
    price_output: float,
    price_cache_read: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Prompt size and cost per query of R2 for larger catalogs (computed, not run).

    prompt tokens = fixed + per_book x books; cost = prompt tokens x input
    price + output tokens x output price (USD per token). With a cache-read
    price, a second cost assumes the catalog block (identical for every
    query) is read from the provider's prompt cache.
    """
    rows = []
    for books in sizes:
        prompt = fixed + per_book * books
        row = {
            "books": books,
            "prompt_tokens": prompt,
            "fits_context": prompt + max_output_tokens <= context_length,
            "cost_usd": prompt * price_input + output_tokens * price_output,
            "cost_cached_catalog_usd": None,
        }
        if price_cache_read is not None:
            row["cost_cached_catalog_usd"] = (
                fixed * price_input + per_book * books * price_cache_read + output_tokens * price_output
            )
        rows.append(row)
    return rows


def determinism(first: Mapping[str, Sequence[BookId]], second: Mapping[str, Sequence[BookId]], k: int = DEPTH) -> Dict[str, Any]:
    """Queries whose top-k list (order included) and top-k set are the same in two runs."""
    query_ids = [q for q in first if q in second]
    return {
        "queries": len(query_ids),
        "identical_lists": sum(1 for q in query_ids if list(first[q])[:k] == list(second[q])[:k]),
        "identical_sets": sum(1 for q in query_ids if set(list(first[q])[:k]) == set(list(second[q])[:k])),
    }


def same_runs(a: Mapping[str, Sequence[str]], b: Mapping[str, Sequence[str]]) -> Dict[str, int]:
    """How many queries have the same top 10 in two runs (as strings)."""
    query_ids = [q for q in a if q in b]
    return {"queries": len(query_ids), "identical": sum(1 for q in query_ids if list(a[q]) == list(b[q]))}


# =============================================================================
# Judgments
# =============================================================================


def llm_judgment_keys(raw: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Tuple[str, str, str]]:
    """(query_id, source, source_id) of the LLM judgments of a judgments file."""
    return [(q, item["source"], item["source_id"]) for q, items in raw.items() for item in items
            if item.get("judge") == "llm"]


def fill_judgments(
    runs: Mapping[str, Mapping[str, Sequence[BookId]]],
    queries: Sequence[TestQuery],
    books: Sequence[Book],
    original_raw: Mapping[str, Sequence[Mapping[str, Any]]],
    extended_raw: Mapping[str, Sequence[Mapping[str, Any]]],
    judges: Sequence[Any],
    cache: JudgmentCache,
    *,
    log: Callable[[str], None] = print,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Grade the top-10 holes of `runs` and return the extended judgments with them.

    The judge must be the one that graded the existing file, so the file
    keeps one LLM judge. The holes are graded with the stored query text
    (as in V06), at most len(judges) at a time, and cached as they come.
    The result holds the original judgments and the judge's grades of the
    file's LLM pairs plus the holes; every existing grade is checked to be
    unchanged.

    Returns:
        (merged judgments in the file format, facts of the judging run)

    Raises:
        ValueError: If the judge is not the file's judge, or a grade changed
        IncompleteJudgingError: If holes remain (failures); nothing to write
    """
    from app.evaluation.judge_sensitivity import judge_in_parallel

    judge = judges[0]
    file_judges = {item.get("model") for items in extended_raw.values() for item in items if item.get("judge") == "llm"}
    if file_judges != {judge.model_name}:
        raise ValueError(f"The judge {judge.model_name} is not the judge of the judgments file {sorted(map(str, file_judges))}")
    texts = {query.query_id: query.text for query in queries}
    books_by_id = {book.id: book for book in books}
    holes = find_unjudged(runs, resolve_grades(extended_raw, books), DEPTH)
    tasks = [JudgeTask(query_id, texts[query_id], books_by_id[book_id]) for query_id, book_id in holes]
    log(f"Judging {len(tasks)} new (query, book) pairs with {judge.model_name} ({len(judges)} in flight)")
    stats = judge_in_parallel(tasks, list(judges), cache)
    if stats.remaining:
        raise IncompleteJudgingError(
            f"{stats.remaining} of {len(tasks)} pairs still have no grade ({stats.failed} failures); "
            f"run the same command again (the outputs and grades made are cached). Nothing was written."
        )
    keys = sorted(set(llm_judgment_keys(extended_raw)) | {task.key for task in tasks})
    missing = [key for key in keys if cache.get(key) is None]
    if missing:
        raise ValueError(f"{len(missing)} LLM judgments of the file are not in the judge cache, e.g. {missing[:3]}")
    merged = merge_judgments(original_raw, [cache.get(key) for key in keys])
    after = {(entry["query_id"], item["source"], item["source_id"]): item["relevance"]
             for entry in merged for item in entry["judgments"]}
    changed = [(q, item["source"], item["source_id"]) for q, items in extended_raw.items() for item in items
               if after.get((q, item["source"], item["source_id"])) != item["relevance"]]
    if changed:
        raise ValueError(f"{len(changed)} existing grades would change, e.g. {changed[:3]}")
    costs = [j.usage_recorder.summary()["total"]["reported_cost_usd"] for j in judges
             if getattr(j, "usage_recorder", None) is not None]
    return merged, {
        "judge_model": judge.model_name,
        "prompt_version": judge.prompt_version,
        "holes": len(tasks),
        "judged_in_this_run": stats.judged,
        "already_cached": stats.cached,
        "failed_in_this_run": stats.failed,
        "seconds_per_judgment": stats.seconds_per_judgment,
        "grade_distribution": [Counter(cache.get(task.key).grade for task in tasks).get(g, 0) for g in GRADES],
        "reported_cost_usd": _sum(costs),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# =============================================================================
# The report (no model is called from here on)
# =============================================================================


@dataclass
class Measurements:
    """Everything the systems produced in one run of this module."""

    runs: Dict[str, Dict[str, List[BookId]]]  # system (and system@run2) -> query -> top 10
    rag: Dict[int, Dict[str, ListwiseOutcome]]  # run -> query -> outcome
    rag_local_ms: Dict[int, Dict[str, float]]
    long_context: Dict[int, Dict[str, ListwiseOutcome]]
    closed_book: Dict[str, ClosedBookOutcome]
    closed_book_matches: Dict[str, List[Optional[Book]]]
    pipeline_latency: Dict[str, Any]
    long_context_prompt_chars: Dict[str, int]  # query -> characters of the R2 prompt
    long_context_catalog_chars: int  # characters of the catalog block in that prompt
    title_index: Optional[Dict[str, List[Book]]] = None  # R3's catalog index


def _outcome_record(outcome: ListwiseOutcome) -> Dict[str, Any]:
    """A listwise outcome, readable: what the model returned and the final top 10."""
    return {
        "returned_ids": outcome.returned_ids,
        "unknown_ids": outcome.unknown_ids,
        "duplicate_ids": outcome.duplicate_ids,
        "filled": outcome.filled,
        "error": outcome.output.error,
        "top10": [{"id": outcome.shown_ids[i], "source_id": outcome.books[i].source_id,
                   "title": outcome.books[i].title, "reason": outcome.reasons.get(outcome.shown_ids[i])}
                  for i in outcome.order[:DEPTH]],
    }


def build_report(
    m: Measurements,
    queries: Sequence[TestQuery],
    judgments: Mapping[str, Mapping[BookId, int]],
    model_info: Mapping[str, Any],
    *,
    max_output_tokens: int = RECOMMENDER_MAX_TOKENS,
    n_resamples: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Quality, contrasts, grounding, cost, scaling and determinism from the measurements.

    Raises:
        UnjudgedResultsError: If Judged@10 is not 1.0 for R0, R1 or R2 (both runs)
    """
    from app.evaluation.retrieval_ablation import UnjudgedResultsError

    report = compare_systems(m.runs, judgments, queries, None, n_resamples=n_resamples, seed=seed)
    must_be_judged = [R0, R1, R2, R1 + RUN2_SUFFIX, R2 + RUN2_SUFFIX]
    not_judged = {s: report["systems"][s]["metrics"]["judged_at_10"]["mean"] for s in must_be_judged
                  if report["systems"][s]["metrics"]["judged_at_10"]["mean"] != 1.0}
    if not_judged:
        raise UnjudgedResultsError(f"Judged@10 is not 1.0: {not_judged}")
    contrasts = compare_pairs(report, [(c.name, c.system, c.reference) for c in CONTRASTS],
                              n_resamples=n_resamples, seed=seed)
    language_of = {q.query_id: q.language for q in queries}

    # Cost: R0 measured locally (no LLM); R1 = local part + LLM call; R2, R3 = LLM call.
    rag_outputs = [m.rag[r][q].output for r in sorted(m.rag) for q in m.rag[r]]
    rag_local = [m.rag_local_ms[r][q] for r in sorted(m.rag) for q in m.rag[r]]
    long_outputs = [m.long_context[r][q].output for r in sorted(m.long_context) for q in m.long_context[r]]
    closed_outputs = [o.output for o in m.closed_book.values()]
    cost = {
        R0: {"latency_median_s": m.pipeline_latency["median_ms"] / 1000,
             "latency_p95_s": m.pipeline_latency["p95_ms"] / 1000, "calls": 0, "cost_mean_usd": 0.0,
             "note": "local models only (e5 + BM25 + bge on the local GPU); no LLM call"},
        R1: llm_cost(rag_outputs, rag_local),
        R2: llm_cost(long_outputs),
        R3: llm_cost(closed_outputs),
    }

    # Scaling of R2 from its reported prompt tokens.
    pairs = [(o.output.input_tokens, m.long_context_prompt_chars[q])
             for r in sorted(m.long_context) for q, o in m.long_context[r].items() if o.output.input_tokens]
    num_books = len(next(iter(m.long_context[1].values())).shown_ids)
    per_book, fixed = tokens_per_book([t for t, _ in pairs], [c for _, c in pairs],
                                      m.long_context_catalog_chars, num_books)
    pricing = model_info["pricing"]
    price_cache = float(pricing["input_cache_read"]) if pricing.get("input_cache_read") else None
    output_tokens = cost[R2]["output_tokens_mean"] or 0.0
    scaling = {
        "method": (
            "tokens per book = median over the R2 calls of reported prompt tokens x (catalog characters / "
            "prompt characters) / books; fixed = the rest of the prompt; prompt(N) = fixed + per_book x N"
        ),
        "tokens_per_book": per_book,
        "fixed_prompt_tokens": fixed,
        "description_max_chars": DESCRIPTION_MAX_CHARS,
        "output_tokens_mean": output_tokens,
        "max_output_tokens": max_output_tokens,
        "context_length": model_info["context_length"],
        "prices_usd_per_token": {"input": float(pricing["prompt"]), "output": float(pricing["completion"]),
                                 "input_cache_read": price_cache},
        "max_books_in_context": max_catalog_size(per_book, fixed, model_info["context_length"], max_output_tokens),
        "rows": scaling_rows(per_book, fixed, output_tokens, (num_books, *SCALING_SIZES),
                             context_length=model_info["context_length"], max_output_tokens=max_output_tokens,
                             price_input=float(pricing["prompt"]), price_output=float(pricing["completion"]),
                             price_cache_read=price_cache),
    }

    run2 = {s: report["systems"][s + RUN2_SUFFIX]["metrics"]["ndcg_at_10"]["mean"] for s in (R1, R2)}
    deterministic = {
        s: {**determinism(m.runs[s], m.runs[s + RUN2_SUFFIX]), "ndcg_at_10_run1": report["systems"][s]["metrics"]["ndcg_at_10"]["mean"],
            "ndcg_at_10_run2": run2[s],
            "mean_abs_ndcg_difference": statistics.fmean(
                abs(report["systems"][s]["per_query"][q]["ndcg_at_10"]
                    - report["systems"][s + RUN2_SUFFIX]["per_query"][q]["ndcg_at_10"])
                for q in report["query_ids"])}
        for s in (R1, R2)
    }

    # Descriptive: the two contrasts recomputed on run 2 (not part of the pre-declared family).
    run2_contrasts = compare_pairs(report, [(c.name + RUN2_SUFFIX, c.system + RUN2_SUFFIX, c.reference)
                                            for c in CONTRASTS], n_resamples=n_resamples, seed=seed)
    for name, contrast in run2_contrasts.items():
        contrast.pop("p_value_holm", None)  # descriptive: no correction reported
    grounding = {
        R1: {f"run{r}": listwise_grounding(list(m.rag[r].values())) for r in sorted(m.rag)},
        R2: {f"run{r}": listwise_grounding(list(m.long_context[r].values())) for r in sorted(m.long_context)},
        R3: closed_book_grounding(m.closed_book_matches, m.closed_book, language_of, m.title_index),
    }
    spend = {
        "recommender_calls_usd": _sum([o.cost_usd for o in rag_outputs + long_outputs + closed_outputs]),
        "by_system_usd": {R1: cost[R1]["cost_total_usd"], R2: cost[R2]["cost_total_usd"], R3: cost[R3]["cost_total_usd"]},
    }
    return {
        "num_queries": report["num_queries"],
        "languages": report["languages"],
        "queries_without_relevant": report["queries_without_relevant"],
        "systems": report["systems"],
        "contrasts": contrasts,
        "grounding": grounding,
        "cost": cost,
        "scaling": scaling,
        "determinism": deterministic,
        "run2_contrasts_descriptive": run2_contrasts,
        "spend": spend,
    }


def outputs_for_json(m: Measurements) -> Dict[str, Any]:
    """Per-query outputs of the LLM systems, readable, for the JSON report."""
    return {
        R1: {f"run{r}": {q: _outcome_record(o) for q, o in m.rag[r].items()} for r in sorted(m.rag)},
        R2: {f"run{r}": {q: _outcome_record(o) for q, o in m.long_context[r].items()} for r in sorted(m.long_context)},
        R3: {q: {"error": o.output.error, "recommendations": [
            {**rec, "in_catalog_source_id": book.source_id if book else None}
            for rec, book in zip(o.recommendations, m.closed_book_matches[q])]}
            for q, o in m.closed_book.items()},
    }


# =============================================================================
# Markdown
# =============================================================================


def _usd(value: Optional[float], digits: int = 5) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _num(value: Optional[float], digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _pct(count: int, share: Optional[float]) -> str:
    return f"{count} ({_fmt(share)})"


def build_markdown(document: Mapping[str, Any]) -> str:
    """LLM_RECOMMENDER.md: design, prompts, results tables and provenance."""
    run, results, models = document["run"], document["results"], document["models"]
    systems, contrasts = results["systems"], results["contrasts"]
    commit = run.get("git_commit") or "unknown"
    dirty = " (with uncommitted changes in tracked files)" if run.get("git_worktree_dirty") else ""
    llm, judge = models["system_llm"], models["judge"]
    spend = document["spend"]
    lines: List[str] = [
        "# The LLM as a recommender versus the retrieval pipeline (V07)",
        "",
        f"Generated by `{run['command']}` on {run['date']} (git `{commit}`{dirty}). "
        f"{results['num_queries']} test queries ({', '.join(f'{n} {lang}' for lang, n in results['languages'].items())}), "
        f"catalog of {run['num_books']} books, 10 results per query."
        + (f" The LLM outputs were made by the run of {run['live_run']['date']} ({run['live_run']['llm_calls_made']} "
           f"calls) and are read from the cache." if run.get("live_run") else ""),
        "",
        f"- System LLM: `{llm['provider']}/{llm['model']}`, temperature 0, thinking {llm['reasoning_text']}, "
        f"up to {llm['max_tokens']} output tokens, strict JSON-schema output. Context window "
        f"{models['openrouter_model_info']['context_length']:,} tokens; list prices "
        f"{float(models['openrouter_model_info']['pricing']['prompt']) * 1e6:.3f} / "
        f"{float(models['openrouter_model_info']['pricing']['completion']) * 1e6:.3f} USD per million input / "
        f"output tokens (OpenRouter models API, read {models['openrouter_model_info']['fetched_at']}).",
        f"- Relevance judge for the new holes: `{judge['model']}` (prompt {judge['prompt_version']}), another vendor than the system LLM.",
        f"- Recommender prompts: version {document['prompts']['version']}.",
        f"- Total spend: {_usd(spend['total_usd'], 3)} USD ({_usd(spend['recommender_calls_usd'], 3)} recommender calls, "
        f"{_usd(spend['judge_usd'], 3)} judge), as reported by OpenRouter per call."
        + (f" API key usage counter: {_usd(spend['key_usage_delta_usd'], 3)} USD over the run that made the calls."
           if spend.get("key_usage_delta_usd") is not None else ""),
        "",
        "## Design (fixed in advance)",
        "",
        "Systems:",
        "",
    ]
    lines += [f"- `{name}`: {text}" for name, text in document["design"]["systems"].items()]
    lines += [
        "",
        "Pre-declared contrasts (nDCG@10, system - reference): "
        + "; ".join(f"{c['name']} `{c['system']}` - `{c['reference']}` ({c['question']})" for c in document["design"]["contrasts"])
        + f". {document['design']['statistics']}",
        "",
        f"Descriptive only: {', '.join(document['design']['descriptive_only'])}.",
        "",
        f"Grounding rules: {document['design']['id_rule']} {document['design']['completion_rule']}",
        "",
        f"R3 matching rule: {document['design']['matching_rule']}",
        "",
        "## Prompts (as used)",
        "",
    ]
    for name, text in document["prompts"]["templates"]:
        lines += [f"{name}:", "", "```", text, "```", ""]
    lines += ["Each book in the list (example, first book of the R2 catalog order):", "", "```",
              document["prompts"]["book_entry_example"], "```", ""]

    order = [R0, R1, R2, R3, REFERENCE]
    lines += ["## Quality", "", "Run 1 of every system; Judged@10 = 1.000 is asserted for R0, R1 and R2. "
              "R3 is scored on its matched books only (a short list counts the missing positions as not "
              "relevant, and Judged@10 counts them as unjudged).", ""]
    rows = []
    for name in order:
        metrics = systems[name]["metrics"]
        rows.append([f"`{name}`", _fmt_ci(metrics["ndcg_at_10"]["mean"], metrics["ndcg_at_10"]["ci_low"],
                                          metrics["ndcg_at_10"]["ci_high"]),
                     _fmt(metrics["recall_at_10"]["mean"]), _fmt(metrics["precision_at_5"]["mean"]),
                     _fmt(metrics["mrr"]["mean"]), _fmt(metrics["judged_at_10"]["mean"])])
    lines += [_table(["System", "nDCG@10 [95% CI]", "Recall@10", "P@5", "MRR", "Judged@10"], rows), ""]

    lines += ["### Pre-declared contrasts", ""]
    rows = []
    questions = {c["name"]: c["question"] for c in document["design"]["contrasts"]}
    for name, c in contrasts.items():
        rows.append([f"{name}: {questions[name]}", f"`{c['system']}`", f"`{c['reference']}`", _fmt(c["mean_system"]),
                     _fmt(c["mean_reference"]), _fmt_ci(c["mean_difference"], c["ci_low"], c["ci_high"], signed=True),
                     _fmt_p(c["p_value"]), _fmt_p(c["p_value_holm"])])
    lines += [_table(["Contrast", "System", "Reference", "nDCG@10 system", "nDCG@10 reference", "Diff. [95% CI]",
                      "p", "p (Holm)"], rows), ""]

    languages = list(results["languages"])
    lines += ["### By query language (nDCG@10, descriptive)", ""]
    rows = [[f"`{name}`"] + [
        _fmt_ci(systems[name]["by_language"][lang]["ndcg_at_10"]["mean"],
                systems[name]["by_language"][lang]["ndcg_at_10"]["ci_low"],
                systems[name]["by_language"][lang]["ndcg_at_10"]["ci_high"]) for lang in languages] for name in order]
    lines += [_table(["System"] + [f"{lang} (n={results['languages'][lang]})" for lang in languages], rows), ""]

    g = results["grounding"]
    lines += ["## Grounding", "", "R1 and R2: ids the model returned that were not in the list it was given, "
              "repeated ids, and lists with fewer than 10 valid ids (completed with the remaining candidates in the "
              "order shown).", ""]
    rows = []
    for name in (R1, R2):
        for run_name, block in g[name].items():
            rows.append([f"`{name}` {run_name}", str(block["lists"]), str(block["ids_returned"]),
                         _pct(block["unknown_ids"], block["unknown_share"]),
                         _pct(block["duplicate_ids"], block["duplicate_share"]),
                         _pct(block["lists_completed"], block["lists_completed_share"]),
                         str(block["positions_filled"]), str(block["empty_answers"]), str(block["failed_calls"])])
    lines += [_table(["System", "Lists", "Ids returned", "Not in the list (share)", "Repeated (share)",
                      "Lists completed (share)", "Positions filled", "Empty answers", "Failed calls"], rows), ""]
    lines += ["An empty answer is a valid answer with no recommendation (the model declined to choose); its "
              "list is the order shown. Failed calls are answers that could not be parsed.", ""]
    r3 = g[R3]
    lines += ["R3: recommendations found in the catalog (matching rule above).", ""]
    rows = [["all", str(r3["queries"]), str(r3["recommendations"]), _pct(r3["in_catalog"], r3["in_catalog_share"]),
             str(r3["queries_with_a_match"])]]
    rows += [[lang, str(b["queries"]), str(b["recommendations"]), _pct(b["in_catalog"], b["in_catalog_share"]),
              str(b["queries_with_a_match"])] for lang, b in r3["by_language"].items()]
    lines += [_table(["Queries", "n", "Recommendations", "In the catalog (share)", "Queries with at least one match"],
                     rows), ""]
    others = r3.get("same_title_other_author")
    if others is not None:
        lines += [f"Check of the rule: {len(others)} recommendations share their main title with a catalog book "
                  f"whose authors do not match (so they are counted as not in the catalog):", ""]
        lines += [f"- {o['query_id']}: \"{o['title']}\" by {o['author']}; catalog: {'; '.join(o['catalog_authors'])}"
                  for o in others]
        lines.append("")
    lines += [f"Failed R3 calls: {r3['failed_calls']}. A recommendation outside this 177-book catalog is not "
              "necessarily an invented book: most are real books the catalog does not hold, which the system "
              "could not show or link either way.", ""]

    c = results["cost"]
    lines += ["## Cost", "", "Per query. R0 runs on local models (measured in this run, after one warm-up query). "
              "R1 = retrieval and fusion measured in this run + the LLM call; R2 and R3 = the LLM call. LLM latency, "
              "tokens and cost are those recorded when each call was made (both runs pooled for R1 and R2); cost is "
              "the cost OpenRouter reported.", ""]
    rows = [[f"`{R0}`", _num(c[R0]["latency_median_s"], 2), _num(c[R0]["latency_p95_s"], 2), "0", "0", "0", "0", "0"]]
    for name in (R1, R2, R3):
        b = c[name]
        rows.append([f"`{name}`", _num(b["latency_median_s"], 2), _num(b["latency_p95_s"], 2), str(b["calls"]),
                     _num(b["input_tokens_mean"]), _num(b["output_tokens_mean"]), _usd(b["cost_mean_usd"], 6),
                     _usd(b["cost_total_usd"], 4)])
    lines += [_table(["System", "Latency median (s)", "Latency p95 (s)", "LLM calls", "Input tokens (mean)",
                      "Output tokens (mean)", "Cost per query (USD)", "Cost of all calls (USD)"], rows), ""]
    lines += [f"R1 time outside the LLM call: median {_num(c[R1]['local_median_ms'], 1)} ms. "
              f"Input tokens read from the provider's prompt cache: R1 {_num(c[R1]['cached_input_tokens_total'])}, "
              f"R2 {_num(c[R2]['cached_input_tokens_total'])}, R3 {_num(c[R3]['cached_input_tokens_total'])} (totals).", ""]

    s = results["scaling"]
    lines += ["## Scaling of R2 (computed, not run)", "",
              f"Measured on this catalog: {s['tokens_per_book']:.1f} prompt tokens per book (descriptions cut to "
              f"{s['description_max_chars']} characters) and {s['fixed_prompt_tokens']:.0f} fixed tokens per prompt. "
              f"Method: {s['method']}. Output: {s['output_tokens_mean']:.0f} tokens per query (R2 mean). The prompt "
              f"plus the {s['max_output_tokens']}-token output cap fits the {s['context_length']:,}-token context "
              f"window up to **{s['max_books_in_context']:,} books**.", ""]
    rows = [[f"{row['books']:,}", f"{row['prompt_tokens']:,.0f}", "yes" if row["fits_context"] else "no",
             _usd(row["cost_usd"], 4), _usd(row["cost_cached_catalog_usd"], 4)] for row in s["rows"]]
    lines += [_table(["Books", "Prompt tokens per query", "Fits the context", "Cost per query, list price (USD)",
                      "Cost per query, catalog read from prompt cache (USD)"], rows), ""]
    lines += [f"Prices: input {s['prices_usd_per_token']['input'] * 1e6:.3f}, output "
              f"{s['prices_usd_per_token']['output'] * 1e6:.3f}"
              + (f", prompt-cache read {s['prices_usd_per_token']['input_cache_read'] * 1e6:.3f}"
                 if s['prices_usd_per_token']['input_cache_read'] is not None else "")
              + " USD per million tokens. Catalogs that do not fit cannot be sent in one call at all; the cost of "
              "those rows is what the call would cost if the window were large enough.", ""]

    d = results["determinism"]
    lines += ["## Determinism (R1 and R2 run twice)", ""]
    rows = [[f"`{name}`", f"{d[name]['identical_lists']} / {d[name]['queries']}",
             f"{d[name]['identical_sets']} / {d[name]['queries']}", _fmt(d[name]["ndcg_at_10_run1"]),
             _fmt(d[name]["ndcg_at_10_run2"]), _fmt(d[name]["mean_abs_ndcg_difference"])] for name in (R1, R2)]
    lines += [_table(["System", "Identical top-10 lists", "Same top-10 books (any order)", "nDCG@10 run 1",
                      "nDCG@10 run 2", "Mean abs. per-query difference"], rows), ""]
    run2 = results.get("run2_contrasts_descriptive") or {}
    if run2:
        lines += ["The same two contrasts on run 2 (descriptive, not part of the pre-declared family, no correction):", ""]
        rows = [[name, f"`{c['system']}`", f"`{c['reference']}`", _fmt(c["mean_system"]), _fmt(c["mean_reference"]),
                 _fmt_ci(c["mean_difference"], c["ci_low"], c["ci_high"], signed=True), _fmt_p(c["p_value"])]
                for name, c in run2.items()]
        lines += [_table(["Contrast", "System", "Reference", "nDCG@10 system", "nDCG@10 reference",
                          "Diff. [95% CI]", "p (unadjusted)"], rows), ""]

    j = document["judgments"]
    lines += ["## Judgments", "",
              f"`{run['extended_judgments_path']}`: {j['num_final']} judgments ({j['num_original']} original, "
              f"{j['num_llm']} by `{judge['model']}`); this task added {j['added_by_this_task']} LLM grades "
              f"(0/1/2/3: {' / '.join(str(n) for n in j['added_grade_distribution'])}) for the new top-10 books "
              f"of R0 to R3 (both runs), graded with the stored query text as in V06.", ""]
    for judged in j["judging_runs"]:
        lines += [f"- Judging run of {judged.get('date')}: {judged.get('holes')} holes, "
                  f"{judged.get('judged_in_this_run')} judged, {judged.get('failed_in_this_run')} failures, "
                  f"{_num(judged.get('seconds_per_judgment'), 2)} s per judgment, "
                  f"{_usd(judged.get('reported_cost_usd'), 4)} USD."]
    if j["judging_runs"]:
        lines.append("")
    check = document.get("reproduction_check")
    if check:
        lines += [f"R0 reproduction: the top 10 of `{R0}` is identical to the run of `{R0_SPEC.name}` stored by "
                  f"V06/V10 (`{check['file']}`) for {check['identical']} of {check['queries']} queries "
                  f"(`{REFERENCE}`: {check['reference_identical']} of {check['queries']}). On the judgments without "
                  f"the grades this task added, R0 scores nDCG@10 {_fmt(check.get('r0_ndcg_without_added_judgments'))} "
                  f"(V10 published {_fmt(check.get('r0_ndcg_published'))}); it scores lower in the tables above "
                  f"because the new grades include relevant books that R1 and R2 found, which raise the ideal DCG "
                  f"of those queries for every system.", ""]
    lines += ["## Reproduce", "", "```", *run["reproduce"], "```", "",
              "The first command calls the system LLM and the judge only for what is not cached "
              f"(`{run['cache_file']}`, `{run['judge_cache_file']}`); the second calls no LLM and fails if any "
              "output or judgment is missing. Both run the local retrieval models (e5, bge), so their latencies "
              "are re-measured on every run; every other number comes from the caches.", ""]
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def _reasoning_text(reasoning: Any) -> str:
    if reasoning is False:
        return "off"
    if reasoning is True:
        return "on"
    return "at the model default" if reasoning is None else f"at effort `{reasoning}`"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface (see the module docstring)."""
    parser = argparse.ArgumentParser(description="The LLM as a recommender versus the retrieval pipeline (V07)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Call no LLM and no judge: recompute everything from the caches; fail if anything is missing")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="LLM requests in flight (R2, R3, judge)")
    parser.add_argument("--model", default=None, help="System LLM id (default: LLM_MODEL from the environment / .env)")
    parser.add_argument("--device", default=None, help="Torch device of the local models (default: cuda, mps or cpu)")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--indexes-dir", default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--ablation-dir", default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--queries-path", default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--judgments-path", default=DEFAULT_JUDGMENTS_PATH, help="Original judgments (read only)")
    parser.add_argument("--extended-judgments-path", default=DEFAULT_EXTENDED_JUDGMENTS,
                        help="Extended judgments: read, and written back with the new holes graded")
    parser.add_argument("--judge-cache-dir", default=DEFAULT_JUDGE_CACHE_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache of the recommender's LLM outputs")
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    return parser


def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _model_info(cache_dir: str, model: str, live: bool) -> Dict[str, Any]:
    """Context length and prices of the model: fetched and saved when live, read from the snapshot otherwise."""
    path = Path(cache_dir) / MODEL_INFO_FILE
    if live:
        from app.evaluation.llm_model_selection import fetch_model_catalog

        entry = fetch_model_catalog()[model]
        info = {"model": model, "context_length": entry["context_length"], "pricing": entry["pricing"],
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "source": "https://openrouter.ai/api/v1/models"}
        _write(str(path), json.dumps(info, indent=2) + "\n")
        return info
    if not path.exists():
        raise MissingLLMOutputError(f"{path} is missing: run once without --no-llm to read the model's context and prices")
    info = json.loads(path.read_text(encoding="utf-8"))
    if info.get("model") != model:
        raise MissingLLMOutputError(f"{path} describes {info.get('model')}, not {model}")
    return info


def _previous(path: str) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def design(n_resamples: int, seed: int) -> Dict[str, Any]:
    """The design as declared before the run, for the report."""
    return {
        "systems": {name: SYSTEM_DESCRIPTIONS[name] for name in (R0, R1, R2, R3, REFERENCE)},
        "depth": DEPTH,
        "candidate_depth": RERANK_DEPTH,
        "top_n": TOP_N,
        "runs_of_r1_and_r2": NUM_RUNS,
        "description_max_chars": DESCRIPTION_MAX_CHARS,
        "primary_metric": PRIMARY_METRIC,
        "contrasts": [{"name": c.name, "system": c.system, "reference": c.reference, "question": c.question}
                      for c in CONTRASTS],
        "statistics": (
            f"Two-sided paired randomisation test (sign flips) on per-query nDCG@10, {n_resamples} resamples, "
            f"seed {seed}; Holm correction across exactly L1 and L2; paired percentile bootstrap 95% CI over "
            f"queries ({n_resamples} resamples, seed {seed})."
        ),
        "descriptive_only": [
            "Recall@10, P@5 and MRR", "the en/es breakdown", f"R3 and the `{REFERENCE}` reference row",
            "grounding", "cost", "scaling", "determinism (run 2)",
        ],
        "id_rule": (
            "Each book is shown with an id B1..Bn (R1: fused order; R2: catalog sorted by source and source_id). "
            "A returned id is read trimmed, without brackets and in upper case; ids that were not shown are "
            "dropped and counted, and so are repeated ids."
        ),
        "completion_rule": (
            "If fewer than 10 valid ids come back, the top 10 is completed with the remaining candidates in the "
            "order they were shown (fused order for R1, catalog order for R2); an unusable answer counts as zero "
            "valid ids."
        ),
        "matching_rule": (
            "a closed-book recommendation matches a catalog book when both titles are equal after normalisation "
            "(lower case, accents removed, every run of non-alphanumeric characters as one space), each cut before "
            "the first ':', '(', '[' or ' - ' (subtitles and edition notes ignored), and the last word of the "
            "recommended author's surname (the words before a comma, otherwise the last word; only the first of "
            "several authors) equals that of one of the book's authors. The first matching catalog book is taken; "
            "a book matched twice in one list is ranked once."
        ),
    }


def run_systems(
    stack: Any,
    recommender_for: Callable[[str, int], Tuple[LLMRecommender, CachedCompletion]],
    catalog: Sequence[Book],
    queries: Sequence[TestQuery],
    workers: int,
    *,
    log: Callable[[str], None] = print,
) -> Measurements:
    """
    Run R0 and the reference on the local models, R1 on the same stack, then R2 and R3.

    Args:
        stack: retrieval_ablation.ModelStack over the catalog
        recommender_for: (system, run) -> recommender and its cached completion
        catalog: The catalog in catalog_order (R2's prompt order)
        queries: Test queries (stored texts are searched)
        workers: LLM requests in flight for R2 and R3
    """
    texts = {query.query_id: query.text for query in queries}
    runs: Dict[str, Dict[str, List[BookId]]] = {}
    log(f"Running {R0} ({R0_SPEC.name})")
    runs[R0], r0_ms, warmup_ms = collect_run(stack.searcher_for(R0_SPEC), texts, warmup_text=queries[0].text)
    log(f"Running the reference {REFERENCE}")
    runs[REFERENCE], _, _ = collect_run(stack.searcher_for(CANDIDATES_SPEC), texts)

    rag: Dict[int, Dict[str, ListwiseOutcome]] = {}
    rag_local_ms: Dict[int, Dict[str, float]] = {}
    for number in range(1, NUM_RUNS + 1):
        log(f"Running {R1} run {number}")
        recommender, completion = recommender_for(R1, number)
        observed: List[ListwiseOutcome] = []
        search = stack.searcher_for(CANDIDATES_SPEC, reranker=LLMListwiseReranker(recommender, observer=observed.append))
        name = R1 if number == 1 else R1 + RUN2_SUFFIX
        runs[name], rag[number], rag_local_ms[number] = run_rag(search, observed, completion, texts, log=log)

    long_context: Dict[int, Dict[str, ListwiseOutcome]] = {}
    for number in range(1, NUM_RUNS + 1):
        log(f"Running {R2} run {number} ({len(catalog)} books per prompt, {workers} in flight)")
        recommender, _ = recommender_for(R2, number)
        name = R2 if number == 1 else R2 + RUN2_SUFFIX
        runs[name], long_context[number] = run_long_context(recommender, catalog, texts, workers)

    log(f"Running {R3} ({workers} in flight)")
    recommender, _ = recommender_for(R3, 1)
    index = title_index(catalog)
    runs[R3], closed_book, matches = run_closed_book(recommender, index, texts, workers)

    catalog_chars = len("\n\n".join(book_entry(i, b) for i, b in zip(candidate_ids(len(catalog)), catalog)))
    prompt_chars = {}
    for query_id, text in texts.items():
        request = listwise_request(text, catalog, LONG_CONTEXT_OPERATION)
        prompt_chars[query_id] = len(request.system_prompt) + len(request.user_prompt)
    return Measurements(
        runs=runs, rag=rag, rag_local_ms=rag_local_ms, long_context=long_context, closed_book=closed_book,
        closed_book_matches=matches, pipeline_latency={**latency_summary(list(r0_ms.values())), "warmup_ms": warmup_ms},
        long_context_prompt_chars=prompt_chars, long_context_catalog_chars=catalog_chars, title_index=index,
    )


def judgments_block(
    final_raw: Mapping[str, Sequence[Mapping[str, Any]]],
    books: Sequence[Book],
    previous: Mapping[str, Any],
    judging: Optional[Mapping[str, Any]],
    new_pairs: Sequence[Tuple[str, str, str]],
) -> Dict[str, Any]:
    """
    Counts of the final judgments and of the pairs this task added.

    The added pairs and the judging runs are kept across runs of this module
    (a later run finds no hole to grade), so they are carried over from the
    previous JSON report and checked against the final file.
    """
    llm_keys = set(llm_judgment_keys(final_raw))
    before = previous.get("judgments", {})
    added = {tuple(key) for key in before.get("added_pairs", [])} | set(new_pairs)
    added = {key for key in added if key in llm_keys}
    runs = list(before.get("judging_runs", []))
    if judging is not None and judging["holes"]:
        runs.append(dict(judging))
    grade = {(q, item["source"], item["source_id"]): item["relevance"]
             for q, items in final_raw.items() for item in items}
    models = sorted({str(item.get("model")) for items in final_raw.values() for item in items
                     if item.get("judge") == "llm"})
    versions = sorted({str(item.get("prompt_version")) for items in final_raw.values() for item in items
                       if item.get("judge") == "llm"})
    return {
        "num_final": sum(len(items) for items in final_raw.values()),
        "num_original": sum(1 for items in final_raw.values() for item in items if item.get("judge") != "llm"),
        "num_llm": len(llm_keys),
        "llm_judge_models": models,
        "llm_judge_prompt_versions": versions,
        "added_by_this_task": len(added),
        "added_grade_distribution": [sum(1 for key in added if grade[key] == g) for g in GRADES],
        "added_pairs": sorted(list(key) for key in added),
        "judging_runs": runs,
        "judge_cost_usd": _sum([r.get("reported_cost_usd") for r in runs]),
        "unresolved_books": sum(1 for q, items in final_raw.items() for item in items
                                if (item["source"], item["source_id"]) not in {(b.source, b.source_id) for b in books}),
    }


def reproduction_check(
    runs: Mapping[str, Mapping[str, Sequence[BookId]]],
    final_raw: Mapping[str, Sequence[Mapping[str, Any]]],
    added_pairs: Sequence[Sequence[str]],
    books: Sequence[Book],
    path: str = V06_RESULTS_JSON,
) -> Optional[Dict[str, Any]]:
    """
    R0 against the V06/V10 run of hybrid_norm_e5_bge (None without that file).

    Compares the top 10 per query (and the reference's), and R0's nDCG@10 on
    the final judgments without the grades this task added (the judgments
    V10 scored it on) with V10's published value.
    """
    previous = _previous(path)
    stored = previous.get("runs")
    if not stored:
        return None
    as_text = {name: {q: [str(b) for b in ids] for q, ids in runs[name].items()} for name in (R0, REFERENCE)}
    r0 = same_runs(as_text[R0], stored[R0_SPEC.name])
    reference = same_runs(as_text[REFERENCE], stored[CANDIDATES_SPEC.name])
    added = {tuple(key) for key in added_pairs}
    before = resolve_grades({q: [item for item in items if (q, item["source"], item["source_id"]) not in added]
                             for q, items in final_raw.items()}, books)
    scores = [ndcg_at_k(list(runs[R0][q]), before[q], DEPTH) for q in runs[R0] if before.get(q)]
    published = previous.get("systems", {}).get(R0_SPEC.name, {}).get("metrics", {}).get("ndcg_at_10", {}).get("mean")
    return {"file": path, "queries": r0["queries"], "identical": r0["identical"],
            "reference_identical": reference["identical"],
            "r0_ndcg_without_added_judgments": statistics.fmean(scores) if scores else None,
            "r0_ndcg_published": published}


def system_llm_identity(settings: Any, model_arg: Optional[str], live: bool,
                        previous: Mapping[str, Any]) -> Tuple[str, str, Any]:
    """
    Provider, model and reasoning setting of the system LLM the report names.

    A live run uses the environment (LLM_PROVIDER, LLM_MODEL, LLM_REASONING).
    A --no-llm run re-renders the outputs cached by the saved run, so,
    unless --model is given, it names that run's LLM from the saved results
    and not from the environment, which may describe another model.
    """
    saved = (previous.get("models") or {}).get("system_llm") or {}
    if not live and model_arg is None and saved.get("model"):
        return saved.get("provider"), saved["model"], saved.get("reasoning")
    return settings.provider, model_arg or settings.model, settings.reasoning


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the four systems, complete the judgments and write the reports."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    live = not args.no_llm

    from app.evaluation.retrieval_ablation import ModelStack, _hardware, _versions, resolve_device

    device = args.device or resolve_device()  # imports torch before faiss (macOS)
    from dotenv import load_dotenv

    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.infrastructure.config.settings import LLMSettings
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
    from app.infrastructure.llm.llm_recommender import LangChainCompletion

    def log(message: str) -> None:
        print(message, flush=True)

    load_dotenv(Path(".env"))
    started = datetime.now(UTC)
    settings = LLMSettings.from_env()
    previous = _previous(args.output_json)
    provider, model, reasoning = system_llm_identity(settings, args.model, live, previous)
    if live and (settings.provider != "openrouter" or model != settings.model):
        log("The live run uses the system LLM of the environment: LLM_PROVIDER=openrouter and LLM_MODEL (no --model).")
        return 2

    books = SqliteBookCatalogRepository(Path(args.db_path)).get_all()
    catalog = catalog_order(books)
    queries = load_test_queries(args.queries_path)
    original_raw = load_relevance_judgments(args.judgments_path)
    extended_raw = load_relevance_judgments(args.extended_judgments_path)
    cache_path = Path(args.cache_dir) / cache_file_name(model, RECOMMENDER_PROMPT_VERSION)
    cache = OutputCache(cache_path)
    meter = SpendMeter()
    completions: List[CachedCompletion] = []
    inner = None
    key_before = None
    try:
        model_info = _model_info(args.cache_dir, model, live)
        if live:
            from app.evaluation.llm_model_selection import key_usage_usd

            inner = LangChainCompletion.from_settings(settings)
            key_before = key_usage_usd()

        def recommender_for(system: str, run: int) -> Tuple[LLMRecommender, CachedCompletion]:
            completion = CachedCompletion(cache, system=system, run=run, model=model, inner=inner, meter=meter)
            completions.append(completion)
            return LLMRecommender(completion, model), completion

        stack = ModelStack(books, indexes_dir=args.indexes_dir, ablation_dir=args.ablation_dir, device=device, log=log)
        m = run_systems(stack, recommender_for, catalog, queries, args.workers, log=log)
        del stack
    except (MissingLLMOutputError, SpendLimitError, RerankerFailedError) as error:
        log(f"Stopped: {error}")
        return 1
    calls_made = sum(c.calls_made for c in completions)
    log(f"LLM calls made in this run: {calls_made} ({meter.spent_usd:.4f} USD reported)")

    # Judgments: grade the new holes (live) or require that there are none (--no-llm).
    judging = None
    new_pairs: List[Tuple[str, str, str]] = []
    judge_cache_path = None
    if live:
        from app.evaluation.relevance_judge import _build_judge
        from app.infrastructure.llm.relevance_judge import LangChainRelevanceJudge

        judge_settings, judge = _build_judge()
        judges = [judge] + [LangChainRelevanceJudge.from_settings(judge_settings) for _ in range(args.workers - 1)]
        judge_cache = JudgmentCache.for_judge(args.judge_cache_dir, judge.model_name, judge.prompt_version)
        judge_cache_path = str(judge_cache.path)
        before_keys = set(llm_judgment_keys(extended_raw))
        try:
            merged, judging = fill_judgments(m.runs, queries, books, original_raw, extended_raw, judges, judge_cache,
                                             log=log)
        except IncompleteJudgingError as error:
            log(str(error))
            return 1
        final_raw = {entry["query_id"]: entry["judgments"] for entry in merged}
        new_pairs = sorted(set(llm_judgment_keys(final_raw)) - before_keys)
        if new_pairs:
            _write(args.extended_judgments_path, json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    else:
        final_raw = extended_raw
    final = resolve_grades(final_raw, books)
    require_judged(m.runs, final, DEPTH)
    key_after = key_usage_usd() if live else None

    results = build_report(m, queries, final, model_info, n_resamples=args.n_resamples, seed=args.seed)
    judgments = judgments_block(final_raw, books, previous, judging, new_pairs)
    judge_model = (judgments["llm_judge_models"] or ["unknown"])[0]
    if judge_cache_path is None:
        from app.evaluation.relevance_judge import cache_file_name as judge_cache_file_name

        judge_cache_path = str(Path(args.judge_cache_dir) / judge_cache_file_name(
            judge_model, (judgments["llm_judge_prompt_versions"] or ["v1.0"])[0]))
    key_delta = (key_after - key_before) if key_before is not None and key_after is not None and calls_made else None
    if key_delta is None:
        key_delta = previous.get("spend", {}).get("key_usage_delta_usd")
    recommender_usd = results["spend"]["recommender_calls_usd"] or 0.0
    judge_usd = judgments["judge_cost_usd"] or 0.0

    command = " ".join(["python -m app.evaluation.llm_recommender_evaluation", *map(shlex.quote, raw_args)]).strip()
    # The run that made the LLM calls: this one when it made any, otherwise the one recorded before.
    before = previous.get("run", {})
    live_run = before.get("live_run")
    if live_run is None and before.get("mode") == "live" and before.get("llm_calls_made"):
        live_run = {"date": before.get("date"), "command": before.get("command"), "git_commit": before.get("git_commit"),
                    "llm_calls_made": before.get("llm_calls_made"),
                    "reported_cost_usd": previous.get("spend", {}).get("new_calls_this_run_usd"),
                    "key_usage_delta_usd": previous.get("spend", {}).get("key_usage_delta_usd")}
    if calls_made:
        live_run = {"date": started.isoformat(timespec="seconds"), "command": command,
                    "git_commit": _git("rev-parse", "--short", "HEAD"), "llm_calls_made": calls_made,
                    "reported_cost_usd": meter.spent_usd, "key_usage_delta_usd": key_delta}
    document = {
        "design": design(args.n_resamples, args.seed),
        "prompts": {"version": RECOMMENDER_PROMPT_VERSION, "templates": prompt_texts(),
                    "book_entry_example": book_entry("B1", catalog[0]), "description_max_chars": DESCRIPTION_MAX_CHARS},
        "models": {
            "system_llm": {"provider": provider, "model": model, "temperature": 0.0,
                           "reasoning": reasoning, "reasoning_text": _reasoning_text(reasoning),
                           "max_tokens": RECOMMENDER_MAX_TOKENS, "structured_output": "json_schema (strict)"},
            "judge": {"model": judge_model, "prompt_version": (judgments["llm_judge_prompt_versions"] or [None])[0]},
            "openrouter_model_info": model_info,
        },
        "results": results,
        "judgments": judgments,
        "reproduction_check": reproduction_check(m.runs, final_raw, judgments["added_pairs"], books),
        "spend": {
            "total_usd": recommender_usd + judge_usd,
            "recommender_calls_usd": recommender_usd,
            "judge_usd": judge_usd,
            "new_calls_this_run_usd": meter.spent_usd,
            "key_usage_delta_usd": key_delta,
        },
        "outputs": outputs_for_json(m),
        "runs": {name: {q: [str(b) for b in ids] for q, ids in run.items()} for name, run in m.runs.items()},
        "run": {
            "date": started.isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_worktree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "command": command,
            "mode": "live" if live else "no-llm",
            "llm_calls_made": calls_made,
            "live_run": live_run,
            "reproduce": [
                "python -m app.evaluation.llm_recommender_evaluation",
                "python -m app.evaluation.llm_recommender_evaluation --no-llm",
            ],
            "device": device,
            "hardware": _hardware(),
            "library_versions": _versions(),
            "num_books": len(books),
            "db_path": args.db_path,
            "queries_path": args.queries_path,
            "judgments_path": args.judgments_path,
            "extended_judgments_path": args.extended_judgments_path,
            "cache_file": str(cache_path),
            "judge_cache_file": judge_cache_path,
            "queries_sha256": _sha256(args.queries_path),
            "judgments_sha256": _sha256(args.judgments_path),
            "extended_judgments_sha256": _sha256(args.extended_judgments_path),
            "cache_sha256": _sha256(str(cache_path)),
        },
    }
    _write(args.output_json, json.dumps(document, ensure_ascii=False, indent=2, default=str) + "\n")
    _write(args.output_md, build_markdown(document))

    for name, c in results["contrasts"].items():
        log(f"{name} {c['system']} - {c['reference']}: {c['mean_difference']:+.3f} "
            f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] p={c['p_value']:.3f} p_holm={c['p_value_holm']:.3f}")
    log(f"Total spend {document['spend']['total_usd']:.4f} USD. Saved {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
