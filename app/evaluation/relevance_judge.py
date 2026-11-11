"""
LLM relevance judgments to fill the holes of the test collection.

Why: only about half of the top 10 of each system has a judgment, and an
unjudged book counts as not relevant. A new system that retrieves books
nobody judged is penalised for it (pool bias). Following the UMBRELA line
of work, an LLM grades the missing (query, book) pairs pointwise, with one
judge at temperature 0, and is first checked against the existing judgments.

This module holds the evaluation logic only; it does not import LangChain.
It depends on the small RelevanceJudge protocol below, which the adapter
app/infrastructure/llm/relevance_judge.py implements.

- JudgmentCache: resumable on-disk cache of LLM judgments, one JSON Lines
  file per judge model and prompt version under data/evaluation/llm_judgments/.
  Each judgment is appended and flushed as soon as it is made, so re-running
  never re-judges a cached pair and an interrupted run loses at most the
  pair in flight.
- judge_pairs: judges the pairs that are not cached yet.
- find_unjudged: the (query, book) pairs in any system's top k without a
  judgment.
- merge_judgments: original + LLM judgments in the relevance_judgments.json
  format, with provenance per judgment; an original judgment always wins.
- agreement_report: agreement of the LLM grades with the original ones
  (exact agreement, quadratic-weighted Cohen's kappa, binary kappa at grade
  >= 1 and >= 2, confusion matrix, mean signed difference), overall and by
  query language.
- stratified_sample: seeded sample stratified by (grade, query language).

CLI (run from the repository root):

    python -m app.evaluation.relevance_judge agreement --sample 200 --seed 0
    python -m app.evaluation.relevance_judge fill --k 10

`agreement` judges a stratified sample of the existing judged pairs blind
(the judge never sees the existing grade) and writes
data/evaluation/JUDGE_AGREEMENT.md and data/evaluation/judge_agreement.json.
`fill` runs the four non-LLM systems, judges every unjudged pair in their
top k, writes app/evaluation/relevance_judgments_extended.json (the original
file is not touched) and compares the systems on the original and on the
extended judgments (data/evaluation/COMPARISON_EXTENDED_QRELS.md).
Both accept --limit N to judge at most N new pairs and stop; the cache keeps
them for the next run.
"""

import argparse
import json
import logging
import os
import random
import re
import shlex
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from app.domain.entities import Book
from app.evaluation.comparison import (
    BASELINE_SYSTEMS,
    DEFAULT_BASELINE,
    SYSTEM_DESCRIPTIONS,
    _git,
    compare_systems,
    render_markdown,
    run_system,
)
from app.evaluation.evaluation_service import BookId
from app.evaluation.types import TestQuery

logger = logging.getLogger(__name__)

GRADES = (0, 1, 2, 3)
BINARY_THRESHOLDS = (1, 2)
UNKNOWN_LANGUAGE = "unknown"

DEFAULT_DB_PATH = "data/catalog.db"
DEFAULT_INDEXES_DIR = "data/indexes"
DEFAULT_QUERIES_PATH = "app/evaluation/test_queries.json"
DEFAULT_JUDGMENTS_PATH = "app/evaluation/relevance_judgments.json"
DEFAULT_CACHE_DIR = "data/evaluation/llm_judgments"
DEFAULT_AGREEMENT_JSON = "data/evaluation/judge_agreement.json"
DEFAULT_AGREEMENT_MD = "data/evaluation/JUDGE_AGREEMENT.md"
DEFAULT_EXTENDED_JUDGMENTS = "app/evaluation/relevance_judgments_extended.json"
DEFAULT_EXTENDED_MD = "data/evaluation/COMPARISON_EXTENDED_QRELS.md"

# Stop after this many judge failures in a row: the server is probably down.
MAX_CONSECUTIVE_FAILURES = 3

PairKey = Tuple[str, str, str]  # (query_id, source, source_id)
T = TypeVar("T")


# =============================================================================
# Judge protocol and judgments
# =============================================================================


class RelevanceJudge(Protocol):
    """
    Grades how relevant a book is to a query.

    model_name and prompt_version are recorded with every judgment; judge
    gets only the query text and the book, never an existing grade.
    """

    model_name: str
    prompt_version: str

    def judge(self, query_text: str, book: Book) -> Tuple[int, str]:
        """Return (grade 0-3, one-sentence rationale)."""
        ...


@dataclass(frozen=True)
class LLMJudgment:
    """One LLM judgment of a (query, book) pair, with its provenance."""

    query_id: str
    source: str
    source_id: str
    grade: int
    rationale: str
    model: str
    prompt_version: str
    date: str
    query_text: str = ""  # for readers of the cache file
    title: str = ""

    @property
    def key(self) -> PairKey:
        return (self.query_id, self.source, self.source_id)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LLMJudgment":
        return cls(**{name: data[name] for name in cls.__dataclass_fields__ if name in data})


def cache_file_name(model: str, prompt_version: str) -> str:
    """File name of the cache of one judge model and prompt version."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
    return f"{slug}__prompt-{prompt_version}.jsonl"


class JudgmentCache:
    """
    Append-only JSON Lines file of the LLM judgments of one judge model and
    prompt version, keyed by (query_id, source, source_id).

    Every judgment is written, flushed and synced to disk as soon as it is
    made, so a run that is interrupted loses at most the pair in flight. A
    run killed in the middle of a write leaves at most one partial last line,
    which is skipped on load.
    """

    def __init__(self, path: Path | str, model: str, prompt_version: str) -> None:
        self.path = Path(path)
        self.model = model
        self.prompt_version = prompt_version
        self._entries: Dict[PairKey, LLMJudgment] = {}
        self._needs_newline = False
        if self.path.exists():
            self._load()

    @classmethod
    def for_judge(cls, directory: Path | str, model: str, prompt_version: str) -> "JudgmentCache":
        """The cache file of a judge model and prompt version inside directory."""
        return cls(Path(directory) / cache_file_name(model, prompt_version), model, prompt_version)

    def _load(self) -> None:
        content = self.path.read_text(encoding="utf-8")
        self._needs_newline = bool(content) and not content.endswith("\n")
        for number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                judgment = LLMJudgment.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError) as error:
                logger.warning("Skipping unreadable line %d of %s: %s", number, self.path, error)
                continue
            self._check_provenance(judgment)
            self._entries[judgment.key] = judgment

    def _check_provenance(self, judgment: LLMJudgment) -> None:
        if (judgment.model, judgment.prompt_version) != (self.model, self.prompt_version):
            raise ValueError(
                f"{self.path} holds judgments of {self.model} / prompt {self.prompt_version}, "
                f"got one of {judgment.model} / prompt {judgment.prompt_version}"
            )

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: PairKey) -> Optional[LLMJudgment]:
        return self._entries.get(key)

    def judgments(self) -> List[LLMJudgment]:
        """All cached judgments, in the order they were made."""
        return list(self._entries.values())

    def add(self, judgment: LLMJudgment) -> None:
        """Append a judgment to the file (flushed and synced) and to memory."""
        self._check_provenance(judgment)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            if self._needs_newline:
                handle.write("\n")
                self._needs_newline = False
            handle.write(json.dumps(judgment.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._entries[judgment.key] = judgment


# =============================================================================
# Judging
# =============================================================================


class JudgeTask(NamedTuple):
    """A (query, book) pair to judge."""

    query_id: str
    query_text: str
    book: Book

    @property
    def key(self) -> PairKey:
        return (self.query_id, self.book.source, self.book.source_id or "")


def _print_now(message: str) -> None:
    """Progress line, flushed at once so a background run's log stays current."""
    print(message, flush=True)


@dataclass
class JudgingStats:
    """What a call to judge_pairs did."""

    total: int = 0  # pairs asked for
    cached: int = 0  # already judged before this run
    judged: int = 0  # judged in this run
    failed: int = 0  # judge errors in this run (not cached, retried next run)
    remaining: int = 0  # still without a judgment after this run
    seconds: float = 0.0  # time spent in successful judge calls

    @property
    def seconds_per_judgment(self) -> Optional[float]:
        return self.seconds / self.judged if self.judged else None


def judge_pairs(
    tasks: Sequence[JudgeTask],
    judge: RelevanceJudge,
    cache: JudgmentCache,
    *,
    limit: Optional[int] = None,
    log: Callable[[str], None] = _print_now,
    clock: Callable[[], float] = time.perf_counter,
) -> JudgingStats:
    """
    Judge, one at a time, the tasks that are not in the cache yet.

    Each judgment is added to the cache as soon as it is made. A judge error
    on one pair is logged and the pair is left for the next run; after
    MAX_CONSECUTIVE_FAILURES errors in a row the run stops with RuntimeError
    (the LLM server is probably down). Ctrl-C stops the run as well; the
    judgments already made stay in the cache.

    Args:
        tasks: Pairs to judge (duplicates are judged once)
        judge: The relevance judge
        cache: Cache of the same judge model and prompt version
        limit: Judge at most this many new pairs (None: all)
        log: Progress output
        clock: Timer, injectable for tests

    Returns:
        JudgingStats of the run
    """
    unique = list({task.key: task for task in tasks}.values())
    todo = [task for task in unique if task.key not in cache]
    stats = JudgingStats(total=len(unique), cached=len(unique) - len(todo))
    batch = todo if limit is None else todo[: max(limit, 0)]
    log(f"{stats.total} pairs: {stats.cached} already judged, {len(todo)} to judge, {len(batch)} in this run")

    consecutive_failures = 0
    for number, task in enumerate(batch, start=1):
        started = clock()
        try:
            grade, rationale = judge.judge(task.query_text, task.book)
            if grade not in GRADES:
                raise ValueError(f"grade {grade!r} is not in {GRADES}")
        except Exception as error:  # one bad output must not stop the whole run
            stats.failed += 1
            consecutive_failures += 1
            log(f"[{number}/{len(batch)}] {task.query_id} {task.book.source_id}: judge failed: {error}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{consecutive_failures} judge failures in a row; is the LLM server running?"
                ) from error
            continue
        elapsed = clock() - started
        consecutive_failures = 0
        cache.add(
            LLMJudgment(
                query_id=task.query_id,
                source=task.book.source,
                source_id=task.book.source_id or "",
                grade=grade,
                rationale=rationale,
                model=judge.model_name,
                prompt_version=judge.prompt_version,
                date=datetime.now(UTC).isoformat(timespec="seconds"),
                query_text=task.query_text,
                title=task.book.title,
            )
        )
        stats.judged += 1
        stats.seconds += elapsed
        mean = stats.seconds / stats.judged
        eta_minutes = mean * (len(batch) - number) / 60
        log(
            f"[{number}/{len(batch)}] {task.query_id} {task.book.source_id} grade={grade} "
            f"{elapsed:.1f}s (mean {mean:.1f}s, ETA {eta_minutes:.1f} min)"
        )
    stats.remaining = len(todo) - stats.judged
    return stats


# =============================================================================
# Holes in the judgments and merging
# =============================================================================


def find_unjudged(
    results_by_system: Mapping[str, Mapping[str, Sequence[BookId]]],
    judgments: Mapping[str, Mapping[BookId, int]],
    k: int,
) -> List[Tuple[str, BookId]]:
    """
    The (query, book) pairs in any system's top k that have no judgment.

    Args:
        results_by_system: System name -> query id -> ranked book ids (the
            runs compare_systems consumes)
        judgments: Query id -> book id -> grade
        k: Depth of the top k

    Returns:
        Distinct (query_id, book_id) pairs, ordered by the best rank at which
        any system returns them (then by query, then by first appearance), so
        an interrupted fill has judged the positions that weigh most in
        nDCG@k first.
    """
    best_rank: Dict[Tuple[str, BookId], int] = {}
    query_order: Dict[str, int] = {}
    for run in results_by_system.values():
        for query_id, ranked in run.items():
            query_order.setdefault(query_id, len(query_order))
            graded = judgments.get(query_id, {})
            for rank, book_id in enumerate(list(ranked)[:k], start=1):
                if book_id in graded:
                    continue
                pair = (query_id, book_id)
                best_rank[pair] = min(rank, best_rank.get(pair, rank))
    first_seen = {pair: index for index, pair in enumerate(best_rank)}
    return sorted(best_rank, key=lambda pair: (best_rank[pair], query_order[pair[0]], first_seen[pair]))


def merge_judgments(
    original: Mapping[str, Sequence[Mapping[str, Any]]],
    llm: Iterable[LLMJudgment],
) -> List[Dict[str, Any]]:
    """
    Original and LLM judgments in the relevance_judgments.json format.

    Every judgment carries `judge`: "original" for the existing ones (whose
    provenance is unknown) or "llm", and LLM ones also carry `model`,
    `prompt_version`, `date` and `rationale`. An original judgment always
    wins over an LLM one for the same (query, book) pair.

    Args:
        original: Query id -> list of {source, source_id, relevance, ...}, as
            returned by load_relevance_judgments
        llm: LLM judgments (e.g. JudgmentCache.judgments())

    Returns:
        [{"query_id", "judgments": [...]}]: the queries of `original` in their
        order, then any other query in id order; within a query the original
        judgments in their order, then the LLM ones by (source, source_id).
    """
    llm_by_query: Dict[str, Dict[Tuple[str, str], LLMJudgment]] = defaultdict(dict)
    for judgment in llm:
        llm_by_query[judgment.query_id][(judgment.source, judgment.source_id)] = judgment

    query_ids = list(original) + sorted(q for q in llm_by_query if q not in original)
    merged = []
    for query_id in query_ids:
        items = []
        for item in original.get(query_id, []):
            entry = dict(item)
            entry.setdefault("judge", "original")
            items.append(entry)
        judged = {(item["source"], item["source_id"]) for item in items}
        for (source, source_id), judgment in sorted(llm_by_query.get(query_id, {}).items()):
            if (source, source_id) in judged:
                continue
            items.append({
                "source": source,
                "source_id": source_id,
                "relevance": judgment.grade,
                "judge": "llm",
                "model": judgment.model,
                "prompt_version": judgment.prompt_version,
                "date": judgment.date,
                "rationale": judgment.rationale,
            })
        merged.append({"query_id": query_id, "judgments": items})
    return merged


# =============================================================================
# Agreement with the original judgments
# =============================================================================


def _kappa(a: np.ndarray, b: np.ndarray, labels: Sequence[int], weights: Optional[str] = None) -> Optional[float]:
    """Cohen's kappa, or None when it is undefined (a single category everywhere)."""
    if a.size == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = cohen_kappa_score(a, b, labels=list(labels), weights=weights)
    return None if np.isnan(value) else float(value)


def _kappa_ci(
    original: np.ndarray, llm: np.ndarray, n_resamples: int, seed: int, confidence: float = 0.95
) -> Optional[List[float]]:
    """Percentile bootstrap CI of the quadratic-weighted kappa, resampling pairs."""
    if original.size < 2 or n_resamples < 1:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for indices in rng.integers(0, original.size, size=(n_resamples, original.size)):
        value = _kappa(original[indices], llm[indices], GRADES, "quadratic")
        if value is not None:
            values.append(value)
    if not values:
        return None
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return [float(low), float(high)]


def _agreement(original: np.ndarray, llm: np.ndarray, n_resamples: int, seed: int) -> Dict[str, Any]:
    """Agreement statistics of two aligned grade arrays."""
    n = int(original.size)
    difference = llm - original
    binary = {}
    for threshold in BINARY_THRESHOLDS:
        a, b = (original >= threshold).astype(int), (llm >= threshold).astype(int)
        binary[f"at_least_{threshold}"] = {
            "kappa": _kappa(a, b, (0, 1)),
            "agreement": float(np.mean(a == b)) if n else None,
        }
    return {
        "n": n,
        "exact_agreement": float(np.mean(difference == 0)) if n else None,
        "within_one_agreement": float(np.mean(np.abs(difference) <= 1)) if n else None,
        "quadratic_weighted_kappa": _kappa(original, llm, GRADES, "quadratic"),
        "quadratic_weighted_kappa_ci95": _kappa_ci(original, llm, n_resamples, seed),
        "binary": binary,
        "mean_signed_difference": float(difference.mean()) if n else None,
        "confusion_matrix": (
            confusion_matrix(original, llm, labels=list(GRADES)).tolist() if n
            else [[0] * len(GRADES) for _ in GRADES]
        ),
        "grade_distribution": {
            "original": [int(np.sum(original == g)) for g in GRADES],
            "llm": [int(np.sum(llm == g)) for g in GRADES],
        },
    }


def agreement_report(
    original_grades: Sequence[int],
    llm_grades: Sequence[int],
    languages: Optional[Sequence[Optional[str]]] = None,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    How well the LLM grades agree with the original grades.

    Args:
        original_grades: Original grade (0-3) of each pair
        llm_grades: LLM grade (0-3) of the same pairs, in the same order
        languages: Query language of each pair (None: no breakdown)
        n_resamples: Bootstrap resamples for the kappa CI (pairs are resampled)
        seed: Random seed of the bootstrap

    Returns:
        JSON-serializable dict: "overall" and "by_language" blocks with n,
        exact agreement, within-one agreement, quadratic-weighted Cohen's
        kappa (with a 95% bootstrap CI), binary kappa and agreement at grade
        >= 1 and >= 2, mean signed difference (LLM - original: positive means
        the LLM grades higher), the 4x4 confusion matrix (rows: original
        grade, columns: LLM grade) and both grade distributions; plus a
        "config" block that states these conventions.

    Raises:
        ValueError: If the sequences differ in length or a grade is not 0-3
    """
    original = np.asarray(original_grades, dtype=int)
    llm = np.asarray(llm_grades, dtype=int)
    if original.shape != llm.shape or (languages is not None and len(languages) != original.size):
        raise ValueError("original_grades, llm_grades and languages must have the same length")
    if not (np.isin(original, GRADES).all() and np.isin(llm, GRADES).all()):
        raise ValueError(f"Grades must be in {GRADES}")

    by_language = {}
    if languages is not None:
        labels = np.asarray([language or UNKNOWN_LANGUAGE for language in languages])
        for language in sorted(set(labels.tolist())):
            mask = labels == language
            by_language[language] = _agreement(original[mask], llm[mask], n_resamples, seed)
    return {
        "config": {
            "grades": list(GRADES),
            "confusion_matrix": "rows: original grade 0-3, columns: LLM grade 0-3",
            "mean_signed_difference": "mean of LLM grade - original grade (positive: the LLM grades higher)",
            "quadratic_weighted_kappa_ci95": (
                f"percentile bootstrap over pairs, {n_resamples} resamples, seed {seed}"
            ),
            "binary": "kappa and agreement after mapping grades to grade >= threshold",
        },
        "overall": _agreement(original, llm, n_resamples, seed),
        "by_language": by_language,
    }


def stratified_sample(
    items: Sequence[T],
    n: int,
    *,
    seed: int,
    stratum: Callable[[T], Hashable],
) -> List[T]:
    """
    Seeded sample of n items, stratified with proportional allocation.

    Each stratum gets n * stratum size / population items (largest-remainder
    rounding) and, when n is at least the number of strata, at least one, so
    small strata (grade 3, one language) are always represented.

    Args:
        items: Population, in a deterministic order
        n: Sample size (the whole population if n >= len(items))
        seed: Random seed (same seed and population: same sample)
        stratum: Stratum of an item, e.g. (grade, language)

    Returns:
        The sampled items, in population order
    """
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []
    groups: Dict[Hashable, List[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[stratum(item)].append(index)
    keys = sorted(groups, key=repr)

    exact = {key: n * len(groups[key]) / len(items) for key in keys}
    quota = {key: int(exact[key]) for key in keys}
    by_remainder = sorted(keys, key=lambda key: -(exact[key] - quota[key]))
    for key in by_remainder[: n - sum(quota.values())]:
        quota[key] += 1
    if n >= len(keys):
        for key in keys:
            if quota[key] == 0:
                # Take the unit from the most over-allocated stratum that keeps one.
                donor = max((k for k in keys if quota[k] >= 2), key=lambda k: quota[k] - exact[k])
                quota[donor] -= 1
                quota[key] = 1

    rng = random.Random(seed)
    chosen = [index for key in keys for index in rng.sample(groups[key], quota[key])]
    return [items[index] for index in sorted(chosen)]


# =============================================================================
# CLI
# =============================================================================


class JudgedPair(NamedTuple):
    """An original judgment of the test collection."""

    query_id: str
    source: str
    source_id: str
    grade: int
    language: str


def _command(raw_args: Sequence[str]) -> str:
    return " ".join(["python -m app.evaluation.relevance_judge", *map(shlex.quote, raw_args)])


def _provenance(started: datetime, raw_args: Sequence[str]) -> Dict[str, Any]:
    return {
        "date": started.isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_worktree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "command": _command(raw_args),
    }


def _judge_description(settings, judge: RelevanceJudge) -> Dict[str, Any]:
    """The judge configuration recorded in every output."""
    return {
        "provider": settings.provider,
        "model": judge.model_name,
        "base_url": settings.base_url,
        "prompt_version": judge.prompt_version,
        "prompt": "build_relevance_judge_prompt in app/infrastructure/llm/prompts_judge.py",
        "temperature": settings.temperature,
        "reasoning": settings.reasoning,
        "max_tokens": settings.max_tokens,
        "system_llm": settings.system_llm,
        "same_as_system_llm": settings.same_as_system_llm,
    }


def _thinking(judge_info: Mapping[str, Any]) -> str:
    """How the judge's thinking mode was set, in words."""
    reasoning = judge_info.get("reasoning")
    if reasoning is False:
        return "thinking off"
    if reasoning is True:
        return "thinking on"
    if reasoning is None:
        return "the model's default thinking mode"
    return f"thinking at effort `{reasoning}` (it cannot be switched off)"


def _judge_note(judge_info: Mapping[str, Any]) -> str:
    """Markdown note on which model judges and why that matters."""
    if judge_info["same_as_system_llm"]:
        return (
            f"The judge is the same model as the system LLM (`{judge_info['system_llm']}`): it is the "
            "only model available locally. The literature recommends a judge from a different model "
            "family than the system being judged; set JUDGE_PROVIDER / JUDGE_MODEL / JUDGE_BASE_URL "
            "to use another one. The four systems compared here do not use an LLM."
        )
    return (
        f"The judge (`{judge_info['provider']}/{judge_info['model']}`) is not the system LLM "
        f"(`{judge_info['system_llm']}`)."
    )


def _build_judge():
    """JudgeSettings from the environment (and .env) and the LangChain judge."""
    from dotenv import load_dotenv

    from app.infrastructure.config.settings import JudgeSettings
    from app.infrastructure.llm.relevance_judge import LangChainRelevanceJudge

    load_dotenv(Path(".env"))
    settings = JudgeSettings.from_env()
    return settings, LangChainRelevanceJudge.from_settings(settings)


def _fmt(value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _agreement_rows(report: Mapping[str, Any]) -> str:
    header = (
        "| Pairs | n | Exact | Within 1 | Quadratic-weighted kappa [95% CI] | Kappa (grade >= 1) "
        "| Kappa (grade >= 2) | Mean LLM - original |\n|---|---|---|---|---|---|---|---|"
    )
    blocks = [("all", report["overall"])] + [
        (f"{language} queries", block) for language, block in report["by_language"].items()
    ]
    rows = [header]
    for name, block in blocks:
        ci = block["quadratic_weighted_kappa_ci95"]
        kappa = _fmt(block["quadratic_weighted_kappa"]) + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
        rows.append(
            f"| {name} | {block['n']} | {_fmt(block['exact_agreement'])} "
            f"| {_fmt(block['within_one_agreement'])} | {kappa} "
            f"| {_fmt(block['binary']['at_least_1']['kappa'])} | {_fmt(block['binary']['at_least_2']['kappa'])} "
            f"| {_fmt(block['mean_signed_difference'], signed=True)} |"
        )
    return "\n".join(rows)


def _confusion_table(block: Mapping[str, Any]) -> str:
    rows = ["| Original \\ LLM | 0 | 1 | 2 | 3 | Total |", "|---|---|---|---|---|---|"]
    for grade, counts in zip(GRADES, block["confusion_matrix"]):
        rows.append(f"| {grade} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
    totals = [sum(column) for column in zip(*block["confusion_matrix"])]
    rows.append("| Total | " + " | ".join(map(str, totals)) + f" | {sum(totals)} |")
    return "\n".join(rows)


def build_agreement_markdown(document: Mapping[str, Any], max_disagreements: int = 20) -> str:
    """JUDGE_AGREEMENT.md: provenance, agreement tables and the largest disagreements."""
    run, judge, sample, report = document["run"], document["judge"], document["sample"], document["agreement"]
    commit = run["git_commit"] or "unknown"
    dirty = " (with uncommitted changes in tracked files)" if run["git_worktree_dirty"] else ""
    strata = ", ".join(f"{key} {count}" for key, count in sample["strata"].items())
    confusion = "\n\n".join(
        f"{title}:\n\n{_confusion_table(block)}"
        for title, block in [("All pairs", report["overall"])]
        + [(f"{language} queries", block) for language, block in report["by_language"].items()]
    )
    disagreements = sorted(
        (pair for pair in document["pairs"] if abs(pair["llm"] - pair["original"]) >= 2),
        key=lambda pair: (-abs(pair["llm"] - pair["original"]), pair["query_id"], pair["source_id"]),
    )
    rows = ["| Query | Book | Description | Original | LLM | LLM rationale |", "|---|---|---|---|---|---|"]
    for pair in disagreements[:max_disagreements]:
        rows.append(
            f"| {pair['query_id']} \"{pair['query_text']}\" | {pair['title'].replace('|', '/')} "
            f"| {'yes' if pair['has_description'] else 'missing'} | {pair['original']} | {pair['llm']} "
            f"| {pair['rationale'].replace('|', '/')} |"
        )
    shown = min(len(disagreements), max_disagreements)
    seconds = document["timing"]["seconds_per_judgment"]
    timing = (
        f"\n- Time: {document['timing']['judged_in_this_run']} judgments made by this run, "
        f"{seconds:.2f} s per judgment (one request at a time)." if seconds else ""
    )
    return f"""# Agreement of the LLM relevance judge with the original judgments

- Date: {run["date"]}
- Git commit: `{commit}`{dirty}
- Reproduce: `{run["command"]}`
- Judge: `{judge["provider"]}/{judge["model"]}`, prompt `{judge["prompt_version"]}`
  (`{judge["prompt"]}`), temperature {judge["temperature"]}, {_thinking(judge)}, at most
  {judge["max_tokens"]} output tokens, structured output (rationale, then grade 0-3).
- {_judge_note(judge)}{timing}
- Sample: {sample["n"]} of the {sample["population"]} original judgments, seed {sample["seed"]}, stratified
  by (grade, query language) with proportional allocation: {strata}.
- Blind: the judge sees only the query and the catalog fields of the book (title, authors,
  categories, year, language, description), never the original grade, a system or a rank.
- The provenance of the original judgments is unknown (`QRELS_DATASHEET.md`), so this measures
  agreement with them, not correctness. The kappa CI resamples pairs and ignores that pairs of
  the same query are not independent, so it is optimistic.

## Agreement

{_agreement_rows(report)}

Exact: share of pairs with the same grade. Within 1: grades differ by at most one.
Mean LLM - original: positive when the LLM grades higher.

## Confusion matrices (rows: original grade, columns: LLM grade)

{confusion}

## Largest disagreements (grades 2 or more apart)

{len(disagreements)} pairs; {shown} shown. All sampled pairs, with the LLM rationale, are in
`judge_agreement.json`.

{chr(10).join(rows)}
"""


def run_agreement(
    args: argparse.Namespace,
    raw_args: Sequence[str],
    build_judge: Optional[Callable[[], Tuple[Any, RelevanceJudge]]] = None,
) -> int:
    """
    `agreement` subcommand: judge a stratified sample of judged pairs blind and report.

    Args:
        args: Parsed CLI arguments
        raw_args: The CLI arguments as given, recorded in the report
        build_judge: Returns (JudgeSettings, judge); default _build_judge
            (JUDGE_* from the environment). The model comparison
            (llm_model_selection.py) passes one judge per candidate model.
    """
    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

    started = datetime.now(UTC)
    settings, judge = (build_judge or _build_judge)()
    catalog = SqliteBookCatalogRepository(Path(args.db_path))
    books = {(book.source, book.source_id): book for book in catalog.get_all()}
    queries = {query.query_id: query for query in load_test_queries(args.queries_path)}
    raw_judgments = load_relevance_judgments(args.judgments_path)

    population = [
        JudgedPair(query_id, item["source"], item["source_id"], int(item["relevance"]),
                   queries[query_id].language or UNKNOWN_LANGUAGE)
        for query_id, items in raw_judgments.items()
        for item in items
        if query_id in queries and (item["source"], item["source_id"]) in books
    ]
    sample = stratified_sample(population, args.sample, seed=args.seed,
                               stratum=lambda pair: (pair.grade, pair.language))
    cache = JudgmentCache.for_judge(args.cache_dir, judge.model_name, judge.prompt_version)
    tasks = [
        JudgeTask(pair.query_id, queries[pair.query_id].text, books[(pair.source, pair.source_id)])
        for pair in sample
    ]
    stats = judge_pairs(tasks, judge, cache, limit=args.limit)
    _print_stats(stats)
    missing = [pair for pair in sample if (pair.query_id, pair.source, pair.source_id) not in cache]
    if missing:
        print(f"{len(missing)} of {len(sample)} sampled pairs have no LLM judgment yet; "
              f"run the same command again to finish. No report written.")
        return 0

    llm = [cache.get((pair.query_id, pair.source, pair.source_id)) for pair in sample]
    report = agreement_report(
        [pair.grade for pair in sample], [judgment.grade for judgment in llm],
        [pair.language for pair in sample], n_resamples=args.n_resamples, seed=args.seed,
    )
    strata = Counter(f"grade {pair.grade}/{pair.language}" for pair in sample)
    document = {
        "run": _provenance(started, raw_args),
        "judge": _judge_description(settings, judge),
        "cache_file": str(cache.path),
        "timing": {
            "judged_in_this_run": stats.judged,
            "failed_in_this_run": stats.failed,
            "seconds_per_judgment": stats.seconds_per_judgment,
        },
        "sample": {
            "n": len(sample),
            "population": len(population),
            "seed": args.seed,
            "stratified_by": "(original grade, query language), proportional allocation",
            "strata": dict(sorted(strata.items())),
        },
        "agreement": report,
        "pairs": [
            {
                "query_id": pair.query_id,
                "query_text": queries[pair.query_id].text,
                "language": pair.language,
                "source": pair.source,
                "source_id": pair.source_id,
                "title": books[(pair.source, pair.source_id)].title,
                "has_description": books[(pair.source, pair.source_id)].has_description(),
                "original": pair.grade,
                "llm": judgment.grade,
                "rationale": judgment.rationale,
            }
            for pair, judgment in zip(sample, llm)
        ],
    }
    _write(args.output_json, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    _write(args.output_md, build_agreement_markdown(document))
    print(_agreement_rows(report))
    print(f"\nSaved {args.output_json} and {args.output_md}")
    return 0


def build_fill_markdown(document: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    """COMPARISON_EXTENDED_QRELS.md: provenance and the two comparison tables."""
    run, judge, fill = document["run"], document["judge"], document["fill"]
    commit = run["git_commit"] or "unknown"
    dirty = " (with uncommitted changes in tracked files)" if run["git_worktree_dirty"] else ""
    systems = "\n".join(f"- `{name}`: {SYSTEM_DESCRIPTIONS[name]}" for name in before["systems"])
    grades = " / ".join(str(count) for count in fill["llm_grade_distribution"])
    agreement = document.get("agreement")
    agreement_line = (
        f"Agreement with the original judgments on {agreement['n']} sampled pairs "
        f"(`JUDGE_AGREEMENT.md`): quadratic-weighted kappa {_fmt(agreement['quadratic_weighted_kappa'])}, "
        f"exact agreement {_fmt(agreement['exact_agreement'])}."
        if agreement else "No agreement report for this judge model and prompt version (`agreement` subcommand)."
    )
    judged_rows = "\n".join(
        f"| {system} | {before['systems'][system]['metrics']['judged_at_10']['mean']:.3f} "
        f"| {after['systems'][system]['metrics']['judged_at_10']['mean']:.3f} |"
        for system in before["systems"]
    )
    return f"""# Comparison on the original and on the LLM-extended judgments

- Date: {run["date"]}
- Git commit: `{commit}`{dirty}
- Reproduce: `{run["command"]}`
- Judge: `{judge["provider"]}/{judge["model"]}`, prompt `{judge["prompt_version"]}`, temperature
  {judge["temperature"]}, {_thinking(judge)}. {_judge_note(judge)}
- {agreement_line}
- Holes: {fill["unjudged_pairs"]} (query, book) pairs in the top {fill["k"]} of the systems below had no
  judgment; the LLM judged all of them (grades 0 / 1 / 2 / 3: {grades}).
- Extended judgments: `{fill["output_judgments"]}`, {fill["num_original"]} original + {fill["num_llm"]} LLM
  judgments. Every judgment carries `judge` ("original" or "llm"); LLM ones also carry `model`,
  `prompt_version`, `date` and `rationale`. An original judgment always wins over an LLM one.
  `relevance_judgments.json` is unchanged.
- Metrics, confidence intervals and tests as in `COMPARISON_BASELINE.md` (depth {fill["k"]},
  baseline `{before["config"]["baseline"]}`). LLM grades are not human grades: read the extended
  table as a robustness check against pool bias, not as a replacement for the original one.

Systems:

{systems}

## Judged@10

| System | Original judgments | Extended judgments |
|---|---|---|
{judged_rows}

## On the original judgments

{render_markdown(before)}
## On the extended judgments (original + LLM)

{render_markdown(after)}"""


def _load_agreement_summary(path: str, judge: RelevanceJudge) -> Optional[Dict[str, Any]]:
    """Overall agreement block of judge_agreement.json if it belongs to this judge."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (document["judge"]["model"], document["judge"]["prompt_version"]) != (judge.model_name, judge.prompt_version):
        return None
    return document["agreement"]["overall"]


def run_fill(args: argparse.Namespace, raw_args: Sequence[str]) -> int:
    """`fill` subcommand: judge the holes in the top k of the four non-LLM systems."""
    from app.domain.services import SearchService
    from app.evaluation.evaluation_job import (
        load_relevance_judgments,
        load_test_queries,
        resolve_judgments_to_uuids,
    )
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
    from app.infrastructure.search.bm25_search_repository import BM25SearchRepository
    from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
    from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository

    started = datetime.now(UTC)
    settings, judge = _build_judge()

    # Same search stack as `python -m app.evaluation.comparison`.
    catalog = SqliteBookCatalogRepository(Path(args.db_path))
    books = catalog.get_all()
    books_by_id = {book.id: book for book in books}
    # Legacy padding (zero-score books fill the list): reproduces the V05 pool (V13)
    bm25 = BM25SearchRepository(zero_score_padding=True)
    bm25.load_index(str(Path(args.indexes_dir) / "bm25_index.pkl"))
    embeddings = EmbeddingsStoreFaiss()
    embeddings.load_index(str(Path(args.indexes_dir) / "faiss_index"))
    vector = FaissVectorSearchRepository(embeddings, books)
    search_service = SearchService(lexical_search=bm25, vector_search=vector, embeddings_store=embeddings)

    queries: List[TestQuery] = load_test_queries(args.queries_path)
    query_text = {query.query_id: query.text for query in queries}
    raw_judgments = load_relevance_judgments(args.judgments_path)
    original = {qid: dict(j.judgments) for qid, j in resolve_judgments_to_uuids(raw_judgments, catalog).items()}

    runs = {
        system: {
            query.query_id: [
                result.book.id for result in run_system(
                    system, query.text, search_service=search_service, lexical_repo=bm25,
                    vector_repo=vector, embeddings_store=embeddings, depth=args.k,
                )
            ]
            for query in queries
        }
        for system in BASELINE_SYSTEMS
    }
    unjudged = find_unjudged(runs, original, args.k)
    tasks = [JudgeTask(query_id, query_text[query_id], books_by_id[book_id]) for query_id, book_id in unjudged]
    cache = JudgmentCache.for_judge(args.cache_dir, judge.model_name, judge.prompt_version)
    stats = judge_pairs(tasks, judge, cache, limit=args.limit)
    _print_stats(stats)
    if stats.remaining:
        print(f"{stats.remaining} of {len(unjudged)} unjudged pairs still have no LLM judgment; "
              f"run the same command again to finish. No extended judgments written.")
        return 0

    merged = merge_judgments(raw_judgments, cache.judgments())
    _write(args.output_judgments, json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    extended_raw = {entry["query_id"]: entry["judgments"] for entry in merged}
    extended = {qid: dict(j.judgments) for qid, j in resolve_judgments_to_uuids(extended_raw, catalog).items()}

    compare = dict(baseline=args.baseline, n_resamples=args.n_resamples, seed=args.seed)
    before = compare_systems(runs, original, queries, **compare)
    after = compare_systems(runs, extended, queries, **compare)
    llm_grades = Counter(cache.get(task.key).grade for task in tasks)
    items = [item for entry in merged for item in entry["judgments"]]
    document = {
        "run": _provenance(started, raw_args),
        "judge": _judge_description(settings, judge),
        "agreement": _load_agreement_summary(args.agreement_json, judge),
        "fill": {
            "k": args.k,
            "unjudged_pairs": len(unjudged),
            "llm_grade_distribution": [llm_grades.get(grade, 0) for grade in GRADES],
            "output_judgments": args.output_judgments,
            "num_original": sum(1 for item in items if item["judge"] == "original"),
            "num_llm": sum(1 for item in items if item["judge"] == "llm"),
        },
    }
    _write(args.output_md, build_fill_markdown(document, before, after))
    print("On the original judgments:\n" + render_markdown(before))
    print("On the extended judgments (original + LLM):\n" + render_markdown(after))
    print(f"Saved {args.output_judgments} and {args.output_md}")
    return 0


def _print_stats(stats: JudgingStats) -> None:
    per_pair = f"{stats.seconds_per_judgment:.2f}s per judgment" if stats.judged else "no new judgment"
    print(f"Judged {stats.judged} new pairs ({per_pair}), {stats.cached} were cached, "
          f"{stats.failed} failed, {stats.remaining} remaining")


def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface (see the module docstring)."""
    parser = argparse.ArgumentParser(description="LLM relevance judge for the test collection")
    subcommands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--db-path", default=DEFAULT_DB_PATH)
        sub.add_argument("--queries-path", default=DEFAULT_QUERIES_PATH)
        sub.add_argument("--judgments-path", default=DEFAULT_JUDGMENTS_PATH)
        sub.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
        sub.add_argument("--limit", type=int, default=None,
                         help="Judge at most this many new pairs, then stop (cached for the next run)")

    agreement = subcommands.add_parser("agreement", help="Judge a sample of judged pairs blind and report agreement")
    common(agreement)
    agreement.add_argument("--sample", type=int, default=200)
    agreement.add_argument("--seed", type=int, default=0)
    agreement.add_argument("--n-resamples", type=int, default=2000, help="Bootstrap resamples for the kappa CI")
    agreement.add_argument("--output-json", default=DEFAULT_AGREEMENT_JSON)
    agreement.add_argument("--output-md", default=DEFAULT_AGREEMENT_MD)

    fill = subcommands.add_parser("fill", help="Judge the unjudged pairs in the top k of the non-LLM systems")
    common(fill)
    fill.add_argument("--k", type=int, default=10)
    fill.add_argument("--indexes-dir", default=DEFAULT_INDEXES_DIR)
    fill.add_argument("--baseline", default=DEFAULT_BASELINE, choices=BASELINE_SYSTEMS)
    fill.add_argument("--n-resamples", type=int, default=10000)
    fill.add_argument("--seed", type=int, default=0)
    fill.add_argument("--agreement-json", default=DEFAULT_AGREEMENT_JSON)
    fill.add_argument("--output-judgments", default=DEFAULT_EXTENDED_JUDGMENTS)
    fill.add_argument("--output-md", default=DEFAULT_EXTENDED_MD)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per LLM call otherwise
    if args.command == "agreement":
        return run_agreement(args, raw_args)
    return run_fill(args, raw_args)


if __name__ == "__main__":
    sys.exit(main())
