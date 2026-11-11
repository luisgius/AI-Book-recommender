"""
Which hosted LLM should the system use, and which one should judge it? (V10)

Public rankings of cheap hosted models disagree with each other, so the
choice is made on this project's own tasks. Two roles are filled:

- the SYSTEM LLM: query understanding and grounded explanations;
- the JUDGE: relevance grades for the evaluation, from a different vendor
  than the system LLM (a judge tends to prefer its own family's output).

Three measurements, fixed before any result was seen:

A. Judge role. The existing agreement measurement of
   app/evaluation/relevance_judge.py (`agreement --sample 200 --seed 0`,
   the same seeded, stratified sample for every model) with one judge per
   candidate. Quadratic-weighted kappa against the original judgments (with
   its bootstrap interval), exact agreement, mean signed difference,
   failures, seconds per judgment and real cost; plus the pairwise kappa
   between the candidates (and the earlier local judge) on those 200 pairs.
B. Explanation role. The first 10 Spanish and the first 10 English test
   queries (by query id), the top 3 results of the production hybrid
   (BM25 + MiniLM, RRF) for each: 60 grounded explanations per candidate
   through the real LangChainLLMClient. Share of calls that parsed, citation
   precision BEFORE the guardrail (share of the citations the model wrote
   that the guardrail's own check accepts), share of explanations with no
   valid citation, share written in the language of the query (rule:
   explanation_language below), median latency and real cost per
   explanation.
C. Query-understanding role. The 44 test queries through
   extract_query_intent. Share of queries whose three graph steps parsed,
   share of queries where the model extracted a language or year filter the
   query does not ask for (the deterministic rule of filter_guards.py),
   median latency and real cost per query.

Selection rule (fixed before any result was seen; select_system_llm and
select_judge apply it mechanically):

- System LLM: among the candidates with 100% parsed calls in B and C, the
  highest pre-guardrail citation precision. Candidates within 0.02 of the
  best are tied; ties go to the higher share of explanations in the query
  language, then to the lower cost of one search (understanding + three
  explanations).
- Judge: among the candidates whose vendor (the prefix of the OpenRouter id)
  differs from the selected system LLM's, the highest quadratic-weighted
  kappa in A. Candidates within 0.02 are tied and the cheaper one wins.

Latency and tokens come from the V08 usage recorder (one LangChain callback
per chat model); the cost is the one OpenRouter reports for each call (the
price of the endpoint that served it). Requests run with at most 4 in
flight: in A, four candidates are judged in parallel, each one request at a
time; in B and C, four worker threads per model, each with its own client.

CLI (from the repository root; needs OPENROUTER_API_KEY in .env, except
`report`):

    python -m app.evaluation.llm_model_selection run
    python -m app.evaluation.llm_model_selection run --models z-ai/glm-5.3=low --roles judge
    python -m app.evaluation.llm_model_selection report

`run` measures A, B and C for the given models (default: the V10
candidates), merges them into data/evaluation/results_llm_model_selection.json
and renders data/evaluation/LLM_MODEL_SELECTION.md. `report` re-renders both
from the saved raw results without calling any model.
"""

import argparse
import json
import logging
import re
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from app.domain.citation_validation import get_book_field_text, is_valid_snippet
from app.domain.entities import Book
from app.evaluation.comparison import _git
from app.evaluation.relevance_judge import agreement_report, cache_file_name
from app.infrastructure.llm.filter_guards import query_requests_language, query_states_years
from app.infrastructure.search.text_normalization import (
    _ENGLISH_STOPWORDS,
    _SPANISH_STOPWORDS,
    fold_accents,
)

logger = logging.getLogger(__name__)

Reasoning = Union[bool, str, None]

# =============================================================================
# Design (fixed before any result was seen)
# =============================================================================

# Candidates of 2026-11-11 (OpenRouter ids) and their thinking mode. Thinking
# is off where the model accepts it; MiniMax and both GLM models answer
# "Reasoning is mandatory for this endpoint" (HTTP 400) to "off", so they run
# at effort "low", as V08 did for MiniMax.
CANDIDATES: Dict[str, Reasoning] = {
    "qwen/qwen3.8-flash": False,
    "qwen/qwen3.7-flash": False,
    "deepseek/deepseek-v4.1-flash": False,
    "minimax/minimax-m3.1-flash": "low",
    "z-ai/glm-5.3-flash": "low",
    "z-ai/glm-5.3": "low",
}

# The earlier local judge (V05/V06), whose grades are in the judgment cache.
LOCAL_JUDGE_MODEL = "qwen3.8:27b-mlx"

AGREEMENT_SAMPLE = 200
AGREEMENT_SEED = 0
AGREEMENT_ATTEMPTS = 3  # re-runs of `agreement` for pairs whose judgment failed
EXPLAINED_QUERIES_PER_LANGUAGE = 10
EXPLAINED_RESULTS = 3  # results explained per search (LLM_MAX_EXPLANATIONS)
SEARCH_DEPTH = 10  # the production hybrid returns 10 results
MAX_IN_FLIGHT = 4
TIE_MARGIN = 0.02
SPEND_LIMIT_USD = 3.0

# Speed-quality rule of 2026-11-11 (task V16), which supersedes the V10 rule:
# the author asked for models "balanced speed-quality", and the V10 choice,
# made on quality alone, was the slowest candidate. Re-applied to the saved
# V10 measurements, without new calls.
LATENCY_BAND = 0.10  # candidates within 10% of the fastest tied one: the cheaper wins
MIN_QUERY_LANGUAGE_SHARE = 0.95  # system LLM: share of explanations in the query language
SPEED_QUALITY_RULE_DATE = "2026-11-11"
SPEED_QUALITY_RULE_REASON = (
    "The author asked for models balanced between speed and quality and for choices that are measured. "
    "The V10 rule chose on quality alone, and the system LLM it chose was the slowest candidate "
    "(6.6 s per explanation, 8.2 s per query understanding)."
)
SPEED_QUALITY_RULE_TEXT = (
    "For each role, the candidates whose primary quality metric is within 0.02 of the best are tied on "
    "quality; among them the lowest median latency wins; if another tied candidate is within 10% of that "
    "latency, the cheaper one wins. A candidate that failed any call in the measurement is excluded. "
    "System LLM: quality = pre-guardrail citation precision (B); a candidate must have parsed every call "
    "(B and C) and written at least 95% of its explanations in the query language; latency = median per "
    "explanation; cost = real cost per explanation. Judge: among the candidates of another vendor than the "
    "selected system LLM, the highest quadratic-weighted kappa (A), with the same 0.02 tie band and the "
    "cheaper winning a tie (speed does not matter for the judge)."
)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"

DEFAULT_RESULTS_JSON = "data/evaluation/results_llm_model_selection.json"
DEFAULT_REPORT_MD = "data/evaluation/LLM_MODEL_SELECTION.md"
DEFAULT_DB_PATH = "data/catalog.db"
DEFAULT_INDEXES_DIR = "data/indexes"
DEFAULT_QUERIES_PATH = "app/evaluation/test_queries.json"
DEFAULT_CACHE_DIR = "data/evaluation/llm_judgments"

ROLES = ("judge", "explanation", "query_understanding")

# Exception types of a model answer that is not valid JSON for the schema
# (LangChain's OpenAI client parses the JSON inside the call, so the usage
# recorder sees these as failed calls).
SCHEMA_ERRORS = ("ValidationError", "OutputParserException", "JSONDecodeError")

# Stopwords that belong to one language only ("a", "me", "he", "has" are in
# both lists and therefore say nothing about the language).
SPANISH_ONLY_STOPWORDS = frozenset(_SPANISH_STOPWORDS - _ENGLISH_STOPWORDS)
ENGLISH_ONLY_STOPWORDS = frozenset(_ENGLISH_STOPWORDS - _SPANISH_STOPWORDS)
_WORD = re.compile(r"\w+")


# =============================================================================
# Metrics (pure functions)
# =============================================================================


def vendor(model_id: str) -> str:
    """Vendor of an OpenRouter model id: the part before the slash."""
    return model_id.split("/", 1)[0]


def explanation_language(text: str) -> Optional[str]:
    """
    Language of an explanation, by a majority vote of stopwords.

    Rule: lowercase and fold accents, split into words, then count the words
    that are Spanish-only and the words that are English-only stopwords (the
    lists of the BM25 tokenizer, text_normalization.py; words in both lists
    are ignored). More Spanish ones: "es"; more English ones: "en"; a tie
    (including none at all): None, which never counts as the query language.
    Quoted book text can add a few stopwords of the other language, but
    function words dominate any sentence, so the vote follows the language
    the explanation is written in.
    """
    words = _WORD.findall(fold_accents(text.lower()))
    spanish = sum(1 for word in words if word in SPANISH_ONLY_STOPWORDS)
    english = sum(1 for word in words if word in ENGLISH_ONLY_STOPWORDS)
    if spanish > english:
        return "es"
    if english > spanish:
        return "en"
    return None


def citation_is_valid(chunk_id: str, snippet: str, book: Book) -> bool:
    """The guardrail's check of one citation: the snippet is in the cited field."""
    return is_valid_snippet(snippet, get_book_field_text(book, chunk_id))


def citation_error(chunk_id: str, snippet: str, book: Book) -> Optional[str]:
    """
    Why a citation fails the check, for the failure-pattern notes.

    Returns None for a valid citation, "wrong field" when the snippet is in
    another field of the book, "not in the book" otherwise.
    """
    if citation_is_valid(chunk_id, snippet, book):
        return None
    fields = ("title", "description", "categories", "authors")
    if any(citation_is_valid(field, snippet, book) for field in fields if field != chunk_id):
        return "wrong field"
    return "not in the book"


def unrequested_filters(query_text: str, filters: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    """
    Which extracted filters the query does not ask for (filter_guards.py rule).

    - language: a non-empty value the guard would drop, i.e. the query does
      not ask for that language with a preposition ("en español", "in
      English"). A value that is not a known language code counts as
      unrequested too.
    - year: a min_year or max_year although the query states no year (no
      run of 2-4 digits).
    """
    filters = filters or {}
    language = (filters.get("language") or "").strip().lower()
    has_year = filters.get("min_year") is not None or filters.get("max_year") is not None
    return {
        "language": bool(language) and not query_requests_language(query_text, language),
        "year": has_year and not query_states_years(query_text),
    }


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _sum_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None


def _share(count: int, total: int) -> Optional[float]:
    return count / total if total else None


def summarize_explanations(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Table B row from the raw explanation records of one model.

    Denominators: parsed_share over all calls; citation precision over every
    citation the model wrote (micro-average); zero-valid and query-language
    shares over the parsed explanations; cost per explanation over all calls.
    """
    parsed = [record for record in records if record["parsed"]]
    citations = [citation for record in parsed for citation in record["citations"]]
    valid = sum(1 for citation in citations if citation["valid"])
    zero_valid = sum(1 for record in parsed if not any(c["valid"] for c in record["citations"]))
    in_language = sum(
        1 for record in parsed
        if explanation_language(f"{record['summary']} {record['reasoning']}") == record["language"]
    )
    cost = _sum_or_none([record["cost_usd"] for record in records])
    return {
        "n": len(records),
        "parsed": len(parsed),
        "parsed_share": _share(len(parsed), len(records)),
        "citations": len(citations),
        "valid_citations": valid,
        "citation_precision": _share(valid, len(citations)),
        "zero_valid_share": _share(zero_valid, len(parsed)),
        "in_query_language_share": _share(in_language, len(parsed)),
        "median_latency_s": _median([record["latency_s"] for record in records]),
        "cost_usd": cost,
        "cost_per_explanation_usd": cost / len(records) if cost is not None and records else None,
    }


def summarize_queries(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Table C row from the raw query-understanding records of one model.

    Denominators: parsed_share over all queries; unrequested-filter shares
    over the queries whose filter step parsed; cost per query over all.
    """
    parsed = [record for record in records if record["parsed"]]
    with_filters = [record for record in records if record["raw_filters"] is not None]
    flags = [unrequested_filters(record["text"], record["raw_filters"]) for record in with_filters]
    cost = _sum_or_none([record["cost_usd"] for record in records])
    return {
        "n": len(records),
        "parsed": len(parsed),
        "parsed_share": _share(len(parsed), len(records)),
        "filters_parsed": len(with_filters),
        "unrequested_language_share": _share(sum(f["language"] for f in flags), len(flags)),
        "unrequested_year_share": _share(sum(f["year"] for f in flags), len(flags)),
        "unrequested_any_share": _share(sum(f["language"] or f["year"] for f in flags), len(flags)),
        "median_latency_s": _median([record["latency_s"] for record in records]),
        "cost_usd": cost,
        "cost_per_query_usd": cost / len(records) if cost is not None and records else None,
    }


def summarize_judge(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Table A row from one candidate's judge result (agreement document + usage)."""
    agreement = result.get("agreement")
    if not agreement:
        return {"complete": False, "failures": result.get("failures"),
                "cost_usd": (result.get("usage") or {}).get("reported_cost_usd")}
    overall = agreement["agreement"]["overall"]
    cost = (result.get("usage") or {}).get("reported_cost_usd")
    n = overall["n"]
    return {
        "complete": True,
        "n": n,
        "kappa": overall["quadratic_weighted_kappa"],
        "kappa_ci95": overall["quadratic_weighted_kappa_ci95"],
        "exact_agreement": overall["exact_agreement"],
        "mean_signed_difference": overall["mean_signed_difference"],
        "failures": result.get("failures", 0),
        "seconds_per_judgment": agreement["timing"]["seconds_per_judgment"],
        "cost_usd": cost,
        "cost_per_judgment_usd": cost / n if cost is not None and n else None,
    }


def pair_key(pair: Mapping[str, Any]) -> str:
    """Key of a (query, book) pair in the saved results: query_id|source|source_id."""
    return f"{pair['query_id']}|{pair['source']}|{pair['source_id']}"


def pairwise_kappa(grades: Mapping[str, Mapping[str, int]]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Quadratic-weighted kappa between every two graders, on the pairs both graded.

    Args:
        grades: Grader name -> pair key -> grade 0-3

    Returns:
        {grader: {grader: kappa}} (symmetric; 1.0 on the diagonal)
    """
    names = list(grades)
    matrix: Dict[str, Dict[str, Optional[float]]] = {name: {} for name in names}
    for a in names:
        for b in names:
            common = sorted(set(grades[a]) & set(grades[b]))
            if not common:
                matrix[a][b] = None
                continue
            report = agreement_report([grades[a][k] for k in common], [grades[b][k] for k in common],
                                      n_resamples=0)
            matrix[a][b] = report["overall"]["quadratic_weighted_kappa"]
    return matrix


def cost_per_search(explanation: Mapping[str, Any], query: Mapping[str, Any]) -> Optional[float]:
    """Real cost of one search with understanding and three explanations."""
    per_explanation = explanation.get("cost_per_explanation_usd")
    per_query = query.get("cost_per_query_usd")
    if per_explanation is None or per_query is None:
        return None
    return per_query + EXPLAINED_RESULTS * per_explanation


def _within(best: float, value: float, margin: float) -> bool:
    """value is tied with best: at most `margin` below it (rounding-safe)."""
    return best - value <= margin + 1e-9


def select_system_llm(
    explanation_rows: Mapping[str, Mapping[str, Any]],
    query_rows: Mapping[str, Mapping[str, Any]],
    margin: float = TIE_MARGIN,
) -> Dict[str, Any]:
    """
    Apply the system-LLM rule to tables B and C.

    Returns:
        winner (None if no candidate is eligible), excluded {model: reason},
        best citation precision, tied models in tie-break order, and
        decided_by (which criterion separated the winner from the runner-up)
    """
    excluded: Dict[str, str] = {}
    eligible: List[str] = []
    for model in explanation_rows:
        b, c = explanation_rows[model], query_rows.get(model)
        if c is None:
            excluded[model] = "no query-understanding measurement"
        elif b["parsed_share"] != 1.0 or c["parsed_share"] != 1.0:
            excluded[model] = (f"not every call parsed (explanations {b['parsed']}/{b['n']}, "
                               f"queries {c['parsed']}/{c['n']})")
        elif b["citation_precision"] is None:
            excluded[model] = "wrote no citation"
        else:
            eligible.append(model)
    if not eligible:
        return {"winner": None, "excluded": excluded, "eligible": [], "tied": [], "decided_by": None}

    best = max(explanation_rows[m]["citation_precision"] for m in eligible)
    tied = [m for m in eligible if _within(best, explanation_rows[m]["citation_precision"], margin)]

    def cost(model: str) -> float:
        value = cost_per_search(explanation_rows[model], query_rows[model])
        return float("inf") if value is None else value

    def language(model: str) -> float:
        return explanation_rows[model]["in_query_language_share"] or 0.0

    ranked = sorted(tied, key=lambda m: (-language(m), cost(m), m))
    decided_by = "highest citation precision (no tie)"
    if len(ranked) > 1:
        first, second = ranked[0], ranked[1]
        if language(first) != language(second):
            decided_by = "tie on citation precision, broken by the share of explanations in the query language"
        elif cost(first) != cost(second):
            decided_by = "tie on citation precision and query-language share, broken by cost per search"
        else:
            decided_by = "complete tie, broken by model id (alphabetical)"
    return {
        "winner": ranked[0],
        "excluded": excluded,
        "eligible": eligible,
        "best_citation_precision": best,
        "tied": ranked,
        "decided_by": decided_by,
    }


def select_judge(
    judge_rows: Mapping[str, Mapping[str, Any]],
    system_llm: Optional[str],
    margin: float = TIE_MARGIN,
) -> Dict[str, Any]:
    """
    Apply the judge rule to table A, given the selected system LLM.

    Returns:
        winner (None if no candidate is eligible), excluded {model: reason},
        best kappa, tied models in tie-break order and decided_by
    """
    excluded: Dict[str, str] = {}
    eligible: List[str] = []
    for model, row in judge_rows.items():
        if system_llm is not None and vendor(model) == vendor(system_llm):
            excluded[model] = f"same vendor as the system LLM ({vendor(model)})"
        elif not row.get("complete") or row.get("kappa") is None:
            excluded[model] = "agreement measurement incomplete"
        else:
            eligible.append(model)
    if not eligible:
        return {"winner": None, "excluded": excluded, "eligible": [], "tied": [], "decided_by": None}

    best = max(judge_rows[m]["kappa"] for m in eligible)
    tied = [m for m in eligible if _within(best, judge_rows[m]["kappa"], margin)]

    def cost(model: str) -> float:
        value = judge_rows[model].get("cost_per_judgment_usd")
        return float("inf") if value is None else value

    ranked = sorted(tied, key=lambda m: (cost(m), m))
    decided_by = "highest kappa (no tie)"
    if len(ranked) > 1:
        decided_by = ("tie on kappa, broken by cost per judgment" if cost(ranked[0]) != cost(ranked[1])
                      else "complete tie, broken by model id (alphabetical)")
    return {
        "winner": ranked[0],
        "excluded": excluded,
        "eligible": eligible,
        "best_kappa": best,
        "tied": ranked,
        "decided_by": decided_by,
    }


# =============================================================================
# Speed-quality rule (2026-11-11, supersedes the V10 rule above)
# =============================================================================


def select_by_speed_quality(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    margin: float = TIE_MARGIN,
    latency_band: float = LATENCY_BAND,
) -> Dict[str, Any]:
    """
    The speed-quality rule, for any role.

    Candidates whose quality is within `margin` of the best are tied on
    quality; among them the lowest median latency wins, and when another
    tied candidate is within `latency_band` (10%) of that latency the
    cheaper one wins. A candidate with any failed call, or a reason in
    `excluded_reason`, is excluded before anything else.

    Args:
        rows: name -> {quality, median_latency_s, cost_usd, failed_calls,
            excluded_reason (optional)}; higher quality is better
        margin: Quality tie band (absolute)
        latency_band: Relative latency band within which cost decides

    Returns:
        winner (None if nobody is eligible), excluded {name: reason},
        eligible, best_quality, tied (by latency), within_latency_band
        (the candidates cost decided between) and decided_by
    """
    excluded: Dict[str, str] = {}
    eligible: List[str] = []
    for name, row in rows.items():
        if row.get("excluded_reason"):
            excluded[name] = str(row["excluded_reason"])
        elif row.get("failed_calls"):
            excluded[name] = f"{row['failed_calls']} failed call(s) in the measurement"
        elif row.get("quality") is None or row.get("median_latency_s") is None:
            excluded[name] = "no quality or latency measurement"
        else:
            eligible.append(name)
    if not eligible:
        return {"winner": None, "excluded": excluded, "eligible": [], "tied": [],
                "within_latency_band": [], "best_quality": None, "decided_by": None}

    def latency(name: str) -> float:
        return float(rows[name]["median_latency_s"])

    def cost(name: str) -> float:
        value = rows[name].get("cost_usd")
        return float("inf") if value is None else float(value)

    best = max(float(rows[name]["quality"]) for name in eligible)
    tied = sorted((n for n in eligible if _within(best, float(rows[n]["quality"]), margin)),
                  key=lambda n: (latency(n), n))
    fastest = latency(tied[0])
    band = [n for n in tied if latency(n) <= fastest * (1 + latency_band) + 1e-12]
    winner = min(band, key=lambda n: (cost(n), latency(n), n))
    if len(tied) == 1:
        decided_by = "highest quality, no other candidate within the tie band"
    elif len(band) == 1:
        decided_by = (f"tie on quality (within {margin}), broken by the lowest median latency "
                      f"(no other tied candidate within {latency_band:.0%} of it)")
    else:
        decided_by = (f"tie on quality (within {margin}) and on latency (within {latency_band:.0%}), "
                      f"broken by the lower cost")
    return {"winner": winner, "excluded": excluded, "eligible": eligible, "best_quality": best,
            "tied": tied, "within_latency_band": band, "decided_by": decided_by}


def select_system_llm_speed_quality(
    explanation_rows: Mapping[str, Mapping[str, Any]],
    query_rows: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    System LLM under the speed-quality rule, from tables B and C of V10.

    Quality: pre-guardrail citation precision (B). Eligible: every call
    parsed in B and C, and at least MIN_QUERY_LANGUAGE_SHARE of the
    explanations in the query language. Latency: median per explanation (B);
    cost: real cost per explanation (B).
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for model, b in explanation_rows.items():
        c = query_rows.get(model)
        reason = None
        if c is None:
            reason = "no query-understanding measurement"
        elif b["parsed_share"] != 1.0 or c["parsed_share"] != 1.0:
            reason = (f"not every call parsed (explanations {b['parsed']}/{b['n']}, "
                      f"queries {c['parsed']}/{c['n']})")
        elif (b["in_query_language_share"] or 0.0) < MIN_QUERY_LANGUAGE_SHARE - 1e-9:
            reason = (f"{_f(b['in_query_language_share'])} of the explanations in the query language "
                      f"(at least {MIN_QUERY_LANGUAGE_SHARE:.2f} required)")
        rows[model] = {
            "quality": b["citation_precision"],
            "median_latency_s": b["median_latency_s"],
            "cost_usd": b["cost_per_explanation_usd"],
            "failed_calls": 0,
            "excluded_reason": reason,
        }
    return {**select_by_speed_quality(rows), "rows": rows,
            "quality": "pre-guardrail citation precision (B)",
            "latency": "median latency per explanation (B)", "cost": "real cost per explanation (B)"}


def select_judge_speed_quality(
    judge_rows: Mapping[str, Mapping[str, Any]],
    system_llm: Optional[str],
    *,
    margin: float = TIE_MARGIN,
) -> Dict[str, Any]:
    """
    Judge under the speed-quality rule: speed does not matter for the judge.

    Among the candidates of another vendor than `system_llm` that completed
    measurement A without a failed judge call, the highest quadratic-weighted
    kappa; candidates within `margin` are tied and the cheaper one (real cost
    per judgment) wins.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for model, row in judge_rows.items():
        reason = None
        if system_llm is not None and vendor(model) == vendor(system_llm):
            reason = f"same vendor as the system LLM ({vendor(model)})"
        elif not row.get("complete") or row.get("kappa") is None:
            reason = "agreement measurement incomplete"
        rows[model] = {
            "quality": row.get("kappa"),
            # Speed is not a criterion: one latency for all, so cost decides every tie.
            "median_latency_s": 0.0,
            "cost_usd": row.get("cost_per_judgment_usd"),
            "failed_calls": row.get("failures") or 0,
            "excluded_reason": reason,
        }
    decision = select_by_speed_quality(rows, margin=margin)
    if decision["winner"] is not None:
        decision["decided_by"] = ("highest kappa, no other candidate within the tie band"
                                  if len(decision["tied"]) == 1 else
                                  f"tie on kappa (within {margin}), broken by the lower cost per judgment")
    return {**decision, "rows": rows, "quality": "quadratic-weighted kappa (A)",
            "cost": "real cost per judgment (A)"}


def strict_failure_exclusion_judge(
    judge_rows: Mapping[str, Mapping[str, Any]],
    explanation_rows: Mapping[str, Mapping[str, Any]],
    query_rows: Mapping[str, Mapping[str, Any]],
    system_llm: Optional[str],
) -> Dict[str, Any]:
    """
    The judge under the other reading of the exclusion clause (reported, not applied).

    The rule says a candidate that failed any call in the measurement is
    excluded. Applied per role, the judge's measurement is A. Read across
    the whole V10 measurement, a failed call in B or C also excludes a model
    from the judge role; this function gives the judge under that reading.
    """
    failed_elsewhere = {
        model: (explanation_rows[model]["n"] - explanation_rows[model]["parsed"]
                + (query_rows.get(model, {}).get("n", 0) - query_rows.get(model, {}).get("parsed", 0)))
        for model in explanation_rows
    }
    rows = {model: dict(row, failures=(row.get("failures") or 0) + failed_elsewhere.get(model, 0))
            for model, row in judge_rows.items()}
    return select_judge_speed_quality(rows, system_llm)


def speed_quality_decision(tables: Mapping[str, Any]) -> Dict[str, Any]:
    """Both roles under the speed-quality rule, from the V10 tables (no model call)."""
    system = select_system_llm_speed_quality(tables["explanation"], tables["query_understanding"])
    judge = select_judge_speed_quality(tables["judge"], system["winner"])
    strict = strict_failure_exclusion_judge(tables["judge"], tables["explanation"],
                                            tables["query_understanding"], system["winner"])
    return {
        "date": SPEED_QUALITY_RULE_DATE,
        "reason": SPEED_QUALITY_RULE_REASON,
        "rule": SPEED_QUALITY_RULE_TEXT,
        "system_llm": system,
        "judge": judge,
        "judge_if_failures_anywhere_exclude": {"winner": strict["winner"], "tied": strict["tied"],
                                               "excluded": strict["excluded"],
                                               "decided_by": strict["decided_by"]},
    }


def check_candidates(
    models: Mapping[str, Reasoning], catalog: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    What OpenRouter's model list says about each candidate.

    A candidate is dropped when it is not listed or when no endpoint of it
    supports structured outputs (JSON schema): every call of the project
    asks for a JSON schema, and the client tells OpenRouter to route only to
    endpoints that support every parameter sent.

    Args:
        models: Model id -> thinking mode
        catalog: OpenRouter model list, id -> {"pricing", "supported_parameters"}

    Returns:
        {model: {listed, input_per_mtok, output_per_mtok, structured_outputs,
        reasoning, dropped (reason or None)}}
    """
    info: Dict[str, Dict[str, Any]] = {}
    for model, reasoning in models.items():
        entry = catalog.get(model)
        if entry is None:
            info[model] = {"listed": False, "reasoning": reasoning,
                           "dropped": "not in the OpenRouter model list"}
            continue
        parameters = list(entry.get("supported_parameters") or [])
        pricing = entry.get("pricing") or {}
        structured = "structured_outputs" in parameters
        info[model] = {
            "listed": True,
            "input_per_mtok": float(pricing.get("prompt", 0)) * 1_000_000,
            "output_per_mtok": float(pricing.get("completion", 0)) * 1_000_000,
            "supported_parameters": parameters,
            "structured_outputs": structured,
            "reasoning": reasoning,
            "dropped": None if structured else (
                "no endpoint supports structured outputs (JSON schema); OpenRouter answers 404 "
                "'No endpoints found that can handle the requested parameters'"
            ),
        }
    return info


def compute_tables(results: Mapping[str, Any]) -> Dict[str, Any]:
    """Tables A, B, C, the pairwise kappa matrix and the decision, from raw results."""
    judge_rows = {model: summarize_judge(r) for model, r in results.get("judge", {}).items()}
    explanation_rows = {model: summarize_explanations(r["records"])
                        for model, r in results.get("explanation", {}).items()}
    query_rows = {model: summarize_queries(r["records"])
                  for model, r in results.get("query_understanding", {}).items()}

    grades: Dict[str, Dict[str, int]] = {}
    original: Dict[str, int] = {}
    for model, result in results.get("judge", {}).items():
        if result.get("agreement"):
            pairs = result["agreement"]["pairs"]
            grades[model] = {pair_key(p): p["llm"] for p in pairs}
            original.update({pair_key(p): p["original"] for p in pairs})
    local = results.get("local_judge", {}).get("grades") or {}
    if local:
        grades[f"{LOCAL_JUDGE_MODEL} (local)"] = dict(local)
    if original:
        grades["original judgments"] = original
    local_row = None
    if local and original:
        common = [key for key in original if key in local]
        local_report = agreement_report([original[k] for k in common], [local[k] for k in common],
                                        n_resamples=2000, seed=AGREEMENT_SEED)["overall"]
        local_row = {
            "n": local_report["n"],
            "kappa": local_report["quadratic_weighted_kappa"],
            "kappa_ci95": local_report["quadratic_weighted_kappa_ci95"],
            "exact_agreement": local_report["exact_agreement"],
            "mean_signed_difference": local_report["mean_signed_difference"],
        }

    system = select_system_llm(explanation_rows, query_rows)
    judge = select_judge(judge_rows, system["winner"])
    tables = {
        "judge": judge_rows,
        "local_judge": local_row,
        "explanation": explanation_rows,
        "query_understanding": query_rows,
        "cost_per_search_usd": {
            model: cost_per_search(explanation_rows[model], query_rows[model])
            for model in explanation_rows if model in query_rows
        },
        "pairwise_kappa": pairwise_kappa(grades) if grades else {},
        # V10 rule (quality only), kept as history
        "decision": {"system_llm": system, "judge": judge},
    }
    # Speed-quality rule (2026-11-11), which supersedes the V10 decision
    tables["decision_speed_quality"] = speed_quality_decision(tables)
    return tables


# =============================================================================
# Markdown report
# =============================================================================


def _f(value: Optional[float], digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _usd(value: Optional[float], digits: int = 5) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _reasoning_text(reasoning: Reasoning) -> str:
    if reasoning is False:
        return "off"
    if reasoning is True:
        return "on"
    return "model default" if reasoning is None else f"effort {reasoning}"


def failure_patterns(results: Mapping[str, Any]) -> List[str]:
    """Plain-language notes on how each model failed, computed from the raw records."""
    notes: List[str] = []
    for model, result in results.get("explanation", {}).items():
        records = result["records"]
        failed = Counter(record["failure"] for record in records if not record["parsed"])
        if failed:
            notes.append(f"`{model}` explanations: {sum(failed.values())} of {len(records)} calls did not "
                         f"parse ({', '.join(f'{k}: {v}' for k, v in sorted(failed.items()))}).")
        wrong_language = Counter()
        for record in records:
            if record["parsed"]:
                detected = explanation_language(f"{record['summary']} {record['reasoning']}")
                if detected != record["language"]:
                    wrong_language[(record["language"], detected or "undetermined")] += 1
        if wrong_language:
            parts = ", ".join(f"{count} {query} queries answered in {got}"
                              for (query, got), count in sorted(wrong_language.items()))
            notes.append(f"`{model}` explanations not in the query language: {parts}.")
        errors = Counter(citation.get("error") for record in records if record["parsed"]
                         for citation in record["citations"] if not citation["valid"])
        if errors:
            notes.append(f"`{model}` invalid citations: "
                         + ", ".join(f"{count} {kind}" for kind, count in sorted(errors.items())) + ".")
    for model, result in results.get("query_understanding", {}).items():
        records = result["records"]
        failed = [record for record in records if not record["parsed"]]
        if failed:
            notes.append(f"`{model}` query understanding: {len(failed)} of {len(records)} queries with a "
                         f"failed step ({'; '.join(sorted({(r['error'] or 'graph error')[:120] for r in failed}))}).")
        examples = []
        for record in records:
            if record["raw_filters"] is None:
                continue
            flags = unrequested_filters(record["text"], record["raw_filters"])
            if flags["language"] or flags["year"]:
                filters = record["raw_filters"]
                shown = ", ".join(
                    f"{name}={filters[name]}" for name in ("language", "min_year", "max_year")
                    if filters.get(name) is not None
                    and (name == "language" and flags["language"] or name != "language" and flags["year"])
                )
                examples.append(f"\"{record['text']}\" ({shown})")
        if examples:
            notes.append(f"`{model}` unrequested filters ({len(examples)}): " + "; ".join(examples) + ".")
    for model, result in results.get("judge", {}).items():
        if result.get("failures"):
            errors = result.get("errors") or {}
            notes.append(f"`{model}` judge: {result['failures']} failed judge calls "
                         f"({', '.join(f'{k}: {v}' for k, v in sorted(errors.items()))}); "
                         f"{result.get('attempts', 1)} runs of `agreement`.")
    return notes


def total_spend(results: Mapping[str, Any]) -> Dict[str, Any]:
    """Real cost per model and role (sum of the costs OpenRouter reported)."""
    per_model: Dict[str, Dict[str, Optional[float]]] = {}
    for role in ROLES:
        for model, result in results.get(role, {}).items():
            per_model.setdefault(model, {})[role] = (result.get("usage") or {}).get("reported_cost_usd")
    total = _sum_or_none([cost for roles in per_model.values() for cost in roles.values()])
    return {"per_model": per_model, "total_usd": total}


def _speed_quality_markdown(current: Mapping[str, Any]) -> str:
    """Section of the speed-quality decision (2026-11-11), which supersedes the V10 one."""
    system, judge = current["system_llm"], current["judge"]

    def status(block: Mapping[str, Any], name: str) -> str:
        if name == block["winner"]:
            return "selected"
        if name in block["excluded"]:
            return f"excluded: {block['excluded'][name]}"
        if name in block["tied"]:
            return "tied on quality"
        return "below the tie band"

    system_rows = [
        [f"`{m}`", _f(r["quality"]), _f(r["median_latency_s"], 2), _usd(r["cost_usd"], 6), status(system, m)]
        for m, r in system["rows"].items()
    ]
    judge_rows = [
        [f"`{m}`", _f(r["quality"]), _usd(r["cost_usd"], 6), str(r["failed_calls"]), status(judge, m)]
        for m, r in judge["rows"].items()
    ]
    alternative = current.get("judge_if_failures_anywhere_exclude") or {}
    alternative_line = ""
    if alternative and alternative.get("winner") != judge["winner"]:
        alternative_line = (
            f"\n\nReading of the exclusion clause: it is applied per role, so a judge candidate is excluded "
            f"for a failed call in A (none failed). If a failed call anywhere in the V10 measurement (B or C) "
            f"excluded a model from every role, the judge would be `{alternative.get('winner')}` "
            f"({alternative.get('decided_by')})."
        )

    def tied_text(block: Mapping[str, Any]) -> str:
        return ", ".join(f"`{m}`" for m in block["tied"]) or "none"

    return f"""## Decision with the speed-quality rule ({current["date"]}; supersedes the V10 decision)

Why the rule changed: {current["reason"]}

Rule: {current["rule"]}

Applied mechanically to the saved V10 measurements (tables A, B and C above; no new model call):

System LLM ({system["quality"]}; {system["latency"]}; {system["cost"]}):

{_table(["Model", "Citation precision", "Median latency / explanation (s)", "Real cost / explanation (USD)",
         "Status"], system_rows)}

- Result: **`{system["winner"]}`**. Best quality {_f(system["best_quality"])}; tied within {TIE_MARGIN}:
  {tied_text(system)}. Decided by: {system["decided_by"]}.

Judge ({judge["quality"]}; {judge["cost"]}):

{_table(["Model", "Kappa", "Real cost / judgment (USD)", "Failed judge calls", "Status"], judge_rows)}

- Result: **`{judge["winner"]}`**. Best kappa {_f(judge["best_quality"])}; tied within {TIE_MARGIN}:
  {tied_text(judge)}. Decided by: {judge["decided_by"]}.{alternative_line}

These are the OpenRouter defaults from {current["date"]} on (`DEFAULT_MODELS["openrouter"]` and
`DEFAULT_JUDGE_MODELS["openrouter"]` in `app/infrastructure/config/settings.py`, and `.env.example`)."""


def build_markdown(results: Mapping[str, Any], tables: Mapping[str, Any]) -> str:
    """LLM_MODEL_SELECTION.md from the raw results and the computed tables."""
    run = results.get("run", {})
    candidates = results.get("candidates", {})
    decision = tables["decision"]
    system, judge = decision["system_llm"], decision["judge"]

    price_rows = []
    for model, info in candidates.items():
        price_rows.append([
            f"`{model}`",
            "yes" if info.get("listed") else "no",
            _f(info.get("input_per_mtok"), 4) if info.get("listed") else "n/a",
            _f(info.get("output_per_mtok"), 4) if info.get("listed") else "n/a",
            "yes" if info.get("structured_outputs") else "no",
            _reasoning_text(info.get("reasoning")),
            info.get("dropped") or "measured",
        ])

    judge_rows = []
    for model, row in tables["judge"].items():
        if not row.get("complete"):
            judge_rows.append([f"`{model}`", "incomplete"] + ["n/a"] * 5 + [str(row.get("failures")),
                                                                            _usd(row.get("cost_usd"))])
            continue
        ci = row["kappa_ci95"]
        judge_rows.append([
            f"`{model}`", str(row["n"]),
            f"{_f(row['kappa'])} [{_f(ci[0])}, {_f(ci[1])}]" if ci else _f(row["kappa"]),
            _f(row["exact_agreement"]), _f(row["mean_signed_difference"], signed=True),
            str(row["failures"]), _f(row["seconds_per_judgment"], 2),
            _usd(row["cost_usd"]), _usd(row["cost_per_judgment_usd"], 6),
        ])
    local = tables.get("local_judge")
    if local:
        ci = local["kappa_ci95"]
        judge_rows.append([
            f"`{LOCAL_JUDGE_MODEL}` (local, V05; reference only)", str(local["n"]),
            f"{_f(local['kappa'])} [{_f(ci[0])}, {_f(ci[1])}]" if ci else _f(local["kappa"]),
            _f(local["exact_agreement"]), _f(local["mean_signed_difference"], signed=True),
            "0", "about 4 (V05)", "n/a (local)", "n/a (local)",
        ])
    table_a = _table(
        ["Model", "n", "Quadratic-weighted kappa [95% CI]", "Exact", "Mean LLM - original", "Failures",
         "s / judgment", "Real cost (USD)", "Real cost / judgment (USD)"],
        judge_rows,
    )

    matrix = tables.get("pairwise_kappa") or {}
    names = list(matrix)
    table_kappa = _table(
        ["Kappa"] + [f"`{n}`" for n in names],
        [[f"`{a}`"] + [_f(matrix[a][b]) for b in names] for a in names],
    ) if names else "No judge results."

    table_b = _table(
        ["Model", "Parsed", "Citations written", "Citation precision (before guardrail)",
         "Zero valid citations", "In query language", "Median latency (s)", "Real cost / explanation (USD)"],
        [[f"`{m}`", f"{r['parsed']}/{r['n']}", str(r["citations"]), _f(r["citation_precision"]),
          _f(r["zero_valid_share"]), _f(r["in_query_language_share"]), _f(r["median_latency_s"], 2),
          _usd(r["cost_per_explanation_usd"], 6)]
         for m, r in tables["explanation"].items()],
    )
    table_c = _table(
        ["Model", "Parsed", "Unrequested language filter", "Unrequested year filter",
         "Unrequested language or year", "Median latency / query (s)", "Real cost / query (USD)",
         "Real cost / search (USD)"],
        [[f"`{m}`", f"{r['parsed']}/{r['n']}", _f(r["unrequested_language_share"]),
          _f(r["unrequested_year_share"]), _f(r["unrequested_any_share"]), _f(r["median_latency_s"], 2),
          _usd(r["cost_per_query_usd"], 6), _usd(tables["cost_per_search_usd"].get(m), 6)]
         for m, r in tables["query_understanding"].items()],
    )

    def excluded_lines(block: Mapping[str, Any]) -> str:
        return "".join(f"\n  - `{m}`: {reason}" for m, reason in block["excluded"].items()) or " none"

    system_text = (
        f"**`{system['winner']}`**. Eligible: {', '.join(f'`{m}`' for m in system['eligible'])}. "
        f"Best pre-guardrail citation precision: {_f(system['best_citation_precision'])}; tied within "
        f"{TIE_MARGIN}: {', '.join(f'`{m}`' for m in system['tied'])}. Decided by: {system['decided_by']}."
        if system["winner"] else "No candidate is eligible."
    )
    judge_text = (
        f"**`{judge['winner']}`**. Eligible: {', '.join(f'`{m}`' for m in judge['eligible'])}. "
        f"Best kappa: {_f(judge['best_kappa'])}; tied within {TIE_MARGIN}: "
        f"{', '.join(f'`{m}`' for m in judge['tied'])}. Decided by: {judge['decided_by']}."
        if judge["winner"] else "No candidate is eligible."
    )

    current = tables.get("decision_speed_quality") or speed_quality_decision(tables)
    speed_quality_section = _speed_quality_markdown(current)
    current_line = (
        f"- **Current decision** (speed-quality rule of {current['date']}, which supersedes the V10 "
        f"decision below): system LLM **`{current['system_llm']['winner']}`**, judge "
        f"**`{current['judge']['winner']}`**. The V10 rule and decision are kept as history."
    )

    spend = total_spend(results)
    spend_rows = [
        [f"`{model}`"] + [_usd(roles.get(role)) for role in ROLES]
        + [_usd(_sum_or_none(list(roles.values())))]
        for model, roles in spend["per_model"].items()
    ]
    key_usage = run.get("key_usage_usd") or {}
    key_line = ""
    if key_usage.get("before") is not None and key_usage.get("after") is not None:
        key_line = (f"\nCross-check: the API key's usage counter (OpenRouter `/key`) went from "
                    f"{key_usage['before']:.4f} to {key_usage['after']:.4f} USD during the run "
                    f"({key_usage['after'] - key_usage['before']:.4f} USD); it is read right after the last "
                    f"call and can lag behind the most recent ones.")
    notes = failure_patterns(results)
    tasks = results.get("explanation_tasks", [])
    explained_queries = ", ".join(sorted({task["query_id"] for task in tasks}))

    return f"""# Choice of the system LLM and of the relevance judge (V10)

- Date of the measurements: {run.get("date", "n/a")}
- Git commit: `{run.get("git_commit") or "unknown"}`{" (with uncommitted changes in tracked files)" if run.get("git_worktree_dirty") else ""}
- Raw results: `{DEFAULT_RESULTS_JSON}`; this report is rendered from them by
  `python -m app.evaluation.llm_model_selection report`.
- All models are called through OpenRouter (the local Ollama model is no longer used).
{current_line}

## Why

The project needs two LLM roles: the system LLM (query understanding and grounded explanations in
Spanish and English) and the judge (relevance grades for the evaluation), from a different vendor
than the system LLM so that the judge does not grade its own family's output. Public rankings of
cheap hosted models disagree with each other, so the choice is made on this project's own tasks.
GPT models are excluded by the author's decision of 2026-11-11.

## Candidates and prices

List prices from `{OPENROUTER_MODELS_URL}` on the date of the run, USD per million tokens. They are
the cheapest endpoint of each model; the client asks OpenRouter to route only to endpoints that
support every parameter sent (JSON schema, thinking switch), so the price actually paid can be
higher. All costs below are the costs OpenRouter reported for each call.

{_table(["Model", "Listed", "Input", "Output", "Structured outputs", "Thinking", "Status"], price_rows)}

## Design (fixed before any result was seen)

- A. Judge: the existing agreement measurement (`python -m app.evaluation.relevance_judge agreement
  --sample {AGREEMENT_SAMPLE} --seed {AGREEMENT_SEED}`, one judge per candidate) on the same seeded
  sample of original judgments, stratified by (grade, query language). Prompt `v1.0`, temperature 0,
  at most 256 output tokens.
- B. Explanations: the first {EXPLAINED_QUERIES_PER_LANGUAGE} Spanish and first
  {EXPLAINED_QUERIES_PER_LANGUAGE} English test queries by id ({explained_queries}), the top
  {EXPLAINED_RESULTS} results of the production hybrid (BM25 + MiniLM, RRF, depth {SEARCH_DEPTH}) for
  each: {len(tasks)} grounded explanations per model through the real `LangChainLLMClient`
  (temperature 0, at most 1024 output tokens). Citation precision before the guardrail is the share
  of the citations the model wrote whose snippet the guardrail's check
  (`app/domain/citation_validation.py`: the normalized snippet is a substring of the cited field)
  accepts, over all citations of the model. The raw output is read through the client's
  `raw_output_observer`.
- Language of an explanation (summary and reasoning, not the quoted snippets): count the words that
  are Spanish-only and English-only stopwords of the BM25 tokenizer
  (`app/infrastructure/search/text_normalization.py`, accents folded; words in both lists ignored);
  the larger count gives the language, a tie gives none (counted as not the query language).
- C. Query understanding: the 44 test queries through `extract_query_intent`. A filter is
  unrequested when the deterministic rule of `app/infrastructure/llm/filter_guards.py` would drop it:
  a language the query does not ask for with a preposition ("en español", "in English"), or a year
  although the query contains no year. Measured on the filters as the model extracted them, before
  the guards.
- Parsed: an explanation parsed when the model output fit the schema; a query parsed when all three
  steps of the graph (intent, filters, reformulation) did. The OpenAI client retries transient HTTP
  errors twice and `LangChainLLMClient` once more.
- Latency and tokens: the V08 usage recorder; latency per explanation is its single call, per query
  the sum of its three calls. Thinking off where the model allows it, otherwise effort `low`.
  At most {MAX_IN_FLIGHT} requests in flight.

## Selection rule of V10 (fixed before any result was seen; superseded on {SPEED_QUALITY_RULE_DATE})

- System LLM: among the candidates with 100% parsed calls in B and C, the highest pre-guardrail
  citation precision; candidates within {TIE_MARGIN} of the best are tied, and ties are broken by the
  higher share of explanations in the query language, then by the lower real cost of one search
  (one query understanding + {EXPLAINED_RESULTS} explanations).
- Judge: among the candidates from a different vendor than the selected system LLM (the vendor is
  the prefix of the id), the highest quadratic-weighted kappa in A; candidates within {TIE_MARGIN} are
  tied and the cheaper one (real cost per judgment) wins.

## A. Judge role (agreement with the original judgments, {AGREEMENT_SAMPLE} pairs)

{table_a}

Mean LLM - original: positive when the model grades higher. The kappa interval resamples pairs and
ignores that pairs of one query are not independent, so it is optimistic. s / judgment: one request
at a time per model.

Pairwise quadratic-weighted kappa on the same {AGREEMENT_SAMPLE} pairs:

{table_kappa}

## B. Explanation role ({len(tasks)} explanations per model)

{table_b}

Citation precision over all citations written; zero-valid and query-language shares over the parsed
explanations (an explanation with no valid citation is replaced by the guardrail's fallback text).

## C. Query-understanding role (44 queries per model)

{table_c}

Unrequested shares over the queries whose filter step parsed. Real cost per search: one query
understanding plus {EXPLAINED_RESULTS} explanations.

## Decision of V10 (mechanical application of the V10 rule; history, superseded)

- System LLM: {system_text}
- Excluded from the system-LLM choice:{excluded_lines(system) if system.get("excluded") is not None else " none"}
- Judge: {judge_text}
- Excluded from the judge choice:{excluded_lines(judge) if judge.get("excluded") is not None else " none"}

{speed_quality_section}

## Failure patterns

{chr(10).join(f"- {note}" for note in notes) if notes else "- None."}

## Spend

Real cost reported by OpenRouter, USD:

{_table(["Model", "A. Judge", "B. Explanations", "C. Query understanding", "Total"], spend_rows)}

Total: {_usd(spend["total_usd"])} USD.{key_line}

## Limits

- One run per model at temperature 0: OpenRouter routes each call to one of several endpoints
  (different hardware and quantization), so a repeat can differ slightly.
- 60 explanations and 44 queries per model: differences of a few points are within noise.
- The agreement is with judgments of unknown provenance (`QRELS_DATASHEET.md`): agreement, not
  correctness. Prices and model versions are those of the run date.

## Reproduce

    python -m app.evaluation.llm_model_selection run      # calls the models (needs OPENROUTER_API_KEY)
    python -m app.evaluation.llm_model_selection report   # re-renders this file, no model call
"""


# =============================================================================
# Measurements (network)
# =============================================================================


@dataclass(frozen=True)
class ExplanationTask:
    """One (query, result) pair to explain."""

    query_id: str
    language: str
    query_text: str
    rank: int
    book: Book

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id, "language": self.language, "query_text": self.query_text,
            "rank": self.rank, "book_id": str(self.book.id), "source": self.book.source,
            "source_id": self.book.source_id, "title": self.book.title,
        }


def explanation_queries(queries: Sequence[Any], per_language: int = EXPLAINED_QUERIES_PER_LANGUAGE) -> List[Any]:
    """The first `per_language` Spanish and English test queries by query id."""
    ordered = sorted(queries, key=lambda query: query.query_id)
    chosen = []
    for language in ("es", "en"):
        chosen += [query for query in ordered if query.language == language][:per_language]
    return sorted(chosen, key=lambda query: query.query_id)


def production_top_results(
    queries: Sequence[Any], books: Sequence[Book], db_path: str, indexes_dir: str
) -> List[ExplanationTask]:
    """The top EXPLAINED_RESULTS of the production hybrid (hybrid_rrf, depth 10) per query."""
    from app.domain.services import SearchService
    from app.evaluation.comparison import run_system
    from app.infrastructure.search.bm25_search_repository import BM25SearchRepository
    from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
    from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository

    # Legacy padding (zero-score books fill the list): reproduces the published results (V13)
    bm25 = BM25SearchRepository(zero_score_padding=True)
    bm25.load_index(str(Path(indexes_dir) / "bm25_index.pkl"))
    embeddings = EmbeddingsStoreFaiss()
    embeddings.load_index(str(Path(indexes_dir) / "faiss_index"))
    vector = FaissVectorSearchRepository(embeddings, list(books))
    service = SearchService(lexical_search=bm25, vector_search=vector, embeddings_store=embeddings)
    tasks = []
    for query in queries:
        results = run_system("hybrid_rrf", query.text, search_service=service, lexical_repo=bm25,
                             vector_repo=vector, embeddings_store=embeddings, depth=SEARCH_DEPTH)
        for rank, result in enumerate(results[:EXPLAINED_RESULTS], start=1):
            tasks.append(ExplanationTask(query.query_id, query.language, query.text, rank, result.book))
    return tasks


def _calls_summary(records: Sequence[Any]) -> Dict[str, Any]:
    """Tokens, latency and reported cost of the LLM calls of one task."""
    return {
        "calls": len(records),
        "call_errors": [record.error for record in records if record.error],
        "latency_s": sum(record.latency_s for record in records),
        "input_tokens": _sum_or_none([record.input_tokens for record in records]),
        "output_tokens": _sum_or_none([record.output_tokens for record in records]),
        "cost_usd": _sum_or_none([record.cost_usd for record in records]),
    }


def explanation_record(task: ExplanationTask, raw: Any, explanation: Any, calls: Sequence[Any],
                       wall_s: float) -> Dict[str, Any]:
    """Raw record of one explanation call: what the model wrote, and what the guardrail kept."""
    usage = _calls_summary(calls)
    record: Dict[str, Any] = {**task.to_dict(), "parsed": raw is not None, **usage, "wall_s": wall_s}
    if raw is None:
        error = usage["call_errors"][-1] if usage["call_errors"] else None
        record.update(failure=("output did not fit the schema" + (f" ({error})" if error else "")
                               if error in SCHEMA_ERRORS or error is None else f"model call failed ({error})"),
                      summary="", reasoning="", confidence=None, citations=[])
    else:
        record.update(
            failure=None,
            summary=raw.summary,
            reasoning=raw.reasoning,
            confidence=raw.confidence,
            citations=[
                {"chunk_id": c.chunk_id, "snippet": c.snippet, "relevance_score": c.relevance_score,
                 "valid": citation_is_valid(c.chunk_id, c.snippet, task.book),
                 "error": citation_error(c.chunk_id, c.snippet, task.book)}
                for c in raw.citations
            ],
        )
    record.update(final_text=explanation.text, final_citations=len(explanation.citations))
    return record


def query_record(query: Any, state: Optional[Mapping[str, Any]], intent: Any, calls: Sequence[Any],
                 wall_s: float) -> Dict[str, Any]:
    """Raw record of one query-understanding call: filters as extracted and as kept."""
    usage = _calls_summary(calls)
    steps = {name: state is not None and state.get(name) is not None
             for name in ("intent", "filters", "reformulated")}
    filters = state["filters"].model_dump() if steps["filters"] else None
    final = intent.extracted_filters
    error = (state.get("error") if state is not None else "the graph raised") or None
    return {
        "query_id": query.query_id,
        "language": query.language,
        "text": query.text,
        "parsed": all(steps.values()),
        "steps_parsed": steps,
        "error": error[:300] if error else None,  # the last failed step's message, shortened
        "raw_filters": filters,
        "final_filters": {"language": final.language, "category": final.category,
                          "min_year": final.min_year, "max_year": final.max_year},
        "intent_type": intent.intent_type,
        "reformulated_query": intent.reformulated_query,
        **usage,
        "wall_s": wall_s,
    }


def build_system_client(model: str, reasoning: Reasoning, categories: Sequence[str]):
    """A LangChainLLMClient for one candidate, built as the API builds it (LLMSettings defaults)."""
    from app.infrastructure.config.settings import DEFAULT_OPENROUTER_URL, LLMSettings
    from app.infrastructure.llm.langchain_llm_client import LangChainLLMClient

    settings = LLMSettings(provider="openrouter", model=model, base_url=DEFAULT_OPENROUTER_URL,
                           num_ctx=None, reasoning=reasoning)
    return LangChainLLMClient(
        provider=settings.provider, model_name=settings.model, temperature=settings.temperature,
        max_retries=settings.max_retries, base_url=settings.base_url, num_ctx=settings.num_ctx,
        reasoning=settings.reasoning, known_categories=list(categories),
        apply_extracted_category=settings.apply_extracted_category, max_tokens=settings.max_tokens,
        timeout_s=settings.timeout_s,
    )


class _WorkerClients:
    """
    One LLM client per worker thread, so each client's usage records and raw
    outputs belong to the one task its thread is running.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._local = threading.local()
        self._lock = threading.Lock()
        self.clients: List[Any] = []

    def get(self) -> Tuple[Any, List[Any]]:
        if not hasattr(self._local, "client"):
            client = self._factory()
            raw: List[Any] = []
            client.raw_output_observer = lambda operation, output: raw.append(output)
            self._local.client, self._local.raw = client, raw
            with self._lock:
                self.clients.append(client)
        return self._local.client, self._local.raw

    def usage(self) -> Dict[str, Any]:
        """Usage summary over every worker's client (one row per operation, plus total)."""
        from app.infrastructure.llm.usage import LLMUsageRecorder

        merged = LLMUsageRecorder()
        for client in self.clients:
            merged._records.extend(client.usage_recorder.records)
        return merged.summary()


def _run_tasks(items: Sequence[Any], clients: _WorkerClients, call: Callable[[Any, Any], Any],
               make_record: Callable[..., Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """Run `call(client, item)` for every item with MAX_IN_FLIGHT workers; one record per item."""
    done = [0]
    lock = threading.Lock()

    def work(item: Any) -> Dict[str, Any]:
        client, raw = clients.get()
        first = len(client.usage_recorder.records)
        raw.clear()
        started = time.perf_counter()
        output = call(client, item)
        wall = time.perf_counter() - started
        record = make_record(item, raw[0] if raw else None, output,
                             client.usage_recorder.records[first:], wall)
        with lock:
            done[0] += 1
            print(f"[{label}] {done[0]}/{len(items)} parsed={record['parsed']} {wall:.1f}s", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT) as pool:
        return list(pool.map(work, items))


def run_explanations(model: str, reasoning: Reasoning, tasks: Sequence[ExplanationTask],
                     categories: Sequence[str]) -> Dict[str, Any]:
    """Measurement B for one model."""
    clients = _WorkerClients(lambda: build_system_client(model, reasoning, categories))
    records = _run_tasks(
        tasks, clients,
        call=lambda client, task: client.generate_grounded_explanation(task.query_text, task.book),
        make_record=explanation_record, label=f"B {model}",
    )
    return {"records": records, "usage": clients.usage()["total"]}


def run_query_understanding(model: str, reasoning: Reasoning, queries: Sequence[Any],
                            categories: Sequence[str]) -> Dict[str, Any]:
    """Measurement C for one model."""
    clients = _WorkerClients(lambda: build_system_client(model, reasoning, categories))
    records = _run_tasks(
        queries, clients,
        call=lambda client, query: client.extract_query_intent(query.text),
        make_record=query_record, label=f"C {model}",
    )
    return {"records": records, "usage": clients.usage()["total"]}


class CountingJudge:
    """Wraps a relevance judge and counts the judge calls that raised (failures)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.model_name = inner.model_name
        self.prompt_version = inner.prompt_version
        self.calls = 0
        self.failures = 0
        self.errors: Counter = Counter()

    def judge(self, query_text: str, book: Book) -> Tuple[int, str]:
        self.calls += 1
        try:
            return self.inner.judge(query_text, book)
        except Exception as error:
            self.failures += 1
            self.errors[type(error).__name__] += 1
            raise


def _reasoning_env(reasoning: Reasoning) -> str:
    """JUDGE_REASONING value of a thinking mode."""
    if isinstance(reasoning, bool):
        return "true" if reasoning else "false"
    return reasoning or "default"


def run_judge(model: str, reasoning: Reasoning, cache_dir: str) -> Dict[str, Any]:
    """
    Measurement A for one model: the `agreement` subcommand with this judge.

    The judge is built from the same environment variables the CLI reads
    (JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_REASONING), passed explicitly instead
    of through os.environ so several candidates can run in parallel. Pairs
    whose judgment failed are retried by running `agreement` again (the
    cache keeps the ones already judged), at most AGREEMENT_ATTEMPTS times.
    """
    from app.evaluation import relevance_judge
    from app.infrastructure.config.settings import JudgeSettings
    from app.infrastructure.llm.relevance_judge import LangChainRelevanceJudge

    env = {"JUDGE_PROVIDER": "openrouter", "JUDGE_MODEL": model, "JUDGE_REASONING": _reasoning_env(reasoning)}
    settings = JudgeSettings.from_env(env)
    inner = LangChainRelevanceJudge.from_settings(settings)
    judge = CountingJudge(inner)
    record_args = ["agreement", "--sample", str(AGREEMENT_SAMPLE), "--seed", str(AGREEMENT_SEED)]
    document = None
    attempts = 0
    with tempfile.TemporaryDirectory() as tmp:
        output_json = str(Path(tmp) / "judge_agreement.json")
        argv = record_args + ["--cache-dir", cache_dir, "--output-json", output_json,
                              "--output-md", str(Path(tmp) / "JUDGE_AGREEMENT.md")]
        args = relevance_judge.build_parser().parse_args(argv)
        for attempts in range(1, AGREEMENT_ATTEMPTS + 1):
            try:
                relevance_judge.run_agreement(args, record_args, build_judge=lambda: (settings, judge))
            except RuntimeError as error:  # several failures in a row
                print(f"[A {model}] attempt {attempts}: {error}", flush=True)
            if Path(output_json).exists():
                document = json.loads(Path(output_json).read_text(encoding="utf-8"))
                break
    if document is not None:
        document["run"]["command"] = (
            " ".join(f"{k}={v}" for k, v in env.items())
            + " python -m app.evaluation.relevance_judge " + " ".join(record_args)
        )
    return {
        "agreement": document,
        "cache_file": str(Path(cache_dir) / cache_file_name(model, inner.prompt_version)),
        "calls": judge.calls,
        "failures": judge.failures,
        "errors": dict(judge.errors),
        "attempts": attempts,
        "usage": inner.usage_recorder.summary()["total"] if inner.usage_recorder else None,
    }


def local_judge_grades(results: Mapping[str, Any], cache_dir: str) -> Dict[str, int]:
    """Grades of the earlier local judge on the agreement sample, from its cache file."""
    from app.evaluation.relevance_judge import JudgmentCache
    from app.infrastructure.llm.prompts_judge import RELEVANCE_JUDGE_PROMPT_VERSION

    cache = JudgmentCache.for_judge(cache_dir, LOCAL_JUDGE_MODEL, RELEVANCE_JUDGE_PROMPT_VERSION)
    grades: Dict[str, int] = {}
    for result in results.get("judge", {}).values():
        for pair in (result.get("agreement") or {}).get("pairs", []):
            judgment = cache.get((pair["query_id"], pair["source"], pair["source_id"]))
            if judgment is not None:
                grades[pair_key(pair)] = judgment.grade
    return grades


def fetch_model_catalog() -> Dict[str, Dict[str, Any]]:
    """OpenRouter's public model list (no API key, no identifying header)."""
    import httpx

    response = httpx.get(OPENROUTER_MODELS_URL, timeout=60)
    response.raise_for_status()
    return {entry["id"]: entry for entry in response.json()["data"]}


def key_usage_usd() -> Optional[float]:
    """Cumulative spend of the API key in USD (OpenRouter /key), or None if unavailable."""
    import httpx

    from app.infrastructure.config.settings import openrouter_api_key

    try:
        response = httpx.get(OPENROUTER_KEY_URL, timeout=30,
                             headers={"Authorization": f"Bearer {openrouter_api_key()}"})
        response.raise_for_status()
        return float(response.json()["data"]["usage"])
    except Exception as error:  # informative only
        logger.warning("Could not read the key usage: %s", type(error).__name__)
        return None


def _parse_models(values: Optional[Sequence[str]]) -> Dict[str, Reasoning]:
    """--models id[=reasoning] ...; default: the V10 candidates."""
    if not values:
        return dict(CANDIDATES)
    from app.infrastructure.config.settings import parse_reasoning

    models: Dict[str, Reasoning] = {}
    for value in values:
        model, _, reasoning = value.partition("=")
        models[model] = parse_reasoning(reasoning) if reasoning else CANDIDATES.get(model, False)
    return models


def _load(path: str) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _save(results: Mapping[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render(results: Dict[str, Any], output_json: str, output_md: str) -> Dict[str, Any]:
    """Recompute the tables and the decision, save them with the raw results and write the report."""
    tables = compute_tables(results)
    results["tables"] = tables
    results["spend"] = total_spend(results)
    _save(results, output_json)
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text(build_markdown(results, tables), encoding="utf-8")
    return tables


def _spent(results: Mapping[str, Any]) -> float:
    return total_spend(results)["total_usd"] or 0.0


def run(args: argparse.Namespace, raw_args: Sequence[str]) -> int:
    """`run` subcommand: measure A, B and C and render the report."""
    from dotenv import load_dotenv

    from app.evaluation.evaluation_job import load_test_queries
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

    load_dotenv(Path(".env"))
    roles = args.roles or list(ROLES)
    results = _load(args.output_json)
    models = _parse_models(args.models)

    catalog_info = check_candidates(models, fetch_model_catalog())
    results.setdefault("candidates", {}).update(catalog_info)
    measured = [model for model in models if catalog_info[model]["dropped"] is None]
    for model in models:
        if model not in measured:
            print(f"Dropping {model}: {catalog_info[model]['dropped']}", flush=True)

    started = datetime.now(UTC)
    results["run"] = {
        "date": started.isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_worktree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "command": " ".join(["python -m app.evaluation.llm_model_selection", *raw_args]),
        "key_usage_usd": {"before": key_usage_usd(), "after": None},
    }

    books = SqliteBookCatalogRepository(Path(args.db_path)).get_all()
    categories = sorted({category for book in books for category in book.categories})
    queries = load_test_queries(args.queries_path)

    if "judge" in roles:
        with ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT) as pool:
            futures = {model: pool.submit(run_judge, model, models[model], args.cache_dir) for model in measured}
            for model, future in futures.items():
                results.setdefault("judge", {})[model] = future.result()
        results["local_judge"] = {"model": LOCAL_JUDGE_MODEL,
                                  "grades": local_judge_grades(results, args.cache_dir)}
        _save(results, args.output_json)

    if "explanation" in roles or "query_understanding" in roles:
        tasks = production_top_results(explanation_queries(queries), books, args.db_path, args.indexes_dir)
        results["explanation_tasks"] = [task.to_dict() for task in tasks]
        for model in measured:
            if _spent(results) > SPEND_LIMIT_USD:
                print(f"Stopping: spend {_spent(results):.4f} USD is above {SPEND_LIMIT_USD} USD", flush=True)
                break
            if "explanation" in roles:
                results.setdefault("explanation", {})[model] = run_explanations(
                    model, models[model], tasks, categories)
                _save(results, args.output_json)
            if "query_understanding" in roles:
                results.setdefault("query_understanding", {})[model] = run_query_understanding(
                    model, models[model], queries, categories)
                _save(results, args.output_json)

    results["run"]["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    results["run"]["key_usage_usd"]["after"] = key_usage_usd()
    tables = render(results, args.output_json, args.output_md)
    _print_decision(tables)
    print(f"Saved {args.output_json} and {args.output_md}")
    return 0


def _print_decision(tables: Mapping[str, Any]) -> None:
    decision = tables["decision"]
    print(f"V10 rule (history) - system LLM: {decision['system_llm']['winner']} "
          f"({decision['system_llm']['decided_by']})")
    print(f"V10 rule (history) - judge: {decision['judge']['winner']} ({decision['judge']['decided_by']})")
    current = tables.get("decision_speed_quality")
    if current:
        print(f"Speed-quality rule ({current['date']}) - system LLM: {current['system_llm']['winner']} "
              f"({current['system_llm']['decided_by']})")
        print(f"Speed-quality rule ({current['date']}) - judge: {current['judge']['winner']} "
              f"({current['judge']['decided_by']})")


def report(args: argparse.Namespace) -> int:
    """`report` subcommand: re-render from the saved raw results (no model call)."""
    results = _load(args.output_json)
    if not results:
        print(f"No saved results in {args.output_json}; run the `run` subcommand first.")
        return 1
    tables = render(results, args.output_json, args.output_md)
    _print_decision(tables)
    print(f"Saved {args.output_json} and {args.output_md}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Choose the system LLM and the judge on the project's tasks")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_parser = subcommands.add_parser("run", help="Measure A, B and C (calls the models)")
    run_parser.add_argument("--models", nargs="+", default=None,
                            help="Model ids, optionally id=reasoning (e.g. z-ai/glm-5.3=low); "
                                 "default: the V10 candidates")
    run_parser.add_argument("--roles", nargs="+", choices=ROLES, default=None, help="Default: all three")
    run_parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    run_parser.add_argument("--indexes-dir", default=DEFAULT_INDEXES_DIR)
    run_parser.add_argument("--queries-path", default=DEFAULT_QUERIES_PATH)
    run_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    report_parser = subcommands.add_parser("report", help="Re-render the report from saved results")
    for sub in (run_parser, report_parser):
        sub.add_argument("--output-json", default=DEFAULT_RESULTS_JSON)
        sub.add_argument("--output-md", default=DEFAULT_REPORT_MD)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.command == "run":
        return run(args, raw_args)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
