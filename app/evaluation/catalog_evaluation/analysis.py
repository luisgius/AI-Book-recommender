"""
Pure analysis of the saved runs and judgments (no network, no database).

- per-query metrics of a run under a judge's grades (nDCG@10, P@10,
  MRR@10, Recall@100 relative to the pool, Judged@10), with the metric
  definitions of app/evaluation/evaluation_service.py;
- system summaries with bootstrap intervals and the declared contrasts
  H1-H6 (paired randomisation test, paired bootstrap, Holm), with the
  statistics module of V03 and its seeds;
- the protocol's rule for a sustained conclusion (same sign and significant
  under both judges; otherwise not conclusive);
- agreement between the two judges; overlaps; the stratum C analysis;
- the embedding decision rule (section 8), the unanswerable-query analysis
  (section 9).
"""

import math
import statistics as st
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.queries import EvalQuery
from app.evaluation.evaluation_service import (
    judged_fraction_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.statistics import bootstrap_ci, holm_correction, paired_bootstrap_ci, paired_permutation_test

METRICS = ("ndcg_at_10", "p_at_10", "mrr_at_10", "recall_at_100", "judged_at_10")
Grades = Mapping[str, Mapping[str, int]]  # need_id -> work id -> grade


def query_metrics(ids: Sequence[str], grades: Mapping[str, int]) -> Dict[str, float]:
    """Metrics of one ranked list against the graded pool of its need."""
    return {
        "ndcg_at_10": ndcg_at_k(ids, grades, D.CUTOFF),
        "p_at_10": precision_at_k(ids, grades, D.CUTOFF),
        "mrr_at_10": reciprocal_rank(list(ids[: D.CUTOFF]), grades),
        "recall_at_100": recall_at_k(ids, grades, D.DEPTH),
        "judged_at_10": judged_fraction_at_k(ids, grades, D.CUTOFF),
    }


def per_query(queries: Sequence[EvalQuery], runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
              grades: Grades, systems: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{system: {query_id: metrics}} for the queries every system has a run line for."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for system in systems:
        out[system] = {}
        for query in queries:
            record = runs.get(system, {}).get(query.query_id)
            if record is None:
                continue
            out[system][query.query_id] = query_metrics(record["ids"], grades.get(query.need_id, {}))
    return out


def summarize(scores: Mapping[str, Mapping[str, Mapping[str, float]]], query_ids: Sequence[str]) -> Dict[str, Any]:
    """Mean of every metric with a 95 % bootstrap interval, per system, over query_ids."""
    out: Dict[str, Any] = {}
    for system, by_query in scores.items():
        ids = [q for q in query_ids if q in by_query]
        if not ids:
            continue
        row: Dict[str, Any] = {"queries": len(ids)}
        for metric in METRICS:
            values = [by_query[q][metric] for q in ids]
            mean, low, high = bootstrap_ci(values, n_resamples=D.N_RESAMPLES, seed=D.SEED)
            row[metric] = {"mean": mean, "ci": [low, high]}
        out[system] = row
    return out


def contrasts(scores: Mapping[str, Mapping[str, Mapping[str, float]]], query_ids: Sequence[str],
              family: Sequence[D.Contrast] = D.CONTRASTS, extra: Sequence[D.Contrast] = (D.REPLICATION,),
              metric: str = "ndcg_at_10") -> Dict[str, Any]:
    """The declared family with Holm-adjusted p-values; `extra` contrasts are tested but kept outside the family."""
    out: Dict[str, Any] = {}
    raw: Dict[str, float] = {}
    for contrast in list(family) + list(extra):
        ids = [q for q in query_ids if q in scores.get(contrast.system, {}) and q in scores.get(contrast.baseline, {})]
        a = [scores[contrast.system][q][metric] for q in ids]
        b = [scores[contrast.baseline][q][metric] for q in ids]
        diff, p = paired_permutation_test(a, b, n_resamples=D.N_RESAMPLES, seed=D.SEED)
        _, low, high = paired_bootstrap_ci(a, b, n_resamples=D.N_RESAMPLES, seed=D.SEED)
        out[contrast.name] = {"system": contrast.system, "baseline": contrast.baseline, "question": contrast.question,
                              "queries": len(ids), "difference": diff, "ci": [low, high], "p": p,
                              "in_family": contrast in family}
        if contrast in family:
            raw[contrast.name] = p
    adjusted = holm_correction(raw)
    for name, row in out.items():
        row["p_holm"] = adjusted.get(name)
    return out


def conclusion(primary: Mapping[str, Any], second: Mapping[str, Any], alpha: float = D.ALPHA) -> str:
    """
    The protocol's rule for one contrast (section 6): "sustained" when the
    difference has the same sign and is significant (Holm) under both
    judges; "not significant" when neither judge finds it significant;
    otherwise "not conclusive" (the judges disagree).
    """
    def significant(row: Mapping[str, Any]) -> bool:
        p = row.get("p_holm") if row.get("p_holm") is not None else row["p"]
        return p < alpha

    sig_a, sig_b = significant(primary), significant(second)
    same_sign = math.copysign(1, primary["difference"]) == math.copysign(1, second["difference"])
    if sig_a and sig_b and same_sign and primary["difference"] != 0:
        return "sustained"
    if not sig_a and not sig_b:
        return "not significant"
    return "not conclusive"


# =============================================================================
# Judges
# =============================================================================


def judge_agreement(first: Grades, second: Grades, languages: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Agreement of the two judges on the pairs both graded (quadratic kappa with CI, exact, mean difference)."""
    from app.evaluation.relevance_judge import agreement_report

    a, b, langs = [], [], []
    for need_id, grades in first.items():
        for work_id, grade in grades.items():
            other = second.get(need_id, {}).get(work_id)
            if other is None:
                continue
            a.append(grade)
            b.append(other)
            langs.append((languages or {}).get(need_id))
    if not a:
        return {"pairs": 0}
    report = agreement_report(a, b, langs if languages else None, n_resamples=2000, seed=D.SEED)
    report["pairs"] = len(a)
    return report


# =============================================================================
# Overlaps and stratum C
# =============================================================================


def overlap_at(a: Sequence[str], b: Sequence[str], k: int) -> float:
    return len(set(a[:k]) & set(b[:k])) / k if k else 0.0


def mean_overlap(runs: Mapping[str, Mapping[str, Mapping[str, Any]]], first: str, second: str,
                 query_ids: Sequence[str], k: int) -> Optional[float]:
    values = [overlap_at(runs[first][q]["ids"], runs[second][q]["ids"], k)
              for q in query_ids if q in runs.get(first, {}) and q in runs.get(second, {})]
    return float(np.mean(values)) if values else None


def stratum_c(queries: Sequence[EvalQuery], runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
              scores: Mapping[str, Mapping[str, Mapping[str, float]]], systems: Sequence[str]) -> Dict[str, Any]:
    """nDCG@10 per language and system, and the mean pairwise top-10 overlap of the four versions of each need."""
    c = [q for q in queries if q.stratum == D.STRATUM_C]
    by_need: Dict[str, List[EvalQuery]] = defaultdict(list)
    for q in c:
        by_need[q.need_id].append(q)
    ndcg = {system: {lang: _mean([scores[system][q.query_id]["ndcg_at_10"] for q in c
                                  if q.language == lang and q.query_id in scores.get(system, {})])
                     for lang in D.C_LANGUAGES} for system in systems}
    overlap = {}
    for system in systems:
        values = []
        for versions in by_need.values():
            for x, y in combinations(versions, 2):
                if x.query_id in runs.get(system, {}) and y.query_id in runs.get(system, {}):
                    values.append(overlap_at(runs[system][x.query_id]["ids"], runs[system][y.query_id]["ids"], D.CUTOFF))
        overlap[system] = _mean(values)
    return {"needs": len(by_need), "ndcg_at_10_by_language": ndcg, "mean_pairwise_top10_overlap": overlap}


def breakdown(queries: Sequence[EvalQuery], scores: Mapping[str, Mapping[str, Mapping[str, float]]],
              systems: Sequence[str], attribute: str, metric: str = "ndcg_at_10") -> Dict[str, Any]:
    """Mean metric per value of a query attribute (language or type), per system (descriptive)."""
    values = sorted({getattr(q, attribute) for q in queries})
    out: Dict[str, Any] = {}
    for value in values:
        ids = [q.query_id for q in queries if getattr(q, attribute) == value]
        out[value] = {"queries": len(ids), **{s: _mean([scores[s][q][metric] for q in ids if q in scores.get(s, {})])
                                                 for s in systems}}
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


# =============================================================================
# Section 8: embedding decision rule
# =============================================================================


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def operational_summary(calls: Sequence[Mapping[str, Any]], top10: Mapping[str, Sequence[Optional[Sequence[str]]]]) -> Dict[str, Any]:
    """
    Operational criteria of one model from its repeated query encodings:
    median and p95 latency of the successful calls, failed or timed-out
    calls, and queries whose top 10 changes between repetitions (counted
    over queries with at least two successful repetitions).
    """
    latencies = [c["latency_ms"] for c in calls if not c.get("error")]
    failed = sum(1 for c in calls if c.get("error"))
    changed = compared = 0
    for lists in top10.values():
        ok = [tuple(x) for x in lists if x is not None]
        if len(ok) < 2:
            continue
        compared += 1
        if len(set(ok)) > 1:
            changed += 1
    return {"calls": len(calls), "failed_calls": failed, "median_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95), "queries_compared": compared, "queries_top10_changed": changed}


def _better(candidate: float, current: float) -> bool:
    return candidate < current


def _worse_by_more_than(candidate: float, current: float, share: float = 0.10) -> bool:
    if candidate <= current:
        return False
    return candidate > current * (1 + share)


def decision_rule(h6_primary: Mapping[str, Any], h6_second: Mapping[str, Any],
                  bge: Mapping[str, Any], qwen: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Protocol section 8, applied mechanically. Switch to baai/bge-m3 only if
    (1) H6 (E2 - E1) does not show qwen significantly better (Holm p < 0.05
    with a negative difference) under either judge, and (2) bge-m3 is better
    on at least two of the three operational criteria and not more than 10 %
    worse on the third.

    Criteria: latency (better = lower median and lower p95; worse by more
    than 10 % = median or p95 more than 10 % higher), failed calls (count),
    queries whose top 10 changes between repetitions (count). A count that
    is 0 for qwen is "more than 10 % worse" for any positive bge count.
    """
    def qwen_better(row: Mapping[str, Any]) -> bool:
        p = row["p_holm"] if row.get("p_holm") is not None else row["p"]
        return row["difference"] < 0 and p < D.ALPHA

    condition_1 = not qwen_better(h6_primary) and not qwen_better(h6_second)
    criteria = {
        "latency": {
            "better": _better(bge["median_ms"], qwen["median_ms"]) and _better(bge["p95_ms"], qwen["p95_ms"]),
            "worse_over_10pct": _worse_by_more_than(bge["median_ms"], qwen["median_ms"])
            or _worse_by_more_than(bge["p95_ms"], qwen["p95_ms"]),
        },
        "failed_calls": {"better": _better(bge["failed_calls"], qwen["failed_calls"]),
                         "worse_over_10pct": _worse_by_more_than(bge["failed_calls"], qwen["failed_calls"])},
        "top10_changes": {"better": _better(bge["queries_top10_changed"], qwen["queries_top10_changed"]),
                          "worse_over_10pct": _worse_by_more_than(bge["queries_top10_changed"], qwen["queries_top10_changed"])},
    }
    better = [name for name, row in criteria.items() if row["better"]]
    others_ok = all(not row["worse_over_10pct"] for name, row in criteria.items() if name not in better)
    condition_2 = len(better) >= 2 and others_ok
    return {"condition_1_h6": condition_1, "criteria": criteria, "bge_better_on": better,
            "condition_2_operational": condition_2,
            "outcome": "propose switching to baai/bge-m3" if condition_1 and condition_2 else "keep qwen/qwen3-embedding-8b"}


# =============================================================================
# Section 9: unanswerable queries
# =============================================================================


def auc(positives: Sequence[float], negatives: Sequence[float]) -> Optional[float]:
    """Area under the ROC curve: P(score of a positive > score of a negative), ties count one half."""
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def best_threshold(positives: Sequence[float], negatives: Sequence[float]) -> Optional[Dict[str, Any]]:
    """
    The threshold that best separates answered (score >= t) from
    unanswerable (score < t) queries by Youden's J (TPR - FPR); ties go to
    the higher accuracy, then the lower threshold. Chosen and measured on
    the same data: exploratory only.
    """
    if not positives or not negatives:
        return None
    best = None
    for t in sorted(set(positives) | set(negatives)):
        tpr = sum(p >= t for p in positives) / len(positives)
        fpr = sum(n >= t for n in negatives) / len(negatives)
        accuracy = (sum(p >= t for p in positives) + sum(n < t for n in negatives)) / (len(positives) + len(negatives))
        key = (tpr - fpr, accuracy)
        if best is None or key > best[0]:
            best = (key, {"threshold": t, "youden_j": tpr - fpr, "tpr": tpr, "fpr": fpr, "accuracy": accuracy})
    return best[1]


def describe(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "median": st.median(values), "mean": float(np.mean(values)),
            "max": max(values)}


def hybrid_rule(h3_primary: Mapping[str, Any], h3_second: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Protocol section 8, rule for the hybrid fusion: the deployed pipeline
    keeps BM25 candidates (its degradation path) even when H1 and H3 are not
    significant; it would change only if S4 (vector + reranker) beat S5
    (hybrid + reranker) significantly, i.e. H3 = S5 - S4 is negative and
    sustained under both judges.
    """
    s4_better = conclusion(h3_primary, h3_second) == "sustained" and h3_primary["difference"] < 0
    return {"s4_significantly_better_than_s5": s4_better,
            "outcome": ("the rule's condition for changing is met: S4 (vector candidates + reranker, without BM25) "
                        "beats the deployed S5 under both judges" if s4_better
                        else "keep the deployed hybrid S5")}
