"""
Pure analysis of the known-item runs (no network, no database).

- per-query metrics: MRR@10, success@1 and success@10 for T and TA (one
  relevant work); nDCG@10 with binary relevance and success@10 for A (every
  work of the author is relevant), with the metric functions of
  app/evaluation/evaluation_service.py;
- the choice of the lexical weight w on the development half;
- the six declared contrasts on the test half (paired randomisation test,
  paired bootstrap interval, Holm over the six), with the statistics module
  the catalog evaluation uses and its resamples and seed;
- the protocol's decision rule per query type.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from app.evaluation.evaluation_service import ndcg_at_k, reciprocal_rank
from app.evaluation.known_item import design as D
from app.evaluation.statistics import bootstrap_ci, holm_correction, paired_bootstrap_ci, paired_permutation_test

Query = Mapping[str, Any]
Scores = Mapping[str, Mapping[str, Mapping[str, float]]]  # system -> query_id -> metric -> value


def query_metrics(ids: Sequence[str], query: Query) -> Dict[str, float]:
    """The metrics of one ranked list for one query (those of its type)."""
    grades = {book_id: 1 for book_id in query["relevant"]}
    top = list(ids[: D.CUTOFF])
    hit_at_10 = float(any(book_id in grades for book_id in top))
    if query["query_type"] == D.A:
        return {"ndcg_at_10": ndcg_at_k(top, grades, D.CUTOFF), "success_at_10": hit_at_10}
    return {"mrr_at_10": reciprocal_rank(top, grades),
            "success_at_1": float(bool(top) and top[0] in grades), "success_at_10": hit_at_10}


def per_query(queries: Sequence[Query], runs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{system: {query_id: metrics}} for the queries a system has a run line for."""
    return {system: {q["query_id"]: query_metrics(by_query[q["query_id"]]["ids"], q)
                     for q in queries if q["query_id"] in by_query}
            for system, by_query in runs.items()}


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def objective(scores: Mapping[str, Mapping[str, float]], queries: Sequence[Query]) -> Optional[float]:
    """
    The selection objective: the mean, over the three query types, of the
    mean primary metric of that type (MRR@10 for T and TA, nDCG@10 for A).
    """
    parts = []
    for query_type in D.QUERY_TYPES:
        metric = D.PRIMARY_METRIC[query_type]
        values = [scores[q["query_id"]][metric] for q in queries
                  if q["query_type"] == query_type and q["query_id"] in scores]
        if not values:
            return None
        parts.append(float(np.mean(values)))
    return float(np.mean(parts))


def select_weight(scores: Scores, dev_queries: Sequence[Query],
                  grid: Sequence[float] = D.WEIGHT_GRID) -> Dict[str, Any]:
    """
    The lexical weight of the weighted fusion: the w of the grid whose run
    (S3w@w, without reranker) maximises the objective on the development
    half. A tie goes to the smallest w.
    """
    values = {w: objective(scores[D.grid_name(w)], dev_queries) for w in grid}
    if any(v is None for v in values.values()):
        raise ValueError("every weight of the grid needs runs for every development query type")
    best = max(grid, key=lambda w: (values[w], -w))
    return {"grid": list(grid), "objective": {D.grid_name(w): values[w] for w in grid}, "chosen_weight": best,
            "dev_queries": len(dev_queries)}


def summarize(scores: Scores, queries: Sequence[Query], systems: Sequence[str]) -> Dict[str, Any]:
    """Per query type and system: queries and every metric's mean with a 95 % bootstrap interval."""
    out: Dict[str, Any] = {}
    for query_type in D.QUERY_TYPES:
        ids = [q["query_id"] for q in queries if q["query_type"] == query_type]
        out[query_type] = {}
        for system in systems:
            done = [i for i in ids if i in scores.get(system, {})]
            if not done:
                continue
            row: Dict[str, Any] = {"queries": len(done)}
            for metric in D.METRICS[query_type]:
                mean, low, high = bootstrap_ci([scores[system][i][metric] for i in done],
                                               n_resamples=D.N_RESAMPLES, seed=D.STAT_SEED)
                row[metric] = {"mean": mean, "ci": [low, high]}
            out[query_type][system] = row
    return out


def _paired(scores: Scores, contrast: D.Contrast, queries: Sequence[Query]) -> Dict[str, Any]:
    metric = D.PRIMARY_METRIC[contrast.query_type]
    ids = [q["query_id"] for q in queries if q["query_type"] == contrast.query_type
           and q["query_id"] in scores.get(contrast.system, {}) and q["query_id"] in scores.get(contrast.baseline, {})]
    a = [scores[contrast.system][i][metric] for i in ids]
    b = [scores[contrast.baseline][i][metric] for i in ids]
    diff, p = paired_permutation_test(a, b, n_resamples=D.N_RESAMPLES, seed=D.STAT_SEED)
    _, low, high = paired_bootstrap_ci(a, b, n_resamples=D.N_RESAMPLES, seed=D.STAT_SEED)
    return {"query_type": contrast.query_type, "system": contrast.system, "baseline": contrast.baseline,
            "metric": metric, "queries": len(ids), "difference": diff, "ci": [low, high], "p": p}


def contrasts(scores: Scores, test_queries: Sequence[Query],
              family: Sequence[D.Contrast] = D.CONTRASTS) -> Dict[str, Any]:
    """The declared family on the test half, with Holm-adjusted p-values over the six contrasts."""
    out = {c.name: _paired(scores, c, test_queries) for c in family}
    adjusted = holm_correction({name: row["p"] for name, row in out.items()})
    for name, row in out.items():
        row["p_holm"] = adjusted[name]
        row["significant"] = adjusted[name] < D.ALPHA
    return out


def secondary_contrasts(scores: Scores, test_queries: Sequence[Query],
                        family: Sequence[D.Contrast] = D.CONTRASTS) -> Dict[str, Any]:
    """The same differences without Chinese and Japanese: outside the family, raw p-values, descriptive."""
    kept = [q for q in test_queries if q["language"] not in D.SECONDARY_DROP_LANGUAGES]
    return {c.name: _paired(scores, c, kept) for c in family}


def decision(family: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """
    The protocol's rule (section 8), per query type: lexical candidates help
    for that type when a fusion (C1: S5, C2: S5w) is significantly better than
    vector only with reranker (S4) after Holm; otherwise the conclusion of V14
    extends to that type on this catalog.
    """
    per_type: Dict[str, Any] = {}
    for query_type in D.QUERY_TYPES:
        rows = [row for row in family.values() if row["query_type"] == query_type]
        helps = [row["system"] for row in rows if row["significant"] and row["difference"] > 0]
        per_type[query_type] = {
            "fusion_significantly_better": helps,
            "outcome": ("limit the conclusion to descriptive queries: for this query type the lexical candidates "
                        "help" if helps else "the conclusion extends to this query type on this catalog"),
        }
    return {"per_type": per_type,
            "lexical_helps_for": [qt for qt, row in per_type.items() if row["fusion_significantly_better"]]}


def by_language(scores: Scores, queries: Sequence[Query], systems: Sequence[str]) -> Dict[str, Any]:
    """Descriptive: mean primary metric per query type, language and system."""
    out: Dict[str, Any] = {}
    for query_type in D.QUERY_TYPES:
        metric = D.PRIMARY_METRIC[query_type]
        out[query_type] = {}
        for language in D.LANGUAGES:
            ids = [q["query_id"] for q in queries if q["query_type"] == query_type and q["language"] == language]
            out[query_type][language] = {s: _mean([scores[s][i][metric] for i in ids if i in scores.get(s, {})])
                                         for s in systems}
    return out


def split(queries: Sequence[Query], half: str) -> List[Query]:
    return [q for q in queries if q["half"] == half]
