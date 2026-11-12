"""
Build the results document from the saved files only (no API call, no
database connection): queries, run files, pool, judgment caches,
operational records, explanations, the parity check and the spend ledger.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from app.evaluation.catalog_evaluation import analysis as A
from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.ledger import LEDGER_FILE, SpendLedger, summarize_ledger
from app.evaluation.catalog_evaluation.queries import EvalQuery, load_queries
from app.evaluation.catalog_evaluation.runs import JsonlStore, load_runs, model_slug

POOL_FILE = "pool.json"
JUDGING_RUNS = "judging_runs.jsonl"


def _read_json(path: Path) -> Optional[Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_grades(data_dir: Path, model: str, pool: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    """{need_id: {work_id: grade}} of one judge, restricted to the pool."""
    from app.evaluation.catalog_evaluation.judging import judge_cache, grades_by_need

    source_ids = {}
    for need_id, works in pool.items():
        for work in works:
            source_ids[(work["source"], work["source_id"])] = work["work_id"]
    grades = grades_by_need(judge_cache(data_dir, model), source_ids)
    allowed = {need_id: {w["work_id"] for w in works} for need_id, works in pool.items()}
    return {need: {w: g for w, g in by.items() if w in allowed.get(need, set())} for need, by in grades.items()}


def _latency(runs: Mapping[str, Mapping[str, Mapping[str, Any]]], query_ids: Sequence[str]) -> Dict[str, Any]:
    out = {}
    for system, records in runs.items():
        stages: Dict[str, List[float]] = defaultdict(list)
        totals = []
        for qid in query_ids:
            record = records.get(qid)
            if not record:
                continue
            timings = record.get("timings_ms") or {}
            for stage, value in timings.items():
                stages[stage].append(value)
            totals.append(sum(timings.values()))
        out[system] = {"median_ms": {s: float(np.median(v)) for s, v in stages.items()},
                       "total_median_ms": float(np.median(totals)) if totals else None,
                       "total_p95_ms": float(np.percentile(totals, 95)) if totals else None}
    return out


def _run_failures(runs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    out = {}
    for system, records in runs.items():
        errors = [r["query_id"] for r in records.values() if r.get("error")]
        extra = Counter()
        for r in records.values():
            e = r.get("extra") or {}
            if e.get("failed_attempts"):
                extra["calls_retried"] += 1
            if e.get("answer_failed"):
                extra["llm_answer_unusable"] += 1
            if e.get("unknown_ids"):
                extra["llm_unknown_ids"] += e["unknown_ids"]
            if e.get("duplicate_ids"):
                extra["llm_duplicate_ids"] += e["duplicate_ids"]
            if e.get("filled") is not None and e.get("filled"):
                extra["llm_lists_completed"] += 1
        out[system] = {"queries": len(records), "fallbacks": errors, **dict(extra)}
    return out


def _s6_replication(runs, scores, ab_ids) -> Dict[str, Any]:
    same = sum(1 for q in ab_ids if q in runs.get(D.S6, {}) and q in runs.get(D.S6B, {})
               and runs[D.S6][q]["ids"][: D.CUTOFF] == runs[D.S6B][q]["ids"][: D.CUTOFF])
    diffs = [abs(scores[D.S6][q]["ndcg_at_10"] - scores[D.S6B][q]["ndcg_at_10"])
             for q in ab_ids if q in scores.get(D.S6, {}) and q in scores.get(D.S6B, {})]
    return {"identical_top10": same, "queries": len(diffs),
            "mean_abs_ndcg_difference": float(np.mean(diffs)) if diffs else None}


def _operational(data_dir: Path, queries: Sequence[EvalQuery]) -> Optional[Dict[str, Any]]:
    out = {}
    for model in (D.QWEN_EMBEDDING, D.BGE_EMBEDDING):
        records = list(JsonlStore(data_dir / "operational" / f"{model_slug(model)}.jsonl", key="key").records.values())
        if not records:
            return None
        top10: Dict[str, List] = defaultdict(lambda: [None] * D.OPERATIONAL_REPETITIONS)
        for r in records:
            top10[r["query_id"]][r["repetition"] - 1] = r.get("top10")
        summary = A.operational_summary(records, top10)
        vectors_changed = 0
        by_query = defaultdict(list)
        for r in records:
            if r.get("vector"):
                by_query[r["query_id"]].append(r["vector"])
        vectors_changed = sum(1 for v in by_query.values() if len(set(v)) > 1)
        summary["queries_vector_changed"] = vectors_changed
        summary["errors"] = dict(Counter((r.get("error") or "").split(":")[0] for r in records if r.get("error")))
        summary["complete"] = len(records) == len(queries) * D.OPERATIONAL_REPETITIONS
        out[model] = summary
    return out


def _unanswerable(queries, runs, primary, second) -> Dict[str, Any]:
    d_queries = [q for q in queries if q.stratum == D.STRATUM_D]
    reclassified = {}
    for q in d_queries:
        p = [w for w, g in primary.get(q.need_id, {}).items() if g >= D.ANSWERED_GRADE]
        s = [w for w, g in second.get(q.need_id, {}).items() if g >= D.ANSWERED_GRADE]
        reclassified[q.query_id] = {"primary_works_graded_2_or_3": len(p), "second_works_graded_2_or_3": len(s),
                                    "reclassified": bool(p)}
    kept = [q for q in d_queries if not reclassified[q.query_id]["reclassified"]]
    answered = [q for q in queries if q.stratum in D.STATISTICAL_STRATA
                and any(g >= D.ANSWERED_GRADE for g in primary.get(q.need_id, {}).values())]
    s5 = runs.get(D.S5, {})

    def score(q):
        record = s5.get(q.query_id)
        return (record or {}).get("extra", {}).get("top_rerank_score")

    pos = [score(q) for q in answered if score(q) is not None]
    neg = [score(q) for q in kept if score(q) is not None]
    return {"reclassification": reclassified, "d_kept": [q.query_id for q in kept],
            "answered_ab_queries": len(answered),
            "ab_without_answer": [q.query_id for q in queries if q.stratum in D.STATISTICAL_STRATA and q not in answered],
            "answered_scores": A.describe(pos), "unanswerable_scores": A.describe(neg),
            "unanswerable_score_by_query": {q.query_id: score(q) for q in kept},
            "auc": A.auc(pos, neg), "best_threshold": A.best_threshold(pos, neg)}


def _explanations(data_dir: Path) -> Optional[Dict[str, Any]]:
    from app.evaluation.llm_model_selection import summarize_explanations

    records = _read_jsonl(data_dir / "explanations" / "explanations.jsonl")
    if not records:
        return None
    summary = summarize_explanations(records)
    parsed = [r for r in records if r.get("parsed")]
    in_language = sum(1 for r in parsed if r.get("detected_language") == r["language"])
    undecided = sum(1 for r in parsed if r.get("detected_language") is None)
    by_language = defaultdict(lambda: [0, 0])
    for r in parsed:
        by_language[r["language"]][1] += 1
        by_language[r["language"]][0] += r.get("detected_language") == r["language"]
    return {"records": len(records), "queries": len({r["query_id"] for r in records}), "v10_summary": summary,
            "in_query_language": in_language, "parsed": len(parsed), "language_undecided": undecided,
            "in_query_language_by_language": {k: {"in_language": v[0], "explanations": v[1]} for k, v in by_language.items()},
            "detected_languages": dict(Counter(str(r.get("detected_language")) for r in parsed))}


def build_results(data_dir: Path) -> Dict[str, Any]:
    queries = load_queries(data_dir / "queries" / "queries.json")
    checks = _read_json(data_dir / "queries" / "checks.json") or {}
    runs = load_runs(data_dir, D.SYSTEM_NAMES)
    pool = _read_json(data_dir / POOL_FILE) or {}
    primary = load_grades(data_dir, D.PRIMARY_JUDGE, pool)
    second = load_grades(data_dir, D.SECOND_JUDGE, pool)
    ab = [q for q in queries if q.stratum in D.STATISTICAL_STRATA]
    ab_ids = [q.query_id for q in ab]
    need_language = {q.need_id: q.language for q in queries}

    per_judge = {}
    for label, grades in (("primary", primary), ("second", second)):
        scores = A.per_query(queries, runs, grades, D.SYSTEM_NAMES)
        per_judge[label] = {
            "summary_ab": A.summarize(scores, ab_ids),
            "contrasts": A.contrasts(scores, ab_ids),
            "by_language": A.breakdown(ab, scores, D.SYSTEM_NAMES, "language"),
            "by_type": A.breakdown(ab, scores, D.SYSTEM_NAMES, "type"),
            "stratum_c": A.stratum_c(queries, runs, scores, D.SYSTEM_NAMES),
            "s6_replication": _s6_replication(runs, scores, ab_ids),
            "per_query": scores,
        }
    conclusions = {}
    for name in [c.name for c in D.CONTRASTS] + [D.REPLICATION.name]:
        p_row = per_judge["primary"]["contrasts"].get(name)
        s_row = per_judge["second"]["contrasts"].get(name)
        if p_row and s_row and p_row["queries"] and s_row["queries"]:
            conclusions[name] = A.conclusion(p_row, s_row)

    pool_sizes = [len(v) for v in pool.values()]
    judging_runs = _read_jsonl(data_dir / "judgments" / JUDGING_RUNS)
    operational = _operational(data_dir, queries)
    decision = None
    if operational and "H6" in per_judge["primary"]["contrasts"]:
        decision = A.decision_rule(per_judge["primary"]["contrasts"]["H6"], per_judge["second"]["contrasts"]["H6"],
                                   operational[D.BGE_EMBEDDING], operational[D.QWEN_EMBEDDING])
    hybrid = None
    if "H3" in per_judge["primary"]["contrasts"] and per_judge["primary"]["contrasts"]["H3"]["queries"]:
        hybrid = A.hybrid_rule(per_judge["primary"]["contrasts"]["H3"], per_judge["second"]["contrasts"]["H3"])
    ledger = SpendLedger(data_dir / LEDGER_FILE, D.SPEND_CAP_USD)
    stage_times = _read_jsonl(data_dir / "stage_times.jsonl")
    seconds = defaultdict(float)
    for row in stage_times:
        seconds[row["stage"]] += row["seconds"]

    return {
        "protocol": {"path": D.PROTOCOL_PATH, "commit": D.PROTOCOL_COMMIT},
        "queries": {
            "counts_by_stratum": dict(Counter(q.stratum for q in queries)),
            "counts_by_stratum_language": dict(Counter(f"{q.stratum}/{q.language}" for q in queries)),
            "counts_by_stratum_type": dict(Counter(f"{q.stratum}/{q.type}" for q in queries)),
            "generator": checks.get("generator"), "first_pass_problems": (checks.get("first_pass_checks") or {}).get("problems"),
            "fixes": checks.get("fixes"), "final_problems": (checks.get("final_checks") or {}).get("problems"),
            "language_summary": (checks.get("final_checks") or {}).get("language_summary"),
            "language_undecided": {k: v for k, v in ((checks.get("final_checks") or {}).get("language") or {}).items()
                                   if v["verdict"] == "undecided"},
            "known_item_targets": checks.get("known_item_targets"),
        },
        "parity": _read_json(data_dir / "parity.json"),
        "runs": {"lines": {s: len(r) for s, r in runs.items()}, "failures": _run_failures(runs),
                 "latency_ab": _latency(runs, ab_ids)},
        "pool": {"needs": len(pool), "pairs": sum(pool_sizes),
                 "per_need": A.describe(pool_sizes),
                 "pairs_by_stratum": dict(Counter({q.need_id: q.stratum for q in queries}[n] for n, w in pool.items() for _ in w))},
        "judgments": {
            "primary": {"model": D.PRIMARY_JUDGE, "graded_pairs": sum(len(v) for v in primary.values())},
            "second": {"model": D.SECOND_JUDGE, "graded_pairs": sum(len(v) for v in second.values())},
            "runs": judging_runs,
            "agreement": A.judge_agreement(primary, second, need_language),
            "grade_distribution": {"primary": dict(Counter(g for v in primary.values() for g in v.values())),
                                   "second": dict(Counter(g for v in second.values() for g in v.values()))},
        },
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "per_query"} for k, v in per_judge.items()},
        "per_query": {k: v["per_query"] for k, v in per_judge.items()},
        "conclusions": conclusions,
        "overlap_s2_e1": {"top10": A.mean_overlap(runs, D.S2, D.E1, ab_ids, 10),
                          "top100": A.mean_overlap(runs, D.S2, D.E1, ab_ids, 100),
                          "all_queries_top10": A.mean_overlap(runs, D.S2, D.E1, [q.query_id for q in queries], 10)},
        "operational": operational,
        "decision_rule": decision,
        "hybrid_rule": hybrid,
        "unanswerable": _unanswerable(queries, runs, primary, second) if pool else None,
        "explanations": _explanations(data_dir),
        "annotation": _read_json(data_dir / "annotation" / "summary.json"),
        "human_agreement": _read_json(data_dir / "annotation" / "human_agreement.json"),
        "spend": summarize_ledger(ledger.entries()),
        "stage_seconds": dict(seconds),
    }
