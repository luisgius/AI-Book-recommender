"""
Pure analysis of the saved turns, reference runs and judgments (no network,
no database).

Definitions this module fixes (the protocol does not spell them out; the
report lists them):
- The list of a reply: the search results on the search route (also when the
  agent failed and the graph fell back to search), the agent's
  recommendations in order on the agent route, and nothing on the clarify
  and out-of-scope routes (nDCG@10 = 0: the user got no books in that turn).
- The recommendations of a reply: the verified picks of the search route,
  the agent's recommendations on the agent route.
- Recommendation precision: the share of recommended works with grade 2 or
  3; P's precision at the same number k of top positions, same threshold.
  A request without recommendations has no k and is left out of H9 (counted).
"""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.evaluation.assistant_evaluation import design as D
from app.evaluation.catalog_evaluation.analysis import conclusion
from app.evaluation.evaluation_service import ndcg_at_k
from app.evaluation.statistics import bootstrap_ci, holm_correction, paired_bootstrap_ci, paired_permutation_test

ROUTES = ("clarify", "search", "agent", "out_of_scope")
ERROR_ROUTE = "error"
Grades = Mapping[str, Mapping[str, int]]


# =============================================================================
# Lists and recommendations of a reply
# =============================================================================


def reply_route(reply: Optional[Mapping[str, Any]]) -> str:
    return ERROR_ROUTE if reply is None else str(reply.get("route"))


def reply_list(reply: Optional[Mapping[str, Any]]) -> List[str]:
    """The ranked work ids the user got in this reply (see the module docstring)."""
    if not reply:
        return []
    if reply.get("route") == "agent" and reply.get("agent"):
        return [r["work_id"] for r in reply["agent"]["recommendations"]]
    if reply.get("route") == "search" and reply.get("search"):
        return list(reply["search"]["ids"])
    return []


def reply_picks(reply: Optional[Mapping[str, Any]]) -> List[str]:
    if not reply:
        return []
    return [r["work_id"] for r in reply.get("recommendations") or []]


def last_reply(conversation: Optional[Mapping[str, Any]], turns: Mapping[str, Mapping[str, Any]],
               request_id: str) -> Optional[Dict[str, Any]]:
    """The assistant's last reply of an E conversation (None if it has none)."""
    if not conversation:
        return None
    last = conversation.get("assistant_turns", 0)
    record = turns.get(f"{D.RUN_1}:{request_id}:{last}")
    return (record or {}).get("reply")


def ndcg10(ids: Sequence[str], grades: Mapping[str, int]) -> float:
    return ndcg_at_k(list(ids), dict(grades), D.CUTOFF) if grades else 0.0


def precision_at(ids: Sequence[str], grades: Mapping[str, int], k: int,
                 threshold: int = D.RECOMMENDED_GRADE) -> Optional[float]:
    """Share of the first k works with grade >= threshold (unjudged counts as 0); None when k is 0."""
    if k <= 0:
        return None
    return sum(1 for w in list(ids)[:k] if grades.get(w, 0) >= threshold) / k


def judged_share(ids: Sequence[str], grades: Mapping[str, int]) -> Optional[float]:
    ids = list(ids)[: D.CUTOFF]
    return sum(1 for w in ids if w in grades) / len(ids) if ids else None


# =============================================================================
# Routes
# =============================================================================


def route_group(request: Any) -> str:
    """Row of the route matrix: A is split into its ambiguous queries and the rest."""
    if request.set == D.SET_A:
        return "A ambiguous" if request.category == D.AMBIGUOUS_CATEGORY else "A other"
    return request.set


def route_matrix(rows: Iterable[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    """{group: {route: count}} from (group, route) pairs."""
    out: Dict[str, Counter] = defaultdict(Counter)
    for group, route in rows:
        out[group][route] += 1
    return {g: {r: c[r] for r in list(ROUTES) + [ERROR_ROUTE] if c[r]} | {"total": sum(c.values())}
            for g, c in out.items()}


def rate(routes: Sequence[str], route: str) -> Dict[str, Any]:
    n = len(routes)
    k = sum(1 for r in routes if r == route)
    return {"count": k, "requests": n, "rate": k / n if n else None}


def route_rates(routes_ab: Mapping[str, str], sets_of: Mapping[str, str], routes_d: Sequence[str]) -> Dict[str, Any]:
    """The protocol's route rates of one run: questions and out of scope in A+B and in B, detection in D."""
    ab = list(routes_ab.values())
    b = [route for rid, route in routes_ab.items() if sets_of[rid] == D.SET_B]
    a = [route for rid, route in routes_ab.items() if sets_of[rid] == D.SET_A]
    return {"clarify_ab": rate(ab, "clarify"), "clarify_b": rate(b, "clarify"), "clarify_a": rate(a, "clarify"),
            "out_of_scope_ab": rate(ab, "out_of_scope"), "out_of_scope_b": rate(b, "out_of_scope"),
            "agent_ab": rate(ab, "agent"), "d_detection": rate(list(routes_d), "out_of_scope")}


def thresholds(rates: Mapping[str, Any]) -> Dict[str, Any]:
    """Section 7, applied mechanically: both must hold before the assistant is the main entry point."""
    oos = rates["out_of_scope_ab"]
    clarify = rates["clarify_b"]
    rows = {
        "out_of_scope_ab": {"value": oos["rate"], "count": oos["count"], "requests": oos["requests"],
                            "maximum": D.OOS_AB_MAX, "met": oos["rate"] is not None and oos["rate"] <= D.OOS_AB_MAX},
        "clarify_b": {"value": clarify["rate"], "count": clarify["count"], "requests": clarify["requests"],
                      "maximum": D.CLARIFY_B_MAX,
                      "met": clarify["rate"] is not None and clarify["rate"] <= D.CLARIFY_B_MAX},
    }
    return {"rows": rows, "both_met": all(r["met"] for r in rows.values())}


def stability(first: Mapping[str, str], second: Mapping[str, str], sets_of: Mapping[str, str]) -> Dict[str, Any]:
    """Agreement of the route between two runs of the same requests, overall and by set, and the changes."""
    common = [rid for rid in first if rid in second]
    same = [rid for rid in common if first[rid] == second[rid]]
    by_set: Dict[str, Any] = {}
    for name in sorted({sets_of[r] for r in common}):
        ids = [r for r in common if sets_of[r] == name]
        agree = sum(1 for r in ids if first[r] == second[r])
        by_set[name] = {"requests": len(ids), "same_route": agree, "agreement": agree / len(ids) if ids else None}
    changes = Counter(f"{first[r]} -> {second[r]}" for r in common if first[r] != second[r])
    return {"requests": len(common), "same_route": len(same), "agreement": len(same) / len(common) if common else None,
            "by_set": by_set, "changes": dict(changes),
            "changed_requests": sorted(r for r in common if first[r] != second[r])}


# =============================================================================
# Contrasts
# =============================================================================


def contrast_row(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    diff, p = paired_permutation_test(list(a), list(b), n_resamples=D.N_RESAMPLES, seed=D.SEED)
    _, low, high = paired_bootstrap_ci(list(a), list(b), n_resamples=D.N_RESAMPLES, seed=D.SEED)
    return {"requests": len(a), "system_mean": float(np.mean(a)) if a else None,
            "reference_mean": float(np.mean(b)) if b else None, "difference": diff, "ci": [low, high], "p": p}


def contrast_family(pairs: Mapping[str, Tuple[Sequence[float], Sequence[float]]]) -> Dict[str, Any]:
    """The declared family (H7-H9) under one judge: difference, interval, raw and Holm-adjusted p."""
    out = {name: contrast_row(a, b) for name, (a, b) in pairs.items() if len(a)}
    adjusted = holm_correction({name: row["p"] for name, row in out.items()})
    for name, row in out.items():
        row["p_holm"] = adjusted.get(name)
    return out


def conclusions(primary: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, str]:
    """The protocol's rule (section 6): sustained only with the same sign and significance under both judges."""
    return {name: conclusion(primary[name], second[name]) for name in primary if name in second}


def mean_ci(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci": [None, None]}
    mean, low, high = bootstrap_ci(list(values), n_resamples=D.N_RESAMPLES, seed=D.SEED)
    return {"n": len(values), "mean": mean, "ci": [low, high]}


# =============================================================================
# Grounding
# =============================================================================


def search_grounding(reply: Mapping[str, Any], answer_raw: Sequence[Mapping[str, Any]],
                     answer_context: int) -> Optional[Dict[str, Any]]:
    """
    What the verify node dropped in one search-route reply: picks proposed by
    the answer call against picks kept, why each was dropped (not among the
    books shown, a repeat, or no citation that passes the check), and
    citations proposed against citations kept.
    """
    if reply.get("route") != "search" or not reply.get("search"):
        return None
    outputs = [a["output"] for a in answer_raw if "output" in a]
    if not outputs:
        return {"answer_call": False, "picks_proposed": 0, "picks_kept": len(reply.get("recommendations") or [])}
    raw = outputs[-1]
    shown = set(reply["search"]["ids"][:answer_context])
    kept = {r["work_id"]: r for r in reply.get("recommendations") or []}
    reasons: Counter = Counter()
    citations_proposed = citations_kept = 0
    seen = set()
    for pick in raw.get("picks") or []:
        book_id = str(pick.get("book_id", "")).strip()
        cites = pick.get("citations") or []
        citations_proposed += len(cites)
        if book_id not in shown:
            reasons["not_shown"] += 1
        elif book_id in seen:
            reasons["repeat"] += 1
        elif book_id not in kept:
            reasons["no_valid_citation"] += 1
        else:
            citations_kept += len(kept[book_id]["citations"])
        seen.add(book_id)
    proposed = len(raw.get("picks") or [])
    return {"answer_call": True, "picks_proposed": proposed, "picks_kept": len(kept),
            "picks_dropped": proposed - len(kept), "dropped_reasons": dict(reasons),
            "citations_proposed": citations_proposed, "citations_kept": citations_kept}


def sum_grounding(rows: Iterable[Optional[Mapping[str, Any]]]) -> Dict[str, Any]:
    total: Counter = Counter()
    reasons: Counter = Counter()
    replies = without_answer = all_dropped = 0
    for row in rows:
        if row is None:
            continue
        replies += 1
        if not row.get("answer_call"):
            without_answer += 1
            continue
        for key in ("picks_proposed", "picks_kept", "picks_dropped", "citations_proposed", "citations_kept"):
            total[key] += row.get(key, 0)
        reasons.update(row.get("dropped_reasons") or {})
        if row.get("picks_proposed") and not row.get("picks_kept"):
            all_dropped += 1
    return {"replies": replies, "replies_without_answer_output": without_answer,
            "replies_with_every_pick_dropped": all_dropped, **dict(total), "dropped_reasons": dict(reasons)}


def agent_grounding(agents: Iterable[Optional[Mapping[str, Any]]]) -> Dict[str, Any]:
    runs = dropped = reflections = recommended = 0
    tools: Counter = Counter()
    rounds: List[int] = []
    for agent in agents:
        if not agent:
            continue
        runs += 1
        dropped += agent.get("dropped_recommendations", 0)
        reflections += 1 if agent.get("reflections") else 0
        recommended += len(agent.get("recommendations") or [])
        tools.update(s["tool"] for s in agent.get("steps") or [])
        rounds.append(len(agent.get("steps") or []))
    return {"runs": runs, "recommended": recommended, "dropped_not_from_tools": dropped,
            "runs_with_reflection": reflections, "tool_calls": dict(tools),
            "tool_calls_per_run": describe(rounds)}


# =============================================================================
# Language of the reply
# =============================================================================


def reply_language_verdict(text: str, language: str) -> Tuple[str, Optional[str]]:
    """
    Is the reply in the request's language? ("ok", "mismatch" or
    "undecided", and the detected language). The catalog's detector
    (twelve languages, at least 40 letters) decides; when it cannot, the
    writing system decides for ru, zh, ja and ar (V14's rule), and other
    languages are "undecided".
    """
    from app.evaluation.catalog_evaluation.queries import NON_LATIN, language_check
    from app.ingestion.language_detection import detect_language

    if not text or not text.strip():
        return "undecided", None
    detected = detect_language(text)
    if detected is not None:
        return ("ok" if detected == language else "mismatch"), detected
    if language not in NON_LATIN:  # function words of a query are no evidence for a reply: undecided
        return "undecided", None
    verdict, _ = language_check(text, language)
    return verdict, None


# =============================================================================
# Cost and time
# =============================================================================


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def describe(values: Sequence[float]) -> Dict[str, Any]:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "median": percentile(values, 50), "p95": percentile(values, 95),
            "mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))}


def turn_cost(record: Mapping[str, Any]) -> Dict[str, Any]:
    """LLM calls, failed calls and reported cost of one turn (its LLM and API calls)."""
    llm = record.get("llm_calls") or []
    api = record.get("api_calls") or []
    cost = sum(float(c["cost_usd"]) for c in llm + api if c.get("cost_usd") is not None)
    return {"llm_calls": len(llm), "llm_failed": sum(1 for c in llm if c.get("error")),
            "api_calls": len(api), "api_failed": sum(1 for c in api if c.get("error") and c["error"] != "cancelled"),
            "api_cancelled": sum(1 for c in api if c.get("error") == "cancelled"),
            "llm_cost_usd": sum(float(c["cost_usd"]) for c in llm if c.get("cost_usd") is not None),
            "cost_usd": cost}


def by_key(rows: Iterable[Tuple[str, Mapping[str, Any]]], fields: Sequence[str]) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for key, row in rows:
        groups[key].append(row)
    return {key: {"turns": len(items), **{f: describe([r.get(f) for r in items]) for f in fields}}
            for key, items in sorted(groups.items())}


def safe_mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(values)) if values else None
