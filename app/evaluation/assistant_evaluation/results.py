"""
Build the results document from the saved files only (no API call, no
database connection): the request sets and their checks, P, the assistant's
turns, the E conversations, the direct agent, pairs.json and the judgment
caches (V14's, read only, and this task's), the ledger and the stage times.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from app.evaluation.assistant_evaluation import analysis as A
from app.evaluation.assistant_evaluation import design as D
from app.evaluation.assistant_evaluation import runs as R
from app.evaluation.assistant_evaluation.judging import PAIRS_FILE, JUDGING_RUNS, judged_counts, load_grades, load_run_data
from app.evaluation.assistant_evaluation.sets import Request, load_requests
from app.evaluation.catalog_evaluation.ledger import LEDGER_FILE, SpendLedger, summarize_ledger


def _read_json(path: Path) -> Optional[Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _turn(turns: Mapping[str, Any], run: str, request_id: str, turn: int = 1) -> Optional[Dict[str, Any]]:
    return turns.get(R.turn_key(run, request_id, turn))


def _reply(record: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    return (record or {}).get("reply")


def per_judge(requests: List[Request], data: Mapping[str, Any], grades: Mapping[str, Mapping[str, int]],
              agent_ids: List[str]) -> Dict[str, Any]:
    """Every quality measure under one judge's grades."""
    p, turns, conversations, agent = data["P"], data["turns"], data["conversations"], data["agent"]
    per_request: Dict[str, Dict[str, Any]] = {}

    def p_ids(text: str) -> List[str]:
        return (p.get(text) or {}).get("ids", [])

    # --- H7: A + B ------------------------------------------------------------
    ab = [r for r in requests if r.set in (D.SET_A, D.SET_B)]
    h7_a, h7_b, search_only = [], [], {"assistant": [], "P": []}
    groups: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"assistant": [], "P": []})
    picks_ab = {"assistant": [], "P": []}
    for r in ab:
        g = grades.get(r.need_id, {})
        reply = _reply(_turn(turns, D.RUN_1, r.request_id))
        assistant = A.ndcg10(A.reply_list(reply), g)
        reference = A.ndcg10(p_ids(r.text), g)
        h7_a.append(assistant)
        h7_b.append(reference)
        group = A.route_group(r)
        groups[group]["assistant"].append(assistant)
        groups[group]["P"].append(reference)
        route = A.reply_route(reply)
        groups[f"route {route}"]["assistant"].append(assistant)
        groups[f"route {route}"]["P"].append(reference)
        if route == "search":
            search_only["assistant"].append(assistant)
            search_only["P"].append(reference)
            rewritten = bool(reply.get("search_query")) and reply["search_query"].strip() != r.text.strip()
            key = "rewritten" if rewritten else "unchanged"
            search_only.setdefault(key, []).append(assistant - reference)
        picks = A.reply_picks(reply)
        if picks:
            picks_ab["assistant"].append(A.precision_at(picks, g, len(picks)))
            picks_ab["P"].append(A.precision_at(p_ids(r.text), g, len(picks)))
        per_request[r.request_id] = {"set": r.set, "route": route, "ndcg_assistant": assistant, "ndcg_P": reference,
                                     "judged_at_10_assistant": A.judged_share(A.reply_list(reply), g),
                                     "query_rewritten": bool(reply and reply.get("search_query")
                                                             and reply["search_query"].strip() != r.text.strip())}

    # --- H8: E ----------------------------------------------------------------
    e = [r for r in requests if r.set == D.SET_E]
    h8_a, h8_b = [], []
    e_rows = []
    for r in e:
        g = grades.get(r.need_id, {})
        conversation = conversations.get(r.request_id) or {}
        reply = A.last_reply(conversation, turns, r.request_id)
        final = A.ndcg10(A.reply_list(reply), g)
        vague = A.ndcg10(p_ids(r.text), g)
        original = A.ndcg10(p_ids(r.source_query or ""), g)
        first = _reply(_turn(turns, D.RUN_1, r.request_id, 1))
        h8_a.append(final)
        h8_b.append(vague)
        row = {"request_id": r.request_id, "language": r.language, "text": r.text, "questions": conversation.get("questions"),
               "assistant_turns": conversation.get("assistant_turns"), "stop": conversation.get("stop"),
               "first_route": A.reply_route(first), "final_route": A.reply_route(reply), "ndcg_final": final,
               "ndcg_first_turn": A.ndcg10(A.reply_list(first), g), "ndcg_P_vague": vague, "ndcg_P_original": original,
               "final_query": (reply or {}).get("search_query")}
        e_rows.append(row)
        per_request[r.request_id] = {"set": r.set, **row}
    asked = [x for x in e_rows if (x["questions"] or 0) > 0]
    not_asked = [x for x in e_rows if not (x["questions"] or 0)]

    def means(rows, field):
        return A.mean_ci([x[field] for x in rows])

    e_summary = {
        "conversations": len(e_rows),
        "asked": len(asked), "clarification_rate": len(asked) / len(e_rows) if e_rows else None,
        "questions": dict(Counter(x["questions"] for x in e_rows)),
        "assistant_turns": dict(Counter(x["assistant_turns"] for x in e_rows)),
        "stop": dict(Counter(x["stop"] for x in e_rows)),
        "final_route": dict(Counter(x["final_route"] for x in e_rows)),
        "ndcg_final": means(e_rows, "ndcg_final"), "ndcg_P_vague": means(e_rows, "ndcg_P_vague"),
        "ndcg_P_original": means(e_rows, "ndcg_P_original"),
        "asked_rows": {f: means(asked, f) for f in ("ndcg_final", "ndcg_P_vague", "ndcg_P_original")},
        "not_asked_rows": {f: means(not_asked, f) for f in ("ndcg_final", "ndcg_P_vague", "ndcg_P_original")},
        "rows": e_rows,
    }

    # --- H9: F ----------------------------------------------------------------
    f = [r for r in requests if r.set == D.SET_F]
    h9_a, h9_b = [], []
    f_rows = []
    for r in f:
        g = grades.get(r.need_id, {})
        reply = _reply(_turn(turns, D.RUN_1, r.request_id))
        picks = A.reply_picks(reply)
        row = {"request_id": r.request_id, "kind": r.category, "language": r.language, "route": A.reply_route(reply),
               "forced_route": (reply or {}).get("forced_route"), "picks": len(picks),
               "precision_picks": A.precision_at(picks, g, len(picks)),
               "precision_P": A.precision_at(p_ids(r.text), g, len(picks)),
               "ndcg_list": A.ndcg10(A.reply_list(reply), g), "ndcg_P": A.ndcg10(p_ids(r.text), g),
               "named_works_recommended": sum(1 for w in r.works if w in picks),
               "named_works_in_P_top_k": sum(1 for w in r.works if w in p_ids(r.text)[: max(len(picks), 1)]),
               "named_works": len(r.works)}
        f_rows.append(row)
        per_request[r.request_id] = {"set": r.set, **row}
        if picks:
            h9_a.append(row["precision_picks"])
            h9_b.append(row["precision_P"])
    f_summary = {"requests": len(f_rows), "with_recommendations": len(h9_a),
                 "without_recommendations": [x["request_id"] for x in f_rows if not x["picks"]],
                 "route": dict(Counter(x["route"] for x in f_rows)),
                 "by_route": {route: {"requests": len(rows), "precision_picks": A.safe_mean([x["precision_picks"] for x in rows]),
                                      "precision_P": A.safe_mean([x["precision_P"] for x in rows])}
                              for route, rows in _group(f_rows, "route").items()},
                 "by_kind": {kind: {"requests": len(rows), "precision_picks": A.safe_mean([x["precision_picks"] for x in rows]),
                                    "precision_P": A.safe_mean([x["precision_P"] for x in rows]),
                                    "routes": dict(Counter(x["route"] for x in rows))}
                             for kind, rows in _group(f_rows, "kind").items()},
                 "named_works_recommended": sum(x["named_works_recommended"] for x in f_rows),
                 "named_works": sum(x["named_works"] for x in f_rows),
                 "named_works_in_P_top_k": sum(x["named_works_in_P_top_k"] for x in f_rows),
                 "ndcg_list": A.mean_ci([x["ndcg_list"] for x in f_rows]), "ndcg_P": A.mean_ci([x["ndcg_P"] for x in f_rows]),
                 "rows": f_rows}

    # --- direct agent on the sample of A and B ------------------------------------
    index = {r.request_id: r for r in requests}
    agent_rows = []
    for request_id in agent_ids:
        r = index[request_id]
        g = grades.get(r.need_id, {})
        record = agent.get(request_id) or {}
        picks = [x["work_id"] for x in ((record.get("agent") or {}).get("recommendations") or [])]
        reply = _reply(_turn(turns, D.RUN_1, request_id))
        assistant_picks = A.reply_picks(reply)
        agent_rows.append({"request_id": request_id, "set": r.set, "picks": len(picks),
                           "precision_agent": A.precision_at(picks, g, len(picks)),
                           "precision_P_same_k": A.precision_at(p_ids(r.text), g, len(picks)),
                           "assistant_route": A.reply_route(reply), "assistant_picks": len(assistant_picks),
                           "precision_assistant": A.precision_at(assistant_picks, g, len(assistant_picks)),
                           "precision_P_assistant_k": A.precision_at(p_ids(r.text), g, len(assistant_picks)),
                           "ndcg_agent_list": A.ndcg10(picks, g), "ndcg_P": A.ndcg10(p_ids(r.text), g),
                           "error": bool(record.get("error"))})
    with_picks = [x for x in agent_rows if x["picks"]]
    both = [x for x in agent_rows if x["picks"] and x["assistant_picks"]]
    agent_summary = {
        "requests": len(agent_rows), "with_recommendations": len(with_picks), "errors": sum(x["error"] for x in agent_rows),
        "precision_agent": A.mean_ci([x["precision_agent"] for x in with_picks]),
        "precision_P_same_k": A.mean_ci([x["precision_P_same_k"] for x in with_picks]),
        "paired_with_assistant": len(both),
        "precision_agent_paired": A.safe_mean([x["precision_agent"] for x in both]),
        "precision_assistant_paired": A.safe_mean([x["precision_assistant"] for x in both]),
        "assistant_routes": dict(Counter(x["assistant_route"] for x in agent_rows)),
        "picks": dict(Counter(x["picks"] for x in agent_rows)),
        "rows": agent_rows,
    }

    contrasts = A.contrast_family({"H7": (h7_a, h7_b), "H8": (h8_a, h8_b), "H9": (h9_a, h9_b)})
    return {
        "contrasts": contrasts,
        "h7": {"assistant": A.mean_ci(h7_a), "P": A.mean_ci(h7_b),
               "by_group": {k: {"requests": len(v["assistant"]), "assistant": A.safe_mean(v["assistant"]),
                                "P": A.safe_mean(v["P"])} for k, v in sorted(groups.items())},
               "search_route_only": {"requests": len(search_only["assistant"]),
                                     "assistant": A.safe_mean(search_only["assistant"]),
                                     "P": A.safe_mean(search_only["P"]),
                                     "test": A.contrast_row(search_only["assistant"], search_only["P"])
                                     if search_only["assistant"] else None,
                                     "query_rewritten": len(search_only.get("rewritten", [])),
                                     "rewritten_mean_difference": A.safe_mean(search_only.get("rewritten", [])),
                                     "rewritten_changed_ndcg": sum(1 for x in search_only.get("rewritten", []) if abs(x) > 1e-9),
                                     "query_unchanged": len(search_only.get("unchanged", []))},
               "recommendations": {"replies": len(picks_ab["assistant"]),
                                   "precision_assistant": A.safe_mean(picks_ab["assistant"]),
                                   "precision_P_same_k": A.safe_mean(picks_ab["P"])}},
        "e": e_summary, "f": f_summary, "agent_direct": agent_summary, "per_request": per_request,
    }


def _group(rows, field):
    out = defaultdict(list)
    for row in rows:
        out[row[field]].append(row)
    return dict(out)


def build_results(data_dir: Path) -> Dict[str, Any]:
    requests = load_requests(data_dir / "sets" / "requests.json")
    meta = _read_json(data_dir / "sets" / "requests.json") or {}
    agent_ids = list(meta.get("agent_sample") or [])
    checks = _read_json(data_dir / "sets" / "checks.json") or {}
    data = load_run_data(data_dir)
    all_turns = data["turns"]
    # Set G exists only in the re-measurement (V25); its turns are reported in
    # their own section, so every other measure covers the same requests in
    # both measurements.
    turns = {k: v for k, v in all_turns.items() if v.get("set") != D.SET_G}
    data = {**data, "turns": turns}
    pairs = _read_json(data_dir / "judgments" / PAIRS_FILE) or {}
    sets_of = {r.request_id: r.set for r in requests}

    # --- routes -------------------------------------------------------------------
    matrices, rates, forced = {}, {}, {}
    route_by_run: Dict[str, Dict[str, str]] = {}
    for run in (D.RUN_1, D.RUN_2):
        rows, routes_ab, routes_d, forced_rows = [], {}, [], Counter()
        for r in requests:
            if r.set in (D.SET_E,):
                if run != D.RUN_1:
                    continue
                group = "E first turn"
            elif r.set in (D.SET_D, D.SET_F) and run != D.RUN_1:
                continue
            else:
                group = A.route_group(r)
            record = _turn(turns, run, r.request_id)
            if record is None:
                continue
            reply = _reply(record)
            route = A.reply_route(reply)
            rows.append((group, route))
            if reply and reply.get("forced_route"):
                forced_rows[f"{r.set}: {reply['forced_route']} -> {route}"] += 1
            if r.set in (D.SET_A, D.SET_B):
                routes_ab[r.request_id] = route
            if r.set == D.SET_D:
                routes_d.append(route)
        if rows:
            matrices[run] = A.route_matrix(rows)
            rates[run] = A.route_rates(routes_ab, sets_of, routes_d)
            forced[run] = dict(forced_rows)
            route_by_run[run] = routes_ab
    e_later = Counter()
    for record in turns.values():
        if record.get("set") == D.SET_E and record.get("turn", 1) > 1:
            reply = _reply(record)
            e_later[f"turn {record['turn']}: {A.reply_route(reply)}"] += 1
            if reply and reply.get("forced_route"):
                e_later[f"turn {record['turn']} forced: {reply['forced_route']}"] += 1
    threshold = {run: A.thresholds(rates[run]) for run in rates}
    stab = A.stability(route_by_run.get(D.RUN_1, {}), route_by_run.get(D.RUN_2, {}), sets_of) if D.RUN_2 in route_by_run else None
    clarify_ab = [{"request_id": rid, "set": sets_of[rid], "text": next(r.text for r in requests if r.request_id == rid),
                   "question": (_reply(_turn(turns, D.RUN_1, rid)) or {}).get("message")}
                  for rid, route in route_by_run.get(D.RUN_1, {}).items() if route in ("clarify", "out_of_scope")]
    d_rows = [{"request_id": r.request_id, "text": r.text,
               "route": A.reply_route(_reply(_turn(turns, D.RUN_1, r.request_id)))} for r in requests if r.set == D.SET_D]

    # --- quality under both judges ---------------------------------------------------
    judges = {}
    for label, model in (("primary", D.PRIMARY_JUDGE), ("second", D.SECOND_JUDGE)):
        grades = load_grades(data_dir, model, pairs) if pairs else {}
        judges[label] = per_judge(requests, data, grades, agent_ids)
    concl = A.conclusions(judges["primary"]["contrasts"], judges["second"]["contrasts"])

    # --- grounding --------------------------------------------------------------------
    # Books shown to the answer call, as saved with the measurement (never the live code's constant).
    config = _read_json(data_dir / "system.json")
    context_books = ((config or {}).get("assistant") or {}).get("answer_context_books")
    if context_books is None:
        raise ValueError(f"{data_dir / 'system.json'} does not record assistant.answer_context_books")
    run1 = [rec for rec in turns.values() if rec.get("run") == D.RUN_1 and rec.get("reply")]
    grounding_rows = defaultdict(list)
    for rec in run1:
        grounding_rows[rec["set"]].append(A.search_grounding(rec["reply"], rec.get("answer_raw") or [], context_books))
    search_grounding = {name: A.sum_grounding(rows) for name, rows in sorted(grounding_rows.items())}
    search_grounding["all"] = A.sum_grounding([x for rows in grounding_rows.values() for x in rows])
    agent_in_assistant = A.agent_grounding(rec["reply"].get("agent") for rec in run1 if rec["reply"].get("agent"))
    agent_direct = A.agent_grounding(rec.get("agent") for rec in data["agent"].values())

    # --- language ---------------------------------------------------------------------
    index = {r.request_id: r for r in requests}
    language_rows = []
    for rec in run1:
        r = index[rec["request_id"]]
        verdict, detected = A.reply_language_verdict(rec["reply"].get("message") or "", r.language)
        fixed = rec["reply"]["route"] == "out_of_scope" or (rec["reply"]["route"] == "search"
                                                               and not rec["reply"].get("recommendations"))
        language_rows.append({"set": r.set, "language": r.language, "route": rec["reply"]["route"],
                              "verdict": verdict, "detected": detected,
                              "text": "fixed reply" if fixed else "written by the LLM"})
    language = {"by_language": _verdicts(language_rows, "language"), "by_route": _verdicts(language_rows, "route"),
                "by_set": _verdicts(language_rows, "set"), "by_text": _verdicts(language_rows, "text"),
                "llm_text_by_language": _verdicts([x for x in language_rows if x["text"] != "fixed reply"], "language"), "overall": dict(Counter(x["verdict"] for x in language_rows)),
                "mismatches_detected_as": dict(Counter(f"{x['language']}->{x['detected']}" for x in language_rows
                                                       if x["verdict"] == "mismatch"))}

    # --- latency and cost --------------------------------------------------------------
    turn_rows = []
    for rec in turns.values():
        if not rec.get("reply"):
            continue
        cost = A.turn_cost(rec)
        turn_rows.append({"run": rec["run"], "set": rec["set"], "route": rec["reply"]["route"],
                          "wall_s": rec["wall_ms"] / 1000.0, **cost,
                          "nodes": rec["reply"].get("timings_ms") or {}})
    node_times: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in turn_rows:
        for node, value in row["nodes"].items():
            node_times[row["route"]][node].append(value / 1000.0)
    fields = ("wall_s", "cost_usd", "llm_calls")
    agent_rows = [{"wall_s": rec["wall_ms"] / 1000.0, **A.turn_cost(rec)} for rec in data["agent"].values() if rec.get("agent")]
    cost_time = {
        "by_route": A.by_key(((row["route"], row) for row in turn_rows), fields),
        "by_set": A.by_key(((row["set"], row) for row in turn_rows), fields),
        "all_turns": A.by_key((("all", row) for row in turn_rows), fields).get("all"),
        "node_seconds_by_route": {route: {node: A.describe(v) for node, v in nodes.items()}
                                  for route, nodes in node_times.items()},
        "agent_direct": A.by_key((("agent", row) for row in agent_rows), fields).get("agent"),
        "llm_calls": sum(r["llm_calls"] for r in turn_rows) + sum(r["llm_calls"] for r in agent_rows),
        "llm_failed": sum(r["llm_failed"] for r in turn_rows) + sum(r["llm_failed"] for r in agent_rows),
        "api_failed": sum(r["api_failed"] for r in turn_rows) + sum(r["api_failed"] for r in agent_rows),
        "api_cancelled": sum(r["api_cancelled"] for r in turn_rows) + sum(r["api_cancelled"] for r in agent_rows),
        "ledger_errors": dict(Counter(f"{e.get('stage')} {e.get('model')}: {e.get('error')}"
                                      for e in SpendLedger(data_dir / LEDGER_FILE, D.spend_cap(data_dir)).entries()
                                      if e.get("error"))),
    }

    # --- failures ----------------------------------------------------------------------
    failures = {
        "turns_without_reply": sorted(k for k, rec in turns.items() if not rec.get("reply")),
        "turns_retried": sorted(k for k, rec in turns.items() if rec.get("failed_attempts")),
        "notices": dict(Counter(n for rec in turns.values() if rec.get("reply") for n in rec["reply"].get("notices") or [])),
        "degraded_searches": sorted(k for k, rec in turns.items() if rec.get("reply") and (rec["reply"].get("search") or {}).get("degraded")),
        "searches_not_reranked": sorted(k for k, rec in turns.items() if rec.get("reply") and rec["reply"].get("search")
                                        and rec["reply"]["search"]["ids"] and not R.reranked(rec["reply"]["search"])),
        "router_errors": sum(1 for rec in turns.values() for x in rec.get("router_raw") or [] if "error" in x),
        "answer_errors": sum(1 for rec in turns.values() for x in rec.get("answer_raw") or [] if "error" in x),
        "agent_direct_errors": sorted(k for k, rec in data["agent"].items() if not rec.get("agent")),
        "P_not_reranked_or_degraded": sorted(t for t, rec in data["P"].items() if not rec.get("reranked") or rec.get("degraded")),
        "P_retried": sum(1 for rec in data["P"].values() if rec.get("failed_attempts")),
    }
    depth = list(R.load_store(data_dir, R.DEPTH_FILE, "text").values())
    vectors = _read_jsonl(data_dir / "query_vectors" / "qwen__qwen3-embedding-8b.jsonl")
    ledger = SpendLedger(data_dir / LEDGER_FILE, D.spend_cap(data_dir))
    stage_seconds = defaultdict(float)
    for row in _read_jsonl(data_dir / "stage_times.jsonl"):
        stage_seconds[row["stage"]] += row["seconds"]
    commits = Counter(rec.get("commit") for rec in list(turns.values()) + list(data["P"].values()) + list(data["agent"].values()))
    dates = sorted(rec.get("date", "") for rec in list(turns.values()) + list(data["agent"].values()) if rec.get("date"))

    out = {
        "protocol": {"path": D.PROTOCOL_PATH, "commit": D.PROTOCOL_COMMIT},
        "system": config, "commits": dict(commits), "dates": [dates[0], dates[-1]] if dates else None,
        "sets": {"counts": checks.get("sets"), "generator": checks.get("generator"),
                 "e_sample": checks.get("e_sample"), "f_tasks": checks.get("f_tasks"),
                 "agent_sample": checks.get("agent_sample"),
                 "E": {k: checks.get("E", {}).get(k) for k in ("fixes",)} | {
                     "first_pass_problems": (checks.get("E", {}).get("first_pass") or {}).get("problems"),
                     "final_problems": (checks.get("E", {}).get("final") or {}).get("problems"),
                     "language_summary": (checks.get("E", {}).get("final") or {}).get("language_summary")},
                 "F": {"fixes": checks.get("F", {}).get("fixes"),
                       "first_pass_problems": (checks.get("F", {}).get("first_pass") or {}).get("problems"),
                       "final_problems": (checks.get("F", {}).get("final") or {}).get("problems"),
                       "language_summary": (checks.get("F", {}).get("final") or {}).get("language_summary"),
                       "works_found": (checks.get("F", {}).get("final") or {}).get("works_found")},
                 "requests": {r.request_id: {"set": r.set, "text": r.text, "language": r.language,
                                             "source_query": r.source_query, "category": r.category}
                              for r in requests if r.set in (D.SET_E, D.SET_F)}},
        "runs": {"P_texts": len(data["P"]), "turns": len(turns),
                 "turns_by_run_set": dict(Counter(f"{rec.get('run')}/{rec.get('set')}" for rec in turns.values())),
                 "conversations": len(data["conversations"]), "agent_direct": len(data["agent"]),
                 "query_vectors": len(vectors),
                 "query_vector_retries": sum(1 for v in vectors if v.get("failed_attempts")),
                 "depth_check": {"texts": len(depth), "same_head_set": sum(1 for d in depth if d["same_head_set"]),
                                 "same_head_order": sum(1 for d in depth if d["same_head_order"]),
                                 "min_overlap": min((d["head_overlap"] for d in depth), default=None)}},
        "routes": {"matrices": matrices, "rates": rates, "forced": forced, "e_later_turns": dict(e_later),
                   "thresholds": threshold, "stability": stab, "clarify_or_out_of_scope_ab_run1": clarify_ab,
                   "d": d_rows},
        "judgments": {"pairs_file_needs": len(pairs), "counts": judged_counts(data_dir, pairs) if pairs else None,
                      "runs": _read_jsonl(data_dir / "judgments" / JUDGING_RUNS)},
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "per_request"} for k, v in judges.items()},
        "conclusions": concl,
        "grounding": {"search_route_by_set": search_grounding, "agent_in_assistant": agent_in_assistant,
                      "agent_direct": agent_direct},
        "language": language,
        "cost_time": cost_time,
        "failures": failures,
        "spend": summarize_ledger(ledger.entries()),
        "stage_seconds": dict(stage_seconds),
        "per_request": {k: v["per_request"] for k, v in judges.items()},
    }
    if D.is_v25(data_dir):
        out["v25"] = v25_results(data_dir, requests, all_turns, rates, ledger)
    return out


def v25_results(data_dir: Path, requests: List[Request], all_turns: Mapping[str, Any], rates: Mapping[str, Any],
                ledger: SpendLedger) -> Dict[str, Any]:
    """
    The re-measurement's own results (V25 addendum): set G, the thresholds as
    section 4 of the addendum applies them, G's cost and time, the spend cap.
    """
    g_rows = []
    for r in requests:
        if r.set != D.SET_G:
            continue
        record = _turn(all_turns, D.RUN_1, r.request_id)
        reply = _reply(record)
        field, form, english = (list(r.notes) + ["", "", ""])[:3]
        g_rows.append({"request_id": r.request_id, "group": r.category, "language": r.language, "field": field,
                       "form": form, "text": r.text, "english": english, "route": A.reply_route(reply),
                       "forced_route": (reply or {}).get("forced_route"),
                       "message": (reply or {}).get("message"), "search_query": (reply or {}).get("search_query"),
                       "reason": next((x.get("output", {}).get("reason") for x in (record or {}).get("router_raw") or []
                                       if isinstance(x.get("output"), dict)), None)})
    g = A.g_routes(g_rows)
    clarify_b = (rates.get(D.RUN_1) or {}).get("clarify_b") or {}
    required = {
        "out_of_scope_g_valid": g["threshold"],
        "clarify_b": {"value": clarify_b.get("rate"), "count": clarify_b.get("count"),
                      "requests": clarify_b.get("requests"), "maximum": D.CLARIFY_B_MAX,
                      "met": clarify_b.get("rate") is not None and clarify_b["rate"] <= D.CLARIFY_B_MAX},
    }
    g_turns = [rec for rec in all_turns.values() if rec.get("set") == D.SET_G and rec.get("reply")]
    g_cost = [{"route": rec["reply"]["route"], "wall_s": rec["wall_ms"] / 1000.0, **A.turn_cost(rec)} for rec in g_turns]
    checks = _read_json(data_dir / "sets" / "g_checks.json") or {}
    return {
        "addendum": {"path": D.ADDENDUM_PATH, "commit": D.ADDENDUM_COMMIT},
        "g": {**g, "rows": g_rows},
        "g_checks": {"generator": checks.get("generator"), "counts": checks.get("counts"),
                     "first_pass_problems": ((checks.get("G") or {}).get("first_pass") or {}).get("problems"),
                     "fixes": (checks.get("G") or {}).get("fixes"),
                     "final_problems": ((checks.get("G") or {}).get("final") or {}).get("problems"),
                     "language_summary": ((checks.get("G") or {}).get("final") or {}).get("language_summary")},
        "thresholds": {"required": required, "required_met": all(x["met"] for x in required.values()),
                       "secondary": {run: A.thresholds(rates[run]) for run in rates}},
        "g_cost_time": {"by_route": A.by_key(((row["route"], row) for row in g_cost), ("wall_s", "cost_usd", "llm_calls")),
                        "llm_failed": sum(row["llm_failed"] for row in g_cost),
                        "turns_without_reply": sorted(k for k, rec in all_turns.items()
                                                      if rec.get("set") == D.SET_G and not rec.get("reply"))},
        "spend_cap_usd": ledger.cap_usd,
    }


def _verdicts(rows, field) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        out[row[field]][row["verdict"]] += 1
    return {k: dict(v) for k, v in sorted(out.items())}
