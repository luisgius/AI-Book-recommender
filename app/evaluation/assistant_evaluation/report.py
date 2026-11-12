"""Markdown report of the assistant evaluation, rendered from the results document only."""

from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.evaluation.assistant_evaluation import design as D

JUDGE_LABEL = {"primary": f"primary `{D.PRIMARY_JUDGE}`", "second": f"second `{D.SECOND_JUDGE}`"}

DEVIATIONS: List[str] = [
    "Simulated user (set E): the protocol asks for an LLM of another vendor than the system LLM; "
    f"`{D.SIMULATED_USER_MODEL}` (Mistral, a vendor used nowhere else in the evaluation) was used at temperature "
    f"{D.SIMULATED_USER_TEMPERATURE}, with the prompt in `simulated_user.py` (it sees the hidden need, the conversation "
    "so far and the latest question with its options, never a result). An empty answer would have been replaced by "
    "the chosen option or the first option (recorded per call).",
    "Set E checks. The protocol names the generator and the size of the requests but not the checks of \"does not give "
    "its need away\"; they were fixed here as: at most one content word shared with the original B query, at most "
    f"{D.E_MAX_WORDS} words ({D.E_MAX_CJK_CHARS} characters in Chinese and Japanese), not the original query, and for "
    "known-item needs no title word and not the author's surname. After the first generation pass, and before any "
    "system run, the shared-word rule was widened to count inflected forms (same first five letters), because "
    "`славянская мифология` passed for a need about Slavic mythology; the generator calls were reused from the cache and "
    "only the new replacements were requested. The checks of that first pass are kept in "
    "`sets/checks_before_prefix_rule.json`.",
    "V14's language check of queries reads the elided French article (`d'`) as Catalan; three correct French E requests "
    "(`récits d'aventure`, `roman d'anticipation`, `livre d'histoire`) were therefore sent back and replaced. The "
    "replacements are also French and are what was evaluated.",
    "Set F design, not fixed by the protocol: 7 comparisons of two works, 7 \"like X but ...\" and 6 questions about one "
    f"work; the first work of each task drawn with seed {D.F_SEED} from the catalog works in the twelve B languages with "
    f"a description of at least {D.F_MIN_DESCRIPTION} characters, the second work of a comparison sharing the first "
    "one's author (else subject, else language); each request written in the first work's catalog language.",
    "Definition of the list a reply gives (needed for nDCG@10, not spelled out by the protocol): the search results on "
    "the search route, the agent's recommendations in order on the agent route, and no list (nDCG@10 = 0) on the "
    "clarify and out-of-scope routes, because the user got no books in that turn. H7 is also shown on the requests the "
    "assistant searched, descriptively.",
    "H9 uses the assistant's turn on each F request whatever route it took (the system as it runs; set F expects the "
    "agent route), and compares the precision of its recommendations (grade 2 or 3) with P's precision at the same "
    "number of top positions. Requests without any recommendation have no number of positions and are left out of "
    "H9; they are listed.",
    "The two thresholds of section 7 are applied to run 1 of A and B; run 2 is shown next to them.",
    "Depth. The assistant was called with the endpoint's default of 10 results (vector candidate depth 60), P with "
    "100 results (depth 200) so that its top-100 list could be saved. The 30 candidates the reranker sees were "
    "compared at both depths for every P text (database only); the result is in the runs section.",
    "Deadlines. The reranker calls used the evaluation's long deadlines (30 s per attempt, 3 retries), as in V14, not the "
    "deployed 5 s, so that a slow provider does not turn a quality measure into a fallback. Query embeddings used the "
    "application's encoder with its deployed settings (DeepInfra, hedge to Nebius after 1 s, DeepInfra again at 2.5 s, "
    "5 s deadline, Zero Data Retention, in-process cache), behind an evaluation cache of vectors by text with up to "
    f"{D.EMBED_ATTEMPTS} attempts per text; a P search that came back without reranking or degraded was repeated (up to "
    f"{D.TURN_ATTEMPTS} searches). A turn of the assistant that raised was repeated up to {D.TURN_ATTEMPTS} times.",
    "LLM settings: those `LLMSettings.from_env` gives with the project's `.env` (OpenRouter, "
    f"`{D.SYSTEM_LLM}`, temperature 0, reasoning off, 1024 output tokens, 180 s timeout). The deployed function's own "
    "environment was not read (V17 records `LLM_TIMEOUT_S=15` there); a slow call that the deployment would cut at 15 s "
    "completed here.",
    "No daily budget: the search service was built without the usage budget, so the live site's counters were neither "
    "read nor spent. The database connection ran every statement with `transaction_read_only=on`, and its write methods "
    "raise.",
    "Judgments: a pair already judged in V14 (same need id, so the same need text, prompt and judge) was reused; nDCG@10 "
    "is computed against the union of V14's pool and the pairs judged here, so its values are not comparable with "
    "V14's report, only within this one.",
    "The agent called directly on the 40-request sample receives the request text, as the assistant would hand it a "
    "condensed request.",
    "Per-turn costs add the LLM calls and the embedding and rerank calls recorded in the request's thread; hedged "
    "embedding requests run on other threads and are only in the ledger. The ledger, not the per-turn sums, gives the "
    "spend.",
    "V14's generator class labels its ledger entries with the stage `1-queries`; in this ledger that label is stage 1.",
    "Before the full runs, each part of stage 3 was run on its first two requests as a smoke test with the same code; "
    "those turns are kept and are part of the data (a re-run skips what is stored). A separate check of the "
    "composition (one P search and one assistant turn, about 0.004 USD) was made outside the evaluation's data "
    "directory, so it is not in the ledger; it falls inside the key-usage window reported on the first page.",
    "Commit: HEAD moved during the task from `e964682` to `36802ee` (another agent's documentation-only commit); every "
    "run of this evaluation was made at `36802ee`, whose application code is that of `e964682`.",
]

LIMITS: List[str] = [
    "Judgments are from two LLM judges only; their agreement with people on this catalog is pending V14's human sheet.",
    "The user of set E is simulated: it always answers, briefly and coherently with the need, which a person does not "
    "guarantee. Any gain measured in H8 is a favourable bound.",
    "Sets E and F were written by an LLM; F requests name catalog works exactly by their catalog titles.",
    "The assistant is not deterministic. Its route was repeated once on A and B; quality was measured once (run 1).",
    "E and F are small (30 and 20 requests); per-language, per-kind and per-route tables are descriptive.",
    "Latencies were measured from a laptop over the internet, four requests in flight, against a Neon compute shared "
    "with the live site, with the evaluation's reranker deadlines; they describe this setting, not the deployed function.",
    "The judges grade a work against a need; they do not grade the reply's text, the comparison an F request asks for, "
    "or whether a clarifying question was useful in itself.",
]


def _f(value: Optional[float], digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.1f} %"


def _p(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _ci(ci: Sequence[Optional[float]], signed: bool = False) -> str:
    return f"[{_f(ci[0], signed=signed)}, {_f(ci[1], signed=signed)}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _usd(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _key_usage(spend: Mapping[str, Any]) -> Optional[float]:
    values = [s.get("key_usage_usd") for s in spend.get("key_usage_snapshots") or [] if s.get("key_usage_usd") is not None]
    return values[-1] - values[0] if len(values) >= 2 else None


def summary_lines(r: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    th = (r["routes"]["thresholds"] or {}).get(D.RUN_1)
    if th:
        oos, cl = th["rows"]["out_of_scope_ab"], th["rows"]["clarify_b"]
        out.append(f"Section 7 thresholds (run 1): out of scope on A+B {oos['count']}/{oos['requests']} = {_pct(oos['value'])} "
                   f"(max {_pct(oos['maximum'])}): **{'met' if oos['met'] else 'not met'}**; questions on B "
                   f"{cl['count']}/{cl['requests']} = {_pct(cl['value'])} (max {_pct(cl['maximum'])}): "
                   f"**{'met' if cl['met'] else 'not met'}**. Both: **{'met' if th['both_met'] else 'not met'}**.")
    rates = (r["routes"]["rates"] or {}).get(D.RUN_1)
    if rates:
        out.append(f"Out-of-scope detection on D: {rates['d_detection']['count']}/{rates['d_detection']['requests']}; "
                   f"questions on A+B: {rates['clarify_ab']['count']}/{rates['clarify_ab']['requests']}; agent route on "
                   f"A+B: {rates['agent_ab']['count']}.")
    stab = r["routes"].get("stability")
    if stab:
        out.append(f"Route stability (A+B run twice): same route on {stab['same_route']}/{stab['requests']} "
                   f"({_pct(stab['agreement'])}).")
    texts = {"H7": "assistant vs P, A+B, nDCG@10", "H8": "assistant after asking vs P on the vague request, E, nDCG@10",
             "H9": "recommendation precision vs P at the same k, F"}
    for name, text in texts.items():
        p = r["metrics"].get("primary", {}).get("contrasts", {}).get(name)
        s = r["metrics"].get("second", {}).get("contrasts", {}).get(name)
        if not p or not s:
            out.append(f"{name} ({text}): not computed.")
            continue
        out.append(f"{name} ({text}): {_f(p['difference'], signed=True)} (Holm p {_p(p['p_holm'])}) under the primary "
                   f"judge, {_f(s['difference'], signed=True)} (Holm p {_p(s['p_holm'])}) under the second: "
                   f"**{r['conclusions'].get(name, 'n/a')}**.")
    h7 = r["metrics"].get("primary", {}).get("h7")
    if h7:
        so = h7["search_route_only"]
        groups = h7["by_group"]
        out.append(f"Where H7 comes from (primary judge): on the {so['requests']} A+B requests the assistant searched, its "
                   f"nDCG@10 equals P's ({_f(so['assistant'])} against {_f(so['P'])}; the query was rewritten on "
                   f"{so.get('query_rewritten')} of them, changing nDCG@10 on {so.get('rewritten_changed_ndcg')}). The whole "
                   f"loss is in the requests routed to a question ({groups.get('route clarify', {}).get('requests', 0)}) or "
                   f"out of scope ({groups.get('route out_of_scope', {}).get('requests', 0)}), which return no books. The "
                   "follow-up section 7 attaches to H7 (search with the user's text instead of the rewritten query) would "
                   "therefore not remove the loss; the routing follow-up of the thresholds addresses it.")
    e = r["metrics"].get("primary", {}).get("e")
    if e and e.get("conversations"):
        out.append(f"E: the assistant asked in {e['asked']}/{e['conversations']} conversations; nDCG@10 (primary) after "
                   f"the conversation {_f(e['ndcg_final']['mean'])}, P on the vague request {_f(e['ndcg_P_vague']['mean'])}, "
                   f"P on the original B query {_f(e['ndcg_P_original']['mean'])}.")
    return out


NOT_RECORDED = "not recorded"


def _recorded(saved: Mapping[str, Any], key: str) -> Any:
    """A value of the system description saved with a measurement, or "not recorded" when it has none.

    The report describes the evaluated system only from what was saved when the measurement
    ran; it never reads the environment, the application's settings or a default of the code.
    """
    value = saved.get(key) if isinstance(saved, Mapping) else None
    return NOT_RECORDED if key not in (saved or {}) else value


def build_markdown(r: Mapping[str, Any]) -> str:
    L: List[str] = []
    spend = r["spend"]
    system = r.get("system") or {}
    llm = system.get("llm") or {}
    emb = system.get("embedding") or {}
    rer = system.get("reranker") or {}
    dates = r.get("dates") or ["n/a", "n/a"]
    L += ["# Evaluation of the assistant and the agent (V22)", ""]
    L += [f"- Protocol: `{r['protocol']['path']}`, committed in `{r['protocol']['commit']}` before any result. This "
          "report implements it; every deviation is listed in [Deviations from the protocol](#deviations-from-the-protocol).",
          f"- System evaluated: the branch at commit `{system.get('commit', 'n/a')}` (commits recorded per run: "
          f"{', '.join(f'`{k}` x{v}' for k, v in r.get('commits', {}).items())}), cloud profile, Neon schema `{D.SCHEMA}` "
          f"({D.CATALOG_WORKS:,} works), called through the `ConversationalAssistant` port with the cloud adapters: no HTTP, "
          "no public URL, no daily budget.",
          "- **Relevance judgments come from LLM judges only** (V14's two judges); no person graded anything.",
          "- **The user in set E is simulated** by an LLM; it answers every question briefly and coherently with the hidden need.",
          f"- Date of the runs: {dates[0][:10]} (from {dates[0]} to {dates[-1]}). Report regenerated offline from the "
          "saved files (`.venv/bin/python -m app.evaluation.assistant_evaluation report --offline`).",
          "- Models and providers (all through OpenRouter): system LLM "
          f"`{_recorded(llm, 'model')}` (temperature {_recorded(llm, 'temperature')}, reasoning {_recorded(llm, 'reasoning')}); "
          f"embeddings `{emb.get('model')}` ({emb.get('dimensions')} d) on {emb.get('provider')} with hedge to "
          f"{emb.get('hedge_provider')}; reranker `{rer.get('model')}` on {rer.get('provider')}; request generator "
          f"`{D.GENERATOR_MODEL}`; simulated user `{D.SIMULATED_USER_MODEL}`; judges `{D.PRIMARY_JUDGE}` (primary) and "
          f"`{D.SECOND_JUDGE}` (second), reasoning effort low.",
          f"- Real spend reported by OpenRouter: **{_usd(spend['total_usd'])} USD** over {spend['calls']} calls (cap "
          f"{D.SPEND_CAP_USD:.2f} USD); the key's own usage counter moved {_usd(_key_usage(spend))} USD between the first "
          "and the last snapshot (it also counts any other use of the key in that window). By stage in "
          "[Cost and time](#cost-and-time).", ""]

    L += ["## Summary", ""] + [f"- {line}" for line in summary_lines(r)] + [""]

    # ------------------------------------------------------------------ sets
    sets = r["sets"]
    counts = (sets.get("counts") or {}).get("counts") or {}
    L += ["## Request sets", ""]
    L += _table(["Set", "Requests", "Origin", "Expected route"],
                [["A", counts.get("A", 0), "V14 stratum A (10 `ambiguous`)", "search"],
                 ["B", counts.get("B", 0), "V14 stratum B", "search"], ["D", counts.get("D", 0), "V14 stratum D", "out of scope"],
                 ["E", counts.get("E", 0), f"generated; needs of B, seed {D.E_SEED}", "clarify"],
                 ["F", counts.get("F", 0), "generated from catalog records", "agent"]])
    gen = sets.get("generator") or {}
    L += [f"E and F were written by `{gen.get('model')}` (temperature {gen.get('temperature')}, reasoning "
          f"{gen.get('reasoning')}, prompt versions {gen.get('prompt_versions')}); prompts, raw outputs and dates are in "
          "`data/evaluation/assistant/sets/generation/`.", ""]
    for name in ("E", "F"):
        part = sets.get(name) or {}
        L += [f"Set {name}: {len(part.get('first_pass_problems') or [])} problems after the first pass, "
              f"{len(part.get('fixes') or [])} replacements, {len(part.get('final_problems') or [])} problems left; "
              f"language verdicts {part.get('language_summary')} (\"undecided\": a short request without function words).", ""]
        for fix in part.get("fixes") or []:
            before = (fix.get("before") or {}).get("text")
            after = (fix.get("after") or {}).get("text")
            L.append(f"- round {fix['round']}, `{fix['request_id']}`: {'; '.join(fix['reasons'])}. `{before}` -> `{after}`")
        L.append("")
    req = sets.get("requests") or {}
    L += ["Set E (vague request, original B query whose need is hidden):", ""]
    L += _table(["Request", "Lang", "Vague request", "Original query"],
                [[k, v["language"], v["text"], v["source_query"]] for k, v in req.items() if v["set"] == "E"])
    L += ["Set F:", ""]
    L += _table(["Request", "Kind", "Lang", "Text"], [[k, v["category"], v["language"], v["text"]]
                                                    for k, v in req.items() if v["set"] == "F"])

    # ------------------------------------------------------------------ runs
    runs = r["runs"]
    depth = runs["depth_check"]
    L += ["## Runs", ""]
    L += [f"- P: {runs['P_texts']} request texts, 100 results each. Assistant turns: {runs['turns']} "
          f"({runs['turns_by_run_set']}); E conversations: {runs['conversations']}; direct agent: {runs['agent_direct']}.",
          f"- Query vectors stored by text: {runs['query_vectors']} ({runs['query_vector_retries']} needed more than one attempt).",
          f"- Depth check: for {depth['same_head_set']}/{depth['texts']} P texts the 30 vector candidates the reranker "
          f"sees are the same set at depth 60 and 200 ({depth['same_head_order']} in the same order; smallest overlap "
          f"{depth['min_overlap']}/30).", ""]
    if system:
        L += ["Configuration evaluated (from the application's settings; no secret):", "", "```json"]
        import json

        L += [json.dumps(system, indent=1, default=str), "```", ""]

    # ------------------------------------------------------------------ routes
    routes = r["routes"]
    L += ["## Routes", ""]
    for run, matrix in routes["matrices"].items():
        L += [f"Route matrix, {run}:", ""]
        cols = ["clarify", "search", "agent", "out_of_scope", "error"]
        L += _table(["Set"] + cols + ["total"], [[g] + [m.get(c, 0) for c in cols] + [m["total"]]
                                                 for g, m in sorted(matrix.items())])
    rows = []
    for run, rates in routes["rates"].items():
        for name in ("clarify_ab", "clarify_a", "clarify_b", "out_of_scope_ab", "out_of_scope_b", "agent_ab", "d_detection"):
            x = rates[name]
            rows.append([run, name, f"{x['count']}/{x['requests']}", _pct(x["rate"])])
    L += ["Route rates:", ""] + _table(["Run", "Measure", "Count", "Rate"], rows)
    L += ["### Section 7 thresholds", ""]
    for run, th in routes["thresholds"].items():
        L += _table(["Run", "Measure", "Value", "Maximum", "Outcome"],
                    [[run, name, f"{row['count']}/{row['requests']} = {_pct(row['value'])}", _pct(row["maximum"]),
                      "met" if row["met"] else "**not met**"] for name, row in th["rows"].items()])
        L += [f"{run}: both thresholds {'met' if th['both_met'] else '**not met**'}."
              + ("" if th["both_met"] else " By the protocol, the routing node's instructions are to be revised and the "
                 "route measure repeated, declaring the change, before the assistant becomes the public page's main "
                 "entry point. That is a separate task; nothing was changed here."), ""]
    if routes.get("clarify_or_out_of_scope_ab_run1"):
        L += ["A and B requests that got a question or an out-of-scope reply (run 1):", ""]
        L += _table(["Request", "Set", "Request text", "Reply"],
                    [[x["request_id"], x["set"], x["text"], (x["question"] or "").replace("|", "/")[:160]]
                     for x in routes["clarify_or_out_of_scope_ab_run1"]])
    L += ["D requests (run 1):", ""] + _table(["Request", "Text", "Route"], [[x["request_id"], x["text"], x["route"]]
                                                                         for x in routes["d"]])
    L += ["Routes forced by code, with their reason:", ""]
    for run, forced in routes["forced"].items():
        L.append(f"- {run}: {forced or 'none'}")
    L += [f"- E, later turns: {routes.get('e_later_turns')}", ""]
    stab = routes.get("stability")
    if stab:
        L += ["### Route stability", "",
              f"A and B were run twice. Same route on {stab['same_route']}/{stab['requests']} requests "
              f"({_pct(stab['agreement'])}); by set: "
              + ", ".join(f"{k} {v['same_route']}/{v['requests']}" for k, v in stab["by_set"].items())
              + f". Changes: {stab['changes'] or 'none'} ({', '.join(stab['changed_requests']) or '-'}).", ""]

    # ------------------------------------------------------------------ contrasts
    L += ["## Contrasts H7-H9", "",
          "One family, two-sided, paired randomisation test (10,000 sign flips, seed 0), 95 % paired bootstrap interval, "
          "Holm correction over the three contrasts, under each judge. A conclusion holds only with the same sign and "
          "significance under both judges.", ""]
    rows = []
    for name in ("H7", "H8", "H9"):
        for label in ("primary", "second"):
            c = r["metrics"].get(label, {}).get("contrasts", {}).get(name)
            if not c:
                continue
            rows.append([name, JUDGE_LABEL[label], c["requests"], _f(c["system_mean"]), _f(c["reference_mean"]),
                         _f(c["difference"], signed=True), _ci(c["ci"], signed=True), _p(c["p"]), _p(c["p_holm"])])
        rows.append([name, "**outcome**", "", "", "", "", "", "", f"**{r['conclusions'].get(name, 'n/a')}**"])
    L += _table(["Id", "Judge", "n", "System", "Reference", "Difference", "95 % CI", "p", "p Holm"], rows)
    L += ["What follows from each (protocol section 7): H7 negative and sustained -> the search route would use the "
          "user's text instead of the rewritten query on single-turn conversations (separate task); H8 not sustained -> "
          "the thesis says the clarifying question showed no measurable gain with this method; H9 informs when the "
          "agent's cost pays off and changes nothing.", ""]
    for label in ("primary", "second"):
        m = r["metrics"].get(label)
        if not m:
            continue
        h7 = m["h7"]
        L += [f"### H7 in detail ({JUDGE_LABEL[label]})", ""]
        L += _table(["Group", "Requests", "Assistant nDCG@10", "P nDCG@10"],
                    [[k, v["requests"], _f(v["assistant"]), _f(v["P"])] for k, v in h7["by_group"].items()])
        so = h7["search_route_only"]
        test = so.get("test") or {}
        L += [f"Only the requests the assistant searched (descriptive): {so['requests']} requests, assistant "
              f"{_f(so['assistant'])}, P {_f(so['P'])}, difference {_f(test.get('difference'), signed=True)} "
              f"{_ci(test.get('ci') or [None, None], signed=True)}, raw p {_p(test.get('p'))}. The condensed query "
              f"differed from the request text on {so.get('query_rewritten')} of them (nDCG@10 changed on "
              f"{so.get('rewritten_changed_ndcg')}, mean difference {_f(so.get('rewritten_mean_difference'), signed=True)}) "
              f"and was the request text itself on {so.get('query_unchanged')}.",
              f"Recommendations on A+B: {h7['recommendations']['replies']} replies with picks; precision (grade >= 2) "
              f"{_f(h7['recommendations']['precision_assistant'])} against P {_f(h7['recommendations']['precision_P_same_k'])} "
              "at the same k.", ""]

    # ------------------------------------------------------------------ E
    L += ["## Set E: vague requests and the simulated user", ""]
    for label in ("primary", "second"):
        e = (r["metrics"].get(label) or {}).get("e")
        if not e:
            continue
        L += [f"{JUDGE_LABEL[label]}: asked in {e['asked']}/{e['conversations']} ({_pct(e['clarification_rate'])}); "
              f"questions per conversation {e['questions']}; assistant turns {e['assistant_turns']}; stop {e['stop']}; "
              f"final route {e['final_route']}.", ""]
        L += _table(["Subset", "n", "Assistant after the conversation", "P on the vague request", "P on the original query (reference)"],
                    [["all", e["conversations"], _f(e["ndcg_final"]["mean"]), _f(e["ndcg_P_vague"]["mean"]), _f(e["ndcg_P_original"]["mean"])],
                     ["asked", e["asked"], _f(e["asked_rows"]["ndcg_final"]["mean"]), _f(e["asked_rows"]["ndcg_P_vague"]["mean"]),
                      _f(e["asked_rows"]["ndcg_P_original"]["mean"])],
                     ["did not ask", e["conversations"] - e["asked"], _f(e["not_asked_rows"]["ndcg_final"]["mean"]),
                      _f(e["not_asked_rows"]["ndcg_P_vague"]["mean"]), _f(e["not_asked_rows"]["ndcg_P_original"]["mean"])]])
        if label == "primary":
            L += _table(["Request", "Lang", "Vague", "Questions", "Final route", "Final query", "nDCG final", "P vague", "P original"],
                        [[x["request_id"], x["language"], x["text"], x["questions"], x["final_route"],
                          (x["final_query"] or "").replace("|", "/"), _f(x["ndcg_final"]), _f(x["ndcg_P_vague"]),
                          _f(x["ndcg_P_original"])] for x in e["rows"]])

    # ------------------------------------------------------------------ F and agent
    L += ["## Set F and the agent", ""]
    for label in ("primary", "second"):
        f = (r["metrics"].get(label) or {}).get("f")
        if not f:
            continue
        L += [f"{JUDGE_LABEL[label]}: routes {f['route']}; {f['with_recommendations']}/{f['requests']} replies with "
              f"recommendations (without: {', '.join(f['without_recommendations']) or 'none'}); works named in the "
              f"requests that were recommended: {f['named_works_recommended']}/{f['named_works']} (in P's top k: "
              f"{f['named_works_in_P_top_k']}); nDCG@10 of the reply's list {_f(f['ndcg_list']['mean'])} against P "
              f"{_f(f['ndcg_P']['mean'])}.", ""]
        L += _table(["By", "Value", "Requests", "Precision of the recommendations", "P at the same k"],
                    [["route", k, v["requests"], _f(v["precision_picks"]), _f(v["precision_P"])] for k, v in f["by_route"].items()]
                    + [["kind", k, v["requests"], _f(v["precision_picks"]), _f(v["precision_P"])] for k, v in f["by_kind"].items()])
    L += ["### The agent called directly (40 requests of A and B, seed 0)", ""]
    for label in ("primary", "second"):
        a = (r["metrics"].get(label) or {}).get("agent_direct")
        if not a:
            continue
        L += [f"{JUDGE_LABEL[label]}: {a['with_recommendations']}/{a['requests']} runs with recommendations "
              f"({a['errors']} failed); precision of the agent's recommendations {_f(a['precision_agent']['mean'])} "
              f"{_ci(a['precision_agent']['ci'])} against P at the same k {_f(a['precision_P_same_k']['mean'])} "
              f"{_ci(a['precision_P_same_k']['ci'])}; on the {a['paired_with_assistant']} requests where the assistant "
              f"(run 1) also recommended: agent {_f(a['precision_agent_paired'])}, assistant "
              f"{_f(a['precision_assistant_paired'])}. Routes the assistant took on these requests: {a['assistant_routes']}; "
              f"number of recommendations: {a['picks']}.", ""]

    # ------------------------------------------------------------------ grounding
    g = r["grounding"]
    L += ["## Grounding", "", "Search route (run 1 and every E turn): picks the answer call proposed and the verify "
          "node kept, why the others were dropped, and citations.", ""]
    L += _table(["Set", "Replies", "No answer output", "Every pick dropped", "Picks proposed", "Kept", "Dropped", "Reasons",
                 "Citations proposed", "Citations of kept picks"],
                [[k, v["replies"], v["replies_without_answer_output"], v["replies_with_every_pick_dropped"],
                  v.get("picks_proposed", 0), v.get("picks_kept", 0), v.get("picks_dropped", 0), v["dropped_reasons"] or "-",
                  v.get("citations_proposed", 0), v.get("citations_kept", 0)] for k, v in g["search_route_by_set"].items()])
    for name, label in (("agent_in_assistant", "Agent route inside the assistant"), ("agent_direct", "Agent called directly")):
        a = g[name]
        L.append(f"- {label}: {a['runs']} runs, {a['recommended']} recommendations kept, {a['dropped_not_from_tools']} "
                 f"recommended works that no tool had returned (dropped), {a['runs_with_reflection']} runs needed the "
                 f"reflection step; tool calls {a['tool_calls']}, per run {_describe(a['tool_calls_per_run'], 1)}.")
    L.append("")

    # ------------------------------------------------------------------ language
    lang = r["language"]
    L += ["## Language of the reply", "",
          "Run 1 and every E turn. The catalog's language detector decides (twelve languages, 40 letters or more); a "
          "reply it cannot decide is judged by its writing system for Russian, Chinese, Japanese and Arabic and is \"undecided\" otherwise. The fixed replies (out of scope, no results, results without "
          "picks) exist only in Spanish and English.", ""]
    L += _table(["Request language", "ok", "mismatch", "undecided"],
                [[k, v.get("ok", 0), v.get("mismatch", 0), v.get("undecided", 0)] for k, v in lang["by_language"].items()])
    L += _table(["Route", "ok", "mismatch", "undecided"],
                [[k, v.get("ok", 0), v.get("mismatch", 0), v.get("undecided", 0)] for k, v in lang["by_route"].items()])
    L += _table(["Text of the reply", "ok", "mismatch", "undecided"],
                [[k, v.get("ok", 0), v.get("mismatch", 0), v.get("undecided", 0)] for k, v in lang["by_text"].items()])
    L += ["Replies written by the LLM only, by request language:", ""]
    L += _table(["Request language", "ok", "mismatch", "undecided"],
                [[k, v.get("ok", 0), v.get("mismatch", 0), v.get("undecided", 0)]
                 for k, v in lang["llm_text_by_language"].items()])
    L += [f"Mismatches (request -> detected): {lang['mismatches_detected_as'] or 'none'}.", ""]

    # ------------------------------------------------------------------ latency and cost
    ct = r["cost_time"]
    L += ["## Latency and cost per turn", "",
          "Wall time of `reply()` from this laptop, four requests in flight; cost as reported by OpenRouter for the calls "
          "recorded in the request's thread.", ""]
    rows = []
    for key, v in list(ct["by_route"].items()) + [("all turns", ct["all_turns"] or {})] + [("direct agent", ct["agent_direct"] or {})]:
        if not v:
            continue
        rows.append([key, v.get("turns"), _describe(v.get("wall_s"), 1), _describe(v.get("cost_usd"), 5),
                     _describe(v.get("llm_calls"), 1)])
    L += _table(["Route", "Turns", "Seconds median / p95", "USD median / p95", "LLM calls median / p95"], rows)
    L += _table(["Set", "Turns", "Seconds median / p95", "USD median / p95"],
                [[k, v["turns"], _describe(v["wall_s"], 1), _describe(v["cost_usd"], 5)] for k, v in ct["by_set"].items()])
    L += ["Median seconds per node, by route:", ""]
    for route, nodes in ct["node_seconds_by_route"].items():
        L.append(f"- {route}: " + ", ".join(f"{n} {_f(d.get('median'), 2)}" for n, d in nodes.items()))
    L += ["", f"LLM calls: {ct['llm_calls']}, failed {ct['llm_failed']}; embedding and rerank calls failed: "
          f"{ct['api_failed']}; hedged embedding requests cancelled because the other provider answered first: "
          f"{ct['api_cancelled']} (in the request's thread). Every call the ledger records as not completed, by stage: "
          f"{ct['ledger_errors']}. `cancelled` is the losing leg of a hedged query embedding, not a failure; "
          "`LengthFinishReasonError` of the system LLM is the answer call of the search route reaching its 1024-token "
          "cap (in 4 of the 5 the provider counted 773 to 1024 of them as reasoning tokens although reasoning is off; read from "
          "`assistant.log`), after which the reply "
          "keeps the results and drops the picks (`assistant_answer_failed`); of the primary judge, a judgment retried "
          "in a later pass.", ""]

    # ------------------------------------------------------------------ failures
    fl = r["failures"]
    L += ["## Failures", ""]
    for k, v in fl.items():
        L.append(f"- {k.replace('_', ' ')}: {v if v not in ([], {}, 0) else 'none'}")
    L.append("")

    # ------------------------------------------------------------------ judgments
    j = r["judgments"]
    L += ["## Judgments", "", f"Needs with measured pairs: {j['pairs_file_needs']}.", ""]
    if j.get("counts"):
        L += _table(["Judge", "Measured pairs", "From V14", "Judged here", "Unjudged"],
                    [[m, c["measured_pairs"], c["from_v14"], c["judged_here"], c["unjudged"]] for m, c in j["counts"].items()])
    for run in j.get("runs") or []:
        L.append(f"- {run.get('model')}: {run.get('pairs')} pairs to judge, {run.get('judged_in_run')} judged in the run, "
                 f"{run.get('failed_calls_in_run')} failed calls, {run.get('unjudged')} left, {run.get('date')}")
    L.append("")

    # ------------------------------------------------------------------ cost
    L += ["## Cost and time", ""]
    L += _table(["Stage", "Calls", "Calls without cost", "Failed", "USD", "Wall-clock s"],
                [[k, v["calls"], v["calls_without_cost"], v["failed_calls"], _usd(v["cost_usd"]),
                  _f(r["stage_seconds"].get(k), 0)] for k, v in sorted(spend["by_stage"].items())])
    L += _table(["Model", "Calls", "USD"], [[k, v["calls"], _usd(v["cost_usd"])] for k, v in sorted(spend["by_model"].items())])
    L += [f"Stage wall-clock (seconds, every run of the stage summed): {r['stage_seconds']}.", ""]

    L += ["## Deviations from the protocol", ""] + [f"- {d}" for d in DEVIATIONS] + [""]
    L += ["## Limits", ""] + [f"- {x}" for x in LIMITS] + [""]
    L += ["## Files", "",
          "- Code: `app/evaluation/assistant_evaluation/`; tests: `tests/evaluation/test_assistant_evaluation.py`.",
          "- Data under `data/evaluation/assistant/`: `sets/` (requests, checks, generator calls), `runs/` (`P.jsonl`, "
          "`depth_check.jsonl`, `turns.jsonl`, `conversations.jsonl`, `agent_direct.jsonl`, `simulated_user/`), "
          "`query_vectors/`, `judgments/` (`pairs.json`, the two caches of this task, `judging_runs.jsonl`), "
          "`system.json`, `spend_ledger.jsonl`, `stage_times.jsonl`, `results.json`, `per_request.json`, the stage logs.",
          "- Regenerate this report: `.venv/bin/python -m app.evaluation.assistant_evaluation report --offline`.", ""]
    return "\n".join(L)


def _describe(d: Optional[Mapping[str, Any]], digits: int) -> str:
    if not d or not d.get("n"):
        return "n/a"
    return f"{_f(d['median'], digits)} / {_f(d['p95'], digits)}"
