"""
Markdown report of the re-measurement (V25 addendum), side by side with the
first measurement (V22), rendered from the two results documents, the
instructions record and the two stage logs only: no API call, no database
connection.

The first measurement's files are only read. Its own report
(ASSISTANT_EVALUATION.md) is still written by report.build_markdown; this
module writes ASSISTANT_EVALUATION_V25.md.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app.evaluation.assistant_evaluation import analysis as A
from app.evaluation.assistant_evaluation import design as D
from app.evaluation.assistant_evaluation.report import JUDGE_LABEL, _recorded, _ci, _describe, _f, _key_usage, _p, _pct, _table, _usd

M1, M2 = "first (V22)", "second (V25)"
GROUPS = ("A ambiguous", "A other", "B", "D", "E first turn", "F", "G valid", "G out of scope")
LOG_FILE = "assistant.log"
_LENGTH = re.compile(r"length limit was reached.*?completion_tokens=(\d+).*?reasoning_tokens=(\d+)")

DEVIATIONS: List[str] = [
    "Where the reasoning defect came from was found by replaying the five answer calls of V22 that hit their cap "
    "(their books read from the database, read only) and then one of them on each OpenRouter provider of the model in "
    "turn; the diagnostic calls that answered are in the ledger as stage `0-diagnosis` (calls refused with HTTP 404 "
    "or 429 by a provider cost nothing and are not listed). They were made before the revised instruction was "
    "committed and do not involve the routing instruction. The key's usage counter is read only around the stages "
    "run through the command line, so it does not cover them.",
    "Reuse. The second measurement copies the request sets A-F, P (no LLM; the retrieval code is the same at both "
    "commits), the depth check and the query vectors from the first one, and reads the first measurement's judgment "
    "caches; only pairs not judged before were judged. The simulated user's calls were not reused (they depend on the "
    "assistant's question), so set E was played again with the same simulated user and prompt.",
    "Set G is measured by its route only, as the addendum's thresholds need; no P, judgments or quality measure was "
    "computed for it. G was run once, not twice.",
    "Set G checks, not fixed by the addendum: language (V14's rule), form (keywords: at most 4 words and no final "
    "punctuation; sentence: at least 5 words), no duplicate within G or with A-F, and, read on the English translation "
    "the generator gives with each request: a valid request shares a content word with its topic and a sentence asks "
    "for something to read (book, guide, manual, ...); an out-of-scope request mentions no book or reading word and "
    "reads as its kind (a time word for present facts, a digit for calculations, a programming word for programs). "
    "Languages and forms were assigned to topics in code with seed 0 (every language: five keyword and five sentence "
    "requests). Some flags were stricter than needed (\"ouvrage\", \"Anleitung\", \"testo\", \"publicação\" are not in "
    "the reading-word list; `viajar a Marruecos` read as Portuguese); every replacement is listed.",
    "The required thresholds are those of the addendum's section 4: out of scope on the 60 valid G requests "
    f"(at most {D.OOS_G_MAX:.0%}) and questions on B (at most {D.CLARIFY_B_MAX:.0%}), both on run 1 of the second "
    "measurement. A and B rates are shown for both runs as secondary data.",
    "The evaluation code that runs and reports the second measurement (set G, the re-measurement set-up, this report) "
    "is not committed; the system it measures is at the commits recorded per turn, whose application code is that of "
    "the two V25 commits.",
    "V14's generator class labels its ledger entries with the stage `1-queries`; for set G that label is stage 1 too.",
    "nDCG@10 is computed in each measurement against the union of V14's pool and the pairs judged in that measurement, "
    "so P's value on the same request can differ slightly between the two columns when the second measurement judged "
    "new works for its need.",
    "Before the full run, every part of stage 3 was run on its first two requests as a smoke test with the same code; "
    "those turns are part of the data.",
]


def _counts(matrix: Optional[Mapping[str, Any]], group: str, route: str) -> str:
    row = (matrix or {}).get(group)
    return "-" if row is None else str(row.get(route, 0))


def _pair(a: str, b: str) -> str:
    return a if a == b else f"{a} -> {b}"


def _rate(x: Optional[Mapping[str, Any]]) -> str:
    if not x:
        return "n/a"
    return f"{x['count']}/{x['requests']} = {_pct(x['rate'])}"


def _met(row: Mapping[str, Any]) -> str:
    return "met" if row["met"] else "**not met**"


def length_cap_events(data_dir: Path) -> Dict[str, Any]:
    """
    Answer calls that reached their token cap, read from the assistant stage's
    log of data_dir: how many, and the reasoning tokens the provider counted.
    """
    path = Path(data_dir) / LOG_FILE
    events: List[Tuple[int, int]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _LENGTH.search(line)
            if match:
                events.append((int(match.group(1)), int(match.group(2))))
    return {"events": len(events), "with_reasoning": sum(1 for _, r in events if r > 0),
            "reasoning_tokens": [r for _, r in events], "log": str(path) if path.exists() else None}


def _instruction_lines(instructions: Mapping[str, Any]) -> List[str]:
    lines = []
    for key, label in (("first", M1), ("second", M2)):
        x = instructions[key]
        oos = next((line for line in x["route_lines"] if line.startswith('- "out_of_scope"')), "")
        lines += [f"- {label}: prompts `{x['prompts_version']}` at commit `{x['commit']}` (routing instruction sha256 "
                  f"`{x['sha256']}`). The `out_of_scope` line:", "", f"  > {oos}", ""]
    first, second = instructions["first"]["router_system_prompt"], instructions["second"]["router_system_prompt"]
    changed = [line for line in second.splitlines() if line not in first.splitlines()]
    lines.append(f"Lines of the routing instruction that differ: {len(changed)} (the `out_of_scope` line only); "
                 "everything else in the prompts, the graph and the code rules is the same.")
    return lines + [""]


def _contrast_cell(c: Optional[Mapping[str, Any]]) -> str:
    if not c:
        return "n/a"
    return f"{_f(c['difference'], signed=True)} {_ci(c['ci'], signed=True)} p {_p(c['p_holm'])}"


def _changes(r1: Mapping[str, Any], r2: Mapping[str, Any]) -> List[Tuple[str, str, str, bool]]:
    """(measure, first, second, worse) for the measures with a direction; worse is mechanical."""
    out = []
    rates1, rates2 = r1["routes"]["rates"].get(D.RUN_1, {}), r2["routes"]["rates"].get(D.RUN_1, {})
    for key, higher_is_worse in (("out_of_scope_ab", True), ("clarify_ab", True), ("clarify_b", True),
                                 ("d_detection", False)):
        a, b = rates1.get(key, {}).get("rate"), rates2.get(key, {}).get("rate")
        if a is not None and b is not None:
            out.append((f"run 1 {key}", _rate(rates1[key]), _rate(rates2[key]), (b > a) if higher_is_worse else (b < a)))
    for label in ("primary", "second"):
        for name in ("H7", "H8", "H9"):
            c1 = r1["metrics"][label]["contrasts"].get(name)
            c2 = r2["metrics"][label]["contrasts"].get(name)
            if c1 and c2:
                out.append((f"{name} system mean ({label})", _f(c1["system_mean"]), _f(c2["system_mean"]),
                            c2["system_mean"] < c1["system_mean"] - 1e-9))
    e1, e2 = r1["metrics"]["primary"]["e"], r2["metrics"]["primary"]["e"]
    out.append(("E asked (primary)", f"{e1['asked']}/{e1['conversations']}", f"{e2['asked']}/{e2['conversations']}",
                False))
    f1, f2 = r1["metrics"]["primary"]["f"], r2["metrics"]["primary"]["f"]
    out.append(("F agent route", str(f1["route"].get("agent", 0)), str(f2["route"].get("agent", 0)),
                f2["route"].get("agent", 0) < f1["route"].get("agent", 0)))
    g1, g2 = r1["grounding"]["search_route_by_set"]["all"], r2["grounding"]["search_route_by_set"]["all"]
    d1, d2 = g1.get("picks_dropped", 0), g2.get("picks_dropped", 0)
    p1, p2 = g1.get("picks_proposed", 0), g2.get("picks_proposed", 0)
    out.append(("search picks dropped by verify", f"{d1}/{p1}", f"{d2}/{p2}", d2 / max(p2, 1) > d1 / max(p1, 1)))
    out.append(("answer calls without output", str(r1["failures"]["answer_errors"]), str(r2["failures"]["answer_errors"]),
                r2["failures"]["answer_errors"] > r1["failures"]["answer_errors"]))
    for route in ("search", "agent"):
        a = (r1["cost_time"]["by_route"].get(route) or {}).get("wall_s") or {}
        b = (r2["cost_time"]["by_route"].get(route) or {}).get("wall_s") or {}
        if a.get("median") is not None and b.get("median") is not None:
            out.append((f"{route} turn seconds, median", _f(a["median"], 1), _f(b["median"], 1),
                        b["median"] > a["median"] * 1.10))
    return out


def build_comparison(r1: Mapping[str, Any], r2: Mapping[str, Any], instructions: Mapping[str, Any],
                     first_dir: Path, second_dir: Path) -> str:
    v = r2["v25"]
    g = v["g"]
    req = v["thresholds"]["required"]
    out: List[str] = ["# Re-measurement of the assistant after revising the routing instruction (V25)", ""]
    commits2 = ", ".join(f"`{k}` x{n}" for k, n in sorted(r2["commits"].items(), key=lambda x: -x[1]))
    commits1 = ", ".join(f"`{k}` x{n}" for k, n in sorted(r1["commits"].items(), key=lambda x: -x[1]))
    out += [
        f"- Contract: the addendum `{v['addendum']['path']}`, committed in `{v['addendum']['commit']}` before the "
        f"instruction changed, completing the protocol `{r1['protocol']['path']}` (`{r1['protocol']['commit']}`).",
        f"- {M1}: `data/evaluation/assistant/`, report `data/evaluation/ASSISTANT_EVALUATION.md`, commits recorded "
        f"per run {commits1}; its files are unchanged.",
        f"- {M2}: `data/evaluation/assistant_v25/`, commits recorded per run {commits2} (the "
        f"{r2['runs']['P_texts']} P searches are the first measurement's, copied); dates "
        f"{(r2.get('dates') or ['?', '?'])[0]} to {(r2.get('dates') or ['?', '?'])[1]}.",
        "- **Relevance judgments come from LLM judges only**; the user of set E is simulated; sets E, F and G were "
        "written by an LLM.",
        f"- Real spend of the second measurement reported by OpenRouter: **{_usd(r2['spend']['total_usd'])} USD** "
        f"(cap {v['spend_cap_usd']:.2f} USD); the key's usage counter moved {_usd(_key_usage(r2['spend']))} USD between "
        "its first and last snapshot.",
        "- Regenerate both reports offline: `.venv/bin/python -m app.evaluation.assistant_evaluation report --offline`.",
        "",
        "## The two instructions", "",
    ]
    out += _instruction_lines(instructions)
    s1, s2 = r1.get("system") or {}, r2.get("system") or {}
    body1, body2 = s1.get("llm_openrouter_extra_body"), s2.get("llm_openrouter_extra_body")
    out += [f"LLM request fields: {M1} reasoning `{_recorded(s1.get('llm') or {}, 'reasoning')}`, request body "
            f"{'`' + str(body1) + '`' if 'llm_openrouter_extra_body' in s1 else 'not recorded'}; "
            f"{M2} reasoning `{_recorded(s2.get('llm') or {}, 'reasoning')}`, request body "
            f"{'`' + str(body2) + '`' if 'llm_openrouter_extra_body' in s2 else 'not recorded'} "
            "(the answer-call defect of the addendum's section 5, below).",
            ""]

    # --- outcome --------------------------------------------------------------------
    out += ["## Outcome", ""]
    out += _table(["Threshold (addendum, section 4)", "Value", "Maximum", "Outcome"], [
        ["out of scope on the 60 valid G requests", _rate({"count": req["out_of_scope_g_valid"]["count"],
                                                           "requests": req["out_of_scope_g_valid"]["requests"],
                                                           "rate": req["out_of_scope_g_valid"]["value"]}),
         _pct(req["out_of_scope_g_valid"]["maximum"]), _met(req["out_of_scope_g_valid"])],
        ["questions on B (run 1)", _rate({"count": req["clarify_b"]["count"], "requests": req["clarify_b"]["requests"],
                                          "rate": req["clarify_b"]["value"]}),
         _pct(req["clarify_b"]["maximum"]), _met(req["clarify_b"])],
    ])
    det = g["out_of_scope"]["detection"]
    out.append(f"Detection on the 20 out-of-scope G requests (reported, no threshold): {_rate(det)}.")
    out.append("")
    if v["thresholds"]["required_met"]:
        out.append("Both required thresholds are **met**.")
    else:
        out.append("The required thresholds are **not met**. By the addendum, the instruction is not touched again in "
                   "this task and the assistant is not presented as the public page's main entry point.")
    out.append("")
    sec = v["thresholds"]["secondary"]
    rows = []
    for run in (D.RUN_1, D.RUN_2):
        for key in ("out_of_scope_ab", "clarify_b"):
            t1 = ((r1["routes"]["thresholds"].get(run) or {}).get("rows") or {}).get(key)
            t2 = ((sec.get(run) or {}).get("rows") or {}).get(key)
            if t1 and t2:
                rows.append([run, key, f"{t1['count']}/{t1['requests']} = {_pct(t1['value'])} ({'met' if t1['met'] else 'not met'})",
                             f"{t2['count']}/{t2['requests']} = {_pct(t2['value'])} ({'met' if t2['met'] else 'not met'})",
                             _pct(t1["maximum"])])
    out += ["Secondary (A and B; the revised instruction was written knowing the first measurement's errors on these "
            "requests, so they are not an independent test):", ""]
    out += _table(["Run", "Measure", M1, M2, "Maximum"], rows)

    # --- set G --------------------------------------------------------------------------
    gc = v["g_checks"]
    gen = gc.get("generator") or {}
    out += ["## Set G (held-out)", "",
            f"Written by `{gen.get('model')}` (temperature {gen.get('temperature')}, reasoning {gen.get('reasoning')}, "
            f"prompt `{gen.get('prompt_version')}`, provider {gen.get('provider')}, {gen.get('date')}), after the "
            f"instruction's commit `{D.INSTRUCTION_COMMIT}`, from the topic list in `sets/g_topics.json` (also in "
            "`set_g.py`); the generator saw only that list and the task's rules, never the routing instruction or a "
            "system output. Prompt and raw outputs: `sets/generation/G*.json`; checks: `sets/g_checks.json`.", "",
            f"Checks: {len(gc.get('first_pass_problems') or [])} problems after the first pass, "
            f"{len(gc.get('fixes') or [])} replacements, {len(gc.get('final_problems') or [])} problems left; language "
            f"verdicts {gc.get('language_summary')} (\"undecided\": keywords without function words).", ""]
    for fix in gc.get("fixes") or []:
        out.append(f"- round {fix['round']}, `{fix['request_id']}`: {'; '.join(fix['reasons'])}. "
                   f"`{(fix.get('before') or {}).get('text')}` -> `{(fix.get('after') or {}).get('text')}`")
    out.append("")
    vr = g["valid"]
    out += _table(["Valid G requests (60)", "Count", "Rate"],
                  [[route, vr[route]["count"], _pct(vr[route]["rate"])] for route in A.ROUTES + (A.ERROR_ROUTE,)])
    for title, key in (("By form", "valid_by_form"), ("By language", "valid_by_language"), ("By field", "valid_by_field")):
        out += _table([title, *A.ROUTES], [[k, *[x.get(r, 0) for r in A.ROUTES]] for k, x in g[key].items()])
    out += _table(["Out-of-scope G requests, by kind", *A.ROUTES],
                  [[k, *[x.get(r, 0) for r in A.ROUTES]] for k, x in g["out_of_scope_by_kind"].items()])
    for title, key in (("Valid requests routed out of scope", "valid_refused"),
                       ("Valid requests that got a question", "valid_clarified"),
                       ("Out-of-scope requests not detected", "out_of_scope_missed")):
        items = g[key]
        out.append(f"{title}: {len(items)}.")
        out.append("")
        if items:
            out += _table(["Request", "Lang", "Field or kind", "Text", "Route", "Router's reason"],
                          [[x["request_id"], x["language"], x["field"], x["text"], x["route"],
                            (x.get("reason") or "")[:160]] for x in items])
    out += ["All G requests:", ""]
    out += _table(["Request", "Group", "Lang", "Field or kind", "Form", "Text", "Route"],
                  [[x["request_id"], x["group"], x["language"], x["field"], x["form"], x["text"], x["route"]]
                   for x in g["rows"]])

    # --- routes -----------------------------------------------------------------------------
    out += ["## Routes, both measurements", "",
            f"Each cell: {M1} -> {M2} (a single number when equal). Run 1:", ""]
    m1, m2 = r1["routes"]["matrices"], r2["routes"]["matrices"]
    for run in (D.RUN_1, D.RUN_2):
        rows = []
        for group in GROUPS:
            if group not in (m1.get(run) or {}) and group not in (m2.get(run) or {}):
                continue
            rows.append([group, *[_pair(_counts(m1.get(run), group, route), _counts(m2.get(run), group, route))
                                  for route in A.ROUTES + (A.ERROR_ROUTE, "total")]])
        if run == D.RUN_2:
            out += ["Run 2 (A and B only):", ""]
        out += _table(["Set", *A.ROUTES, A.ERROR_ROUTE, "total"], rows)
    rows = []
    for run in (D.RUN_1, D.RUN_2):
        for key in ("clarify_ab", "clarify_a", "clarify_b", "out_of_scope_ab", "out_of_scope_b", "agent_ab", "d_detection"):
            a = (r1["routes"]["rates"].get(run) or {}).get(key)
            b = (r2["routes"]["rates"].get(run) or {}).get(key)
            if (a and a["requests"]) or (b and b["requests"]):
                rows.append([run, key, _rate(a), _rate(b)])
    out += _table(["Run", "Measure", M1, M2], rows)
    for label, r in ((M1, r1), (M2, r2)):
        stab = r["routes"].get("stability")
        if stab:
            out.append(f"- Route stability, {label}: same route on {stab['same_route']}/{stab['requests']} "
                       f"({_pct(stab['agreement'])}); changes {stab.get('changes')}.")
        out.append(f"- Routes forced by code, {label}: {r['routes']['forced']}; E later turns {r['routes']['e_later_turns']}.")
    out.append("")
    out += [f"A and B requests that got a question or an out-of-scope reply ({M2}, run 1):", ""]
    out += _table(["Request", "Set", "Request text", "Reply"],
                  [[x["request_id"], x["set"], x["text"], (x["question"] or "")[:120]]
                   for x in r2["routes"]["clarify_or_out_of_scope_ab_run1"]])
    out += [f"D requests, {M2}: " + ", ".join(f"{x['request_id']} {x['route']}" for x in r2["routes"]["d"]), ""]

    # --- contrasts -----------------------------------------------------------------------------
    out += ["## Contrasts H7-H9, both measurements", "",
            "Same test as the first report (paired randomisation test, 10,000 sign flips, seed 0, 95 % paired "
            "bootstrap interval, Holm over the three contrasts, under each judge). Cell: difference [CI] Holm p.", ""]
    rows = []
    for name in ("H7", "H8", "H9"):
        for label in ("primary", "second"):
            c1 = r1["metrics"][label]["contrasts"].get(name)
            c2 = r2["metrics"][label]["contrasts"].get(name)
            rows.append([name, JUDGE_LABEL[label], f"{_f((c1 or {}).get('system_mean'))} / {_f((c1 or {}).get('reference_mean'))}",
                         _contrast_cell(c1), f"{_f((c2 or {}).get('system_mean'))} / {_f((c2 or {}).get('reference_mean'))}",
                         _contrast_cell(c2)])
        rows.append([name, "**outcome**", "", f"**{r1['conclusions'].get(name)}**", "", f"**{r2['conclusions'].get(name)}**"])
    out += _table(["Id", "Judge", f"{M1} system / P", f"{M1} difference", f"{M2} system / P", f"{M2} difference"], rows)
    for label in ("primary", "second"):
        h1, h2 = r1["metrics"][label]["h7"], r2["metrics"][label]["h7"]
        out += [f"H7 by group, {JUDGE_LABEL[label]} (assistant / P):", ""]
        groups = sorted(set(h1["by_group"]) | set(h2["by_group"]))
        out += _table(["Group", f"{M1} n", f"{M1}", f"{M2} n", f"{M2}"],
                      [[k, (h1["by_group"].get(k) or {}).get("requests", 0),
                        f"{_f((h1['by_group'].get(k) or {}).get('assistant'))} / {_f((h1['by_group'].get(k) or {}).get('P'))}",
                        (h2["by_group"].get(k) or {}).get("requests", 0),
                        f"{_f((h2['by_group'].get(k) or {}).get('assistant'))} / {_f((h2['by_group'].get(k) or {}).get('P'))}"]
                       for k in groups])
        for lab, h in ((M1, h1), (M2, h2)):
            so = h["search_route_only"]
            out.append(f"- {lab}: on the {so['requests']} requests the assistant searched, assistant {_f(so['assistant'])}, "
                       f"P {_f(so['P'])}; recommendations on A+B: {h['recommendations']['replies']} replies, precision "
                       f"{_f(h['recommendations']['precision_assistant'])} against P {_f(h['recommendations']['precision_P_same_k'])}.")
        out.append("")

    # --- E -------------------------------------------------------------------------------------
    out += ["## Set E: how often it asked, and quality with and without asking", ""]
    rows = []
    for label in ("primary", "second"):
        for lab, r in ((M1, r1), (M2, r2)):
            e = r["metrics"][label]["e"]
            rows.append([JUDGE_LABEL[label], lab, f"{e['asked']}/{e['conversations']}",
                         f"{_f(e['ndcg_final']['mean'])} / {_f(e['ndcg_P_vague']['mean'])}",
                         f"{e['asked_rows']['ndcg_final']['n']}: {_f(e['asked_rows']['ndcg_final']['mean'])} / "
                         f"{_f(e['asked_rows']['ndcg_P_vague']['mean'])}",
                         f"{e['not_asked_rows']['ndcg_final']['n']}: {_f(e['not_asked_rows']['ndcg_final']['mean'])} / "
                         f"{_f(e['not_asked_rows']['ndcg_P_vague']['mean'])}",
                         _f(e["ndcg_P_original"]["mean"])])
    out += _table(["Judge", "Measurement", "Asked", "All: assistant / P vague", "Asked: n: assistant / P vague",
                   "Did not ask: n: assistant / P vague", "P on the original query"], rows)
    for lab, r in ((M1, r1), (M2, r2)):
        e = r["metrics"]["primary"]["e"]
        out.append(f"- {lab}: questions per conversation {e['questions']}; stop {e['stop']}; final route {e['final_route']}.")
    out.append("")

    # --- F and the agent -----------------------------------------------------------------------
    out += ["## Set F and the agent", ""]
    rows = []
    for label in ("primary", "second"):
        for lab, r in ((M1, r1), (M2, r2)):
            f = r["metrics"][label]["f"]
            a = r["metrics"][label]["agent_direct"]
            rows.append([JUDGE_LABEL[label], lab, f["route"], f"{f['with_recommendations']}/{f['requests']}",
                         f"{f['named_works_recommended']}/{f['named_works']}",
                         " ".join(f"{k} {_f(x['precision_picks'])}/{_f(x['precision_P'])}" for k, x in sorted(f["by_kind"].items())),
                         f"{a['with_recommendations']}/{a['requests']}: {_f(a['precision_agent']['mean'])} / "
                         f"{_f(a['precision_P_same_k']['mean'])}"])
    out += _table(["Judge", "Measurement", "F routes", "F with picks", "Named works recommended",
                   "F precision by kind (picks / P)", "Direct agent: precision / P"], rows)

    # --- grounding ------------------------------------------------------------------------------
    out += ["## Grounding", ""]
    rows = []
    for lab, r in ((M1, r1), (M2, r2)):
        for name, x in r["grounding"]["search_route_by_set"].items():
            rows.append([lab, name, x["replies"], x["replies_without_answer_output"], x["replies_with_every_pick_dropped"],
                         x.get("picks_proposed", 0), x.get("picks_kept", 0), x.get("picks_dropped", 0),
                         x["dropped_reasons"] or "-"])
    out += _table(["Measurement", "Set", "Replies", "No answer output", "Every pick dropped", "Picks proposed", "Kept",
                   "Dropped", "Reasons"], rows)
    for lab, r in ((M1, r1), (M2, r2)):
        for key, name in (("agent_in_assistant", "agent route"), ("agent_direct", "agent called directly")):
            x = r["grounding"][key]
            out.append(f"- {lab}, {name}: {x['runs']} runs, {x['recommended']} recommendations kept, "
                       f"{x['dropped_not_from_tools']} recommended works no tool returned (dropped), "
                       f"{x['runs_with_reflection']} runs with the reflection step; tool calls {x['tool_calls']}.")
    out.append("")

    # --- the answer-call defect ------------------------------------------------------------------
    l1, l2 = length_cap_events(first_dir), length_cap_events(second_dir)
    out += ["## Answer calls that reached their token cap (addendum, section 5)", "",
            "Cause, found before the instruction changed: the request asked for reasoning off with "
            "`{\"enabled\": false}`, which OpenRouter forwards, but one provider of `deepseek/deepseek-v4.1-flash`, "
            "Novita, keeps reasoning whatever the request says (also under `{\"effort\": \"none\"}`, both together, "
            "`{\"max_tokens\": 0}` and the top-level `reasoning_effort`), and OpenRouter routes some calls to it. "
            "Fix, in the client: with reasoning off, `provider.ignore = [\"novita\"]`. The 1024-token cap is unchanged.",
            ""]
    out += _table(["Measurement", "Answer calls at the cap", "with reasoning tokens", "reasoning tokens", "answer errors"],
                  [[M1, l1["events"], l1["with_reasoning"], l1["reasoning_tokens"], r1["failures"]["answer_errors"]],
                   [M2, l2["events"], l2["with_reasoning"], l2["reasoning_tokens"], r2["failures"]["answer_errors"]]])

    # --- latency and cost --------------------------------------------------------------------------
    out += ["## Latency and cost per route", "",
            "Wall time of `reply()` from this laptop, four requests in flight; cost reported by OpenRouter for the "
            "calls in the request's thread. Set G is shown apart.", ""]
    rows = []
    for route in ("agent", "clarify", "out_of_scope", "search", "all turns", "direct agent"):
        cells = []
        for r in (r1, r2):
            ct = r["cost_time"]
            d = ct["all_turns"] if route == "all turns" else ct["agent_direct"] if route == "direct agent" else ct["by_route"].get(route)
            cells += [str((d or {}).get("turns", 0)), _describe((d or {}).get("wall_s"), 1), _describe((d or {}).get("cost_usd"), 5)]
        rows.append([route, *cells])
    out += _table(["Route", f"{M1} turns", f"{M1} s median / p95", f"{M1} USD median / p95",
                   f"{M2} turns", f"{M2} s median / p95", f"{M2} USD median / p95"], rows)
    out += _table(["G route", "Turns", "s median / p95", "USD median / p95"],
                  [[route, d.get("turns", 0), _describe(d.get("wall_s"), 1), _describe(d.get("cost_usd"), 5)]
                   for route, d in sorted(v["g_cost_time"]["by_route"].items())])
    for lab, r in ((M1, r1), (M2, r2)):
        ct = r["cost_time"]
        out.append(f"- {lab}: LLM calls {ct['llm_calls']}, failed {ct['llm_failed']}; embedding and rerank calls failed "
                   f"{ct['api_failed']}; ledger errors {ct['ledger_errors']}.")
    out.append(f"- {M2}, set G: LLM calls failed {v['g_cost_time']['llm_failed']}; turns without reply "
               f"{v['g_cost_time']['turns_without_reply'] or 'none'}.")
    out.append("")

    # --- failures ----------------------------------------------------------------------------------
    out += ["## Failures", ""]
    keys = ("turns_without_reply", "turns_retried", "notices", "degraded_searches", "searches_not_reranked",
            "router_errors", "answer_errors", "agent_direct_errors", "P_not_reranked_or_degraded", "P_retried")
    out += _table(["Failure", M1, M2], [[k, r1["failures"][k] or "none", r2["failures"][k] or "none"] for k in keys])

    # --- what changed ---------------------------------------------------------------------------------
    out += ["## What changed", "",
            "Measures with a direction, run 1; \"worse\" is applied mechanically (a higher out-of-scope or question rate, "
            "a lower detection rate or system mean, more dropped picks or answer errors, a median turn more than 10 % "
            "slower).", ""]
    out += _table(["Measure", M1, M2, "Worse"], [[m, a, b, "**yes**" if w else "no"] for m, a, b, w in _changes(r1, r2)])

    # --- judgments and spend ------------------------------------------------------------------------
    out += ["## Judgments", ""]
    rows = []
    for lab, r in ((M1, r1), (M2, r2)):
        for model, c in ((r["judgments"].get("counts") or {}).items()):
            rows.append([lab, model, c["measured_pairs"], c["from_v14"], c.get("from_first_measurement", "-"),
                         c["judged_here"], c["unjudged"]])
    out += _table(["Measurement", "Judge", "Measured pairs", "From V14", "From the first measurement", "Judged here",
                   "Unjudged"], rows)
    out += ["## Spend of the second measurement", ""]
    spend = r2["spend"]
    out += _table(["Stage", "Calls", "Calls without cost", "Failed", "USD", "Wall-clock s"],
                  [[k, x["calls"], x["calls_without_cost"], x["failed_calls"], _usd(x["cost_usd"]),
                    _f(r2["stage_seconds"].get(k), 0)] for k, x in sorted(spend["by_stage"].items())])
    out += _table(["Model", "Calls", "USD"],
                  [[k, x["calls"], _usd(x["cost_usd"])] for k, x in sorted(spend["by_model"].items())])
    out += [f"First measurement, for reference: {_usd(r1['spend']['total_usd'])} USD.", ""]

    out += ["## Deviations from the addendum", ""] + [f"- {x}" for x in DEVIATIONS] + [""]
    out += ["## Files", "",
            "- Code: `app/evaluation/assistant_evaluation/` (`set_g.py`, `remeasure.py`, `report_v25.py` and the V25 "
            "parts of the other modules); tests: `tests/evaluation/test_assistant_evaluation.py`.",
            "- Data under `data/evaluation/assistant_v25/`: `instructions.json`, `sets/` (requests with G, "
            "`g_topics.json`, `g_checks.json`, `generation/`), `runs/`, `query_vectors/`, `judgments/`, `system.json`, "
            "`spend_ledger.jsonl`, `stage_times.jsonl`, `results.json`, `per_request.json`, `assistant.log`.",
            "- Regenerate: `.venv/bin/python -m app.evaluation.assistant_evaluation report --offline`.", ""]
    return "\n".join(out)
