"""
Sensitivity of the published evaluations to the generated items the author
marked doubtful in the manual review (V27).

The author reviewed by hand the queries and requests an LLM wrote for the
catalog evaluation (V14) and the assistant evaluation (V22 and its
re-measurement V25) and gave each a verdict (data/evaluation/query_review/
review_sheet.csv). This module recomputes, from the saved files only (no
network, no LLM, no database), the figures of the published reports that are
computed over a set holding an item marked `dudosa` or `mal`, without those
items, and says whether a declared decision (a contrast's significance, a
threshold) changes.

The analysis was decided after the results were known and neither protocol
declares it: it is exploratory, it is reported whatever its direction, and it
never replaces or modifies the published results. It writes only
sensitivity.json and SENSITIVITY.md next to the review sheet.

How each evaluation is recomputed, with the functions, resamples and seed of
the published reports:
- Catalog (V14): what catalog_evaluation.results.build_results computes for
  strata A and B (metrics with bootstrap intervals, contrasts H1-H6 and
  H5-rep under both judges), the protocol's conclusion rule, the hybrid rule
  and the embedding decision rule, and the stratum C analysis, on the query
  set without the excluded queries. A need of stratum C keeps its remaining
  versions. The embedding rule's operational measurements (all 168 queries)
  are the published ones.
- Assistant (V22 and V25): assistant_evaluation.results.build_results with
  the excluded requests left out, with their turns, conversation and
  direct-agent run. The assistant's sets B and D are V14's strata B and D
  (same ids), so a doubtful B query is left out there too.

Before reporting, every evaluation is recomputed with no exclusion and
compared with its published results.json; any difference stops the command.

    python -m app.evaluation.query_review sensitivity
"""

import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from app.evaluation import query_review as QR
from app.evaluation.assistant_evaluation import design as AD
from app.evaluation.assistant_evaluation.report import _ci, _f, _p, _pct, _table
from app.evaluation.assistant_evaluation.report_v25 import _changes as v25_report_changes
from app.evaluation.assistant_evaluation.results import build_results as build_assistant_results
from app.evaluation.assistant_evaluation.sets import load_requests
from app.evaluation.catalog_evaluation import analysis as CA
from app.evaluation.catalog_evaluation import design as CD
from app.evaluation.catalog_evaluation.queries import EvalQuery, load_queries
from app.evaluation.catalog_evaluation.results import POOL_FILE, load_grades
from app.evaluation.catalog_evaluation.runs import load_runs

JSON_FILE, MD_FILE = "sensitivity.json", "SENSITIVITY.md"
COMMAND = ".venv/bin/python -m app.evaluation.query_review sensitivity"
JUDGES = ("primary", "second")
CATALOG, V22, V25 = "catalog", "assistant_v22", "assistant_v25"
ASSISTANT_DIRS: Dict[str, Path] = {V22: AD.DATA_DIR, V25: AD.V25_DIR}
TITLES = {CATALOG: "Catalog evaluation (V14)", V22: "Assistant evaluation, first measurement (V22)",
          V25: "Assistant evaluation, re-measurement (V25)"}
SHORT = {CATALOG: "V14", V22: "V22", V25: "V25"}
REPORTS = {CATALOG: "data/evaluation/CATALOG_EVALUATION.md", V22: "data/evaluation/ASSISTANT_EVALUATION.md",
           V25: "data/evaluation/ASSISTANT_EVALUATION_V25.md"}
# Sections of the assistant's results.json that hold measures (the rest is inventory, checks and spend).
ASSISTANT_SECTIONS = ("routes", "metrics", "conclusions", "grounding", "language", "cost_time", "failures", "v25")
CATALOG_METRICS = ("summary_ab", "contrasts", "stratum_c")
SYSTEM_METRICS = (("ndcg_at_10", "nDCG@10 [95 % CI]"), ("p_at_10", "P@10"), ("mrr_at_10", "MRR@10"),
                  ("recall_at_100", "Recall@100 (pool)"), ("judged_at_10", "Judged@10"))
THRESHOLDS = {"out_of_scope_ab": "out of scope on A+B", "clarify_b": "questions on B",
              "out_of_scope_g_valid": "out of scope on the valid G requests"}

# Figures of the assistant reports shown in SENSITIVITY.md (every figure that changes is in sensitivity.json):
# (label, paths of the values shown as "a / b", path of the number of items or None to read it from the first
# value). "{j}" stands for each judge.
ASSISTANT_FIGURES: Tuple[Tuple[str, Tuple[str, ...], Optional[str]], ...] = (
    ("Routes of set B, run 1", ("routes.matrices.run1.B",), None),
    ("Routes of set B, run 2", ("routes.matrices.run2.B",), None),
    ("Routes of set E (first turn)", ("routes.matrices.run1.E first turn",), None),
    ("Routes of set F", ("routes.matrices.run1.F",), None),
    ("Questions on A+B, run 1", ("routes.rates.run1.clarify_ab",), None),
    ("Route stability: same route in runs 1 and 2", ("routes.stability.same_route",), "routes.stability.requests"),
    ("H7, group B: assistant / P nDCG@10 ({j} judge)",
     ("metrics.{j}.h7.by_group.B.assistant", "metrics.{j}.h7.by_group.B.P"), "metrics.{j}.h7.by_group.B.requests"),
    ("H7, requests it searched: assistant / P ({j} judge)",
     ("metrics.{j}.h7.search_route_only.assistant", "metrics.{j}.h7.search_route_only.P"),
     "metrics.{j}.h7.search_route_only.requests"),
    ("A+B recommendations: precision / P at the same k ({j} judge)",
     ("metrics.{j}.h7.recommendations.precision_assistant", "metrics.{j}.h7.recommendations.precision_P_same_k"),
     "metrics.{j}.h7.recommendations.replies"),
    ("E: conversations where it asked", ("metrics.primary.e.asked",), "metrics.primary.e.conversations"),
    ("E: assistant after the conversation / P on the vague request ({j} judge)",
     ("metrics.{j}.e.ndcg_final", "metrics.{j}.e.ndcg_P_vague"), None),
    ("E: P on the original query ({j} judge)", ("metrics.{j}.e.ndcg_P_original",), None),
    ("E, asked: assistant / P on the vague request ({j} judge)",
     ("metrics.{j}.e.asked_rows.ndcg_final", "metrics.{j}.e.asked_rows.ndcg_P_vague"), None),
    ("E, did not ask: assistant / P on the vague request ({j} judge)",
     ("metrics.{j}.e.not_asked_rows.ndcg_final", "metrics.{j}.e.not_asked_rows.ndcg_P_vague"), None),
    ("F: routes", ("metrics.primary.f.route",), "metrics.primary.f.requests"),
    ("F: replies with recommendations", ("metrics.primary.f.with_recommendations",), "metrics.primary.f.requests"),
    ("F: named works recommended / in P's top k",
     ("metrics.primary.f.named_works_recommended", "metrics.primary.f.named_works_in_P_top_k"),
     "metrics.primary.f.named_works"),
    ("F: nDCG@10 of the reply's list / P ({j} judge)", ("metrics.{j}.f.ndcg_list", "metrics.{j}.f.ndcg_P"), None),
    *((f"F, {kind}: precision of the recommendations / P ({{j}} judge)",
       (f"metrics.{{j}}.f.by_kind.{kind}.precision_picks", f"metrics.{{j}}.f.by_kind.{kind}.precision_P"),
       f"metrics.{{j}}.f.by_kind.{kind}.requests") for kind in ("compare", "like_but", "facts")),
    ("Direct agent: precision / P at the same k ({j} judge)",
     ("metrics.{j}.agent_direct.precision_agent", "metrics.{j}.agent_direct.precision_P_same_k"), None),
    ("Search route, all sets: picks proposed / kept / dropped",
     tuple(f"grounding.search_route_by_set.all.{k}" for k in ("picks_proposed", "picks_kept", "picks_dropped")),
     "grounding.search_route_by_set.all.replies"),
    ("Agent route in the assistant: runs / recommendations kept",
     ("grounding.agent_in_assistant.runs", "grounding.agent_in_assistant.recommended"), None),
    ("Language of the reply: ok / mismatch / undecided",
     ("language.overall.ok", "language.overall.mismatch", "language.overall.undecided"), None),
    ("All turns: seconds, median / p95", ("cost_time.all_turns.wall_s",), "cost_time.all_turns.turns"),
    ("All turns: USD, median / p95", ("cost_time.all_turns.cost_usd",), "cost_time.all_turns.turns"),
    ("LLM calls / failed", ("cost_time.llm_calls", "cost_time.llm_failed"), None),
    ("Answer calls without output", ("failures.answer_errors",), None),
)
V25_FIGURES: Tuple[Tuple[str, Tuple[str, ...], Optional[str]], ...] = (
    ("G valid: routed to search", ("v25.g.valid.search",), None),
    ("G valid, routes by form", ("v25.g.valid_by_form",), "v25.g.threshold.requests"),
    ("G valid, routes by language", ("v25.g.valid_by_language",), "v25.g.threshold.requests"),
    ("G out of scope: detected", ("v25.g.out_of_scope.detection",), None),
    ("G turns on the search route: seconds, median / p95", ("v25.g_cost_time.by_route.search.wall_s",), None),
)

# Figures of the published reports computed over sets that may hold an excluded item, left as published.
NOT_RECOMPUTED: Dict[str, List[str]] = {
    "Catalog evaluation (V14)": [
        "Per-language and per-type tables, latency per stage, the S6 replication counts, section 9 (unanswerable "
        "queries), section 10 (explanations), the human annotation sheet and the agreement between the judges: "
        "descriptive or exploratory figures outside the scope of this analysis (strata A and B, and stratum C); no "
        "declared decision rests on them.",
        "Section 8, operational measurements of the two embedding models (all 168 queries): kept as published; "
        "condition 2 of the embedding rule is re-applied with them.",
    ],
    "Assistant evaluation (V22 and V25)": [
        "Judgment counts (measured pairs, from V14, judged here): they describe the judging work per need, and E's "
        "needs are B's needs, so a pair cannot be assigned to one request.",
        "Spend and calls by stage and model, the key-usage cross-check, wall-clock per stage and the ledger's list "
        "of failed calls: they come from the spend ledger, whose entries do not name the request they served.",
        "V25, answer calls that reached their token cap: read from the stage log, whose lines do not name the "
        "request.",
        "Inventory of the runs (P texts, turns, query vectors, depth check, commits, dates) and the generation "
        "checks of E, F and G: they describe what was run and how the sets were written, not the system; P and the "
        "depth check are stored by request text.",
    ],
}


class SensitivityError(ValueError):
    """The analysis cannot be made as asked (the message says why)."""


# =============================================================================
# The exclusion set, from the filled review sheet
# =============================================================================


def flagged_items(sheet: Path, items: Optional[Sequence[Mapping[str, str]]] = None) -> List[Dict[str, str]]:
    """The rows of a filled review sheet whose verdict is dudosa or mal, with their verdict and comment."""
    items = QR.review_items() if items is None else items
    filled = QR.read_sheet(sheet, items)
    return [{**{k: item[k] for k in ("item_id", "evaluation", "stratum", "language", "text")},
             **filled[item["item_id"]]}
            for item in items if filled[item["item_id"]]["verdict"] in QR.FLAGGED]


# =============================================================================
# Figures of each evaluation, published and recomputed
# =============================================================================


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_json(document: Any) -> Any:
    """The document as results.json stores it (tuples become lists; numbers keep their exact value)."""
    return json.loads(json.dumps(document, ensure_ascii=False, default=str))


def published_catalog(data_dir: Path = CD.DATA_DIR) -> Dict[str, Any]:
    """The part of the catalog's published results.json that catalog_figures recomputes."""
    r = _read_json(data_dir / "results.json")
    return {"metrics": {j: {k: r["metrics"][j][k] for k in CATALOG_METRICS} for j in JUDGES},
            "conclusions": r["conclusions"], "hybrid_rule": r["hybrid_rule"], "decision_rule": r["decision_rule"]}


def catalog_figures(data_dir: Path = CD.DATA_DIR, exclude: Collection[str] = (),
                    strata: Collection[str] = CD.STATISTICAL_STRATA) -> Dict[str, Any]:
    """
    The figures of published_catalog, recomputed from the saved runs and
    judgments without the queries in `exclude`, with the calls of
    catalog_evaluation.results.build_results. `strata` are the strata the
    summary and the contrasts are computed over (the protocol's A and B by
    default; the post hoc scope analysis of V29 passes others).
    """
    queries = [q for q in load_queries(data_dir / "queries" / "queries.json") if q.query_id not in exclude]
    runs = load_runs(data_dir, CD.SYSTEM_NAMES)
    pool = _read_json(data_dir / POOL_FILE)
    ab_ids = [q.query_id for q in queries if q.stratum in strata]
    metrics = {}
    for label, judge in zip(JUDGES, (CD.PRIMARY_JUDGE, CD.SECOND_JUDGE)):
        scores = CA.per_query(queries, runs, load_grades(data_dir, judge, pool), CD.SYSTEM_NAMES)
        metrics[label] = {"summary_ab": CA.summarize(scores, ab_ids), "contrasts": CA.contrasts(scores, ab_ids),
                          "stratum_c": CA.stratum_c(queries, runs, scores, CD.SYSTEM_NAMES)}
    first, second = metrics["primary"]["contrasts"], metrics["second"]["contrasts"]
    operational = _read_json(data_dir / "results.json")["operational"]  # all 168 queries, as published
    return _as_json({
        "metrics": metrics,
        "conclusions": {name: CA.conclusion(first[name], second[name]) for name in first},
        "hybrid_rule": CA.hybrid_rule(first["H3"], second["H3"]),
        "decision_rule": CA.decision_rule(first["H6"], second["H6"], operational[CD.BGE_EMBEDDING],
                                          operational[CD.QWEN_EMBEDDING]),
    })


def published_assistant(data_dir: Path) -> Dict[str, Any]:
    return _read_json(data_dir / "results.json")


def assistant_figures(data_dir: Path, exclude: Collection[str] = ()) -> Dict[str, Any]:
    """build_results of one assistant measurement without `exclude`, as its results.json stores it."""
    results = build_assistant_results(data_dir, exclude)
    return _as_json({k: v for k, v in results.items() if k != "per_request"})


# =============================================================================
# Comparing documents
# =============================================================================


def first_difference(a: Any, b: Any, path: str = "") -> Optional[str]:
    """The dotted path of the first value where two documents differ, or None when they are equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in list(a) + [k for k in b if k not in a]:
            where = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                return where
            found = first_difference(a[key], b[key], where)
            if found is not None:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found is not None:
                return found
        return None
    return None if a == b else (path or "(whole document)")


def check_reproduction(name: str, published: Any, recomputed: Any) -> None:
    """Raise SensitivityError naming the first published figure the recomputation without exclusions misses."""
    where = first_difference(published, recomputed)
    if where is not None:
        raise SensitivityError(f"{name}: recomputing with no exclusion does not reproduce the published figure "
                               f"`{where}` of results.json, so no sensitivity is reported. Find what differs "
                               "before trusting any recomputed figure.")


def leaves(document: Any, path: str = "") -> Iterator[Tuple[str, Any]]:
    """
    (dotted path, value) of every figure of a document. A list of numbers
    (an interval) is one figure; lists of records (per-request rows) are
    listings, not figures, and are skipped.
    """
    if isinstance(document, dict):
        for key, value in document.items():
            yield from leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(document, list) and any(isinstance(x, (dict, list)) for x in document):
        return
    else:
        yield path, document


def changed_figures(published: Mapping[str, Any], without: Mapping[str, Any],
                    sections: Sequence[str]) -> Tuple[int, List[Dict[str, Any]]]:
    """
    (figures compared, [{"figure", "published", "without"}] of those that
    differ) over `sections`. The figures are those of the published document:
    leaving items out can only remove figures (then "without" is None), never
    add one.
    """
    before = dict(leaves({k: published.get(k) for k in sections}))
    after = dict(leaves({k: without.get(k) for k in sections}))
    return len(before), [{"figure": p, "published": value, "without": after.get(p)}
                         for p, value in before.items() if value != after.get(p)]


def _get(document: Any, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(document, dict):
            return None
        document = document.get(key)
    return document


# =============================================================================
# Declared decisions
# =============================================================================


DECLARED, REPORT_FLAG = "declared", "report flag"  # a decision of a protocol, or a mechanical flag of a report


def _decision(label: str, published: Any, without: Any, published_detail: str = "", without_detail: str = "",
              kind: str = DECLARED) -> Dict[str, Any]:
    return {"decision": label, "kind": kind, "published": published, "without": without,
            "changes": published != without, "published_detail": published_detail, "without_detail": without_detail}


def _significant(row: Mapping[str, Any]) -> bool:
    """Significant at the declared level: Holm-adjusted p inside the family, raw p outside it (H5-rep)."""
    return (row["p_holm"] if row.get("p_holm") is not None else row["p"]) < CD.ALPHA


def _p_detail(row: Mapping[str, Any]) -> str:
    holm = row.get("p_holm") is not None
    n = row.get("queries", row.get("requests"))
    return f"{'p Holm' if holm else 'p'} {_p(row['p_holm'] if holm else row['p'])}, n {n}"


def contrast_decisions(published: Mapping[str, Any], without: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Per declared contrast: its significance under each judge and the protocol's conclusion (both judges)."""
    out = []
    for name in published["conclusions"]:
        for judge in JUDGES:
            a, b = published["metrics"][judge]["contrasts"][name], without["metrics"][judge]["contrasts"][name]
            out.append(_decision(f"{name} significant at {CD.ALPHA}, {judge} judge", _significant(a), _significant(b),
                                 _p_detail(a), _p_detail(b)))
        out.append(_decision(f"{name} conclusion (both judges)", published["conclusions"][name],
                             without["conclusions"][name]))
    return out


def catalog_rule_decisions(published: Mapping[str, Any], without: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The two rules of section 8 that the contrasts decide."""
    hybrid, embedding = published["hybrid_rule"], published["decision_rule"]
    return [
        _decision("Hybrid rule: S4 significantly better than S5 under both judges",
                  hybrid["s4_significantly_better_than_s5"], without["hybrid_rule"]["s4_significantly_better_than_s5"]),
        _decision("Embedding rule, condition 1: H6 does not show qwen significantly better",
                  embedding["condition_1_h6"], without["decision_rule"]["condition_1_h6"]),
        _decision("Embedding rule, outcome", embedding["outcome"], without["decision_rule"]["outcome"]),
    ]


def _thresholds(label: str, rows: Mapping[str, Any], rows_without: Mapping[str, Any], all_met: bool,
                all_met_without: bool) -> List[Dict[str, Any]]:
    """Each threshold row (count, requests, value, maximum, met) published and without, and whether all are met."""
    def rate(row: Mapping[str, Any]) -> str:
        return f"{row['count']}/{row['requests']} = {_pct(row['value'])}"

    out = [_decision(f"{label}: {THRESHOLDS[key]} at most {_pct(row['maximum'])}", row["met"],
                     rows_without[key]["met"], rate(row), rate(rows_without[key])) for key, row in rows.items()]
    return out + [_decision(f"{label}: all met", all_met, all_met_without)]


def threshold_decisions(name: str, published: Mapping[str, Any], without: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The route thresholds: section 7 of the protocol (V22), section 4 of the addendum (V25, A+B as secondary)."""
    where = "Protocol section 7" if name == V22 else "Addendum section 4, secondary (A+B)"
    out = []
    for run, a in published["routes"]["thresholds"].items():
        b = without["routes"]["thresholds"][run]
        out += _thresholds(f"{where}, {run}", a["rows"], b["rows"], a["both_met"], b["both_met"])
    if name == V25:
        a, b = published["v25"]["thresholds"], without["v25"]["thresholds"]
        out += _thresholds("Addendum section 4, required", a["required"], b["required"], a["required_met"],
                           b["required_met"])
    return out


def v25_report_flags(published: Mapping[str, Mapping[str, Any]],
                     without: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The "What changed" rows of the V25 report (worse: yes or no), published and without the excluded items."""
    before = {m: (a, b, w) for m, a, b, w in v25_report_changes(published[V22], published[V25])}
    after = {m: (a, b, w) for m, a, b, w in v25_report_changes(without[V22], without[V25])}
    return [_decision(f"{measure}: second measurement worse", before[measure][2], after[measure][2],
                      f"{before[measure][0]} -> {before[measure][1]}", f"{after[measure][0]} -> {after[measure][1]}",
                      kind=REPORT_FLAG)
            for measure in before if measure in after]


# =============================================================================
# The analysis
# =============================================================================


def _stratum_c_counts(queries: Sequence[EvalQuery]) -> Dict[str, Any]:
    """Versions per language and pairs of versions of the same need in stratum C."""
    c = [q for q in queries if q.stratum == CD.STRATUM_C]
    versions = Counter(q.need_id for q in c)
    return {"needs": len(versions), "versions_by_language": dict(Counter(q.language for q in c)),
            "pairs": sum(comb(n, 2) for n in versions.values())}


def _figure_rows(published: Mapping[str, Any], without: Mapping[str, Any],
                 specs: Sequence[Tuple[str, Tuple[str, ...], Optional[str]]]) -> List[Dict[str, Any]]:
    """The figures of `specs` (one per judge where the label has "{j}"), published and without."""
    rows = []
    for label, paths, n_path in specs:
        for judge in (JUDGES if "{j}" in label else ("",)):
            values = [(_get(published, p.replace("{j}", judge)), _get(without, p.replace("{j}", judge)))
                      for p in paths]
            if n_path:
                items = [_get(published, n_path.replace("{j}", judge)), _get(without, n_path.replace("{j}", judge))]
            else:
                items = [_count(values[0][0]), _count(values[0][1])]
            rows.append({"figure": label.replace("{j}", judge), "items": items,
                         "published": [a for a, _ in values], "without": [b for _, b in values]})
    return rows


def _contrast_pairs(published: Mapping[str, Any], without: Mapping[str, Any]) -> Dict[str, Any]:
    """{judge: {contrast: {"published": row, "without": row}}}."""
    return {j: {h: {"published": row, "without": without["metrics"][j]["contrasts"][h]}
                for h, row in published["metrics"][j]["contrasts"].items()} for j in JUDGES}


def _count(value: Any) -> Optional[int]:
    """The number of items a figure is computed over, when the figure records it."""
    if isinstance(value, dict):
        for key in ("requests", "n", "total", "turns", "conversations", "replies"):
            if isinstance(value.get(key), int):
                return value[key]
    return None


def sensitivity(exclude: Collection[str], catalog_dir: Path = CD.DATA_DIR,
                assistant_dirs: Mapping[str, Path] = ASSISTANT_DIRS) -> Dict[str, Any]:
    """
    The whole analysis for one exclusion set of item ids: per evaluation,
    the reproduction check, the excluded items it holds, the decisions and
    every figure that changes. Raises SensitivityError for an id that names no
    query or request of the evaluations, or a published figure that the
    recomputation with no exclusion does not reproduce.
    """
    exclude = set(exclude)
    queries = load_queries(catalog_dir / "queries" / "queries.json")
    stratum_of = {q.query_id: q.stratum for q in queries}
    set_of = {name: {r.request_id: r.set for r in load_requests(d / "sets" / "requests.json")}
              for name, d in assistant_dirs.items()}
    unknown = sorted(exclude - set(stratum_of) - {i for ids in set_of.values() for i in ids})
    if unknown:
        raise SensitivityError(f"unknown item ids in the exclusion set: {unknown}; every id must name a query of "
                               f"{catalog_dir / 'queries' / 'queries.json'} or a request of an assistant measurement")

    out: Dict[str, Any] = {}
    # --- catalog --------------------------------------------------------------------
    published = published_catalog(catalog_dir)
    check_reproduction(TITLES[CATALOG], published, catalog_figures(catalog_dir))
    ids = sorted(i for i in exclude if i in stratum_of)
    without = catalog_figures(catalog_dir, ids) if ids else published
    compared, changed = changed_figures(published, without, list(published))
    kept = [q for q in queries if q.query_id not in ids]
    out[CATALOG] = {
        "data_dir": catalog_dir.as_posix(), "report": REPORTS[CATALOG],
        "reproduced": "metrics of A and B, contrasts and stratum C (both judges), conclusions, hybrid_rule and "
                      "decision_rule of results.json: identical",
        "excluded": {i: stratum_of[i] for i in ids},
        "items": {"ab": [sum(q.stratum in CD.STATISTICAL_STRATA for q in qs) for qs in (queries, kept)],
                  "stratum_c": [_stratum_c_counts(queries), _stratum_c_counts(kept)]},
        "decisions": contrast_decisions(published, without) + catalog_rule_decisions(published, without),
        "contrasts": _contrast_pairs(published, without),
        "figures_compared": compared, "changed": changed, "published": published, "without": without,
    }
    # --- assistant ------------------------------------------------------------------
    documents: Dict[str, Dict[str, Any]] = {"published": {}, "without": {}}
    for name, data_dir in assistant_dirs.items():
        published = published_assistant(data_dir)
        check_reproduction(TITLES[name], published, assistant_figures(data_dir))
        ids = sorted(i for i in exclude if i in set_of[name])
        without = assistant_figures(data_dir, ids) if ids else published
        documents["published"][name], documents["without"][name] = published, without
        compared, changed = changed_figures(published, without, ASSISTANT_SECTIONS)
        out[name] = {
            "data_dir": data_dir.as_posix(), "report": REPORTS[name],
            "reproduced": "the whole results.json: identical",
            "excluded": {i: set_of[name][i] for i in ids},
            "decisions": contrast_decisions(published, without) + threshold_decisions(name, published, without),
            "contrasts": _contrast_pairs(published, without),
            "figures": _figure_rows(published, without, ASSISTANT_FIGURES + (V25_FIGURES if name == V25 else ())),
            "figures_compared": compared, "changed": changed,
        }
    if V22 in assistant_dirs and V25 in assistant_dirs:
        out[V25]["decisions"] += v25_report_flags(documents["published"], documents["without"])
    used_in: Dict[str, List[str]] = defaultdict(list)
    for item_id, stratum in stratum_of.items():
        used_in[item_id].append(f"{SHORT[CATALOG]} stratum {stratum}")
    for name, sets in set_of.items():
        for item_id, name_of_set in sets.items():
            used_in[item_id].append(f"{SHORT[name]} set {name_of_set}")
    in_both = sorted(i for i in exclude if i in stratum_of and any(i in ids for ids in set_of.values()))
    return {"used_in": {i: used_in[i] for i in sorted(exclude)}, "in_both_evaluations": in_both,
            "evaluations": out, "not_recomputed": NOT_RECOMPUTED}


# =============================================================================
# Report
# =============================================================================


def _show(value: Any) -> str:
    """A figure as the reports print it."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _f(value) if value == 0 or abs(value) >= 0.01 else _f(value, 5)
    if isinstance(value, dict):
        if {"count", "requests", "rate"} <= value.keys():
            return f"{value['count']}/{value['requests']} = {_pct(value['rate'])}"
        if {"mean", "ci"} <= value.keys():
            return "n/a" if value["mean"] is None else f"{_show(value['mean'])} {_ci(value['ci'])}"
        if {"median", "p95"} <= value.keys():
            return f"{_show(value['median'])} / {_show(value['p95'])}"
        parts = [f"{k} ({_show(v)})" if isinstance(v, dict) else f"{k} {_show(v)}"
                 for k, v in value.items() if k != "total"]
        return ("; " if any(isinstance(v, dict) for v in value.values()) else ", ").join(parts)
    if isinstance(value, list):
        return ", ".join(_show(v) for v in value)
    return str(value)


def _arrow(before: Any, after: Any) -> str:
    """"a -> b", or a single value when both print the same."""
    before, after = str(before), str(after)
    return before if before == after else f"{before} -> {after}"


def _difference(row: Mapping[str, Any]) -> str:
    return f"{_f(row['difference'], signed=True)} {_ci(row['ci'], signed=True)}"


def _cell(value: Any, detail: str) -> str:
    return f"{_show(value)} ({detail})" if detail else _show(value)


def _decision_table(decisions: Sequence[Mapping[str, Any]]) -> List[str]:
    return _table(["Decision", "Published", "Without the doubtful items", "Changes"],
                  [[d["decision"], _cell(d["published"], d["published_detail"]),
                    _cell(d["without"], d["without_detail"]), "**yes**" if d["changes"] else "no"] for d in decisions])


def _changed_decisions(document: Mapping[str, Any], kind: str) -> List[str]:
    return [f"{TITLES[name]}: {d['decision']}: {_cell(d['published'], d['published_detail'])} -> "
            f"{_cell(d['without'], d['without_detail'])}"
            for name, e in document["evaluations"].items()
            for d in e["decisions"] if d["changes"] and d["kind"] == kind]


def _largest_shift(e: Mapping[str, Any]) -> str:
    """The declared contrast whose difference moved most without the excluded items (either judge)."""
    label, a, b = max(((f"{h}, {j} judge", c["published"]["difference"], c["without"]["difference"])
                       for j, by_name in e["contrasts"].items() for h, c in by_name.items()),
                      key=lambda x: abs(x[2] - x[1]))
    return f"{label}, {_f(a, signed=True)} -> {_f(b, signed=True)}"


def _catalog_markdown(e: Mapping[str, Any]) -> List[str]:
    pub, new = e["published"], e["without"]
    excluded = ", ".join(f"`{i}` (stratum {s})" for i, s in e["excluded"].items()) or "none"
    L = [f"## {TITLES[CATALOG]}", "",
         f"Published report: `{e['report']}`. Excluded here: {excluded}. Strata A and B, the statistical unit: "
         f"{_arrow(*e['items']['ab'])} queries. {e['figures_compared']} figures recomputed, {len(e['changed'])} of "
         "them change.", ""]

    def metric(row: Mapping[str, Any], m: str) -> str:
        return f"{_f(row[m]['mean'])} {_ci(row[m]['ci'])}" if m == "ndcg_at_10" else _f(row[m]["mean"])

    for judge, model in zip(JUDGES, CD.JUDGES):
        L += [f"### Strata A and B, {judge} judge `{model}`", "",
              "Each cell: published -> without the doubtful items (a single value when equal).", ""]
        rows = []
        for system, row in pub["metrics"][judge]["summary_ab"].items():
            other = new["metrics"][judge]["summary_ab"][system]
            rows.append([system, _arrow(row["queries"], other["queries"]),
                         *[_arrow(metric(row, m), metric(other, m)) for m, _ in SYSTEM_METRICS]])
        L += _table(["System", "Queries", *[label for _, label in SYSTEM_METRICS]], rows)
    L += ["### Declared contrasts H1-H6 and H5-rep", "",
          "nDCG@10 difference (system - baseline) on A and B, 95 % paired bootstrap interval, paired randomisation "
          f"test ({CD.N_RESAMPLES:,} resamples, seed {CD.SEED}), Holm over H1-H6; H5-rep is outside the family "
          "(raw p).", ""]
    rows = []
    for name, row in pub["metrics"]["primary"]["contrasts"].items():
        for judge in JUDGES:
            a, b = e["contrasts"][judge][name]["published"], e["contrasts"][judge][name]["without"]
            key = "p_holm" if a.get("p_holm") is not None else "p"
            rows.append([name, f"{row['system']} - {row['baseline']}", judge, _arrow(a["queries"], b["queries"]),
                         _arrow(_difference(a), _difference(b)),
                         _arrow(_p(a[key]), _p(b[key])) + ("" if key == "p_holm" else " (raw)")])
    L += _table(["Id", "Contrast", "Judge", "Queries", "Difference [95 % CI]", "p Holm"], rows)
    c_before, c_after = e["items"]["stratum_c"]
    versions = c_before["versions_by_language"], c_after["versions_by_language"]
    languages = [lang for lang in CD.C_LANGUAGES if versions[0].get(lang) != versions[1].get(lang)]
    c_pub, c_new = pub["metrics"]["primary"]["stratum_c"], new["metrics"]["primary"]["stratum_c"]
    L += ["### Stratum C (descriptive)", "",
          f"{c_after['needs']} needs; the need of each excluded version keeps its other versions (the choice made "
          f"here), so the pairs of versions whose top 10 overlap is averaged go from {c_before['pairs']} to "
          f"{c_after['pairs']}. Only the columns that change are shown (primary judge).", ""]
    rows = [[system, *[_arrow(_f(c_pub["ndcg_at_10_by_language"][system][lang]),
                              _f(c_new["ndcg_at_10_by_language"][system][lang])) for lang in languages],
             _arrow(_f(overlap), _f(c_new["mean_pairwise_top10_overlap"][system]))]
            for system, overlap in c_pub["mean_pairwise_top10_overlap"].items()]
    L += _table(["System", *[f"{lang} nDCG@10 ({_arrow(versions[0][lang], versions[1].get(lang, 0))} versions)"
                             for lang in languages],
                 f"Top-10 overlap ({_arrow(c_before['pairs'], c_after['pairs'])} pairs)"], rows)
    L += ["### Declared decisions", ""] + _decision_table(e["decisions"])
    return L


def _assistant_markdown(name: str, e: Mapping[str, Any]) -> List[str]:
    by_set = Counter(e["excluded"].values())
    L = [f"## {TITLES[name]}", "",
         f"Published report: `{e['report']}`. Excluded here: {len(e['excluded'])} requests "
         f"({', '.join(f'{s} {n}' for s, n in sorted(by_set.items()))}: "
         f"{', '.join(f'`{i}`' for i in e['excluded'])}). {e['figures_compared']} figures recomputed, "
         f"{len(e['changed'])} of them change.", "",
         "### Contrasts H7-H9", "",
         f"Paired randomisation test ({AD.N_RESAMPLES:,} sign flips, seed {AD.SEED}), 95 % paired bootstrap "
         "interval, Holm over H7-H9, under each judge. Each cell: published -> without the doubtful items.", ""]
    rows = []
    for judge in JUDGES:
        for h, pair in e["contrasts"][judge].items():
            a, b = pair["published"], pair["without"]
            rows.append([h, judge, _arrow(a["requests"], b["requests"]),
                         _arrow(f"{_f(a['system_mean'])} / {_f(a['reference_mean'])}",
                                f"{_f(b['system_mean'])} / {_f(b['reference_mean'])}"),
                         _arrow(_difference(a), _difference(b)),
                         _arrow(_p(a["p"]), _p(b["p"])), _arrow(_p(a["p_holm"]), _p(b["p_holm"]))])
    L += _table(["Id", "Judge", "n", "System / P", "Difference [95 % CI]", "p", "p Holm"], rows)
    L += ["### Declared decisions", ""]
    if name == V25:
        L += ["The last rows are the report's \"What changed\" table (worse is applied mechanically; not a "
              "decision of the addendum).", ""]
    L += _decision_table(e["decisions"])
    L += ["### Other figures of the report", "",
          f"The report's figures over sets that hold an excluded item; every one of the {len(e['changed'])} figures "
          f"that change is listed in `{JSON_FILE}`.", ""]
    L += _table(["Figure", "Items", "Published", "Without the doubtful items"],
                [[f["figure"], _arrow(_show(f["items"][0]), _show(f["items"][1])),
                  " / ".join(_show(v) for v in f["published"]), " / ".join(_show(v) for v in f["without"])]
                 for f in e["figures"]])
    return L


def build_markdown(document: Mapping[str, Any]) -> str:
    """SENSITIVITY.md from the document `write_report` saves as sensitivity.json."""
    excluded = document["excluded"]
    verdicts = Counter(x["verdict"] for x in excluded)
    changed = _changed_decisions(document, DECLARED)
    shared = ", ".join(f"`{i}`" for i in document["in_both_evaluations"]) or "none today"
    L = ["# Sensitivity of the evaluations to the doubtful generated queries (V27)", "",
         "The author reviewed by hand the 254 queries and requests that an LLM wrote for the catalog evaluation "
         "(V14) and the assistant evaluation (V22 and its re-measurement V25) "
         f"(`{document['sheet']}`, summary in `{(QR.OUT_DIR / QR.SUMMARY_FILE).as_posix()}`, task note "
         "`docs/notes/tareas/V27-hoja-revision-consultas.md`). This report recomputes, from the saved files only "
         "(no network, no LLM, no database), the figures of the published reports that are computed over a set "
         "holding an item marked `dudosa` (doubtful) or `mal` (wrong), without those items, to see whether the "
         "conclusions of the evaluations depend on them.", "",
         "- **Decided after the results.** Neither protocol (`docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`, "
         "`docs/notes/2026-11-12-protocolo-evaluacion-asistente.md` and its addendum) declares this analysis; it was "
         "decided once the published results were known. It is exploratory and is reported whatever its direction. "
         "It does not replace or modify any published result: the reports and their `results.json` are unchanged and "
         "remain the results of the evaluations.",
         f"- **Excluded items**: every item of the sheet whose verdict is `dudosa` or `mal`: {len(excluded)} "
         f"({', '.join(f'{n} {v}' for v, n in sorted(verdicts.items()))}). The author's reason for each is in the "
         "table below.",
         "- **Same method as the published reports**: the same functions, resamples (10,000), seed (0), level "
         "(0.05) and Holm families. Before reporting, each evaluation was recomputed with no exclusion and compared "
         "with its published `results.json` (catalog: the metrics of A and B, the contrasts and stratum C under both "
         "judges, the conclusions and the two rules of section 8; assistant: the whole file of each measurement). "
         "Every compared figure is identical; the command stops on the first one that is not, and the test "
         "`test_no_exclusion_reproduces_the_published_figures_exactly` checks the same.",
         "- **Choices made here.** (1) The assistant's sets B and D are V14's strata B and D (same ids), reviewed "
         "once as catalog rows, so a doubtful query of those strata is also left out of the assistant's set "
         f"({shared}): it changes H7, the route thresholds on A+B "
         "and on B, the route stability and, if it is in it, the direct-agent sample. (2) Stratum C: the doubtful "
         "version is dropped and its need keeps its other versions. (3) Condition 2 of the embedding rule uses the "
         "published operational measurements (all 168 queries).",
         f"- **Regenerate**: `{COMMAND}` (reads the sheet; writes only `{JSON_FILE}` and this file in "
         f"`{QR.OUT_DIR.as_posix()}/`).", "",
         "## Excluded items", ""]
    L += _table(["Item", "Sheet", "Used in", "Verdict", "Author's comment"],
                [[f"`{x['item_id']}`", f"{x['evaluation']} {x['stratum']}",
                  "; ".join(document["used_in"][x["item_id"]]), x["verdict"], x["comment"]] for x in excluded])
    L += ["## Outcome", ""]
    total = sum(d["kind"] == DECLARED for e in document["evaluations"].values() for d in e["decisions"])
    L += ([f"- **No declared decision changes** ({total} checked): every declared contrast keeps its significance "
           "under each judge and its conclusion, every route threshold keeps its outcome, and the two rules of the "
           "catalog protocol's section 8 keep theirs."]
          if not changed else [f"- **{len(changed)} of {total} declared decisions change**:"]
          + [f"  - {c}" for c in changed])
    flags = _changed_decisions(document, REPORT_FLAG)
    if flags:
        L += ["- Mechanical flags of a report that change (they are not declared decisions):"]
        L += [f"  - {c}" for c in flags]
    for name, e in document["evaluations"].items():
        L.append(f"- {TITLES[name]}: {len(e['excluded'])} items excluded; {len(e['changed'])} of "
                 f"{e['figures_compared']} recomputed figures change; the contrast that moves most is "
                 f"{_largest_shift(e)}.")
    L.append("")
    L += _catalog_markdown(document["evaluations"][CATALOG])
    for name in (V22, V25):
        if name in document["evaluations"]:
            L += _assistant_markdown(name, document["evaluations"][name])
    L += ["## Figures not recomputed", ""]
    L += [f"- {where}: {text}" for where, texts in document["not_recomputed"].items() for text in texts]
    L += ["", "## Files", "",
          "- Code: `app/evaluation/review_sensitivity.py` (this analysis), `app/evaluation/query_review.py` (the "
          "`sensitivity` command); `assistant_evaluation/results.build_results` takes the excluded requests.",
          "- Tests: `tests/evaluation/test_review_sensitivity.py`.",
          f"- `{JSON_FILE}`: per evaluation the excluded items, the decisions, and every recomputed figure that "
          "changes (published and without the doubtful items).", ""]
    return "\n".join(L)


def write_report(sheet: Path, out_dir: Path, catalog_dir: Path = CD.DATA_DIR,
                 assistant_dirs: Mapping[str, Path] = ASSISTANT_DIRS) -> Dict[str, Any]:
    """Read the flagged items of the sheet, run the analysis and write sensitivity.json and SENSITIVITY.md."""
    excluded = flagged_items(sheet)
    document = {"analysis": "sensitivity of the evaluations to the doubtful generated items (V27)",
                "decided_after_results": True, "declared_in_protocols": False, "command": COMMAND,
                "sheet": sheet.as_posix(), "excluded": excluded,
                **sensitivity([x["item_id"] for x in excluded], catalog_dir, assistant_dirs)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / JSON_FILE).write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out_dir / MD_FILE).write_text(build_markdown(document), encoding="utf-8")
    return document
