"""
Text of the report that is not a table: the summary (derived from the
results, so it always says what the numbers say), the deviations from the
protocol, the limits and the file list.
"""

from typing import Any, Dict, List, Mapping

from app.evaluation.catalog_evaluation import design as D

DEVIATIONS: List[str] = [
    "The protocol calls the approximate index of S2 \"HNSW\". The deployed schema's index is `lakebase_ann` "
    "(Neon's IVF index with full-precision re-ranking; `catalog_info.vector_index`), and S2 was run on it as "
    "deployed. The S2 against E1 comparison therefore measures that index.",
    "Fusion depth. S3 fuses the top 60 of S1 and of S2, the candidate depth the deployed service uses for a page of "
    "10 results with the reranker on (max(10, 30) x 2); its run file keeps the first 100 of the fused list. S1, S2 "
    "and E1 are separate 100-deep searches. S4 re-ranks the top 30 of S2 (S2 with the deployed reranker), S5 and S6 "
    "the top 30 of S3; the rest of each list keeps its order.",
    "Reranker and LLM calls of S4-S6 used the evaluation's long deadlines (30 s per attempt, 3 retries, and up to "
    "3 attempts of the whole call), not the deployed 5 s deadline, so that a slow provider does not turn a quality "
    "measurement into a fallback. The deployed deadlines are measured for the embeddings in section 8 only.",
    "The judge grades each pair against the need's English description (stratum A: the query itself), as the "
    "protocol says, through the unchanged V05 prompt (v1.0), whose text calls the query \"a short keyword search in "
    "Spanish or in English\" and names only English and Spanish book languages (other languages appear as their "
    "ISO code). The prompt was kept unchanged as the protocol requires.",
    "Section 8, interpretation of the criteria: latency is better when both the median and the p95 are lower, and "
    "more than 10 % worse when either is more than 10 % higher; failed calls and queries with a changed top 10 are "
    "counts, and a count of 0 for qwen makes any positive bge-m3 count more than 10 % worse. Top 10 changes are "
    "counted over queries with at least two successful repetitions, with each model's exact search (E1 in Postgres, "
    "E2 locally).",
    "Section 9 uses the primary judge to decide which A and B queries have an answer and which D queries are "
    "reclassified; the second judge's counts are shown next to it.",
    "Section 10 language check: V10's rule only distinguishes Spanish and English, so the catalog's language "
    "detector (app/ingestion/language_detection.py, 12 languages) is applied to the summary and reasoning instead.",
    "Human annotation sample: drawn from the judged pool pairs of strata A and B (the statistical unit), not from C "
    "or D.",
    "Stages ran partly at the same time in separate processes (the run files while the E2 vectors were being "
    "computed; judging, the operational measurement and the explanations together). Each process checks the cap "
    "against the ledger it read at start plus its own calls, so a check could lag by the other processes' spend; the "
    "total stayed far below the cap and the ledger on disk holds every call.",
    "Query generator: `google/gemini-3.8-flash`, temperature 0.7, reasoning effort low, JSON output, one call per "
    "B language, one for C and one for D; replacements were requested in the same conversation with the reason "
    "(at most three rounds).",
]

LIMITS: List[str] = [
    "Judgments are from two LLM judges only. Their agreement with people was measured on another collection (177 "
    "books, English and Spanish, V05 and V10); on this catalog and its twelve languages it is pending the human sheet.",
    "Strata B, C and D were written by an LLM: the queries may be more regular and better formed than real users' "
    "ones, and the descriptive and known-item queries reflect what that model knows.",
    "The pool is formed by the evaluated systems; a future system cannot be compared on these judgments without "
    "judging its new results. Recall@100 is relative to this pool.",
    "One collection, one run of each deterministic system; S6 was repeated because it is the only one with variable "
    "output. API models can change behind the same id; dates and pinned providers are recorded.",
    "Latencies were measured from a laptop over the internet against a Neon compute shared with the live site, "
    "sequentially; they describe this setting, not the deployed function's.",
    "E2 (bge-m3) is far below E1 (qwen) here (nDCG@10 0.43 against 0.68), a much larger gap than on the 177-book "
    "collection (V16: 0.614 against 0.654). The stored vectors were checked (9 works re-embedded: cosine 1.000 with the "
    "stored vectors, ids aligned) and the per-work text is the loader's; why the gap is larger on this catalog "
    "(longer multilingual texts, provider-side truncation, or the model itself) was not investigated: unverified.",
    "The 116 A+B queries are the statistical unit; stratum C (40 queries, 10 needs) and the per-language and "
    "per-type tables are descriptive, with few queries per cell.",
]


def _contrast_line(results: Mapping[str, Any], name: str, text: str) -> str:
    p = results["metrics"]["primary"]["contrasts"].get(name)
    s = results["metrics"]["second"]["contrasts"].get(name)
    outcome = results["conclusions"].get(name, "n/a")
    if not p or not s:
        return f"{name}: not computed."
    return (f"{name} ({text}): {p['system']} - {p['baseline']} = {p['difference']:+.3f} (Holm p "
            f"{p['p_holm'] if p['p_holm'] is not None else p['p']:.3f}) under the primary judge, {s['difference']:+.3f} "
            f"(Holm p {s['p_holm'] if s['p_holm'] is not None else s['p']:.3f}) under the second: **{outcome}**.")


def report_meta(results: Mapping[str, Any]) -> Dict[str, Any]:
    summary: List[str] = []
    m = results["metrics"]["primary"]["summary_ab"]
    if m:
        best = max(m, key=lambda s: m[s]["ndcg_at_10"]["mean"])
        summary.append("nDCG@10 on A+B (primary judge): " + ", ".join(
            f"{s} {m[s]['ndcg_at_10']['mean']:.3f}" for s in D.SYSTEM_NAMES if s in m) + f"; highest: {best}.")
        texts = {"H1": "hybrid vs vector", "H2": "reranker on the hybrid", "H3": "BM25 candidates under the reranker",
                 "H4": "vector vs BM25", "H5": "LLM re-ranker vs cross-encoder", "H6": "bge-m3 vs qwen, exact search"}
        for name, text in texts.items():
            summary.append(_contrast_line(results, name, text))
        rep = results["conclusions"].get("H5-rep")
        if rep:
            summary.append(f"H5 replication with the second S6 run: {rep}.")
    hybrid = results.get("hybrid_rule")
    if hybrid:
        summary.append(f"Hybrid rule (section 8): {hybrid['outcome']}.")
    decision = results.get("decision_rule")
    if decision:
        summary.append(f"Embedding decision rule (section 8): {decision['outcome']}.")
    ag = results["judgments"]["agreement"]
    if ag.get("pairs"):
        summary.append(f"Judges agree with quadratic kappa {ag['overall']['quadratic_weighted_kappa']:.3f} on {ag['pairs']} pairs.")
    targets = (results["queries"].get("known_item_targets") or {})
    missing = [k for k, v in targets.items() if not v]
    known_note = ("Not found by the mechanical title match: " + ", ".join(f"`{k}`" for k in missing)
                  + "; `b-zh-6` asks for 红楼梦, which the catalog holds under its English title \"Dream of the Red "
                  "Chamber\" (Cao Xueqin), checked by hand.") if missing else ""
    return {
        "date": "2026-11-12",
        "summary": summary,
        "known_item_note": known_note,
        "query_notes": ["Stage 1 ran twice. The first run's known-item check required the author's surname to match, "
                        "which rejected works whose author the catalog stores transliterated (e.g. Cyrillic title, "
                        "\"Fyodor Dostoyevsky\"); the rule was corrected to accept an exact title match, the generator "
                        "calls were reused from the cache, and only the replacements still needed were requested again. "
                        "The 18 replacement calls of the first run are paid and included in the spend, but their output "
                        "is not used.",
                        "No stratum D query was generated with a book-related request; reclassification after judging "
                        "is reported in section 9."],
        "contrast_notes": [],
        "time_notes": ["The frozen query embeddings of qwen were made while DeepInfra was slow (repeated 30 s read "
                       "timeouts, about 20 minutes for 168 queries); that is why the embedding stage dominates the "
                       "latency table and its p95. The operational table of section 8 measures the same model with "
                       "the deployed deadlines one hour later."],
        "deviations": DEVIATIONS,
        "limits": LIMITS,
        "files": [
            "`app/evaluation/catalog_evaluation/` (code), `tests/evaluation/test_catalog_evaluation.py` (tests).",
            "`data/evaluation/catalog50k/queries/` (query set, checks, generator calls), `runs/` (run files), "
            "`query_embeddings/` (frozen query vectors), `pool.json`, `judgments/` (judgment caches with provenance), "
            "`operational/`, `explanations/`, `annotation/`, `parity.json`, `spend_ledger.jsonl`, `results.json`, "
            "`per_query_metrics.json`.",
            "`data/catalog_deploy/bge_m3/` (E2 vectors, git-ignored).",
            "Regenerate this report: `.venv/bin/python -m app.evaluation.catalog_evaluation report --offline`.",
        ],
    }
