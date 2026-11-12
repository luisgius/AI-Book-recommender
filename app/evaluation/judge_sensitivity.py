"""
Do the V06 ablation conclusions depend on the LLM that graded the holes? (V10)

V06 graded with the local judge (qwen3.8:27b-mlx) the 919 (query, book)
pairs of the extended judgments that have no original judgment; the seven
pre-declared contrasts were tested on the 637 original + those 919 grades.
This module re-grades exactly the same 919 pairs with another judge (by
default the one configured in JUDGE_* / .env), writes a second judgments
file with the 637 original judgments and the new judge's grades, and, once
retrieval_ablation.py has been re-run on that file, compares the seven
contrasts under both judges.

A contrast's V06 conclusion is its direction (sign of the nDCG@10
difference) and whether it is significant after the Holm correction
(p < 0.05). The conclusion holds under the new judge when both are the same.

CLI (from the repository root):

    python -m app.evaluation.judge_sensitivity grade --workers 4
    python -m app.evaluation.retrieval_ablation --no-judge \\
        --extended-judgments-path app/evaluation/relevance_judgments_extended_<judge-slug>.json \\
        --llm-pairs-from app/evaluation/relevance_judgments_extended.json \\
        --output-json data/evaluation/results_retrieval_ablation_<judge-slug>.json --output-md ""
    python -m app.evaluation.judge_sensitivity report --judge-model <judge-model>

Both judges are compared on the same pair set: the pairs the old judge
graded by LLM for the V06 pool. Grades that other tasks later added to the
new judge's file (V07 added 146) are left out, by --llm-pairs-from in the
ablation run and by build_document in the report.

`grade` calls the judge (cached per model in data/evaluation/llm_judgments/,
resumable); `report` calls no model and writes
data/evaluation/RETRIEVAL_ABLATION_JUDGE_SENSITIVITY.md.
"""

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.evaluation.comparison import _git
from app.evaluation.relevance_judge import (
    GRADES,
    JudgeTask,
    JudgingStats,
    JudgmentCache,
    agreement_report,
    judge_pairs,
    merge_judgments,
)

DEFAULT_DB_PATH = "data/catalog.db"
DEFAULT_QUERIES_PATH = "app/evaluation/test_queries.json"
DEFAULT_JUDGMENTS_PATH = "app/evaluation/relevance_judgments.json"
DEFAULT_EXTENDED_JUDGMENTS = "app/evaluation/relevance_judgments_extended.json"
DEFAULT_CACHE_DIR = "data/evaluation/llm_judgments"
DEFAULT_ABLATION_JSON = "data/evaluation/results_retrieval_ablation.json"
DEFAULT_RESULTS_JSON = "data/evaluation/results_judge_sensitivity.json"
DEFAULT_OUTPUT_MD = "data/evaluation/RETRIEVAL_ABLATION_JUDGE_SENSITIVITY.md"

ALPHA = 0.05

PairKey = Tuple[str, str, str]  # (query_id, source, source_id)


def judge_slug(model: str) -> str:
    """File-name slug of a judge model id ("z-ai/glm-5.3" -> "z-ai-glm-5.3")."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def new_judgments_path(model: str) -> str:
    """Second extended-judgments file, graded by `model`."""
    return f"app/evaluation/relevance_judgments_extended_{judge_slug(model)}.json"


def new_ablation_json(model: str) -> str:
    """retrieval_ablation.py output on the judgments graded by `model`."""
    return f"data/evaluation/results_retrieval_ablation_{judge_slug(model)}.json"


def llm_grades(raw: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[PairKey, int]:
    """(query_id, source, source_id) -> grade of every LLM judgment in a judgments file."""
    return {
        (query_id, item["source"], item["source_id"]): int(item["relevance"])
        for query_id, items in raw.items()
        for item in items
        if item.get("judge") == "llm"
    }


def llm_models(raw: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
    """Models of the LLM judgments of a judgments file."""
    return sorted({item["model"] for items in raw.values() for item in items if item.get("judge") == "llm"})


def direction(difference: float) -> str:
    """Sign of a mean difference: "+", "-" or "0"."""
    return "+" if difference > 0 else "-" if difference < 0 else "0"


def contrast_conclusion(contrast: Mapping[str, Any], alpha: float = ALPHA) -> Dict[str, Any]:
    """Direction and significance (Holm-adjusted p < alpha) of one contrast."""
    return {
        "direction": direction(contrast["mean_difference"]),
        "significant_holm": contrast["p_value_holm"] < alpha,
        "significant_unadjusted": contrast["p_value"] < alpha,
    }


def compare_contrasts(
    old_report: Mapping[str, Any], new_report: Mapping[str, Any], alpha: float = ALPHA
) -> List[Dict[str, Any]]:
    """
    The pre-declared contrasts side by side under both judges.

    A conclusion holds when the direction and the Holm-adjusted significance
    are the same under both judges.
    """
    questions = {c["name"]: c["question"] for c in old_report["design"]["contrasts"]}
    rows = []
    for name, old in old_report["contrasts"]["final"].items():
        new = new_report["contrasts"]["final"][name]
        if (old["system"], old["reference"]) != (new["system"], new["reference"]):
            raise ValueError(f"Contrast {name} compares different systems in the two reports")
        old_conclusion, new_conclusion = contrast_conclusion(old, alpha), contrast_conclusion(new, alpha)
        rows.append({
            "name": name,
            "question": questions.get(name, ""),
            "system": old["system"],
            "reference": old["reference"],
            "old": {**old, **old_conclusion},
            "new": {**new, **new_conclusion},
            "holds": (old_conclusion["direction"], old_conclusion["significant_holm"])
            == (new_conclusion["direction"], new_conclusion["significant_holm"]),
        })
    return rows


def same_runs(old_report: Mapping[str, Any], new_report: Mapping[str, Any]) -> bool:
    """True when every system returned the same top 10 in both ablation runs."""
    return old_report.get("runs") == new_report.get("runs")


def _reproduction_matches(report: Mapping[str, Any], label: str) -> Optional[bool]:
    """Whether the ablation's V05 reproduction check matched for one judgment set."""
    check = (report.get("reproduction_check") or {}).get(label)
    if check is None:
        return None
    return all(metric["match"] for system in check.values() for metric in system.values())


def conclusion_text(row: Mapping[str, Any]) -> str:
    """One plain sentence on whether a contrast's conclusion holds."""
    old, new = row["old"], row["new"]

    def verdict(block: Mapping[str, Any]) -> str:
        word = {"+": "improves", "-": "lowers", "0": "does not change"}[block["direction"]]
        significance = "significant after Holm" if block["significant_holm"] else "not significant after Holm"
        return f"{word} nDCG@10 ({block['mean_difference']:+.3f}, p (Holm) {block['p_value_holm']:.3f}, {significance})"

    if row["holds"]:
        return f"{row['name']} holds: `{row['system']}` {verdict(old)} with the old judge and {verdict(new)} with the new one."
    return f"{row['name']} does not hold: `{row['system']}` {verdict(old)} with the old judge but {verdict(new)} with the new one."


# =============================================================================
# Markdown
# =============================================================================


def _f(value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def build_markdown(document: Mapping[str, Any]) -> str:
    """RETRIEVAL_ABLATION_JUDGE_SENSITIVITY.md from the sensitivity document."""
    old_judge, new_judge = document["old_judge"], document["new_judge"]
    agreement = document["agreement"]
    overall = agreement["overall"]
    rows = document["contrasts"]
    grading = document.get("grading") or {}

    contrast_rows = []
    for row in rows:
        old, new = row["old"], row["new"]
        contrast_rows.append([
            f"{row['name']}: {row['question']}",
            f"{_f(old['mean_difference'], True)} [{_f(old['ci_low'], True)}, {_f(old['ci_high'], True)}]",
            _p(old["p_value"]), _p(old["p_value_holm"]),
            f"{_f(new['mean_difference'], True)} [{_f(new['ci_low'], True)}, {_f(new['ci_high'], True)}]",
            _p(new["p_value"]), _p(new["p_value_holm"]),
            "yes" if row["holds"] else "**no**",
        ])
    system_rows = [
        [f"`{name}`", _f(values["old"]), _f(values["new"]), _f(values["new"] - values["old"], True)]
        for name, values in document["systems_ndcg"].items()
    ]
    agreement_rows = []
    for label, block in [("all", overall)] + [(f"{lang} queries", b) for lang, b in agreement["by_language"].items()]:
        bci = block["quadratic_weighted_kappa_ci95"]
        agreement_rows.append([
            label, str(block["n"]), _f(block["exact_agreement"]), _f(block["within_one_agreement"]),
            f"{_f(block['quadratic_weighted_kappa'])}" + (f" [{bci[0]:.3f}, {bci[1]:.3f}]" if bci else ""),
            _f(block["binary"]["at_least_1"]["kappa"]), _f(block["binary"]["at_least_2"]["kappa"]),
            _f(block["mean_signed_difference"], True),
        ])
    confusion = overall["confusion_matrix"]
    confusion_rows = [[str(g)] + [str(c) for c in counts] + [str(sum(counts))] for g, counts in zip(GRADES, confusion)]
    confusion_rows.append(["Total"] + [str(sum(col)) for col in zip(*confusion)] + [str(sum(map(sum, confusion)))])
    confusion_table = _table(["Old \\ New", "0", "1", "2", "3", "Total"], confusion_rows)
    crossed = [
        f"{row['name']} ({_p(row['old']['p_value'])} to {_p(row['new']['p_value'])})"
        for row in rows if row["old"]["significant_unadjusted"] != row["new"]["significant_unadjusted"]
    ]
    unadjusted_line = (
        f"\nNot part of the conclusions (the V06 family is Holm-corrected), but worth knowing: the "
        f"unadjusted p crosses {ALPHA} between the judges for {', '.join(crossed)}."
        if crossed else f"\nThe unadjusted p is on the same side of {ALPHA} under both judges for every contrast."
    )
    holding = [row["name"] for row in rows if row["holds"]]
    failing = [row["name"] for row in rows if not row["holds"]]
    grading_line = ""
    if grading:
        cost = grading.get("reported_cost_usd")
        seconds = grading.get("seconds_per_judgment")
        grading_line = (
            f"\n- Grading run of {grading.get('date', 'n/a')}: {grading.get('judged_in_this_run')} pairs judged "
            f"({grading.get('already_cached')} already cached), {grading.get('failed')} judge failures, "
            + (f"{seconds:.2f} s per judgment ({grading.get('workers', 1)} parallel judges, each one request "
               f"at a time), " if seconds else "")
            + (f"real cost {cost:.4f} USD (OpenRouter)." if cost is not None else "cost not reported.")
        )
    reproduction = document.get("reproduction_original_matches")
    reproduction_line = (
        "" if reproduction is None else
        "\n- The re-run reproduces the V05 numbers on the original judgments"
        + (" (all match); " if reproduction else " **only partly**; ")
        + "its check on the V05 extended judgments reports a mismatch, as expected, because those "
          "numbers include the old judge's grades."
    )
    excluded = new_judge.get("num_llm_excluded", 0)
    pinning_line = (
        f"- Pair set pinned to the {new_judge['num_llm']} pairs both judges graded for the V06 ablation pool "
        f"(the old judge's LLM-graded pairs). The new judge's file now holds {new_judge.get('num_llm_in_file')} "
        f"LLM grades: the other {excluded} were added later by other tasks for pairs the old judge never "
        f"graded, so they are left out of the agreement and of the new ablation run "
        f"(`retrieval_ablation --llm-pairs-from {old_judge['judgments_path']}`); the file itself is unchanged."
        if excluded else
        f"- Pair set: the {new_judge['num_llm']} pairs both judges graded for the V06 ablation pool "
        f"(the new judge's file holds no other LLM grade)."
    )
    runs_line = ("Both ablation runs returned the same top 10 for every system and query, so only the "
                 "judgments differ." if document["same_runs"] else
                 "**The two ablation runs did not return the same top 10** (catalog or index changed?); "
                 "the comparison mixes retrieval and judge effects.")

    return f"""# Sensitivity of the V06 ablation to the LLM judge

- Date: {document["date"]}
- Git commit: `{document.get("git_commit") or "unknown"}`
- Old judge (V06): `{", ".join(old_judge["models"])}`, {old_judge["num_llm"]} LLM grades in
  `{old_judge["judgments_path"]}`.
- New judge: `{new_judge["model"]}` (OpenRouter), prompt `{new_judge["prompt_version"]}`, the same
  {new_judge["num_llm"]} pairs, in `{new_judge["judgments_path"]}` (cache
  `{new_judge["cache_file"]}`).{grading_line}
{pinning_line}
- Both files hold the same {document["num_original"]} original judgments; only the LLM grades of the
  {overall["n"]} holes differ. {runs_line}{reproduction_line}
- Contrasts, statistics and Holm family exactly as in `RETRIEVAL_ABLATION.md` (two-sided paired
  randomisation test, 10000 resamples, seed 0, Holm over the seven contrasts, paired bootstrap 95% CI).
- A conclusion is the direction of the difference and whether it is significant after Holm
  (p < {ALPHA}); it holds when both are the same under the two judges.

## Agreement between the two judges on the {overall["n"]} pairs

{_table(["Pairs", "n", "Exact", "Within 1", "Quadratic-weighted kappa [95% CI]", "Kappa (grade >= 1)",
         "Kappa (grade >= 2)", "Mean new - old"], agreement_rows)}

Grades 0 / 1 / 2 / 3: old judge {" / ".join(map(str, overall["grade_distribution"]["original"]))}; new judge
{" / ".join(map(str, overall["grade_distribution"]["llm"]))}.

Confusion matrix (rows: old judge, columns: new judge):

{confusion_table}

The kappa interval resamples pairs and ignores that pairs of one query are not independent.

## The seven pre-declared contrasts under both judges (nDCG@10, system - reference)

{_table(["Contrast", "Old judge: diff. [95% CI]", "p", "p (Holm)", "New judge: diff. [95% CI]", "p",
         "p (Holm)", "Conclusion holds"], contrast_rows)}

## nDCG@10 of every system under both judges (descriptive)

{_table(["System", "Old judge", "New judge", "New - old"], system_rows)}

## Which V06 conclusions hold

{chr(10).join(f"- {conclusion_text(row)}" for row in rows)}

In short: {len(holding)} of {len(rows)} conclusions hold under both judges
({", ".join(holding) or "none"}){"; " + ", ".join(failing) + " do not" if failing else ""}.
{unadjusted_line}

## Reproduce

    python -m app.evaluation.judge_sensitivity grade --workers 4
    python -m app.evaluation.retrieval_ablation --no-judge --extended-judgments-path {new_judge["judgments_path"]} --llm-pairs-from {old_judge["judgments_path"]} --output-json {document["new_ablation_json"]} --output-md ""
    python -m app.evaluation.judge_sensitivity report --judge-model {new_judge["model"]}

The first command calls the judge set in `JUDGE_PROVIDER` / `JUDGE_MODEL` (see `.env.example`) and
resumes from its cache; the other two call no LLM.
"""


# =============================================================================
# CLI
# =============================================================================


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _judge_from_env():
    from app.evaluation.relevance_judge import _build_judge

    return _build_judge()


class LockedCache:
    """A JudgmentCache shared by worker threads: membership and appends under one lock."""

    def __init__(self, cache: JudgmentCache) -> None:
        self._cache = cache
        self._lock = threading.Lock()

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._cache

    def add(self, judgment) -> None:
        with self._lock:
            self._cache.add(judgment)


def judge_in_parallel(
    tasks: Sequence[JudgeTask], judges: Sequence[Any], cache: JudgmentCache, *, limit: Optional[int] = None
) -> JudgingStats:
    """
    judge_pairs on len(judges) round-robin shards of the pending tasks, one
    thread and one judge per shard, so at most len(judges) requests are in
    flight. Each shard is judged one request at a time, so seconds per
    judgment means the same as in judge_pairs. One judge: judge_pairs itself.
    """
    if len(judges) == 1:
        return judge_pairs(tasks, judges[0], cache, limit=limit)
    unique = list({task.key: task for task in tasks}.values())
    todo = [task for task in unique if task.key not in cache]
    batch = todo if limit is None else todo[: max(limit, 0)]
    shared = LockedCache(cache)

    def work(index: int) -> Optional[JudgingStats]:
        try:
            return judge_pairs(batch[index::len(judges)], judges[index], shared)
        except RuntimeError as error:  # several failures in a row in this shard
            print(f"Worker {index} stopped: {error}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=len(judges)) as pool:
        shards = [stats for stats in pool.map(work, range(len(judges))) if stats is not None]
    judged = sum(1 for task in batch if task.key in cache)
    return JudgingStats(
        total=len(unique),
        cached=len(unique) - len(todo),
        judged=judged,
        failed=sum(stats.failed for stats in shards),
        remaining=sum(1 for task in todo if task.key not in cache),
        seconds=sum(stats.seconds for stats in shards),
    )


def run_grade(args: argparse.Namespace) -> int:
    """`grade`: re-grade the LLM-judged pairs of the extended judgments with the configured judge."""
    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
    from app.infrastructure.llm.relevance_judge import LangChainRelevanceJudge

    started = datetime.now(UTC)
    settings, judge = _judge_from_env()
    judges = [judge] + [LangChainRelevanceJudge.from_settings(settings) for _ in range(args.workers - 1)]
    books = {(b.source, b.source_id): b for b in SqliteBookCatalogRepository(Path(args.db_path)).get_all()}
    queries = {q.query_id: q.text for q in load_test_queries(args.queries_path)}
    extended = load_relevance_judgments(args.extended_judgments_path)
    if judge.model_name in llm_models(extended):
        print(f"{judge.model_name} already graded {args.extended_judgments_path}; choose another judge.")
        return 1

    pairs = sorted(llm_grades(extended))
    tasks = [JudgeTask(query_id, queries[query_id], books[(source, source_id)])
             for query_id, source, source_id in pairs]
    cache = JudgmentCache.for_judge(args.cache_dir, judge.model_name, judge.prompt_version)
    stats = judge_in_parallel(tasks, judges, cache, limit=args.limit)
    print(f"Judged {stats.judged} new pairs, {stats.cached} cached, {stats.failed} failed, {stats.remaining} remaining")
    costs = [j.usage_recorder.summary()["total"]["reported_cost_usd"]
             for j in judges if getattr(j, "usage_recorder", None)]
    grading = {
        "date": started.isoformat(timespec="seconds"),
        "judge_provider": settings.provider,
        "judge_model": judge.model_name,
        "judge_reasoning": settings.reasoning,
        "prompt_version": judge.prompt_version,
        "pairs": len(tasks),
        "judged_in_this_run": stats.judged,
        "already_cached": stats.cached,
        "failed": stats.failed,
        "remaining": stats.remaining,
        "seconds_per_judgment": stats.seconds_per_judgment,
        "workers": args.workers,
        "reported_cost_usd": sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None,
        "cache_file": str(cache.path),
    }
    if stats.remaining:
        print("Run the same command again to grade the remaining pairs; no judgments file written.")
        return 1

    output = args.output_judgments or new_judgments_path(judge.model_name)
    merged = merge_judgments(load_relevance_judgments(args.judgments_path), [cache.get(t.key) for t in tasks])
    _write(output, json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    previous = _load_json(args.results_json) if Path(args.results_json).exists() else {}
    _write(args.results_json, json.dumps({**previous, "grading": grading}, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {output} ({sum(len(e['judgments']) for e in merged)} judgments) and {args.results_json}")
    return 0


def build_document(
    old_raw: Mapping[str, Sequence[Mapping[str, Any]]],
    new_raw: Mapping[str, Sequence[Mapping[str, Any]]],
    old_report: Mapping[str, Any],
    new_report: Mapping[str, Any],
    query_languages: Mapping[str, Optional[str]],
    *,
    new_model: str,
    paths: Mapping[str, str],
    n_resamples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """The sensitivity document: judge agreement on the shared pairs and the contrasts side by side."""
    old, new_all = llm_grades(old_raw), llm_grades(new_raw)
    missing = set(old) - set(new_all)
    if missing:
        raise ValueError(f"The new judge did not grade {len(missing)} pairs that the old judge graded")
    # Pinned to the pairs both judges graded for the ablation pool (the old
    # judge's pairs): grades the new judge's file received later for other
    # tasks (V07 added 146) are left out of the agreement and of the ablation.
    new = {k: new_all[k] for k in old}
    excluded = len(new_all) - len(new)
    pinned_llm = new_report.get("judgments", {}).get("num_llm_in_final_file")
    if pinned_llm is not None and pinned_llm != len(old):
        raise ValueError(
            f"The new ablation run used {pinned_llm} LLM grades, not the {len(old)} pairs of the old judge; "
            "re-run it with --llm-pairs-from " + paths["old_judgments"]
        )
    keys = sorted(old)
    agreement = agreement_report(
        [old[k] for k in keys], [new[k] for k in keys], [query_languages.get(k[0]) for k in keys],
        n_resamples=n_resamples, seed=seed,
    )
    num_original = sum(1 for items in old_raw.values() for item in items if item.get("judge") != "llm")
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "old_judge": {"models": llm_models(old_raw), "num_llm": len(old), "judgments_path": paths["old_judgments"]},
        "new_judge": {"model": new_model, "num_llm": len(new), "judgments_path": paths["new_judgments"],
                      "prompt_version": ", ".join(sorted({i.get("prompt_version") for items in new_raw.values()
                                                          for i in items if i.get("judge") == "llm"})),
                      "cache_file": paths.get("cache_file", ""),
                      "num_llm_in_file": len(new_all), "num_llm_excluded": excluded},
        "num_original": num_original,
        "agreement": agreement,
        "agreement_note": "rows / 'original': old judge; columns / 'llm': new judge",
        "contrasts": compare_contrasts(old_report, new_report),
        "systems_ndcg": {
            name: {"old": old_report["systems"][name]["metrics"]["ndcg_at_10"]["mean"],
                   "new": new_report["systems"][name]["metrics"]["ndcg_at_10"]["mean"]}
            for name in old_report["systems"]
        },
        "same_runs": same_runs(old_report, new_report),
        "reproduction_original_matches": _reproduction_matches(new_report, "original"),
        "old_ablation_json": paths["old_ablation"],
        "new_ablation_json": paths["new_ablation"],
    }


def run_report(args: argparse.Namespace) -> int:
    """`report`: compare the contrasts under both judges (no model call)."""
    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.evaluation.relevance_judge import cache_file_name
    from app.infrastructure.llm.prompts_judge import RELEVANCE_JUDGE_PROMPT_VERSION

    model = args.judge_model
    if model is None:
        from dotenv import load_dotenv

        from app.infrastructure.config.settings import JudgeSettings

        load_dotenv(Path(".env"))
        model = JudgeSettings.from_env().model
    paths = {
        "old_judgments": args.extended_judgments_path,
        "new_judgments": args.new_judgments_path or new_judgments_path(model),
        "old_ablation": args.old_ablation_json,
        "new_ablation": args.new_ablation_json or new_ablation_json(model),
        "cache_file": str(Path(args.cache_dir) / cache_file_name(model, RELEVANCE_JUDGE_PROMPT_VERSION)),
    }
    languages = {q.query_id: q.language for q in load_test_queries(args.queries_path)}
    document = build_document(
        load_relevance_judgments(paths["old_judgments"]), load_relevance_judgments(paths["new_judgments"]),
        _load_json(paths["old_ablation"]), _load_json(paths["new_ablation"]), languages,
        new_model=model, paths=paths,
    )
    previous = _load_json(args.results_json) if Path(args.results_json).exists() else {}
    document["grading"] = previous.get("grading")
    _write(args.results_json, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    _write(args.output_md, build_markdown(document))
    for row in document["contrasts"]:
        print(conclusion_text(row))
    print(f"Saved {args.results_json} and {args.output_md}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Sensitivity of the V06 ablation to the LLM judge")
    subcommands = parser.add_subparsers(dest="command", required=True)
    grade = subcommands.add_parser("grade", help="Re-grade the LLM-judged pairs with the configured judge")
    grade.add_argument("--db-path", default=DEFAULT_DB_PATH)
    grade.add_argument("--judgments-path", default=DEFAULT_JUDGMENTS_PATH, help="Original judgments (read only)")
    grade.add_argument("--output-judgments", default=None,
                       help="Default: app/evaluation/relevance_judgments_extended_<judge-slug>.json")
    grade.add_argument("--limit", type=int, default=None, help="Judge at most this many new pairs, then stop")
    grade.add_argument("--workers", type=int, default=1, choices=range(1, 5), metavar="1-4",
                       help="Parallel judges, each one request at a time (default 1)")
    report = subcommands.add_parser("report", help="Compare the contrasts under both judges (no model call)")
    report.add_argument("--judge-model", default=None, help="New judge model id (default: JUDGE_MODEL)")
    report.add_argument("--new-judgments-path", default=None)
    report.add_argument("--old-ablation-json", default=DEFAULT_ABLATION_JSON)
    report.add_argument("--new-ablation-json", default=None,
                        help="Default: data/evaluation/results_retrieval_ablation_<judge-slug>.json")
    report.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    for sub in (grade, report):
        sub.add_argument("--extended-judgments-path", default=DEFAULT_EXTENDED_JUDGMENTS,
                         help="V06 extended judgments (read only)")
        sub.add_argument("--queries-path", default=DEFAULT_QUERIES_PATH)
        sub.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
        sub.add_argument("--results-json", default=DEFAULT_RESULTS_JSON)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "grade":
        return run_grade(args)
    return run_report(args)


if __name__ == "__main__":
    sys.exit(main())
