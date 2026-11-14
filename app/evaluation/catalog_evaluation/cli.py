"""Command line of the catalog evaluation: one subcommand per stage."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.design import SPEND_CAP_USD
from app.evaluation.catalog_evaluation.ledger import LEDGER_FILE, SpendLedger


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _load_env() -> None:
    """Read .env (secrets stay in the process environment; nothing is printed)."""
    from dotenv import load_dotenv

    load_dotenv(Path(".env"))


def _ledger(data_dir: Path) -> SpendLedger:
    return SpendLedger(data_dir / LEDGER_FILE, SPEND_CAP_USD)


def _key_snapshot(ledger: SpendLedger, stage: str, when: str) -> None:
    """Record the key's cumulative usage reported by OpenRouter (cross-check of the ledger)."""
    from app.evaluation.llm_model_selection import key_usage_usd

    value = key_usage_usd()
    ledger.add(stage, "key_usage", "-", None, key_usage_usd=value, when=when)


def _stage(stage: str, data_dir: Path, run: Callable[[SpendLedger], Any]) -> int:
    _load_env()
    ledger = _ledger(data_dir)
    _key_snapshot(ledger, stage, "start")
    started = time.perf_counter()
    try:
        run(ledger)
    finally:
        _key_snapshot(ledger, stage, "end")
        timing_path = data_dir / "stage_times.jsonl"
        with timing_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": stage, "seconds": round(time.perf_counter() - started, 1),
                                     "ended": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        log(f"Recorded spend so far: {ledger.total():.4f} USD")
    return 0


def cmd_queries(args: argparse.Namespace) -> int:
    from app.evaluation.catalog_evaluation.generation import generate_queries

    def run(ledger: SpendLedger) -> None:
        test_queries = json.loads(Path(args.test_queries).read_text(encoding="utf-8"))
        document = generate_queries(test_queries, args.data_dir / "queries", ledger, log=log)
        log(f"Final problems: {document['final_checks']['problems']}")
        log(f"Fixes: {len(document['fixes'])}")

    return _stage("1-queries", args.data_dir, run)


def _queries(data_dir: Path):
    from app.evaluation.catalog_evaluation.queries import load_queries

    return load_queries(data_dir / "queries" / "queries.json")


def _context(args: argparse.Namespace, ledger: SpendLedger, stage: str):
    from app.evaluation.catalog_evaluation.pipeline import Context

    return Context(args.data_dir, ledger, stage, log)


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def cmd_parity(args: argparse.Namespace) -> int:
    from app.evaluation.catalog_evaluation.design import SCHEMA
    from app.evaluation.catalog_evaluation.pipeline import explain_exact_scan
    from app.evaluation.catalog_evaluation.runs import check_parity

    def run(ledger: SpendLedger) -> None:
        ctx = _context(args, ledger, "2-runs")
        try:
            size = ctx.db.query("SELECT pg_database_size(current_database()) AS bytes")[0]["bytes"]
            parity = check_parity(ctx.db, SCHEMA, ctx.catalog)
            explain = explain_exact_scan(ctx)
            info = ctx.vector("approximate").index_type()
        finally:
            ctx.close()
        document = {"database_bytes": size, "parity": parity, "explain": explain, "vector_index": info,
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _write_json(args.data_dir / "parity.json", document)
        log(json.dumps({k: v for k, v in document.items() if k != "explain"}, default=str))
        log(f"exact plan sequential: {explain['exact_is_sequential_scan']}; approximate uses ANN: "
            f"{explain['approximate_uses_ann_index']}")

    return _stage("2-parity", args.data_dir, run)


def cmd_embed(args: argparse.Namespace) -> int:
    from app.evaluation.catalog_evaluation.design import BGE_EMBEDDING, QWEN_EMBEDDING
    from app.evaluation.catalog_evaluation.pipeline import build_e2, embed_queries

    def run(ledger: SpendLedger) -> None:
        ctx = _context(args, ledger, "2-embed")
        queries = _queries(args.data_dir)
        for model in (QWEN_EMBEDDING, BGE_EMBEDDING):
            embed_queries(ctx, model, queries)
        if not args.queries_only:
            build_e2(ctx)

    return _stage("2-embed", args.data_dir, run)


def cmd_runs(args: argparse.Namespace) -> int:
    from app.evaluation.catalog_evaluation import pipeline

    def run(ledger: SpendLedger) -> None:
        ctx = _context(args, ledger, "2-runs")
        queries = _queries(args.data_dir)
        try:
            if "db" in args.parts:
                pipeline.run_database_systems(ctx, queries)
        finally:
            ctx.close()
        if "e2" in args.parts:
            pipeline.run_e2(ctx, queries)
        if "rerank" in args.parts:
            pipeline.run_reranked(ctx, queries)
        if "llm" in args.parts:
            pipeline.run_llm(ctx, queries)

    return _stage("2-runs", args.data_dir, run)


def cmd_judge(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from app.evaluation.catalog_evaluation import design as D
    from app.evaluation.catalog_evaluation.judging import build_pool, judge_cache, judge_pool, needs_of
    from app.evaluation.catalog_evaluation.results import JUDGING_RUNS, POOL_FILE
    from app.evaluation.catalog_evaluation.runs import LocalCatalog, load_runs

    def run(ledger: SpendLedger) -> None:
        queries = _queries(args.data_dir)
        runs = load_runs(args.data_dir, D.POOL_SYSTEMS)
        missing = {s: len(queries) - len(r) for s, r in runs.items() if len(r) < len(queries)}
        if missing:
            raise SystemExit(f"Run files incomplete: {missing}")
        pool = build_pool(queries, runs)
        catalog = LocalCatalog(D.SOURCE_CATALOG)
        needs = needs_of(queries)
        document, tasks = {}, []
        for need_id, ids in pool.items():
            books = catalog.books(ids)
            document[need_id] = [{"work_id": i, "source": b.source, "source_id": b.source_id or ""} for i, b in zip(ids, books)]
            tasks += [(need_id, needs[need_id].need, b) for b in books]
        _write_json(args.data_dir / POOL_FILE, document)
        log(f"Pool: {len(pool)} needs, {len(tasks)} pairs")

        def judge(model: str):
            stats = judge_pool(model, tasks, judge_cache(args.data_dir, model), ledger, log=log)
            stats["date"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            with (args.data_dir / "judgments" / JUDGING_RUNS).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stats) + "\n")
            log(json.dumps(stats))

        with ThreadPoolExecutor(max_workers=2) as pool_:
            list(pool_.map(judge, D.JUDGES))

    return _stage("3-judge", args.data_dir, run)


def cmd_operational(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from app.evaluation.catalog_evaluation.design import BGE_EMBEDDING, QWEN_EMBEDDING
    from app.evaluation.catalog_evaluation.extras import measure_operational

    def run(ledger: SpendLedger) -> None:
        ctx = _context(args, ledger, "5-operational")
        queries = _queries(args.data_dir)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda m: measure_operational(ctx, m, queries), (QWEN_EMBEDDING, BGE_EMBEDDING)))
        finally:
            ctx.close()

    return _stage("5-operational", args.data_dir, run)


def cmd_explain(args: argparse.Namespace) -> int:
    from app.evaluation.catalog_evaluation.extras import run_explanations

    def run(ledger: SpendLedger) -> None:
        run_explanations(_context(args, ledger, "6-explanations"), _queries(args.data_dir))

    return _stage("6-explanations", args.data_dir, run)


def cmd_annotation(args: argparse.Namespace) -> int:
    from app.evaluation.catalog_evaluation import design as D
    from app.evaluation.catalog_evaluation.annotation import build_sheet
    from app.evaluation.catalog_evaluation.results import POOL_FILE, load_grades
    from app.evaluation.catalog_evaluation.runs import LocalCatalog

    queries = _queries(args.data_dir)
    pool = json.loads((args.data_dir / POOL_FILE).read_text(encoding="utf-8"))
    grades = load_grades(args.data_dir, D.PRIMARY_JUDGE, pool)
    ids = sorted({w for v in grades.values() for w in v})
    catalog = LocalCatalog(D.SOURCE_CATALOG)
    books = dict(zip(ids, catalog.books(ids)))
    summary = build_sheet(queries, grades, books, args.data_dir / "annotation")
    _write_json(args.data_dir / "annotation" / "summary.json", summary)
    log(json.dumps(summary))
    return 0


def cmd_annotation_agreement(args: argparse.Namespace) -> int:
    from app.evaluation.catalog_evaluation import design as D
    from app.evaluation.catalog_evaluation.annotation import KEY_FILE, annotation_agreement
    from app.evaluation.catalog_evaluation.results import POOL_FILE, load_grades

    pool = json.loads((args.data_dir / POOL_FILE).read_text(encoding="utf-8"))
    key = json.loads((args.data_dir / "annotation" / KEY_FILE).read_text(encoding="utf-8"))
    judges = {m: load_grades(args.data_dir, m, pool) for m in D.JUDGES}
    languages = {q.query_id: q.language for q in _queries(args.data_dir)}
    document = annotation_agreement(Path(args.sheet), key, judges, languages)
    _write_json(args.output or args.data_dir / "annotation" / "human_agreement.json", document)
    for model, row in document["judges"].items():
        overall = row.get("overall") or {}
        log(f"{model}: {row['pairs']} pairs, quadratic kappa {overall.get('quadratic_weighted_kappa')}, "
            f"exact {overall.get('exact_agreement')}")
    return 0


def _judge_grades(data_dir: Path) -> Dict[str, Any]:
    """{judge model: {need_id: {work_id: grade}}} from the saved judgment caches (no network)."""
    from app.evaluation.catalog_evaluation.results import POOL_FILE, load_grades

    pool = json.loads((data_dir / POOL_FILE).read_text(encoding="utf-8"))
    return {m: load_grades(data_dir, m, pool) for m in D.JUDGES}


def cmd_order_sheet(args: argparse.Namespace) -> int:
    """V31: the blind two-block sheet of the top 5 of S2-S5, its key, instructions and power (no network)."""
    from app.evaluation.catalog_evaluation import annotation_order as O
    from app.evaluation.catalog_evaluation.annotation import KEY_FILE as EARLIER_KEY
    from app.evaluation.catalog_evaluation.annotation import read_sheet
    from app.evaluation.catalog_evaluation.runs import LocalCatalog, load_runs

    queries = _queries(args.data_dir)
    block_1, sample_1 = O.select_queries(queries)
    block_2, sample_2 = O.remaining_queries(queries, block_1)
    lists = O.top_lists(block_1 + block_2, load_runs(args.data_dir, O.ORDER_SYSTEMS))
    ids = sorted({w for by in lists.values() for top in by.values() for w in top})
    books = dict(zip(ids, LocalCatalog(D.SOURCE_CATALOG).books(ids)))
    earlier_key = json.loads((args.data_dir / "annotation" / EARLIER_KEY).read_text(encoding="utf-8"))
    out_dir = args.data_dir / O.ORDER_DIR
    sheet = O.build_order_sheet([block_1, block_2], lists, books, out_dir, earlier_key)
    earlier_filled = args.data_dir / O.EARLIER_FILLED_SHEET
    if earlier_filled.exists():
        graded = read_sheet(earlier_filled)
        for block in sheet["blocks"].values():
            overlap = block["overlap_with_earlier_sheet"]
            overlap["graded_in_earlier_sheet"] = sum(1 for p in overlap["earlier_pair_ids"] if p in graded)
    key = json.loads((out_dir / O.KEY_FILE).read_text(encoding="utf-8"))
    judges = _judge_grades(args.data_dir)
    power = {"block_1": O.design_power(key, judges, [q.query_id for q in block_1]),
             "blocks_1_and_2": O.design_power(key, judges)}
    _write_json(out_dir / O.SAMPLE_FILE, {"sample": {"block_1": sample_1, "block_2": sample_2},
                                          "sheet": sheet, "power": power})
    for name, block in sheet["blocks"].items():
        log(f"{name}: {block['queries']} queries, {block['pairs']} pairs ({block['pairs_per_query_summary']}), "
            f"overlap with the 200-pair sheet {block['overlap_with_earlier_sheet']['pairs']}, "
            f"{block['time_estimate']['minutes']} minutes")
    for name, block in power.items():
        for model, row in block["by_judge"].items():
            log(f"{name} ({block['queries']} queries) {model}: " + ", ".join(
                f"{c} mean {r['mean_difference']:+.3f} sd {r['sd']:.3f} mde {r['minimum_detectable_difference']:.3f}"
                for c, r in row["contrasts"].items()))
    log(f"Wrote {out_dir / O.SHEET_FILE}, {O.KEY_FILE}, {O.INSTRUCTIONS_FILE}, {O.SAMPLE_FILE}")
    return 0


def cmd_order_agreement(args: argparse.Namespace) -> int:
    """V31: the declared analysis of a filled order sheet (no network)."""
    from app.evaluation.catalog_evaluation import annotation_order as O
    from app.evaluation.catalog_evaluation.runs import load_runs

    out_dir = args.data_dir / O.ORDER_DIR
    key = json.loads((out_dir / O.KEY_FILE).read_text(encoding="utf-8"))
    mismatched = O.check_key_against_runs(key, load_runs(args.data_dir, O.ORDER_SYSTEMS))
    if mismatched:
        raise SystemExit(f"The key does not match the stored run files for {mismatched}")
    earlier = Path(args.earlier_sheet) if args.earlier_sheet else args.data_dir / O.EARLIER_FILLED_SHEET
    second = Path(args.second_sheet) if args.second_sheet else None
    document = O.order_agreement(Path(args.sheet), key, _judge_grades(args.data_dir), earlier, second)
    _write_json(args.output or out_dir / O.RESULT_FILE, document)
    cells = document["cells"][O.FIRST]
    log(f"{cells['graded']} of {cells['rows']} pairs graded {cells['graded_by_block']}, {cells['empty']} empty, "
        f"{len(cells['invalid'])} invalid")
    block = document[O.FIRST]
    selection = block["selection"]
    log(f"Primary set: {selection['primary_set']} ({len(selection['primary'])} queries); "
        f"partial block 2 queries not used: {selection['partial_not_used']}")
    for name, figures in block["primary"]["figures"].items():
        log(f"{name}: " + ", ".join(f"{c} {r.get('difference', float('nan')):+.3f}"
                                    for c, r in figures["contrasts"].items()) + f"; order {figures['ordering']}")
    log(f"Reading rule: {block['primary']['reading_rule']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from app.evaluation.catalog_evaluation.report import build_markdown
    from app.evaluation.catalog_evaluation.report_meta import report_meta
    from app.evaluation.catalog_evaluation.results import build_results

    if not args.offline:
        log("The report never calls an API or the database; --offline only states it.")
    results = build_results(args.data_dir)
    _write_json(args.data_dir / "results.json", {k: v for k, v in results.items() if k != "per_query"})
    _write_json(args.data_dir / "per_query_metrics.json", results["per_query"])
    meta = report_meta(results)
    # The report follows --data-dir unless given explicitly (see design.report_md).
    output = D.report_md(args.data_dir) if args.output is None else args.output
    output.write_text(build_markdown(results, meta), encoding="utf-8")
    log(f"Wrote {output}")
    return 0


def cmd_scope(args: argparse.Namespace) -> int:
    """V29 part A: the post hoc scope analysis on subsets of the saved results (no network)."""
    from app.evaluation.catalog_evaluation.scope import write_scope

    out_dir = args.data_dir / "scope"
    write_scope(args.data_dir, out_dir)
    log(f"Wrote {out_dir / 'SCOPE.md'} and {out_dir / 'scope.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation.catalog_evaluation", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=D.DATA_DIR)
    sub = parser.add_subparsers(dest="stage", required=True)
    q = sub.add_parser("queries", help="stage 1: generate and check strata B, C, D")
    q.add_argument("--test-queries", default="app/evaluation/test_queries.json")
    q.set_defaults(func=cmd_queries)
    sub.add_parser("parity", help="local catalog copy vs Neon, EXPLAIN of E1").set_defaults(func=cmd_parity)
    e = sub.add_parser("embed", help="frozen query embeddings and E2 document vectors")
    e.add_argument("--queries-only", action="store_true")
    e.set_defaults(func=cmd_embed)
    r = sub.add_parser("runs", help="run files of every system")
    r.add_argument("--parts", nargs="+", default=["db", "e2", "rerank", "llm"], choices=["db", "e2", "rerank", "llm"])
    r.set_defaults(func=cmd_runs)
    sub.add_parser("judge", help="pool and judge with both judges").set_defaults(func=cmd_judge)
    sub.add_parser("operational", help="embedding operational measurements").set_defaults(func=cmd_operational)
    sub.add_parser("explain", help="explanation check").set_defaults(func=cmd_explain)
    sub.add_parser("annotation", help="blind human annotation sheet").set_defaults(func=cmd_annotation)
    a = sub.add_parser("annotation-agreement", help="agreement of a filled sheet with each judge")
    a.add_argument("--sheet", required=True)
    a.add_argument("--output", type=Path, default=None)
    a.set_defaults(func=cmd_annotation_agreement)
    sub.add_parser("order-sheet", help="V31: blind sheet of the top 5 of S2-S5 on es/en queries, two blocks (no network)"
                   ).set_defaults(func=cmd_order_sheet)
    o = sub.add_parser("order-agreement", help="V31: system order with a person's grades vs the judges (no network)")
    o.add_argument("--sheet", required=True, help="the filled order sheet (column grade, optionally grade_2)")
    o.add_argument("--second-sheet", default=None, help="a separate copy where a second person filled grade_2")
    o.add_argument("--earlier-sheet", default=None,
                   help="the filled 200-pair sheet (default: annotation/annotation_sheet_filled.csv)")
    o.add_argument("--output", type=Path, default=None)
    o.set_defaults(func=cmd_order_agreement)
    rp = sub.add_parser("report", help="results.json and the markdown report from saved files")
    rp.add_argument("--offline", action="store_true", help="no API call and no database connection (always true)")
    rp.add_argument("--output", type=Path, default=None,
                    help=f"default: {D.REPORT_MD.name} next to --data-dir ({D.REPORT_MD})")
    rp.set_defaults(func=cmd_report)
    sub.add_parser("scope", help="post hoc scope analysis on subsets of the queries (V29, no network)"
                   ).set_defaults(func=cmd_scope)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    return args.func(args)
