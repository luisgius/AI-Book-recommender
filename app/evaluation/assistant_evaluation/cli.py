"""Command line of the assistant evaluation: one subcommand per stage."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from app.evaluation.assistant_evaluation import design as D
from app.evaluation.catalog_evaluation.ledger import LEDGER_FILE, SpendCapReached, SpendLedger

STAGE_TIMES = "stage_times.jsonl"


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _load_env() -> None:
    """Read .env into the process environment (secrets are never printed or written)."""
    from dotenv import load_dotenv

    load_dotenv(Path(".env"))


def _ledger(data_dir: Path) -> SpendLedger:
    return SpendLedger(data_dir / LEDGER_FILE, D.spend_cap(data_dir))


def _key_snapshot(ledger: SpendLedger, stage: str, when: str) -> None:
    """The key's cumulative usage reported by OpenRouter (a cross-check of the ledger)."""
    from app.evaluation.llm_model_selection import key_usage_usd

    ledger.add(stage, "key_usage", "-", None, key_usage_usd=key_usage_usd(), when=when)


def _stage(stage: str, data_dir: Path, run: Callable[[SpendLedger], Any]) -> int:
    _load_env()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ledger = _ledger(data_dir)
    _key_snapshot(ledger, stage, "start")
    started = time.perf_counter()
    status = 0
    try:
        run(ledger)
    except SpendCapReached as error:
        log(f"STOPPED: {error}")
        status = 2
    finally:
        _key_snapshot(ledger, stage, "end")
        with (data_dir / STAGE_TIMES).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": stage, "seconds": round(time.perf_counter() - started, 1),
                                     "ended": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": status}) + "\n")
        log(f"Recorded spend so far: {ledger.total():.4f} USD (cap {ledger.cap_usd:.2f})")
    return status


def cmd_sets(args: argparse.Namespace) -> int:
    from app.evaluation.assistant_evaluation.generation import build_sets

    def run(ledger: SpendLedger) -> None:
        document = build_sets(args.data_dir / "sets", ledger, log=log)
        for name in ("E", "F"):
            part = document[name]
            log(f"{name}: first-pass problems {len(part['first_pass']['problems'])}, fixes {len(part['fixes'])}, "
                f"final problems {len(part['final']['problems'])}")
        log(json.dumps(document["sets"]["counts"]))

    return _stage("1-sets", args.data_dir, run)


def cmd_seed_v25(args: argparse.Namespace) -> int:
    from app.evaluation.assistant_evaluation.remeasure import seed

    seed(D.DATA_DIR, D.V25_DIR, log=log)
    return 0


def cmd_set_g(args: argparse.Namespace) -> int:
    from app.evaluation.assistant_evaluation.set_g import build_set_g

    def run(ledger: SpendLedger) -> None:
        document = build_set_g(D.V25_DIR / "sets", ledger, log=log)
        part = document["G"]
        log(f"G: first-pass problems {len(part['first_pass']['problems'])}, fixes {len(part['fixes'])}, "
            f"final problems {len(part['final']['problems'])}; language {part['final']['language_summary']}")

    return _stage("1-set-g", D.V25_DIR, run)


def _system(args: argparse.Namespace, ledger: SpendLedger, stage: str):
    from app.evaluation.assistant_evaluation.system import CloudSystem

    return CloudSystem(args.data_dir, ledger, stage, log=log)


def _requests(data_dir: Path):
    from app.evaluation.assistant_evaluation.sets import load_requests

    return load_requests(data_dir / "sets" / "requests.json")


def cmd_reference(args: argparse.Namespace) -> int:
    from app.evaluation.assistant_evaluation import runs

    def run(ledger: SpendLedger) -> None:
        system = _system(args, ledger, "2-reference")
        try:
            runs.run_reference(system, _requests(args.data_dir), limit=args.limit)
            if not args.skip_depth_check:
                runs.run_depth_check(system, _requests(args.data_dir), limit=args.limit)
        finally:
            system.close()

    return _stage("2-reference", args.data_dir, run)


def cmd_assistant(args: argparse.Namespace) -> int:
    from app.evaluation.assistant_evaluation import runs

    def run(ledger: SpendLedger) -> None:
        if D.is_v25(args.data_dir):
            from app.evaluation.assistant_evaluation.remeasure import current_instruction_matches

            if not current_instruction_matches(args.data_dir):
                raise SystemExit("The routing instruction in the code is not the one recorded for this measurement")
        system = _system(args, ledger, "3-assistant")
        (args.data_dir / "system.json").write_text(json.dumps(system.describe(), indent=2, default=str),
                                                   encoding="utf-8")
        requests = _requests(args.data_dir)
        try:
            for part in args.parts:
                started = time.perf_counter()
                runs.run_part(system, requests, part, limit=args.limit)
                log(f"part {part} done in {time.perf_counter() - started:.0f} s; spend {ledger.total():.4f} USD")
        finally:
            system.close()

    return _stage("3-assistant", args.data_dir, run)


def cmd_judge(args: argparse.Namespace) -> int:
    from app.evaluation.assistant_evaluation.judging import judge_all

    def run(ledger: SpendLedger) -> None:
        judge_all(args.data_dir, _requests(args.data_dir), ledger, log=log)

    return _stage("4-judge", args.data_dir, run)


def cmd_report(args: argparse.Namespace) -> int:
    from app.evaluation.assistant_evaluation.report import build_markdown
    from app.evaluation.assistant_evaluation.report_v25 import build_comparison

    if not args.offline:
        log("The report never calls an API or the database; --offline only states it.")
    results = _write_results(args.data_dir)
    Path(args.output).write_text(build_markdown(results), encoding="utf-8")
    log(f"Wrote {args.output}")
    # The re-measurement (V25), side by side with the first measurement.
    second_dir = D.V25_DIR if args.second_data_dir is None else args.second_data_dir
    if not args.skip_second and (second_dir / "runs" / "turns.jsonl").exists():
        second = _write_results(second_dir)
        instructions = json.loads((second_dir / "instructions.json").read_text(encoding="utf-8"))
        Path(args.second_output).write_text(build_comparison(results, second, instructions, args.data_dir, second_dir), encoding="utf-8")
        log(f"Wrote {args.second_output}")
    return 0


def _write_results(data_dir: Path) -> dict:
    """build_results(data_dir), saved as results.json and per_request.json in data_dir."""
    from app.evaluation.assistant_evaluation.results import build_results

    results = build_results(data_dir)
    (data_dir / "results.json").write_text(json.dumps({k: v for k, v in results.items() if k != "per_request"},
                                                      ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (data_dir / "per_request.json").write_text(json.dumps(results["per_request"], ensure_ascii=False, indent=1,
                                                          default=str), encoding="utf-8")
    return results


PARTS = ("ab1", "d", "f", "e", "ab2", "agent")
V25_PARTS = PARTS + ("g",)  # G exists only in the re-measurement's request sets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation.assistant_evaluation", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=D.DATA_DIR)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("sets", help="stage 1: A, B, D from V14; generate and check E and F").set_defaults(func=cmd_sets)
    r = sub.add_parser("reference", help="stage 2: P on every request text (top 100)")
    r.add_argument("--limit", type=int, default=None, help="only the first N texts (smoke test)")
    r.add_argument("--skip-depth-check", action="store_true")
    r.set_defaults(func=cmd_reference)
    a = sub.add_parser("assistant", help="stage 3: the assistant's turns and the direct agent")
    a.add_argument("--parts", nargs="+", default=list(PARTS), choices=V25_PARTS)
    a.add_argument("--limit", type=int, default=None, help="only the first N requests of each part (smoke test)")
    a.set_defaults(func=cmd_assistant)
    sub.add_parser("judge", help="stage 4: judge every new pair with both judges").set_defaults(func=cmd_judge)
    rp = sub.add_parser("report", help="stage 5: results.json and the markdown report from saved files")
    rp.add_argument("--offline", action="store_true", help="no API call and no database connection (always true)")
    rp.add_argument("--output", default=str(D.REPORT_MD))
    rp.add_argument("--second-data-dir", type=Path, default=None,
                    help=f"the re-measurement's data (default {D.V25_DIR}); its report is written when it has runs")
    rp.add_argument("--second-output", default=str(D.V25_REPORT_MD))
    rp.add_argument("--skip-second", action="store_true", help="only the first measurement's report")
    rp.set_defaults(func=cmd_report)
    sub.add_parser("seed-v25", help=f"V25: copy the reusable files into {D.V25_DIR} and record both instructions"
                   ).set_defaults(func=cmd_seed_v25)
    sub.add_parser("set-g", help="V25: generate and check the held-out set G").set_defaults(func=cmd_set_g)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    return args.func(args)
