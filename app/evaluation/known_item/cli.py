"""Command line of the known-item evaluation: sample | run | report."""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

from app.evaluation.known_item import design as D


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def cmd_sample(args: argparse.Namespace) -> int:
    """Draw the sample from the local copy of the deployed catalog (read-only, no network)."""
    from app.evaluation.catalog_evaluation.runs import LocalCatalog
    from app.evaluation.known_item.sampling import build_sample

    path = args.data_dir / D.SAMPLE_FILE
    if path.exists() and not args.force:
        log(f"{path} exists; the sample is fixed once drawn (use --force only before any run)")
        return 1
    document = build_sample(LocalCatalog(D.SOURCE_CATALOG).iter_all())
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {path}: {len(document['books'])} books, {len(document['queries'])} queries")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Phase 2: embeddings, database runs, fusions, weight choice and test-half reranking (paid, capped)."""
    from app.evaluation.catalog_evaluation.cli import _key_snapshot, _load_env
    from app.evaluation.catalog_evaluation.ledger import SpendCapReached, SpendLedger
    from app.evaluation.known_item.runner import EstimateAboveCap, run_all

    _load_env()
    ledger = SpendLedger(args.data_dir / D.LEDGER_FILE, D.SPEND_CAP_USD)
    _key_snapshot(ledger, "known-item", "start")
    started = time.perf_counter()
    try:
        run_all(args.data_dir, ledger, log)
    except (SpendCapReached, EstimateAboveCap) as stop:
        log(f"Stopped: {stop}")
        return 2
    finally:
        _key_snapshot(ledger, "known-item", "end")
        log(f"Recorded spend: {ledger.total():.4f} USD; {time.perf_counter() - started:.0f} s")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from app.evaluation.known_item.report import write_report

    write_report(args.data_dir)
    log(f"Wrote {args.data_dir / D.RESULTS_FILE} and {args.data_dir / D.REPORT_FILE}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation.known_item", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=D.DATA_DIR)
    sub = parser.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("sample", help="draw the 300 books and their 900 queries (local, read-only)")
    s.add_argument("--force", action="store_true", help="overwrite an existing sample")
    s.set_defaults(func=cmd_sample)
    sub.add_parser("run", help="phase 2: run every system against the deployed catalog (paid, cap "
                               f"{D.SPEND_CAP_USD:.2f} USD)").set_defaults(func=cmd_run)
    sub.add_parser("report", help="results.json and KNOWN_ITEM.md from the saved files").set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    return args.func(args)
