#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_relevance(value: Any, *, ctx: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Invalid relevance (must be int 0-3 or null) at {ctx}: {value!r}")
    if value < 0 or value > 3:
        raise ValueError(f"Invalid relevance (must be 0-3) at {ctx}: {value}")
    return value


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def build_judgments_from_pool(pool: List[dict]) -> List[dict]:
    out: List[dict] = []

    for qi, q in enumerate(pool):
        query_id = q.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"Missing/invalid query_id at pool[{qi}]")

        candidates = q.get("candidates")
        if candidates is None:
            raise ValueError(f"Missing candidates for query_id={query_id}")
        if not isinstance(candidates, list):
            raise ValueError(f"Invalid candidates type for query_id={query_id}: expected list")

        judgments: List[dict] = []

        for ci, c in enumerate(candidates):
            if not isinstance(c, dict):
                raise ValueError(f"Invalid candidate type at {query_id}.candidates[{ci}]")

            rel = _validate_relevance(c.get("relevance"), ctx=f"{query_id}.candidates[{ci}].relevance")
            if rel is None:
                continue

            source = c.get("source")
            source_id = c.get("source_id")

            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"Missing/invalid source at {query_id}.candidates[{ci}]")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(f"Missing/invalid source_id at {query_id}.candidates[{ci}]")

            judgments.append(
                {
                    "source": source,
                    "source_id": source_id,
                    "relevance": rel,
                    "title": c.get("title"),
                }
            )

        # Sort deterministically: desc relevance, then title, then source_id
        judgments.sort(
            key=lambda j: (
                -int(j["relevance"]),
                _safe_str(j.get("title")).lower(),
                _safe_str(j.get("source_id")).lower(),
            )
        )

        # Drop helper field "title" from output
        cleaned = [
            {"source": j["source"], "source_id": j["source_id"], "relevance": j["relevance"]}
            for j in judgments
        ]

        out.append({"query_id": query_id, "judgments": cleaned})

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build app/evaluation/relevance_judgments.json from a labeled pooling file",
    )
    parser.add_argument(
        "--pool",
        type=str,
        required=True,
        help="Path to data/evaluation/pool_candidates.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for app/evaluation/relevance_judgments.json",
    )

    args = parser.parse_args()

    pool_path = Path(args.pool)
    out_path = Path(args.out)

    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    pool_data = _load_json(pool_path)
    if not isinstance(pool_data, list):
        raise ValueError("pool_candidates.json must be a JSON array")

    judgments = build_judgments_from_pool(pool_data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(judgments, f, indent=2, ensure_ascii=False)

    # Tiny summary to stdout
    labeled_counts = [len(q["judgments"]) for q in judgments]
    total_labeled = sum(labeled_counts)
    print(
        f"Wrote {out_path} | queries={len(judgments)} | labeled_judgments={total_labeled} | "
        f"min_per_query={min(labeled_counts) if labeled_counts else 0} | "
        f"max_per_query={max(labeled_counts) if labeled_counts else 0}"
    )


if __name__ == "__main__":
    main()
