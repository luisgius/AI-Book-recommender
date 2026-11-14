"""Tests of the comparison of two known-item runs (V29 replication): pure, no network, no database."""

import base64
import json
from typing import Any, Dict, List

import numpy as np

from app.evaluation.known_item import design as D
from app.evaluation.known_item.cli import main
from app.evaluation.known_item.compare import build_markdown, compare, run_time


def _vec(values: List[float]) -> str:
    return base64.b64encode(np.asarray(values, dtype=np.float32).tobytes()).decode()


def _line(qid: str, ids: List[str]) -> Dict[str, Any]:
    return {"query_id": qid, "ids": ids, "scores": [None] * len(ids), "timings_ms": {}, "error": None, "extra": {}}


QUERIES = [
    {"query_id": "q1-T", "query_type": "T", "half": "test", "language": "en", "text": "x", "relevant": ["a"]},
    {"query_id": "q2-T", "query_type": "T", "half": "test", "language": "en", "text": "y", "relevant": ["b"]},
]


def _folder(name: str, s2_q2: List[str], vec_q2: List[float], rerank_b: float, weight: float) -> Dict[str, Any]:
    runs = {
        D.S1: {"q1-T": _line("q1-T", ["a", "b"]), "q2-T": _line("q2-T", ["b", "a"])},
        D.S2: {"q1-T": _line("q1-T", ["a", "c"]), "q2-T": _line("q2-T", s2_q2)},
        D.S4: {"q1-T": _line("q1-T", ["a", "c"]), "q2-T": _line("q2-T", ["b", "c"])},
    }
    contrast = {"query_type": "T", "system": "S5", "baseline": "S4", "metric": "mrr_at_10", "queries": 2,
                "difference": 0.0, "ci": [0.0, 0.0], "p": 1.0, "p_holm": 1.0, "significant": False}
    return {
        "folder": name, "queries": QUERIES, "runs": runs,
        "embeddings": {"q1-T": {"vector": _vec([1.0, 0.0]), "provider": "p"},
                       "q2-T": {"vector": _vec(vec_q2), "provider": "p"}},
        "rerank": {"q1-T": {"a": 0.9, "c": 0.1}, "q2-T": {"b": rerank_b, "c": 0.2}},
        "ledger": [
            {"date": "2026-11-14T10:00:00+00:00", "operation": "key_usage", "when": "start", "cost_usd": None},
            {"date": "2026-11-14T10:05:00+00:00", "operation": "embedding", "model": "m", "stage": "k", "cost_usd": 0.5},
            {"date": "2026-11-14T10:10:00+00:00", "operation": "key_usage", "when": "end", "cost_usd": None},
        ],
        "results": {"selection": {"chosen_weight": weight, "objective": {"S3w@0.25": 0.9}},
                    "contrasts": {"C1-T": contrast}, "decision": {"per_type": {}}, "deviations": []},
    }


def test_identical_folders_have_no_differences():
    a = _folder("a", ["b", "c"], [0.0, 1.0], 0.8, 0.25)
    c = compare(a, _folder("b", ["b", "c"], [0.0, 1.0], 0.8, 0.25))
    assert all(r["different_full"] == 0 and r["different_top10"] == 0 for r in c["lists"].values())
    assert c["embeddings"]["different"] == 0 and c["rerank"]["different"] == 0
    assert all(not v for v in c["causes"].values())
    assert c["metric_changes_test"]["T"][D.S2]["largest_change"] == 0
    assert c["contrasts"]["C1-T"]["same_decision"] is True
    assert "No hay listas distintas" in build_markdown(c)


def test_a_different_vector_explains_a_different_vector_list():
    a = _folder("a", ["b", "c"], [0.0, 1.0], 0.8, 0.25)
    b = _folder("b", ["c", "b"], [0.1, 1.0], 0.7, 0.75)
    c = compare(a, b)
    assert c["lists"][D.S2] == {"queries": 2, "identical_top10": 1, "identical_full": 1, "different_top10": 1,
                                "different_full": 1, "different_top10_test": 1}
    assert c["lists"][D.S1]["different_full"] == 0
    assert c["causes"][D.S2] == {"vector de consulta distinto": 1}
    # q2's relevant book drops from rank 1 to rank 2 in the replication's S2
    assert c["metric_changes_test"]["T"][D.S2]["largest_change"] == -0.5
    assert c["weight"]["a"] == 0.25 and c["weight"]["b"] == 0.75
    assert c["rerank"]["different"] == 1 and c["rerank"]["different_book_sets"] == 0
    assert c["embeddings"]["different_same_provider"] == 1


def test_same_vector_with_a_different_list_points_at_the_approximate_index():
    c = compare(_folder("a", ["b", "c"], [0.0, 1.0], 0.8, 0.25), _folder("b", ["c", "b"], [0.0, 1.0], 0.8, 0.25))
    assert c["causes"][D.S2] == {"mismo vector, lista distinta (índice aproximado)": 1}


def test_run_time_spans_the_ledger_and_counts_invocations():
    ledger = [{"date": "2026-11-14T10:00:00+00:00", "operation": "key_usage", "when": "start"},
              {"date": "2026-11-14T10:30:00+00:00", "operation": "embedding"},
              {"date": "2026-11-14T11:00:00+00:00", "operation": "key_usage", "when": "start"},
              {"date": "2026-11-14T11:10:00+00:00", "operation": "key_usage", "when": "end"}]
    t = run_time(ledger)
    assert t["invocations"] == 2 and t["span_seconds"] == 4200


def test_compare_command_writes_both_files(tmp_path):
    for name in ("orig", "rep"):
        folder = tmp_path / name
        (folder / "runs").mkdir(parents=True)
        (folder / D.SAMPLE_FILE).write_text(json.dumps({"queries": QUERIES}), encoding="utf-8")
        (folder / "runs" / f"{D.S1}.jsonl").write_text(
            "\n".join(json.dumps(_line(q["query_id"], ["a", "b"])) for q in QUERIES) + "\n", encoding="utf-8")
    assert main(["--data-dir", str(tmp_path / "orig"), "compare", "--with", str(tmp_path / "rep")]) == 0
    written = json.loads((tmp_path / "rep" / "replication.json").read_text(encoding="utf-8"))
    assert written["lists"][D.S1]["identical_full"] == 2
    assert (tmp_path / "rep" / "REPLICATION.md").read_text(encoding="utf-8").startswith("# Réplica")
