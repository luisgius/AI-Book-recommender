"""
The V13 BM25 padding measurement and the guards that keep published runs reproducible.

No model, judge or database: the design as data, the pure analysis, the
ModelStack's lexical side, and a source check that every evaluation path
states which BM25 behaviour it uses.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from app.domain.entities import Book
from app.evaluation import bm25_padding
from app.evaluation.bm25_padding import CONTRASTS, SYSTEMS, analyse, build_markdown, list_lengths
from app.evaluation.retrieval_ablation import SYSTEMS as ABLATION_SYSTEMS, ModelStack, UnjudgedResultsError
from app.evaluation.types import TestQuery

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- the design -----------------------------------------------------------------------


def test_published_arms_keep_the_legacy_padding_explicitly():
    lexical = [spec for spec in ABLATION_SYSTEMS if spec.lexical]
    assert lexical and all(spec.zero_score_padding for spec in lexical)


def test_v13_arms_are_the_published_ones_with_the_fixed_adapter():
    by_name = {spec.name: spec for spec in SYSTEMS}
    published = {spec.name: spec for spec in ABLATION_SYSTEMS}
    for name in ("bm25_normalized", "vec_e5", "hybrid_norm_e5", "hybrid_norm_e5_bge"):
        assert by_name[name] is published[name]
    for fixed, legacy in (("bm25_normalized_nopad", "bm25_normalized"), ("hybrid_norm_e5_nopad", "hybrid_norm_e5"),
                          ("hybrid_norm_e5_bge_nopad", "hybrid_norm_e5_bge")):
        a, b = by_name[fixed], by_name[legacy]
        assert (a.lexical, a.embedding, a.reranker) == (b.lexical, b.embedding, b.reranker)
        assert a.zero_score_padding is False and b.zero_score_padding is True


def test_three_contrasts_declared_in_advance():
    assert [(n, s, r) for n, s, r, _ in CONTRASTS] == [
        ("F1", "hybrid_norm_e5_nopad", "hybrid_norm_e5"),
        ("F2", "hybrid_norm_e5_nopad", "vec_e5"),
        ("F3", "hybrid_norm_e5_bge_nopad", "hybrid_norm_e5_bge"),
    ]


def test_every_bm25_built_by_an_evaluation_path_states_its_padding():
    """A new evaluation path must choose: reproduce a published run (True) or measure the fix (False)."""
    offenders = []
    for path in (REPO_ROOT / "app" / "evaluation").glob("*.py"):
        for match in re.finditer(r"BM25SearchRepository\(([^)]*)\)", path.read_text(encoding="utf-8")):
            if "zero_score_padding" not in match.group(1):
                offenders.append(f"{path.name}: {match.group(0)}")
    assert offenders == []


# --- the harness's lexical side -------------------------------------------------------------


def _book(title: str) -> Book:
    return Book(id=uuid4(), title=title, authors=["A"], source="test", source_id=title,
                created_at=datetime(2025, 1, 1, tzinfo=UTC), updated_at=datetime(2025, 1, 1, tzinfo=UTC))


def test_model_stack_builds_bm25_with_the_requested_padding_once_per_mode():
    books = [_book("python programming"), _book("cooking at home"), _book("garden plants")]
    stack = ModelStack(books, indexes_dir="unused", ablation_dir="unused", device="cpu", log=lambda m: None)

    legacy = stack._lexical("normalized", zero_score_padding=True)
    fixed = stack._lexical("normalized", zero_score_padding=False)

    assert legacy.zero_score_padding is True and fixed.zero_score_padding is False
    assert stack._lexical("normalized", zero_score_padding=True) is legacy
    assert len(legacy.search("python", max_results=3)) == 3
    assert [r.book.title for r in fixed.search("python", max_results=3)] == ["python programming"]
    assert "bm25:normalized" in stack.costs and "bm25:normalized:matching_only" in stack.costs


def test_padding_facts_on_a_tiny_catalog():
    books = [_book("python programming"), _book("cooking at home"), _book("garden plants")]

    facts = bm25_padding.padding_facts(books, {"q1": "python", "q2": "nothing here"})

    assert facts["10"]["mean_padding_books"] == pytest.approx((2 + 3) / 2)
    assert facts["10"]["queries_with_padding"] == 2
    assert facts["10"]["fixed_list_is_legacy_prefix"] is True


# --- the analysis -------------------------------------------------------------------------


def _queries():
    return [TestQuery(query_id=f"q{i}", text=f"query {i}", language="en" if i % 2 else "es") for i in range(6)]


def test_analyse_runs_the_family_with_holm_and_reports_short_lists():
    books = [uuid4() for _ in range(4)]
    queries = _queries()
    judgments = {q.query_id: {books[0]: 3, books[1]: 1, books[2]: 0, books[3]: 0} for q in queries}
    full = {q.query_id: [books[2], books[1], books[0]] for q in queries}
    better = {q.query_id: [books[0], books[1], books[2]] for q in queries}
    runs = {spec.name: full for spec in SYSTEMS}
    runs["hybrid_norm_e5_nopad"] = better
    runs["bm25_normalized_nopad"] = {q.query_id: [books[0]] for q in queries}

    report = analyse(runs, queries, judgments, n_resamples=200)

    assert list(report["contrasts"]) == ["F1", "F2", "F3"]
    assert report["contrasts"]["F1"]["mean_difference"] > 0
    assert report["contrasts"]["F3"]["mean_difference"] == 0.0
    assert all("p_value_holm" in c for c in report["contrasts"].values())
    lengths = report["systems"]["bm25_normalized_nopad"]["list_lengths"]
    assert lengths == {"mean": 1.0, "queries_with_fewer_than_10": 6, "queries_with_no_result": 0,
                       "judged_share_of_returned": 1.0}
    assert report["systems"]["bm25_normalized_nopad"]["metrics"]["judged_at_10"]["mean"] == pytest.approx(0.1)
    assert report["systems"]["hybrid_norm_e5"]["not_relevant_share_of_returned"] == pytest.approx(1 / 3)


def test_analyse_refuses_an_unjudged_returned_result():
    queries = _queries()
    book, stranger = uuid4(), uuid4()
    judgments = {q.query_id: {book: 1} for q in queries}
    with pytest.raises(UnjudgedResultsError):
        analyse({"vec_e5": {q.query_id: [stranger] for q in queries}}, queries, judgments, n_resamples=50)


def test_list_lengths_counts_empty_lists():
    lengths = list_lengths({"a": {"q1": [], "q2": [1, 2]}}, {"q2": {1: 0, 2: 1}}, ["q1", "q2"])
    assert lengths["a"]["queries_with_no_result"] == 1 and lengths["a"]["mean"] == 1.0


def test_markdown_renders_from_the_analysis():
    books = [uuid4() for _ in range(3)]
    queries = _queries()
    judgments = {q.query_id: {books[0]: 2, books[1]: 0, books[2]: 1} for q in queries}
    runs = {spec.name: {q.query_id: books for q in queries} for spec in SYSTEMS}
    report = {
        "date": "2026-11-11T00:00:00+00:00", "git_commit": "abc1234", "device": "cpu",
        "judgments": {"v13_file": "v13.json", "base_file": "cloud.json", "base_sha256": "0" * 64, "holes_at_start": 0},
        "padding": {"10": {"mean_padding_books": 1.0, "max_padding_books": 3, "queries_with_padding": 2,
                           "fixed_list_is_legacy_prefix": True}},
        "analysis": analyse(runs, queries, judgments, n_resamples=50),
    }
    text = build_markdown(report)
    assert "## Pre-declared contrasts" in text and "`hybrid_norm_e5_bge_nopad`" in text and "| F3 |" in text


# --- the published report -------------------------------------------------------------------


def test_published_report_matches_its_saved_rankings():
    """The saved rankings carry the contrasts the report prints (no model, cheap check)."""
    path = REPO_ROOT / bm25_padding.DEFAULT_OUTPUT_JSON
    if not path.is_file():
        pytest.skip("results_bm25_padding.json not generated yet")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert set(saved["runs"]) == {spec.name for spec in SYSTEMS}
    assert list(saved["analysis"]["contrasts"]) == ["F1", "F2", "F3"]
    for system, check in saved["analysis"]["reproduction"].items():
        assert check["identical_lists"] == check["num_queries"] == 44, system
    for name, data in saved["analysis"]["systems"].items():
        assert data["list_lengths"]["judged_share_of_returned"] == 1.0, name
