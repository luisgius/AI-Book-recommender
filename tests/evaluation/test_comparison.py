"""
Tests for the shared system comparison (app/evaluation/comparison.py).

The synthetic collection below is small enough to score by hand:

    judgments  q1 (en): a=3, b=1, c=0        q2 (es): d=2, e=0      q3 (en): none
    base       q1: [a, x, b]                 q2: [e, d]             q3: [a]
    alt        q1: [x, b, a]                 q2: [d]                q3: []

x is never judged. With gain 2^g - 1 and discount log2(rank + 1):
    IDCG(q1) = 7 + 1/log2(3)                IDCG(q2) = 3
"""

import json
import math
from uuid import uuid4

import pytest

from app.domain.entities import Book, SearchResult
from app.evaluation import llm_path_evaluation
from app.evaluation.comparison import compare_systems, describe_judgments, render_markdown, score_query
from app.evaluation.evaluation_service import (
    EvaluationService,
    condensed_ranking,
    judged_at_k,
    judged_fraction_at_k,
    ndcg_at_k,
)
from app.evaluation.statistics import holm_correction
from app.evaluation.types import RelevanceJudgment, TestQuery

L3 = math.log2(3)
IDCG_Q1 = 7 + 1 / L3

JUDGMENTS = {"q1": {"a": 3, "b": 1, "c": 0}, "q2": {"d": 2, "e": 0}}
QUERIES = [
    TestQuery("q1", "first", language="en"),
    TestQuery("q2", "segunda", language="es"),
    TestQuery("q3", "third", language="en"),
]
BASE = {"q1": ["a", "x", "b"], "q2": ["e", "d"], "q3": ["a"]}
ALT = {"q1": ["x", "b", "a"], "q2": ["d"], "q3": []}

EXPECTED = {
    "base": {
        "q1": {"ndcg_at_10": 7.5 / IDCG_Q1, "recall_at_10": 1.0, "precision_at_5": 0.4, "mrr": 1.0,
               "judged_at_10": 0.2, "condensed_ndcg_at_10": 1.0},
        "q2": {"ndcg_at_10": 1 / L3, "recall_at_10": 1.0, "precision_at_5": 0.2, "mrr": 0.5,
               "judged_at_10": 0.2, "condensed_ndcg_at_10": 1 / L3},
    },
    "alt": {
        "q1": {"ndcg_at_10": (1 / L3 + 3.5) / IDCG_Q1, "recall_at_10": 1.0, "precision_at_5": 0.4, "mrr": 0.5,
               "judged_at_10": 0.2, "condensed_ndcg_at_10": (1 + 7 / L3) / IDCG_Q1},
        "q2": {"ndcg_at_10": 1.0, "recall_at_10": 1.0, "precision_at_5": 0.2, "mrr": 1.0,
               "judged_at_10": 0.1, "condensed_ndcg_at_10": 1.0},
    },
}


@pytest.fixture(scope="module")
def report():
    return compare_systems({"base": BASE, "alt": ALT}, JUDGMENTS, QUERIES, baseline="base", n_resamples=2000)


def test_per_query_scores_match_the_hand_computed_values(report):
    for system, by_query in EXPECTED.items():
        for query_id, metrics in by_query.items():
            assert report["systems"][system]["per_query"][query_id] == pytest.approx(metrics)


def test_aggregate_means_and_intervals(report):
    for system, by_query in EXPECTED.items():
        for name, stat in report["systems"][system]["metrics"].items():
            expected_mean = (by_query["q1"][name] + by_query["q2"][name]) / 2
            assert stat["mean"] == pytest.approx(expected_mean)
            assert stat["ci_low"] <= stat["mean"] <= stat["ci_high"]


def test_queries_without_judgments_are_excluded(report):
    assert report["num_queries"] == 2
    assert report["query_ids"] == ["q1", "q2"]
    assert report["excluded_query_ids"] == ["q3"]
    assert report["queries_without_relevant"] == []


def test_language_breakdown(report):
    assert report["languages"] == {"en": 1, "es": 1}
    base = report["systems"]["base"]["by_language"]
    assert base["en"]["num_queries"] == 1 and base["es"]["num_queries"] == 1
    # one query per language: the interval collapses onto that query's score
    assert base["en"]["ndcg_at_10"]["mean"] == pytest.approx(7.5 / IDCG_Q1)
    assert base["en"]["ndcg_at_10"]["ci_low"] == pytest.approx(7.5 / IDCG_Q1)
    assert base["es"]["ndcg_at_10"]["mean"] == pytest.approx(1 / L3)
    assert report["systems"]["alt"]["by_language"]["es"]["ndcg_at_10"]["ci_high"] == pytest.approx(1.0)


def test_unlabelled_queries_are_grouped_as_unknown():
    queries = [TestQuery("q1", "first"), TestQuery("q2", "segunda", language="es")]
    out = compare_systems({"base": BASE}, JUDGMENTS, queries, baseline="base", n_resamples=200)
    assert out["languages"] == {"es": 1, "unknown": 1}


def test_difference_against_the_baseline(report):
    expected = ((EXPECTED["alt"]["q1"]["ndcg_at_10"] - EXPECTED["base"]["q1"]["ndcg_at_10"])
                + (EXPECTED["alt"]["q2"]["ndcg_at_10"] - EXPECTED["base"]["q2"]["ndcg_at_10"])) / 2
    diff = report["vs_baseline"]["alt"]
    assert diff["metric"] == "ndcg_at_10"
    assert diff["mean_difference"] == pytest.approx(expected)
    assert diff["ci_low"] <= diff["mean_difference"] <= diff["ci_high"]
    # with two queries every sign flip is at least as extreme as the observed difference
    assert diff["p_value"] == pytest.approx(1.0)
    assert diff["p_value_holm"] == pytest.approx(1.0)
    assert "base" not in report["vs_baseline"]


def test_holm_is_applied_across_all_compared_systems():
    runs = {"base": BASE, "alt": ALT, "empty": {"q1": [], "q2": [], "q3": []}}
    out = compare_systems(runs, JUDGMENTS, QUERIES, baseline="base", n_resamples=2000)
    raw = {name: c["p_value"] for name, c in out["vs_baseline"].items()}
    assert set(raw) == {"alt", "empty"}
    for name, adjusted in holm_correction(raw).items():
        assert out["vs_baseline"][name]["p_value_holm"] == pytest.approx(adjusted)


def test_report_is_json_serializable(report):
    assert json.loads(json.dumps(report))["config"]["baseline"] == "base"


def test_queries_whose_judgments_are_all_zero_stay_and_score_zero():
    judgments = {**JUDGMENTS, "q3": {"c": 0}}
    out = compare_systems({"base": BASE}, judgments, QUERIES, baseline="base", n_resamples=200)
    assert out["num_queries"] == 3
    assert out["queries_without_relevant"] == ["q3"]
    scores = out["systems"]["base"]["per_query"]["q3"]
    assert scores["ndcg_at_10"] == 0.0 and scores["condensed_ndcg_at_10"] == 0.0
    assert scores["judged_at_10"] == 0.0  # [a] is not judged for q3


def test_rejects_inconsistent_input():
    with pytest.raises(ValueError, match="Baseline"):
        compare_systems({"alt": ALT}, JUDGMENTS, QUERIES, baseline="base")
    with pytest.raises(ValueError, match="no ranking"):
        compare_systems({"base": BASE, "alt": {"q1": []}}, JUDGMENTS, QUERIES, baseline="base")
    with pytest.raises(ValueError, match="No query"):
        compare_systems({"base": BASE}, {}, QUERIES, baseline="base")


def test_markdown_table_uses_three_decimals(report):
    md = render_markdown(report)
    base_mean = report["systems"]["base"]["metrics"]["ndcg_at_10"]["mean"]
    assert "| base (baseline) | " + f"{base_mean:.3f} [" in md
    diff = report["vs_baseline"]["alt"]["mean_difference"]
    assert f"{diff:+.3f} [" in md
    assert "nDCG@10 en (n=1) [95% CI]" in md and "nDCG@10 es (n=1) [95% CI]" in md
    assert "| alt | " in md


# ---------------------------------------------------------------------------
# Condensed lists and Judged@k
# ---------------------------------------------------------------------------


def test_condensed_ranking_drops_unjudged_books_and_keeps_the_order():
    assert condensed_ranking(["x", "b", "y", "a"], JUDGMENTS["q1"]) == ["b", "a"]


def test_condensed_ndcg_is_not_penalised_by_unjudged_books():
    ranked = ["x", "b", "a"]
    assert ndcg_at_k(ranked, JUDGMENTS["q1"]) == pytest.approx((1 / L3 + 3.5) / IDCG_Q1)
    assert score_query(ranked, JUDGMENTS["q1"])["condensed_ndcg_at_10"] == pytest.approx((1 + 7 / L3) / IDCG_Q1)


def test_judged_fraction_uses_k_as_denominator():
    assert judged_fraction_at_k(["a", "x", "b"], JUDGMENTS["q1"], k=10) == pytest.approx(0.2)
    assert judged_fraction_at_k(["a", "x", "b"], JUDGMENTS["q1"], k=2) == pytest.approx(0.5)
    assert judged_fraction_at_k([], JUDGMENTS["q1"], k=10) == 0.0
    with pytest.raises(ValueError):
        judged_fraction_at_k(["a"], JUDGMENTS["q1"], k=0)


def _results(books):
    return [SearchResult(book=b, final_score=1.0, rank=i + 1, source="hybrid") for i, b in enumerate(books)]


def test_judged_at_k_averages_over_judged_queries_only():
    books = [Book(id=uuid4(), title=f"B{i}", authors=["A"]) for i in range(4)]
    judgments = {
        "q1": RelevanceJudgment("q1", {books[0].id: 3, books[1].id: 0}),
        "q2": RelevanceJudgment("q2", {books[2].id: 1}),
        "q3": RelevanceJudgment("q3", {}),
    }
    results = {"q1": _results(books[:2]), "q2": _results([books[3], books[2]]), "q3": _results(books)}
    # q1: 2 of top 2 judged -> 1.0; q2: 1 of top 2 -> 0.5; q3 has no judgments and is skipped
    assert judged_at_k(results, judgments, k=2) == pytest.approx(0.75)


def test_llm_path_evaluation_uses_the_shared_judged_at_k():
    assert llm_path_evaluation.judged_at_k is judged_at_k


def test_service_methods_and_id_functions_agree():
    books = [Book(id=uuid4(), title=f"B{i}", authors=["A"]) for i in range(5)]
    judgment = RelevanceJudgment("q1", {books[0].id: 1, books[3].id: 3, books[4].id: 2})
    ranked = [books[i] for i in (2, 3, 0, 1)]
    service = EvaluationService()
    expected = score_query([b.id for b in ranked], judgment.judgments)
    assert service.compute_ndcg(_results(ranked), judgment, k=10) == pytest.approx(expected["ndcg_at_10"])
    assert service.compute_recall_at_k(_results(ranked), judgment, k=10) == pytest.approx(expected["recall_at_10"])
    assert service.compute_precision_at_k(_results(ranked), judgment, k=5) == pytest.approx(expected["precision_at_5"])
    assert service.compute_mrr(_results(ranked), judgment) == pytest.approx(expected["mrr"])


# ---------------------------------------------------------------------------
# Qrels statistics
# ---------------------------------------------------------------------------


def test_describe_judgments_counts_and_tolerates_extra_fields():
    raw = {
        "q1": [
            {"source": "g", "source_id": "a", "relevance": 3, "judge": "llm", "model": "m", "date": "2026-11-11"},
            {"source": "g", "source_id": "b", "relevance": 0, "prompt_version": "v1"},
            {"source": "g", "source_id": "c", "relevance": 1},
        ],
        "q2": [{"source": "g", "source_id": "a", "relevance": 0}],
    }
    queries = [TestQuery("q1", "one", language="en"), TestQuery("q2", "dos", language="es"),
               TestQuery("q3", "tres", language="es")]
    stats = describe_judgments(raw, queries, num_books=5)
    assert stats["num_judgments"] == 4
    assert stats["num_distinct_query_book_pairs"] == 4
    assert stats["judged_share_of_matrix"] == pytest.approx(4 / 15)
    assert stats["judgments_per_query"] == {"min": 0, "median": 1, "max": 3}
    assert stats["grade_distribution"] == {"0": 2, "1": 1, "2": 0, "3": 1}
    assert stats["num_relevant_judgments"] == 2
    assert stats["queries_without_judgments"] == ["q3"]
    assert stats["queries_without_relevant"] == ["q2", "q3"]
    assert stats["by_language"]["es"] == {
        "num_queries": 2, "num_judgments": 1, "num_relevant_judgments": 0, "queries_without_relevant": ["q2", "q3"],
    }
    assert stats["extra_judgment_fields"] == ["date", "judge", "model", "prompt_version"]
