"""Pure parts of the profile parity measurement (no database, no model)."""

from uuid import uuid4

import pytest

from app.evaluation.profile_parity import (
    CONTRASTS,
    MATCHES_ONLY,
    CachedQueryEmbeddings,
    analyse,
    build_markdown,
    capacity,
    matches_only_run,
    overlap,
    overlap_report,
    recall_against,
    score_differences,
)
from app.evaluation.retrieval_ablation import UnjudgedResultsError
from app.evaluation.types import TestQuery


def test_overlap_and_recall():
    assert overlap([1, 2, 3], [3, 2, 1]) == 1.0
    assert overlap([1, 2], [1, 2, 3, 4]) == 0.5  # divided by the longer list
    assert overlap([], []) == 1.0
    assert recall_against([1, 9], [1, 2]) == 0.5
    assert recall_against([1], []) == 1.0
    assert overlap(list(range(20)), list(range(10, 30))) == 0.0  # only the top 10 count


def test_overlap_report_counts_identical_lists_and_sets():
    runs = {"a": {"q1": [1, 2], "q2": [3, 4]}, "b": {"q1": [1, 2], "q2": [4, 3]}}
    [entry] = overlap_report(runs, [("a", "b"), ("a", "missing")]).values()
    assert entry["identical_lists"] == 1 and entry["identical_sets"] == 2 and entry["mean_overlap"] == 1.0


def test_matches_only_drops_zero_scores_and_score_differences():
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    runs = {"bm25_normalized": {"q1": [a, b, c]}}
    scores = {"bm25_normalized": {"q1": {a: 2.0, b: 0.5, c: 0.0}}}
    assert matches_only_run(runs, scores) == {"q1": [a, b]}
    assert matches_only_run(runs, {}) is None
    diffs = score_differences({"x": {"q1": {a: 0.5, b: 0.25}}, "y": {"q1": {a: 0.5001, c: 0.1}}}, "x", "y")
    assert diffs["pairs"] == 1 and diffs["max_abs_difference"] == pytest.approx(0.0001)


def _queries():
    return [TestQuery(query_id=f"q{i}", text=f"query {i}", language="en" if i % 2 else "es") for i in range(6)]


def test_analyse_runs_the_three_contrasts_with_holm_and_reports_short_lists():
    books = [uuid4() for _ in range(4)]
    queries = _queries()
    judgments = {q.query_id: {books[0]: 3, books[1]: 1, books[2]: 0, books[3]: 0} for q in queries}
    full = {q.query_id: [books[0], books[1], books[2]] for q in queries}
    short = {q.query_id: [books[1]] for q in queries}
    runs = {name: full for name in ("bm25_normalized", "vec_e5", "hybrid_norm_e5", "pg_vec_exact", "pg_hybrid")}
    runs["pg_bm25"] = short

    report = analyse(runs, queries, judgments, n_resamples=200)

    assert list(report["contrasts"]) == [name for name, _, _ in CONTRASTS]
    assert all("p_value_holm" in c for c in report["contrasts"].values())
    assert report["contrasts"]["vector_exact"]["mean_difference"] == 0.0
    assert report["contrasts"]["lexical"]["mean_difference"] < 0
    lengths = report["systems"]["pg_bm25"]["list_lengths"]
    assert lengths["mean"] == 1.0 and lengths["queries_with_fewer_than_10"] == 6
    assert lengths["judged_share_of_returned"] == 1.0
    assert report["systems"]["pg_bm25"]["metrics"]["judged_at_10"]["mean"] == pytest.approx(0.1)


def test_analyse_refuses_unjudged_results():
    queries = _queries()
    book, stranger = uuid4(), uuid4()
    judgments = {q.query_id: {book: 1} for q in queries}
    with pytest.raises(UnjudgedResultsError):
        analyse({"pg_bm25": {q.query_id: [stranger] for q in queries}}, queries, judgments, n_resamples=50)


def test_cached_query_embeddings_encodes_each_text_once():
    class Store:
        calls = 0

        def generate_query_embedding(self, text):
            Store.calls += 1
            return [float(len(text))]

        def is_ready(self):
            return True

    cached = CachedQueryEmbeddings(Store())
    assert cached.generate_query_embedding("abc") == [3.0]
    assert cached.generate_query_embedding("abc") == [3.0]
    assert Store.calls == 1 and cached.is_ready() is True


def test_capacity_extrapolates_from_bytes_per_work():
    scale = {
        "catalog_info": {"works": 1000},
        "schema_total_bytes": 10_000_000,
        "relations": [{"kind": "index", "relation": "works_embedding_ann", "bytes": 4_000_000}],
    }
    cap = capacity(scale, database_bytes=30_000_000, limit=110_000_000)
    assert cap["bytes_per_work"] == 10_000 and cap["bytes_per_work_without_ann_index"] == 6_000
    assert cap["available_bytes"] == 90_000_000  # the other 20 MB stay where they are
    assert cap["works_that_fit"] == 9_000 and cap["works_that_fit_without_ann_index"] == 15_000


def test_markdown_renders_a_minimal_report():
    books = [uuid4() for _ in range(3)]
    queries = _queries()
    judgments = {q.query_id: {books[0]: 2, books[1]: 0, books[2]: 1} for q in queries}
    runs = {name: {q.query_id: books for q in queries}
            for name in ("bm25_normalized", "vec_e5", "hybrid_norm_e5", "pg_bm25", "pg_vec_exact", "pg_hybrid")}
    report = {
        "date": "2026-11-11T00:00:00+00:00", "git_commit": "abc1234", "schema": "eval177", "hnsw_schema": None,
        "database": {"postgres": "18.6", "database_bytes": 1, "extensions": {}},
        "judgments": {"cloud_file": "c.json", "base_file": "b.json", "base_sha256": "0" * 64, "holes_at_start": 0},
        "latency": {"pg_bm25": {"median_ms": 50.0, "p95_ms": 60.0, "warmup_ms": 100.0}},
        "analysis": analyse(runs, queries, judgments, n_resamples=50),
    }
    text = build_markdown(report)
    assert "Pre-declared contrasts" in text and "`pg_hybrid`" in text and MATCHES_ONLY not in text
