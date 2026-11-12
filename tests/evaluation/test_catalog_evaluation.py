"""Tests of the catalog evaluation (V14): pure logic only, no network, no database."""

import csv
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.domain.entities import Book, SearchResult
from app.domain.services import SearchService
from app.evaluation.catalog_evaluation import analysis as A
from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.annotation import (
    KEY_FILE,
    SHEET_COLUMNS,
    SHEET_FILE,
    annotation_agreement,
    build_sheet,
    read_sheet,
)
from app.evaluation.catalog_evaluation.e2_index import ExactIndex
from app.evaluation.catalog_evaluation.generation import find_target
from app.evaluation.catalog_evaluation.judging import build_pool, judge_cache, judge_pool, needs_of
from app.evaluation.catalog_evaluation.ledger import SpendCapReached, SpendLedger, summarize_ledger
from app.evaluation.catalog_evaluation.queries import (
    EvalQuery,
    b_queries,
    c_queries,
    check_queries,
    d_queries,
    language_check,
    parse_json_answer,
    stratum_a,
    title_overlap,
)
from app.evaluation.catalog_evaluation.runs import apply_head_order, decode_vector, encode_vector, rrf_fuse


def q(qid, stratum="B", need=None, language="en", qtype="thematic", text=None):
    return EvalQuery(qid, need or qid, text or f"text {qid}", language, qtype, stratum, f"need {need or qid}")


# ------------------------------------------------------------------ queries


def test_stratum_a_keeps_text_and_uses_it_as_need():
    out = stratum_a([{"query_id": "q01", "text": "machine learning python", "language": "en"}])
    assert out[0].need == out[0].text == "machine learning python"
    assert (out[0].stratum, out[0].need_id, out[0].type) == ("A", "q01", D.A_TYPE)


def test_c_queries_share_need_across_languages():
    answer = {"needs": [{"need": "n", "type": "thematic", "es": "a", "en": "b", "fr": "c", "de": "d"}]}
    out = c_queries(answer)
    assert [x.query_id for x in out] == ["c01-es", "c01-en", "c01-fr", "c01-de"]
    assert {x.need_id for x in out} == {"c01"}


def test_parse_json_answer_accepts_fenced_block():
    assert parse_json_answer('```json\n{"queries": []}\n```') == {"queries": []}


@pytest.mark.parametrize("text,language,verdict", [
    ("una novela sobre la guerra y los soldados", "es", "ok"),
    ("a novel about the sea and the whales", "es", "mismatch"),
    ("советская научная фантастика", "ru", "ok"),
    ("硬科幻小说 太空殖民", "zh", "ok"),
    ("孤島に取り残された少年たちの物語", "ja", "ok"),
    ("孤島に取り残された", "zh", "mismatch"),
    ("post-apocalyptic science fiction", "en", "undecided"),
])
def test_language_check(text, language, verdict):
    assert language_check(text, language)[0] == verdict


def test_title_overlap_latin_and_cjk():
    assert title_overlap("the novel about Rochester at Thornfield", "Jane Eyre") == []
    assert title_overlap("Kafka Verwandlung Gregor", "Die Verwandlung") == ["verwandlung"]
    assert title_overlap("曹雪芹的红楼故事", "红楼梦") == ["红楼"]


def test_check_queries_flags_counts_duplicates_and_title_words():
    queries = [q("b-en-1", text="same text"), q("b-en-2", text="Same  TEXT"),
               EvalQuery("b-en-3", "b-en-3", "Kafka Verwandlung", "de", "known_item", "B", "n", target_title="Die Verwandlung")]
    checks = check_queries(queries)
    kinds = {p["check"] for p in checks["problems"]}
    assert {"count", "duplicate", "title words"} <= kinds
    assert checks["duplicates"] == [["b-en-1", "b-en-2"]]


def test_find_target_exact_title_without_author_match():
    works = [("1", "Преступление и наказание", ["Fyodor Dostoyevsky"]), ("2", "Jane Eyre: An Autobiography", ["Charlotte Brontë"])]
    assert find_target("Преступление и наказание", "Фёдор Достоевский", works) == ["1"]
    assert find_target("Jane Eyre", "Charlotte Bronte", works) == ["2"]
    assert find_target("Jane Eyre", "Someone Else", works) == []


# ------------------------------------------------------------------ runs


def _book(title):
    return Book(id=uuid4(), title=title, authors=["a"], source="t", source_id=title)


def test_rrf_fuse_matches_search_service():
    books = {name: _book(name) for name in "abcdef"}
    lexical = ["a", "b", "c", "d"]
    vector = ["c", "e", "a", "f"]
    as_results = lambda ids, source: [SearchResult(book=books[i], final_score=1.0, rank=r, source=source)  # noqa: E731
                                      for r, i in enumerate(ids, start=1)]
    domain = SearchService._fuse_results_rrf(None, as_results(lexical, "lexical"), as_results(vector, "vector"), k=60)
    expected = [r.book.title for r in domain]
    assert [i for i, _ in rrf_fuse(lexical, vector)] == expected


def test_apply_head_order_keeps_tail():
    assert apply_head_order(["a", "b", "c", "d"], [1, 0], 2) == ["b", "a", "c", "d"]
    with pytest.raises(ValueError):
        apply_head_order(["a", "b", "c"], [0, 0], 2)


def test_vector_roundtrip_is_exact():
    v = np.random.default_rng(0).normal(size=8).astype(np.float32)
    assert np.array_equal(decode_vector(encode_vector(v)), v)


def test_exact_index_cosine_order():
    index = ExactIndex(["x", "y", "z"], np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    assert [i for i, _ in index.search([1, 0.1], 2)] == ["x", "z"]


# ------------------------------------------------------------------ pooling and judging


def test_pool_is_union_of_top_k_per_need():
    queries = [q("c01-es", "C", "c01"), q("c01-en", "C", "c01"), q("b1")]
    runs = {"S1": {"c01-es": {"ids": ["a", "b", "z"]}, "c01-en": {"ids": ["b", "c"]}, "b1": {"ids": ["a"]}},
            "S2": {"c01-es": {"ids": ["d"]}}}
    pool = build_pool(queries, runs, systems=("S1", "S2"), k=2)
    assert pool == {"c01": ["a", "b", "d", "c"], "b1": ["a"]}
    assert needs_of(queries)["c01"].query_id == "c01-es"


class FakeJudge:
    model_name = "fake/judge"
    prompt_version = "v1.0"

    def __init__(self):
        self.calls = 0
        self.bad_calls = 0

    def judge(self, text, book):
        self.calls += 1
        if book.title == "bad":
            self.bad_calls += 1
            if self.bad_calls == 1:
                raise RuntimeError("flaky")
        return (3 if "x" in book.title else 0), "because"


def test_judge_pool_caches_and_retries(tmp_path):
    ledger = SpendLedger(tmp_path / "l.jsonl", 5.0)
    cache = judge_cache(tmp_path, "fake/judge")
    judge = FakeJudge()
    tasks = [("n1", "need", _book("x1")), ("n1", "need", _book("y")), ("n2", "need", _book("bad"))]
    stats = judge_pool("fake/judge", tasks, cache, ledger, log=lambda m: None, workers=1, judge_factory=lambda: judge)
    assert stats["unjudged"] == 0 and stats["failed_calls_in_run"] >= 1
    calls = judge.calls
    again = judge_pool("fake/judge", tasks, cache, ledger, log=lambda m: None, workers=1, judge_factory=lambda: judge)
    assert judge.calls == calls and again["judged_in_run"] == 0


def test_ledger_cap_and_summary(tmp_path):
    ledger = SpendLedger(tmp_path / "l.jsonl", 0.01)
    ledger.add("s1", "embedding", "m", 0.004)
    ledger.add("s1", "key_usage", "-", None, key_usage_usd=1.0)
    ledger.check()
    ledger.add("s2", "judge", "j", 0.007, error=None)
    with pytest.raises(SpendCapReached):
        SpendLedger(tmp_path / "l.jsonl", 0.01).check()
    summary = summarize_ledger(ledger.entries())
    assert summary["by_stage"]["s2"]["cost_usd"] == pytest.approx(0.007)
    assert summary["total_usd"] == pytest.approx(0.011)


# ------------------------------------------------------------------ metrics and rules


def test_query_metrics_and_contrast_family():
    grades = {"a": 3, "b": 1}
    m = A.query_metrics(["a", "x", "b"], grades)
    assert m["ndcg_at_10"] > 0.9 and m["p_at_10"] == pytest.approx(0.2) and m["judged_at_10"] == pytest.approx(0.2)
    scores = {s: {f"q{i}": {"ndcg_at_10": (0.5 if s in (D.S3, D.S5) else 0.2) + i * 0.001} for i in range(20)}
              for s in D.SYSTEM_NAMES}
    out = A.contrasts(scores, [f"q{i}" for i in range(20)])
    assert out["H1"]["difference"] == pytest.approx(0.3)
    assert out["H1"]["p_holm"] >= out["H1"]["p"]
    assert out["H5-rep"]["in_family"] is False and out["H5-rep"]["p_holm"] is None


@pytest.mark.parametrize("primary,second,expected", [
    ({"difference": 0.1, "p": 0.001, "p_holm": 0.01}, {"difference": 0.05, "p": 0.001, "p_holm": 0.02}, "sustained"),
    ({"difference": 0.1, "p": 0.001, "p_holm": 0.01}, {"difference": 0.05, "p": 0.2, "p_holm": 0.4}, "not conclusive"),
    ({"difference": 0.1, "p": 0.001, "p_holm": 0.01}, {"difference": -0.05, "p": 0.001, "p_holm": 0.01}, "not conclusive"),
    ({"difference": 0.1, "p": 0.3, "p_holm": 0.6}, {"difference": 0.05, "p": 0.2, "p_holm": 0.4}, "not significant"),
])
def test_conclusion_rule(primary, second, expected):
    assert A.conclusion(primary, second) == expected


def _op(median, p95, failed, changed):
    return {"median_ms": median, "p95_ms": p95, "failed_calls": failed, "queries_top10_changed": changed}


def test_decision_rule():
    h6_ok = {"difference": -0.01, "p": 0.5, "p_holm": 0.9}
    h6_qwen_better = {"difference": -0.08, "p": 0.001, "p_holm": 0.004}
    qwen = _op(1500, 4000, 5, 20)
    assert A.decision_rule(h6_ok, h6_ok, _op(500, 900, 0, 0), qwen)["outcome"].startswith("propose")
    assert A.decision_rule(h6_ok, h6_qwen_better, _op(500, 900, 0, 0), qwen)["outcome"].startswith("keep")
    # better on two criteria, more than 10 % worse on the third: keep
    assert A.decision_rule(h6_ok, h6_ok, _op(500, 900, 0, 30), qwen)["outcome"].startswith("keep")
    # better on two, worse on the third by less than 10 %: switch
    assert A.decision_rule(h6_ok, h6_ok, _op(500, 900, 0, 21), qwen)["outcome"].startswith("propose")
    # qwen has 0 failures: any bge failure is more than 10 % worse
    assert A.decision_rule(h6_ok, h6_ok, _op(500, 900, 1, 0), _op(1500, 4000, 0, 20))["outcome"].startswith("keep")


def test_operational_summary_counts_changes():
    calls = [{"latency_ms": 100}, {"latency_ms": 200}, {"latency_ms": 5000, "error": "Timeout"}]
    top = {"q1": [["a"], ["a"], None], "q2": [["a"], ["b"], ["a"]], "q3": [None, None, ["a"]]}
    s = A.operational_summary(calls, top)
    assert (s["failed_calls"], s["queries_compared"], s["queries_top10_changed"]) == (1, 2, 1)
    assert s["median_ms"] == pytest.approx(150)


def test_auc_and_threshold():
    assert A.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert A.auc([0.5], [0.5]) == 0.5
    best = A.best_threshold([0.9, 0.8, 0.3], [0.1, 0.4])
    assert best["threshold"] == 0.8 and best["youden_j"] == pytest.approx(2 / 3)


def test_stratum_c_overlap():
    queries = [q("c01-es", "C", "c01", "es"), q("c01-en", "C", "c01", "en")]
    runs = {D.S1: {"c01-es": {"ids": list("abcdefghij")}, "c01-en": {"ids": list("abcdeklmno")}}}
    scores = {D.S1: {"c01-es": {"ndcg_at_10": 0.4}, "c01-en": {"ndcg_at_10": 0.6}}}
    out = A.stratum_c(queries, runs, scores, [D.S1])
    assert out["mean_pairwise_top10_overlap"][D.S1] == pytest.approx(0.5)
    assert out["ndcg_at_10_by_language"][D.S1]["es"] == pytest.approx(0.4)


# ------------------------------------------------------------------ annotation sheet


def test_annotation_sheet_is_blind_and_agreement_command(tmp_path):
    queries = [q(f"b{i}", "B", language=("en" if i % 2 else "es")) for i in range(6)] + [q("d1", "D")]
    books, grades = {}, {}
    for i in range(6):
        grades[f"b{i}"] = {}
        for j in range(10):
            work = f"w{i}-{j}"
            books[work] = Book(id=uuid4(), title=f"T{work}", authors=["A"], source="s", source_id=work, description="d")
            grades[f"b{i}"][work] = j % 4
    grades["d1"] = {"wd": 0}
    summary = build_sheet(queries, grades, books, tmp_path, n=20)
    assert summary["pairs"] == 20 and summary["population"] == 60
    with (tmp_path / SHEET_FILE).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0].keys()) == SHEET_COLUMNS
    assert all(row["grade"] == "" for row in rows)
    key = json.loads((tmp_path / KEY_FILE).read_text())
    assert all(v["need_id"] != "d1" for v in key.values())
    # a synthetic filled sheet: the human copies the judge's grade except on two rows, one row left empty
    for n, row in enumerate(rows):
        pair = key[row["pair_id"]]
        grade = grades[pair["need_id"]][pair["work_id"]]
        row["grade"] = "" if n == 0 else str((grade + 1) % 4 if n in (1, 2) else grade)
    filled = tmp_path / "filled.csv"
    with filled.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert len(read_sheet(filled)) == 19
    result = annotation_agreement(filled, key, {"judge": grades}, {x.query_id: x.language for x in queries})
    row = result["judges"]["judge"]
    assert row["pairs"] == 19
    assert row["overall"]["exact_agreement"] == pytest.approx(17 / 19)


# ------------------------------------------------------------------ offline report


def test_offline_report_from_saved_files(tmp_path, monkeypatch):
    """The report builds from files only: any socket use fails the test."""
    import socket

    from app.evaluation.catalog_evaluation.cli import main
    from app.evaluation.relevance_judge import LLMJudgment

    data = tmp_path / "data"
    queries = ([EvalQuery(f"q{i:02d}", f"q{i:02d}", f"query {i}", "en" if i % 2 else "es", D.A_TYPE, "A", f"query {i}")
                for i in range(12)]
               + [EvalQuery("d-en", "d-en", "train times", "en", "unanswerable", "D", "train")])
    (data / "queries").mkdir(parents=True)
    (data / "queries" / "queries.json").write_text(json.dumps({"queries": [x.to_dict() for x in queries]}))
    rng = np.random.default_rng(0)
    works = [f"w{i}" for i in range(30)]
    pool = {}
    for system in D.SYSTEM_NAMES:
        path = data / "runs" / f"{system}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for x in queries:
                ids = list(rng.permutation(works))
                extra = {"top_rerank_score": float(rng.random())} if system == D.S5 else {}
                handle.write(json.dumps({"query_id": x.query_id, "ids": ids, "scores": [None] * 30,
                                         "timings_ms": {"lexical": 10.0}, "error": None, "extra": extra}) + "\n")
                pool.setdefault(x.need_id, set()).update(ids[:10])
    pool_doc = {n: [{"work_id": w, "source": "s", "source_id": w} for w in sorted(ws)] for n, ws in pool.items()}
    (data / "pool.json").write_text(json.dumps(pool_doc))
    for model in D.JUDGES:
        cache = judge_cache(data, model)
        for need, ws in pool_doc.items():
            for w in ws:
                cache.add(LLMJudgment(query_id=need, source="s", source_id=w["source_id"], grade=int(w["work_id"][1:]) % 4,
                                      rationale="r", model=model, prompt_version="v1.0", date="2026-11-12"))
    SpendLedger(data / "spend_ledger.jsonl", 5.0).add("2-runs", "rerank", D.RERANKER, 0.001)

    def no_network(*args, **kwargs):
        raise AssertionError("the offline report opened a network connection")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    output = tmp_path / "REPORT.md"
    assert main(["--data-dir", str(data), "report", "--offline", "--output", str(output)]) == 0
    text = output.read_text()
    assert "LLM judges only" in text and "Deviations from the protocol" in text and "## Limits" in text
    assert "H6" in text and "S6 replication" in text
    results = json.loads((data / "results.json").read_text())
    assert results["pool"]["needs"] == 13
    assert set(results["conclusions"]) >= {"H1", "H2", "H3", "H4", "H5", "H6"}


def test_hybrid_rule():
    s4_better = {"difference": -0.03, "p": 0.001, "p_holm": 0.002}
    unclear = {"difference": -0.03, "p": 0.2, "p_holm": 0.4}
    assert A.hybrid_rule(s4_better, s4_better)["s4_significantly_better_than_s5"] is True
    assert A.hybrid_rule(s4_better, unclear)["outcome"] == "keep the deployed hybrid S5"
    s5_better = {"difference": 0.03, "p": 0.001, "p_holm": 0.002}
    assert A.hybrid_rule(s5_better, s5_better)["s4_significantly_better_than_s5"] is False
