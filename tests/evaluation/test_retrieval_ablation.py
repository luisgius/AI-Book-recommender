"""
Tests for the retrieval ablation (app/evaluation/retrieval_ablation.py).

Everything runs on fakes: searchers that return fixed rankings, a fake
judge, fake repositories and rerankers. No model is downloaded and no LLM
is called.

The small collection used by the run_ablation tests has 12 books (b00-b11)
and two queries, q1 (English) and q2 (Spanish, with an accented variant).
Every fake system returns 10 books, as the real systems do, so Judged@10 can
reach 1.0.
"""

import json
import math
from typing import Dict, List, Sequence
from uuid import UUID

import pytest

from app.domain.entities import Book, SearchResult
from app.domain.value_objects import SearchResponse
from app.evaluation import comparison
from app.evaluation.comparison import compare_pairs
from app.evaluation.evaluation_job import DEFAULT_QUERIES_PATH, load_test_queries
from app.evaluation.relevance_judge import JudgmentCache, merge_judgments
from app.evaluation.retrieval_ablation import (
    CONTRASTS,
    SYSTEMS,
    Contrast,
    IncompleteJudgingError,
    RerankerFailedError,
    SearchDegradedError,
    SystemSpec,
    UnjudgedResultsError,
    accented_texts,
    build_markdown,
    check_hybrid_response,
    execution_order,
    make_searcher,
    previous_judging_run,
    require_judged,
    run_ablation,
)
from app.evaluation.types import TestQuery
from app.infrastructure.search.text_normalization import fold_accents

# =============================================================================
# The declared design
# =============================================================================


def test_design_declares_eleven_systems_and_seven_contrasts():
    names = [spec.name for spec in SYSTEMS]
    assert len(names) == 11 and len(set(names)) == 11
    assert [c.name for c in CONTRASTS] == ["C1", "C2", "C3", "C4a", "C4b", "C5", "C6"]
    for contrast in CONTRASTS:
        assert contrast.system in names and contrast.reference in names
    by_name = {spec.name: spec for spec in SYSTEMS}
    assert by_name["hybrid_legacy"].lexical == "whitespace"
    assert by_name["hybrid_legacy"].embedding == "all-MiniLM-L6-v2"
    assert by_name["hybrid_legacy"].reranker is None
    assert by_name["hybrid_norm_e5_bge"].reranker == "BAAI/bge-reranker-v2-m3"
    assert by_name["hybrid_norm_e5_mmarco"].reranker == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def test_execution_order_loads_each_embedding_model_once():
    order = execution_order(SYSTEMS)
    models = [spec.embedding for spec in order]
    assert models[:2] == [None, None]  # lexical-only systems first
    changes = sum(1 for a, b in zip(models, models[1:]) if a != b)
    assert changes == len(set(models)) - 1  # one contiguous group per model
    assert sorted(spec.name for spec in order) == sorted(spec.name for spec in SYSTEMS)


def test_a_reranker_needs_a_hybrid_system():
    with pytest.raises(ValueError):
        SystemSpec("bad", lexical="normalized", reranker="some/model")


# =============================================================================
# Contrasts: paired tests and Holm across exactly the declared family
# =============================================================================


def _report(scores: Dict[str, List[float]]) -> dict:
    """A minimal compare_systems-like report with per-query nDCG@10."""
    query_ids = [f"q{i}" for i in range(1, 5)]
    return {
        "query_ids": query_ids,
        "systems": {
            system: {"per_query": {qid: {"ndcg_at_10": value} for qid, value in zip(query_ids, values)}}
            for system, values in scores.items()
        },
    }


SCORES = {
    "ref": [0.5, 0.5, 0.5, 0.5],
    "x": [0.7, 0.7, 0.7, 0.7],  # +0.2 on every query
    "y": [0.4, 0.6, 0.5, 0.3],  # -0.1, +0.1, 0, -0.2: mean -0.05
    "z": [0.6, 0.5, 0.5, 0.5],  # +0.1, 0, 0, 0: mean +0.025
}
PAIRS = [("X", "x", "ref"), ("Y", "y", "ref"), ("Z", "z", "ref")]


def test_contrasts_with_holm_on_a_hand_computed_example(monkeypatch):
    # Fixed raw p-values so the Holm step can be computed by hand:
    # sorted 0.01 (X), 0.03 (Z), 0.04 (Y) -> 3 * 0.01 = 0.03, 2 * 0.03 = 0.06,
    # 1 * 0.04 = 0.04 raised to the running maximum 0.06.
    raw = iter([0.01, 0.04, 0.03])
    monkeypatch.setattr(
        comparison, "paired_permutation_test",
        lambda a, b, n_resamples, seed: (sum(a) / len(a) - sum(b) / len(b), next(raw)),
    )
    result = compare_pairs(_report(SCORES), PAIRS, n_resamples=200, seed=0)

    assert list(result) == ["X", "Y", "Z"]
    assert result["X"]["p_value"] == 0.01 and result["Y"]["p_value"] == 0.04 and result["Z"]["p_value"] == 0.03
    assert result["X"]["p_value_holm"] == pytest.approx(0.03)
    assert result["Z"]["p_value_holm"] == pytest.approx(0.06)
    assert result["Y"]["p_value_holm"] == pytest.approx(0.06)
    assert result["X"]["mean_difference"] == pytest.approx(0.2)
    assert result["Y"]["mean_difference"] == pytest.approx(-0.05)
    assert result["Z"]["mean_difference"] == pytest.approx(0.025)
    assert result["Y"]["mean_system"] == pytest.approx(0.45) and result["Y"]["mean_reference"] == 0.5
    # A constant difference has a degenerate bootstrap interval
    assert result["X"]["ci_low"] == pytest.approx(0.2) and result["X"]["ci_high"] == pytest.approx(0.2)
    assert (result["Y"]["system"], result["Y"]["reference"]) == ("y", "ref")


def test_holm_family_is_only_the_given_contrasts(monkeypatch):
    monkeypatch.setattr(comparison, "paired_permutation_test", lambda a, b, n_resamples, seed: (0.0, 0.02))
    two = compare_pairs(_report(SCORES), PAIRS[:2], n_resamples=50)
    three = compare_pairs(_report(SCORES), PAIRS, n_resamples=50)
    assert two["X"]["p_value_holm"] == pytest.approx(0.04)  # 2 * 0.02
    assert three["X"]["p_value_holm"] == pytest.approx(0.06)  # 3 * 0.02


def test_identical_systems_have_p_one_with_the_real_test():
    result = compare_pairs(_report({"a": [0.1, 0.2, 0.3, 0.4], "b": [0.1, 0.2, 0.3, 0.4]}), [("K", "a", "b")],
                           n_resamples=100)
    assert result["K"]["mean_difference"] == 0.0
    assert result["K"]["p_value"] == 1.0 and result["K"]["p_value_holm"] == 1.0


def test_contrast_names_must_be_unique_and_systems_known():
    with pytest.raises(ValueError):
        compare_pairs(_report(SCORES), [("X", "x", "ref"), ("X", "y", "ref")], n_resamples=10)
    with pytest.raises(ValueError):
        compare_pairs(_report(SCORES), [("X", "missing", "ref")], n_resamples=10)


# =============================================================================
# Fakes for run_ablation
# =============================================================================


BOOKS = [Book(id=UUID(int=i + 1), title=f"Book {i:02d}", authors=["A"], source="test", source_id=f"b{i:02d}")
         for i in range(12)]
BOOK = {book.source_id: book for book in BOOKS}

Q1 = TestQuery("q1", "history books", language="en")
Q2 = TestQuery("q2", "novela historica", language="es", text_accented="novela histórica")
QUERIES = [Q1, Q2]

# Rankings by query text. The accented q2 brings in b11, which nothing else returns.
TOP = [f"b{i:02d}" for i in range(10)]
RANKINGS = {
    "lex": {"history books": TOP, "novela historica": TOP, "novela histórica": TOP[1:] + ["b11"]},
    "vec": {"history books": TOP[::-1], "novela historica": TOP[::-1], "novela histórica": TOP[::-1]},
}
FAKE_SYSTEMS = [SystemSpec("lex", lexical="whitespace"), SystemSpec("vec", embedding="fake-model")]
FAKE_CONTRASTS = [Contrast("K1", "vec", "lex", "Is vec better than lex?")]


class FakeSearcher:
    """Returns the fixed ranking of a system for a query text; records every text."""

    def __init__(self, system: str):
        self.system = system
        self.texts: List[str] = []

    def __call__(self, text: str) -> List[SearchResult]:
        self.texts.append(text)
        ranking = RANKINGS[self.system].get(text, TOP)
        return [SearchResult(book=BOOK[sid], final_score=1.0 / rank, rank=rank, source="lexical")
                for rank, sid in enumerate(ranking, start=1)]


class FakeJudge:
    """Grades 1 for every pair; records the query text it was shown."""

    model_name = "fake-judge:1b"
    prompt_version = "v1.0"

    def __init__(self):
        self.calls: List[tuple] = []

    def judge(self, query_text: str, book: Book) -> tuple:
        self.calls.append((query_text, book.source_id))
        return 1, "Fake rationale."


def raw_judgments(pairs: Dict[str, Sequence[str]], grade: int = 2) -> Dict[str, List[dict]]:
    """Original-style judgments: query id -> source ids, all with the same grade."""
    return {qid: [{"source": "test", "source_id": sid, "relevance": grade} for sid in sids]
            for qid, sids in pairs.items()}


def run(searchers: Dict[str, FakeSearcher], original, **kwargs):
    return run_ablation(
        lambda spec: searchers[spec.name], QUERIES, BOOKS, original,
        systems=FAKE_SYSTEMS, contrasts=FAKE_CONTRASTS, n_resamples=200, seed=0, log=lambda _m: None, **kwargs,
    )


@pytest.fixture
def searchers() -> Dict[str, FakeSearcher]:
    return {"lex": FakeSearcher("lex"), "vec": FakeSearcher("vec")}


# =============================================================================
# --no-judge mode
# =============================================================================


def test_no_judge_fails_loudly_on_an_unjudged_result(searchers):
    # b11 (retrieved only by the accented q2) is not judged
    complete_but_b11 = raw_judgments({"q1": TOP, "q2": TOP})
    with pytest.raises(UnjudgedResultsError, match=r"1 \(query, book\) pairs .*without --no-judge"):
        run(searchers, complete_but_b11, extended_raw=complete_but_b11)


def test_no_judge_runs_when_every_top_10_result_is_judged(searchers):
    judged = raw_judgments({"q1": TOP, "q2": TOP + ["b11"]})
    report, merged = run(searchers, raw_judgments({"q1": TOP[:3], "q2": TOP[:3]}), extended_raw=judged)
    assert merged is None
    assert report["judgments"]["judging_run"] is None
    assert all(data["metrics"]["judged_at_10"]["mean"] == 1.0 for data in report["systems"].values())
    # On the original judgments (3 per query) only 3 of the 10 are judged
    assert report["original_judgments"]["metrics"]["lex"]["judged_at_10"] == pytest.approx(0.3)
    assert set(report["contrasts"]["final"]) == {"K1"} and set(report["contrasts"]["original"]) == {"K1"}


def test_no_judge_keeps_the_judging_facts_only_for_the_same_judgments_file(tmp_path):
    previous = tmp_path / "results.json"
    judging = {"judged_in_this_run": 5, "already_cached": 0, "failed_in_this_run": 0, "seconds_per_judgment": 4.0}
    previous.write_text(json.dumps({
        "judgments": {"judging_run": judging},
        "run": {"date": "2026-11-11T15:00:00+00:00", "extended_judgments_sha256": "abc"},
    }), encoding="utf-8")
    carried = previous_judging_run(str(previous), "abc")
    assert carried == {**judging, "carried_from_run_of": "2026-11-11T15:00:00+00:00"}
    assert previous_judging_run(str(previous), "another file") is None
    assert previous_judging_run(str(tmp_path / "missing.json"), "abc") is None


def test_require_judged_lists_the_holes():
    require_judged({"s": {"q1": ["a", "b"]}}, {"q1": {"a": 1, "b": 0}}, k=10)  # complete: no error
    with pytest.raises(UnjudgedResultsError, match=r"\(q1, b\)"):
        require_judged({"s": {"q1": ["a", "b"]}}, {"q1": {"a": 1}}, k=10)


def test_exactly_one_mode_must_be_configured(searchers, tmp_path):
    original = raw_judgments({"q1": TOP, "q2": TOP})
    with pytest.raises(ValueError):
        run(searchers, original)
    with pytest.raises(ValueError):
        run(searchers, original, extended_raw=original, judge=FakeJudge(),
            cache=JudgmentCache.for_judge(tmp_path, "fake-judge:1b", "v1.0"))


# =============================================================================
# Judge mode and the accented variants
# =============================================================================


def test_judge_fills_the_holes_with_the_stored_query_text(searchers, tmp_path):
    judge = FakeJudge()
    cache = JudgmentCache.for_judge(tmp_path, judge.model_name, judge.prompt_version)
    original = raw_judgments({"q1": TOP, "q2": TOP[:5]})
    report, merged = run(searchers, original, judge=judge, cache=cache)

    # Holes: q2 b05-b09 (stored runs) and q2 b11 (accented run only)
    assert sorted(sid for _, sid in judge.calls) == ["b05", "b06", "b07", "b08", "b09", "b11"]
    assert {text for text, _ in judge.calls} == {"novela historica"}  # never the accented text
    added = report["judgments"]["added_by_v06"]
    assert added["pairs"] == 6 and added["from_accented_runs_only"] == 1
    assert added["grade_distribution"] == [0, 6, 0, 0]
    assert report["judgments"]["judging_run"]["judged_in_this_run"] == 6
    assert report["judgments"]["llm_judges"] == [
        {"model": "fake-judge:1b", "prompt_version": "v1.0", "num_judgments": 6}
    ]
    # The merged file keeps the originals and adds the LLM grades with provenance
    assert merged == merge_judgments(original, cache.judgments())
    q2 = next(entry for entry in merged if entry["query_id"] == "q2")["judgments"]
    assert [item["judge"] for item in q2].count("llm") == 6

    # A second run finds nothing left to judge
    again = FakeJudge()
    run({"lex": FakeSearcher("lex"), "vec": FakeSearcher("vec")}, original, judge=again, cache=cache)
    assert again.calls == []


def test_a_judging_run_that_stops_early_writes_nothing(searchers, tmp_path):
    judge = FakeJudge()
    cache = JudgmentCache.for_judge(tmp_path, judge.model_name, judge.prompt_version)
    with pytest.raises(IncompleteJudgingError, match="Nothing was written"):
        run(searchers, raw_judgments({"q1": TOP, "q2": TOP[:5]}), judge=judge, cache=cache, limit=2)
    assert len(judge.calls) == 2


def test_accented_variants_are_searched_and_reported(searchers):
    judged = raw_judgments({"q1": TOP, "q2": TOP + ["b11"]})
    judged["q2"][0]["relevance"] = 3  # b00: only the stored q2 ranks it first
    report, _ = run(searchers, raw_judgments({"q1": TOP, "q2": TOP}), extended_raw=judged)

    # Each system searched the stored texts (+ one warm-up) and the accented q2 once
    assert searchers["lex"].texts.count("novela histórica") == 1
    assert searchers["lex"].texts.count("novela historica") == 1
    accent = report["accent"]
    assert accent["num_spanish_queries"] == 1 and accent["changed_query_ids"] == ["q2"]
    lex = accent["systems"]["lex"]["changed"]
    assert lex["ndcg_at_10_accented"] < lex["ndcg_at_10_stored"]
    assert lex["difference"] == pytest.approx(lex["ndcg_at_10_accented"] - lex["ndcg_at_10_stored"])
    assert accent["systems"]["lex"]["identical_top10_lists"] == 0
    assert accent["systems"]["vec"]["identical_top10_lists"] == 1
    assert accent["systems"]["vec"]["changed"]["difference"] == 0.0


def test_accented_texts_only_lists_variants_that_differ():
    queries = [Q1, Q2, TestQuery("q3", "cocina", language="es", text_accented="cocina")]
    assert accented_texts(queries) == {"q2": "novela histórica"}


def test_load_test_queries_reads_text_accented(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps([
        {"query_id": "q1", "text": "programacion", "language": "es", "text_accented": "programación"},
        {"query_id": "q2", "text": "python", "language": "en"},
    ], ensure_ascii=False), encoding="utf-8")
    first, second = load_test_queries(str(path))
    assert first.text_accented == "programación" and second.text_accented is None


def test_shipped_accented_variants_differ_only_in_diacritics():
    queries = load_test_queries(DEFAULT_QUERIES_PATH)
    with_variant = [q for q in queries if q.text_accented is not None]
    assert len(with_variant) == 14
    for query in with_variant:
        assert query.language == "es"
        assert query.text_accented != query.text
        assert fold_accents(query.text_accented) == query.text


def test_markdown_renders_every_section(searchers):
    judged = raw_judgments({"q1": TOP, "q2": TOP + ["b11"]})
    report, _ = run(searchers, raw_judgments({"q1": TOP, "q2": TOP}), extended_raw=judged)
    report["costs"] = {"fake-model": {"role": "embedding", "device": "cpu", "size_on_disk_bytes": 2**20,
                                      "load_seconds": 0.5, "index_build_seconds": 1.0, "dimension": 2,
                                      "index_source": "fake"}}
    report["run"] = {"date": "2026-11-11", "git_commit": "abc1234", "git_worktree_dirty": False,
                     "command": "python -m app.evaluation.retrieval_ablation --no-judge",
                     "reproduce": ["python -m app.evaluation.retrieval_ablation"], "device": "cpu",
                     "hardware": "test", "llm_server_note": "", "num_books": 12,
                     "extended_judgments_path": "x.json"}
    markdown = build_markdown(report)
    for heading in ("## Design", "## Reproduction check", "## Pre-declared contrasts", "## Sensitivity",
                    "## All systems", "## By query language", "## Accent robustness", "## Cost"):
        assert heading in markdown
    assert "K1: Is vec better than lex?" in markdown


# =============================================================================
# Hybrid systems: a silently failed reranker or a degraded search
# =============================================================================


class Repo:
    """Lexical or vector repository returning the books in catalog order."""

    def __init__(self, ready: bool = True):
        self.ready = ready

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        return [SearchResult(book=book, final_score=1.0 / (i + 1), rank=i + 1, source="lexical")
                for i, book in enumerate(BOOKS[:max_results])]

    def is_ready(self) -> bool:
        return self.ready


class Embeddings:
    def generate_query_embedding(self, text: str) -> List[float]:
        return [1.0, 0.0]

    def is_ready(self) -> bool:
        return True


class Reranker:
    """Reverses the head it receives, or raises like a model that cannot load."""

    def __init__(self, fail: bool = False):
        self.fail = fail

    def rerank(self, query_text, results):
        if self.fail:
            raise RuntimeError("model failed to load")
        from dataclasses import replace

        ordered = list(reversed(results))
        return [replace(r, rank=i, rerank_score=float(-i)) for i, r in enumerate(ordered, start=1)]

    def get_model_name(self) -> str:
        return "fake/reranker"

    def is_ready(self) -> bool:
        return not self.fail


HYBRID = SystemSpec("hyb", lexical="normalized", embedding="fake", reranker="fake/reranker")


def hybrid_search(reranker=None, vector_ready: bool = True):
    spec = HYBRID if reranker is not None else SystemSpec("hyb", lexical="normalized", embedding="fake")
    return make_searcher(spec, lexical=Repo(), vector=Repo(vector_ready), embeddings=Embeddings(),
                         reranker=reranker, depth=10, rerank_depth=5)


def test_a_failed_reranker_is_detected():
    with pytest.raises(RerankerFailedError, match="hyb"):
        hybrid_search(Reranker(fail=True))("anything")


def test_a_working_reranker_passes_and_reorders_the_head():
    results = hybrid_search(Reranker())("anything")
    assert len(results) == 10
    assert results[0].rerank_score is not None
    assert [r.book.source_id for r in results[:5]] == ["b04", "b03", "b02", "b01", "b00"]


def test_a_degraded_hybrid_search_is_detected():
    with pytest.raises(SearchDegradedError, match="hyb"):
        hybrid_search(vector_ready=False)("anything")


def test_check_hybrid_response_without_reranker():
    results = Repo().search(max_results=3)
    check_hybrid_response(SearchResponse(results=results), system="s", expects_reranker=False)
    with pytest.raises(RerankerFailedError):
        check_hybrid_response(SearchResponse(results=results), system="s", expects_reranker=True)
    check_hybrid_response(SearchResponse(results=[]), system="s", expects_reranker=True)  # nothing to check


def test_metrics_in_the_report_match_a_hand_computation(searchers):
    # q1 judged: b00 grade 3, b01-b09 grade 0. lex ranks b00 first (nDCG 1),
    # vec ranks it last: DCG = 7 / log2(11), IDCG = 7.
    judged = raw_judgments({"q1": TOP, "q2": TOP + ["b11"]}, grade=0)
    judged["q1"][0]["relevance"] = 3
    report, _ = run(searchers, judged, extended_raw=judged)
    per_query = report["systems"]
    assert per_query["lex"]["per_query"]["q1"]["ndcg_at_10"] == pytest.approx(1.0)
    assert per_query["vec"]["per_query"]["q1"]["ndcg_at_10"] == pytest.approx(1 / math.log2(11))
    assert report["contrasts"]["final"]["K1"]["mean_difference"] == pytest.approx((1 / math.log2(11) - 1) / 2)
