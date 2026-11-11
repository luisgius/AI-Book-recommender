"""
Tests for the LLM-recommender evaluation (app/evaluation/llm_recommender_evaluation.py).

No model is called: completions are plain functions and the systems'
outputs are built by hand.
"""

import json
from datetime import datetime
from typing import List
from uuid import uuid4

import pytest

from app.domain.entities import Book
from app.evaluation import llm_recommender_evaluation as ev
from app.evaluation.types import TestQuery
from app.infrastructure.llm.llm_recommender import (
    LONG_CONTEXT_OPERATION,
    RERANK_OPERATION,
    ClosedBookOutcome,
    LLMOutput,
    LLMRecommender,
    LLMRequest,
    listwise_request,
)
from app.infrastructure.llm.schemas_recommender import CandidateRecommendationsLLM


def make_book(title: str, authors: List[str], i: int = 0) -> Book:
    return Book(id=uuid4(), title=title, authors=authors, description="d", language="en", categories=["C"],
                published_date=datetime(2000, 1, 1), source="google_books", source_id=f"s{i:03d}")


def make_request(query: str = "q", user: str = "user") -> LLMRequest:
    return LLMRequest(RERANK_OPERATION, query, "system", user, CandidateRecommendationsLLM)


class CountingCompletion:
    def __init__(self, output: LLMOutput) -> None:
        self.output = output
        self.calls = 0

    def __call__(self, request: LLMRequest) -> LLMOutput:
        self.calls += 1
        return self.output


OUTPUT = LLMOutput(parsed={"recommendations": [{"id": "B1", "reason": "r"}]}, latency_s=2.0, input_tokens=100,
                   output_tokens=10, cost_usd=0.001)


# =============================================================================
# Cache
# =============================================================================


class TestCache:
    def test_a_cache_hit_avoids_a_call(self, tmp_path):
        inner = CountingCompletion(OUTPUT)
        cache = ev.OutputCache(tmp_path / "c.jsonl")
        completion = ev.CachedCompletion(cache, system="s", run=1, model="m", inner=inner)
        assert completion(make_request()) == OUTPUT
        assert completion(make_request()) == OUTPUT
        assert inner.calls == 1 and completion.calls_made == 1

    def test_the_key_depends_on_run_query_and_prompt(self, tmp_path):
        inner = CountingCompletion(OUTPUT)
        cache = ev.OutputCache(tmp_path / "c.jsonl")
        ev.CachedCompletion(cache, system="s", run=1, model="m", inner=inner)(make_request())
        ev.CachedCompletion(cache, system="s", run=2, model="m", inner=inner)(make_request())
        ev.CachedCompletion(cache, system="s", run=1, model="m", inner=inner)(make_request(query="q2", user="u2"))
        ev.CachedCompletion(cache, system="s", run=1, model="m", inner=inner)(make_request(user="other candidates"))
        assert inner.calls == 4

    def test_the_cache_survives_a_restart_and_no_llm_reads_it(self, tmp_path):
        path = tmp_path / "c.jsonl"
        ev.CachedCompletion(ev.OutputCache(path), system="s", run=1, model="m",
                            inner=CountingCompletion(OUTPUT))(make_request())
        reloaded = ev.CachedCompletion(ev.OutputCache(path), system="s", run=1, model="m", inner=None)
        assert reloaded(make_request()) == OUTPUT
        entry = json.loads(path.read_text().splitlines()[0])
        assert (entry["system"], entry["run"], entry["model"], entry["query_text"]) == ("s", 1, "m", "q")

    def test_no_llm_fails_loudly_on_a_miss(self, tmp_path):
        completion = ev.CachedCompletion(ev.OutputCache(tmp_path / "c.jsonl"), system="s", run=1, model="m")
        with pytest.raises(ev.MissingLLMOutputError, match="No cached output of s"):
            completion(make_request())
        assert isinstance(completion.last_error, ev.MissingLLMOutputError)

    def test_spend_limit(self, tmp_path):
        meter = ev.SpendMeter(limit_usd=0.0015)
        completion = ev.CachedCompletion(ev.OutputCache(tmp_path / "c.jsonl"), system="s", run=1, model="m",
                                         inner=CountingCompletion(OUTPUT), meter=meter)
        completion(make_request(user="a"))
        with pytest.raises(ev.SpendLimitError):
            completion(make_request(user="b"))

    def test_recommender_through_the_cache(self, tmp_path):
        books = [make_book(f"Book {i}", ["A B"], i) for i in range(3)]
        inner = CountingCompletion(OUTPUT)
        cache = ev.OutputCache(tmp_path / "c.jsonl")
        for _ in range(2):
            recommender = LLMRecommender(ev.CachedCompletion(cache, system="r2", run=1, model="m", inner=inner), "m")
            assert recommender.rank_books("q", books, LONG_CONTEXT_OPERATION).valid_ids == ["B1"]
        assert inner.calls == 1


# =============================================================================
# R3 matching rule
# =============================================================================


class TestMatching:
    def test_normalize_text(self):
        assert ev.normalize_text("  La Sombra del Viento!  ") == "la sombra del viento"
        assert ev.normalize_text("Programación en PYTHON 3.11") == "programacion en python 3 11"

    def test_main_title_drops_subtitles_and_notes(self):
        assert ev.main_title("Clean Code: A Handbook of Agile Software Craftsmanship") == "clean code"
        assert ev.main_title("Python Crash Course (2nd Edition)") == "python crash course"
        assert ev.main_title("Dune - The Graphic Novel") == "dune"
        assert ev.main_title("Fortunata y Jacinta") == "fortunata y jacinta"

    @pytest.mark.parametrize("name, expected", [
        ("Benito Pérez Galdós", "galdos"),
        ("Pérez Galdós, Benito", "perez galdos"),
        ("J. R. R. Tolkien", "tolkien"),
        ("Aurélien Géron and Someone Else", "geron"),
        ("", ""),
    ])
    def test_surname(self, name, expected):
        assert ev.surname(name) == expected

    def test_match_on_title_and_surname(self):
        catalog = [make_book("Fortunata y Jacinta", ["Benito Pérez Galdós"], 1),
                   make_book("Clean Code: A Handbook", ["Robert C. Martin"], 2)]
        index = ev.title_index(catalog)
        assert ev.match_book("Fortunata y Jacinta", "Perez Galdos, Benito", index) is catalog[0]
        assert ev.match_book("CLEAN CODE", "Robert Martin", index) is catalog[1]

    def test_no_match_with_another_author_or_title(self):
        index = ev.title_index([make_book("Dune", ["Frank Herbert"])])
        assert ev.match_book("Dune", "Brian Herbert Smith", index) is None
        assert ev.match_book("Dune Messiah", "Frank Herbert", index) is None
        assert ev.match_book("Dune", "", index) is None

    def test_first_catalog_match_wins(self):
        first, second = make_book("Dune", ["Frank Herbert"], 1), make_book("Dune", ["F. Herbert"], 2)
        assert ev.match_book("Dune", "Frank Herbert", ev.title_index([first, second])) is first

    def test_run_closed_book_ranks_matches_once(self):
        catalog = [make_book("Dune", ["Frank Herbert"], 1), make_book("Emma", ["Jane Austen"], 2)]
        output = LLMOutput(parsed={"recommendations": [
            {"title": "Dune", "author": "Frank Herbert", "reason": "r"},
            {"title": "Not Here", "author": "Nobody", "reason": "r"},
            {"title": "Dune: Deluxe Edition", "author": "Herbert, Frank", "reason": "r"},
            {"title": "Emma", "author": "Jane Austen", "reason": "r"}]})
        recommender = LLMRecommender(CountingCompletion(output), "m")
        run, outcomes, matches = ev.run_closed_book(recommender, ev.title_index(catalog), {"q1": "x"}, workers=2)
        assert run["q1"] == [catalog[0].id, catalog[1].id]
        assert [b is not None for b in matches["q1"]] == [True, False, True, True]
        grounding = ev.closed_book_grounding(matches, outcomes, {"q1": "en"})
        assert (grounding["recommendations"], grounding["in_catalog"]) == (4, 3)
        assert grounding["by_language"]["en"]["in_catalog_share"] == 0.75
        assert grounding["same_title_other_author"] is None

    def test_same_title_other_author_is_listed(self):
        catalog = [make_book("Machine Learning", ["Ethem Alpaydin"], 1)]
        output = LLMOutput(parsed={"recommendations": [
            {"title": "Machine Learning: A Probabilistic Perspective", "author": "Kevin P. Murphy", "reason": ""},
            {"title": "Pattern Recognition", "author": "Christopher Bishop", "reason": ""}]})
        index = ev.title_index(catalog)
        _, outcomes, matches = ev.run_closed_book(LLMRecommender(CountingCompletion(output), "m"), index,
                                                  {"q1": "x"}, workers=1)
        others = ev.closed_book_grounding(matches, outcomes, {"q1": "en"}, index)["same_title_other_author"]
        assert [(o["author"], o["catalog_authors"]) for o in others] == [("Kevin P. Murphy", ["Ethem Alpaydin"])]


# =============================================================================
# Summaries
# =============================================================================


def outcome(ids, count=12):
    books = [make_book(f"B{i}", ["A"], i) for i in range(count)]
    output = LLMOutput(parsed={"recommendations": [{"id": i, "reason": ""} for i in ids]})
    return LLMRecommender(CountingCompletion(output), "m").rank_books("q", books)


class TestSummaries:
    def test_listwise_grounding(self):
        outcomes = [outcome([f"B{i}" for i in range(1, 11)]), outcome(["B1", "B1", "Z9", "B40"]),
                    outcome([f"B{i}" for i in range(1, 13)]), outcome([])]
        g = ev.listwise_grounding(outcomes)
        assert g["lists"] == 4 and g["ids_returned"] == 26 and g["empty_answers"] == 1
        assert (g["unknown_ids"], g["duplicate_ids"]) == (2, 1)
        assert g["unknown_share"] == pytest.approx(2 / 26)
        assert (g["lists_completed"], g["positions_filled"]) == (2, 19)
        assert g["lists_with_more_than_10_ids"] == 1 and g["failed_calls"] == 0

    def test_determinism(self):
        a = {"q1": [1, 2, 3], "q2": [1, 2, 3], "q3": [1, 2, 3]}
        b = {"q1": [1, 2, 3], "q2": [2, 1, 3], "q3": [1, 2, 4]}
        assert ev.determinism(a, b) == {"queries": 3, "identical_lists": 1, "identical_sets": 2}

    def test_llm_cost_adds_the_local_part(self):
        outputs = [LLMOutput(parsed={}, latency_s=s, input_tokens=100, output_tokens=10, cost_usd=0.01)
                   for s in (1.0, 2.0, 3.0)]
        cost = ev.llm_cost(outputs, local_ms=[100.0, 100.0, 100.0])
        assert cost["latency_median_s"] == pytest.approx(2.1)
        assert cost["llm_latency_median_s"] == pytest.approx(2.0)
        assert cost["cost_total_usd"] == pytest.approx(0.03) and cost["input_tokens_mean"] == 100


# =============================================================================
# Scaling arithmetic
# =============================================================================


class TestScaling:
    def test_tokens_per_book_splits_by_characters(self):
        # 10,000 tokens over 40,000 characters, 39,000 of them the catalog of 100 books
        per_book, fixed = ev.tokens_per_book([10_000], [40_000], 39_000, 100)
        assert per_book == pytest.approx(97.5) and fixed == pytest.approx(250)

    def test_rows_and_context_limit(self):
        rows = ev.scaling_rows(100.0, 200.0, 50.0, [1_000, 20_000], context_length=1_000_000,
                               max_output_tokens=2_000, price_input=1e-7, price_output=1e-6,
                               price_cache_read=1e-8)
        assert rows[0]["prompt_tokens"] == 100_200 and rows[0]["fits_context"]
        assert rows[0]["cost_usd"] == pytest.approx(100_200 * 1e-7 + 50 * 1e-6)
        assert rows[0]["cost_cached_catalog_usd"] == pytest.approx(200 * 1e-7 + 100_000 * 1e-8 + 50 * 1e-6)
        assert rows[1]["prompt_tokens"] == 2_000_200 and not rows[1]["fits_context"]
        assert ev.max_catalog_size(100.0, 200.0, 1_000_000, 2_000) == 9_978

    def test_tokens_per_book_needs_matching_inputs(self):
        with pytest.raises(ValueError):
            ev.tokens_per_book([1, 2], [3], 1, 1)


# =============================================================================
# Report and Markdown (synthetic measurements)
# =============================================================================


def synthetic_measurements():
    books = [make_book(f"Book {i}", ["A"], i) for i in range(12)]
    queries = [TestQuery("q1", "uno", language="es"), TestQuery("q2", "two", language="en")]
    ids = [b.id for b in books]
    runs = {name: {"q1": ids[:10], "q2": ids[2:12]} for name in
            (ev.R0, ev.R1, ev.R2, ev.REFERENCE, ev.R1 + ev.RUN2_SUFFIX, ev.R2 + ev.RUN2_SUFFIX)}
    runs[ev.R3] = {"q1": ids[:2], "q2": []}
    out = LLMOutput(parsed={"recommendations": []}, latency_s=1.0, input_tokens=5000, output_tokens=300,
                    cost_usd=0.001)
    listwise = {r: {q.query_id: LLMRecommender(CountingCompletion(out), "m").rank_books(q.text, books)
                    for q in queries} for r in (1, 2)}
    closed = {q.query_id: ClosedBookOutcome(q.text, [{"title": "x", "author": "y", "reason": ""}], out)
              for q in queries}
    request = listwise_request("uno", books, LONG_CONTEXT_OPERATION)
    m = ev.Measurements(
        runs=runs, rag=listwise, rag_local_ms={r: {"q1": 10.0, "q2": 12.0} for r in (1, 2)}, long_context=listwise,
        closed_book=closed, closed_book_matches={"q1": [books[0]], "q2": [None]},
        pipeline_latency={"median_ms": 2000.0, "p95_ms": 2500.0},
        long_context_prompt_chars={"q1": len(request.user_prompt) + len(request.system_prompt),
                                   "q2": len(request.user_prompt) + len(request.system_prompt)},
        long_context_catalog_chars=len(request.user_prompt) - 300,
    )
    judgments = {"q1": {i: (1 if n < 3 else 0) for n, i in enumerate(ids)},
                 "q2": {i: (2 if n == 5 else 0) for n, i in enumerate(ids)}}
    info = {"model": "m", "context_length": 1_000_000, "fetched_at": "2026-11-11",
            "pricing": {"prompt": "0.00000015", "completion": "0.00000047", "input_cache_read": "0.000000016"}}
    return m, queries, judgments, info, books


class TestReport:
    def test_report_and_markdown(self):
        m, queries, judgments, info, books = synthetic_measurements()
        results = ev.build_report(m, queries, judgments, info, n_resamples=200)
        assert set(results["contrasts"]) == {"L1", "L2"}
        assert results["determinism"][ev.R1]["identical_lists"] == 2
        assert results["grounding"][ev.R1]["run1"]["lists_completed"] == 2
        assert results["scaling"]["rows"][0]["books"] == 12
        document = {
            "design": ev.design(200, 0),
            "prompts": {"version": "v1.0", "templates": [("t", "text")], "book_entry_example": "[B1] ..."},
            "models": {"system_llm": {"provider": "openrouter", "model": "m", "reasoning_text": "off",
                                      "max_tokens": 2048},
                       "judge": {"model": "j", "prompt_version": "v1.0"}, "openrouter_model_info": info},
            "results": results,
            "judgments": {"num_final": 24, "num_original": 10, "num_llm": 14, "added_by_this_task": 3,
                          "added_grade_distribution": [1, 1, 1, 0], "judging_runs": []},
            "reproduction_check": {"file": "f.json", "queries": 2, "identical": 2, "reference_identical": 2},
            "spend": {"total_usd": 0.5, "recommender_calls_usd": 0.4, "judge_usd": 0.1, "key_usage_delta_usd": None},
            "run": {"command": "python -m x", "date": "2026-11-11", "git_commit": "abc", "num_books": 12,
                    "extended_judgments_path": "e.json", "reproduce": ["a", "b"], "cache_file": "c",
                    "judge_cache_file": "j"},
        }
        text = ev.build_markdown(document)
        for heading in ("## Design (fixed in advance)", "## Prompts (as used)", "### Pre-declared contrasts",
                        "## Grounding", "## Cost", "## Scaling of R2", "## Determinism", "## Reproduce"):
            assert heading in text
        assert "L1:" in text and "1,000,000" in text

    def test_unjudged_r1_fails(self):
        from app.evaluation.retrieval_ablation import UnjudgedResultsError

        m, queries, judgments, info, _ = synthetic_measurements()
        m.runs[ev.R1]["q1"] = m.runs[ev.R1]["q1"][:9] + [uuid4()]
        with pytest.raises(UnjudgedResultsError):
            ev.build_report(m, queries, judgments, info, n_resamples=50)
