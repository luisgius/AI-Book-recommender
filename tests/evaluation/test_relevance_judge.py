"""
Tests for the LLM relevance judgments (app/evaluation/relevance_judge.py).

A fake judge stands in for the LLM, so no model is called.

The agreement example is small enough to compute by hand:

    original  [0, 0, 1, 1, 2, 3]     (languages en, en, en, es, es, es)
    llm       [0, 1, 1, 2, 2, 2]

    confusion (rows original, columns llm):
        [1, 1, 0, 0]
        [0, 1, 1, 0]      row totals    [2, 2, 1, 1]
        [0, 0, 1, 0]      column totals [1, 2, 3, 0]
        [0, 0, 1, 0]

    Quadratic weights (i - j)^2. Observed weighted disagreement: 1 + 1 + 1 = 3.
    Expected: sum_i sum_j r_i c_j (i - j)^2 / n = (2*14 + 2*4 + 1*6 + 1*20) / 6 = 62 / 6.
    kappa = 1 - 3 / (62 / 6) = 22 / 31.

    grade >= 1: original [0,0,1,1,1,1], llm [0,1,1,1,1,1]: p_o = 5/6,
        p_e = (4/6)(5/6) + (2/6)(1/6) = 22/36, kappa = (30 - 22) / (36 - 22) = 4/7.
    grade >= 2: original [0,0,0,0,1,1], llm [0,0,0,1,1,1]: p_o = 5/6,
        p_e = (2/6)(3/6) + (4/6)(3/6) = 1/2, kappa = (5/6 - 1/2) / (1/2) = 2/3.
    es only: original [1,2,3], llm [2,2,2] (constant): kappa = 1 - 2/2 = 0.
"""

import json
from uuid import uuid4

import pytest

from app.domain.entities import Book
from app.evaluation.comparison import compare_systems, describe_judgments
from app.evaluation.evaluation_job import load_relevance_judgments
from app.evaluation.relevance_judge import (
    MAX_CONSECUTIVE_FAILURES,
    JudgeTask,
    JudgmentCache,
    LLMJudgment,
    agreement_report,
    build_agreement_markdown,
    build_fill_markdown,
    cache_file_name,
    find_unjudged,
    judge_pairs,
    merge_judgments,
    stratified_sample,
)
from app.evaluation.types import TestQuery

MODEL = "fake-judge:1b"
PROMPT = "v1.0"


def make_book(source_id: str, title: str = "A book", description: str | None = "About things") -> Book:
    return Book(id=uuid4(), title=title, authors=["Someone"], description=description,
                source="google_books", source_id=source_id)


class FakeJudge:
    """Grades by title length; can fail on chosen calls."""

    model_name = MODEL
    prompt_version = PROMPT

    def __init__(self, fail_on: dict | None = None):
        self.calls: list[tuple[str, Book]] = []
        self.fail_on = fail_on or {}  # call number (1-based) -> exception to raise

    def judge(self, query_text: str, book: Book) -> tuple[int, str]:
        self.calls.append((query_text, book))
        error = self.fail_on.get(len(self.calls))
        if error is not None:
            raise error
        return len(book.title) % 4, f"Fake rationale for {book.title}."


def make_tasks(n: int) -> list[JudgeTask]:
    return [JudgeTask(f"q{i}", f"query {i}", make_book(f"B{i}", title="T" * (i + 1))) for i in range(n)]


def silent(_message: str) -> None:
    pass


@pytest.fixture
def cache(tmp_path) -> JudgmentCache:
    return JudgmentCache.for_judge(tmp_path, MODEL, PROMPT)


# =============================================================================
# Cache
# =============================================================================


class TestJudgmentCache:
    def test_one_file_per_model_and_prompt_version(self, tmp_path):
        assert cache_file_name("qwen3.8:27b-mlx", "v1.0") == "qwen3.8-27b-mlx__prompt-v1.0.jsonl"
        assert JudgmentCache.for_judge(tmp_path, "a/b:c", "v2").path == tmp_path / "a-b-c__prompt-v2.jsonl"

    def test_entries_carry_provenance(self, cache):
        judge_pairs(make_tasks(1), FakeJudge(), cache, log=silent)
        entry = json.loads(cache.path.read_text(encoding="utf-8").splitlines()[0])
        assert entry["query_id"] == "q0" and entry["source"] == "google_books" and entry["source_id"] == "B0"
        assert entry["grade"] == 1 and entry["rationale"] == "Fake rationale for T."
        assert entry["model"] == MODEL and entry["prompt_version"] == PROMPT
        assert entry["date"] and entry["query_text"] == "query 0" and entry["title"] == "T"

    def test_rerun_does_not_rejudge_cached_pairs(self, tmp_path, cache):
        tasks = make_tasks(3)
        first = judge_pairs(tasks, FakeJudge(), cache, limit=2, log=silent)
        assert (first.judged, first.remaining) == (2, 1)

        reloaded = JudgmentCache.for_judge(tmp_path, MODEL, PROMPT)
        assert len(reloaded) == 2
        judge = FakeJudge()
        second = judge_pairs(tasks, judge, reloaded, log=silent)
        assert [book.source_id for _, book in judge.calls] == ["B2"]
        assert (second.cached, second.judged, second.remaining) == (2, 1, 0)
        assert len(JudgmentCache.for_judge(tmp_path, MODEL, PROMPT)) == 3

    def test_interrupted_run_loses_only_the_pair_in_flight(self, tmp_path, cache):
        judge = FakeJudge(fail_on={3: KeyboardInterrupt()})
        with pytest.raises(KeyboardInterrupt):
            judge_pairs(make_tasks(5), judge, cache, log=silent)
        reloaded = JudgmentCache.for_judge(tmp_path, MODEL, PROMPT)
        assert sorted(key[0] for key in reloaded._entries) == ["q0", "q1"]

    def test_partial_last_line_is_skipped_and_appending_still_works(self, tmp_path, cache):
        judge_pairs(make_tasks(1), FakeJudge(), cache, log=silent)
        with cache.path.open("a", encoding="utf-8") as handle:
            handle.write('{"query_id": "q1", "sou')  # killed in the middle of a write
        reloaded = JudgmentCache.for_judge(tmp_path, MODEL, PROMPT)
        assert len(reloaded) == 1

        judge_pairs(make_tasks(3), FakeJudge(), reloaded, log=silent)
        assert len(JudgmentCache.for_judge(tmp_path, MODEL, PROMPT)) == 3

    def test_refuses_judgments_of_another_model(self, tmp_path, cache):
        judge_pairs(make_tasks(1), FakeJudge(), cache, log=silent)
        with pytest.raises(ValueError, match="holds judgments of other-model"):
            JudgmentCache(cache.path, "other-model", PROMPT)


class TestJudgePairs:
    def test_duplicate_pairs_are_judged_once(self, cache):
        task = make_tasks(1)[0]
        judge = FakeJudge()
        stats = judge_pairs([task, task], judge, cache, log=silent)
        assert len(judge.calls) == 1 and stats.total == 1

    def test_a_failed_pair_is_left_for_the_next_run(self, cache):
        judge = FakeJudge(fail_on={1: ValueError("bad output")})
        stats = judge_pairs(make_tasks(3), judge, cache, log=silent)
        assert (stats.judged, stats.failed, stats.remaining) == (2, 1, 1)
        assert ("q0", "google_books", "B0") not in cache

    def test_out_of_range_grade_counts_as_failure(self, cache):
        class BadJudge(FakeJudge):
            def judge(self, query_text, book):
                return 5, "too high"

        stats = judge_pairs(make_tasks(1), BadJudge(), cache, log=silent)
        assert (stats.judged, stats.failed) == (0, 1) and len(cache) == 0

    def test_stops_after_consecutive_failures(self, cache):
        fail_on = {n: ConnectionError("down") for n in range(1, MAX_CONSECUTIVE_FAILURES + 1)}
        with pytest.raises(RuntimeError, match="in a row"):
            judge_pairs(make_tasks(MAX_CONSECUTIVE_FAILURES + 2), FakeJudge(fail_on), cache, log=silent)

    def test_measures_seconds_per_judgment(self, cache):
        ticks = iter([0.0, 2.0, 10.0, 14.0])
        stats = judge_pairs(make_tasks(2), FakeJudge(), cache, log=silent, clock=lambda: next(ticks))
        assert stats.seconds_per_judgment == pytest.approx(3.0)


# =============================================================================
# Holes and merging
# =============================================================================


class TestFindUnjudged:
    RUNS = {
        "a": {"q1": ["x", "y", "z"], "q2": ["u"]},
        "b": {"q1": ["z", "w"], "q2": ["v", "u"]},
    }
    JUDGMENTS = {"q1": {"y": 1}}

    def test_union_of_top_k_without_judgment_best_rank_first(self):
        assert find_unjudged(self.RUNS, self.JUDGMENTS, k=2) == [
            ("q1", "x"), ("q1", "z"), ("q2", "u"), ("q2", "v"), ("q1", "w"),
        ]

    def test_only_the_top_k_counts(self):
        assert find_unjudged(self.RUNS, self.JUDGMENTS, k=1) == [("q1", "x"), ("q1", "z"), ("q2", "u"), ("q2", "v")]

    def test_nothing_to_judge_when_everything_is_judged(self):
        judgments = {"q1": {"x": 0, "y": 1, "z": 2, "w": 3}, "q2": {"u": 0, "v": 0}}
        assert find_unjudged(self.RUNS, judgments, k=10) == []


def llm_judgment(query_id: str, source_id: str, grade: int) -> LLMJudgment:
    return LLMJudgment(query_id=query_id, source="google_books", source_id=source_id, grade=grade,
                       rationale=f"Because {source_id}.", model=MODEL, prompt_version=PROMPT,
                       date="2026-11-11T12:00:00+00:00")


class TestMergeJudgments:
    ORIGINAL = {
        "q1": [{"source": "google_books", "source_id": "A", "relevance": 2}],
        "q2": [{"source": "google_books", "source_id": "D", "relevance": 0}],
    }

    def merged(self):
        llm = [llm_judgment("q1", "C", 1), llm_judgment("q1", "A", 0), llm_judgment("q1", "B", 3),
               llm_judgment("q0", "E", 2)]
        return merge_judgments(self.ORIGINAL, llm)

    def test_original_wins_and_llm_fills_the_rest(self):
        merged = self.merged()
        assert [entry["query_id"] for entry in merged] == ["q1", "q2", "q0"]
        q1 = merged[0]["judgments"]
        assert [(j["source_id"], j["relevance"], j["judge"]) for j in q1] == [
            ("A", 2, "original"), ("B", 3, "llm"), ("C", 1, "llm"),
        ]
        assert merged[1]["judgments"] == [
            {"source": "google_books", "source_id": "D", "relevance": 0, "judge": "original"}
        ]

    def test_llm_judgments_carry_provenance(self):
        llm_item = self.merged()[0]["judgments"][1]
        assert llm_item == {
            "source": "google_books", "source_id": "B", "relevance": 3, "judge": "llm",
            "model": MODEL, "prompt_version": PROMPT, "date": "2026-11-11T12:00:00+00:00",
            "rationale": "Because B.",
        }

    def test_original_input_is_not_modified(self):
        self.merged()
        assert "judge" not in self.ORIGINAL["q1"][0]

    def test_existing_loader_reads_the_merged_file(self, tmp_path):
        path = tmp_path / "extended.json"
        path.write_text(json.dumps(self.merged()), encoding="utf-8")
        loaded = load_relevance_judgments(str(path))
        assert [item["relevance"] for item in loaded["q1"]] == [2, 3, 1]
        queries = [TestQuery("q0", "zero"), TestQuery("q1", "one"), TestQuery("q2", "two")]
        stats = describe_judgments(loaded, queries, num_books=10)
        assert stats["num_judgments"] == 5
        assert "judge" in stats["extra_judgment_fields"]


# =============================================================================
# Agreement and sampling
# =============================================================================


ORIGINAL = [0, 0, 1, 1, 2, 3]
LLM = [0, 1, 1, 2, 2, 2]
LANGUAGES = ["en", "en", "en", "es", "es", "es"]


class TestAgreementReport:
    def test_hand_computed_example(self):
        overall = agreement_report(ORIGINAL, LLM, LANGUAGES, n_resamples=200)["overall"]
        assert overall["n"] == 6
        assert overall["exact_agreement"] == pytest.approx(0.5)
        assert overall["within_one_agreement"] == pytest.approx(1.0)
        assert overall["quadratic_weighted_kappa"] == pytest.approx(22 / 31)
        assert overall["binary"]["at_least_1"]["kappa"] == pytest.approx(4 / 7)
        assert overall["binary"]["at_least_1"]["agreement"] == pytest.approx(5 / 6)
        assert overall["binary"]["at_least_2"]["kappa"] == pytest.approx(2 / 3)
        assert overall["mean_signed_difference"] == pytest.approx(1 / 6)
        assert overall["confusion_matrix"] == [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
        assert overall["grade_distribution"] == {"original": [2, 2, 1, 1], "llm": [1, 2, 3, 0]}
        low, high = overall["quadratic_weighted_kappa_ci95"]
        assert low <= high

    def test_split_by_language(self):
        by_language = agreement_report(ORIGINAL, LLM, LANGUAGES, n_resamples=50)["by_language"]
        assert list(by_language) == ["en", "es"]
        assert by_language["en"]["exact_agreement"] == pytest.approx(2 / 3)
        assert by_language["en"]["mean_signed_difference"] == pytest.approx(1 / 3)
        assert by_language["es"]["quadratic_weighted_kappa"] == pytest.approx(0.0)
        assert by_language["es"]["mean_signed_difference"] == pytest.approx(0.0)

    def test_undefined_kappa_is_none(self):
        overall = agreement_report([1, 1], [1, 1], n_resamples=10)["overall"]
        assert overall["quadratic_weighted_kappa"] is None
        assert overall["exact_agreement"] == 1.0

    def test_rejects_misaligned_or_invalid_grades(self):
        with pytest.raises(ValueError):
            agreement_report([0, 1], [0])
        with pytest.raises(ValueError):
            agreement_report([0, 4], [0, 1])


class TestStratifiedSample:
    # 100 items: (grade, language) strata of sizes 40, 30, 20, 9 and 1.
    POPULATION = (
        [(0, "en")] * 40 + [(1, "en")] * 30 + [(1, "es")] * 20 + [(2, "es")] * 9 + [(3, "es")] * 1
    )
    ITEMS = [(index, grade, language) for index, (grade, language) in enumerate(POPULATION)]

    @staticmethod
    def stratum(item):
        return item[1], item[2]

    def sample(self, n, seed=0):
        return stratified_sample(self.ITEMS, n, seed=seed, stratum=self.stratum)

    def test_same_seed_same_sample(self):
        assert self.sample(20, seed=0) == self.sample(20, seed=0)
        assert self.sample(20, seed=0) != self.sample(20, seed=1)

    def test_proportional_allocation(self):
        counts = {}
        for item in self.sample(20):
            counts[self.stratum(item)] = counts.get(self.stratum(item), 0) + 1
        assert counts == {(0, "en"): 8, (1, "en"): 6, (1, "es"): 4, (2, "es"): 1, (3, "es"): 1}

    def test_every_stratum_and_both_languages_represented(self):
        sample = self.sample(10)
        assert len(sample) == 10
        assert {self.stratum(item) for item in sample} == set(map(self.stratum, self.ITEMS))
        assert {item[2] for item in sample} == {"en", "es"}

    def test_population_order_and_no_duplicates(self):
        sample = self.sample(30)
        assert sample == sorted(sample) and len(set(sample)) == 30

    def test_whole_population_when_n_is_large(self):
        assert self.sample(500) == self.ITEMS


# =============================================================================
# Report documents
# =============================================================================


JUDGE_INFO = {
    "provider": "ollama", "model": MODEL, "base_url": None, "prompt_version": PROMPT,
    "prompt": "build_relevance_judge_prompt in app/infrastructure/llm/prompts_judge.py",
    "temperature": 0.0, "reasoning": False, "max_tokens": 256,
    "system_llm": f"ollama/{MODEL}", "same_as_system_llm": True,
}
RUN_INFO = {"date": "2026-11-11T12:00:00+00:00", "git_commit": "abc1234", "git_worktree_dirty": False,
            "command": "python -m app.evaluation.relevance_judge agreement --sample 6 --seed 0"}


def test_agreement_markdown_lists_tables_and_large_disagreements():
    pairs = [
        {"query_id": f"q{i}", "query_text": "text", "language": language, "source": "google_books",
         "source_id": f"B{i}", "title": "Title | with bar", "has_description": i % 2 == 0,
         "original": original, "llm": llm, "rationale": "Reason."}
        for i, (original, llm, language) in enumerate(zip(ORIGINAL, LLM, LANGUAGES))
    ]
    pairs[0]["llm"] = 3  # one disagreement of 3 grades
    document = {
        "run": RUN_INFO, "judge": JUDGE_INFO,
        "timing": {"judged_in_this_run": 6, "seconds_per_judgment": 4.2},
        "sample": {"n": 6, "population": 6, "seed": 0, "strata": {"grade 0/en": 2}},
        "agreement": agreement_report(ORIGINAL, LLM, LANGUAGES, n_resamples=20),
        "pairs": pairs,
    }
    text = build_agreement_markdown(document)
    assert "same model as the system LLM" in text and "4.20 s per judgment" in text
    assert "| all | 6 | 0.500 | 1.000 | 0.710" in text
    assert "| en queries | 3 |" in text and "| es queries | 3 |" in text
    assert "| 0 | 1 | 1 | 0 | 0 | 2 |" in text  # first confusion row
    assert "1 pairs; 1 shown" in text and "Title / with bar" in text


def test_fill_markdown_shows_both_comparisons_and_judged_at_10():
    queries = [TestQuery("q1", "first", language="en"), TestQuery("q2", "segunda", language="es")]
    runs = {"hybrid_rrf": {"q1": ["a", "x"], "q2": ["d"]}, "lexical_only": {"q1": ["x"], "q2": ["e", "d"]}}
    original = {"q1": {"a": 3}, "q2": {"d": 2}}
    extended = {"q1": {"a": 3, "x": 1}, "q2": {"d": 2, "e": 0}}
    before = compare_systems(runs, original, queries, "hybrid_rrf", n_resamples=20)
    after = compare_systems(runs, extended, queries, "hybrid_rrf", n_resamples=20)
    document = {
        "run": RUN_INFO, "judge": JUDGE_INFO,
        "agreement": {"n": 200, "quadratic_weighted_kappa": 0.7, "exact_agreement": 0.6},
        "fill": {"k": 10, "unjudged_pairs": 2, "llm_grade_distribution": [1, 1, 0, 0],
                 "output_judgments": "extended.json", "num_original": 2, "num_llm": 2},
    }
    text = build_fill_markdown(document, before, after)
    assert "quadratic-weighted kappa 0.700" in text and "grades 0 / 1 / 2 / 3: 1 / 1 / 0 / 0" in text
    # Judged@10 before / after: hybrid_rrf (0.1 + 0.1) / 2 -> (0.2 + 0.1) / 2;
    # lexical_only (0 + 0.1) / 2 -> (0.1 + 0.2) / 2
    assert "| hybrid_rrf | 0.100 | 0.150 |" in text and "| lexical_only | 0.050 | 0.150 |" in text
    assert "## On the original judgments" in text and "## On the extended judgments" in text
    assert text.count("| System | nDCG@10 [95% CI]") == 2
