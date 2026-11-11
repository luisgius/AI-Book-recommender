"""
Tests for the judge sensitivity of the V06 ablation (app/evaluation/judge_sensitivity.py).

No network and no search: judgments files and ablation reports are built by hand.
"""

import pytest

from app.evaluation.judge_sensitivity import (
    build_document,
    build_markdown,
    compare_contrasts,
    conclusion_text,
    judge_slug,
    llm_grades,
    new_judgments_path,
)


def judgments(llm_grades_by_pair, model):
    """A judgments file: one original judgment per query plus the LLM grades."""
    raw = {}
    for (query_id, source_id), grade in llm_grades_by_pair.items():
        raw.setdefault(query_id, [{"source": "s", "source_id": "orig", "relevance": 2, "judge": "original"}])
        raw[query_id].append({"source": "s", "source_id": source_id, "relevance": grade, "judge": "llm",
                              "model": model, "prompt_version": "v1.0"})
    return raw


def contrast(diff, p, p_holm, system="sys", reference="ref"):
    return {"system": system, "reference": reference, "metric": "ndcg_at_10", "mean_difference": diff,
            "ci_low": diff - 0.05, "ci_high": diff + 0.05, "p_value": p, "p_value_holm": p_holm}


def ablation_report(contrasts, ndcg, runs=None):
    return {
        "design": {"contrasts": [{"name": name, "question": f"Question {name}?"} for name in contrasts]},
        "contrasts": {"final": contrasts},
        "systems": {name: {"metrics": {"ndcg_at_10": {"mean": value}}} for name, value in ndcg.items()},
        "runs": runs or {"sys": {"q1": ["b1"]}},
    }


class TestConclusions:
    def test_same_direction_and_holm_significance_holds(self):
        old = ablation_report({"C1": contrast(0.10, 0.001, 0.008)}, {"sys": 0.6, "ref": 0.5})
        new = ablation_report({"C1": contrast(0.07, 0.004, 0.030)}, {"sys": 0.6, "ref": 0.5})
        [row] = compare_contrasts(old, new)
        assert row["holds"]
        assert conclusion_text(row).startswith("C1 holds")

    def test_losing_significance_after_holm_does_not_hold(self):
        old = ablation_report({"C5": contrast(0.10, 0.001, 0.008)}, {})
        new = ablation_report({"C5": contrast(0.05, 0.03, 0.12)}, {})
        [row] = compare_contrasts(old, new)
        assert not row["holds"]
        assert "does not hold" in conclusion_text(row)

    def test_a_flipped_direction_does_not_hold_even_if_both_are_not_significant(self):
        old = ablation_report({"C3": contrast(0.06, 0.08, 0.16)}, {})
        new = ablation_report({"C3": contrast(-0.01, 0.80, 1.0)}, {})
        [row] = compare_contrasts(old, new)
        assert not row["holds"]

    def test_unadjusted_significance_alone_does_not_matter(self):
        old = ablation_report({"C1": contrast(0.05, 0.02, 0.15)}, {})
        new = ablation_report({"C1": contrast(0.04, 0.09, 0.30)}, {})
        [row] = compare_contrasts(old, new)
        assert row["holds"]  # same direction, not significant after Holm under both
        assert row["old"]["significant_unadjusted"] and not row["new"]["significant_unadjusted"]

    def test_contrasts_must_compare_the_same_systems(self):
        old = ablation_report({"C1": contrast(0.1, 0.01, 0.01, system="a")}, {})
        new = ablation_report({"C1": contrast(0.1, 0.01, 0.01, system="b")}, {})
        with pytest.raises(ValueError):
            compare_contrasts(old, new)


class TestDocument:
    def test_agreement_is_on_the_llm_pairs_only_and_rows_are_the_old_judge(self):
        old_raw = judgments({("q1", "b1"): 0, ("q1", "b2"): 1, ("q2", "b3"): 3, ("q2", "b4"): 2}, "old-model")
        new_raw = judgments({("q1", "b1"): 0, ("q1", "b2"): 2, ("q2", "b3"): 3, ("q2", "b4"): 2}, "z-ai/new")
        report = ablation_report({"C1": contrast(0.1, 0.001, 0.007)}, {"sys": 0.6, "ref": 0.5})
        document = build_document(
            old_raw, new_raw, report, report, {"q1": "en", "q2": "es"}, new_model="z-ai/new",
            paths={"old_judgments": "old.json", "new_judgments": "new.json",
                   "old_ablation": "old_ab.json", "new_ablation": "new_ab.json"},
            n_resamples=50,
        )
        overall = document["agreement"]["overall"]
        assert overall["n"] == 4  # the original judgments are not compared
        assert overall["exact_agreement"] == pytest.approx(0.75)
        assert overall["mean_signed_difference"] == pytest.approx(0.25)  # new - old
        assert overall["confusion_matrix"][1][2] == 1  # old 1 -> new 2
        assert set(document["agreement"]["by_language"]) == {"en", "es"}
        assert document["num_original"] == 2 and document["same_runs"]
        assert document["old_judge"]["models"] == ["old-model"]

        text = build_markdown(document)
        assert "Agreement between the two judges on the 4 pairs" in text
        assert "1 of 1 conclusions hold under both judges" in text
        assert "`z-ai/new`" in text

    def test_files_grading_different_pairs_are_rejected(self):
        old_raw = judgments({("q1", "b1"): 0}, "old")
        new_raw = judgments({("q1", "b2"): 0}, "new")
        report = ablation_report({}, {})
        with pytest.raises(ValueError):
            build_document(old_raw, new_raw, report, report, {}, new_model="new",
                           paths={"old_judgments": "", "new_judgments": "", "old_ablation": "", "new_ablation": ""})

    def test_different_runs_are_flagged(self):
        raw = judgments({("q1", "b1"): 0}, "m")
        old = ablation_report({}, {}, runs={"sys": {"q1": ["b1"]}})
        new = ablation_report({}, {}, runs={"sys": {"q1": ["b2"]}})
        document = build_document(raw, raw, old, new, {}, new_model="m", n_resamples=10,
                                  paths={"old_judgments": "", "new_judgments": "", "old_ablation": "",
                                         "new_ablation": ""})
        assert not document["same_runs"]
        assert "did not return the same top 10" in build_markdown(document)


def test_llm_grades_and_paths():
    raw = judgments({("q1", "b1"): 3}, "m")
    assert llm_grades(raw) == {("q1", "s", "b1"): 3}
    assert judge_slug("z-ai/glm-5.3") == "z-ai-glm-5.3"
    assert new_judgments_path("z-ai/glm-5.3") == "app/evaluation/relevance_judgments_extended_z-ai-glm-5.3.json"


def test_parallel_judging_covers_every_pending_pair_once(tmp_path):
    from uuid import uuid4

    from app.domain.entities import Book
    from app.evaluation.judge_sensitivity import judge_in_parallel
    from app.evaluation.relevance_judge import JudgeTask, JudgmentCache

    class FakeJudge:
        model_name, prompt_version = "fake/judge", "v1.0"

        def __init__(self):
            self.seen = []

        def judge(self, query_text, book):
            self.seen.append(book.source_id)
            return 1, "ok"

    def task(i):
        book = Book(id=uuid4(), title=f"Book {i}", authors=["A"], description=None, language="en", categories=[],
                    published_date=None, source="s", source_id=f"b{i}", metadata={})
        return JudgeTask(f"q{i % 3}", "query", book)

    tasks = [task(i) for i in range(10)]
    cache = JudgmentCache.for_judge(tmp_path, "fake/judge", "v1.0")
    first = FakeJudge()
    judge_in_parallel(tasks[:2], [first], cache)  # two pairs already cached
    judges = [FakeJudge() for _ in range(3)]

    stats = judge_in_parallel(tasks, judges, cache)

    assert (stats.total, stats.cached, stats.judged, stats.failed, stats.remaining) == (10, 2, 8, 0, 0)
    seen = [source_id for judge in judges for source_id in judge.seen]
    assert sorted(seen) == sorted(f"b{i}" for i in range(2, 10))  # each pending pair judged exactly once
    assert all(judge.seen for judge in judges)  # the work was shared
    assert len(JudgmentCache.for_judge(tmp_path, "fake/judge", "v1.0")) == 10  # all on disk


def test_unadjusted_crossings_are_reported_but_do_not_change_the_conclusion():
    raw = judgments({("q1", "b1"): 0}, "m")
    old = ablation_report({"C4a": contrast(0.062, 0.045, 0.159)}, {"sys": 0.5, "ref": 0.4})
    new = ablation_report({"C4a": contrast(0.059, 0.070, 0.280)}, {"sys": 0.5, "ref": 0.4})
    document = build_document(raw, raw, old, new, {}, new_model="m", n_resamples=10,
                              paths={"old_judgments": "", "new_judgments": "", "old_ablation": "", "new_ablation": ""})
    assert document["contrasts"][0]["holds"]
    assert "unadjusted p crosses 0.05 between the judges for C4a (0.045 to 0.070)" in build_markdown(document)
