"""
Tests for the LLM model selection (app/evaluation/llm_model_selection.py).

No network: the metrics, the selection rule and the report are pure
functions over saved raw records, which the tests build by hand.
"""

import json
from uuid import uuid4

import pytest

from app.domain.entities import Book
from app.evaluation import llm_model_selection as selection
from app.evaluation.llm_model_selection import (
    CountingJudge,
    ExplanationTask,
    check_candidates,
    explanation_language,
    explanation_queries,
    explanation_record,
    pairwise_kappa,
    select_judge,
    select_system_llm,
    summarize_explanations,
    summarize_queries,
    unrequested_filters,
    vendor,
)
from app.evaluation.types import TestQuery as Query  # alias: pytest would try to collect Test*
from app.infrastructure.llm.schemas import CitationLLM, GroundedExplanationLLM


def make_book() -> Book:
    return Book(
        id=uuid4(),
        title="Cien años de soledad",
        authors=["Gabriel García Márquez"],
        description="La historia de la familia Buendía en el pueblo de Macondo.",
        language="es",
        categories=["Fiction", "Magical realism"],
        published_date="1967",
        source="test",
        source_id="b1",
        metadata={},
    )


# =============================================================================
# Language of an explanation
# =============================================================================


class TestExplanationLanguage:
    def test_spanish(self):
        assert explanation_language("Este libro es una novela sobre la familia y el pueblo de Macondo.") == "es"

    def test_english(self):
        assert explanation_language("This book is a novel about the family and the town of Macondo.") == "en"

    def test_accents_are_folded(self):
        # "según" and "había" are Spanish stopwords once the accents are folded
        assert explanation_language("Según la descripción, había magia") == "es"

    def test_words_in_both_lists_do_not_vote(self):
        # "a", "me", "he" and "has" are stopwords in both languages
        assert explanation_language("a me he has Macondo") is None

    def test_tie_or_no_stopword_gives_none(self):
        assert explanation_language("Macondo Buendía") is None
        assert explanation_language("the libro de") is None  # one English-only, one Spanish-only word

    def test_quoted_english_text_does_not_flip_a_spanish_explanation(self):
        text = ('El libro trata de la historia de una familia y es relevante para la consulta; '
                'la descripción dice "a story of the family"')
        assert explanation_language(text) == "es"


# =============================================================================
# Citation precision before the guardrail
# =============================================================================


def raw_explanation(book: Book, *citations: tuple) -> GroundedExplanationLLM:
    return GroundedExplanationLLM(
        summary="Una novela sobre la familia Buendía.",
        reasoning="La descripción habla de la familia Buendía en Macondo [CITE: description].",
        citations=[CitationLLM(book_id=str(book.id), chunk_id=chunk, snippet=snippet, relevance_score=0.8)
                   for chunk, snippet in citations],
        confidence=0.8,
    )


class FakeExplanation:
    def __init__(self, text: str, citations: int) -> None:
        self.text = text
        self.citations = [object()] * citations


class TestExplanationRecord:
    def test_each_citation_is_checked_like_the_guardrail(self):
        book = make_book()
        task = ExplanationTask("q02", "es", "novela familia", 1, book)
        raw = raw_explanation(
            book,
            ("description", "familia Buendía"),        # valid
            ("title", "Cien Años de Soledad"),          # valid (case and spaces are normalized)
            ("title", "familia Buendía"),               # in the description, not in the title
            ("description", "a novel about Macondo"),   # not in the book at all
        )
        record = explanation_record(task, raw, FakeExplanation("text", 2), [], 1.0)

        assert [c["valid"] for c in record["citations"]] == [True, True, False, False]
        assert [c["error"] for c in record["citations"]] == [None, None, "wrong field", "not in the book"]
        assert record["parsed"] and record["final_citations"] == 2

    def test_unparsed_call_has_no_citations(self):
        book = make_book()
        record = explanation_record(ExplanationTask("q02", "es", "x", 1, book), None,
                                    FakeExplanation("Unable to generate explanation.", 0), [], 1.0)
        assert not record["parsed"] and record["citations"] == []
        assert record["failure"] == "output did not fit the schema"


def b_record(language: str, valid_flags, *, parsed: bool = True, text: str = None, cost: float = 0.001,
             latency: float = 2.0) -> dict:
    text = text if text is not None else (
        "Este libro es una novela sobre la familia." if language == "es" else "This book is a novel about the family."
    )
    return {
        "language": language,
        "parsed": parsed,
        "failure": None if parsed else "output did not fit the schema",
        "summary": text if parsed else "",
        "reasoning": "",
        "citations": [{"valid": flag, "error": None if flag else "not in the book"} for flag in valid_flags]
        if parsed else [],
        "latency_s": latency,
        "cost_usd": cost,
    }


class TestSummarizeExplanations:
    def test_precision_is_over_all_citations_written(self):
        records = [
            b_record("es", [True, True, False]),
            b_record("en", [False]),                 # zero valid citations
            b_record("en", [True]),
            b_record("es", [], parsed=False),         # does not count for precision
        ]
        row = summarize_explanations(records)

        assert (row["parsed"], row["n"], row["parsed_share"]) == (3, 4, 0.75)
        assert (row["citations"], row["valid_citations"]) == (5, 3)
        assert row["citation_precision"] == pytest.approx(3 / 5)
        assert row["zero_valid_share"] == pytest.approx(1 / 3)
        assert row["cost_per_explanation_usd"] == pytest.approx(0.001)

    def test_query_language_share(self):
        records = [
            b_record("es", [True]),
            b_record("en", [True], text="Este libro es una novela sobre la familia."),  # wrong language
            b_record("en", [True]),
            b_record("es", [True], text="Macondo"),  # undetermined: not the query language
        ]
        assert summarize_explanations(records)["in_query_language_share"] == pytest.approx(2 / 4)


# =============================================================================
# Unrequested filters
# =============================================================================


class TestUnrequestedFilters:
    def test_language_the_query_does_not_ask_for(self):
        assert unrequested_filters("cocina española", {"language": "es"})["language"] is True
        assert unrequested_filters("novelas en español", {"language": "es"})["language"] is False
        assert unrequested_filters("books in English", {"language": " EN "})["language"] is False

    def test_a_language_name_instead_of_a_code_is_unrequested(self):
        assert unrequested_filters("spanish novels", {"language": "Spanish"})["language"] is True

    def test_year_without_digits_in_the_query(self):
        assert unrequested_filters("literatura del siglo XIX", {"min_year": 1800, "max_year": 1899})["year"] is True
        assert unrequested_filters("novels from the 90s", {"min_year": 1990})["year"] is False

    def test_no_filters(self):
        assert unrequested_filters("anything", {"language": None, "min_year": None}) == {"language": False,
                                                                                        "year": False}
        assert unrequested_filters("anything", None) == {"language": False, "year": False}


def c_record(text: str, filters, parsed: bool = True, latency: float = 3.0, cost: float = 0.0005) -> dict:
    return {"text": text, "parsed": parsed, "raw_filters": filters, "latency_s": latency, "cost_usd": cost,
            "error": None if parsed else "Filter extraction failed: boom"}


class TestSummarizeQueries:
    def test_shares_and_denominators(self):
        records = [
            c_record("cocina española", {"language": "es", "min_year": None, "max_year": None}),
            c_record("historia siglo XIX", {"language": None, "min_year": 1800, "max_year": 1899}),
            c_record("libros en inglés de 1990", {"language": "en", "min_year": 1990, "max_year": None}),
            c_record("python", {"language": None, "min_year": None, "max_year": None}),
            c_record("broken", None, parsed=False),  # filter step failed: not in the shares
        ]
        row = summarize_queries(records)

        assert (row["parsed"], row["n"], row["filters_parsed"]) == (4, 5, 4)
        assert row["unrequested_language_share"] == pytest.approx(1 / 4)
        assert row["unrequested_year_share"] == pytest.approx(1 / 4)
        assert row["unrequested_any_share"] == pytest.approx(2 / 4)
        assert row["median_latency_s"] == 3.0
        assert row["cost_per_query_usd"] == pytest.approx(0.0005)


# =============================================================================
# Selection rule
# =============================================================================


def b_row(precision, language=1.0, parsed=60, per_explanation=0.001):
    return {"n": 60, "parsed": parsed, "parsed_share": parsed / 60, "citation_precision": precision,
            "in_query_language_share": language, "cost_per_explanation_usd": per_explanation}


def c_row(parsed=44, per_query=0.001):
    return {"n": 44, "parsed": parsed, "parsed_share": parsed / 44, "cost_per_query_usd": per_query}


class TestSelectSystemLLM:
    def test_highest_precision_wins_without_tie(self):
        decision = select_system_llm(
            {"a/x": b_row(0.95), "b/y": b_row(0.90)}, {"a/x": c_row(), "b/y": c_row()})
        assert decision["winner"] == "a/x"
        assert decision["tied"] == ["a/x"]
        assert decision["decided_by"] == "highest citation precision (no tie)"

    def test_tie_within_margin_is_broken_by_query_language(self):
        decision = select_system_llm(
            {"a/x": b_row(0.95, language=0.90), "b/y": b_row(0.94, language=1.00), "c/z": b_row(0.90)},
            {"a/x": c_row(), "b/y": c_row(), "c/z": c_row()},
        )
        assert decision["tied"] == ["b/y", "a/x"]
        assert decision["winner"] == "b/y"
        assert "query language" in decision["decided_by"]

    def test_exactly_the_margin_below_is_still_tied(self):
        decision = select_system_llm(
            {"a/x": b_row(0.95, language=0.9), "b/y": b_row(0.93, language=1.0)}, {"a/x": c_row(), "b/y": c_row()})
        assert set(decision["tied"]) == {"a/x", "b/y"}
        assert decision["winner"] == "b/y"

    def test_then_by_cost_per_search(self):
        decision = select_system_llm(
            {"a/x": b_row(0.95, per_explanation=0.002), "b/y": b_row(0.95, per_explanation=0.001)},
            {"a/x": c_row(per_query=0.001), "b/y": c_row(per_query=0.003)},
        )
        # cost per search: a/x 0.001 + 3 * 0.002 = 0.007; b/y 0.003 + 3 * 0.001 = 0.006
        assert decision["winner"] == "b/y"
        assert "cost" in decision["decided_by"]

    def test_a_model_with_unparsed_calls_is_excluded_even_if_best(self):
        decision = select_system_llm(
            {"a/x": b_row(0.99, parsed=59), "b/y": b_row(0.80)}, {"a/x": c_row(), "b/y": c_row(parsed=44)})
        assert decision["winner"] == "b/y"
        assert "a/x" in decision["excluded"]

    def test_unparsed_query_understanding_also_excludes(self):
        decision = select_system_llm({"a/x": b_row(0.99)}, {"a/x": c_row(parsed=43)})
        assert decision["winner"] is None and "a/x" in decision["excluded"]


class TestSelectJudge:
    def test_vendor_is_the_id_prefix(self):
        assert vendor("z-ai/glm-5.3") == "z-ai"

    def test_same_vendor_as_the_system_llm_is_excluded(self):
        rows = {"qwen/a": {"complete": True, "kappa": 0.90, "cost_per_judgment_usd": 0.0001},
                "z-ai/b": {"complete": True, "kappa": 0.70, "cost_per_judgment_usd": 0.0001}}
        decision = select_judge(rows, system_llm="qwen/other")
        assert decision["winner"] == "z-ai/b"
        assert "same vendor" in decision["excluded"]["qwen/a"]

    def test_tie_within_margin_goes_to_the_cheaper(self):
        rows = {"m/a": {"complete": True, "kappa": 0.75, "cost_per_judgment_usd": 0.0003},
                "d/b": {"complete": True, "kappa": 0.74, "cost_per_judgment_usd": 0.0001},
                "z/c": {"complete": True, "kappa": 0.70, "cost_per_judgment_usd": 0.00001}}
        decision = select_judge(rows, system_llm="qwen/x")
        assert decision["tied"] == ["d/b", "m/a"]
        assert decision["winner"] == "d/b"
        assert "cost" in decision["decided_by"]

    def test_incomplete_agreement_is_excluded(self):
        rows = {"m/a": {"complete": False, "failures": 5},
                "d/b": {"complete": True, "kappa": 0.60, "cost_per_judgment_usd": 0.0001}}
        decision = select_judge(rows, system_llm="qwen/x")
        assert decision["winner"] == "d/b" and "incomplete" in decision["excluded"]["m/a"]


# =============================================================================
# Candidates, queries, kappa matrix, counting judge
# =============================================================================


def test_candidates_without_structured_outputs_are_dropped():
    catalog = {
        "q/ok": {"pricing": {"prompt": "0.00000015", "completion": "0.00000047"},
                 "supported_parameters": ["response_format", "structured_outputs", "reasoning"]},
        "q/json-mode-only": {"pricing": {"prompt": "0.00000003", "completion": "0.00000013"},
                             "supported_parameters": ["response_format", "reasoning"]},
    }
    info = check_candidates({"q/ok": False, "q/json-mode-only": False, "q/gone": False}, catalog)
    assert info["q/ok"]["dropped"] is None
    assert info["q/ok"]["input_per_mtok"] == pytest.approx(0.15)
    assert "structured outputs" in info["q/json-mode-only"]["dropped"]
    assert info["q/gone"]["dropped"] == "not in the OpenRouter model list"


def test_explanation_queries_are_the_first_per_language_by_id():
    queries = [Query(query_id=f"q{i:02d}", text=f"query {i}", language="es" if i % 2 else "en")
               for i in range(1, 9)]
    chosen = explanation_queries(queries, per_language=2)
    assert [q.query_id for q in chosen] == ["q01", "q02", "q03", "q04"]


def test_pairwise_kappa_is_symmetric_with_ones_on_the_diagonal():
    grades = {"a": {"k1": 0, "k2": 1, "k3": 2, "k4": 3},
              "b": {"k1": 0, "k2": 1, "k3": 3, "k4": 3},
              "c": {"k1": 3, "k2": 2, "k3": 1, "k4": 0}}
    matrix = pairwise_kappa(grades)
    assert matrix["a"]["a"] == pytest.approx(1.0)
    assert matrix["a"]["b"] == pytest.approx(matrix["b"]["a"])
    assert matrix["a"]["c"] < 0 < matrix["a"]["b"]


def test_counting_judge_counts_failures_and_reraises():
    class Inner:
        model_name, prompt_version = "m", "v1.0"

        def __init__(self):
            self.answers = iter([(2, "ok"), ValueError("bad json")])

        def judge(self, query_text, book):
            answer = next(self.answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

    judge = CountingJudge(Inner())
    assert judge.judge("q", None) == (2, "ok")
    with pytest.raises(ValueError):
        judge.judge("q", None)
    assert (judge.calls, judge.failures, dict(judge.errors)) == (2, 1, {"ValueError": 1})


# =============================================================================
# Report rendered from saved results (no model call)
# =============================================================================


def agreement_doc(llm_grades, original_grades):
    pairs = [{"query_id": f"q{i:02d}", "source": "s", "source_id": f"b{i}", "llm": g, "original": o}
             for i, (g, o) in enumerate(zip(llm_grades, original_grades))]
    return {
        "agreement": {"overall": {"n": len(pairs), "quadratic_weighted_kappa": 0.8,
                                  "quadratic_weighted_kappa_ci95": [0.7, 0.9], "exact_agreement": 0.7,
                                  "mean_signed_difference": -0.1}},
        "timing": {"seconds_per_judgment": 1.2},
        "pairs": pairs,
    }


def saved_results() -> dict:
    original = [0, 1, 2, 3, 1, 0]
    b_records = [dict(b_record("es", [True, False]), query_id="q02", rank=1, title="T")] * 2
    c_records = [dict(c_record("cocina española", {"language": "es", "min_year": None, "max_year": None}),
                      query_id="q01", language="es", steps_parsed={}, final_filters={})]
    return {
        "run": {"date": "2026-11-11T15:00:00+00:00", "git_commit": "abc1234", "git_worktree_dirty": False,
                "key_usage_usd": {"before": 0.1, "after": 0.2}},
        "candidates": {
            "qwen/a": {"listed": True, "input_per_mtok": 0.15, "output_per_mtok": 0.47,
                       "structured_outputs": True, "reasoning": False, "dropped": None},
            "z-ai/b": {"listed": True, "input_per_mtok": 0.04, "output_per_mtok": 0.08,
                       "structured_outputs": True, "reasoning": "low", "dropped": None},
            "qwen/c": {"listed": True, "input_per_mtok": 0.03, "output_per_mtok": 0.13,
                       "structured_outputs": False, "reasoning": False, "dropped": "no structured outputs"},
        },
        "explanation_tasks": [{"query_id": "q02"}],
        "judge": {
            "qwen/a": {"agreement": agreement_doc([0, 1, 2, 3, 1, 1], original), "failures": 0,
                       "usage": {"reported_cost_usd": 0.01}},
            "z-ai/b": {"agreement": agreement_doc([0, 1, 1, 3, 1, 0], original), "failures": 1,
                       "errors": {"ValueError": 1}, "attempts": 2, "usage": {"reported_cost_usd": 0.004}},
        },
        "local_judge": {"model": "qwen3.8:27b-mlx",
                        "grades": {f"q{i:02d}|s|b{i}": g for i, g in enumerate([0, 1, 2, 2, 1, 0])}},
        "explanation": {
            "qwen/a": {"records": b_records, "usage": {"reported_cost_usd": 0.002}},
            "z-ai/b": {"records": [dict(r, summary="This book is about the family.") for r in b_records],
                       "usage": {"reported_cost_usd": 0.001}},
        },
        "query_understanding": {
            "qwen/a": {"records": c_records, "usage": {"reported_cost_usd": 0.0005}},
            "z-ai/b": {"records": c_records, "usage": {"reported_cost_usd": 0.0004}},
        },
    }


def test_report_is_rendered_from_saved_results_without_calling_a_model(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("the report must not call any model or API")

    monkeypatch.setattr(selection, "run_judge", no_network)
    monkeypatch.setattr(selection, "build_system_client", no_network)
    monkeypatch.setattr(selection, "fetch_model_catalog", no_network)
    monkeypatch.setattr(selection, "key_usage_usd", no_network)
    results_path, report_path = tmp_path / "results.json", tmp_path / "REPORT.md"
    results_path.write_text(json.dumps(saved_results()), encoding="utf-8")

    assert selection.main(["report", "--output-json", str(results_path), "--output-md", str(report_path)]) == 0

    text = report_path.read_text(encoding="utf-8")
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    decision = saved["tables"]["decision"]
    # Same precision (0.5); qwen/a writes in the query language (Spanish), z-ai/b does not.
    assert decision["system_llm"]["winner"] == "qwen/a"
    assert decision["judge"]["winner"] == "z-ai/b"  # the only other vendor
    assert "same vendor" in decision["judge"]["excluded"]["qwen/a"]
    for heading in ("## A. Judge role", "## B. Explanation role", "## C. Query-understanding role",
                    "## Decision", "## Spend", "## Reproduce"):
        assert heading in text
    assert "**`qwen/a`**" in text and "**`z-ai/b`**" in text
    assert "qwen3.8:27b-mlx (local)" in text  # pairwise matrix includes the local judge
    assert "no structured outputs" in text  # dropped candidate is listed
    assert "2 es queries answered in en" in text  # failure pattern of z-ai/b
    assert saved["spend"]["total_usd"] == pytest.approx(0.01 + 0.004 + 0.002 + 0.001 + 0.0005 + 0.0004)


def test_report_without_saved_results_fails_cleanly(tmp_path):
    assert selection.main(["report", "--output-json", str(tmp_path / "none.json"),
                           "--output-md", str(tmp_path / "r.md")]) == 1
