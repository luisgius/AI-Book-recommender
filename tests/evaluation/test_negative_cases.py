"""
Tests for negative/adversarial test suite.

These tests verify:
1. negative_tests.json is well-structured and complete
2. NegativeTestCase dataclass validates input correctly
3. NegativeTestResult captures outcomes properly
4. _evaluate_negative_case function handles pass/fail with mocked search
5. load_negative_tests reads and parses the JSON correctly
"""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.evaluation.types import (
    NegativeTestCase,
    NegativeTestResult,
    NEGATIVE_TEST_CATEGORIES,
)
from app.evaluation.evaluation_job import (
    load_negative_tests,
    _evaluate_negative_case,
    NEGATIVE_TIMEOUT_MS,
)


# =============================================================================
# Load test data at module level for parametrize
# =============================================================================

_NEGATIVE_TESTS_PATH = (
    Path(__file__).parent.parent.parent / "app" / "evaluation" / "negative_tests.json"
)

with open(_NEGATIVE_TESTS_PATH, encoding="utf-8") as _f:
    _RAW_CASES = json.load(_f)


# =============================================================================
# Tests: JSON file structure
# =============================================================================


class TestNegativeTestsJsonStructure:
    """Validate the negative_tests.json file is well-formed and complete."""

    def test_file_is_non_empty(self):
        """JSON file should contain at least one test case."""
        assert len(_RAW_CASES) > 0

    def test_all_required_fields_present(self):
        """Every test case must have query_id, text, category, expected_behavior."""
        required = {"query_id", "text", "category", "expected_behavior"}
        for case in _RAW_CASES:
            missing = required - case.keys()
            assert not missing, (
                f"Missing fields {missing} in {case.get('query_id', '?')}"
            )

    def test_all_categories_are_valid(self):
        """Every category must be from the allowed set."""
        for case in _RAW_CASES:
            assert case["category"] in NEGATIVE_TEST_CATEGORIES, (
                f"Invalid category '{case['category']}' in {case['query_id']}"
            )

    def test_query_ids_are_unique(self):
        """No duplicate query IDs."""
        ids = [c["query_id"] for c in _RAW_CASES]
        assert len(ids) == len(set(ids)), "Duplicate query IDs found"

    def test_all_categories_have_at_least_2_cases(self):
        """Every defined category should have sufficient test coverage."""
        counts = Counter(c["category"] for c in _RAW_CASES)
        for cat in NEGATIVE_TEST_CATEGORIES:
            assert counts.get(cat, 0) >= 2, (
                f"Category '{cat}' has fewer than 2 test cases ({counts.get(cat, 0)})"
            )

    def test_texts_are_non_empty(self):
        """Every test case must have non-empty query text."""
        for case in _RAW_CASES:
            assert case["text"].strip(), f"Empty text in {case['query_id']}"

    def test_query_ids_follow_naming_convention(self):
        """Query IDs should follow neg_{category_prefix}_{nn} pattern."""
        for case in _RAW_CASES:
            qid = case["query_id"]
            assert qid.startswith("neg_"), (
                f"Query ID '{qid}' should start with 'neg_'"
            )


# =============================================================================
# Tests: NegativeTestCase dataclass
# =============================================================================


class TestNegativeTestCase:
    """Tests for NegativeTestCase frozen dataclass."""

    def test_create_from_json_dict(self):
        """Should create NegativeTestCase from a raw JSON dict."""
        case = NegativeTestCase(**_RAW_CASES[0])
        assert case.query_id == _RAW_CASES[0]["query_id"]
        assert case.text == _RAW_CASES[0]["text"]
        assert case.category == _RAW_CASES[0]["category"]

    def test_is_immutable(self):
        """NegativeTestCase should be frozen."""
        case = NegativeTestCase(
            query_id="test", text="test query",
            category="gibberish", expected_behavior="no crash",
        )
        with pytest.raises(Exception):
            case.text = "changed"

    def test_rejects_invalid_category(self):
        """Should raise ValueError for unknown category."""
        with pytest.raises(ValueError, match="Invalid category"):
            NegativeTestCase(
                query_id="bad", text="test",
                category="nonexistent_category",
                expected_behavior="test",
            )

    def test_rejects_empty_text(self):
        """Should raise ValueError for empty or whitespace-only text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            NegativeTestCase(
                query_id="bad", text="   ",
                category="gibberish", expected_behavior="test",
            )

    def test_all_json_cases_load_successfully(self):
        """Every case in the JSON file should be loadable into the dataclass."""
        for raw in _RAW_CASES:
            case = NegativeTestCase(**raw)
            assert case.query_id


# =============================================================================
# Tests: NegativeTestResult dataclass
# =============================================================================


class TestNegativeTestResult:
    """Tests for NegativeTestResult dataclass."""

    def test_create_passing_result(self):
        """Should create a result that passed."""
        result = NegativeTestResult(
            query_id="neg_gb_01", category="gibberish",
            query_text="asdf jkl", crashed=False, error_message=None,
            num_results=0, latency_ms=50.0,
            passed=True, failure_reason=None,
        )
        assert result.passed is True
        assert result.crashed is False
        assert result.num_results == 0

    def test_create_crashed_result(self):
        """Should create a result that crashed."""
        result = NegativeTestResult(
            query_id="neg_gb_02", category="gibberish",
            query_text="!!!", crashed=True,
            error_message="BM25 index error",
            num_results=0, latency_ms=5.0,
            passed=False, failure_reason="Crashed: BM25 index error",
        )
        assert result.passed is False
        assert result.crashed is True
        assert "BM25" in result.error_message

    def test_serializable_to_dict(self):
        """Should be convertible to dict via asdict()."""
        result = NegativeTestResult(
            query_id="test", category="gibberish",
            query_text="test", crashed=False, error_message=None,
            num_results=3, latency_ms=100.0,
            passed=True, failure_reason=None,
        )
        d = asdict(result)
        assert isinstance(d, dict)
        assert d["query_id"] == "test"
        assert d["num_results"] == 3

    def test_serializable_to_json(self):
        """Should be JSON-serializable after asdict()."""
        result = NegativeTestResult(
            query_id="test", category="ambiguous",
            query_text="something", crashed=False, error_message=None,
            num_results=5, latency_ms=200.0,
            passed=True, failure_reason=None,
        )
        json_str = json.dumps(asdict(result))
        parsed = json.loads(json_str)
        assert parsed["category"] == "ambiguous"


# =============================================================================
# Tests: load_negative_tests function
# =============================================================================


class TestLoadNegativeTests:
    """Tests for the JSON loading function."""

    def test_loads_from_default_path(self):
        """Should load all test cases from the default JSON file."""
        cases = load_negative_tests(str(_NEGATIVE_TESTS_PATH))
        assert len(cases) == len(_RAW_CASES)
        assert all(isinstance(c, NegativeTestCase) for c in cases)

    def test_raises_on_missing_file(self):
        """Should raise FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_negative_tests("/nonexistent/path/tests.json")

    def test_loaded_cases_match_raw_data(self):
        """Loaded dataclasses should contain same data as raw JSON."""
        cases = load_negative_tests(str(_NEGATIVE_TESTS_PATH))
        for case, raw in zip(cases, _RAW_CASES):
            assert case.query_id == raw["query_id"]
            assert case.text == raw["text"]
            assert case.category == raw["category"]


# =============================================================================
# Tests: _evaluate_negative_case with mocked SearchService
# =============================================================================


class TestEvaluateNegativeCaseWithMock:
    """Test the evaluation function with a mocked search service."""

    def _make_mock_service(self, num_results=3, side_effect=None):
        """Create a mock SearchService returning a configurable response."""
        mock_service = MagicMock()
        if side_effect:
            mock_service.search_with_fallback.side_effect = side_effect
        else:
            mock_response = MagicMock()
            mock_response.results = [MagicMock() for _ in range(num_results)]
            mock_service.search_with_fallback.return_value = mock_response
        return mock_service

    def _make_case(self, category="gibberish", text="test query"):
        return NegativeTestCase(
            query_id="test_case", text=text,
            category=category, expected_behavior="no crash",
        )

    def test_successful_search_passes(self):
        """A search that returns results without crashing should pass."""
        service = self._make_mock_service(num_results=5)
        result = _evaluate_negative_case(self._make_case(), service)

        assert result.passed is True
        assert result.crashed is False
        assert result.num_results == 5
        assert result.failure_reason is None

    def test_zero_results_still_passes(self):
        """Zero results is valid for negative tests (not a crash)."""
        service = self._make_mock_service(num_results=0)
        result = _evaluate_negative_case(self._make_case(), service)

        assert result.passed is True
        assert result.num_results == 0

    def test_exception_marks_as_crashed(self):
        """An unhandled exception should mark the result as crashed/failed."""
        service = self._make_mock_service(side_effect=RuntimeError("boom"))
        result = _evaluate_negative_case(self._make_case(), service)

        assert result.passed is False
        assert result.crashed is True
        assert "boom" in result.error_message
        assert result.failure_reason.startswith("Crashed:")

    def test_records_correct_metadata(self):
        """Result should capture query_id, category, and query_text."""
        case = NegativeTestCase(
            query_id="neg_am_01", text="something good",
            category="ambiguous", expected_behavior="no crash",
        )
        service = self._make_mock_service(num_results=2)
        result = _evaluate_negative_case(case, service)

        assert result.query_id == "neg_am_01"
        assert result.category == "ambiguous"
        assert result.query_text == "something good"

    def test_latency_is_recorded(self):
        """Latency should be a positive number."""
        service = self._make_mock_service(num_results=1)
        result = _evaluate_negative_case(self._make_case(), service)

        assert result.latency_ms >= 0

    def test_respects_max_results_parameter(self):
        """Should pass max_results to the SearchQuery."""
        service = self._make_mock_service(num_results=0)
        _evaluate_negative_case(self._make_case(), service, max_results=5)

        call_args = service.search_with_fallback.call_args
        query = call_args[0][0]
        assert query.max_results == 5

    @pytest.mark.parametrize("category", sorted(NEGATIVE_TEST_CATEGORIES))
    def test_all_categories_handled(self, category):
        """Every category should be processable without error."""
        case = NegativeTestCase(
            query_id=f"test_{category}", text="test input",
            category=category, expected_behavior="no crash",
        )
        service = self._make_mock_service(num_results=1)
        result = _evaluate_negative_case(case, service)
        assert result.passed is True
