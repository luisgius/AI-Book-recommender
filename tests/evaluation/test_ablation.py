"""
Tests for ablation testing support.

These tests verify:
1. AblationConfig dataclass creation, validation, and immutability
2. get_standard_ablation_configs() returns well-formed configs
3. _run_ablation_search() routes to correct search function per config
4. _format_ablation_table() produces correct comparison output
"""

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from app.evaluation.types import (
    AblationConfig,
    get_standard_ablation_configs,
)
from app.evaluation.evaluation_job import (
    _run_ablation_search,
    _format_ablation_table,
)


# =============================================================================
# Tests: AblationConfig dataclass
# =============================================================================


class TestAblationConfig:
    """Tests for AblationConfig frozen dataclass."""

    def test_create_with_defaults(self):
        """Default config should have all components enabled."""
        config = AblationConfig(label="test")
        assert config.use_lexical is True
        assert config.use_vector is True
        assert config.use_mmr is True
        assert config.mmr_lambda == 0.6

    def test_create_with_overrides(self):
        """Should accept custom toggle values."""
        config = AblationConfig(
            label="custom", use_lexical=False,
            use_vector=True, use_mmr=False, mmr_lambda=0.3,
        )
        assert config.use_lexical is False
        assert config.use_mmr is False
        assert config.mmr_lambda == 0.3

    def test_is_immutable(self):
        """AblationConfig should be frozen."""
        config = AblationConfig(label="test")
        with pytest.raises(Exception):
            config.use_mmr = False

    def test_rejects_empty_label(self):
        """Should raise ValueError for empty label."""
        with pytest.raises(ValueError, match="label cannot be empty"):
            AblationConfig(label="")

    def test_rejects_whitespace_label(self):
        """Should raise ValueError for whitespace-only label."""
        with pytest.raises(ValueError, match="label cannot be empty"):
            AblationConfig(label="   ")

    def test_rejects_both_search_disabled(self):
        """Should raise if both lexical and vector are disabled."""
        with pytest.raises(ValueError, match="At least one"):
            AblationConfig(label="broken", use_lexical=False, use_vector=False)

    def test_rejects_lambda_below_zero(self):
        """Should reject mmr_lambda < 0."""
        with pytest.raises(ValueError, match="mmr_lambda"):
            AblationConfig(label="bad", mmr_lambda=-0.1)

    def test_rejects_lambda_above_one(self):
        """Should reject mmr_lambda > 1."""
        with pytest.raises(ValueError, match="mmr_lambda"):
            AblationConfig(label="bad", mmr_lambda=1.5)

    def test_boundary_lambda_zero(self):
        """Lambda=0 (pure diversity) should be valid."""
        config = AblationConfig(label="diverse", mmr_lambda=0.0)
        assert config.mmr_lambda == 0.0

    def test_boundary_lambda_one(self):
        """Lambda=1 (pure relevance) should be valid."""
        config = AblationConfig(label="relevant", mmr_lambda=1.0)
        assert config.mmr_lambda == 1.0

    def test_serializable_to_dict(self):
        """Should be convertible via asdict()."""
        config = AblationConfig(label="test", use_mmr=False)
        d = asdict(config)
        assert d["label"] == "test"
        assert d["use_mmr"] is False


# =============================================================================
# Tests: get_standard_ablation_configs
# =============================================================================


class TestGetStandardAblationConfigs:
    """Tests for the standard config generator."""

    def test_returns_non_empty_list(self):
        """Should return at least one configuration."""
        configs = get_standard_ablation_configs()
        assert len(configs) > 0

    def test_first_config_is_baseline(self):
        """First config should be the full pipeline baseline."""
        configs = get_standard_ablation_configs()
        baseline = configs[0]
        assert baseline.label == "full_pipeline"
        assert baseline.use_lexical is True
        assert baseline.use_vector is True
        assert baseline.use_mmr is True

    def test_labels_are_unique(self):
        """No duplicate labels."""
        configs = get_standard_ablation_configs()
        labels = [c.label for c in configs]
        assert len(labels) == len(set(labels))

    def test_all_configs_are_valid(self):
        """Every config should be a valid AblationConfig."""
        configs = get_standard_ablation_configs()
        for config in configs:
            assert isinstance(config, AblationConfig)
            assert config.label

    def test_has_no_mmr_variant(self):
        """Should include a config with MMR disabled."""
        configs = get_standard_ablation_configs()
        labels = {c.label for c in configs}
        assert "no_mmr" in labels

    def test_has_single_source_variants(self):
        """Should include lexical-only and vector-only configs."""
        configs = get_standard_ablation_configs()
        labels = {c.label for c in configs}
        assert "no_vector" in labels  # lexical only
        assert "no_lexical" in labels  # vector only

    def test_has_diversity_variants(self):
        """Should include high and low diversity configs."""
        configs = get_standard_ablation_configs()
        labels = {c.label for c in configs}
        assert "high_diversity" in labels
        assert "low_diversity" in labels

    def test_diversity_lambdas_differ(self):
        """High/low diversity should have different lambda values."""
        configs = get_standard_ablation_configs()
        by_label = {c.label: c for c in configs}
        assert by_label["high_diversity"].mmr_lambda < by_label["full_pipeline"].mmr_lambda
        assert by_label["low_diversity"].mmr_lambda > by_label["full_pipeline"].mmr_lambda


# =============================================================================
# Tests: _run_ablation_search routing
# =============================================================================


class TestRunAblationSearch:
    """Test that _run_ablation_search routes to the correct search function."""

    def _make_mocks(self):
        """Create mock objects for all search dependencies."""
        return {
            "bm25_repo": MagicMock(),
            "vector_repo": MagicMock(),
            "embeddings_store": MagicMock(),
            "search_service": MagicMock(),
        }

    @patch("app.evaluation.evaluation_job._mmr_rerank")
    @patch("app.evaluation.evaluation_job.run_hybrid_search")
    @patch("app.evaluation.evaluation_job.run_vector_search")
    @patch("app.evaluation.evaluation_job.run_lexical_search")
    def test_lexical_only_calls_lexical(self, mock_lex, mock_vec, mock_hyb, mock_mmr):
        """use_vector=False should call only lexical search."""
        mock_lex.return_value = [MagicMock()]
        mocks = self._make_mocks()
        config = AblationConfig(label="test", use_vector=False, use_mmr=False)

        results = _run_ablation_search("test query", config, **mocks)

        assert mock_lex.called
        assert not mock_vec.called
        assert not mock_hyb.called
        assert len(results) == 1

    @patch("app.evaluation.evaluation_job._mmr_rerank")
    @patch("app.evaluation.evaluation_job.run_hybrid_search")
    @patch("app.evaluation.evaluation_job.run_vector_search")
    @patch("app.evaluation.evaluation_job.run_lexical_search")
    def test_vector_only_calls_vector(self, mock_lex, mock_vec, mock_hyb, mock_mmr):
        """use_lexical=False should call only vector search."""
        mock_vec.return_value = [MagicMock()]
        mocks = self._make_mocks()
        config = AblationConfig(label="test", use_lexical=False, use_mmr=False)

        results = _run_ablation_search("test query", config, **mocks)

        assert mock_vec.called
        assert not mock_lex.called
        assert not mock_hyb.called

    @patch("app.evaluation.evaluation_job._mmr_rerank")
    @patch("app.evaluation.evaluation_job.run_hybrid_search")
    @patch("app.evaluation.evaluation_job.run_vector_search")
    @patch("app.evaluation.evaluation_job.run_lexical_search")
    def test_hybrid_without_mmr(self, mock_lex, mock_vec, mock_hyb, mock_mmr):
        """Both enabled + no MMR should call hybrid only."""
        mock_hyb.return_value = [MagicMock(), MagicMock()]
        mocks = self._make_mocks()
        config = AblationConfig(label="test", use_mmr=False)

        results = _run_ablation_search("test query", config, **mocks)

        assert mock_hyb.called
        assert not mock_mmr.called
        assert len(results) == 2

    @patch("app.evaluation.evaluation_job._mmr_rerank")
    @patch("app.evaluation.evaluation_job.run_hybrid_search")
    @patch("app.evaluation.evaluation_job.run_vector_search")
    @patch("app.evaluation.evaluation_job.run_lexical_search")
    def test_hybrid_with_mmr(self, mock_lex, mock_vec, mock_hyb, mock_mmr):
        """Both enabled + MMR should call hybrid then MMR rerank."""
        mock_hyb.return_value = [MagicMock()]
        mock_mmr.return_value = [MagicMock()]
        mocks = self._make_mocks()
        config = AblationConfig(label="test", use_mmr=True, mmr_lambda=0.3)

        _run_ablation_search("test query", config, **mocks)

        assert mock_hyb.called
        assert mock_mmr.called
        # Verify lambda was passed through
        _, kwargs = mock_mmr.call_args
        assert kwargs["lambda_param"] == 0.3

    @patch("app.evaluation.evaluation_job._mmr_rerank")
    @patch("app.evaluation.evaluation_job.run_hybrid_search")
    @patch("app.evaluation.evaluation_job.run_vector_search")
    @patch("app.evaluation.evaluation_job.run_lexical_search")
    def test_full_pipeline_config(self, mock_lex, mock_vec, mock_hyb, mock_mmr):
        """Full pipeline should call hybrid + MMR with default lambda."""
        mock_hyb.return_value = [MagicMock()]
        mock_mmr.return_value = [MagicMock()]
        mocks = self._make_mocks()
        config = AblationConfig(label="full_pipeline")

        _run_ablation_search("test query", config, **mocks)

        assert mock_hyb.called
        assert mock_mmr.called
        _, kwargs = mock_mmr.call_args
        assert kwargs["lambda_param"] == 0.6


# =============================================================================
# Tests: _format_ablation_table
# =============================================================================


class TestFormatAblationTable:
    """Tests for the comparison table formatter."""

    def _sample_metrics(self):
        return {
            "full_pipeline": {
                "ndcg_at_10": 0.7200,
                "recall_at_100": 0.8500,
                "mrr": 0.6500,
                "ild_at_10": 0.4500,
            },
            "no_mmr": {
                "ndcg_at_10": 0.6800,
                "recall_at_100": 0.8500,
                "mrr": 0.6300,
                "ild_at_10": 0.3000,
            },
        }

    def test_returns_string(self):
        """Should return a formatted string."""
        table = _format_ablation_table(self._sample_metrics())
        assert isinstance(table, str)
        assert len(table) > 0

    def test_contains_all_config_labels(self):
        """Table should contain every config label."""
        metrics = self._sample_metrics()
        table = _format_ablation_table(metrics)
        assert "full_pipeline" in table
        assert "no_mmr" in table

    def test_baseline_shows_dashes(self):
        """Baseline row should show '---' instead of delta values."""
        table = _format_ablation_table(self._sample_metrics())
        lines = table.split("\n")
        baseline_line = [l for l in lines if "full_pipeline" in l][0]
        assert "---" in baseline_line

    def test_non_baseline_shows_deltas(self):
        """Non-baseline rows should show numeric deltas."""
        table = _format_ablation_table(self._sample_metrics())
        lines = table.split("\n")
        no_mmr_line = [l for l in lines if "no_mmr" in l][0]
        # nDCG delta should be negative (-0.04)
        assert "-0.0400" in no_mmr_line

    def test_positive_delta_has_plus(self):
        """Positive deltas should show + sign."""
        metrics = {
            "baseline": {"ndcg_at_10": 0.5, "recall_at_100": 0.5, "mrr": 0.5, "ild_at_10": 0.5},
            "better": {"ndcg_at_10": 0.7, "recall_at_100": 0.5, "mrr": 0.5, "ild_at_10": 0.5},
        }
        table = _format_ablation_table(metrics, baseline_label="baseline")
        better_line = [l for l in table.split("\n") if "better" in l][0]
        assert "+0.2000" in better_line

    def test_has_header_and_separator(self):
        """Table should start with header and separator line."""
        table = _format_ablation_table(self._sample_metrics())
        lines = table.split("\n")
        assert "Config" in lines[0]
        assert "---" in lines[1]

    def test_empty_metrics(self):
        """Should handle empty metrics dict."""
        table = _format_ablation_table({})
        assert isinstance(table, str)
