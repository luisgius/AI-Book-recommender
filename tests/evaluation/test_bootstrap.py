"""Tests for the seeded percentile bootstrap (bootstrap_ci, paired_bootstrap_ci)."""

import random
from statistics import mean

import pytest

from app.evaluation.statistics import bootstrap_ci, paired_bootstrap_ci

_RNG = random.Random(3)
SCORES = [_RNG.random() for _ in range(40)]


def test_same_seed_gives_the_same_interval():
    assert bootstrap_ci(SCORES, n_resamples=2000, seed=11) == bootstrap_ci(SCORES, n_resamples=2000, seed=11)


def test_different_seeds_give_different_resamples():
    assert bootstrap_ci(SCORES, n_resamples=2000, seed=1) != bootstrap_ci(SCORES, n_resamples=2000, seed=2)


def test_returns_the_sample_mean_inside_the_interval():
    value, low, high = bootstrap_ci(SCORES, n_resamples=2000)
    assert value == pytest.approx(mean(SCORES))
    assert low < value < high


def test_higher_confidence_gives_a_wider_interval():
    _, low_80, high_80 = bootstrap_ci(SCORES, n_resamples=2000, confidence=0.80)
    _, low_99, high_99 = bootstrap_ci(SCORES, n_resamples=2000, confidence=0.99)
    assert low_99 < low_80 and high_99 > high_80


def test_single_score_collapses_the_interval():
    assert bootstrap_ci([0.42]) == pytest.approx((0.42, 0.42, 0.42))


def test_constant_scores_collapse_the_interval():
    assert bootstrap_ci([0.5] * 10, n_resamples=500) == pytest.approx((0.5, 0.5, 0.5))


def test_rejects_degenerate_arguments():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, 0.2], confidence=1.0)
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, 0.2], confidence=0.0)
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, 0.2], n_resamples=0)


def test_paired_interval_is_the_interval_of_the_differences():
    other = [s * 0.8 for s in SCORES]
    differences = [a - b for a, b in zip(SCORES, other)]
    assert paired_bootstrap_ci(SCORES, other, n_resamples=1000, seed=5) == bootstrap_ci(
        differences, n_resamples=1000, seed=5
    )


def test_paired_interval_of_identical_systems_is_zero():
    assert paired_bootstrap_ci(SCORES, SCORES, n_resamples=500) == (0.0, 0.0, 0.0)


def test_paired_constant_shift_is_exact():
    shifted = [s - 0.2 for s in SCORES]
    assert paired_bootstrap_ci(SCORES, shifted, n_resamples=500) == pytest.approx((0.2, 0.2, 0.2))


def test_paired_rejects_unpaired_input():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1], [0.1, 0.2])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])
