"""Tests for the paired permutation test and Holm correction."""

import pytest

from app.evaluation.statistics import holm_correction, paired_permutation_test


def test_identical_systems_are_not_different():
    assert paired_permutation_test([0.5, 0.2, 0.9], [0.5, 0.2, 0.9]) == (0.0, 1.0)


def test_consistent_improvement_is_significant():
    a = [0.6 + 0.01 * i for i in range(30)]
    b = [x - 0.2 for x in a]
    diff, p = paired_permutation_test(a, b, n_resamples=2000)
    assert diff == pytest.approx(0.2)
    assert p < 0.01


def test_noise_is_not_significant():
    a = [0.5, 0.6, 0.4, 0.7, 0.3, 0.5]
    b = [0.6, 0.5, 0.5, 0.6, 0.4, 0.4]
    _, p = paired_permutation_test(a, b, n_resamples=2000)
    assert p > 0.5


def test_is_reproducible_with_the_same_seed():
    a, b = [0.1, 0.9, 0.4, 0.3], [0.2, 0.5, 0.4, 0.1]
    assert paired_permutation_test(a, b, seed=7) == paired_permutation_test(a, b, seed=7)


def test_rejects_unpaired_input():
    with pytest.raises(ValueError):
        paired_permutation_test([0.1], [0.1, 0.2])
    with pytest.raises(ValueError):
        paired_permutation_test([], [])


def test_holm_adjusts_in_step_down_order_and_stays_monotone():
    adjusted = holm_correction({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)   # 3 * 0.01
    assert adjusted["c"] == pytest.approx(0.06)   # 2 * 0.03
    assert adjusted["b"] == pytest.approx(0.06)   # max(1 * 0.04, 0.06)


def test_holm_caps_at_one():
    assert holm_correction({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}
