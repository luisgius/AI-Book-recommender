"""
Statistical tests for comparing retrieval systems on the same queries.

With a small test collection (44 queries) a difference in mean nDCG@10 can
easily be noise, so every comparison against the baseline is tested:

- paired_permutation_test: two-sided randomisation test on per-query score
  differences (randomly flipping the sign of each difference). Standard in IR
  evaluation and free of normality assumptions.
- holm_correction: Holm-Bonferroni step-down adjustment when several systems
  are compared with the same baseline (controls the family-wise error rate).
- bootstrap_ci / paired_bootstrap_ci: seeded percentile bootstrap over
  queries, giving a 95% confidence interval for a mean score or for the mean
  difference between two systems.
"""

import random
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import numpy as np


def paired_permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Two-sided paired randomisation test of mean(a) - mean(b).

    Args:
        scores_a: Per-query scores of system A
        scores_b: Per-query scores of system B, same queries in the same order
        n_resamples: Random sign-flip permutations
        seed: Random seed (results are reproducible)

    Returns:
        (mean difference a - b, p-value)

    Raises:
        ValueError: If the lists differ in length or are empty
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Both systems must be scored on the same queries")
    if not scores_a:
        raise ValueError("No scores to compare")

    differences = [a - b for a, b in zip(scores_a, scores_b)]
    observed = abs(mean(differences))
    if observed == 0.0:
        return 0.0, 1.0

    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(n_resamples):
        permuted = mean(d if rng.random() < 0.5 else -d for d in differences)
        if abs(permuted) >= observed - 1e-12:
            at_least_as_extreme += 1
    # +1 smoothing: the observed assignment is one of the permutations
    p_value = (at_least_as_extreme + 1) / (n_resamples + 1)
    return mean(differences), p_value


def holm_correction(p_values: Dict[str, float]) -> Dict[str, float]:
    """
    Holm-Bonferroni adjusted p-values (monotone, capped at 1).

    Args:
        p_values: Raw p-value per comparison

    Returns:
        Adjusted p-value per comparison (same keys)
    """
    ordered: List[Tuple[str, float]] = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(ordered):
        value = min(1.0, (m - rank) * p)
        running_max = max(running_max, value)
        adjusted[name] = running_max
    return adjusted


def bootstrap_ci(
    scores: Sequence[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval for the mean of per-query scores.

    Each resample draws len(scores) queries with replacement and records the
    mean; the interval is given by the (1 - confidence) / 2 and
    (1 + confidence) / 2 quantiles of those means. Queries are the unit of
    resampling because they are the independent units of the test collection.

    Args:
        scores: One score per query
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level, strictly between 0 and 1
        seed: Random seed (results are reproducible)

    Returns:
        (mean, lower bound, upper bound). With a single score, or with all
        scores equal, the three values coincide.

    Raises:
        ValueError: If scores is empty, confidence is not in (0, 1) or
            n_resamples is not positive
    """
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("No scores to resample")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    resampled_means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(resampled_means, [tail, 1.0 - tail])
    return float(values.mean()), float(low), float(high)


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval for mean(a) - mean(b).

    The two systems are scored on the same queries, so the queries are
    resampled as pairs, which is the same as resampling the per-query
    differences a - b.

    Args:
        scores_a: Per-query scores of system A
        scores_b: Per-query scores of system B, same queries in the same order
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level, strictly between 0 and 1
        seed: Random seed (results are reproducible)

    Returns:
        (mean difference a - b, lower bound, upper bound)

    Raises:
        ValueError: If the lists differ in length or are empty
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Both systems must be scored on the same queries")
    differences = [a - b for a, b in zip(scores_a, scores_b)]
    return bootstrap_ci(differences, n_resamples=n_resamples, confidence=confidence, seed=seed)
