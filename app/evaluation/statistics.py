"""
Statistical tests for comparing retrieval systems on the same queries.

With a small test collection (44 queries) a difference in mean nDCG@10 can
easily be noise, so every comparison against the baseline is tested:

- paired_permutation_test: two-sided randomisation test on per-query score
  differences (randomly flipping the sign of each difference). Standard in IR
  evaluation and free of normality assumptions.
- holm_correction: Holm-Bonferroni step-down adjustment when several systems
  are compared with the same baseline (controls the family-wise error rate).
"""

import random
from statistics import mean
from typing import Dict, List, Sequence, Tuple


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
