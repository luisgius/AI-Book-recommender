"""
Weighted Reciprocal Rank Fusion (pure).

The deployed fusion (SearchService._fuse_results_rrf, reproduced by
catalog_evaluation.runs.rrf_fuse) gives both lists the same weight:
score(d) = 1 / (k + rank_lexical(d)) + 1 / (k + rank_vector(d)).
The weighted variant multiplies each term by the weight of its list:

    score(d) = w / (k + rank_lexical(d)) + (1 - w) / (k + rank_vector(d))

With w = 0.5 the order is the same as the unweighted fusion (every score is
halved). Ties are broken as in rrf_fuse: books are inserted lexical first,
then the vector-only ones, and the sort is stable. This function lives in the
evaluation package: the domain's fusion and its defaults are not changed.
"""

from typing import Dict, List, Sequence, Tuple

from app.evaluation.catalog_evaluation.design import RRF_K


def weighted_rrf(lexical: Sequence[str], vector: Sequence[str], lexical_weight: float,
                 k: int = RRF_K) -> List[Tuple[str, float]]:
    """
    Fuse two ranked id lists with weights w (lexical) and 1 - w (vector).

    Args:
        lexical: Book ids of the lexical branch, best first
        vector: Book ids of the vector branch, best first
        lexical_weight: w, between 0 and 1
        k: The RRF constant (60, as deployed)

    Returns:
        (book id, fused score) pairs, best first
    """
    if not 0.0 <= lexical_weight <= 1.0:
        raise ValueError(f"lexical_weight must be between 0 and 1, got {lexical_weight}")
    vector_weight = 1.0 - lexical_weight
    scores: Dict[str, float] = {}
    for rank, book_id in enumerate(lexical, start=1):
        scores[book_id] = scores.get(book_id, 0.0) + lexical_weight / (k + rank)
    for rank, book_id in enumerate(vector, start=1):
        scores[book_id] = scores.get(book_id, 0.0) + vector_weight / (k + rank)
    ordered = sorted(scores, key=lambda book_id: scores[book_id], reverse=True)
    return [(book_id, scores[book_id]) for book_id in ordered]
