"""
The incremental MMR must select exactly what the original quadratic version
selected (same formula, same tie-breaking), only faster.
"""

import random
import time
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import pytest

from app.domain.entities import Book, SearchResult
from app.domain.services import SearchService


class Embeddings:
    def __init__(self, vectors: Dict[UUID, List[float]]):
        self.vectors = vectors

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return self.vectors.get(book_id)

    def is_ready(self) -> bool:
        return True


def reference_mmr(results, embeddings, top_k, lambda_param):
    """The original O(k^2 n) implementation, kept as the oracle."""
    book_embeddings = {r.book.id: embeddings[r.book.id] for r in results if r.book.id in embeddings}
    if not book_embeddings:
        return results[:top_k]
    max_score = max(r.final_score for r in results)
    min_score = min(r.final_score for r in results)
    score_range = max_score - min_score if max_score != min_score else 1.0
    selected, candidates = [], list(results)
    while len(selected) < top_k and candidates:
        best, best_idx = float("-inf"), 0
        for i, c in enumerate(candidates):
            if c.book.id not in book_embeddings:
                continue
            relevance = (c.final_score - min_score) / score_range
            sims = [SearchService._cosine_similarity(book_embeddings[c.book.id], book_embeddings[s.book.id])
                    for s in selected if s.book.id in book_embeddings]
            score = lambda_param * relevance - (1 - lambda_param) * (max(sims) if sims else 0.0)
            if score > best:
                best, best_idx = score, i
        selected.append(candidates.pop(best_idx))
    return selected


def random_case(rng, n, dim, missing_rate=0.1, tie_rate=0.2):
    results, vectors = [], {}
    for i in range(n):
        book = Book(id=uuid4(), title=f"B{i}", authors=["A"])
        score = round(rng.random(), 1) if rng.random() < tie_rate else rng.random()
        results.append(SearchResult(book=book, final_score=score, rank=i + 1, source="hybrid"))
        if rng.random() > missing_rate:
            vectors[book.id] = [rng.uniform(-1, 1) for _ in range(dim)]
    results.sort(key=lambda r: r.final_score, reverse=True)
    return results, vectors


@pytest.mark.parametrize("seed", range(25))
def test_matches_the_original_algorithm(seed):
    rng = random.Random(seed)
    results, vectors = random_case(rng, n=rng.randint(2, 40), dim=8)
    top_k, lam = rng.randint(1, 15), rng.choice([0.0, 0.3, 0.5, 0.6, 1.0])
    service = SearchService(None, None, Embeddings(vectors))

    expected = [r.book.id for r in reference_mmr(list(results), vectors, top_k, lam)]
    actual = [r.book.id for r in service._apply_mmr_diversification(list(results), top_k, lam)]

    assert actual == expected


def test_large_k_is_fast():
    rng = random.Random(0)
    results, vectors = random_case(rng, n=180, dim=384, missing_rate=0.0)
    service = SearchService(None, None, Embeddings(vectors))

    start = time.time()
    selected = service._apply_mmr_diversification(results, top_k=100, lambda_param=0.5)

    assert len(selected) == 100
    assert time.time() - start < 10
