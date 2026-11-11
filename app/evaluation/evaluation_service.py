"""
Evaluation service for computing IR metrics.

The ranking metrics are defined once, as functions over a ranked list of book
ids and the graded judgments of one query (book id -> grade 0-3). The
EvaluationService methods apply them to lists of SearchResult, and
app/evaluation/comparison.py applies them to stored rankings, so every report
uses the same definitions:

- gain of a book with grade g is 2^g - 1 (nDCG);
- a book is relevant when its grade is >= 1 (Recall, Precision, MRR);
- a book without a judgment counts as not relevant (grade 0).
"""

import math
from itertools import combinations
from typing import Dict, Hashable, List, Mapping, Sequence, TypeVar

from app.domain.entities import SearchResult
from app.evaluation.types import RelevanceJudgment

from app.domain.entities import Book, Explanation
from app.domain.citation_validation import get_book_field_text, is_valid_snippet

RELEVANT_GRADE = 1
"""Minimum grade for a book to count as relevant in the binary metrics."""

BookId = TypeVar("BookId", bound=Hashable)
"""Any hashable book identifier (UUID in the catalog, str in tests)."""


def ndcg_at_k(ranked_ids: Sequence[BookId], grades: Mapping[BookId, int], k: int = 10) -> float:
    """
    nDCG@k with gain 2^grade - 1 and a log2(rank + 1) discount.

    Args:
        ranked_ids: Book ids in ranked order
        grades: Graded judgments of the query (book id -> 0-3)
        k: Cutoff position

    Returns:
        nDCG@k in [0, 1]; 0.0 when the query has no relevant book
    """
    dcg = 0.0
    for i, book_id in enumerate(ranked_ids[:k], start=1):
        gain = (2 ** grades.get(book_id, 0)) - 1
        dcg += gain / math.log2(i + 1)

    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    idcg = 0.0
    for i, grade in enumerate(ideal_grades, start=1):
        gain = (2 ** grade) - 1
        idcg += gain / math.log2(i + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(ranked_ids: Sequence[BookId], grades: Mapping[BookId, int], k: int = 100) -> float:
    """
    Recall@k = relevant books in the top k / all books judged relevant.

    Returns 0.0 when the query has no relevant book.
    """
    retrieved_relevant = {
        book_id for book_id in ranked_ids[:k] if grades.get(book_id, 0) >= RELEVANT_GRADE
    }
    total_relevant = {book_id for book_id, grade in grades.items() if grade >= RELEVANT_GRADE}
    if not total_relevant:
        return 0.0
    return len(retrieved_relevant) / len(total_relevant)


def precision_at_k(ranked_ids: Sequence[BookId], grades: Mapping[BookId, int], k: int = 5) -> float:
    """
    Precision@k = relevant books in the top k / k.

    The denominator is k even when fewer than k books are returned, so a
    short list is penalised (standard trec_eval behaviour).

    Raises:
        ValueError: If k is not positive
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant_in_top_k = sum(1 for book_id in ranked_ids[:k] if grades.get(book_id, 0) >= RELEVANT_GRADE)
    return relevant_in_top_k / k


def reciprocal_rank(ranked_ids: Sequence[BookId], grades: Mapping[BookId, int]) -> float:
    """
    1 / rank of the first relevant book over the whole list (0.0 if none).

    Averaged over queries this is MRR; with lists of depth 10 it is MRR@10.
    """
    for i, book_id in enumerate(ranked_ids, start=1):
        if grades.get(book_id, 0) >= RELEVANT_GRADE:
            return 1.0 / i
    return 0.0


def judged_fraction_at_k(ranked_ids: Sequence[BookId], grades: Mapping[BookId, int], k: int = 10) -> float:
    """
    Share of the top-k positions filled by a book that has a judgment (Judged@k).

    The denominator is k, as in ir-measures; a short list therefore lowers it.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return sum(1 for book_id in ranked_ids[:k] if book_id in grades) / k


def condensed_ranking(ranked_ids: Sequence[BookId], grades: Mapping[BookId, int]) -> List[BookId]:
    """
    Ranked list with the unjudged books removed (condensed list, Sakai 2007).

    Scoring the condensed list measures a system only on the books that were
    judged, so it is not penalised for surfacing books outside the pool.
    """
    return [book_id for book_id in ranked_ids if book_id in grades]


def judged_at_k(
    results_by_query: Dict[str, List[SearchResult]],
    judgments: Mapping[str, RelevanceJudgment],
    k: int = 10,
) -> float:
    """
    Mean Judged@k over the queries that have judgments.

    Judgments were pooled from earlier systems; a system that surfaces books
    nobody judged is scored as if those books were irrelevant. Reporting
    Judged@k shows how much of a difference may be pool bias.

    Args:
        results_by_query: Search results per query id
        judgments: Relevance judgments per query id
        k: Cutoff position

    Returns:
        Mean share of judged books in the top k (0.0 if no query is judged)
    """
    shares = []
    for query_id, results in results_by_query.items():
        judged = judgments.get(query_id)
        if judged is None or not judged.judgments:
            continue
        shares.append(judged_fraction_at_k([r.book.id for r in results], judged.judgments, k=k))
    return sum(shares) / len(shares) if shares else 0.0


class EvaluationService:
    """
    Service for computing Information Retrieval metrics.
    
    This service implements standard IR evaluation metrics:
    - nDCG@k (Normalized Discounted Cumulative Gain)
    - Recall@k
    - MRR (Mean Reciprocal Rank)
    - ILD@k (Intra-List Diversity)
    """

    @staticmethod
    def _jaccard_distance(a: set, b: set) -> float | None:
        union = a | b
        if not union:
            return None
        inter = a & b
        return 1.0 - (len(inter) / len(union))

    @classmethod
    def _pair_distance(
        cls,
        *,
        authors_i: set,
        authors_j: set,
        cats_i: set,
        cats_j: set,
    ) -> float | None:
        distances: List[float] = []

        author_distance = cls._jaccard_distance(authors_i, authors_j)
        if author_distance is not None:
            distances.append(author_distance)

        cat_distance = cls._jaccard_distance(cats_i, cats_j)
        if cat_distance is not None:
            distances.append(cat_distance)

        if not distances:
            return None

        return sum(distances) / len(distances)

    def compute_ndcg(
        self,
        results: List[SearchResult],
        judgment: RelevanceJudgment,
        k: int = 10
    ) -> float:
        """
        Compute nDCG@k for a query.
        
        Args:
            results: Search results (already ranked)
            judgment: Ground truth relevance judgments
            k: Cutoff position (default 10)
            
        Returns:
            nDCG@k score in [0, 1], where 1 is perfect ranking
        """
        return ndcg_at_k([r.book.id for r in results], judgment.judgments, k=k)

    def compute_recall_at_k(
        self,
        results: List[SearchResult],
        judgment: RelevanceJudgment,
        k: int = 100
    ) -> float:
        """
        Compute Recall@k.
        
        Recall@k = (# relevant docs in top-k) / (# total relevant docs)
        
        Args:
            results: Search results
            judgment: Ground truth relevance judgments
            k: Cutoff position (default 100)
            
        Returns:
            Recall@k score in [0, 1]
        """
        return recall_at_k([r.book.id for r in results], judgment.judgments, k=k)

    def compute_precision_at_k(
        self,
        results: List[SearchResult],
        judgment: RelevanceJudgment,
        k: int = 5
    ) -> float:
        """
        Compute Precision@k.

        Precision@k = (# relevant docs in top-k) / k

        The denominator is k even when fewer than k results are returned, so a
        short result list is penalised (standard trec_eval behaviour).

        Args:
            results: Search results (already ranked)
            judgment: Ground truth relevance judgments
            k: Cutoff position (default 5)

        Returns:
            Precision@k score in [0, 1]
        """
        return precision_at_k([r.book.id for r in results], judgment.judgments, k=k)

    def compute_mrr(
        self,
        results: List[SearchResult],
        judgment: RelevanceJudgment
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        MRR = 1 / rank_of_first_relevant_doc
        
        Args:
            results: Search results
            judgment: Ground truth relevance judgments
            
        Returns:
            MRR score, or 0.0 if no relevant docs found
        """
        return reciprocal_rank([r.book.id for r in results], judgment.judgments)

    def compute_ild(
        self,
        results: List[SearchResult],
        k: int = 10
    ) -> float:
        """
        Compute Intra-List Diversity (ILD@k).
        
        ILD measures diversity by computing average pairwise distance
        between documents in the result list.
        
        We use a simple proxy: diversity based on unique authors and categories.
        
        Args:
            results: Search results
            k: Cutoff position (default 10)
            
        Returns:
            ILD score in [0, 1], higher is more diverse
        """
        ild, _coverage = self.compute_ild_with_coverage(results, k=k)
        return ild

    def compute_ild_with_coverage(
        self,
        results: List[SearchResult],
        k: int = 10
    ) -> tuple[float, float]:
        top_k = results[:k]
        
        if len(top_k) < 2:
            return 0.0, 0.0
        
        # Collect features for diversity computation
        authors_sets = []
        categories_sets = []
        
        for result in top_k:
            authors_sets.append(set(result.book.authors))
            categories_sets.append(set(result.book.categories))
        
        # Compute pairwise diversity
        total_distance = 0.0
        used_pairs = 0
        total_pairs = (len(top_k) * (len(top_k) - 1)) // 2

        items = list(zip(authors_sets, categories_sets))
        for (authors_i, cats_i), (authors_j, cats_j) in combinations(items, 2):
            pair_distance = self._pair_distance(
                authors_i=authors_i,
                authors_j=authors_j,
                cats_i=cats_i,
                cats_j=cats_j,
            )
            if pair_distance is None:
                continue

            total_distance += pair_distance
            used_pairs += 1
        
        if used_pairs == 0:
            return 0.0, 0.0
        
        return (total_distance / used_pairs), (used_pairs / total_pairs if total_pairs else 0.0)


    def compute_citation_precision(
        self,
        explanation: Explanation,
        book: Book
    ) -> float:
        """
        
        Compute citation precision: % of citations that are valid.
        
        Citation precision measures what percentage of the LLM's citations
        are actually grounded in the book data (snippet exists in field).
        
        Args:
            explanation: The explanation with citations to validate
            book: The book entity being cited
            
        Returns:
            Precision in [0.0, 1.0], or 0.0 if no citations
            
        Example:
            >>> # 4 out of 5 citations are valid
            >>> precision = service.compute_citation_precision(explanation, book)
            >>> precision
            0.8  # 80% precision
        """
        if not explanation.citations:
            return 0.0 # no citations -> no precision

        # 2. For each citation, validate using is_valid_snippet()
        valid_count = 0
        total_count = len(explanation.citations)

        for citation in explanation.citations:
            # Get the actual field text from the book
            field_text = get_book_field_text(book, citation.chunk_id)
            
            # Check if the snippet actually exists in that field
            if is_valid_snippet(citation.snippet, field_text):
                valid_count += 1

        # 3. Return valid_count / total_count
        return valid_count / total_count

    def compute_citation_recall(
        self,
        explanation: Explanation,
        book: Book
    ) -> float:
        
        # 1. Identify which fields have content
        fields_with_content = set()

        
        if book.title:
            fields_with_content.add("title")

        if book.description:
            fields_with_content.add("description")

        if book.authors:  # Non-empty list
            fields_with_content.add("authors")

        if book.categories:  # Non-empty list
            fields_with_content.add("categories")

        # 2. Check which fields are referenced in citations
        cited_fields = set()
    
        for citation in explanation.citations:
            cited_fields.add(citation.chunk_id)
    
        # 3. Return cited_fields / total_fields_with_content
        if not fields_with_content:
            return 0.0  # Edge case: book has no content
            
        # Recall = fields cited / fields with content
        return len(cited_fields & fields_with_content) / len(fields_with_content)