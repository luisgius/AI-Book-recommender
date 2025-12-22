"""
Evaluation service for computing IR metrics.
"""

import math
from typing import List, Dict
from uuid import UUID

from app.domain.entities import SearchResult
from app.evaluation.types import RelevanceJudgment


class EvaluationService:
    """
    Service for computing Information Retrieval metrics.
    
    This service implements standard IR evaluation metrics:
    - nDCG@k (Normalized Discounted Cumulative Gain)
    - Recall@k
    - MRR (Mean Reciprocal Rank)
    - ILD@k (Intra-List Diversity)
    """

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
        # Limit to top-k results
        top_k = results[:k]
        
        # Compute DCG
        dcg = 0.0
        for i, result in enumerate(top_k, start=1):
            book_id = result.book.id
            relevance = judgment.judgments.get(book_id, 0)
            dcg += relevance / math.log2(i + 1)
        
        # Compute IDCG (ideal DCG)
        ideal_relevances = sorted(judgment.judgments.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances, start=1):
            idcg += rel / math.log2(i + 1)
        
        # Handle edge case: no relevant documents
        if idcg == 0:
            return 0.0
        
        return dcg / idcg

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
        # Count relevant documents in top-k (relevance >= 1)
        top_k = results[:k]
        retrieved_relevant_ids = {
            r.book.id for r in top_k
            if judgment.judgments.get(r.book.id, 0) >= 1
        }
        
        # Count total relevant documents
        total_relevant_ids = {
            book_id for book_id, rel in judgment.judgments.items()
            if rel >= 1
        }
        
        if len(total_relevant_ids) == 0:
            return 0.0
        
        return len(retrieved_relevant_ids) / len(total_relevant_ids)

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
        for i, result in enumerate(results, start=1):
            book_id = result.book.id
            relevance = judgment.judgments.get(book_id, 0)
            if relevance >= 1:  # First relevant document
                return 1.0 / i
        
        # No relevant documents found
        return 0.0

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
        top_k = results[:k]
        
        if len(top_k) < 2:
            return 0.0
        
        # Collect features for diversity computation
        authors_sets = []
        categories_sets = []
        
        for result in top_k:
            authors_sets.append(set(result.book.authors))
            categories_sets.append(set(result.book.categories))
        
        # Compute pairwise diversity
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(top_k)):
            for j in range(i + 1, len(top_k)):
                # Jaccard distance on authors
                authors_i = authors_sets[i]
                authors_j = authors_sets[j]
                union_authors = authors_i | authors_j
                inter_authors = authors_i & authors_j
                
                if len(union_authors) > 0:
                    author_distance = 1.0 - (len(inter_authors) / len(union_authors))
                else:
                    author_distance = 1.0
                
                # Jaccard distance on categories
                cats_i = categories_sets[i]
                cats_j = categories_sets[j]
                union_cats = cats_i | cats_j
                inter_cats = cats_i & cats_j
                
                if len(union_cats) > 0:
                    cat_distance = 1.0 - (len(inter_cats) / len(union_cats))
                else:
                    cat_distance = 1.0
                
                # Average distance
                pair_distance = (author_distance + cat_distance) / 2.0
                total_distance += pair_distance
                num_pairs += 1
        
        # Average over all pairs
        if num_pairs == 0:
            return 0.0
        
        return total_distance / num_pairs