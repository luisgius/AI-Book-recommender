#!/usr/bin/env python3
"""
Evaluation Job for comparing search modes and computing IR metrics.

This script runs evaluation experiments comparing 4 search modes:
1. Lexical-only (BM25)
2. Vector-only (FAISS)
3. Hybrid (RRF fusion)
4. Hybrid + MMR (with diversification)

Usage:
    python -m app.evaluation.evaluation_job \
        --queries-path app/evaluation/test_queries.json \
        --judgments-path app/evaluation/relevance_judgments.json \
        --output data/evaluation/results.json
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Infrastructure imports
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
from app.infrastructure.search.bm25_search_repository import BM25SearchRepository
from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository

# Domain imports
from app.domain.services import SearchService
from app.domain.value_objects import SearchQuery
from app.domain.entities import SearchResult

# Evaluation imports
from app.evaluation.types import TestQuery, RelevanceJudgment
from app.evaluation.evaluation_service import EvaluationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = "data/catalog.db"
DEFAULT_INDEXES_DIR = "data/indexes"
DEFAULT_QUERIES_PATH = "app/evaluation/test_queries.json"
DEFAULT_JUDGMENTS_PATH = "app/evaluation/relevance_judgments.json"
DEFAULT_OUTPUT_PATH = "data/evaluation/results.json"
DEFAULT_POOL_PER_MODE_LIMIT = 50

NDCG_KS = (5, 10, 20)
RECALL_KS = (10, 20, 100)
ILD_KS = (10, 20)


def load_test_queries(path: str) -> List[TestQuery]:
    """Load test queries from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    
    queries = []
    for item in data:
        query = TestQuery(
            query_id=item["query_id"],
            text=item["text"],
            category=item.get("category") 
        )
        queries.append(query)

    return queries


def load_relevance_judgments(path: str) -> Dict[str, List[dict]]:
    """Load relevance judgments from JSON file (raw, before UUID resolution)."""
    with open(path, "r") as f:
        data = json.load(f)

    judgments = {}
    for item in data:
        judgments[item["query_id"]] = item["judgments"]

    return judgments


def resolve_judgments_to_uuids(
    raw_judgments: Dict[str, List[dict]],
    catalog_repo: SqliteBookCatalogRepository
) -> Dict[str, RelevanceJudgment]:
    """Convert (source, source_id) judgments to UUID-based RelevanceJudgment objects."""
    result = {}

    for query_id, judgment_list in raw_judgments.items():
        uuid_judgments = {}

        for j in judgment_list:
            book = catalog_repo.get_by_source_id(j["source"], j["source_id"])
            if book:
                uuid_judgments[book.id] = j["relevance"]

        result[query_id] = RelevanceJudgment(
            query_id=query_id,
            judgments = uuid_judgments
            )

    return result


def run_lexical_search(
    query_text: str,
    bm25_repo: BM25SearchRepository,
    max_results: int = 100,
):
    """Mode 1: Lexical-only search (BM25)."""
    
    return bm25_repo.search(query_text, max_results)


def run_vector_search(
    query_text: str,
    vector_repo: FaissVectorSearchRepository,
    embeddings_store: EmbeddingsStoreFaiss,
    max_results: int = 100
):
    """Mode 2: Vector-only search (FAISS)."""
    query_embedding = embeddings_store.generate_embedding(query_text)
    return vector_repo.search(query_embedding, max_results=max_results)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _build_book_embeddings(
    results: list,
    *,
    embeddings_store: EmbeddingsStoreFaiss,
) -> Dict[str, List[float]]:
    book_embeddings: Dict[str, List[float]] = {}
    for r in results:
        emb = embeddings_store.get_embedding(r.book.id)
        if emb is not None:
            book_embeddings[str(r.book.id)] = emb
    return book_embeddings


def _make_score_normalizer(results: list) -> Callable[[float], float]:
    max_score = max(r.final_score for r in results) if results else 1.0
    min_score = min(r.final_score for r in results) if results else 0.0
    score_range = max_score - min_score if max_score != min_score else 1.0

    def _norm(score: float) -> float:
        return (score - min_score) / score_range

    return _norm


def _max_similarity(
    *,
    candidate_id: str,
    selected: list,
    book_embeddings: Dict[str, List[float]],
) -> float:
    max_sim = 0.0
    for s in selected:
        s_id = str(s.book.id)
        sim = _cosine_similarity(book_embeddings[candidate_id], book_embeddings[s_id])
        if sim > max_sim:
            max_sim = sim
    return max_sim


def _pick_best_candidate_index(
    *,
    candidates: list,
    selected: list,
    book_embeddings: Dict[str, List[float]],
    norm: Callable[[float], float],
    lambda_param: float,
) -> Optional[int]:
    best_score = float("-inf")
    best_idx: Optional[int] = None

    for i, cand in enumerate(candidates):
        cand_id = str(cand.book.id)
        relevance = norm(cand.final_score)
        max_sim = _max_similarity(
            candidate_id=cand_id,
            selected=selected,
            book_embeddings=book_embeddings,
        ) if selected else 0.0
        mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim
        if mmr_score > best_score:
            best_score = mmr_score
            best_idx = i

    return best_idx


def _mmr_rerank(
    results: list,
    *,
    embeddings_store: EmbeddingsStoreFaiss,
    lambda_param: float,
    mmr_top_k: int,
) -> list:
    if len(results) <= 1:
        return results

    mmr_top_k = min(mmr_top_k, len(results))

    book_embeddings = _build_book_embeddings(results, embeddings_store=embeddings_store)
    if not book_embeddings:
        return results

    norm = _make_score_normalizer(results)

    selected: list = []
    candidates = [r for r in results if str(r.book.id) in book_embeddings]
    while len(selected) < mmr_top_k and candidates:
        best_idx = _pick_best_candidate_index(
            candidates=candidates,
            selected=selected,
            book_embeddings=book_embeddings,
            norm=norm,
            lambda_param=lambda_param,
        )
        if best_idx is None:
            break
        selected.append(candidates.pop(best_idx))

    selected_ids = {str(r.book.id) for r in selected}
    remainder = [r for r in results if str(r.book.id) not in selected_ids]
    reranked = selected + remainder

    for i, r in enumerate(reranked, start=1):
        r.rank = i

    return reranked


def run_hybrid_search(
    query_text: str,
    search_service: SearchService,
    embeddings_store: EmbeddingsStoreFaiss,
    max_results: int = 100,
    use_diversification: bool = False,
    diversity_lambda: float = 0.6,
    mmr_top_k: int = 20,
):
    """Mode 3 & 4: Hybrid search (with optional MMR diversification)."""
    query = SearchQuery(
        text=query_text,
        max_results=max_results,
        use_explanations=False,
        use_diversification=False,
        diversity_lambda=diversity_lambda,
    )
    response = search_service.search_with_fallback(query)
    results = response.results
    if use_diversification:
        results = _mmr_rerank(
            results,
            embeddings_store=embeddings_store,
            lambda_param=diversity_lambda,
            mmr_top_k=mmr_top_k,
        )
    return results


def _serialize_top_titles(results: list, *, top_k: int = 10) -> List[dict]:
    out: List[dict] = []
    for r in results[:top_k]:
        out.append(
            {
                "rank": r.rank,
                "title": r.book.title,
                "book_id": str(r.book.id),
                "source": r.book.source,
                "source_id": r.book.source_id,
                "final_score": r.final_score,
            }
        )
    return out


def _candidate_key(result: Any) -> tuple[str, str]:
    if result.book.source_id is not None:
        return result.book.source, result.book.source_id
    return result.book.source, str(result.book.id)


def _build_existing_judgments_map(
    existing_raw_judgments: Dict[str, List[dict]],
) -> Dict[str, Dict[tuple[str, str], int]]:
    existing_map: Dict[str, Dict[tuple[str, str], int]] = {}
    for qid, items in existing_raw_judgments.items():
        existing_map[qid] = {
            (it.get("source"), it.get("source_id")): it.get("relevance")
            for it in items
        }
    return existing_map


def _pool_candidates_for_query(
    *,
    query: TestQuery,
    results_by_mode: Dict[str, Dict[str, list]],
    existing_map: Dict[str, Dict[tuple[str, str], int]],
    per_mode_limit: int = DEFAULT_POOL_PER_MODE_LIMIT,
) -> List[dict]:
    seen: Dict[tuple[str, str], Any] = {}
    for mode_results in results_by_mode.values():
        for r in mode_results.get(query.query_id, [])[:per_mode_limit]:
            seen.setdefault(_candidate_key(r), r)

    rel_map = existing_map.get(query.query_id, {})
    return [
        {
            "source": r.book.source,
            "source_id": r.book.source_id,
            "title": r.book.title,
            "authors": list(r.book.authors),
            "relevance": rel_map.get((r.book.source, r.book.source_id)),
        }
        for r in seen.values()
    ]


def _pool_payload_entry(
    *,
    query: TestQuery,
    results_by_mode: Dict[str, Dict[str, list]],
    existing_map: Dict[str, Dict[tuple[str, str], int]],
    per_mode_limit: int,
) -> dict:
    return {
        "query_id": query.query_id,
        "text": query.text,
        "category": query.category,
        "candidates": _pool_candidates_for_query(
            query=query,
            results_by_mode=results_by_mode,
            existing_map=existing_map,
            per_mode_limit=per_mode_limit,
        ),
    }


def export_pool_candidates(
    *,
    output_path: str,
    queries: List[TestQuery],
    results_by_mode: Dict[str, Dict[str, list]],
    existing_raw_judgments: Dict[str, List[dict]],
    per_mode_limit: int = DEFAULT_POOL_PER_MODE_LIMIT,
) -> None:
    existing_map = _build_existing_judgments_map(existing_raw_judgments)
    payload: List[dict] = [
        _pool_payload_entry(
            query=q,
            results_by_mode=results_by_mode,
            existing_map=existing_map,
            per_mode_limit=per_mode_limit,
        )
        for q in queries
    ]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def evaluate_mode(
    mode_name: str,
    results_by_query: Dict[str, list],
    judgments: Dict[str, RelevanceJudgment],
    eval_service: EvaluationService
) -> Dict[str, Any]:
    """Compute metrics for a single search mode across all queries."""

    per_query: Dict[str, Dict[str, float]] = {}
    ndcg_sums: Dict[int, float] = dict.fromkeys(NDCG_KS, 0.0)
    recall_sums: Dict[int, float] = dict.fromkeys(RECALL_KS, 0.0)
    ild_sums: Dict[int, float] = dict.fromkeys(ILD_KS, 0.0)
    ild_coverage_sums: Dict[int, float] = dict.fromkeys(ILD_KS, 0.0)
    mrr_sum = 0.0
    used = 0

    skipped_missing_judgment = 0
    skipped_no_resolved_mapping = 0
    queries_no_relevant = 0

    skipped_missing_judgment_query_ids: List[str] = []
    skipped_no_resolved_mapping_query_ids: List[str] = []
    queries_no_relevant_query_ids: List[str] = []

    for query_id, results in results_by_query.items():
        judgment = judgments.get(query_id)
        if judgment is None:
            skipped_missing_judgment += 1
            skipped_missing_judgment_query_ids.append(query_id)
            continue
        if not judgment.judgments:
            # No resolvable ground-truth docs in the catalog.
            skipped_no_resolved_mapping += 1
            skipped_no_resolved_mapping_query_ids.append(query_id)
            continue

        if not any(rel >= 1 for rel in judgment.judgments.values()):
            queries_no_relevant += 1
            queries_no_relevant_query_ids.append(query_id)

        metrics: Dict[str, float] = {}
        for k in NDCG_KS:
            val = eval_service.compute_ndcg(results, judgment, k=k)
            ndcg_sums[k] += val
            metrics[f"ndcg_at_{k}"] = val

        for k in RECALL_KS:
            val = eval_service.compute_recall_at_k(results, judgment, k=k)
            recall_sums[k] += val
            metrics[f"recall_at_{k}"] = val

        mrr = eval_service.compute_mrr(results, judgment)
        mrr_sum += mrr
        metrics["mrr"] = mrr

        for k in ILD_KS:
            val, coverage = eval_service.compute_ild_with_coverage(results, k=k)
            ild_sums[k] += val
            ild_coverage_sums[k] += coverage
            metrics[f"ild_at_{k}"] = val
            metrics[f"ild_coverage_at_{k}"] = coverage

        per_query[query_id] = metrics
        used += 1

    if used == 0:
        logger.warning("No evaluable queries for mode '%s' (no judgments resolved)", mode_name)
        out: Dict[str, Any] = {
            "n_total_queries": len(results_by_query),
            "num_queries": 0,
            "n_skipped_missing_judgment": skipped_missing_judgment,
            "n_skipped_no_resolved_mapping": skipped_no_resolved_mapping,
            "n_queries_no_relevant": queries_no_relevant,
            "skipped_missing_judgment_query_ids": skipped_missing_judgment_query_ids,
            "skipped_no_resolved_mapping_query_ids": skipped_no_resolved_mapping_query_ids,
            "queries_no_relevant_query_ids": queries_no_relevant_query_ids,
            "per_query_metrics": per_query,
        }
        for k in NDCG_KS:
            out[f"ndcg_at_{k}"] = 0.0
        for k in RECALL_KS:
            out[f"recall_at_{k}"] = 0.0
        out["mrr"] = 0.0
        for k in ILD_KS:
            out[f"ild_at_{k}"] = 0.0
            out[f"ild_coverage_at_{k}"] = 0.0
        return out

    out2: Dict[str, Any] = {
        "n_total_queries": len(results_by_query),
        "num_queries": used,
        "n_skipped_missing_judgment": skipped_missing_judgment,
        "n_skipped_no_resolved_mapping": skipped_no_resolved_mapping,
        "n_queries_no_relevant": queries_no_relevant,
        "skipped_missing_judgment_query_ids": skipped_missing_judgment_query_ids,
        "skipped_no_resolved_mapping_query_ids": skipped_no_resolved_mapping_query_ids,
        "queries_no_relevant_query_ids": queries_no_relevant_query_ids,
        "per_query_metrics": per_query,
    }
    for k in NDCG_KS:
        out2[f"ndcg_at_{k}"] = ndcg_sums[k] / used
    for k in RECALL_KS:
        out2[f"recall_at_{k}"] = recall_sums[k] / used
    out2["mrr"] = mrr_sum / used
    for k in ILD_KS:
        out2[f"ild_at_{k}"] = ild_sums[k] / used
        out2[f"ild_coverage_at_{k}"] = ild_coverage_sums[k] / used
    return out2


def main(
    db_path: str = DEFAULT_DB_PATH,
    indexes_dir: str = DEFAULT_INDEXES_DIR,
    queries_path: str = DEFAULT_QUERIES_PATH,
    judgments_path: str = DEFAULT_JUDGMENTS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    export_pool_path: str | None = None,
    pool_per_mode_limit: int = DEFAULT_POOL_PER_MODE_LIMIT,
    max_results: int = 100,
    mmr_top_k: int = 20,
    mmr_lambdas: List[float] | None = None,
) -> None:
    """
    Main entry point for evaluation job.

    Steps:
    1. Load test queries and relevance judgments from JSON
    2. Load search indices (BM25, FAISS) and catalog
    3. Resolve (source, source_id) -> UUID for all judgments
    4. For each query, run 4 search modes
    5. Compute metrics (nDCG@10, Recall@100, MRR, ILD@10) per mode
    6. Aggregate and save results
    """
    logger.info("=" * 70)
    logger.info("EVALUATION JOB")
    logger.info("=" * 70)

    # Step 1: Load test queries and judgments
    logger.info("Step 1: Loading test queries and relevance judgments...")
    queries = load_test_queries(queries_path)
    raw_judgments = load_relevance_judgments(judgments_path)

    total_queries = len(queries)
    judgment_sets_in_file = len(raw_judgments)

    # Converters may include empty judgments lists for convenience; treat those as missing.
    queries_with_any_judgment_entries = {qid for qid, items in raw_judgments.items() if items}

    # A query is considered "relevance-judged" only if it has at least 1 relevant document (rel>=1)
    queries_with_any_relevant_entries = {
        qid
        for qid, items in raw_judgments.items()
        if items and any((it.get("relevance") or 0) >= 1 for it in items)
    }

    queries_missing_judgments_ids = [
        q.query_id for q in queries if q.query_id not in queries_with_any_judgment_entries
    ]

    logger.info(
        "Loaded %s queries and %s judgment sets in file (non-empty=%s, with>=1 relevant=%s, missing=%s)",
        total_queries,
        judgment_sets_in_file,
        len(queries_with_any_judgment_entries),
        len(queries_with_any_relevant_entries),
        len(queries_missing_judgments_ids),
    )

    # Step 2: Load indices and catalog
    logger.info("Step 2: Loading search indices and catalog...")
    catalog_repo = SqliteBookCatalogRepository(Path(db_path))
    books = catalog_repo.get_all()
    logger.info("Catalog loaded: %s books", len(books))

    indexes_dir_path = Path(indexes_dir)

    bm25_repo = BM25SearchRepository()
    bm25_index_path = indexes_dir_path / "bm25_index.pkl"
    if bm25_index_path.exists():
        bm25_repo.load_index(str(bm25_index_path))
        logger.info("BM25 index loaded from %s", bm25_index_path)
    else:
        logger.warning("BM25 index not found at %s (lexical search will return empty)", bm25_index_path)

    embeddings_store = EmbeddingsStoreFaiss()
    faiss_dir = indexes_dir_path / "faiss_index"
    if faiss_dir.exists():
        embeddings_store.load_index(str(faiss_dir))
        logger.info("FAISS index loaded from %s", faiss_dir)
    else:
        logger.warning("FAISS index not found at %s (vector search may return empty)", faiss_dir)

    vector_repo = FaissVectorSearchRepository(embeddings_store, books)
    search_service = SearchService(
        lexical_search=bm25_repo,
        vector_search=vector_repo,
        embeddings_store=embeddings_store,
    )

    # Step 3: Resolve judgments (source, source_id) -> UUID
    logger.info("Step 3: Resolving judgments to UUIDs...")
    judgments = resolve_judgments_to_uuids(raw_judgments, catalog_repo)

    resolved_counts = {qid: len(j.judgments) for qid, j in judgments.items()}
    logger.info("Resolved judgments (UUID-mapped) per query: %s", resolved_counts)

    evaluable_query_ids = [
        qid for qid, j in judgments.items() if j.judgments
    ]
    non_evaluable_query_ids = [
        qid for qid, j in judgments.items() if (not j.judgments)
    ]
    logger.info(
        "Evaluable queries (after UUID resolution): %s/%s",
        len(evaluable_query_ids),
        total_queries,
    )

    # Step 4: Run searches for each mode
    logger.info("Step 4: Running searches for all modes...")
    lexical_results_by_query: Dict[str, list] = {}
    vector_results_by_query: Dict[str, list] = {}
    hybrid_results_by_query: Dict[str, list] = {}
    hybrid_mmr_results_by_query: Dict[str, list] = {}
    hybrid_mmr_by_lambda: Dict[str, Dict[str, list]] = {}

    if mmr_lambdas is None:
        mmr_lambdas = [0.3, 0.6, 0.8]
    mmr_lambdas_sorted = sorted(mmr_lambdas)
    for lam in mmr_lambdas_sorted:
        hybrid_mmr_by_lambda[str(lam)] = {}

    for q in queries:
        logger.info("Query %s: %s", q.query_id, q.text)
        lexical_results_by_query[q.query_id] = run_lexical_search(q.text, bm25_repo, max_results=max_results)
        vector_results_by_query[q.query_id] = run_vector_search(
            q.text,
            vector_repo,
            embeddings_store,
            max_results=max_results,
        )
        hybrid_results_by_query[q.query_id] = run_hybrid_search(
            q.text,
            search_service,
            embeddings_store,
            max_results=max_results,
            use_diversification=False,
        )

        base_hybrid: List[SearchResult] = hybrid_results_by_query[q.query_id]

        # Default MMR for backward-compatible key (lambda=0.6)
        default_copy = copy.deepcopy(base_hybrid)
        hybrid_mmr_results_by_query[q.query_id] = _mmr_rerank(
            default_copy,
            embeddings_store=embeddings_store,
            lambda_param=0.6,
            mmr_top_k=mmr_top_k,
        )

        for lam in mmr_lambdas_sorted:
            lam_copy = copy.deepcopy(base_hybrid)
            hybrid_mmr_by_lambda[str(lam)][q.query_id] = _mmr_rerank(
                lam_copy,
                embeddings_store=embeddings_store,
                lambda_param=float(lam),
                mmr_top_k=mmr_top_k,
            )

    # Step 5: Compute metrics
    logger.info("Step 5: Computing metrics...")
    eval_service = EvaluationService()

    results: Dict[str, Any] = {
        "config": {
            "db_path": db_path,
            "indexes_dir": indexes_dir,
            "queries_path": queries_path,
            "judgments_path": judgments_path,
            "total_queries": total_queries,
            "judgment_sets_in_file": judgment_sets_in_file,
            "queries_with_any_judgment_entries": len(queries_with_any_judgment_entries),
            "queries_with_any_relevant_entries": len(queries_with_any_relevant_entries),
            "queries_missing_judgments": len(queries_missing_judgments_ids),
            "evaluable_queries_after_uuid_resolution": len(evaluable_query_ids),
            "ndcg_gain": "exp2_minus_1",
            "ndcg_ks": list(NDCG_KS),
            "recall_ks": list(RECALL_KS),
            "ild_ks": list(ILD_KS),
            "max_results": max_results,
            "mmr_top_k": mmr_top_k,
            "mmr_lambdas": mmr_lambdas_sorted,
            "pool_per_mode_limit": pool_per_mode_limit,
        },
        "modes": {},
        "debug": {
            "top10_by_mode": {},
            "missing_judgments_query_ids": queries_missing_judgments_ids,
            "non_evaluable_query_ids": non_evaluable_query_ids,
        },
    }

    results["modes"]["lexical_only"] = evaluate_mode(
        "lexical_only",
        lexical_results_by_query,
        judgments,
        eval_service,
    )
    results["modes"]["vector_only"] = evaluate_mode(
        "vector_only",
        vector_results_by_query,
        judgments,
        eval_service,
    )
    results["modes"]["hybrid_rrf"] = evaluate_mode(
        "hybrid_rrf",
        hybrid_results_by_query,
        judgments,
        eval_service,
    )
    results["modes"]["hybrid_rrf_mmr"] = evaluate_mode(
        "hybrid_rrf_mmr",
        hybrid_mmr_results_by_query,
        judgments,
        eval_service,
    )

    results["modes"]["hybrid_rrf_mmr_lambdas"] = {}
    for lam in mmr_lambdas_sorted:
        results["modes"]["hybrid_rrf_mmr_lambdas"][str(lam)] = evaluate_mode(
            f"hybrid_rrf_mmr_{lam}",
            hybrid_mmr_by_lambda[str(lam)],
            judgments,
            eval_service,
        )

    results["debug"]["top10_by_mode"]["lexical_only"] = {
        qid: _serialize_top_titles(res, top_k=10) for qid, res in lexical_results_by_query.items()
    }
    results["debug"]["top10_by_mode"]["vector_only"] = {
        qid: _serialize_top_titles(res, top_k=10) for qid, res in vector_results_by_query.items()
    }
    results["debug"]["top10_by_mode"]["hybrid_rrf"] = {
        qid: _serialize_top_titles(res, top_k=10) for qid, res in hybrid_results_by_query.items()
    }
    results["debug"]["top10_by_mode"]["hybrid_rrf_mmr"] = {
        qid: _serialize_top_titles(res, top_k=10) for qid, res in hybrid_mmr_results_by_query.items()
    }
    results["debug"]["top10_by_mode"]["hybrid_rrf_mmr_lambdas"] = {}
    for lam in mmr_lambdas_sorted:
        results["debug"]["top10_by_mode"]["hybrid_rrf_mmr_lambdas"][str(lam)] = {
            qid: _serialize_top_titles(res, top_k=10)
            for qid, res in hybrid_mmr_by_lambda[str(lam)].items()
        }

    # Step 6: Save results
    logger.info("Step 6: Saving results...")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Saved results to %s", output_path_obj)

    if export_pool_path is not None:
        logger.info("Exporting pooling candidates to %s", export_pool_path)
        export_pool_candidates(
            output_path=export_pool_path,
            queries=queries,
            results_by_mode={
                "lexical_only": lexical_results_by_query,
                "vector_only": vector_results_by_query,
                "hybrid_rrf": hybrid_results_by_query,
                "hybrid_rrf_mmr": hybrid_mmr_results_by_query,
            },
            existing_raw_judgments=raw_judgments,
            per_mode_limit=pool_per_mode_limit,
        )

    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation experiments on search modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--indexes-dir",
        type=str,
        default=DEFAULT_INDEXES_DIR,
        help=f"Directory with search indices (default: {DEFAULT_INDEXES_DIR})",
    )
    parser.add_argument(
        "--queries-path",
        type=str,
        default=DEFAULT_QUERIES_PATH,
        help=f"Path to test queries JSON (default: {DEFAULT_QUERIES_PATH})",
    )
    parser.add_argument(
        "--judgments-path",
        type=str,
        default=DEFAULT_JUDGMENTS_PATH,
        help=f"Path to relevance judgments JSON (default: {DEFAULT_JUDGMENTS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for output results JSON (default: {DEFAULT_OUTPUT_PATH})",
    )

    parser.add_argument(
        "--export-pool",
        type=str,
        default=None,
        help="Optional path to export pooling candidates JSON (union of top-N across modes)",
    )

    parser.add_argument(
        "--pool-per-mode-limit",
        type=int,
        default=DEFAULT_POOL_PER_MODE_LIMIT,
        help=f"When using --export-pool, how many top results per mode to include (default: {DEFAULT_POOL_PER_MODE_LIMIT})",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Max results to retrieve per mode (default: 100)",
    )

    parser.add_argument(
        "--mmr-top-k",
        type=int,
        default=20,
        help="How many items MMR selects/reranks at the top (default: 20)",
    )

    parser.add_argument(
        "--mmr-lambdas",
        nargs="*",
        type=float,
        default=[0.3, 0.6, 0.8],
        help="MMR lambda values to evaluate (default: 0.3 0.6 0.8)",
    )

    args = parser.parse_args()

    try:
        main(
            db_path=args.db_path,
            indexes_dir=args.indexes_dir,
            queries_path=args.queries_path,
            judgments_path=args.judgments_path,
            output_path=args.output,
            export_pool_path=args.export_pool,
            pool_per_mode_limit=args.pool_per_mode_limit,
            max_results=args.max_results,
            mmr_top_k=args.mmr_top_k,
            mmr_lambdas=args.mmr_lambdas,
        )
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
