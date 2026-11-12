"""
Evaluation of the LLM search path (query understanding + multi-query retrieval).

The LLM is called ONCE per test query to obtain its QueryIntent (intent type,
extracted filters, reformulated query, query variations). Intents are cached
in a JSON file, so the run is reproducible and every ablation below reuses
the same LLM output without further LLM calls:

    routing_only               original query, no filters, no variations
                               (isolates intent -> strategy routing and the
                               MMR forced for exploratory intents)
    filters                    original query + extracted filters
    reformulation              reformulated query, no filters, no variations
    reformulation_multi_query  reformulated query + query variations
    full                       everything the LLM produced (production path)

Each variant runs through SearchService.search_with_understanding and is
compared with the plain hybrid baseline (hybrid_rrf) using the same metrics as
the evaluation job, a paired permutation test on nDCG@10 per query and Holm
correction across the variants.

Usage:
    python -m app.evaluation.llm_path_evaluation \\
        --output data/evaluation/results_llm_path.json
"""

import argparse
import json
import logging
import subprocess
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.domain.entities import Book, Explanation
from app.domain.ports import LLMClient
from app.domain.services import SearchService
from app.domain.value_objects import QueryIntent, SearchFilters
from app.evaluation.evaluation_job import (
    DEFAULT_JUDGMENTS_PATH,
    DEFAULT_QUERIES_PATH,
    evaluate_mode,
    load_relevance_judgments,
    load_test_queries,
    resolve_judgments_to_uuids,
    run_hybrid_search,
)
from app.evaluation.evaluation_service import EvaluationService, judged_at_k
from app.evaluation.statistics import holm_correction, paired_permutation_test
from app.infrastructure.llm.filter_guards import FILTER_GUARDS_VERSION, guard_filters

logger = logging.getLogger(__name__)

BASELINE = "hybrid_rrf"
VARIANTS = ("routing_only", "filters", "reformulation", "reformulation_multi_query", "full")
PRIMARY_METRIC = "ndcg_at_10"


# =============================================================================
# Intent caching
# =============================================================================


def intent_to_dict(intent: QueryIntent) -> Dict[str, Any]:
    """Serialise a QueryIntent for the cache file."""
    return {
        "intent_type": intent.intent_type,
        "original_query": intent.original_query,
        "reformulated_query": intent.reformulated_query,
        "extracted_filters": asdict(intent.extracted_filters),
        "confidence": intent.confidence,
        "reasoning": intent.reasoning,
        "query_variations": list(intent.query_variations),
    }


def intent_from_dict(data: Dict[str, Any]) -> QueryIntent:
    """Rebuild a QueryIntent from the cache file."""
    return QueryIntent(
        intent_type=data["intent_type"],
        original_query=data["original_query"],
        reformulated_query=data["reformulated_query"],
        extracted_filters=SearchFilters(**data["extracted_filters"]),
        confidence=data["confidence"],
        reasoning=data["reasoning"],
        query_variations=list(data.get("query_variations", [])),
    )


def variant_intent(intent: QueryIntent, variant: str) -> QueryIntent:
    """Keep only the parts of the LLM output that the variant evaluates."""
    original = intent.original_query
    if variant == "routing_only":
        return replace(intent, reformulated_query=original, extracted_filters=SearchFilters(), query_variations=[])
    if variant == "filters":
        return replace(intent, reformulated_query=original, query_variations=[])
    if variant == "reformulation":
        return replace(intent, extracted_filters=SearchFilters(), query_variations=[])
    if variant == "reformulation_multi_query":
        return replace(intent, extracted_filters=SearchFilters())
    if variant == "full":
        return intent
    raise ValueError(f"Unknown variant '{variant}'")


class FixedIntentClient:
    """LLMClient double that returns a precomputed intent (no LLM calls)."""

    def __init__(self) -> None:
        self.intent: Optional[QueryIntent] = None

    def extract_query_intent(self, query_text: str) -> QueryIntent:
        if self.intent is None:
            raise RuntimeError("No intent set for this query")
        return self.intent

    def generate_grounded_explanation(self, query_text: str, book: Book) -> Explanation:
        raise RuntimeError("Explanations are not used in retrieval evaluation")

    def judge_explanation(self, query_text: str, book: Book, explanation: Explanation):
        raise RuntimeError("Judging is not used in retrieval evaluation")

    def get_model_name(self) -> str:
        return "cached-intents"


# =============================================================================
# Evaluation
# =============================================================================


def collect_intents(
    queries, llm_client: Optional[LLMClient], cache_path: Optional[Path], refresh: bool = False
) -> Dict[str, QueryIntent]:
    """Load cached intents, calling the LLM only for queries not in the cache."""
    cached: Dict[str, Dict[str, Any]] = {}
    if cache_path is not None and cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))["intents"]

    intents: Dict[str, QueryIntent] = {}
    for i, q in enumerate(queries, start=1):
        if q.query_id in cached:
            intents[q.query_id] = intent_from_dict(cached[q.query_id])
            continue
        if llm_client is None:
            raise RuntimeError(f"No cached intent for {q.query_id} and no LLM client")
        start = time.time()
        intents[q.query_id] = llm_client.extract_query_intent(q.text)
        logger.info("[%s/%s] %s understood in %.1fs", i, len(queries), q.query_id, time.time() - start)
        if cache_path is not None:  # save as we go: a long run can be resumed
            cached[q.query_id] = intent_to_dict(intents[q.query_id])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"intents": cached}, ensure_ascii=False, indent=2), encoding="utf-8")
    return intents


def evaluate_llm_path(
    *,
    lexical_repo,
    vector_repo,
    embeddings_store,
    catalog_repo,
    intents: Dict[str, QueryIntent],
    queries_path: str = DEFAULT_QUERIES_PATH,
    judgments_path: str = DEFAULT_JUDGMENTS_PATH,
    max_results: int = 100,
    n_resamples: int = 10000,
) -> Dict[str, Any]:
    """
    Evaluate the baseline and every variant on the test collection.

    Returns:
        Dict with per-mode aggregate metrics, significance of each variant
        against the baseline (mean nDCG@10 difference, p, Holm-adjusted p),
        and a per-query record of the LLM output and nDCG@10 per mode.
    """
    queries = load_test_queries(queries_path)
    judgments = resolve_judgments_to_uuids(load_relevance_judgments(judgments_path), catalog_repo)
    eval_service = EvaluationService()

    # Cached intents may predate the current filter guards; the guards are
    # deterministic, idempotent and only ever remove values, so re-applying
    # them equals re-running the LLM with the current guards.
    intents = {
        qid: replace(i, extracted_filters=guard_filters(i.extracted_filters, i.original_query))
        for qid, i in intents.items()
    }

    fixed = FixedIntentClient()
    service = SearchService(lexical_repo, vector_repo, embeddings_store, llm_client=fixed)
    baseline_service = SearchService(lexical_repo, vector_repo, embeddings_store)

    results: Dict[str, Dict[str, list]] = {BASELINE: {}}
    for q in queries:
        results[BASELINE][q.query_id] = run_hybrid_search(q.text, baseline_service, embeddings_store,
                                                          max_results=max_results)
    for variant in VARIANTS:
        results[variant] = {}
        for q in queries:
            fixed.intent = variant_intent(intents[q.query_id], variant)
            response, _ = service.search_with_understanding(q.text, max_results=max_results)
            results[variant][q.query_id] = response.results

    modes: Dict[str, Dict[str, Any]] = {}
    for mode, by_query in results.items():
        modes[mode] = evaluate_mode(mode, by_query, judgments, eval_service)
        modes[mode]["judged_at_10"] = judged_at_k(by_query, judgments, k=10)

    def per_query(mode: str) -> List[float]:
        metrics = modes[mode]["per_query_metrics"]
        return [metrics[q.query_id][PRIMARY_METRIC] for q in queries if q.query_id in metrics]

    raw_p: Dict[str, float] = {}
    differences: Dict[str, float] = {}
    for variant in VARIANTS:
        differences[variant], raw_p[variant] = paired_permutation_test(
            per_query(variant), per_query(BASELINE), n_resamples=n_resamples
        )
    adjusted = holm_correction(raw_p)

    per_query_records = []
    for q in queries:
        intent = intents[q.query_id]
        record = {"query_id": q.query_id, "text": q.text, **intent_to_dict(intent)}
        record["n_results_full"] = len(results["full"][q.query_id])
        record[PRIMARY_METRIC] = {
            mode: modes[mode]["per_query_metrics"].get(q.query_id, {}).get(PRIMARY_METRIC) for mode in results
        }
        per_query_records.append(record)

    return {
        "modes": {
            mode: {k: v for k, v in m.items() if isinstance(v, float) and not k.startswith("n_")}
            for mode, m in modes.items()
        },
        "significance_vs_baseline": {
            variant: {
                "metric": PRIMARY_METRIC,
                "mean_difference": differences[variant],
                "p_value": raw_p[variant],
                "p_value_holm": adjusted[variant],
            }
            for variant in VARIANTS
        },
        "per_query": per_query_records,
    }


# =============================================================================
# CLI
# =============================================================================


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


LLM_LABELS = ("llm_provider", "llm_model", "temperature", "apply_extracted_category")


def cached_run_labels(output_path: Path, cache_path: Path, query_ids) -> Optional[Dict[str, Any]]:
    """
    The LLM labels of the saved results when this run needs no LLM.

    When the intents cache holds every query, no LLM is called and the
    intents are those of the model that wrote the cache; the saved results
    for the same cache name that model (provider, model, temperature, the
    category option). Taking them from there, not from the environment,
    keeps a re-run from relabelling the published results.

    Returns:
        The labels, or None when an intent is missing or there are no saved results for this cache
    """
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))["intents"]
        config = json.loads(output_path.read_text(encoding="utf-8"))["config"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    if any(query_id not in cached for query_id in query_ids) or config.get("intents_cache") != str(cache_path):
        return None
    if any(label not in config for label in LLM_LABELS):
        return None
    return {label: config[label] for label in LLM_LABELS}


def main() -> None:
    from dotenv import load_dotenv

    from app.infrastructure.config.settings import LLMSettings
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
    from app.infrastructure.llm.prompts import GROUNDED_EXPLANATION_PROMPT_VERSION
    from app.infrastructure.llm.prompts_query import PROMPTS_VERSION as QUERY_PROMPTS_VERSION
    from app.infrastructure.search.bm25_search_repository import BM25SearchRepository
    from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
    from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository

    parser = argparse.ArgumentParser(description="Evaluate the LLM search path against hybrid search")
    parser.add_argument("--db-path", default="data/catalog.db")
    parser.add_argument("--indexes-dir", default="data/indexes")
    parser.add_argument("--output", default="data/evaluation/results_llm_path.json")
    parser.add_argument("--intents-cache", default=None,
                        help="JSON cache of LLM intents (default: data/evaluation/llm_intents_<model>.json)")
    parser.add_argument("--refresh-intents", action="store_true", help="Ignore the cache and call the LLM again")
    parser.add_argument("--max-results", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    settings = LLMSettings.from_env()

    catalog = SqliteBookCatalogRepository(Path(args.db_path))
    books = catalog.get_all()
    # Legacy padding (zero-score books fill the list): reproduces the published results (V13)
    bm25 = BM25SearchRepository(zero_score_padding=True)
    bm25.load_index(str(Path(args.indexes_dir) / "bm25_index.pkl"))
    embeddings = EmbeddingsStoreFaiss()
    embeddings.load_index(str(Path(args.indexes_dir) / "faiss_index"))
    vector = FaissVectorSearchRepository(embeddings, books)

    cache_path = Path(args.intents_cache or
                      f"data/evaluation/llm_intents_{settings.model.replace(':', '-').replace('/', '-')}.json")
    queries = load_test_queries(DEFAULT_QUERIES_PATH)
    cached_labels = None if args.refresh_intents else cached_run_labels(
        Path(args.output), cache_path, [q.query_id for q in queries])
    labels = cached_labels or {
        "llm_provider": settings.provider, "llm_model": settings.model,
        "temperature": settings.temperature, "apply_extracted_category": settings.apply_extracted_category,
    }
    if cached_labels:
        print(f"Every intent is cached: no LLM client; LLM labels from the saved {args.output}")
    llm_client = None
    if settings.enabled and cached_labels is None:
        from app.infrastructure.llm.langchain_llm_client import LangChainLLMClient
        llm_client = LangChainLLMClient(
            provider=settings.provider, model_name=settings.model, temperature=settings.temperature,
            max_retries=settings.max_retries, base_url=settings.base_url, num_ctx=settings.num_ctx,
            reasoning=settings.reasoning, max_tokens=settings.max_tokens, timeout_s=settings.timeout_s,
            known_categories=sorted({c for b in books for c in b.categories}),
            apply_extracted_category=settings.apply_extracted_category,
        )

    started = datetime.now(UTC)
    intents = collect_intents(queries, llm_client, cache_path, refresh=args.refresh_intents)
    report = evaluate_llm_path(lexical_repo=bm25, vector_repo=vector, embeddings_store=embeddings,
                               catalog_repo=catalog, intents=intents, max_results=args.max_results)
    report["config"] = {
        "date": started.isoformat(),
        "git_commit": _git_commit(),
        **labels,
        "filter_guards_version": FILTER_GUARDS_VERSION,
        "query_prompts_version": QUERY_PROMPTS_VERSION,
        "explanation_prompt_version": GROUNDED_EXPLANATION_PROMPT_VERSION,
        "intents_cache": str(cache_path),
        "max_results": args.max_results,
        "baseline": BASELINE,
        "variants": list(VARIANTS),
        "significance": "paired permutation test on per-query nDCG@10, 10000 resamples, Holm correction",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'mode':28} nDCG@10  R@10   P@5    MRR    J@10  | diff vs {BASELINE}  p     p(Holm)")
    for mode, m in report["modes"].items():
        sig = report["significance_vs_baseline"].get(mode)
        tail = (f"| {sig['mean_difference']:+.4f}  {sig['p_value']:.3f}  {sig['p_value_holm']:.3f}"
                if sig else "| baseline")
        print(f"{mode:28} {m['ndcg_at_10']:.4f}  {m['recall_at_10']:.4f} {m['precision_at_5']:.4f} "
              f"{m['mrr']:.4f} {m['judged_at_10']:.2f}  {tail}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
