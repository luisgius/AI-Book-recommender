# LLM path evaluation (2026-11-03)

Does LLM query understanding (intent, filters, reformulation, multi-query
retrieval) improve retrieval over plain hybrid search (BM25 + FAISS, RRF)?

- Collection: 44 test queries, 637 graded judgments (pooled from the non-LLM
  systems), 177-book catalog.
- LLM: local Ollama `qwen3.8:27b-mlx`, temperature 0; query prompts v1.0;
  filter guards v2 (`app/infrastructure/llm/filter_guards.py`).
- Intents cached in `llm_intents_qwen3.8-27b-mlx.json` (one LLM call per query,
  about 12 s each); all variants reuse them.
- Significance: paired permutation test on per-query nDCG@10 (10,000
  resamples) against `hybrid_rrf`, Holm correction across the five variants.
- Judged@10: share of the top 10 that has a relevance judgment.
- Reproduce: `python -m app.evaluation.llm_path_evaluation --max-results 10`.

## Production depth (max_results = 10): `results_llm_path_k10.json`

| Mode | nDCG@10 | R@10 | P@5 | MRR | Judged@10 | Diff. nDCG@10 | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| hybrid_rrf (baseline) | 0.531 | 0.555 | 0.486 | 0.678 | 0.55 | | | |
| routing_only | 0.499 | 0.515 | 0.436 | 0.687 | 0.53 | -0.032 | 0.031 | 0.125 |
| filters | 0.491 | 0.500 | 0.432 | 0.682 | 0.52 | -0.040 | 0.016 | 0.079 |
| reformulation | 0.494 | 0.484 | 0.418 | 0.706 | 0.45 | -0.037 | 0.446 | 0.446 |
| reformulation_multi_query | 0.473 | 0.471 | 0.409 | 0.717 | 0.42 | -0.058 | 0.172 | 0.343 |
| full | 0.454 | 0.448 | 0.400 | 0.695 | 0.41 | -0.077 | 0.066 | 0.198 |

## Deep lists (max_results = 100): `results_llm_path_k100.json`

`full` -0.133 (p Holm 0.016), `routing_only` -0.100 (0.001), `filters` -0.107
(0.001), `reformulation` -0.096 (0.055). At this depth the MMR that the
system forces for "exploratory" intents (lambda 0.5) re-ranks a 100-item list
by diversity, which pushes relevant books out of the top 10. The production
UI and API use 10 results, where the same rule only reorders a short list.

## Reading the results

1. With the current judgments, the LLM path brings **no significant
   improvement** at the production depth, and no variant is significantly
   worse after Holm correction.
2. **Pool bias limits any conclusion about reformulation.** The judgments were
   pooled from BM25, FAISS and hybrid runs. Reformulated queries surface books
   nobody judged (Judged@10 falls from 0.55 to 0.41-0.45), and those count as
   non-relevant. MRR is slightly higher with reformulation (0.706-0.717 vs
   0.678), consistent with the relevant books that were judged moving up.
   Judging the unjudged books in the new top 10s (backlog T05 in
   `review/06-research-improvements-2026-11.md`) is needed before claiming a
   loss or a gain.
3. **Hard filters from the LLM needed guards.** With v1 guards, `filters` was
   significantly worse at depth 10 (-0.074, p Holm 0.024). Per query:
   "Italian pasta recipes" became language=it, "French pastry" became fr,
   "literatura española siglo XIX" became years 1800-1899. The v2 guards keep
   a language only if the query asks for it ("en español", "in English") and
   years only if the query contains digits. The loss halved and stopped
   being significant.
4. **Forced MMR for exploratory intents** is the largest remaining cost at
   depth 100 and the main candidate to revise (make it opt-in or raise
   lambda).

With 44 queries, differences below about 0.08 nDCG@10 cannot be detected
reliably (backlog T03/T06 propose 80-100 queries).
