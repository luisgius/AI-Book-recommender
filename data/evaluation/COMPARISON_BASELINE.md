# Baseline comparison of the non-LLM retrieval systems

- Date: 2026-11-11T10:17:47+00:00
- Git commit: `1a1a9ad` (with uncommitted changes in tracked files)
- Reproduce: `python -m app.evaluation.comparison`
- Collection: 44 queries (24 en, 20 es), 637 graded judgments (0-3),
  177-book catalog; see `QRELS_DATASHEET.md`. Judgments file SHA-256
  `81bc727205d4`, queries file SHA-256 `76c4dcce85e2`.
- Depth: every system returns at most 10 results per query (`max_results`).

Systems:

- `lexical_only`: BM25 only (rank-bm25 index in data/indexes/bm25_index.pkl)
- `vector_only`: FAISS only (dense embeddings, index in data/indexes/faiss_index)
- `hybrid_rrf`: BM25 + FAISS fused with Reciprocal Rank Fusion (SearchService)
- `hybrid_rrf_mmr`: hybrid_rrf re-ranked with MMR (lambda 0.6 over the top 20)

Method:

- Metrics as in `app/evaluation/evaluation_service.py`: gain 2^grade - 1, a book is relevant
  when its grade is >= 1, a book without a judgment counts as not relevant. MRR is computed
  over the returned list, so at depth 10 it is MRR@10.
- Primary metric: nDCG@10. 95% CI: percentile bootstrap over queries
  (10000 resamples, seed 0).
- Diff.: mean per-query nDCG@10 difference against `hybrid_rrf`, with a paired bootstrap
  CI; p: two-sided paired permutation test (10000 sign-flip resamples); p (Holm):
  Holm correction across the 3 comparisons with the baseline.
- Judged@10: share of the top 10 that has a judgment (denominator 10).
- Condensed nDCG@10: nDCG@10 after removing unjudged books from the ranking. It is computed
  from the same depth-10 lists, so it can only move judged books up within them.
- Queries whose judgments are all grade 0 (q22, q32, q38, q39)
  are kept and score 0 on every metric.

| System | nDCG@10 [95% CI] | Recall@10 | P@5 | MRR | Judged@10 | Condensed nDCG@10 | Diff. nDCG@10 vs hybrid_rrf [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| lexical_only | 0.460 [0.356, 0.564] | 0.441 | 0.423 | 0.571 | 0.507 | 0.522 | -0.071 [-0.132, -0.010] | 0.028 | 0.083 |
| vector_only | 0.530 [0.436, 0.623] | 0.498 | 0.500 | 0.712 | 0.532 | 0.578 | -0.001 [-0.062, +0.057] | 0.979 | 1.000 |
| hybrid_rrf (baseline) | 0.531 [0.436, 0.626] | 0.555 | 0.486 | 0.678 | 0.550 | 0.630 |  |  |  |
| hybrid_rrf_mmr | 0.529 [0.436, 0.622] | 0.555 | 0.473 | 0.684 | 0.550 | 0.633 | -0.002 [-0.008, +0.005] | 0.647 | 1.000 |

| System | nDCG@10 en (n=24) [95% CI] | nDCG@10 es (n=20) [95% CI] |
|---|---|---|
| lexical_only | 0.396 [0.253, 0.543] | 0.537 [0.389, 0.679] |
| vector_only | 0.495 [0.358, 0.627] | 0.572 [0.445, 0.701] |
| hybrid_rrf | 0.492 [0.362, 0.622] | 0.578 [0.440, 0.709] |
| hybrid_rrf_mmr | 0.498 [0.370, 0.627] | 0.567 [0.430, 0.698] |
