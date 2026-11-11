# Comparison on the original and on the LLM-extended judgments

- Date: 2026-11-11T12:56:47+00:00
- Git commit: `97cbae7` (with uncommitted changes in tracked files)
- Reproduce: `python -m app.evaluation.relevance_judge fill --k 10`
- Judge: `ollama/qwen3.8:27b-mlx`, prompt `v1.0`, temperature
  0.0, thinking off. The judge is the same model as the system LLM (`ollama/qwen3.8:27b-mlx`): it is the only model available locally. The literature recommends a judge from a different model family than the system being judged; set JUDGE_PROVIDER / JUDGE_MODEL / JUDGE_BASE_URL to use another one. The four systems compared here do not use an LLM.
- Agreement with the original judgments on 200 sampled pairs (`JUDGE_AGREEMENT.md`): quadratic-weighted kappa 0.729, exact agreement 0.685.
- Holes: 405 (query, book) pairs in the top 10 of the systems below had no
  judgment; the LLM judged all of them (grades 0 / 1 / 2 / 3: 309 / 55 / 22 / 19).
- Extended judgments: `app/evaluation/relevance_judgments_extended.json`, 637 original + 405 LLM
  judgments. Every judgment carries `judge` ("original" or "llm"); LLM ones also carry `model`,
  `prompt_version`, `date` and `rationale`. An original judgment always wins over an LLM one.
  `relevance_judgments.json` is unchanged.
- Metrics, confidence intervals and tests as in `COMPARISON_BASELINE.md` (depth 10,
  baseline `hybrid_rrf`). LLM grades are not human grades: read the extended
  table as a robustness check against pool bias, not as a replacement for the original one.

Systems:

- `lexical_only`: BM25 only (rank-bm25 index in data/indexes/bm25_index.pkl)
- `vector_only`: FAISS only (dense embeddings, index in data/indexes/faiss_index)
- `hybrid_rrf`: BM25 + FAISS fused with Reciprocal Rank Fusion (SearchService)
- `hybrid_rrf_mmr`: hybrid_rrf re-ranked with MMR (lambda 0.6 over the top 20)

## Judged@10

| System | Original judgments | Extended judgments |
|---|---|---|
| lexical_only | 0.507 | 1.000 |
| vector_only | 0.532 | 1.000 |
| hybrid_rrf | 0.550 | 1.000 |
| hybrid_rrf_mmr | 0.550 | 1.000 |

## On the original judgments

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

## On the extended judgments (original + LLM)

| System | nDCG@10 [95% CI] | Recall@10 | P@5 | MRR | Judged@10 | Condensed nDCG@10 | Diff. nDCG@10 vs hybrid_rrf [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| lexical_only | 0.491 [0.391, 0.591] | 0.415 | 0.518 | 0.646 | 1.000 | 0.491 | -0.097 [-0.149, -0.051] | <0.001 | <0.001 |
| vector_only | 0.598 [0.524, 0.670] | 0.572 | 0.618 | 0.787 | 1.000 | 0.598 | +0.010 [-0.053, +0.069] | 0.762 | 1.000 |
| hybrid_rrf (baseline) | 0.588 [0.500, 0.676] | 0.549 | 0.605 | 0.768 | 1.000 | 0.588 |  |  |  |
| hybrid_rrf_mmr | 0.590 [0.502, 0.677] | 0.549 | 0.595 | 0.774 | 1.000 | 0.590 | +0.002 [-0.006, +0.010] | 0.715 | 1.000 |

| System | nDCG@10 en (n=24) [95% CI] | nDCG@10 es (n=20) [95% CI] |
|---|---|---|
| lexical_only | 0.468 [0.324, 0.612] | 0.519 [0.379, 0.653] |
| vector_only | 0.572 [0.470, 0.668] | 0.630 [0.521, 0.738] |
| hybrid_rrf | 0.568 [0.441, 0.688] | 0.614 [0.481, 0.738] |
| hybrid_rrf_mmr | 0.579 [0.454, 0.697] | 0.603 [0.471, 0.728] |
