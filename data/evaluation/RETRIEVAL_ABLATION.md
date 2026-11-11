# Retrieval ablation (V06)

- Date: 2026-11-11T14:57:22+00:00
- Git commit: `9993ad6` (with uncommitted changes in tracked files)
- This run: `python -m app.evaluation.retrieval_ablation --no-judge`
- Collection: 44 queries (24 en, 20 es), 177-book catalog.
- Judgments: `app/evaluation/relevance_judgments_extended.json`, 1556 judgments (637
  original resolved to the catalog + 919 LLM). Before this task it held
  1042. V06 added 514 LLM-graded pairs (grades 0 / 1 / 2 / 3: 291 / 150 / 52 / 21;
  7 of them retrieved only by the accented query variants). The judging run of 2026-11-11T13:48:00+00:00 judged 514 pairs (0 were already in the cache, 0 judge failures, 4.84 s per judgment, one request at a time). This run (`--no-judge`) judged nothing and found every top-10 result judged.
- LLM judges in the file: `qwen3.8:27b-mlx`, prompt `v1.0` (919 judgments). The original `relevance_judgments.json` is unchanged.

Reproduce (from the repository root):

    python -m app.evaluation.retrieval_ablation
    python -m app.evaluation.retrieval_ablation --no-judge

The first command judges any missing pair with the local LLM (see `.env.example`, `JUDGE_*`); the
second needs no LLM and fails if any top-10 result has no judgment.

## Design (fixed before the results were seen)

The design is held as data in `app/evaluation/retrieval_ablation.py` (`SYSTEMS`, `CONTRASTS`) and was
written before any system of this ablation was scored. Every system returns 10 results
per query (production depth). Systems:

- `bm25_whitespace`: BM25, legacy whitespace tokenizer (as the shipped index)
- `bm25_normalized`: BM25, normalizing tokenizer (accents folded, punctuation split, stopwords removed)
- `vec_minilm`: FAISS, all-MiniLM-L6-v2 (shipped index data/indexes/faiss_index)
- `vec_e5`: FAISS, intfloat/multilingual-e5-large-instruct (cosine index)
- `vec_qwen3`: FAISS, Qwen/Qwen3-Embedding-0.6B (cosine index)
- `hybrid_legacy`: RRF(bm25_whitespace, vec_minilm): the system in production today
- `hybrid_norm_minilm`: RRF(bm25_normalized, vec_minilm)
- `hybrid_norm_e5`: RRF(bm25_normalized, vec_e5)
- `hybrid_norm_qwen3`: RRF(bm25_normalized, vec_qwen3)
- `hybrid_norm_e5_bge`: hybrid_norm_e5 + BAAI/bge-reranker-v2-m3 over the top 30
- `hybrid_norm_e5_mmarco`: hybrid_norm_e5 + cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 over the top 30

Primary metric: nDCG@10 (gain 2^grade - 1, a book without a judgment counts as not relevant), on
judgments in which every top-10 result of every system is judged: the holes are graded by the V05
LLM judge with the stored query text and merged into the extended judgments file. Judged@10 is
asserted to be 1.000 for every system.

Pre-declared primary contrasts (system - reference):

- C1: `bm25_normalized` vs `bm25_whitespace`. Does the normalizing tokenizer improve BM25?
- C2: `vec_e5` vs `vec_minilm`. Does multilingual-e5-large-instruct improve the vector side?
- C3: `vec_qwen3` vs `vec_minilm`. Does Qwen3-Embedding-0.6B improve the vector side?
- C4a: `hybrid_norm_e5` vs `hybrid_legacy`. Do tokenizer + e5 together improve the production hybrid?
- C4b: `hybrid_norm_qwen3` vs `hybrid_legacy`. Do tokenizer + Qwen3 together improve the production hybrid?
- C5: `hybrid_norm_e5_bge` vs `hybrid_norm_e5`. Does the bge cross-encoder improve the e5 hybrid?
- C6: `hybrid_norm_e5_mmarco` vs `hybrid_norm_e5`. Does the mMARCO cross-encoder improve the e5 hybrid?

Statistics: two-sided paired randomisation test (sign flips) on per-query ndcg_at_10, 10000 resamples, seed 0; Holm correction across exactly the 7 contrasts; paired percentile bootstrap 95% CI over queries (10000 resamples, seed 0).

Descriptive only, no test: the full systems table, Recall@10, P@5, MRR, the en/es breakdown, the contrasts on the original judgments, the accent table, the cost table. The contrasts on the original
judgments are a sensitivity check: there, books outside the original pool count as not relevant,
which penalises the new systems.

With 44 queries, differences in nDCG@10 below about 0.08 are hard to detect.

## Reproduction check

`hybrid_legacy` must reproduce the `hybrid_rrf` numbers of `COMPARISON_EXTENDED_QRELS.md` on the
judgments as they were before this task (original + the LLM grades of the V05 pool, the top 10 of
the three legacy systems; 405 pairs).

All values match: `hybrid_legacy` is the production system measured in V05.

| Judgments | System | Metric | This run | V05 published | Match |
|---|---|---|---|---|---|
| extended, before V06 | `bm25_whitespace` (lexical_only in V05) | ndcg_at_10 | 0.491 | 0.491 | yes |
| extended, before V06 | `bm25_whitespace` (lexical_only in V05) | recall_at_10 | 0.415 | 0.415 | yes |
| extended, before V06 | `bm25_whitespace` (lexical_only in V05) | precision_at_5 | 0.518 | 0.518 | yes |
| extended, before V06 | `bm25_whitespace` (lexical_only in V05) | mrr | 0.646 | 0.646 | yes |
| extended, before V06 | `bm25_whitespace` (lexical_only in V05) | judged_at_10 | 1.000 | 1.000 | yes |
| extended, before V06 | `vec_minilm` (vector_only in V05) | ndcg_at_10 | 0.598 | 0.598 | yes |
| extended, before V06 | `vec_minilm` (vector_only in V05) | recall_at_10 | 0.572 | 0.572 | yes |
| extended, before V06 | `vec_minilm` (vector_only in V05) | precision_at_5 | 0.618 | 0.618 | yes |
| extended, before V06 | `vec_minilm` (vector_only in V05) | mrr | 0.787 | 0.787 | yes |
| extended, before V06 | `vec_minilm` (vector_only in V05) | judged_at_10 | 1.000 | 1.000 | yes |
| extended, before V06 | `hybrid_legacy` (hybrid_rrf in V05) | ndcg_at_10 | 0.588 | 0.588 | yes |
| extended, before V06 | `hybrid_legacy` (hybrid_rrf in V05) | recall_at_10 | 0.549 | 0.549 | yes |
| extended, before V06 | `hybrid_legacy` (hybrid_rrf in V05) | precision_at_5 | 0.605 | 0.605 | yes |
| extended, before V06 | `hybrid_legacy` (hybrid_rrf in V05) | mrr | 0.768 | 0.768 | yes |
| extended, before V06 | `hybrid_legacy` (hybrid_rrf in V05) | judged_at_10 | 1.000 | 1.000 | yes |
| original | `bm25_whitespace` (lexical_only in V05) | ndcg_at_10 | 0.460 | 0.460 | yes |
| original | `bm25_whitespace` (lexical_only in V05) | recall_at_10 | 0.441 | 0.441 | yes |
| original | `bm25_whitespace` (lexical_only in V05) | precision_at_5 | 0.423 | 0.423 | yes |
| original | `bm25_whitespace` (lexical_only in V05) | mrr | 0.571 | 0.571 | yes |
| original | `bm25_whitespace` (lexical_only in V05) | judged_at_10 | 0.507 | 0.507 | yes |
| original | `vec_minilm` (vector_only in V05) | ndcg_at_10 | 0.530 | 0.530 | yes |
| original | `vec_minilm` (vector_only in V05) | recall_at_10 | 0.498 | 0.498 | yes |
| original | `vec_minilm` (vector_only in V05) | precision_at_5 | 0.500 | 0.500 | yes |
| original | `vec_minilm` (vector_only in V05) | mrr | 0.712 | 0.712 | yes |
| original | `vec_minilm` (vector_only in V05) | judged_at_10 | 0.532 | 0.532 | yes |
| original | `hybrid_legacy` (hybrid_rrf in V05) | ndcg_at_10 | 0.531 | 0.531 | yes |
| original | `hybrid_legacy` (hybrid_rrf in V05) | recall_at_10 | 0.555 | 0.555 | yes |
| original | `hybrid_legacy` (hybrid_rrf in V05) | precision_at_5 | 0.486 | 0.486 | yes |
| original | `hybrid_legacy` (hybrid_rrf in V05) | mrr | 0.678 | 0.678 | yes |
| original | `hybrid_legacy` (hybrid_rrf in V05) | judged_at_10 | 0.550 | 0.550 | yes |

## Pre-declared contrasts (final judgments)

| Contrast | System | Reference | nDCG@10 system | nDCG@10 reference | Diff. [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| C1: Does the normalizing tokenizer improve BM25? | `bm25_normalized` | `bm25_whitespace` | 0.471 | 0.418 | +0.054 [+0.009, +0.108] | 0.026 | 0.155 |
| C2: Does multilingual-e5-large-instruct improve the vector side? | `vec_e5` | `vec_minilm` | 0.621 | 0.523 | +0.098 [+0.006, +0.187] | 0.040 | 0.159 |
| C3: Does Qwen3-Embedding-0.6B improve the vector side? | `vec_qwen3` | `vec_minilm` | 0.592 | 0.523 | +0.069 [-0.007, +0.145] | 0.082 | 0.164 |
| C4a: Do tokenizer + e5 together improve the production hybrid? | `hybrid_norm_e5` | `hybrid_legacy` | 0.570 | 0.508 | +0.062 [+0.002, +0.118] | 0.045 | 0.159 |
| C4b: Do tokenizer + Qwen3 together improve the production hybrid? | `hybrid_norm_qwen3` | `hybrid_legacy` | 0.580 | 0.508 | +0.072 [+0.010, +0.134] | 0.026 | 0.155 |
| C5: Does the bge cross-encoder improve the e5 hybrid? | `hybrid_norm_e5_bge` | `hybrid_norm_e5` | 0.668 | 0.570 | +0.098 [+0.046, +0.155] | 0.001 | 0.008 |
| C6: Does the mMARCO cross-encoder improve the e5 hybrid? | `hybrid_norm_e5_mmarco` | `hybrid_norm_e5` | 0.538 | 0.570 | -0.032 [-0.113, +0.045] | 0.437 | 0.437 |

## Sensitivity: the same contrasts on the original judgments only (descriptive)

| Contrast | System | Reference | nDCG@10 system | nDCG@10 reference | Diff. [95% CI] | p | p (Holm) | Judged@10 system / reference |
|---|---|---|---|---|---|---|---|---|
| C1: Does the normalizing tokenizer improve BM25? | `bm25_normalized` | `bm25_whitespace` | 0.503 | 0.460 | +0.043 [-0.007, +0.103] | 0.159 | 0.890 | 0.464 / 0.507 |
| C2: Does multilingual-e5-large-instruct improve the vector side? | `vec_e5` | `vec_minilm` | 0.468 | 0.530 | -0.063 [-0.145, +0.020] | 0.148 | 0.890 | 0.386 / 0.532 |
| C3: Does Qwen3-Embedding-0.6B improve the vector side? | `vec_qwen3` | `vec_minilm` | 0.498 | 0.530 | -0.032 [-0.128, +0.060] | 0.504 | 1.000 | 0.457 / 0.532 |
| C4a: Do tokenizer + e5 together improve the production hybrid? | `hybrid_norm_e5` | `hybrid_legacy` | 0.543 | 0.531 | +0.012 [-0.056, +0.080] | 0.739 | 1.000 | 0.482 / 0.550 |
| C4b: Do tokenizer + Qwen3 together improve the production hybrid? | `hybrid_norm_qwen3` | `hybrid_legacy` | 0.570 | 0.531 | +0.039 [-0.032, +0.111] | 0.288 | 0.946 | 0.520 / 0.550 |
| C5: Does the bge cross-encoder improve the e5 hybrid? | `hybrid_norm_e5_bge` | `hybrid_norm_e5` | 0.586 | 0.543 | +0.043 [-0.026, +0.111] | 0.237 | 0.946 | 0.486 / 0.482 |
| C6: Does the mMARCO cross-encoder improve the e5 hybrid? | `hybrid_norm_e5_mmarco` | `hybrid_norm_e5` | 0.431 | 0.543 | -0.112 [-0.201, -0.028] | 0.019 | 0.130 | 0.380 / 0.482 |

## All systems (final judgments; descriptive)

| System | nDCG@10 [95% CI] | Recall@10 | P@5 | MRR | Judged@10 | nDCG@10 (original judgments) | Judged@10 (original judgments) |
|---|---|---|---|---|---|---|---|
| `bm25_whitespace` | 0.418 [0.329, 0.509] | 0.277 | 0.518 | 0.646 | 1.000 | 0.460 | 0.507 |
| `bm25_normalized` | 0.471 [0.378, 0.564] | 0.302 | 0.564 | 0.684 | 1.000 | 0.503 | 0.464 |
| `vec_minilm` | 0.523 [0.442, 0.602] | 0.415 | 0.618 | 0.787 | 1.000 | 0.530 | 0.532 |
| `vec_e5` | 0.621 [0.542, 0.696] | 0.481 | 0.709 | 0.796 | 1.000 | 0.468 | 0.386 |
| `vec_qwen3` | 0.592 [0.506, 0.675] | 0.460 | 0.691 | 0.791 | 1.000 | 0.498 | 0.457 |
| `hybrid_legacy` | 0.508 [0.423, 0.593] | 0.392 | 0.605 | 0.768 | 1.000 | 0.531 | 0.550 |
| `hybrid_norm_minilm` | 0.529 [0.440, 0.617] | 0.399 | 0.636 | 0.762 | 1.000 | 0.548 | 0.539 |
| `hybrid_norm_e5` | 0.570 [0.484, 0.651] | 0.427 | 0.686 | 0.794 | 1.000 | 0.543 | 0.482 |
| `hybrid_norm_qwen3` | 0.580 [0.494, 0.663] | 0.444 | 0.645 | 0.765 | 1.000 | 0.570 | 0.520 |
| `hybrid_norm_e5_bge` | 0.668 [0.582, 0.745] | 0.518 | 0.718 | 0.838 | 1.000 | 0.586 | 0.486 |
| `hybrid_norm_e5_mmarco` | 0.538 [0.454, 0.618] | 0.409 | 0.618 | 0.743 | 1.000 | 0.431 | 0.380 |

## By query language (final judgments; descriptive)

| System | nDCG@10 en (n=24) [95% CI] | nDCG@10 es (n=20) [95% CI] |
|---|---|---|
| `bm25_whitespace` | 0.385 [0.257, 0.516] | 0.457 [0.322, 0.585] |
| `bm25_normalized` | 0.378 [0.251, 0.508] | 0.584 [0.453, 0.703] |
| `vec_minilm` | 0.499 [0.391, 0.604] | 0.552 [0.430, 0.671] |
| `vec_e5` | 0.538 [0.414, 0.651] | 0.722 [0.648, 0.793] |
| `vec_qwen3` | 0.502 [0.382, 0.621] | 0.700 [0.595, 0.797] |
| `hybrid_legacy` | 0.480 [0.365, 0.595] | 0.541 [0.409, 0.668] |
| `hybrid_norm_minilm` | 0.473 [0.356, 0.592] | 0.596 [0.463, 0.719] |
| `hybrid_norm_e5` | 0.486 [0.362, 0.604] | 0.670 [0.564, 0.767] |
| `hybrid_norm_qwen3` | 0.474 [0.359, 0.585] | 0.708 [0.597, 0.809] |
| `hybrid_norm_e5_bge` | 0.554 [0.427, 0.671] | 0.804 [0.730, 0.869] |
| `hybrid_norm_e5_mmarco` | 0.524 [0.402, 0.639] | 0.554 [0.438, 0.667] |

## Accent robustness (Spanish queries; descriptive)

The Spanish test queries are stored without diacritics. 14 of the 20 have an accented variant
(`text_accented` in `test_queries.json`) that differs from the stored text only in diacritics (the
stored capitalization is kept). Each system was run on both; a query without a variant keeps its
stored ranking in the accented column. Same judgments for both texts (same information need).

| System | es stored (n=20) | es accented (n=20) | Diff. accented - stored [95% CI] | changed stored (n=14) | changed accented (n=14) | Diff. [95% CI] | Identical top 10 |
|---|---|---|---|---|---|---|---|
| `bm25_whitespace` | 0.457 | 0.501 | +0.044 [-0.007, +0.109] | 0.416 | 0.479 | +0.063 [-0.010, +0.152] | 5 of 14 |
| `bm25_normalized` | 0.584 | 0.584 | +0.000 [+0.000, +0.000] | 0.591 | 0.591 | +0.000 [+0.000, +0.000] | 14 of 14 |
| `vec_minilm` | 0.552 | 0.552 | +0.000 [+0.000, +0.000] | 0.534 | 0.534 | +0.000 [+0.000, +0.000] | 14 of 14 |
| `vec_e5` | 0.722 | 0.715 | -0.007 [-0.025, +0.005] | 0.702 | 0.692 | -0.010 [-0.035, +0.007] | 0 of 14 |
| `vec_qwen3` | 0.700 | 0.713 | +0.013 [+0.001, +0.029] | 0.726 | 0.744 | +0.018 [+0.002, +0.041] | 1 of 14 |
| `hybrid_legacy` | 0.541 | 0.558 | +0.016 [-0.003, +0.048] | 0.516 | 0.538 | +0.023 [-0.005, +0.068] | 5 of 14 |
| `hybrid_norm_minilm` | 0.596 | 0.596 | +0.000 [+0.000, +0.000] | 0.607 | 0.607 | +0.000 [+0.000, +0.000] | 14 of 14 |
| `hybrid_norm_e5` | 0.670 | 0.672 | +0.001 [-0.003, +0.005] | 0.669 | 0.671 | +0.002 [-0.004, +0.007] | 2 of 14 |
| `hybrid_norm_qwen3` | 0.708 | 0.708 | -0.000 [-0.002, +0.002] | 0.737 | 0.737 | -0.000 [-0.003, +0.002] | 5 of 14 |
| `hybrid_norm_e5_bge` | 0.804 | 0.799 | -0.005 [-0.027, +0.015] | 0.817 | 0.811 | -0.007 [-0.040, +0.022] | 0 of 14 |
| `hybrid_norm_e5_mmarco` | 0.554 | 0.556 | +0.001 [-0.026, +0.030] | 0.612 | 0.613 | +0.002 [-0.038, +0.042] | 0 of 14 |

## Cost (descriptive)

Device: `mps` (Apple M5 Pro). Latency is per query over the 44
queries, after one warm-up query, measured in the same process right after each system was built.
No local LLM server answered when the run started; a GPU shared with other work would still make the timings indicative only.

| System | Median ms / query | p95 ms / query | Warm-up query ms |
|---|---|---|---|
| `bm25_whitespace` | 0.1 | 0.1 | 0 |
| `bm25_normalized` | 0.1 | 0.1 | 0 |
| `vec_minilm` | 3.6 | 19.0 | 30 |
| `vec_e5` | 11.3 | 26.2 | 651 |
| `vec_qwen3` | 19.3 | 72.8 | 953 |
| `hybrid_legacy` | 3.4 | 3.5 | 4 |
| `hybrid_norm_minilm` | 3.4 | 3.6 | 4 |
| `hybrid_norm_e5` | 11.3 | 12.2 | 12 |
| `hybrid_norm_qwen3` | 19.6 | 22.2 | 21 |
| `hybrid_norm_e5_bge` | 2242.3 | 2392.4 | 2856 |
| `hybrid_norm_e5_mmarco` | 342.4 | 398.5 | 402 |

| Component | Role | Device | Size on disk | Load | Index build (177 books) | Dimension | Index |
|---|---|---|---|---|---|---|---|
| `bm25:whitespace` | lexical | cpu | n/a | n/a | 0.00 s |  | built in memory from the catalog |
| `bm25:normalized` | lexical | cpu | n/a | n/a | 0.01 s |  | built in memory from the catalog |
| `all-MiniLM-L6-v2` | embedding | mps:0 | 87 MiB | 2.23 s | 0.74 s | 384 | shipped index data/indexes/faiss_index (read only); build time measured by a throwaway rebuild |
| `intfloat/multilingual-e5-large-instruct` | embedding | mps:0 | 1,089 MiB | 3.59 s | 7.22 s | 1024 | loaded from data/indexes_ablation/intfloat_multilingual-e5-large-instruct/faiss_index (built 2026-11-11T13:44:09+00:00) |
| `BAAI/bge-reranker-v2-m3` | reranker | mps | 2,187 MiB | 2.68 s | n/a |  |  |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | reranker | mps | 470 MiB | 1.30 s | n/a |  |  |
| `Qwen/Qwen3-Embedding-0.6B` | embedding | mps:0 | 1,152 MiB | 5.89 s | 18.65 s | 1024 | loaded from data/indexes_ablation/qwen_qwen3-embedding-0.6b/faiss_index (built 2026-11-11T13:47:19+00:00) |

Size on disk: files of the model's snapshot in the local Hugging Face cache. Load: model load
(rerankers: load plus scoring one pair). Index build: encode the 177 books and build the FAISS index.

## Notes

- `vec_minilm` (and the two hybrids built on it) search the shipped index `data/indexes/faiss_index`
  (IndexFlatL2 over MiniLM's unit-length vectors, which ranks as cosine), because that index is the
  production one that `hybrid_legacy` must reproduce. A cosine index rebuilt with the V02 code has the
  same vectors (largest difference 1.5e-7) and differs only in the order of exactly tied books.
- The reranked systems retrieve 60 candidates per retriever instead of 20
  (`SearchService` asks for at least the rerank depth), so C5 and C6 measure the re-ranking stage as it
  would run in production, larger candidate pool included.
- BM25 always returns 10 books; when fewer than 10 match, the rest have score 0 and come in catalog
  order. Those positions are judged like any other.
- LLM grades are not human grades (agreement with the original judgments: `JUDGE_AGREEMENT.md`).
- Run the commands online even with every model cached: with transformers 4.57.3, `HF_HUB_OFFLINE=1`
  makes loading the e5 and Qwen3 tokenizers fail (the library asks the Hub for model metadata; the
  same problem V04 worked around for the cross-encoders).
