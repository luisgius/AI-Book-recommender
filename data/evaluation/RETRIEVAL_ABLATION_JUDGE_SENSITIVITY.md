# Sensitivity of the V06 ablation to the LLM judge

- Date: 2026-11-12T11:11:31+00:00
- Git commit: `b308017`
- Old judge (V06): `qwen3.8:27b-mlx`, 919 LLM grades in
  `app/evaluation/relevance_judgments_extended.json`.
- New judge: `z-ai/glm-5.3-flash` (OpenRouter), prompt `v1.0`, the same
  919 pairs, in `app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json` (cache
  `data/evaluation/llm_judgments/z-ai-glm-5.3-flash__prompt-v1.0.jsonl`).
- Grading run of 2026-11-11T15:41:21+00:00: 919 pairs judged (0 already cached), 0 judge failures, 1.95 s per judgment (4 parallel judges, each one request at a time), real cost 0.0591 USD (OpenRouter).
- Pair set pinned to the 919 pairs both judges graded for the V06 ablation pool (the old judge's LLM-graded pairs). The new judge's file now holds 1065 LLM grades: the other 146 were added later by other tasks for pairs the old judge never graded, so they are left out of the agreement and of the new ablation run (`retrieval_ablation --llm-pairs-from app/evaluation/relevance_judgments_extended.json`); the file itself is unchanged.
- Both files hold the same 637 original judgments; only the LLM grades of the
  919 holes differ. Both ablation runs returned the same top 10 for every system and query, so only the judgments differ.
- The re-run reproduces the V05 numbers on the original judgments (all match); its check on the V05 extended judgments reports a mismatch, as expected, because those numbers include the old judge's grades.
- Contrasts, statistics and Holm family exactly as in `RETRIEVAL_ABLATION.md` (two-sided paired
  randomisation test, 10000 resamples, seed 0, Holm over the seven contrasts, paired bootstrap 95% CI).
- A conclusion is the direction of the difference and whether it is significant after Holm
  (p < 0.05); it holds when both are the same under the two judges.

## Agreement between the two judges on the 919 pairs

| Pairs | n | Exact | Within 1 | Quadratic-weighted kappa [95% CI] | Kappa (grade >= 1) | Kappa (grade >= 2) | Mean new - old |
|---|---|---|---|---|---|---|---|
| all | 919 | 0.812 | 0.993 | 0.847 [0.822, 0.871] | 0.772 | 0.769 | +0.038 |
| en queries | 529 | 0.841 | 0.992 | 0.872 [0.837, 0.901] | 0.831 | 0.740 | +0.042 |
| es queries | 390 | 0.772 | 0.995 | 0.808 [0.758, 0.846] | 0.695 | 0.807 | +0.033 |

Grades 0 / 1 / 2 / 3: old judge 600 / 205 / 74 / 40; new judge
571 / 229 / 78 / 41.

Confusion matrix (rows: old judge, columns: new judge):

| Old \ New | 0 | 1 | 2 | 3 | Total |
|---|---|---|---|---|---|
| 0 | 537 | 63 | 0 | 0 | 600 |
| 1 | 34 | 145 | 21 | 5 | 205 |
| 2 | 0 | 20 | 41 | 13 | 74 |
| 3 | 0 | 1 | 16 | 23 | 40 |
| Total | 571 | 229 | 78 | 41 | 919 |

The kappa interval resamples pairs and ignores that pairs of one query are not independent.

## The seven pre-declared contrasts under both judges (nDCG@10, system - reference)

| Contrast | Old judge: diff. [95% CI] | p | p (Holm) | New judge: diff. [95% CI] | p | p (Holm) | Conclusion holds |
|---|---|---|---|---|---|---|---|
| C1: Does the normalizing tokenizer improve BM25? | +0.054 [+0.009, +0.108] | 0.026 | 0.155 | +0.045 [+0.004, +0.095] | 0.047 | 0.236 | yes |
| C2: Does multilingual-e5-large-instruct improve the vector side? | +0.098 [+0.006, +0.187] | 0.040 | 0.159 | +0.111 [+0.023, +0.196] | 0.017 | 0.103 | yes |
| C3: Does Qwen3-Embedding-0.6B improve the vector side? | +0.069 [-0.007, +0.145] | 0.082 | 0.164 | +0.060 [-0.024, +0.144] | 0.173 | 0.280 | yes |
| C4a: Do tokenizer + e5 together improve the production hybrid? | +0.062 [+0.002, +0.118] | 0.045 | 0.159 | +0.059 [-0.004, +0.118] | 0.070 | 0.280 | yes |
| C4b: Do tokenizer + Qwen3 together improve the production hybrid? | +0.072 [+0.010, +0.134] | 0.026 | 0.155 | +0.067 [-0.005, +0.138] | 0.075 | 0.280 | yes |
| C5: Does the bge cross-encoder improve the e5 hybrid? | +0.098 [+0.046, +0.155] | 0.001 | 0.008 | +0.100 [+0.043, +0.160] | 0.003 | 0.019 | yes |
| C6: Does the mMARCO cross-encoder improve the e5 hybrid? | -0.032 [-0.113, +0.045] | 0.437 | 0.437 | -0.073 [-0.164, +0.011] | 0.112 | 0.280 | yes |

## nDCG@10 of every system under both judges (descriptive)

| System | Old judge | New judge | New - old |
|---|---|---|---|
| `bm25_whitespace` | 0.418 | 0.451 | +0.033 |
| `bm25_normalized` | 0.471 | 0.496 | +0.025 |
| `vec_minilm` | 0.523 | 0.563 | +0.040 |
| `vec_e5` | 0.621 | 0.674 | +0.052 |
| `vec_qwen3` | 0.592 | 0.623 | +0.031 |
| `hybrid_legacy` | 0.508 | 0.543 | +0.035 |
| `hybrid_norm_minilm` | 0.529 | 0.563 | +0.034 |
| `hybrid_norm_e5` | 0.570 | 0.601 | +0.031 |
| `hybrid_norm_qwen3` | 0.580 | 0.610 | +0.030 |
| `hybrid_norm_e5_bge` | 0.668 | 0.701 | +0.033 |
| `hybrid_norm_e5_mmarco` | 0.538 | 0.528 | -0.010 |

## Which V06 conclusions hold

- C1 holds: `bm25_normalized` improves nDCG@10 (+0.054, p (Holm) 0.155, not significant after Holm) with the old judge and improves nDCG@10 (+0.045, p (Holm) 0.236, not significant after Holm) with the new one.
- C2 holds: `vec_e5` improves nDCG@10 (+0.098, p (Holm) 0.159, not significant after Holm) with the old judge and improves nDCG@10 (+0.111, p (Holm) 0.103, not significant after Holm) with the new one.
- C3 holds: `vec_qwen3` improves nDCG@10 (+0.069, p (Holm) 0.164, not significant after Holm) with the old judge and improves nDCG@10 (+0.060, p (Holm) 0.280, not significant after Holm) with the new one.
- C4a holds: `hybrid_norm_e5` improves nDCG@10 (+0.062, p (Holm) 0.159, not significant after Holm) with the old judge and improves nDCG@10 (+0.059, p (Holm) 0.280, not significant after Holm) with the new one.
- C4b holds: `hybrid_norm_qwen3` improves nDCG@10 (+0.072, p (Holm) 0.155, not significant after Holm) with the old judge and improves nDCG@10 (+0.067, p (Holm) 0.280, not significant after Holm) with the new one.
- C5 holds: `hybrid_norm_e5_bge` improves nDCG@10 (+0.098, p (Holm) 0.008, significant after Holm) with the old judge and improves nDCG@10 (+0.100, p (Holm) 0.019, significant after Holm) with the new one.
- C6 holds: `hybrid_norm_e5_mmarco` lowers nDCG@10 (-0.032, p (Holm) 0.437, not significant after Holm) with the old judge and lowers nDCG@10 (-0.073, p (Holm) 0.280, not significant after Holm) with the new one.

In short: 7 of 7 conclusions hold under both judges
(C1, C2, C3, C4a, C4b, C5, C6).

Not part of the conclusions (the V06 family is Holm-corrected), but worth knowing: the unadjusted p crosses 0.05 between the judges for C4a (0.045 to 0.070), C4b (0.026 to 0.075).

## Reproduce

    python -m app.evaluation.judge_sensitivity grade --workers 4
    python -m app.evaluation.retrieval_ablation --no-judge --extended-judgments-path app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json --llm-pairs-from app/evaluation/relevance_judgments_extended.json --output-json data/evaluation/results_retrieval_ablation_z-ai-glm-5.3-flash.json --output-md ""
    python -m app.evaluation.judge_sensitivity report --judge-model z-ai/glm-5.3-flash

The first command calls the judge set in `JUDGE_PROVIDER` / `JUDGE_MODEL` (see `.env.example`) and
resumes from its cache; the other two call no LLM.
