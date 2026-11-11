# Agreement of the LLM relevance judge with the original judgments

- Date: 2026-11-11T12:42:51+00:00
- Git commit: `97cbae7` (with uncommitted changes in tracked files)
- Reproduce: `python -m app.evaluation.relevance_judge agreement --sample 200 --seed 0`
- Judge: `ollama/qwen3.8:27b-mlx`, prompt `v1.0`
  (`build_relevance_judge_prompt in app/infrastructure/llm/prompts_judge.py`), temperature 0.0, thinking off, at most
  256 output tokens, structured output (rationale, then grade 0-3).
- The judge is the same model as the system LLM (`ollama/qwen3.8:27b-mlx`): it is the only model available locally. The literature recommends a judge from a different model family than the system being judged; set JUDGE_PROVIDER / JUDGE_MODEL / JUDGE_BASE_URL to use another one. The four systems compared here do not use an LLM.
- Time: 190 judgments made by this run, 4.35 s per judgment (one request at a time).
- Sample: 200 of the 637 original judgments, seed 0, stratified
  by (grade, query language) with proportional allocation: grade 0/en 34, grade 0/es 42, grade 1/en 52, grade 1/es 26, grade 2/en 18, grade 2/es 20, grade 3/en 3, grade 3/es 5.
- Blind: the judge sees only the query and the catalog fields of the book (title, authors,
  categories, year, language, description), never the original grade, a system or a rank.
- The provenance of the original judgments is unknown (`QRELS_DATASHEET.md`), so this measures
  agreement with them, not correctness. The kappa CI resamples pairs and ignores that pairs of
  the same query are not independent, so it is optimistic.

## Agreement

| Pairs | n | Exact | Within 1 | Quadratic-weighted kappa [95% CI] | Kappa (grade >= 1) | Kappa (grade >= 2) | Mean LLM - original |
|---|---|---|---|---|---|---|---|
| all | 200 | 0.685 | 0.965 | 0.729 [0.645, 0.796] | 0.702 | 0.609 | -0.070 |
| en queries | 107 | 0.682 | 0.981 | 0.721 [0.585, 0.810] | 0.588 | 0.718 | -0.037 |
| es queries | 93 | 0.688 | 0.946 | 0.735 [0.618, 0.829] | 0.807 | 0.504 | -0.108 |

Exact: share of pairs with the same grade. Within 1: grades differ by at most one.
Mean LLM - original: positive when the LLM grades higher.

## Confusion matrices (rows: original grade, columns: LLM grade)

All pairs:

| Original \ LLM | 0 | 1 | 2 | 3 | Total |
|---|---|---|---|---|---|
| 0 | 68 | 8 | 0 | 0 | 76 |
| 1 | 17 | 52 | 7 | 2 | 78 |
| 2 | 4 | 12 | 13 | 9 | 38 |
| 3 | 0 | 1 | 3 | 4 | 8 |
| Total | 89 | 73 | 23 | 15 | 200 |

en queries:

| Original \ LLM | 0 | 1 | 2 | 3 | Total |
|---|---|---|---|---|---|
| 0 | 27 | 7 | 0 | 0 | 34 |
| 1 | 11 | 38 | 3 | 0 | 52 |
| 2 | 2 | 4 | 6 | 6 | 18 |
| 3 | 0 | 0 | 1 | 2 | 3 |
| Total | 40 | 49 | 10 | 8 | 107 |

es queries:

| Original \ LLM | 0 | 1 | 2 | 3 | Total |
|---|---|---|---|---|---|
| 0 | 41 | 1 | 0 | 0 | 42 |
| 1 | 6 | 14 | 4 | 2 | 26 |
| 2 | 2 | 8 | 7 | 3 | 20 |
| 3 | 0 | 1 | 2 | 2 | 5 |
| Total | 49 | 24 | 13 | 7 | 93 |

## Largest disagreements (grades 2 or more apart)

7 pairs; 7 shown. All sampled pairs, with the LLM rationale, are in
`judge_agreement.json`.

| Query | Book | Description | Original | LLM | LLM rationale |
|---|---|---|---|---|---|
| q07 "algoritmos de aprendizaje automatico sin matematicas avanzadas" | Automated Machine Learning for Business | yes | 1 | 3 | The book is highly relevant as it explicitly teaches machine learning algorithms using AutoML, a method designed to remove the need for advanced math and statistics, which directly addresses the user's request for learning without advanced mathematics. |
| q07 "algoritmos de aprendizaje automatico sin matematicas avanzadas" | Artificial Intelligence For Dummies | yes | 3 | 1 | The book is a general introduction to Artificial Intelligence rather than a specific guide on machine learning algorithms, and it does not explicitly address the user's constraint of avoiding advanced mathematics, making it only somewhat relevant. |
| q07 "algoritmos de aprendizaje automatico sin matematicas avanzadas" | Machine Learning and Artificial Intelligence | yes | 1 | 3 | The book covers machine learning algorithms and explicitly states that it makes minimal use of mathematics to make topics accessible, which aligns perfectly with the user's request for a resource without advanced mathematics. |
| q09 "data science pandas numpy tutorial" | Machine Learning, Optimization, and Data Science | yes | 2 | 0 | The book is a collection of academic conference proceedings focused on theoretical machine learning and optimization, which does not provide the practical, step-by-step tutorial on specific Python libraries like Pandas and NumPy that the user is seeking. |
| q11 "computer vision opencv beginner guide" | Hands-On Artificial Intelligence with TensorFlow | yes | 2 | 0 | The book focuses on Artificial Intelligence and TensorFlow rather than OpenCV, and it requires prior knowledge of machine learning, making it unsuitable for a beginner looking for a specific OpenCV guide. |
| q15 "novela guerra civil espanola" | España no se rinde | yes | 2 | 0 | The book is a historical novel set in the 8th century regarding the start of the Reconquista, which is a different historical period from the Spanish Civil War (1936-1939) requested in the query. |
| q25 "recetas tradicionales cocina mediterranea" | La cocina mexicana de Socorro y Fernando del Paso | yes | 2 | 0 | The book focuses on Mexican cuisine, which is distinct from the Mediterranean cuisine requested in the query, making it not relevant. |
