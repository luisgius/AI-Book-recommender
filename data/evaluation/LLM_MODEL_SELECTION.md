# Choice of the system LLM and of the relevance judge (V10)

- Date of the measurements: 2026-11-11T15:20:30+00:00
- Git commit: `b8d6ee3` (with uncommitted changes in tracked files)
- Raw results: `data/evaluation/results_llm_model_selection.json`; this report is rendered from them by
  `python -m app.evaluation.llm_model_selection report`.
- All models are called through OpenRouter (the local Ollama model is no longer used).
- **Current decision** (speed-quality rule of 2026-11-11, which supersedes the V10 decision below): system LLM **`deepseek/deepseek-v4.1-flash`**, judge **`z-ai/glm-5.3-flash`**. The V10 rule and decision are kept as history.

## Why

The project needs two LLM roles: the system LLM (query understanding and grounded explanations in
Spanish and English) and the judge (relevance grades for the evaluation), from a different vendor
than the system LLM so that the judge does not grade its own family's output. Public rankings of
cheap hosted models disagree with each other, so the choice is made on this project's own tasks.
GPT models are excluded by the author's decision of 2026-11-11.

## Candidates and prices

List prices from `https://openrouter.ai/api/v1/models` on the date of the run, USD per million tokens. They are
the cheapest endpoint of each model; the client asks OpenRouter to route only to endpoints that
support every parameter sent (JSON schema, thinking switch), so the price actually paid can be
higher. All costs below are the costs OpenRouter reported for each call.

| Model | Listed | Input | Output | Structured outputs | Thinking | Status |
|---|---|---|---|---|---|---|
| `qwen/qwen3.8-flash` | yes | 0.1500 | 0.4700 | yes | off | measured |
| `qwen/qwen3.7-flash` | yes | 0.0300 | 0.1300 | no | off | no endpoint supports structured outputs (JSON schema); OpenRouter answers 404 'No endpoints found that can handle the requested parameters' |
| `deepseek/deepseek-v4.1-flash` | yes | 0.0510 | 0.5670 | yes | off | measured |
| `minimax/minimax-m3.1-flash` | yes | 0.0750 | 0.3000 | yes | effort low | measured |
| `z-ai/glm-5.3-flash` | yes | 0.0399 | 0.0784 | yes | effort low | measured |
| `z-ai/glm-5.3` | yes | 0.0699 | 1.4000 | yes | effort low | measured |

## Design (fixed before any result was seen)

- A. Judge: the existing agreement measurement (`python -m app.evaluation.relevance_judge agreement
  --sample 200 --seed 0`, one judge per candidate) on the same seeded
  sample of original judgments, stratified by (grade, query language). Prompt `v1.0`, temperature 0,
  at most 256 output tokens.
- B. Explanations: the first 10 Spanish and first
  10 English test queries by id (q01, q02, q03, q04, q05, q06, q07, q08, q09, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q21), the top
  3 results of the production hybrid (BM25 + MiniLM, RRF, depth 10) for
  each: 60 grounded explanations per model through the real `LangChainLLMClient`
  (temperature 0, at most 1024 output tokens). Citation precision before the guardrail is the share
  of the citations the model wrote whose snippet the guardrail's check
  (`app/domain/citation_validation.py`: the normalized snippet is a substring of the cited field)
  accepts, over all citations of the model. The raw output is read through the client's
  `raw_output_observer`.
- Language of an explanation (summary and reasoning, not the quoted snippets): count the words that
  are Spanish-only and English-only stopwords of the BM25 tokenizer
  (`app/infrastructure/search/text_normalization.py`, accents folded; words in both lists ignored);
  the larger count gives the language, a tie gives none (counted as not the query language).
- C. Query understanding: the 44 test queries through `extract_query_intent`. A filter is
  unrequested when the deterministic rule of `app/infrastructure/llm/filter_guards.py` would drop it:
  a language the query does not ask for with a preposition ("en español", "in English"), or a year
  although the query contains no year. Measured on the filters as the model extracted them, before
  the guards.
- Parsed: an explanation parsed when the model output fit the schema; a query parsed when all three
  steps of the graph (intent, filters, reformulation) did. The OpenAI client retries transient HTTP
  errors twice and `LangChainLLMClient` once more.
- Latency and tokens: the V08 usage recorder; latency per explanation is its single call, per query
  the sum of its three calls. Thinking off where the model allows it, otherwise effort `low`.
  At most 4 requests in flight.

## Selection rule of V10 (fixed before any result was seen; superseded on 2026-11-11)

- System LLM: among the candidates with 100% parsed calls in B and C, the highest pre-guardrail
  citation precision; candidates within 0.02 of the best are tied, and ties are broken by the
  higher share of explanations in the query language, then by the lower real cost of one search
  (one query understanding + 3 explanations).
- Judge: among the candidates from a different vendor than the selected system LLM (the vendor is
  the prefix of the id), the highest quadratic-weighted kappa in A; candidates within 0.02 are
  tied and the cheaper one (real cost per judgment) wins.

## A. Judge role (agreement with the original judgments, 200 pairs)

| Model | n | Quadratic-weighted kappa [95% CI] | Exact | Mean LLM - original | Failures | s / judgment | Real cost (USD) | Real cost / judgment (USD) |
|---|---|---|---|---|---|---|---|---|
| `qwen/qwen3.8-flash` | 200 | 0.752 [0.678, 0.814] | 0.655 | -0.020 | 0 | 2.28 | 0.01681 | 0.000084 |
| `deepseek/deepseek-v4.1-flash` | 200 | 0.714 [0.629, 0.786] | 0.650 | -0.020 | 0 | 1.02 | 0.01673 | 0.000084 |
| `minimax/minimax-m3.1-flash` | 200 | 0.739 [0.660, 0.805] | 0.665 | +0.115 | 0 | 1.81 | 0.01696 | 0.000085 |
| `z-ai/glm-5.3-flash` | 200 | 0.765 [0.704, 0.817] | 0.650 | +0.045 | 0 | 2.03 | 0.01414 | 0.000071 |
| `z-ai/glm-5.3` | 200 | 0.715 [0.635, 0.784] | 0.670 | -0.005 | 0 | 1.06 | 0.03788 | 0.000189 |
| `qwen3.8:27b-mlx` (local, V05; reference only) | 200 | 0.729 [0.645, 0.796] | 0.685 | -0.070 | 0 | about 4 (V05) | n/a (local) | n/a (local) |

Mean LLM - original: positive when the model grades higher. The kappa interval resamples pairs and
ignores that pairs of one query are not independent, so it is optimistic. s / judgment: one request
at a time per model.

Pairwise quadratic-weighted kappa on the same 200 pairs:

| Kappa | `qwen/qwen3.8-flash` | `deepseek/deepseek-v4.1-flash` | `minimax/minimax-m3.1-flash` | `z-ai/glm-5.3-flash` | `z-ai/glm-5.3` | `qwen3.8:27b-mlx (local)` | `original judgments` |
|---|---|---|---|---|---|---|---|
| `qwen/qwen3.8-flash` | 1.000 | 0.879 | 0.849 | 0.876 | 0.824 | 0.900 | 0.752 |
| `deepseek/deepseek-v4.1-flash` | 0.879 | 1.000 | 0.888 | 0.872 | 0.843 | 0.892 | 0.714 |
| `minimax/minimax-m3.1-flash` | 0.849 | 0.888 | 1.000 | 0.879 | 0.876 | 0.838 | 0.739 |
| `z-ai/glm-5.3-flash` | 0.876 | 0.872 | 0.879 | 1.000 | 0.850 | 0.859 | 0.765 |
| `z-ai/glm-5.3` | 0.824 | 0.843 | 0.876 | 0.850 | 1.000 | 0.853 | 0.715 |
| `qwen3.8:27b-mlx (local)` | 0.900 | 0.892 | 0.838 | 0.859 | 0.853 | 1.000 | 0.729 |
| `original judgments` | 0.752 | 0.714 | 0.739 | 0.765 | 0.715 | 0.729 | 1.000 |

## B. Explanation role (60 explanations per model)

| Model | Parsed | Citations written | Citation precision (before guardrail) | Zero valid citations | In query language | Median latency (s) | Real cost / explanation (USD) |
|---|---|---|---|---|---|---|---|
| `qwen/qwen3.8-flash` | 60/60 | 137 | 0.964 | 0.017 | 1.000 | 6.61 | 0.000283 |
| `deepseek/deepseek-v4.1-flash` | 60/60 | 179 | 0.972 | 0.000 | 0.983 | 2.68 | 0.000409 |
| `minimax/minimax-m3.1-flash` | 60/60 | 236 | 0.958 | 0.017 | 0.633 | 2.98 | 0.000407 |
| `z-ai/glm-5.3-flash` | 60/60 | 170 | 0.959 | 0.000 | 0.950 | 4.35 | 0.000310 |
| `z-ai/glm-5.3` | 59/60 | 166 | 0.970 | 0.000 | 1.000 | 2.85 | 0.000935 |

Citation precision over all citations written; zero-valid and query-language shares over the parsed
explanations (an explanation with no valid citation is replaced by the guardrail's fallback text).

## C. Query-understanding role (44 queries per model)

| Model | Parsed | Unrequested language filter | Unrequested year filter | Unrequested language or year | Median latency / query (s) | Real cost / query (USD) | Real cost / search (USD) |
|---|---|---|---|---|---|---|---|
| `qwen/qwen3.8-flash` | 44/44 | 0.227 | 0.045 | 0.250 | 8.19 | 0.000187 | 0.001037 |
| `deepseek/deepseek-v4.1-flash` | 44/44 | 0.386 | 0.023 | 0.386 | 2.81 | 0.000272 | 0.001498 |
| `minimax/minimax-m3.1-flash` | 44/44 | 0.341 | 0.023 | 0.341 | 3.73 | 0.000245 | 0.001466 |
| `z-ai/glm-5.3-flash` | 43/44 | 0.205 | 0.045 | 0.227 | 5.06 | 0.000319 | 0.001249 |
| `z-ai/glm-5.3` | 44/44 | 0.273 | 0.000 | 0.273 | 3.21 | 0.000815 | 0.003620 |

Unrequested shares over the queries whose filter step parsed. Real cost per search: one query
understanding plus 3 explanations.

## Decision of V10 (mechanical application of the V10 rule; history, superseded)

- System LLM: **`qwen/qwen3.8-flash`**. Eligible: `qwen/qwen3.8-flash`, `deepseek/deepseek-v4.1-flash`, `minimax/minimax-m3.1-flash`. Best pre-guardrail citation precision: 0.972; tied within 0.02: `qwen/qwen3.8-flash`, `deepseek/deepseek-v4.1-flash`, `minimax/minimax-m3.1-flash`. Decided by: tie on citation precision, broken by the share of explanations in the query language.
- Excluded from the system-LLM choice:
  - `z-ai/glm-5.3-flash`: not every call parsed (explanations 60/60, queries 43/44)
  - `z-ai/glm-5.3`: not every call parsed (explanations 59/60, queries 44/44)
- Judge: **`z-ai/glm-5.3-flash`**. Eligible: `deepseek/deepseek-v4.1-flash`, `minimax/minimax-m3.1-flash`, `z-ai/glm-5.3-flash`, `z-ai/glm-5.3`. Best kappa: 0.765; tied within 0.02: `z-ai/glm-5.3-flash`. Decided by: highest kappa (no tie).
- Excluded from the judge choice:
  - `qwen/qwen3.8-flash`: same vendor as the system LLM (qwen)

## Decision with the speed-quality rule (2026-11-11; supersedes the V10 decision)

Why the rule changed: The author asked for models balanced between speed and quality and for choices that are measured. The V10 rule chose on quality alone, and the system LLM it chose was the slowest candidate (6.6 s per explanation, 8.2 s per query understanding).

Rule: For each role, the candidates whose primary quality metric is within 0.02 of the best are tied on quality; among them the lowest median latency wins; if another tied candidate is within 10% of that latency, the cheaper one wins. A candidate that failed any call in the measurement is excluded. System LLM: quality = pre-guardrail citation precision (B); a candidate must have parsed every call (B and C) and written at least 95% of its explanations in the query language; latency = median per explanation; cost = real cost per explanation. Judge: among the candidates of another vendor than the selected system LLM, the highest quadratic-weighted kappa (A), with the same 0.02 tie band and the cheaper winning a tie (speed does not matter for the judge).

Applied mechanically to the saved V10 measurements (tables A, B and C above; no new model call):

System LLM (pre-guardrail citation precision (B); median latency per explanation (B); real cost per explanation (B)):

| Model | Citation precision | Median latency / explanation (s) | Real cost / explanation (USD) | Status |
|---|---|---|---|---|
| `qwen/qwen3.8-flash` | 0.964 | 6.61 | 0.000283 | tied on quality |
| `deepseek/deepseek-v4.1-flash` | 0.972 | 2.68 | 0.000409 | selected |
| `minimax/minimax-m3.1-flash` | 0.958 | 2.98 | 0.000407 | excluded: 0.633 of the explanations in the query language (at least 0.95 required) |
| `z-ai/glm-5.3-flash` | 0.959 | 4.35 | 0.000310 | excluded: not every call parsed (explanations 60/60, queries 43/44) |
| `z-ai/glm-5.3` | 0.970 | 2.85 | 0.000935 | excluded: not every call parsed (explanations 59/60, queries 44/44) |

- Result: **`deepseek/deepseek-v4.1-flash`**. Best quality 0.972; tied within 0.02:
  `deepseek/deepseek-v4.1-flash`, `qwen/qwen3.8-flash`. Decided by: tie on quality (within 0.02), broken by the lowest median latency (no other tied candidate within 10% of it).

Judge (quadratic-weighted kappa (A); real cost per judgment (A)):

| Model | Kappa | Real cost / judgment (USD) | Failed judge calls | Status |
|---|---|---|---|---|
| `qwen/qwen3.8-flash` | 0.752 | 0.000084 | 0 | tied on quality |
| `deepseek/deepseek-v4.1-flash` | 0.714 | 0.000084 | 0 | excluded: same vendor as the system LLM (deepseek) |
| `minimax/minimax-m3.1-flash` | 0.739 | 0.000085 | 0 | below the tie band |
| `z-ai/glm-5.3-flash` | 0.765 | 0.000071 | 0 | selected |
| `z-ai/glm-5.3` | 0.715 | 0.000189 | 0 | below the tie band |

- Result: **`z-ai/glm-5.3-flash`**. Best kappa 0.765; tied within 0.02:
  `qwen/qwen3.8-flash`, `z-ai/glm-5.3-flash`. Decided by: tie on kappa (within 0.02), broken by the lower cost per judgment.

Reading of the exclusion clause: it is applied per role, so a judge candidate is excluded for a failed call in A (none failed). If a failed call anywhere in the V10 measurement (B or C) excluded a model from every role, the judge would be `qwen/qwen3.8-flash` (tie on kappa (within 0.02), broken by the lower cost per judgment).

These are the OpenRouter defaults from 2026-11-11 on (`DEFAULT_MODELS["openrouter"]` and
`DEFAULT_JUDGE_MODELS["openrouter"]` in `app/infrastructure/config/settings.py`, and `.env.example`).

## Failure patterns

- `qwen/qwen3.8-flash` invalid citations: 5 not in the book.
- `deepseek/deepseek-v4.1-flash` explanations not in the query language: 1 en queries answered in es.
- `deepseek/deepseek-v4.1-flash` invalid citations: 5 not in the book.
- `minimax/minimax-m3.1-flash` explanations not in the query language: 22 en queries answered in es.
- `minimax/minimax-m3.1-flash` invalid citations: 10 not in the book.
- `z-ai/glm-5.3-flash` explanations not in the query language: 3 en queries answered in es.
- `z-ai/glm-5.3-flash` invalid citations: 7 not in the book.
- `z-ai/glm-5.3` explanations: 1 of 60 calls did not parse (output did not fit the schema (ValidationError): 1).
- `z-ai/glm-5.3` invalid citations: 5 not in the book.
- `qwen/qwen3.8-flash` unrequested filters (11): "Novela historica espana" (language=es); "recetas cocina rumana" (language=ro); "algoritmos de aprendizaje automatico sin matematicas avanzadas" (language=es); "novela guerra civil espanola" (language=es); "literatura espanola siglo XIX realismo" (language=es, min_year=1800, max_year=1899); "novelas sobre la inquisicion espanola" (language=es); "Victorian era mystery novels detective stories" (min_year=1837, max_year=1901); "recetas tradicionales cocina mediterranea" (language=es); "cocina espanola tapas y pinchos" (language=es); "recetas de mariscos y pescados frescos" (language=es); "recetas faciles" (language=es).
- `deepseek/deepseek-v4.1-flash` unrequested filters (17): "Novela historica espana" (language=es); "algoritmos de aprendizaje automatico sin matematicas avanzadas" (language=es); "programacion en python para analisis de datos" (language=es); "redes neuronales convolucionales ejemplos practicos" (language=es); "novela guerra civil espanola" (language=es); "literatura espanola siglo XIX realismo" (language=es, min_year=1800, max_year=1899); "novelas sobre la inquisicion espanola" (language=es); "ficcion historica conquista de america" (language=es); "novelas ambientadas en el Camino de Santiago" (language=es); "recetas tradicionales cocina mediterranea" (language=es); "postres caseros faciles sin horno" (language=es); "Italian pasta recipes homemade sauce" (language=it); "cocina espanola tapas y pinchos" (language=es); "recetas de mariscos y pescados frescos" (language=es); "French pastry desserts professional techniques" (language=fr); "historia" (language=es); "cocina moderna" (language=es).
- `minimax/minimax-m3.1-flash` unrequested filters (15): "Novela historica espana" (language=es); "recetas cocina rumana" (language=es); "introduccion a la inteligencia artificial para principiantes" (language=es); "programacion en python para analisis de datos" (language=es); "novela guerra civil espanola" (language=es); "literatura espanola siglo XIX realismo" (language=es, min_year=1800, max_year=1899); "novelas sobre la inquisicion espanola" (language=es); "ficcion historica conquista de america" (language=es); "novelas ambientadas en el Camino de Santiago" (language=es); "cocina espanola tapas y pinchos" (language=es); "recetas de mariscos y pescados frescos" (language=es); "comida mexicana autentica tacos enchiladas" (language=es); "French pastry desserts professional techniques" (language=fr); "recetas faciles" (language=es); "cocina moderna" (language=es).
- `z-ai/glm-5.3-flash` query understanding: 1 of 44 queries with a failed step (Intent parsing failed: Could not parse response content as the length limit was reached - CompletionUsage(completion_tok).
- `z-ai/glm-5.3-flash` unrequested filters (10): "Novela historica espana" (language=es); "recetas cocina rumana" (language=ro); "introduccion a la inteligencia artificial para principiantes" (language=es); "algoritmos de aprendizaje automatico sin matematicas avanzadas" (language=es); "programacion en python para analisis de datos" (language=es); "literatura espanola siglo XIX realismo" (language=es, min_year=1800, max_year=1899); "Victorian era mystery novels detective stories" (min_year=1837, max_year=1901); "postres caseros faciles sin horno" (language=es); "cocina espanola tapas y pinchos" (language=es); "recetas faciles" (language=es).
- `z-ai/glm-5.3` unrequested filters (12): "Novela historica espana" (language=es); "recetas cocina rumana" (language=es); "algoritmos de aprendizaje automatico sin matematicas avanzadas" (language=es); "redes neuronales convolucionales ejemplos practicos" (language=es); "novela guerra civil espanola" (language=es); "literatura espanola siglo XIX realismo" (language=es); "ficcion historica conquista de america" (language=es); "novelas ambientadas en el Camino de Santiago" (language=es); "Italian pasta recipes homemade sauce" (language=it); "recetas de mariscos y pescados frescos" (language=es); "French pastry desserts professional techniques" (language=fr); "historia" (language=es).

## Spend

Real cost reported by OpenRouter, USD:

| Model | A. Judge | B. Explanations | C. Query understanding | Total |
|---|---|---|---|---|
| `qwen/qwen3.8-flash` | 0.01681 | 0.01700 | 0.00822 | 0.04203 |
| `deepseek/deepseek-v4.1-flash` | 0.01673 | 0.02453 | 0.01196 | 0.05322 |
| `minimax/minimax-m3.1-flash` | 0.01696 | 0.02443 | 0.01078 | 0.05217 |
| `z-ai/glm-5.3-flash` | 0.01414 | 0.01860 | 0.01402 | 0.04677 |
| `z-ai/glm-5.3` | 0.03788 | 0.05610 | 0.03585 | 0.12982 |

Total: 0.32400 USD.
Cross-check: the API key's usage counter (OpenRouter `/key`) went from 0.0809 to 0.3924 USD during the run (0.3115 USD); it is read right after the last call and can lag behind the most recent ones.

## Limits

- One run per model at temperature 0: OpenRouter routes each call to one of several endpoints
  (different hardware and quantization), so a repeat can differ slightly.
- 60 explanations and 44 queries per model: differences of a few points are within noise.
- The agreement is with judgments of unknown provenance (`QRELS_DATASHEET.md`): agreement, not
  correctness. Prices and model versions are those of the run date.

## Reproduce

    python -m app.evaluation.llm_model_selection run      # calls the models (needs OPENROUTER_API_KEY)
    python -m app.evaluation.llm_model_selection report   # re-renders this file, no model call
