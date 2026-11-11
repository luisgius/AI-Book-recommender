# Live check of the cloud profile (V13)

Run of 2026-11-11T18:54:07+00:00 (git `86c5cdc`): the API started locally in the cloud configuration against the Neon schema `eval177_api` (the 177-book collection, documents embedded by the cloud encoder), reached over HTTP at http://127.0.0.1:8765. Descriptive: no contrast was declared in advance.

## Configuration reported by /api/v1/health

Profile: `{"name": "cloud", "schema": "eval177_api", "lexical_ranking": "bm25", "vector_search": "approximate"}`; components ready: `{"lexical_search": true, "vector_search": true, "embeddings_store": true}`; reranker: `{"configured": true, "model": "qwen/qwen3-reranker-8b", "ready": true}`; LLM: `{"configured": true, "provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash"}` (no LLM call is made: the searches use neither understanding nor explanations).

## Every endpoint answers

| Endpoint | Answer |
|---|---|
| GET /api/v1/health | ok |
| GET /ui/ | 200 |
| GET /api/v1/books/{id} | 200 |
| GET /api/v1/books/{id}/similar | 200 (10 results) |
| GET /api/v1/books/{id}/similar?use_diversification=true | 200 (10 results) |
| POST /api/v1/search (use_diversification) | 200 (10 results, degraded False) |
| POST /mcp (initialize) | 200 |
| GET /api/v1/usage | 200 |

## Latency of POST /api/v1/search over the test queries

44 queries, one at a time after one warm-up query, 10 results, from this machine to Neon (Frankfurt) and OpenRouter. Server = `latency_ms` of the response; embedding and re-ranking = the calls recorded by the usage endpoint; database = server - embedding - re-ranking (BM25 and vector statements, RRF and filters).

| Part | Median ms | p95 ms |
|---|---|---|
| Client wall clock | 1755 | 7080 |
| Server search | 1753 | 7077 |
| Query embedding (API) | 494 | 3812 |
| Database search (Postgres BM25 + vector, fusion) | 276 | 762 |
| Re-ranking (API) | 677 | 922 |

Warm-up query (first search of the process: connection to Neon, catalog facts): 5192 ms wall clock (627 embedding, 1582 database, 1723 re-ranking). Degraded searches: 0.

## Cost per search (reported by the providers, from the usage endpoint)

Mean 0.001500 USD per search (0.0660 USD over 44 searches); 1.0 embedding and 1.0 rerank calls per search.

## nDCG@10 on the V13 judgments (descriptive)

`cloud_stack`: API query embedding (cloud default), Postgres BM25 (`lakebase_text`) and vector search (`halfvec`, the ANN index of the schema), RRF, API reranker over the top 30. Local references from `results_bm25_padding.json`. Holes of the cloud top 10 graded by the judge: 41 (grades 0/1/2/3: 35 / 5 / 1 / 0; 0.0005 USD).

| System | nDCG@10 [95% CI] | Recall@10 | MRR | Results per query | Judged share of returned | nDCG@10 en | nDCG@10 es |
|---|---|---|---|---|---|---|---|
| `cloud_stack` | 0.722 [0.635, 0.799] | 0.473 | 0.902 | 10.0 | 1.000 | 0.639 | 0.823 |
| `vec_e5` | 0.661 [0.585, 0.732] | 0.488 | 0.849 | 10.0 | 1.000 | 0.591 | 0.745 |
| `hybrid_norm_e5_nopad` | 0.622 [0.540, 0.700] | 0.429 | 0.836 | 10.0 | 1.000 | 0.551 | 0.707 |
| `hybrid_norm_e5_bge` | 0.688 [0.613, 0.757] | 0.527 | 0.851 | 10.0 | 1.000 | 0.614 | 0.777 |
| `hybrid_norm_e5_bge_nopad` | 0.660 [0.576, 0.739] | 0.475 | 0.822 | 10.0 | 1.000 | 0.562 | 0.778 |

| Pair | nDCG@10 system | nDCG@10 reference | Difference [95% CI] | p (unadjusted) |
|---|---|---|---|---|
| cloud_stack vs hybrid_norm_e5_bge | 0.722 | 0.688 | +0.034 [-0.027, +0.093] | 0.275 |
| cloud_stack vs hybrid_norm_e5_bge_nopad | 0.722 | 0.660 | +0.062 [+0.003, +0.123] | 0.054 |
| cloud_stack vs vec_e5 | 0.722 | 0.661 | +0.061 [-0.004, +0.127] | 0.079 |

## Limits

- One run from one machine and network; hosted latency depends on the providers' load at that minute.
- 44 queries and 177 books; the added grades come from an LLM judge.
- The cloud stack differs from the local reference in three places at once (embedding model, BM25 engine, reranker), so the difference is not attributable to one of them.

## Reproduce

```
SEARCH_PROFILE=cloud python -m scripts.load_postgres --from data/catalog.db --schema eval177_api
SEARCH_PROFILE=cloud DATABASE_SCHEMA=eval177_api USAGE_ENDPOINT_ENABLED=true uvicorn app.main:app --port 8765
python -m app.evaluation.cloud_live_check run --base-url http://127.0.0.1:8765
python -m app.evaluation.cloud_live_check report   # no API, no judge
```
