# Deployment check of the cloud profile (V17; catalog enlarged to 50,000 works in V18; redeployed in V32)

Run of 2026-11-17T14:53:18+00:00 against https://ai-book-recommender-one.vercel.app from the measuring machine (Spain; requests enter Vercel's network at `arn1` and run in `fra1`).

## Deployment

- **Public URL**: https://ai-book-recommender-one.vercel.app (web UI at /ui/, MCP at /mcp, API docs at /docs)
- **Date**: 2026-11-11 (V17, 30,000 works); 2026-11-12 (V18: catalog enlarged to 50,000 works, then redeployed for the web UI text); 2026-11-17 (V32: redeploy of commit ca2ea97 of branch tfg/validacion-2026-11, which adds the web UI of V19/V26 as committed, the assistant of V20, the query-embedding hedge, cache and zero-retention routing of V24 and the hybrid candidate default of V30)
- **Vercel project**: ai-book-recommender (team luis-projects-f5315dfc, Hobby plan), framework preset FastAPI, no Git repository connected: deployed from the working directory with `vercel deploy --prod`
- **Function**: one Python function from app/main.py (variable app): python3.13, region fra1 (Frankfurt, the region of the Neon database), maxDuration 120 s, memory 2048 MB / 1 vCPU (fixed on Hobby), Fluid compute (default of new projects); V17 bundle 49.8 MB, build 10 s, 95 files uploaded. V18 redeploy on 2026-11-12 (deployment dpl_D8cAmq5hHoeWrHAThWXniQXroMkE, build 37 s) from a clean export of the last commit plus the V18 changes under app/, because the working directory held other uncommitted work; the previous production deployment (ai-book-recommender-e1wtlhals) remains available for a rollback. V32 redeploy on 2026-11-17 at 14:52 UTC (deployment dpl_HfWkUj7afK4jNFh9ZaDEcvLQr8cf, ai-book-recommender-1n8emdp1r, bundle 47.57 MB, deploy command 41 s) from a clean git worktree of HEAD plus .vercel/project.json; the previous production deployment (dpl_D8cAmq5hHoeWrHAThWXniQXroMkE, ai-book-recommender-a66qlahvk) remains available for a rollback
- **Dependencies**: pyproject.toml (the 66 pins of requirements-cloud.txt); requirements.txt is not uploaded
- **Static files**: app/ui/static is promoted to Vercel's CDN at build time (x-vercel-cache: HIT on /ui/)
- **Deployment protection**: Vercel Authentication (Hobby default) on the per-deployment URLs; the production alias is public
- **Environment variables (production; names only)**: SEARCH_PROFILE, DATABASE_SCHEMA, DATABASE_URL (secret, pooled), OPENROUTER_API_KEY (secret), LLM_PROVIDER, LLM_MODEL, LLM_TIMEOUT_S, LLM_MAX_RETRIES, LLM_CLIENT_MAX_RETRIES, DAILY_SEARCH_LIMIT, DAILY_LLM_LIMIT, BUDGET_SCHEMA, SEARCH_RATE_LIMIT_PER_MINUTE, LLM_RATE_LIMIT_PER_MINUTE, CLIENT_IP_HEADER, MCP_ALLOWED_HOSTS, MCP_ALLOWED_ORIGINS, MCP_RATE_LIMIT_PER_MINUTE, EVALUATE_ENDPOINT_ENABLED, USAGE_ENDPOINT_ENABLED. Unchanged in V32 (the same 20 names, checked with `vercel env ls production`): SEARCH_CANDIDATES and the EMBEDDING_* settings are not set, so the code defaults apply (hybrid candidates; hedge to Nebius after 1.0 s, query-vector cache, zero-retention routing)
- **Database**: Neon project twilight-poetry-45310085 (Free plan, storage limit 1 GiB per branch), branch production, schema catalog_deploy (50000 works; 30000 in V17) and usage_budget (daily counters); database size 626.4 MiB (423.9 MiB with 30,000 works)

## Health

```json
{
  "status": "ok",
  "profile": {
    "name": "cloud",
    "candidates": "hybrid",
    "schema": "catalog_deploy",
    "lexical_ranking": "bm25",
    "vector_search": "approximate"
  },
  "database": {
    "reachable": true,
    "latency_ms": 1.9
  },
  "budget": {
    "model_search": 400,
    "llm_request": 60
  },
  "uptime_s": 30.4,
  "components": {
    "lexical_search": true,
    "vector_search": true,
    "embeddings_store": true
  },
  "llm": {
    "configured": true,
    "provider": "openrouter",
    "model": "deepseek/deepseek-v4.1-flash"
  },
  "reranker": {
    "configured": true,
    "model": "qwen/qwen3-reranker-8b",
    "ready": true
  },
  "overall": true
}
```

## Web UI

| Path | Status | Bytes | Vercel cache |
|---|---|---|---|
| `/ui/` | 200 | 24237 | HIT |
| `/ui/app.js` | 200 | 54725 | HIT |
| `/ui/styles.css` | 200 | 22530 | HIT |
| `/` | 200 | 118 | MISS |

## Queries (round 1)

| Query | Lang. | Kind | Mode | Reranked | Top 3 (language) |
|---|---|---|---|---|---|
| novelas sobre la guerra civil española | es | topic | hybrid | True | Los cipreses creen en Dios (es); A sangre y fuego. Héroes, bestias y mártires de España (es); Las figuras de cera: novela (es) |
| libros de filosofía para principiantes | es | topic | hybrid | True | Introducción a la filosofía (es); Historia de la filosofía sin temor ni temblor (es); Ética para Amador (es) |
| Cien años de soledad | es | exact | hybrid | True | Cien años de soledad (es); Vivir para contarla (es); La hojarasca (es) |
| science fiction novels about artificial intelligence | en | topic | hybrid | True | Virtual Girl (en); Dreaming Metal (en); Robopocalypse (en) |
| history of the Roman Empire | en | topic | hybrid | True | The History of the Decline and Fall of the Roman Empire (en); The History of Rome, Books 09 to 26 (en); Annals (la) |
| Jane Austen | en | exact | hybrid | True | The Complete Project Gutenberg Works of Jane Austen: A Linked Index of all PG Editions of Jane Austen (en); A Memoir of Jane Austen (en); Jane Austen, Her Life and Letters: A Family Record (en) |
| romans policiers qui se déroulent à Paris | fr | topic | hybrid | True | Maigret en meublé (fr); Maigret and the Hotel Majestic (fr); L'Ombre chinoise (fr) |
| Romane über den Ersten Weltkrieg | de | topic | hybrid | True | All quiet on the Western Front (en); Clérambault (en); Above the Battle (fr) |
| romanzi storici ambientati a Firenze | it | topic | hybrid | True | Romola (en); Marietta de' Ricci, ovvero Firenze al tempo dell'assedio (it); Historias florentinas (it) |
| romances sobre o mar e os pescadores | pt | topic | hybrid | True | Pêcheur d'Islande (fr); O Filho do Pescador (pt); Blow The Man Down: A Romance Of The Coast (en) |
| manual de reparación del motor de un Seat Ibiza de 2004 | es | no_answer | hybrid | True | PANA (es); The Driver's Seat (en); Life and Death of a Spanish Town (en) |
| how to configure a Kubernetes ingress controller | en | no_answer | hybrid | True | Análisis estratégico en salud y gestión a través de la escucha (es); The Control Revolution (en); Investigación aplicada en Ingeniería (es) |

## Warm latency (36 searches: 12 queries x 3 rounds, after one warm-up search; 0 degraded, 0 not 200)

| Part | n | Median ms | p95 ms |
|---|---|---|---|
| Client wall clock | 36 | 1319.8 | 3464.7 |
| Server (latency_ms) | 36 | 1151.6 | 2245.6 |
| Stage `embedding` | 36 | 237.3 | 681.0 |
| Stage `lexical` | 36 | 36.5 | 473.2 |
| Stage `rerank` | 36 | 540.5 | 2046.4 |
| Stage `vector` | 36 | 18.4 | 241.4 |

## Warm latency before and after the enlargement

Same queries, rounds and client; runs: 30,000 works (V17) at 2026-11-11T22:38:37+00:00; 50,000 works (V18) at 2026-11-12T09:30:00+00:00; redeploy 2026-11-17 (V32) at 2026-11-17T14:53:18+00:00.

| Part | 30,000 works (V17): median ms | p95 ms | 50,000 works (V18): median ms | p95 ms | redeploy 2026-11-17 (V32): median ms | p95 ms |
|---|---|---|---|---|---|---|
| Client wall clock | 1430.2 | 3276.4 | 1120.7 | 2040.4 | 1319.8 | 3464.7 |
| Server (latency_ms) | 1284.3 | 3083.0 | 960.4 | 1918.0 | 1151.6 | 2245.6 |
| Stage `embedding` | 384.4 | 1346.5 | 400.1 | 1021.5 | 237.3 | 681.0 |
| Stage `lexical` | 33.9 | 468.8 | 31.6 | 54.6 | 36.5 | 473.2 |
| Stage `rerank` | 545.5 | 2530.6 | 479.4 | 675.1 | 540.5 | 2046.4 |
| Stage `vector` | 19.4 | 175.8 | 17.2 | 24.8 | 18.4 | 241.4 |
| Degraded of searches | 0 of 36 | - | 0 of 36 | - | 0 of 36 | - |

## Cold start

The first search after the function and the database had been idle (Neon suspends its compute after 5 minutes), then /health:

| At (UTC) | Idle min | Client wall ms | Server ms | Mode | Stages (ms) | Instance uptime at /health (s) | Catalog works |
|---|---|---|---|---|---|---|---|
| 2026-11-11T23:00:50+00:00 | 16.0 | 12601.6 | 6654.3 | lexical_only (degraded) | {"lexical": 23.4, "rerank": 590.9} | 10.4 | 30000 |
| 2026-11-11T23:54:01+00:00 | 53 | 22343.8 | None | GET /api/v1/health (no search) | {} | None | 30000 |
| 2026-11-12T09:47:17+00:00 | 12.6 | 12829.9 | 6575.5 | lexical_only (degraded) | {"lexical": 21.6, "rerank": 673.6} | 10.8 | 50000 |

2026-11-11T23:00:50+00:00: Split from the Vercel runtime log of this request: the function instance started and ran the app's startup (the MCP session manager) about 2.2 s after the request reached Vercel; building the search service took about 3.5 s more (the Postgres connection to the resumed Neon compute, the catalog facts and the categories query, and the LLM client: 0.8 s); then the search itself took the 6.65 s the backend reports: the BM25 query, two embedding attempts that timed out (3 s, then 1 s left of the 5 s deadline after a 1 s wait), the lexical-only fallback (23 ms) and the re-ranking (591 ms). The embedding failure was the provider's (ReadTimeout from OpenRouter/DeepInfra on the first call after the idle period), not the platform's; the search degraded as designed.

2026-11-11T23:54:01+00:00: A second cold search, sent at 23:31 UTC after 30 idle minutes, never reached the function: the request hung on the measuring machine's side (no runtime log entry; the client was stopped after more than 20 minutes), as one earlier check attempt did. The next request, GET /api/v1/health with curl at 23:54 after 53 idle minutes, took 22.3 s: its runtime log shows the instance starting the app at 23:54:03.96 and the LLM client being built at 23:54:22.98, so about 19 s went into the first Postgres query of the search service (the categories) while Neon resumed the suspended compute and the connection was opened (connect timeout 10 s, one retry). V12 measured 1.8 s for the same resume from the local machine; this one is a single observation.

2026-11-12T09:47:17+00:00: V18, 50,000 works, after the redeploy. Neon's control plane reported the compute suspended since 09:39:47 UTC (last active 09:34:13), and /health right after the search reported an instance uptime of 10.8 s, so both the function instance and the database were cold. The search degraded to lexical-only as in the first V17 observation: the embedding request timed out twice at the provider ("POST embeddings: ReadTimeout (after 2 attempts)"), the lexical query took 21.6 ms and the re-ranking 673.6 ms. Wall clock 12.8 s against 12.6 s in V17. This is a single observation, and the Vercel runtime log was not read to split the start-up time.

## Book, similar books, explanations

- `GET /api/v1/books/0b15d0a1-8cb5-532c-8746-1907932e2509`: 200 (Cien años de soledad; description source {"source": "wikipedia", "language": "es", "license": "CC BY-SA 3.0", "attribution_url": "https://es.wikipedia.org/wiki/Cien_a%C3%B1os_de_soledad", "auto_generated": false})
- `GET /api/v1/books/{id}/similar`: 200 (El coronel no tiene quien le escriba; El amor en los tiempos del cólera; Memoria de mis putas tristes; Vivir para contarla; The Autumn of the Patriarch)
- Search with explanations: 200, 4223.6 ms, mode hybrid, stages {"lexical": 233.0, "embedding": 562.6, "vector": 97.2, "rerank": 508.0, "explanations": 2725.0}
  - The Murder Room: grounded=True: El libro pertenece a los géneros de literatura criminal y ficción de misterio, lo que coincide con la búsqueda de 'novelas de misterio'. Además, la descripción indica explícitamente que es una novela detectivesca y que su acción transcurre en Londres, satisfaciendo el criterio de ambientación londinense.
  - A Silent Witness: grounded=True: El libro es relevante porque pertenece al género de misterio y detectives, y está ambientado en Londres, como indica explícitamente la categoría 'London (England)'. Además, la descripción lo confirma como una novela detectivesca y menciona escenarios londinenses como 'London’s lanes and lodgings'.
  - The Dark Eyes of London: grounded=True: La obra pertenece al género de misterio y detectives, tal como se indica en sus categorías. Además, la trama se desarrolla en Londres, ya que la ciudad aparece tanto en las categorías como en el título. La descripción confirma que se trata de una novela de crimen con una investigación policial, elementos centrales del misterio.

## Disabled endpoints and rate limit

- `POST /api/v1/evaluate`: 404
- `GET /api/v1/usage`: 404
- 25 quick `GET /books/{id}/similar` from this client: 20 answered 200, 5 answered 429 (first at request 21, Retry-After 57 s): "Too many searches from this client: at most 20 per minute. Retry in 57 s."

## Web UI in a browser

Observed with 30,000 works (V17, 2026-11-11); not repeated in a browser in V18. After the V18 redeploy, `/ui/` serves "Sobre los datos" with "unas 50 000 obras" (checked with curl).

Opened https://ai-book-recommender-one.vercel.app/ui/ in a browser, typed "novelas de Gabriel García Márquez" and pressed Buscar:

- First attempt, right after the rate-limit check above: the page showed "Error en la búsqueda: 429: Too many searches from this client: at most 20 per minute. Retry in 39 s." (the limit is visible to a user).
- Second attempt a minute later: "10 resultados en 1401 ms (hybrid) · candidatos BM25 60, vector 60"; the health box read "Índices listos · LLM deepseek/deepseek-v4.1-flash"; the language filter lists the twelve catalog languages; the "Sobre los datos" section is present.
- Attribution under the first three results (with links to the article, the licence deed and the Wikidata item):
  - Cien años de soledad (Gabriel García Márquez, 1967): "Extracto del artículo de Wikipedia (es), CC BY-SA 3.0 · Ficha: Wikidata Q178869"
  - En agosto nos vemos (2024): "Extracto del artículo de Wikipedia (en), CC BY-SA 4.0 · Ficha: Wikidata Q123133089"
  - La mala hora (1962): "Extracto del artículo de Wikipedia (en), CC BY-SA 3.0 · Ficha: Wikidata Q1630561"

## Deployed catalog

Selection: `Score = Wikipedia editions + Gutenberg downloads/1000. Quotas for 50000 works: en 20000, es 12500, fr 1750, de 1750, it 1750, pt 1750, ca 1750, nl 1750, ru 1750, zh 1750, ja 1750, ar 1750. Within a language, works whose language is stated by a source first, then highest score (ties: more description languages, then id); unused places go to the best remaining works of any language.`

The 50,000-work selection contains every one of the 30,000 works deployed before (V17), unchanged; the 20,000 added works are new. The source catalog has only 7,231 Spanish, 971 Catalan and 1,617 Chinese works, all of them selected, so the unused places of those quotas (6,181) went to the best remaining works of any language (French 1,349, Latin 977, English 924, German 308, other languages 2,623).

| Language of the work | 30,000 (V17) | 50,000 (V18) |
|---|---|---|
| en | 12,000 | 20,924 |
| es | 7,231 | 7,231 |
| fr | 1,050 | 3,099 |
| de | 1,050 | 2,058 |
| it | 1,050 | 1,750 |
| pt | 1,050 | 1,750 |
| nl | 1,050 | 1,750 |
| ru | 1,050 | 1,750 |
| ja | 1,050 | 1,750 |
| ar | 1,050 | 1,750 |
| zh | 1,050 | 1,617 |
| la | 134 | 977 |
| ca | 971 | 971 |
| other | 214 (33 languages) | 2,623 (55 languages) |

| Sources of the work | 30,000 (V17) | 50,000 (V18) |
|---|---|---|
| wikidata | 23,821 | 40,794 |
| gutenberg | 2,095 | 4,106 |
| gutenberg + wikidata | 2,438 | 3,198 |
| doab | 1,642 | 1,897 |
| doab + wikidata | 3 | 4 |
| doab + gutenberg | 1 | 1 |

| Description language | With a description in it, 30,000 | 50,000 | Primary description in it, 30,000 | 50,000 |
|---|---|---|---|---|
| en | 19,652 | 33,633 | 15,300 | 28,165 |
| es | 12,596 | 14,688 | 5,727 | 6,265 |
| fr | 8,351 | 11,887 | 1,202 | 2,932 |
| de | 6,282 | 8,747 | 1,083 | 2,157 |
| it | 8,309 | 11,660 | 1,045 | 2,065 |
| pt | 5,617 | 6,750 | 933 | 1,157 |
| ca | 3,400 | 3,859 | 918 | 938 |
| nl | 3,803 | 4,299 | 945 | 1,011 |
| ru | 6,779 | 9,005 | 969 | 1,752 |
| zh | 2,104 | 2,859 | 279 | 804 |
| ja | 2,328 | 3,026 | 631 | 1,032 |
| ar | 5,338 | 7,534 | 968 | 1,722 |

Descriptions by licence (50,000): CC BY-SA 3.0 74,435, CC BY-SA 4.0 36,941, Project Gutenberg catalog (no explicit licence) 4,666, CC0 1.0 1,905.

Licences and attribution per source (verified 2026-11-11, V11): Wikidata CC0 1.0 (no attribution needed); Wikipedia text CC BY-SA 3.0 (via DBpedia) or 4.0 (API), attributed per record with the article URL, shown under every result in the UI with a link to the licence; Project Gutenberg catalog metadata without an explicit licence (attributed with the link to the book); DOAB metadata CC0 1.0 (each record links its DOAB page).

The same counts, the selection rule and the licence of each source are stored in the database (`catalog_deploy.catalog_info`, key `catalog_contents`).

## Footprint and sizes

The first load of 30,000 works (V17, 2026-11-11); the V18 enlargement is in the next section.

| Measure | Value |
|---|---|
| Database before the load (after dropping `catalog_small`) | 18.8 MiB (178.6 MiB before) |
| Stage 1 (5,000 works, GIN and texts kept) after VACUUM FULL | 20,401 bytes per work |
| GIN index (only ts_rank_cd uses it) | 13.0 MiB for 5,000 works = 2,726 bytes per work saved |
| Stored description texts in `work_texts.body` | 9.8 MiB for 5,000 works = 2,048 bytes per work saved |
| Dead rows left by the load (insert then tsvector update) | 13.8 MiB for 5,000 works, removed by VACUUM FULL |
| Stage 2 (15,000 works, without GIN and texts) | database 256.2 MiB |
| Stage 3 (30,000 works) before VACUUM FULL | database 462.1 MiB |
| Final, after VACUUM FULL | database 423.9 MiB, schema 405.1 MiB, 14,158 bytes per work |
| Extrapolated saving at 30,000 works | GIN about 78 MiB, texts about 59 MiB |
| Embeddings (qwen/qwen3-embedding-8b, 1024 d, DeepInfra) | 5,603,598 tokens, 0.0560 USD reported by OpenRouter (1,403 requests) |

## Enlargement to 50,000 works (V18)

Plan limit: the Neon project is on the Free plan, whose storage limit is 1 GiB: the Neon API reports `branch_logical_size_limit` 1024 MiB (1,073,741,824 bytes) for this project (`neon projects get twilight-poetry-45310085`), and Neon's plans page (https://neon.com/docs/introduction/plans, updated 2026-11-11) gives 1 GB of storage per project on the Free plan, past which writes that grow storage fail. The project has one branch (production). The load ran under a guard of 780 MiB of `pg_database_size`, checked after every batch of 500 works, so that no batch could take the database past 800 MiB.

Selection: `python -m scripts.build_catalog select --size 50000` contains all 30,000 works deployed in V17, with identical records (their content hashes in the database matched, so none was rewritten or re-embedded); 20,000 works are new. This holds by construction: the quotas only grow with the size, each language's ranking is fixed, and the places left unused grow at least as much as the quotas (test `test_a_larger_selection_contains_every_smaller_one`).

Stages (each run of the loader adds 5,000 new works, keeping every work already loaded, then rebuilds the indexes; sizes after the stage):

| Works | Database MiB | Schema MiB | Schema bytes per work | BM25 index MiB | Vector index MiB | Peak database MiB | Vector index absent (s) | Embedding requests / tokens / USD | Loader run (s) |
|---|---|---|---|---|---|---|---|---|---|
| 30,000 | 424.5 | 405.1 | 14,158 | 56.7 | 88.5 | - | - | - | - |
| 35,000 | 473.9 | 454.4 | 13,614 | 60.6 | 103.2 | - | 8.2 | 220 / 863,272 / 0.0086 | 340.1 |
| 40,000 | 523.5 | 504.1 | 13,215 | 65.1 | 117.9 | - | 8.0 | 234 / 868,464 / 0.0087 | 217.0 |
| 45,000 | 574.9 | 555.6 | 12,946 | 69.6 | 132.6 | - | 10.8 | 227 / 886,078 / 0.0089 | 339.2 |
| 50,000 | 626.4 | 607.0 | 12,731 | 74.1 | 147.3 | 636.3 | 9.6 | 231 / 877,430 / 0.0088 | 222.5 |

- Projection after the first stage: 49.4 MiB for 5,000 added works (10,358 bytes each), so 50,000 works would take about 622 MiB; measured 626.4 MiB. The added works take 10,585 bytes each (201.9 MiB for 20,000), less than the 14,158 bytes of the first 30,000 (fewer description languages per work).
- Peak: the database was largest just before the last index rebuild, 636.3 MiB (the indexes kept up to date by the inserts are about 10 MiB larger than rebuilt ones). The loader keeps the per-batch size log of its last run only, so the peaks inside stages 1 to 3 were not kept; their end sizes rose by about 50 MiB a stage, every batch was checked against the 780 MiB guard, and the guard never stopped a stage.
- Embeddings (qwen/qwen3-embedding-8b, 1024 d, DeepInfra, through OpenRouter): 912 requests, 3,495,244 tokens, 0.0350 USD reported by OpenRouter for the 20,000 added works (V17: 5,603,598 tokens and 0.0560 USD for 30,000).
- Dead rows: the V17 loader inserted each work without its vector and each text row without its tsvector and filled both with UPDATEs (its write counters: 30,000 updates of works, 98,617 of work_texts), one dead row per work and per text row (13.8 MiB for 5,000 works), which needed VACUUM FULL, and VACUUM FULL rewrites a table and needs room for the copy. The V18 loader writes each row once: a new work's vector is computed before the INSERT and goes into it, and each text row is inserted with its tsvector computed in the same statement. Postgres write counters over the whole enlargement: works 20,000 inserts, 0 updates, 0 deletes; work_texts 33,388 inserts, 0 updates, 0 deletes; `n_dead_tup` 0 on both after the load. Only catalog_info (27 rows of facts) was updated. No VACUUM FULL ran; each load ends with a plain VACUUM ANALYZE.
- Live site during the load: no loaded work was deleted or rewritten, and new works became searchable as each batch was committed. Each stage rebuilt the indexes (BM25 statistics are fixed at build time): the BM25 index was built beside the one in use and swapped in one transaction, so the lexical search never lost it (builds of 4.5, 4.3, 5.5 and 5.4 s); the vector index was dropped first, so the database never held two BM25 indexes and the vector index at once, and rebuilt last (3.4, 3.3, 4.4 and 3.9 s). While it was absent, vector searches ran as exact scans, with correct results but slower: 36.6 s in total over the four stages. At 50,000 works the exact scan takes 264 ms in the database, against 4.2 ms with the index. The API treats such a search as a normal hybrid search (an exact scan is not a failure, so nothing is marked degraded); no public search was observed during those windows.
- Idempotence: running the same command again wrote nothing (0 new or changed works, no embedding request, "Indexes are up to date", 21 s).

Database checks at 50,000 works: 50,000 works, 50,000 vectors, 50,000 works with lexical documents (117,947 lexical documents, none with an empty tsvector); every index valid. EXPLAIN ANALYZE of the API's own statements (60 candidates, as deployed): the vector query runs `Index Scan using works_embedding_ann` (4.2 ms), the BM25 query `Index Scan using work_texts_bm25` (18.3 ms).

Redeploy: not needed for the data. The function reads the catalog from the database on every search, and the catalog facts it caches (text-search configurations, at most 12 texts per work) did not change. The four added works below were found through the deployment made on 2026-11-11, before any redeploy. The redeploy was needed only for the static "Sobre los datos" text of the web UI, which said "unas 30 000 obras" and now says "unas 50 000 obras".

Added works searched through the public API (`POST /api/v1/search`, field `text`, 2026-11-12T09:28:57+00:00):

| Query | Language | Rank of the added work | Mode | Server ms | Top 3 |
|---|---|---|---|---|---|
| Zuckerman Bound Philip Roth | en | 1 (Zuckerman Bound, d0ad7ec1) | hybrid | 1115 | Zuckerman Bound; Zuckerman Unbound; The Ghost Writer |
| Une partie de campagne Maupassant | fr | 1 (Une partie de campagne, 9a390924) | hybrid | 985 | Une partie de campagne; Complete Original Short Stories of Guy De Maupassant; Deux amis |
| Zettels Traum Arno Schmidt | de | 1 (Zettels Traum, f9389804) | hybrid | 958 | Zettels Traum; Leviathan; Das steinerne Herz |
| 鹿男あをによし | ja | 1 (鹿男あをによし, 966728f9) | hybrid | 1923 | 鹿男あをによし; 鹿の王; Densha Otoko |

Daily budget used by this V18 work (counter `usage_budget.daily_usage`, 2026-11-12): model searches from 5 to 49 (44 of them by these checks: 4 added-work searches, 38 searches of the deployment check, 1 MCP search and 1 cold-start search), LLM requests from 0 to 1 (the one search with explanations); limits 400 and 60.

## Budget guard (end to end)

Run in V17 (2026-11-11, 30,000 works); the guard's code and settings are unchanged in V18.

Run against a temporary Neon branch (`v17-it`, a copy of production with the deployed catalog) with the API started locally in the cloud configuration and tiny limits (DAILY_SEARCH_LIMIT=2, DAILY_LLM_LIMIT=1, BUDGET_SCHEMA=usage_budget_e2e), real OpenRouter calls; the production day's budget was not touched.

| Step | Status | Mode | Notices | Explanations | Re-ranked | Wall ms |
|---|---|---|---|---|---|---|
| 1 plain search | 200 | hybrid | - | 0 | True | 4579 |
| 2 search with explanations | 200 | hybrid | - | 3 | True | 10193 |
| 3 plain search after the search budget | 200 | lexical_only | search_budget_spent | 0 | False | 379 |
| 4 search with explanations after both budgets | 200 | lexical_only | llm_budget_spent, search_budget_spent | 0 | False | 468 |
| 5 agent after the LLM budget | 429 | - | - | - | - | - |

Agent answer: The day's budget of LLM requests of this public demo is spent, so the agent is not available until 00:00 UTC. Plain search still works.

Counter table afterwards: model_search 2, llm_request 1 (2026-11-11). The integration tests (`tests/integration/test_postgres_usage_budget.py`, same branch) add six instances asking at once: exactly the limit (25 of 60) was granted.

## MCP endpoint

`python -m scripts.mcp_demo --url https://ai-book-recommender-one.vercel.app/mcp` (protocol 2026-07-28), run after the V18 deployment check (50,000 works; "A Human Algorithm" among the similar books is one of the added works):

```
Connected to https://ai-book-recommender-one.vercel.app/mcp
  server: ai-book-recommender 1.0.0 | protocol: 2026-07-28 | 224 ms

tools/list (106 ms)
  - search_books(query, max_results?, language?, min_year?, max_year?, category?)  read-only=True
      Search this server's book catalog and return the best matches, best first. Retrieval is hybrid: BM25 keywor...
  - get_book(book_id)  read-only=True
      Get the full catalog record of one book by its book_id (a UUID returned by search_books or similar_books): ...
  - similar_books(book_id, max_results?)  read-only=True
      Find books similar in content to a given catalog book (item-to-item recommendation by the cosine similarity...

search_books(query='introduction to machine learning with python', max_results=3)  (2383 ms)
  mode=hybrid reranked=True results=3
     1. The Master Algorithm                                         Pedro Domingos         2015 en  score=0.0253
     2. Artificial Intelligence: A Modern Approach                   Stuart J. Russell      1995 en  score=0.0154
     3. Introduction to Algorithms                                   Clifford Stein         1990 en  score=0.0156

get_book(book_id=86ca64b9-bab6-53e3-bc46-c15205810b10)  (119 ms)
  The Master Algorithm | Pedro Domingos | 2015-01-01 | en | wikidata:Q21428158
  publisher=None isbn13=None pages=None
  description (222 chars): The Master Algorithm: How the Quest for the Ultimate Learning Machine Will Remake Our World is a book by Pe...

similar_books(book_id=86ca64b9-bab6-53e3-bc46-c15205810b10, max_results=3)  (123 ms)
     1. A Human Algorithm                                            Unknown                ---- en  similarity=0.6247
     2. Artificial Intelligence: A Modern Approach                   Stuart J. Russell      1995 en  similarity=0.5980
     3. Weapons of Math Destruction                                  Cathy O'Neil           2016 en  similarity=0.5861

get_book(unknown id)  (109 ms)  is_error=True
  Error executing tool get_book: No book with id 00000000-0000-0000-0000-000000000000 in the catalog. Use search_books ...
```

V17 runs (30,000 works, 2026-11-11), kept for the legacy protocol mode and the MCP Inspector CLI:

Two earlier runs at 20:40 UTC, sent at the same moment in both protocol modes (`--mode legacy` negotiates 2025-11-25), completed every call, but their search degraded to lexical-only during a provider overload (see the last section). The legacy run's first lines:

```
Connected to https://ai-book-recommender-one.vercel.app/mcp
  server: ai-book-recommender 1.0.0 | protocol: 2025-11-25 | 376 ms

```

MCP Inspector CLI:

```
$ npx @modelcontextprotocol/inspector@2.14.0 --cli --server-url https://ai-book-recommender-one.vercel.app/mcp --transport http --method tools/list
  search_books  readOnlyHint=True
  get_book  readOnlyHint=True
  similar_books  readOnlyHint=True
$ ... --method tools/call --tool-name search_books --tool-args-json '{"query":"novelas de caballerías","max_results":3,"language":"es"}'
  isError=False search_mode=hybrid reranked=True
  1. Libros de Rocinante | Unknown | es
  2. Le Romant des Romans | Unknown | es
  3. Romances de materia caballeresca | Martí de Riquer i Morera | es
```

## Protections

| Guard | Setting in production | When it is hit | What it cannot prevent |
|---|---|---|---|
| Daily budget of model-backed searches (Postgres counter, one atomic statement per search, shared by every instance) | DAILY_SEARCH_LIMIT=400 per UTC day | the search runs lexical-only without re-ranking (no model call), `degraded` with the reason and the notice `search_budget_spent`; the UI shows a notice | it does not stop database traffic (lexical searches still wake Neon and use compute hours); MCP searches share the same budget |
| Daily budget of LLM requests | DAILY_LLM_LIMIT=60 per UTC day (one unit per request with understanding or explanations, one per agent question) | understanding and explanations are skipped with the notice `llm_budget_spent`; `POST /agent/ask` answers 429 with a message | an agent question makes several LLM calls for one unit; a budget that cannot be read counts as spent (fail closed) |
| Per-client rate limit, REST | SEARCH_RATE_LIMIT_PER_MINUTE=20 (search, similar books), LLM_RATE_LIMIT_PER_MINUTE=4 (LLM options, agent), client = `x-real-ip` (set by Vercel, not forgeable) | 429 with Retry-After | counters live in each instance's memory: several instances give a client several allowances, and many clients together are not limited; the daily budget is the hard ceiling |
| Per-client rate limit, MCP | MCP_RATE_LIMIT_PER_MINUTE=30, same client header | 429 (JSON-RPC error) | same as above; assistants such as Claude call from shared provider addresses, so their users share one counter |
| Model API deadlines | API_QUERY_* cloud defaults: 3 s per attempt, 1 retry, 5 s per call; LLM 15 s, 1 retry, provider client retries 0 | embedding failure: lexical-only; rerank failure: fused order; LLM failure: results without explanations | a slow provider still costs a search up to about 5 s per call |
| Disabled endpoints | EVALUATE_ENDPOINT_ENABLED=false, USAGE_ENDPOINT_ENABLED=false | 404 | - |
| Function duration | maxDuration 120 s | the platform ends the request | - |

Worst case of the model spending per day with these limits: 400 searches x about 0.0015 USD (embedding and re-ranking, measured in V13) + 60 LLM requests x about 0.002 USD at most (three explanations or an agent question with DeepSeek V4.1 Flash at 0.051 / 0.567 USD per million input / output tokens) = about 0.72 USD; the agent's own searches take search units, so they are inside the 400.
The owner should also set a credit limit on the OpenRouter key.

## Commands

Redeploy (from the repository root, linked to the project in `.vercel/`):

```
vercel deploy --prod --yes
```

When the working directory holds uncommitted work that must not be published, deploy a clean copy instead (as in V18): `git archive HEAD | tar -x -C <empty dir>`, copy the intended changes and `.vercel/` into it, and run the same command there.

Change a setting: `printf '%s' VALUE | vercel env add NAME production --force --yes`, then redeploy.

Reload or enlarge the catalog (idempotent: unchanged works keep their rows and vectors; `--limit N` loads in stages, keeping every work already loaded and adding the first new works by id up to N; each row is written once, so no VACUUM FULL is needed):

```
python -m scripts.build_catalog select --size 50000 --from data/catalog_core/catalog.db --out data/catalog_deploy/catalog.db
python -m scripts.build_catalog report --out data/catalog_deploy/catalog.db
SEARCH_PROFILE=cloud python -m scripts.load_postgres --from data/catalog_deploy/catalog.db --schema catalog_deploy \
    --no-gin --no-text-bodies --embed-workers 4 --max-database-mib 780 --limit 35000
# then --limit 40000, --limit 45000, and once more without --limit; after a sleep or a failure, run the same command again
python -m scripts.load_postgres --schema catalog_deploy --report
```

Check the deployment again: `python -m scripts.deployment_check run --base-url https://ai-book-recommender-one.vercel.app --catalog-works 50000`, after 10+ idle minutes `... cold --base-url ... --idle-minutes N --catalog-works 50000`, then `python -m scripts.deployment_check report`.

Switch it off:

```
vercel project pause ai-book-recommender      # visitors get an error page; undo: vercel project resume ai-book-recommender
vercel project delete ai-book-recommender     # removes the project and its deployments
python -m scripts.load_postgres --schema catalog_deploy --drop   # drops the deployed catalog (direct connection of .env)
python -m scripts.load_postgres --schema usage_budget --drop     # drops the daily counters
```

## Behaviour on Vercel compared with local runs

- Provider overload: at 20:40 UTC two MCP searches sent at the same moment (the demo client in both protocol modes) degraded to lexical-only after 11 s, because the pinned embedding provider (DeepInfra) answered HTTP 429 "Model busy, retry later" and then timed out (Vercel runtime log: "Hybrid search failed, falling back to lexical: POST embeddings: ... (after 2 attempts)"). The re-ranking still ran. A REST search a minute later was hybrid, and none of the 36 measured searches degraded. Locally (V13) the same provider was seen busy; the deadlines now bound the wait to about 5 s per call instead of minutes.
- LLM explanations are slow on the deployment: three explanations took 23.5 s in sequence (about 8 s each with DeepSeek V4.1 Flash on OpenRouter), inside LLM_TIMEOUT_S=15 and the 120 s function limit.
- The function runs in fra1 next to the database: `SELECT 1` from /health takes 1.5 to 2.4 ms. Requests from the measuring machine enter Vercel's network at arn1 (Stockholm) and are routed to fra1 (`x-vercel-id: arn1::fra1::...`).
- Fluid compute often serves a request with a recently started instance (/health reported uptimes of a few seconds between checks, and a /health a minute after the second cold observation took 5.8 s on a new instance), so the first request on each instance pays the app start and a new Postgres connection, and in-memory rate-limit counters reset often: the per-client limit triggered at the 21st request of 25 sent within seconds, as configured, but it is not a guarantee across instances.
- The web UI files are copied to Vercel's CDN at build time (`x-vercel-cache: HIT` on /ui/); the API and /mcp run in the function.
- Queries with no good answer in the catalog (an engine repair manual, Kubernetes) still return ten results (the vector side always finds neighbours); their top results are unrelated, as expected.
- Requests from the measuring machine sometimes hung on its side: one check attempt aborted on a client TCP timeout (Errno 60) in round 2, another stalled, and a cold-start search never reached Vercel. The script now records a client network error and goes on; the run reported here had none.
- Before a demonstration, open the page (or call /api/v1/health) a minute ahead: the first request after an idle period paid 12.6 s and 22.3 s in the two cold observations.
- `vercel link` appended `.vercel` and `.env*` to .gitignore and wrote `.env.local` (a Vercel OIDC token, already ignored by `.env.local`); the .gitignore lines were removed again, so `.vercel/` (project and team ids, no secret) shows as untracked and must not be committed.
- V18: with 50,000 works the warm searches were not slower than with 30,000 (table above). The database stages stayed in the same range (vector median 17.2 ms, BM25 31.6 ms), and the totals were lower, mostly because the model calls were faster on that day. The embedding median was 15.7 ms higher. The two runs are 11 hours apart and both depend on the providers' load, so this is not a controlled comparison.
- V18: the search with explanations took 14.1 s (12.6 s of explanations), against 26.3 s in V17.

## Redeployment 2026-11-17 (V32)

Redeploy of the committed code of branch tfg/validacion-2026-11 at commit ca2ea97, authorised by the owner on 2026-11-17. The working directory held another session's uncommitted edits to the web UI, so the deployment ran from a clean `git worktree add --detach <scratch>/deploy HEAD` with only `.vercel/project.json` copied in. No database schema or data change and no environment variable change: since the V18 deployment (commit 04569cf) nothing under app/infrastructure/db/, the usage budget, vercel.json, .vercelignore or pyproject.toml changed, and the new settings have code defaults. Before deploying, the committed tree passed the test suite (1894 passed, 22 skipped) and the uploaded file set (app/ without app/evaluation, pyproject.toml, vercel.json) imported with SEARCH_PROFILE=cloud.

| Check | Before (V18 deployment dpl_D8cAmq5h..., 14:49 UTC) | After (V32 deployment dpl_HfWkUj7a..., 14:52-14:58 UTC) |
|---|---|---|
| GET /api/v1/health | 200, overall true; no `profile.candidates` field (older code) | 200, overall true, `profile.candidates` hybrid, schema catalog_deploy, budget 400 / 60, LLM deepseek/deepseek-v4.1-flash, reranker qwen/qwen3-reranker-8b ready |
| /ui/ | 5212 bytes, SHA-256 857816af... (V18 page) | 24237 bytes, SHA-256 910fb1fc..., identical to `git show HEAD:app/ui/static/index.html`; app.js (54725 bytes) and styles.css (22530 bytes) also identical to HEAD |
| POST /api/v1/assistant/chat | 404 (endpoint not deployed) | 200 in 6.5 s, route search, 3 recommendations (The Murder Room; A Silent Witness; The Dark Eyes of London), 2 LLM calls, 0.00072 USD reported |
| /mcp (`python -m scripts.mcp_demo`) | - | protocol 2026-07-28, three tools, search_books hybrid and re-ranked (1255 ms), get_book, similar_books, unknown id answered as a tool error |

Warm latency: the table "Warm latency" above and the third column of the comparison table are this run (same script, queries, rounds, spacing and client as V17 and V18; the heading of the comparison table is the script's and still says "enlargement"). 36 of 36 searches answered 200 in hybrid mode, none degraded. Client wall clock median 1319.8 ms, p95 3464.7 ms (V18: 1120.7 and 2040.4); server median 1151.6 ms, p95 2245.6 ms (V18: 960.4 and 1918.0).

- Round 2 was answered from the query-vector cache of V24 (embedding stage 0.0 ms in all 12 searches of that round; rounds 1 and 3 called the provider), so the embedding figures are not comparable with V18 (median of the 24 searches that called the provider: 262.7 ms).
- The higher p95 comes from the re-ranking provider: rerank p95 2046.4 ms against 675.1 ms in V18. In one search (round 2, "Jane Austen", 5148.9 ms) the reranker timed out twice (Vercel runtime log: "Re-ranking failed, keeping the fused order: POST rerank: ReadTimeout (after 2 attempts)") and the results kept the fused order. The API does not mark this as degraded, so the 0 degraded above does not count it; it is 1 of 36 searches without re-ranking.
- Search with explanations: 4.2 s (2.7 s of explanations) against 14.1 s in V18.
- Runtime log of the new deployment from 14:52 to 14:58 UTC: no error on /api/v1; three POST /mcp requests ended with "Task timed out after 120 seconds". They came from another MCP client, not from this check, and the previous deployment logged the same error every few minutes from 14:37 UTC on, before the redeploy, so it is not caused by the new code. Vercel's Hobby plan keeps runtime logs for about 30 minutes, so the earlier history could not be read.
- Rollback, if needed: `vercel promote ai-book-recommender-a66qlahvk-luis-projects-f5315dfc.vercel.app` (or `vercel rollback`). Not needed.
- Daily budget used by the V32 checks (estimate; /api/v1/usage is off and the counter table was not read): about 40 model-backed searches (37 of the deployment check, the search with explanations, the MCP search, the assistant's search) and 2 LLM requests, against limits of 400 and 60. Model spend about 0.07 USD at the measured 0.0015 to 0.0017 USD per search plus about 0.002 USD of LLM calls.
- Not repeated in V32: the cold start, the budget guard end to end and a check in a browser.
