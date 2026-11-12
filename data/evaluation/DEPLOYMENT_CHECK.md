# Deployment check of the cloud profile (V17)

Run of 2026-11-11T22:38:37+00:00 against https://ai-book-recommender-one.vercel.app from the measuring machine (Spain; requests enter Vercel's network at `arn1` and run in `fra1`).

## Deployment

- **Public URL**: https://ai-book-recommender-one.vercel.app (web UI at /ui/, MCP at /mcp, API docs at /docs)
- **Date**: 2026-11-11
- **Vercel project**: ai-book-recommender (team luis-projects-f5315dfc, Hobby plan), framework preset FastAPI, no Git repository connected: deployed from the working directory with `vercel deploy --prod`
- **Function**: one Python function from app/main.py (variable app): python3.13, region fra1 (Frankfurt, the region of the Neon database), maxDuration 120 s, memory 2048 MB / 1 vCPU (fixed on Hobby), Fluid compute (default of new projects); bundle 49.8 MB; build 10 s, 95 files uploaded
- **Dependencies**: pyproject.toml (the 66 pins of requirements-cloud.txt); requirements.txt is not uploaded
- **Static files**: app/ui/static is promoted to Vercel's CDN at build time (x-vercel-cache: HIT on /ui/)
- **Deployment protection**: Vercel Authentication (Hobby default) on the per-deployment URLs; the production alias is public
- **Environment variables (production; names only)**: SEARCH_PROFILE, DATABASE_SCHEMA, DATABASE_URL (secret, pooled), OPENROUTER_API_KEY (secret), LLM_PROVIDER, LLM_MODEL, LLM_TIMEOUT_S, LLM_MAX_RETRIES, LLM_CLIENT_MAX_RETRIES, DAILY_SEARCH_LIMIT, DAILY_LLM_LIMIT, BUDGET_SCHEMA, SEARCH_RATE_LIMIT_PER_MINUTE, LLM_RATE_LIMIT_PER_MINUTE, CLIENT_IP_HEADER, MCP_ALLOWED_HOSTS, MCP_ALLOWED_ORIGINS, MCP_RATE_LIMIT_PER_MINUTE, EVALUATE_ENDPOINT_ENABLED, USAGE_ENDPOINT_ENABLED
- **Database**: Neon project twilight-poetry-45310085, branch production, schema catalog_deploy (30000 works) and usage_budget (daily counters); database size 423.9 MiB

## Health

```json
{
  "status": "ok",
  "profile": {
    "name": "cloud",
    "schema": "catalog_deploy",
    "lexical_ranking": "bm25",
    "vector_search": "approximate"
  },
  "database": {
    "reachable": true,
    "latency_ms": 2.7
  },
  "budget": {
    "model_search": 400,
    "llm_request": 60
  },
  "uptime_s": 3.5,
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
| `/ui/` | 200 | 5212 | HIT |
| `/ui/app.js` | 200 | 10551 | HIT |
| `/ui/styles.css` | 200 | 4613 | HIT |
| `/` | 200 | 118 | MISS |

## Queries (round 1)

| Query | Lang. | Kind | Mode | Reranked | Top 3 (language) |
|---|---|---|---|---|---|
| novelas sobre la guerra civil española | es | topic | hybrid | True | Los cipreses creen en Dios (es); A sangre y fuego. Héroes, bestias y mártires de España (es); Las figuras de cera: novela (es) |
| libros de filosofía para principiantes | es | topic | hybrid | True | Introducción a la filosofía (es); Historia de la filosofía sin temor ni temblor (es); Ética para Amador (es) |
| Cien años de soledad | es | exact | hybrid | True | Cien años de soledad (es); La hojarasca (es); Vivir para contarla (es) |
| science fiction novels about artificial intelligence | en | topic | hybrid | True | Robopocalypse (en); Neuromancer (en); The Invincible (pl) |
| history of the Roman Empire | en | topic | hybrid | True | The History of the Decline and Fall of the Roman Empire (en); Annals (la); Römische Geschichte (de) |
| Jane Austen | en | exact | hybrid | True | The Complete Project Gutenberg Works of Jane Austen: A Linked Index of all PG Editions of Jane Austen (en); A Memoir of Jane Austen (en); Pride and Prejudice (en) |
| romans policiers qui se déroulent à Paris | fr | topic | hybrid | True | L'Ombre chinoise (fr); La Tête d'un homme (fr); The Mysteries of Paris (fr) |
| Romane über den Ersten Weltkrieg | de | topic | hybrid | True | Clérambault (en); All quiet on the Western Front (en); Im Westen nichts Neues (de) |
| romanzi storici ambientati a Firenze | it | topic | hybrid | True | Romola (en); Historias florentinas (it); Florentine Histories (it) |
| romances sobre o mar e os pescadores | pt | topic | hybrid | True | The Sound of Waves (ja); O Filho do Pescador (pt); Pêcheur d'Islande (fr) |
| manual de reparación del motor de un Seat Ibiza de 2004 | es | no_answer | hybrid | True | PANA (es); Frascati Manual (en); Manual de Merck (es) |
| how to configure a Kubernetes ingress controller | en | no_answer | hybrid | True | Fundamentación de un modelo de gestión del conocimiento para la Universidad Católica Luis Amigó (es); Análisis estratégico en salud y gestión a través de la escucha (es); La gobernanza de los puertos atlánticos, siglos xiv-xx : Políticas y estructuras portuarias (es) |

## Warm latency (36 searches: 12 queries x 3 rounds, after one warm-up search; 0 degraded, 0 not 200)

| Part | n | Median ms | p95 ms |
|---|---|---|---|
| Client wall clock | 36 | 1430.2 | 3276.4 |
| Server (latency_ms) | 36 | 1284.3 | 3083.0 |
| Stage `embedding` | 36 | 384.4 | 1346.5 |
| Stage `lexical` | 36 | 33.9 | 468.8 |
| Stage `rerank` | 36 | 545.5 | 2530.6 |
| Stage `vector` | 36 | 19.4 | 175.8 |

## Cold start

The first search after the function and the database had been idle (Neon suspends its compute after 5 minutes), then /health:

| At (UTC) | Idle min | Client wall ms | Server ms | Mode | Stages (ms) | Instance uptime at /health (s) |
|---|---|---|---|---|---|---|
| 2026-11-11T23:00:50+00:00 | 16.0 | 12601.6 | 6654.3 | lexical_only (degraded) | {"lexical": 23.4, "rerank": 590.9} | 10.4 |
| 2026-11-11T23:54:01+00:00 | 53 | 22343.8 | None | GET /api/v1/health (no search) | {} | None |

2026-11-11T23:00:50+00:00: Split from the Vercel runtime log of this request: the function instance started and ran the app's startup (the MCP session manager) about 2.2 s after the request reached Vercel; building the search service took about 3.5 s more (the Postgres connection to the resumed Neon compute, the catalog facts and the categories query, and the LLM client: 0.8 s); then the search itself took the 6.65 s the backend reports: the BM25 query, two embedding attempts that timed out (3 s, then 1 s left of the 5 s deadline after a 1 s wait), the lexical-only fallback (23 ms) and the re-ranking (591 ms). The embedding failure was the provider's (ReadTimeout from OpenRouter/DeepInfra on the first call after the idle period), not the platform's; the search degraded as designed.

2026-11-11T23:54:01+00:00: A second cold search, sent at 23:31 UTC after 30 idle minutes, never reached the function: the request hung on the measuring machine's side (no runtime log entry; the client was stopped after more than 20 minutes), as one earlier check attempt did. The next request, GET /api/v1/health with curl at 23:54 after 53 idle minutes, took 22.3 s: its runtime log shows the instance starting the app at 23:54:03.96 and the LLM client being built at 23:54:22.98, so about 19 s went into the first Postgres query of the search service (the categories) while Neon resumed the suspended compute and the connection was opened (connect timeout 10 s, one retry). V12 measured 1.8 s for the same resume from the local machine; this one is a single observation.

## Book, similar books, explanations

- `GET /api/v1/books/0b15d0a1-8cb5-532c-8746-1907932e2509`: 200 (Cien años de soledad; description source {"source": "wikipedia", "language": "es", "license": "CC BY-SA 3.0", "attribution_url": "https://es.wikipedia.org/wiki/Cien_a%C3%B1os_de_soledad", "auto_generated": false})
- `GET /api/v1/books/{id}/similar`: 200 (El coronel no tiene quien le escriba; El amor en los tiempos del cólera; Memoria de mis putas tristes; Vivir para contarla; The Autumn of the Patriarch)
- Search with explanations: 200, 26277.7 ms, mode hybrid, stages {"lexical": 215.5, "embedding": 1656.3, "vector": 210.6, "rerank": 568.4, "explanations": 23521.6}
  - The Murder Room: grounded=True: El libro pertenece a las categorías de literatura criminal y ficción de misterio [CITE: categories], lo que responde al género de 'novelas de misterio' solicitado. Además, la descripción indica explícitamente que es una novela detectivesca y que la acción transcurre en Londres [CITE: description], coincidiendo con la ambientación londinense pedida. También se confirma que forma parte de una serie detectivesca, reforzando su carácter de misterio [CITE: description].
  - A Silent Witness: grounded=True: La novela es relevante porque pertenece al género de ficción detectivesca y misterio [CITE: categories]. Además, la trama está claramente ambientada en Londres, con escenarios como Hampstead y Millfield Lane [CITE: description], y la ciudad aparece también como tema en las categorías [CITE: categories]. La historia gira en torno a una investigación forense dirigida por el Dr. John Thorndyke [CITE: description], lo que confirma su carácter de novela de misterio.
  - The Man in the Brown Suit: grounded=True: La obra pertenece al género de misterio y ficción detectivesca, como indican sus categorías [CITE: categories]. Además, la descripción confirma que es una novela de misterio [CITE: description] y que parte de la acción está ambientada en Londres, ya que 'The setting for the early chapters is London' [CITE: description]. También se menciona que hay un asesinato en Inglaterra, lo que refuerza el componente de misterio y crimen [CITE: description].

## Disabled endpoints and rate limit

- `POST /api/v1/evaluate`: 404
- `GET /api/v1/usage`: 404
- 25 quick `GET /books/{id}/similar` from this client: 20 answered 200, 5 answered 429 (first at request 21, Retry-After 58 s): "Too many searches from this client: at most 20 per minute. Retry in 58 s."

## Web UI in a browser

Opened https://ai-book-recommender-one.vercel.app/ui/ in a browser, typed "novelas de Gabriel García Márquez" and pressed Buscar:

- First attempt, right after the rate-limit check above: the page showed "Error en la búsqueda: 429: Too many searches from this client: at most 20 per minute. Retry in 39 s." (the limit is visible to a user).
- Second attempt a minute later: "10 resultados en 1401 ms (hybrid) · candidatos BM25 60, vector 60"; the health box read "Índices listos · LLM deepseek/deepseek-v4.1-flash"; the language filter lists the twelve catalog languages; the "Sobre los datos" section is present.
- Attribution under the first three results (with links to the article, the licence deed and the Wikidata item):
  - Cien años de soledad (Gabriel García Márquez, 1967): "Extracto del artículo de Wikipedia (es), CC BY-SA 3.0 · Ficha: Wikidata Q178869"
  - En agosto nos vemos (2024): "Extracto del artículo de Wikipedia (en), CC BY-SA 4.0 · Ficha: Wikidata Q123133089"
  - La mala hora (1962): "Extracto del artículo de Wikipedia (en), CC BY-SA 3.0 · Ficha: Wikidata Q1630561"

## Deployed catalog

Selection: `Score = Wikipedia editions + Gutenberg downloads/1000. Quotas for 30000 works: en 12000, es 7500, fr 1050, de 1050, it 1050, pt 1050, ca 1050, nl 1050, ru 1050, zh 1050, ja 1050, ar 1050. Within a language, works whose language is stated by a source first, then highest score (ties: more description languages, then id); unused places go to the best remaining works of any language.`

| Language of the work | Works |
|---|---|
| en | 12,000 |
| es | 7,231 |
| ru | 1,050 |
| it | 1,050 |
| pt | 1,050 |
| zh | 1,050 |
| ja | 1,050 |
| nl | 1,050 |
| de | 1,050 |
| ar | 1,050 |
| fr | 1,050 |
| ca | 971 |
| la | 134 |
| other (33 languages) | 214 |

| Sources of the work | Works |
|---|---|
| wikidata | 23,821 |
| gutenberg + wikidata | 2,438 |
| gutenberg | 2,095 |
| doab | 1,642 |
| doab + wikidata | 3 |
| doab + gutenberg | 1 |

| Description language | Works with a description in it | Works whose primary description is in it |
|---|---|---|
| en | 19,652 | 15,300 |
| es | 12,596 | 5,727 |
| fr | 8,351 | 1,202 |
| de | 6,282 | 1,083 |
| it | 8,309 | 1,045 |
| pt | 5,617 | 933 |
| ca | 3,400 | 918 |
| nl | 3,803 | 945 |
| ru | 6,779 | 969 |
| zh | 2,104 | 279 |
| ja | 2,328 | 631 |
| ar | 5,338 | 968 |

Descriptions by licence: CC BY-SA 3.0 57,232, CC BY-SA 4.0 23,215, Project Gutenberg catalog (no explicit licence) 2,465, CC0 1.0 1,647.

Licences and attribution per source (verified 2026-11-11, V11): Wikidata CC0 1.0 (no attribution needed); Wikipedia text CC BY-SA 3.0 (via DBpedia) or 4.0 (API), attributed per record with the article URL, shown under every result in the UI with a link to the licence; Project Gutenberg catalog metadata without an explicit licence (attributed with the link to the book); DOAB metadata CC0 1.0 (each record links its DOAB page).

The same counts, the selection rule and the licence of each source are stored in the database (`catalog_deploy.catalog_info`, key `catalog_contents`).

## Footprint and sizes

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

## Budget guard (end to end)

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

`python -m scripts.mcp_demo --url https://ai-book-recommender-one.vercel.app/mcp` (protocol 2026-07-28), run at the end of the check:

```
Connected to https://ai-book-recommender-one.vercel.app/mcp
  server: ai-book-recommender 1.0.0 | protocol: 2026-07-28 | 347 ms

tools/list (164 ms)
  - search_books(query, max_results?, language?, min_year?, max_year?, category?)  read-only=True
      Search this server's book catalog and return the best matches, best first. Retrieval is hybrid: BM25 keywor...
  - get_book(book_id)  read-only=True
      Get the full catalog record of one book by its book_id (a UUID returned by search_books or similar_books): ...
  - similar_books(book_id, max_results?)  read-only=True
      Find books similar in content to a given catalog book (item-to-item recommendation by the cosine similarity...

search_books(query='introduction to machine learning with python', max_results=3)  (2695 ms)
  mode=hybrid reranked=True results=3
     1. The Master Algorithm                                         Pedro Domingos         2015 en  score=0.0274
     2. Artificial Intelligence: A Modern Approach                   Stuart J. Russell      1995 en  score=0.0154
     3. Introduction to Algorithms                                   Clifford Stein         1990 en  score=0.0255

get_book(book_id=86ca64b9-bab6-53e3-bc46-c15205810b10)  (133 ms)
  The Master Algorithm | Pedro Domingos | 2015-01-01 | en | wikidata:Q21428158
  publisher=None isbn13=None pages=None
  description (222 chars): The Master Algorithm: How the Quest for the Ultimate Learning Machine Will Remake Our World is a book by Pe...

similar_books(book_id=86ca64b9-bab6-53e3-bc46-c15205810b10, max_results=3)  (225 ms)
     1. Artificial Intelligence: A Modern Approach                   Stuart J. Russell      1995 en  similarity=0.5980
     2. Weapons of Math Destruction                                  Cathy O'Neil           2016 en  similarity=0.5861
     3. The Alignment Problem                                        Brian Christian        ---- en  similarity=0.5858

get_book(unknown id)  (198 ms)  is_error=True
  Error executing tool get_book: No book with id 00000000-0000-0000-0000-000000000000 in the catalog. Use search_books ...
```

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

Change a setting: `printf '%s' VALUE | vercel env add NAME production --force --yes`, then redeploy.

Reload the catalog (idempotent: unchanged works keep their vectors; `--limit N` loads in stages):

```
python -m scripts.build_catalog select --size 30000 --from data/catalog_core/catalog.db --out data/catalog_deploy/catalog.db
python -m scripts.build_catalog report --out data/catalog_deploy/catalog.db
SEARCH_PROFILE=cloud python -m scripts.load_postgres --from data/catalog_deploy/catalog.db --schema catalog_deploy \
    --no-gin --no-text-bodies --embed-workers 4 --max-database-mib 800
python -m scripts.load_postgres --schema catalog_deploy --report
```

Check the deployment again: `python -m scripts.deployment_check run --base-url https://ai-book-recommender-one.vercel.app`, after 10+ idle minutes `... cold --base-url ...`, then `python -m scripts.deployment_check report`.

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
