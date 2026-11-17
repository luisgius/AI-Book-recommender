# V32 · Redespliegue del sitio público con el código de la rama `tfg/validacion-2026-11`

- **Fecha:** 2026-11-17 (todas las horas en UTC)
- **Estado:** hecha y validada; el sitio público ejecuta el commit `ca2ea97`. No se hizo marcha atrás
- **Capa:** despliegue (Vercel) y comprobación del despliegue (`scripts/deployment_check.py`, sin cambios de código)
- **Autorización:** el autor autorizó el despliegue el 2026-11-17
- **Commit desplegado:** `ca2ea97a15ddb064416545431b4bd7390b79ab00` (HEAD de `tfg/validacion-2026-11`)

## Qué se ha hecho

- Se publicó en producción (`https://ai-book-recommender-one.vercel.app`) el código ya guardado en git de la rama `tfg/validacion-2026-11`. El sitio llevaba desde el 2026-11-12 con el despliegue de V18. Con este redespliegue llegan al sitio público:
  - la página web de V19 y V26 tal como está en el commit;
  - el asistente conversacional de V20 (`POST /api/v1/assistant/chat`);
  - la petición de embedding de la consulta con respaldo a un segundo proveedor (hedge), la caché de vectores de consulta y la restricción a proveedores sin retención de datos, todo de V24;
  - el modo de candidatos híbrido como valor por defecto (V30).
- Despliegue `dpl_HfWkUj7afK4jNFh9ZaDEcvLQr8cf` (`https://ai-book-recommender-1n8emdp1r-luis-projects-f5315dfc.vercel.app`), creado a las 14:52:16 y listo a las 14:52:49. El alias de producción apunta a él. Función de 47.57 MB en `fra1` y 37.45 MiB de despliegue.
- Se repitió la comprobación del despliegue de V17 y V18 con el mismo script (`scripts.deployment_check run`), y también la medida de latencia pública. Además se comprobaron el asistente y el servidor MCP.
- Se regeneró `data/evaluation/DEPLOYMENT_CHECK.md` con la orden del proyecto (`scripts.deployment_check report`). La ejecución de V18 se guardó como columna de comparación (`previous_latency`) y se añadió la sección «Redeployment 2026-11-17 (V32)».

## Por qué

- La URL pública seguía con la configuración de V18: búsqueda híbrida con una sola petición de embedding y sin asistente ni página nueva. V24 y V30 dejaron pendiente desplegar y comprobar `/health` (`profile.candidates`) desde la región de la función.
- Se desplegó desde una copia limpia porque el árbol de trabajo tenía cambios sin guardar de otra sesión (`app/ui/static/*`, `tests/api/test_ui.py` y otros) que no se debían publicar.

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| `vercel deploy --prod` desde la raíz del repositorio | Habría subido los cambios sin guardar de la página web de otra sesión |
| `git archive HEAD` como en V18 | Equivale a lo hecho. Se eligió `git worktree` porque el encargo lo pedía y porque permite comprobar con `git status` que la copia está limpia |
| Escribir un script propio para la medida de latencia | Las cifras no serían comparables con las de V17 y V18. Se usó el mismo script sin cambios |
| Medir solo la latencia y saltarse la búsqueda con explicaciones y la ráfaga del límite por cliente | El script lo hace todo en una ejecución y V17 lo comprobó así. Esas dos partes cuestan una petición de LLM y ninguna búsqueda con modelo |
| Marcha atrás al despliegue anterior | No hacía falta: ningún error en `/api/v1`, ninguna búsqueda degradada y los fallos de `/mcp` ya ocurrían antes (véase «Validación») |

## Comprobaciones previas al despliegue (reglas de la tarea)

- **Esquema y datos de la base de datos.** Entre el commit de V18 (`04569cf`) y `ca2ea97` no cambia nada en `app/infrastructure/db/`, `app/infrastructure/usage_budget.py`, los repositorios de búsqueda de Postgres, `vercel.json`, `.vercelignore` ni `pyproject.toml` (`git diff --stat 04569cf..HEAD`). El código nuevo no necesita tablas ni columnas nuevas, y no se tocó Neon.
- **Variables de entorno.** `vercel env ls production` (solo nombres) muestra los mismos 20 nombres que V17 y V18. `SEARCH_CANDIDATES` **no** está definida, así que no hay ningún `vector` que anule el valor por defecto híbrido. Tampoco hay `EMBEDDING_*`: los valores por defecto del código, en el perfil nube y con el proveedor fijado, activan el hedge (Nebius a 1.0 s, DeepInfra otra vez a 2.5 s), la caché y la restricción de retención (`app/infrastructure/config/settings.py`, como dice V24). No se añadió, cambió ni quitó ninguna variable.
- **Tests sobre el árbol del commit.** 1894 pasados y 22 omitidos (los de integración con Neon, porque no había conexión) en 81 s, con el `.venv` del repositorio.
- **Importación de lo que se sube.** Se copió `app/` sin `app/evaluation` junto con `pyproject.toml` y `vercel.json`, que es el conjunto de `.vercelignore`. Esa copia se importó con `SEARCH_PROFILE=cloud`: 15 rutas, entre ellas `/api/v1/assistant/chat`. `app.evaluation` solo se importa dentro de `POST /evaluate`, que está apagado.

## Estado antes (14:49)

- `vercel ls`: producción en `ai-book-recommender-a66qlahvk` (`dpl_D8cAmq5hHoeWrHAThWXniQXroMkE`, creado el 2026-11-12 a las 09:29 UTC, el redespliegue de V18).
- `GET /api/v1/health`: 200, `overall` true, perfil `cloud`, esquema `catalog_deploy`, presupuesto 400 / 60, LLM `deepseek/deepseek-v4.1-flash`, reranker `qwen/qwen3-reranker-8b`. **No había campo `profile.candidates`**, porque el código era anterior a V24.
- `/ui/`: 5212 bytes, SHA-256 `857816af…` (la página de V18).
- `POST /api/v1/assistant/chat`: 404 (no existía). `GET /api/v1/usage`: 404 (apagado).

## Validación (después)

| Comprobación | Resultado |
|---|---|
| `GET /api/v1/health` (14:52:57 y 14:58:23) | 200, `overall` true, `profile.candidates` **hybrid**, esquema `catalog_deploy`, búsqueda léxica bm25, vectorial approximate, presupuesto `model_search` 400 y `llm_request` 60, LLM `deepseek/deepseek-v4.1-flash`, reranker listo. Base de datos alcanzable en 1.4 y 2.1 ms. La primera llamada tardó 3.0 s (instancia nueva, 9.0 s de vida) |
| `/ui/` | 200, 24237 bytes, SHA-256 `910fb1fc…`, idéntico a `git show HEAD:app/ui/static/index.html`. `app.js` (54725 bytes) y `styles.css` (22530 bytes) también son idénticos a HEAD. Los cambios sin guardar de otra sesión **no** están publicados |
| `POST /api/v1/search` (campo `text`) | 38 búsquedas de la comprobación con 200: calentamiento, 36 medidas y la búsqueda con explicaciones |
| Búsqueda con explicaciones | 200 en 4.2 s (2.7 s de explicaciones), tres explicaciones con `grounded` true. En V18 fueron 14.1 s |
| Libro y similares | `GET /books/{id}` 200 (Cien años de soledad), `GET /books/{id}/similar` 200 (cinco obras de García Márquez) |
| `POST /api/v1/assistant/chat` | 200 en 6.5 s, ruta `search`, tres recomendaciones (The Murder Room; A Silent Witness; The Dark Eyes of London), 2 llamadas al LLM, 1520 + 762 tokens, 0.00072 USD notificados |
| `/mcp` (`python -m scripts.mcp_demo`) | Protocolo 2026-07-28, tres herramientas de solo lectura, `search_books` híbrida con reranker (1255 ms), `get_book` (231 ms), `similar_books` (119 ms) e id desconocido devuelto como error de herramienta |
| Endpoints apagados | `POST /evaluate` 404, `GET /usage` 404 |
| Límite por cliente | 25 `GET /similar` seguidos: 20 con 200 y 5 con 429 (el primero en la petición 21, Retry-After 57 s), igual que en V17 y V18 |

### Latencia pública (mismo procedimiento que V17 y V18)

36 búsquedas: 12 consultas por 3 rondas, tras una búsqueda de calentamiento y con 3.2 s entre búsquedas, desde la máquina de medida (España, entrada `arn1`, ejecución `fra1`). Ejecución de 14:53:18 a 14:57:37, con el servicio ya caliente por el `/health` de las 14:52:57.

| Parte | V18 mediana ms | V18 p95 ms | V32 mediana ms | V32 p95 ms |
|---|---|---|---|---|
| Reloj del cliente | 1120.7 | 2040.4 | 1319.8 | 3464.7 |
| Servidor (`latency_ms`) | 960.4 | 1918.0 | 1151.6 | 2245.6 |
| Etapa `embedding` | 400.1 | 1021.5 | 237.3 | 681.0 |
| Etapa `lexical` | 31.6 | 54.6 | 36.5 | 473.2 |
| Etapa `rerank` | 479.4 | 675.1 | 540.5 | 2046.4 |
| Etapa `vector` | 17.2 | 24.8 | 18.4 | 241.4 |
| Búsquedas degradadas | 0 de 36 | - | 0 de 36 | - |
| Respuestas distintas de 200 | 0 | - | 0 | - |

- **Caché de vectores.** La ronda 2 entera salió de la caché de V24 (etapa `embedding` de 0.0 ms en las 12 búsquedas). Las rondas 1 y 3 llamaron al proveedor. La mediana de `embedding` no es comparable con la de V18. En las 24 búsquedas que llamaron al proveedor fue de 262.7 ms.
- **De dónde sale el p95 más alto.** Del reranker: p95 de 2046.4 ms frente a 675.1 ms en V18. En una búsqueda (ronda 2, «Jane Austen», 5148.9 ms) el reranker agotó dos veces el tiempo. El registro de Vercel dice «Re-ranking failed, keeping the fused order: POST rerank: ReadTimeout (after 2 attempts)», y los resultados salieron en el orden de la fusión. La API **no** marca ese caso como degradado. Por eso «0 degradadas» no lo cuenta: fue 1 de 36 búsquedas sin reranking. Los p95 altos de `lexical` (473.2 ms) y `vector` (241.4 ms) vienen de la primera búsqueda de una instancia nueva (conexión nueva a Postgres).
- **Cliente frente a servidor.** El p95 del cliente (3464.7 ms) corresponde a una búsqueda cuyo servidor tardó 1266.1 ms. Los 2.2 s restantes se fueron en la red o en la plataforma, fuera de la función.
- **Conclusión.** No hay errores ni búsquedas degradadas nuevas, y la mediana queda en el mismo orden que en V17 y V18. La cola es más alta por un proveedor externo (el reranker) en una sola medida de 36 búsquedas. Con el criterio del encargo (errores o búsquedas degradadas que el despliegue anterior no tenía) no procedía la marcha atrás.

### Registro de ejecución de Vercel

- Despliegue nuevo, de 14:52 a 14:58: ningún error en `/api/v1`. Los únicos avisos son los dos del reranker citados arriba.
- `POST /mcp`: tres peticiones terminaron con «Vercel Runtime Timeout Error: Task timed out after 120 seconds» (iniciadas a las 14:53:19, 14:53:50 y 14:54:39). No las hizo esta comprobación: la demo MCP corrió a las 14:58 y terminó bien. El despliegue **anterior** registró el mismo error cada pocos minutos desde las 14:37, antes de tocar nada, después de un «StreamableHTTP session manager started». Es otro cliente MCP que deja abierta una petición hasta el límite de 120 s de la función. No lo causa el código nuevo. El plan Hobby solo guarda unos 30 minutos de registro, así que no se pudo ver desde cuándo ocurre.

## Gasto y presupuesto diario

- No se pudo leer el contador del día antes ni después: `/api/v1/usage` está apagado en producción (404), `/health` solo da los límites y la tarea no permite usar la conexión a la base de datos (secreto). Las cifras siguientes son estimaciones a partir de lo enviado.
- Unidades del presupuesto diario que gastó V32: unas 40 búsquedas con modelo (37 de la comprobación, la búsqueda con explicaciones, la búsqueda MCP y la búsqueda del asistente) y 2 peticiones de LLM (explicaciones y asistente). Los límites son 400 y 60. Las 25 peticiones a `/similar` no llaman a ningún modelo.
- Gasto en modelos: unos 0.07 USD. Sale de 40 búsquedas a 0.0015-0.0017 USD (V13, y 0.8981 USD en 528 llamadas al reranker en las evaluaciones del asistente) más unos 0.002 USD de LLM (0.00072 USD notificados por el asistente). La petición de reranking que agotó el tiempo puede facturarse igualmente. El gasto quedó por debajo de 0.10 USD, pero no «muy por debajo»: casi todo es el coste fijo de las 37 búsquedas del procedimiento de medida.

## Órdenes exactas

Desde la raíz del repositorio, salvo indicación:

```
vercel ls
vercel inspect https://ai-book-recommender-a66qlahvk-luis-projects-f5315dfc.vercel.app
vercel env ls production                      # solo nombres
curl -s https://ai-book-recommender-one.vercel.app/api/v1/health
curl -s https://ai-book-recommender-one.vercel.app/ui/ | shasum -a 256
git show HEAD:app/ui/static/index.html | shasum -a 256
git diff --stat 04569cf..HEAD -- app/ vercel.json pyproject.toml .vercelignore requirements-cloud.txt

S=/private/tmp/claude-501/deploy-2026-11-17
git worktree add --detach $S/deploy HEAD
mkdir -p $S/deploy/.vercel && cp .vercel/project.json $S/deploy/.vercel/
(cd $S/deploy && caffeinate -i <raíz del repositorio>/.venv/bin/python -m pytest -q)   # 1894 pasados, 22 omitidos
(cd $S/deploy && caffeinate -i vercel deploy --prod --yes)

# comprobación (el JSON de V18 se conservó antes en previous_latency)
caffeinate -i .venv/bin/python -m scripts.deployment_check run --base-url https://ai-book-recommender-one.vercel.app --catalog-works 50000
curl -s -X POST https://ai-book-recommender-one.vercel.app/api/v1/assistant/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Recomiéndame novelas de misterio ambientadas en Londres"}],"max_results":3}'
.venv/bin/python -m scripts.mcp_demo --url https://ai-book-recommender-one.vercel.app/mcp
vercel logs --deployment dpl_HfWkUj7afK4jNFh9ZaDEcvLQr8cf --since 2026-11-17T14:52:00Z --json -n 1000
vercel logs --since 30m --json -n 1000 -q 'requestPath:/mcp'
.venv/bin/python -m scripts.deployment_check report
git worktree remove $S/deploy
```

Marcha atrás, si hiciera falta: `vercel promote ai-book-recommender-a66qlahvk-luis-projects-f5315dfc.vercel.app` (o `vercel rollback`).

## Publicación de la rama y primera ejecución de la integración continua

El mismo 2026-11-17 se publicó en GitHub la rama `tfg/validacion-2026-11`, hasta entonces solo local, con el commit `ca2ea97`. Fue la primera vez que se ejecutó el flujo `.github/workflows/tests.yml`:

- Ejecución 44107471816 de GitHub Actions, sobre `ca2ea97`, con resultado `success`.
- Trabajo `pytest` en `ubuntu-latest` con Python 3.13: de 14:50:14 a 14:55:18 UTC (5 min 4 s).
- Paso «Run the tests» (`make test`): de 14:50:29 a 14:55:07 UTC (4 min 38 s). Línea final de pytest: `1894 passed, 22 skipped, 9 warnings in 224.41s (0:03:44)`.

Los datos proceden de la API de GitHub (`gh api repos/luisgius/AI-Book-recommender/actions/runs/44107471816/jobs`) y del registro de la ejecución (`gh run view 44107471816 --log`).

## Límites

- Una sola medida de 36 búsquedas en un momento del día. La diferencia con V18 (p95 del cliente de 3.46 s frente a 2.04 s) depende de la carga de los proveedores y no es una comparación controlada.
- La caché de vectores hace que la ronda 2 no sea comparable con las mismas rondas de V17 y V18.
- `degraded` no cuenta las búsquedas en que falla el reranker. Una de las 36 salió sin reranking.
- No se repitieron el arranque en frío, la prueba de extremo a extremo del presupuesto ni la comprobación en un navegador.
- El gasto y el uso del presupuesto son estimaciones (contador del día no leído).
- Las tablas «Queries (round 1)», «Health» y «Web UI» de `DEPLOYMENT_CHECK.md` ya son las de V32. Las de V18 siguen en el historial de git (`04569cf`) y en la columna de comparación. El título de la tabla de comparación lo pone el script y sigue diciendo «enlargement».

## Para decidir el autor

- **Timeouts de `/mcp`.** Algún cliente MCP (quizá una sesión local configurada con el servidor `book-recommender` contra la URL pública) deja abiertas peticiones de 120 s cada pocos minutos. No rompe nada, pero gasta tiempo de función. Conviene ver de dónde vienen.
- **Búsquedas sin reranking.** Si se quiere, podrían contarse como degradadas o avisarse en la respuesta. Sería un cambio de código, fuera de esta tarea.

## Ficheros

- `docs/notes/tareas/V32-redespliegue-2026-11-17.md` (esta nota).
- `data/evaluation/results_deployment_check.json` (resultados de V32, V18 en `previous_latency`, sección nueva y datos del despliegue).
- `data/evaluation/DEPLOYMENT_CHECK.md` (regenerado con `scripts.deployment_check report`).
