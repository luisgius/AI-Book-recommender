# V17 · Despliegue público: catálogo de 30 000 obras en Neon y backend en Vercel, con límites de gasto

- **Fecha:** 2026-11-12 (código en el commit `df2d890`; las medidas del despliegue se tomaron el 2026-11-11, en UTC)
- **Estado:** hecha y validada; la ampliación del catálogo pedida el 2026-11-12 está en curso (ver "Trabajo pendiente")
- **Capa:** infraestructura (`app/infrastructure/`), dominio (`app/domain/`), aplicación (`app/api/`), interfaz (`app/ui/static/`), scripts, configuración de despliegue

## Qué se ha hecho

- El perfil nube del buscador de libros se ha publicado en una dirección pública que el tribunal puede abrir sin instalar nada: `https://ai-book-recommender-one.vercel.app`. La interfaz web está en `/ui/`, la API REST en `/api/v1/` (documentación en `/docs`) y el servidor MCP (Model Context Protocol, el protocolo con que un asistente de IA llama a herramientas externas) en `/mcp`.
- El backend corre como una única función de Python en Vercel (plataforma de alojamiento sin servidor), en la región `fra1` (Fráncfort, la misma que la base de datos). La base de datos es Postgres gestionado en Neon, en su plan gratuito, y contiene un catálogo de 30 000 obras seleccionadas de las fuentes de V11.
- Se ha escrito la configuración de despliegue (`vercel.json`, `.vercelignore`, `pyproject.toml`) para que el despliegue se pueda repetir desde la línea de órdenes.
- Se ha reducido el tamaño del catálogo en la base de datos hasta que cupiera en el plan gratuito (14 158 bytes por obra).
- Se han añadido protecciones del gasto, porque la dirección es pública y cada búsqueda llama a modelos de pago: un presupuesto diario guardado en Postgres, límites de peticiones por cliente, plazos cortos para las llamadas a modelos con degradación a búsqueda léxica, y endpoints costosos desactivados.
- La interfaz muestra ahora el origen y la licencia de cada descripción, y avisa cuando una búsqueda se degrada por presupuesto.
- Se ha escrito un script de comprobación del despliegue (`scripts/deployment_check.py`) y se ha medido el sistema desplegado. El informe es `data/evaluation/DEPLOYMENT_CHECK.md`.

## Por qué

El autor decidió el 2026-11-11 que el sistema debe vivir también en la nube (V12 y propuesta `docs/notes/2026-11-11-propuesta-nube-y-mcp.md`) y que los examinadores deben poder probarlo sin instalarlo; V09, la ejecución en un comando, sigue pendiente. Tomó estas decisiones, que aquí se citan como las dio:

1. Sobre la base de datos: "neon seems better for a TFG plus is cheaper". El proyecto de Neon existe en la región de Fráncfort, en el plan gratuito.
2. Sobre el alojamiento del backend: "Vamos con Vercel, lo mejor son opciones gratuitas o de muy bajo coste".
3. Sobre el método: "lets go with it, but we measure and test". Por eso cada pieza de esta tarea se ha medido, y la comprobación del despliegue es parte del resultado.

El objetivo de coste fijo cero se apoya en dos planes gratuitos: el plan Hobby de Vercel y el plan gratuito de Neon. El único gasto variable son las llamadas a modelos por OpenRouter, que el presupuesto diario acota (ver más abajo). Una dirección pública abierta a cualquiera sin esa cota permitiría a un tercero gastar sin límite, de modo que las protecciones forman parte del despliegue y no un añadido.

## Objetivo

1. Publicar el perfil nube en una dirección estable y pública, sin coste fijo, con la interfaz, la API y el servidor MCP.
2. Hacer que un catálogo de 30 000 obras quepa en el plan gratuito de Neon (1 GB por proyecto, según V12). V12 había fijado esa cifra como decisión: en el plan gratuito, el catálogo desplegado es una selección de unas 30 000 obras.
3. Acotar el gasto máximo diario en modelos y hacer que, cuando falle o se agote un servicio de pago, la búsqueda se degrade en lugar de bloquearse.
4. Medir el sistema desplegado: salud, latencia en caliente y en frío, explicaciones, MCP y protecciones.

## Qué se ha construido

### Configuración de Vercel

- `vercel.json`: región `fra1`, una función (`app/main.py`) con `maxDuration` de 120 segundos y `git.deploymentEnabled` en `false`.
  - La región es la de la base de datos, de modo que cada consulta a Postgres no sale de una ciudad. Desde `/health`, `SELECT 1` tarda de 1.5 a 2.4 ms. Las peticiones desde la máquina de medida entran en la red de Vercel en `arn1` (Estocolmo) y se encaminan a `fra1`.
  - 120 segundos queda por debajo del techo de 300 s del plan Hobby y por encima de la petición legítima más lenta (una búsqueda con tres explicaciones del LLM). La memoria no se fija: en Hobby es fija (2048 MB, 1 vCPU) y Fluid compute (el modo por defecto de los proyectos nuevos, que reutiliza instancias entre peticiones) no admite fijarla en `vercel.json`.
  - Los despliegues por Git están desactivados: publicar es siempre un paso deliberado, `vercel deploy --prod`, desde el directorio del repositorio. El proyecto (`ai-book-recommender`, plan Hobby, preajuste FastAPI) no tiene ningún repositorio de Git conectado.
- `.vercelignore` es una lista de permitidos: se sube solo `app/`, `pyproject.toml` y `vercel.json`. Quedan fuera `data/` (varios GB de catálogos, índices y salidas de evaluación), `tests/`, `docs/`, `review/`, `scripts/`, `requirements.txt` y todo fichero local, y dentro de `app/` queda fuera el paquete `evaluation`, que solo sirve a `POST /evaluate`, desactivado en producción. Resultado medido: 95 ficheros subidos, paquete de 49.8 MB, construcción de 10 s, Python 3.13.
- Los ficheros estáticos de la interfaz (`app/ui/static`) se copian a la CDN de Vercel en la construcción (`x-vercel-cache: HIT` en `/ui/`); la API y `/mcp` se ejecutan en la función.
- Protección de despliegues: la autenticación de Vercel (por defecto en Hobby) cubre las direcciones de cada despliegue; el alias de producción es público.

### Lista de dependencias del perfil nube

Vercel busca las dependencias primero en `pyproject.toml`, luego en `Pipfile` y por último en `requirements*.txt`. Si solo existiera `requirements.txt`, la función instalaría el fichero completo del trabajo local (`pip freeze`, con torch, FAISS y sentence-transformers, varios GB), que no cabe en una función. Por eso `pyproject.toml` lista las 66 versiones fijadas de `requirements-cloud.txt` (las del perfil nube, sin bibliotecas de modelos, creado en V13), y `tests/test_requirements_cloud.py` falla si las dos listas difieren. La regla de mantenimiento es editar `requirements-cloud.txt` y copiar sus versiones a `pyproject.toml`. `requirements.txt` no se sube.

### Selección de 30 000 obras y carga en Neon

- La selección sale del catálogo de fuentes múltiples (`data/catalog_core/catalog.db`) con `python -m scripts.build_catalog select --size 30000`. Cada obra pasa antes el filtro de calidad (título, autor y una descripción de al menos 200 caracteres en uno de los doce idiomas admitidos).
- Regla de selección: puntuación igual a ediciones de Wikipedia más descargas de Gutenberg divididas entre 1000, con cuotas por idioma (inglés 12 000, español 7500 y 1050 para cada uno de fr, de, it, pt, ca, nl, ru, zh, ja y ar). Dentro de cada idioma van primero las obras cuyo idioma declara una fuente y después las de mayor puntuación (desempate: más idiomas de descripción, luego el identificador); los puestos que quedan sin usar los ocupan las mejores obras restantes de cualquier idioma.
- Composición real, del informe del catálogo:

| Idioma de la obra | Obras |
|---|---|
| en | 12 000 |
| es | 7231 |
| ru, it, pt, zh, ja, nl, de, ar, fr (cada uno) | 1050 |
| ca | 971 |
| la | 134 |
| otros (33 idiomas, según el informe de despliegue) | 214 |

  El español y el catalán no llegan a su cuota (7231 de 7500 y 971 de 1050); el informe no indica el motivo.

| Fuentes de la obra | Obras |
|---|---|
| wikidata | 23 821 |
| gutenberg + wikidata | 2438 |
| gutenberg | 2095 |
| doab | 1642 |
| doab + wikidata | 3 |
| doab + gutenberg | 1 |

- Cobertura de campos: año 70.6 %, materias 62.3 %, `isbn13` 6.0 %, identificador de Wikidata 87.5 %. 17 423 obras (58.1 %) tienen descripciones en más de un idioma, y 2245 descripciones principales son resúmenes generados automáticamente por Project Gutenberg y marcados como tales.
- Licencias de las descripciones: CC BY-SA 3.0 (57 232), CC BY-SA 4.0 (23 215), catálogo de Project Gutenberg sin licencia explícita (2465) y CC0 1.0 (1647). Wikidata es CC0 1.0; el texto de Wikipedia es CC BY-SA y se atribuye por registro con la dirección del artículo, y la interfaz lo muestra bajo cada resultado con enlaces al artículo, a la licencia y al elemento de Wikidata. DOAB aporta metadatos CC0 1.0. Los recuentos, la regla y la licencia de cada fuente se guardan también en la base de datos (`catalog_deploy.catalog_info`, clave `catalog_contents`).
- Carga: `scripts/load_postgres.py` calcula los vectores con `qwen/qwen3-embedding-8b` a 1024 dimensiones por la API de OpenRouter (proveedor DeepInfra). La carga se hizo por etapas (5000, 15 000 y 30 000 obras), con un tope declarado de tamaño de la base de datos (`--max-database-mib 800`), y es idempotente: las obras sin cambios conservan sus vectores. Los embeddings costaron 0.0560 USD según OpenRouter (5 603 598 tokens en 1403 peticiones).

### Medidas para que el catálogo cupiera en el plan gratuito

Medidas sobre la etapa de 5000 obras, con el catálogo cargado y tras `VACUUM FULL` (que compacta las tablas):

| Medida | Efecto |
|---|---|
| Etapa 1 de partida (5000 obras, con el índice GIN y los textos almacenados) | 20 401 bytes por obra |
| No crear el índice GIN (solo lo usa `ts_rank_cd`, la ordenación de comparación; la búsqueda léxica de producción usa el índice BM25) | 13.0 MiB en 5000 obras: 2726 bytes por obra |
| No guardar el texto de las descripciones en `work_texts.body` | 9.8 MiB en 5000 obras: 2048 bytes por obra |
| Filas muertas que deja la carga (inserción y luego actualización del `tsvector`) | 13.8 MiB en 5000 obras, eliminadas con `VACUUM FULL` |

Tamaños de la base de datos durante la carga: 18.8 MiB antes de cargar (tras borrar `catalog_small`, que ocupaba 178.6 MiB); 256.2 MiB con 15 000 obras (sin GIN y sin textos); 462.1 MiB con las 30 000 antes de `VACUUM FULL`; y al final 423.9 MiB la base de datos, 405.1 MiB el esquema y 14 158 bytes por obra. El ahorro extrapolado a 30 000 obras es de unos 78 MiB (GIN) y unos 59 MiB (textos). El plan gratuito admite 1 GB por proyecto (V12); la ampliación en curso parte de los 423.9 MiB finales.

El cargador admite para ello las opciones `--no-gin` y `--no-text-bodies`, y almacena cada descripción una sola vez.

### Presupuesto diario guardado en Postgres

- Dos contadores por día UTC y clase: `model_search` (búsquedas que llaman al modelo de embeddings y al reranker; límite 400 al día) y `llm_request` (peticiones que usan el LLM, ya sea la comprensión de la consulta, las explicaciones o una pregunta al agente; límite 60 al día). Viven en un esquema propio (`usage_budget`), distinto del catálogo, de modo que recargar o borrar el catálogo no reinicia la cuenta.
- Un puerto del dominio (`UsageBudget`) y su implementación en Postgres (`app/infrastructure/db/postgres_usage_budget.py`). Tomar una unidad es una sola sentencia atómica (`INSERT ... ON CONFLICT DO UPDATE ... WHERE used < límite`), de modo que todas las instancias de la función comparten la cuenta y el límite no se supera aunque varias pregunten a la vez. El día lo marca el reloj de la base de datos (00:00 UTC), por lo que todas las instancias coinciden.
- Cuando se agota: la búsqueda se ejecuta solo con BM25 y sin reranking (ninguna llamada a un modelo), se marca como degradada con el motivo y el aviso `search_budget_spent`, y la interfaz muestra el aviso. Con el presupuesto del LLM agotado se omiten la comprensión y las explicaciones (aviso `llm_budget_spent`) y `POST /agent/ask` responde 429 con un mensaje.
- Si el contador no se puede leer (base de datos inaccesible), la unidad se trata como agotada (se falla cerrando). Si una conexión muerta obliga a reintentar, una unidad puede contarse dos veces: el error va del lado de gastar menos.
- Límites: no impide el tráfico a la base de datos (las búsquedas léxicas siguen despertando Neon y gastando horas de cómputo); las búsquedas de MCP comparten el mismo presupuesto; una pregunta al agente hace varias llamadas al LLM por una sola unidad, aunque las búsquedas del propio agente consumen unidades de `model_search`.

### Límites por cliente, endpoints desactivados y plazos cortos

| Protección | Ajuste en producción | Al alcanzarse | Lo que no puede evitar |
|---|---|---|---|
| Límite por cliente, REST | `SEARCH_RATE_LIMIT_PER_MINUTE=20` (búsqueda y libros similares), `LLM_RATE_LIMIT_PER_MINUTE=4` (opciones con LLM y agente); el cliente se identifica por `x-real-ip`, que pone Vercel y no es falsificable | 429 con `Retry-After` | Los contadores viven en la memoria de cada instancia: varias instancias dan a un cliente varias cuotas, y muchos clientes juntos no quedan limitados; el techo duro es el presupuesto diario |
| Límite por cliente, MCP | `MCP_RATE_LIMIT_PER_MINUTE=30` | 429 (error JSON-RPC) | Lo mismo; además, asistentes como Claude llaman desde direcciones compartidas del proveedor, de modo que sus usuarios comparten un contador |
| Plazos de las llamadas a la API de modelos | 3 s por intento, 1 reintento, 5 s por llamada; LLM 15 s con 1 reintento y sin reintentos en el cliente del proveedor | Fallo de embeddings: búsqueda solo léxica; fallo del reranker: orden de la fusión; fallo del LLM: resultados sin explicaciones | Un proveedor lento cuesta todavía hasta unos 5 s por llamada |
| Endpoints desactivados | `EVALUATE_ENDPOINT_ENABLED=false`, `USAGE_ENDPOINT_ENABLED=false` | 404 | - |
| Duración de la función | `maxDuration` de 120 s | La plataforma termina la petición | - |

Estos plazos sustituyen a los de V13 (30 s por intento y cuatro intentos), que la decisión 4 de V13 daba por demasiado largos para una búsqueda en vivo. El resto del trabajo en código fue: `app/api/rate_limiting.py`, la inyección del presupuesto en `SearchService`, la lectura de los ajustes en `settings.py` y los cambios en `openrouter_http.py`, `openrouter_embedding_encoder.py` y `api_reranker.py`.

Peor caso del gasto diario en modelos con estos límites: 400 búsquedas por unos 0.0015 USD (embedding y reranking, medido en V13) más 60 peticiones al LLM por unos 0.002 USD como máximo (tres explicaciones o una pregunta al agente con DeepSeek V4.1 Flash, a 0.051 y 0.567 USD por millón de tokens de entrada y de salida), es decir, unos 0.72 USD. Las búsquedas del agente consumen unidades de las 400, de modo que quedan dentro de esa cifra. El informe indica además que el propietario debería fijar un límite de crédito en la clave de OpenRouter; es una medida adicional que no consta en las fuentes como hecha.

### Cambios en la interfaz

- Bajo cada resultado, la fuente y la licencia de la descripción con enlaces al artículo, a la licencia y al elemento de Wikidata.
- Una sección "Sobre los datos" y el filtro de idioma con los doce idiomas del catálogo.
- Los avisos de degradación (`search_budget_spent`, `llm_budget_spent`) y los errores 429 se muestran al usuario (se vio "Too many searches from this client: at most 20 per minute" en el navegador, y a la repetición un minuto después, "10 resultados en 1401 ms (hybrid) · candidatos BM25 60, vector 60").

### Script de comprobación del despliegue

`scripts/deployment_check.py` envía las consultas por HTTP al despliegue real y escribe el informe y los datos (`data/evaluation/DEPLOYMENT_CHECK.md` y `results_deployment_check.json`). Mide la salud, la interfaz, doce consultas en tres rondas (latencia en caliente), el arranque en frío (tras 10 minutos de inactividad o más), un libro, sus similares, las explicaciones, los endpoints desactivados y el límite por cliente. Tras un fallo de red del cliente, registra el error y sigue (varias peticiones desde la máquina de medida se quedaron colgadas de su lado). Tiene un test (`tests/evaluation/test_deployment_check.py`). El MCP se comprobó con `scripts/mcp_demo` y con el Inspector de MCP.

## Resultados medidos

Todo lo siguiente sale de `data/evaluation/DEPLOYMENT_CHECK.md`. La ejecución principal fue del 2026-11-11 a las 22:38 UTC contra la dirección pública, desde una máquina en España. Son medidas descriptivas, sin contraste estadístico ni juicios de relevancia.

### Salud

`GET /api/v1/health`: estado `ok`, perfil `cloud`, esquema `catalog_deploy`, ordenación léxica `bm25`, búsqueda vectorial aproximada, base de datos alcanzable (2.7 ms), presupuestos 400 y 60, LLM configurado (`deepseek/deepseek-v4.1-flash` por OpenRouter), reranker configurado (`qwen/qwen3-reranker-8b`). La interfaz responde 200 en `/ui/` (5212 bytes, `HIT` de la CDN), `/ui/app.js` (10 551) y `/ui/styles.css` (4613).

### Latencia en caliente

36 búsquedas (12 consultas por 3 rondas, tras una búsqueda de calentamiento); ninguna degradada ni distinta de 200.

| Parte | n | Mediana ms | p95 ms |
|---|---|---|---|
| Petición completa (reloj del cliente) | 36 | 1430.2 | 3276.4 |
| Servidor (`latency_ms`) | 36 | 1284.3 | 3083.0 |
| Etapa `embedding` | 36 | 384.4 | 1346.5 |
| Etapa `lexical` | 36 | 33.9 | 468.8 |
| Etapa `rerank` | 36 | 545.5 | 2530.6 |
| Etapa `vector` | 36 | 19.4 | 175.8 |

Las doce consultas cubren temas y títulos o autores exactos en español, inglés, francés, alemán, italiano y portugués, y dos consultas sin respuesta en el catálogo. Todas se ejecutaron en modo híbrido y con reranking. El informe lista los tres primeros resultados de cada una (por ejemplo, "Cien años de soledad" devuelve esa obra en primer lugar), pero no se ha juzgado la relevancia: es una inspección, no una evaluación.

### Arranque en frío

Dos observaciones, una sola medida cada una:

| Momento (UTC) | Inactividad | Reloj del cliente | Servidor | Resultado |
|---|---|---|---|---|
| 2026-11-11 23:00:50 | 16.0 min | 12 601.6 ms | 6654.3 ms | búsqueda solo léxica (degradada) |
| 2026-11-11 23:54:01 | 53 min | 22 343.8 ms | - | `GET /api/v1/health`, sin búsqueda |

- Primera observación: la instancia de la función arrancó la aplicación unos 2.2 s después de llegar la petición; construir el servicio de búsqueda tardó unos 3.5 s más (conexión al cómputo de Neon reanudado, datos del catálogo y consulta de categorías); la búsqueda tardó los 6.65 s que informa el backend: la consulta BM25, dos intentos de embedding que agotaron el plazo (3 s y luego 1 s de los 5 s) por un fallo del proveedor (`ReadTimeout` de OpenRouter/DeepInfra en la primera llamada tras la inactividad), la caída a modo solo léxico (23 ms) y el reranking (591 ms). La búsqueda se degradó como estaba diseñado.
- Segunda observación: una búsqueda fría enviada a las 23:31 tras 30 minutos de inactividad no llegó a la función (la petición quedó colgada del lado de la máquina de medida, sin entrada en el registro de ejecución). La petición siguiente, `GET /api/v1/health` tras 53 minutos, tardó 22.3 s: unos 19 s fueron la primera consulta a Postgres mientras Neon reanudaba el cómputo suspendido (plazo de conexión de 10 s, un reintento). V12 había medido 1.8 s para la misma reanudación desde la máquina local; esta es una sola observación.
- Fluid compute suele atender con instancias recién arrancadas: un `/health` un minuto después de la segunda observación tardó 5.8 s en una instancia nueva.

### Libro, similares, explicaciones

`GET /api/v1/books/{id}` y `.../similar` respondieron 200 (para "Cien años de soledad", con su descripción de Wikipedia en español, CC BY-SA 3.0, y como similares otras obras de García Márquez). Una búsqueda con explicaciones respondió 200 en 26 277.7 ms, modo híbrido; las etapas fueron 215.5 ms léxica, 1656.3 embedding, 210.6 vectorial, 568.4 reranking y 23 521.6 explicaciones. Las tres explicaciones, en secuencia, tardaron unos 8 s cada una con DeepSeek V4.1 Flash por OpenRouter, dentro de `LLM_TIMEOUT_S=15` y del límite de 120 s de la función. Las tres salieron marcadas como ancladas (`grounded=True`) con citas a `categories` y `description`; el informe reproduce los textos.

### Protecciones

- `POST /api/v1/evaluate` y `GET /api/v1/usage` responden 404.
- 25 peticiones rápidas a `GET /books/{id}/similar` desde un cliente: 20 respondieron 200 y 5 respondieron 429, la primera en la petición 21 con `Retry-After` de 58 s. Los contadores son por instancia; el límite saltó como se configuró, pero no es una garantía entre instancias.

### Presupuesto diario, de extremo a extremo

Se ejecutó contra una rama temporal de Neon (`v17-it`, copia de producción con el catálogo desplegado), con la API local en la configuración nube, límites mínimos (`DAILY_SEARCH_LIMIT=2`, `DAILY_LLM_LIMIT=1`, esquema `usage_budget_e2e`) y llamadas reales a OpenRouter; el presupuesto del día de producción no se tocó.

| Paso | Estado | Modo | Avisos | Explicaciones | Reranking | ms |
|---|---|---|---|---|---|---|
| 1 búsqueda simple | 200 | hybrid | - | 0 | sí | 4579 |
| 2 búsqueda con explicaciones | 200 | hybrid | - | 3 | sí | 10 193 |
| 3 búsqueda simple tras agotar las búsquedas | 200 | lexical_only | `search_budget_spent` | 0 | no | 379 |
| 4 búsqueda con explicaciones tras agotar ambos | 200 | lexical_only | `llm_budget_spent`, `search_budget_spent` | 0 | no | 468 |
| 5 agente tras agotar el LLM | 429 | - | - | - | - | - |

El agente respondió que el presupuesto diario de peticiones al LLM de esta demostración pública está agotado hasta las 00:00 UTC y que la búsqueda simple sigue funcionando. La tabla de contadores quedó en `model_search` 2 y `llm_request` 1. Una prueba de integración (`tests/integration/test_postgres_usage_budget.py`, sobre la misma rama) lanza seis instancias a la vez y comprueba que se concede exactamente el límite (25 de 60).

### MCP

`python -m scripts.mcp_demo --url https://ai-book-recommender-one.vercel.app/mcp` (protocolo 2026-07-28; el modo `legacy` negocia 2025-11-25):

- Conexión en 347 ms y `tools/list` en 164 ms, con tres herramientas de solo lectura: `search_books`, `get_book` y `similar_books`.
- `search_books("introduction to machine learning with python", max_results=3)` tardó 2695 ms en modo híbrido con reranking; `get_book` 133 ms; `similar_books` 225 ms. Un identificador desconocido devuelve un error de herramienta con un mensaje claro (198 ms).
- El Inspector de MCP (`@modelcontextprotocol/inspector@2.14.0`) lista las tres herramientas con `readOnlyHint` y una llamada a `search_books` con filtro de idioma español devuelve resultados en modo híbrido.
- Dos ejecuciones anteriores, a las 20:40 UTC y enviadas a la vez en ambos modos de protocolo, completaron todas las llamadas pero sus búsquedas se degradaron a solo léxicas durante una sobrecarga del proveedor (ver "Límites").

### Coste del gasto medido

Embeddings de la carga: 0.0560 USD. Peor caso diario con los límites de producción: unos 0.72 USD (cálculo en la sección anterior).

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Supabase en lugar de Neon | Su plan gratuito es de 500 MB y se pausa tras siete días de poca actividad; el mismo SQL cuesta de 25 a 30 USD al mes (V12 y propuesta). Neon no tiene cuota mínima y se suspende en reposo. El adaptador es SQL estándar, de modo que cambiar después sería cambiar la cadena de conexión |
| El plan de pago de Neon para un catálogo mayor | Los 30 000 obras caben en el plan gratuito; un catálogo mayor exige el plan de pago por uso, que V12 deja como decisión del autor. 200 000 obras de este tipo necesitarían unos 6.2 GiB según la medida de V12, que usó el esquema antiguo con GIN y textos; aquí se midieron 14 158 bytes por obra. La ampliación en curso se hace dentro del plan gratuito |
| Un contenedor único que se apaga cuando no se usa (Cloud Run), con SQLite y FAISS dentro | La propuesta lo dio como alternativa más barata y conservadora, pero no escala bien a un millón de obras (imagen de varios GB, arranque en frío largo) y exige una cuenta de Google Cloud con tarjeta. El autor eligió Vercel |
| Ejecutar los modelos dentro de la función de Vercel | Las bibliotecas de modelos (torch, FAISS, sentence-transformers) ocupan varios GB instalados y una función serverless no puede alojarlas; por eso los embeddings y el reranker van por API y el paquete desplegado pesa 49.8 MB |
| Un solo fichero de dependencias para los dos perfiles (`requirements.txt` en Vercel) | Vercel prioriza `pyproject.toml`; con solo `requirements.txt` instalaría el fichero completo. `pyproject.toml` replica el reducido y un test los contrasta |
| Despliegue automático por cada `git push` | Publicar debe ser un paso deliberado; con una dirección pública y un presupuesto de gasto, un push no debe cambiar lo que ve el tribunal sin que el autor lo decida |
| Confiar solo en límites por cliente en memoria | Cada instancia tiene sus propios contadores: varias instancias dan a un cliente varias cuotas y muchos clientes juntos no se limitan. El techo duro es el presupuesto diario en Postgres, compartido por todas las instancias |
| Que un contador ilegible se trate como disponible | Si la base de datos falla, dejaría pasar gasto sin límite. Se falla cerrando: la unidad se trata como agotada |
| Mantener el índice GIN y el texto almacenado de las descripciones | Cuestan 2726 y 2048 bytes por obra y solo sirven a la ordenación de comparación `ts_rank_cd`; la búsqueda de producción usa el índice BM25 y los textos ya están en el catálogo |
| Mantener los plazos de 30 s por intento y cuatro intentos de V13 | Una búsqueda en vivo esperaría minutos si el proveedor se atascaba; ahora se degrada en unos 5 s por llamada |

## Cómo se ha validado

- Medida del despliegue real (arriba), repetible con `scripts/deployment_check`.
- Presupuesto diario probado de extremo a extremo contra una rama temporal de Neon con llamadas reales, y con una prueba de integración de concurrencia (seis instancias a la vez).
- Tests añadidos en el commit: `tests/api/test_public_deployment_api.py`, `tests/domain/test_usage_budget_in_search.py`, `tests/infrastructure/test_public_deployment_guards.py`, `tests/integration/test_postgres_usage_budget.py`, `tests/integration/test_postgres_cloud_profile.py`, `tests/evaluation/test_deployment_check.py`, `tests/test_requirements_cloud.py` y ampliaciones de los tests del cargador, del catálogo y de la interfaz. Las pruebas de integración se omiten sin cadena de conexión de prueba.

## Criterios de diseño aplicados

1. El perfil nube queda publicado en la dirección pública; el perfil local sigue siendo el evaluado.
2. Las llamadas a modelos de pago se acotan con un presupuesto diario en la base de datos, y lo que supere el presupuesto se degrada a búsqueda léxica en lugar de fallar.
3. Los endpoints `/evaluate` y `/usage` quedan desactivados en producción.
4. Cualquier cambio de ajuste se hace con `vercel env add` y un nuevo despliegue manual.

## Límites que debe conocer el tribunal

- **El perfil desplegado no es el evaluado.** Las cifras de calidad de la memoria (nDCG@10 y demás) proceden del perfil local sobre la colección de 177 libros, con 44 consultas, y la paridad entre perfiles se midió en V12 y V13 sobre esa misma colección. El sistema desplegado, con 30 000 obras y otros modelos (embeddings y reranker por API), no se ha evaluado todavía: las doce consultas del informe son una inspección de los tres primeros resultados, no una medida de calidad. Esa evaluación es V14.
- **Arranque en frío.** La primera petición tras un periodo de inactividad tardó 12.6 s y, en otra observación, 22.3 s; sus causas son el arranque de la aplicación, la reanudación de Neon (se suspende tras 5 minutos de inactividad, no configurable) y, en la primera, un fallo del proveedor de embeddings. Son dos observaciones sueltas, no una distribución. Conviene abrir la página, o llamar a `/api/v1/health`, un minuto antes de una demostración.
- **Las explicaciones del LLM son lentas:** unos 8 s cada una (23.5 s las tres), 26.3 s la petición completa.
- **Sin umbral de "nada relevante".** Las consultas sin buena respuesta en el catálogo (una avería de un Seat Ibiza, la configuración de un controlador de entrada de Kubernetes) devuelven igualmente diez resultados, porque la rama vectorial siempre encuentra vecinos; sus primeros resultados no tienen relación con la consulta. El sistema no tiene un umbral que diga que no hay nada relevante.
- **Sobrecarga del proveedor de embeddings.** El 2026-11-11 a las 20:40 UTC, dos búsquedas MCP enviadas a la vez se degradaron a solo léxicas tras 11 s, porque DeepInfra respondió HTTP 429 ("Model busy, retry later") y luego agotó el plazo; el reranking sí se ejecutó. Una búsqueda REST un minuto después fue híbrida y ninguna de las 36 búsquedas medidas se degradó. Los plazos acotan la espera a unos 5 s por llamada, no la eliminan.
- **Límites por cliente, no garantías.** Los contadores viven en la memoria de cada instancia y se reinician a menudo; el techo duro es el presupuesto diario. El presupuesto no impide el tráfico a la base de datos, y los usuarios de un mismo asistente de IA pueden compartir un contador de MCP.
- **Presupuesto compartido.** Con 400 búsquedas con modelos al día para todos los visitantes, un uso intensivo (por ejemplo, de un tribunal en varios equipos) agota el presupuesto y el resto del día la búsqueda es solo léxica. El peor caso de gasto diario en modelos es de unos 0.72 USD.
- **Medidas desde una máquina y una red.** Las latencias vienen de una máquina en España hacia Fráncfort. Varias peticiones desde esa máquina se quedaron colgadas de su lado (un tiempo de espera TCP del cliente y una búsqueda en frío que no llegó a Vercel); la ejecución que se informa no tuvo ninguna.
- **Catálogo.** Los idiomas con cuota no llenan todos los puestos (español 7231 de 7500, catalán 971 de 1050). Cobertura de `isbn13` del 6.0 % y de año del 70.6 %. 2245 descripciones principales son resúmenes generados automáticamente por Project Gutenberg. Las licencias de las descripciones son CC BY-SA 3.0 o 4.0 (con obligación de compartir igual), CC0 1.0 y, para el catálogo de Gutenberg, sin licencia explícita.
- **Chino y japonés.** V12 documentó que Postgres no trae lematizador para estos idiomas y que su texto no se divide en palabras, de modo que la recuperación léxica en ellos es pobre. Con 1050 obras de cada uno en el catálogo, esa limitación se aplica al sistema desplegado; no se ha medido en este catálogo.
- **Modelo de embeddings alojado.** V13 (con la medida de V16) indica que devuelve vectores ligeramente distintos en llamadas repetidas, de modo que los resultados en vivo no son exactamente repetibles.
- **Notas de operación.** `vercel link` añadió `.vercel` y `.env*` a `.gitignore` y escribió `.env.local` (un token OIDC de Vercel, ya ignorado por `.env.local`); se revirtieron las líneas de `.gitignore`, de modo que `.vercel/` (identificadores de proyecto y equipo, sin secretos) aparece sin seguimiento y no debe versionarse.

## Trabajo pendiente

- **V14**, la evaluación sobre el catálogo desplegado: consultas ampliadas y multilingües, juicios de relevancia y, con ello, la medida de calidad que hoy falta para el sistema de 30 000 obras (incluida la comparación de la búsqueda híbrida y la vectorial en el catálogo grande, que V13 dejó abierta).
- **V09**, la ejecución en un comando (README, Makefile, demostración, integración continua).
- **Ampliación del catálogo.** El 2026-11-12 el autor pidió cargar más obras dentro del plan gratuito de Neon, con 423.9 MiB usados de 1 GB. Se está haciendo como tarea de seguimiento y se añadirá a esta nota como una sección de actualización, con sus medidas.
- Fijar un límite de crédito en la clave de OpenRouter (recomendación del informe, que no consta como hecha).

## Cómo reproducirlo

```
# Seleccionar y cargar el catálogo (idempotente; --limit N carga por etapas)
python -m scripts.build_catalog select --size 30000 --from data/catalog_core/catalog.db --out data/catalog_deploy/catalog.db
python -m scripts.build_catalog report --out data/catalog_deploy/catalog.db
SEARCH_PROFILE=cloud python -m scripts.load_postgres --from data/catalog_deploy/catalog.db --schema catalog_deploy \
    --no-gin --no-text-bodies --embed-workers 4 --max-database-mib 800
python -m scripts.load_postgres --schema catalog_deploy --report

# Desplegar (desde la raíz del repositorio, enlazada al proyecto con `vercel link`)
vercel deploy --prod --yes

# Comprobar el despliegue (el modo cold, tras 10 minutos de inactividad o más)
python -m scripts.deployment_check run --base-url https://ai-book-recommender-one.vercel.app
python -m scripts.deployment_check cold --base-url https://ai-book-recommender-one.vercel.app
python -m scripts.deployment_check report
python -m scripts.mcp_demo --url https://ai-book-recommender-one.vercel.app/mcp
```

- Cambiar un ajuste: `vercel env add NOMBRE production --force --yes` con el valor por la entrada estándar, y desplegar de nuevo. Las cadenas de conexión y la clave de OpenRouter son secretos del entorno de Vercel y del entorno local, y nunca se versionan.
- Apagarlo: `vercel project pause ai-book-recommender` (se deshace con `resume`) o `vercel project delete ai-book-recommender`; y, para la base de datos, `python -m scripts.load_postgres --schema catalog_deploy --drop` y `--schema usage_budget --drop`.
- `report` no necesita red ni base de datos.
- Tests: `pytest tests/api/test_public_deployment_api.py tests/domain/test_usage_budget_in_search.py tests/infrastructure/test_public_deployment_guards.py tests/evaluation/test_deployment_check.py tests/test_requirements_cloud.py`, y la suite completa con `pytest`.

## Ficheros

- **Configuración de despliegue:** `vercel.json`, `.vercelignore`, `pyproject.toml` (espejo de `requirements-cloud.txt`), `.env.example` (ajustes del despliegue público), `.gitignore`.
- **Dominio:** `app/domain/ports.py` (puerto `UsageBudget`), `app/domain/services.py` (el presupuesto y la degradación en `SearchService`), `app/domain/value_objects.py`.
- **Infraestructura:** `app/infrastructure/db/postgres_usage_budget.py`, `app/infrastructure/usage_budget.py`, `app/infrastructure/config/settings.py` (presupuesto, límites por cliente y plazos de la nube), `app/infrastructure/search/openrouter_http.py`, `openrouter_embedding_encoder.py`, `api_reranker.py`, `app/infrastructure/db/postgres_connection.py`, `postgres_schema.py`, `postgres_book_catalog_repository.py`, `app/infrastructure/llm/langchain_llm_client.py`.
- **Aplicación:** `app/api/rate_limiting.py`, `app/api/mcp/rate_limit.py`, `app/api/mcp/server.py`, `app/api/v1/dependencies.py`, `search_endpoints.py`, `agent_endpoints.py`, `evaluation_endpoints.py`, `schemas.py`, `converters.py`.
- **Interfaz:** `app/ui/static/index.html`, `app.js`, `styles.css`.
- **Scripts:** `scripts/load_postgres.py` (opciones `--no-gin`, `--no-text-bodies`, `--max-database-mib`), `scripts/build_catalog.py`, `scripts/deployment_check.py`.
- **Informes:** `data/evaluation/DEPLOYMENT_CHECK.md`, `data/evaluation/results_deployment_check.json`.
- **Tests:** `tests/api/test_public_deployment_api.py`, `tests/api/test_ui.py`, `tests/api/cloud_profile_driver.py`, `tests/domain/test_usage_budget_in_search.py`, `tests/infrastructure/test_public_deployment_guards.py`, `test_load_postgres_script.py`, `test_search_profile_settings.py`, `tests/ingestion/test_build_catalog_script.py`, `tests/integration/test_postgres_usage_budget.py`, `test_postgres_cloud_profile.py`, `tests/evaluation/test_deployment_check.py`, `tests/test_requirements_cloud.py`.
