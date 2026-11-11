# V13 · Integración del perfil nube y corrección del relleno de BM25

- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** dominio (`app/domain/`), infraestructura (`app/infrastructure/`), aplicación (`app/api/v1/`, `app/main.py`), evaluación (`app/evaluation/`), scripts

## Qué se ha hecho

- Se ha corregido el adaptador BM25 local: devuelve solo los libros que contienen al menos un término de la consulta. El comportamiento anterior queda tras la opción `zero_score_padding=True`, que solo usan las rutas de evaluación que reproducen resultados publicados. Un test falla si una ruta de evaluación construye BM25 sin declarar qué comportamiento usa.
- `SearchService` admite una lista léxica corta o vacía sin fallar (tests añadidos).
- Se ha medido el efecto de la corrección con un diseño fijado de antemano (`app/evaluation/bm25_padding.py`, informe en `data/evaluation/BM25_PADDING.md`).
- Se han unido las piezas del perfil nube (adaptadores de Postgres, embeddings y reranking por API) para que el backend corra en una función serverless pequeña que no carga ningún modelo: importaciones perezosas por perfil, `requirements-cloud.txt`, cargador con identidad completa del codificador, método de embeddings por lotes en el puerto, categorías con una sola consulta, `GET /api/v1/usage` y valores por defecto del perfil nube.
- Se ha comprobado el conjunto en vivo contra Neon (`app/evaluation/cloud_live_check.py`, informe en `data/evaluation/CLOUD_LIVE_CHECK.md`).

## Por qué

Las piezas del perfil nube se habían construido y medido por separado (adaptadores de Postgres en V12, embeddings y reranking por API en V16). Había que unirlas para poder desplegar el backend en una función serverless, que no puede alojar las bibliotecas de modelos. Además, las medidas de V12 habían dejado a la vista un defecto del adaptador BM25 local. La regla del autor es "medimos y probamos": ambas cosas se han tratado con medida y con tests, no por criterio.

## Objetivo

1. Corregir el defecto del relleno de BM25 y medir de antemano qué cambia, incluida la pregunta que V06 dejó abierta: si el híbrido mejora a su parte vectorial una vez eliminado el defecto.
2. Dejar el perfil nube listo para desplegar y comprobar, contra la base de datos y las APIs reales, que responde y qué cuesta.

## Parte A: el relleno de BM25

### El defecto y la corrección

El adaptador local `BM25SearchRepository` devolvía siempre `max_results` libros. `rank_bm25` puntúa todos los libros del corpus y la lista se cortaba sin mirar las puntuaciones, de modo que los libros sin ningún término de la consulta (puntuación cero) completaban la lista. En la fusión RRF esos libros entraban como si la rama léxica los hubiera recuperado. Entraron 7.4 libros de relleno por consulta con 20 candidatos y 36.1 por consulta con 60 candidatos (la profundidad del reranker).

La corrección hace que BM25 devuelva solo libros que contienen algún término de la consulta; las listas pueden ser cortas o estar vacías. La opción `zero_score_padding=True` conserva el comportamiento antiguo y no forma parte del índice guardado. La API, el cableado por defecto y los brazos de evaluación nuevos usan el comportamiento corregido. Una repetición completa de la ablación de V06 reprodujo los once sistemas con 44 de 44 listas idénticas, de modo que ninguna cifra publicada cambió.

### Diseño de la medida, fijado de antemano

- Colección de 177 libros, 44 consultas (24 en inglés, 20 en español), profundidad 10, con el mismo arnés de V06 (`retrieval_ablation`). El `SearchService` no cambia: RRF con k = 60, 20 candidatos por rama y 60 con el reranker, que reordena los 30 primeros.
- Los brazos publicados se ejecutan tal como se declararon (con relleno); los brazos `_nopad` son los mismos sistemas con el adaptador corregido.
- Tres contrastes sobre nDCG@10, con corrección de Holm sobre esa familia: F1, híbrido corregido frente a híbrido con relleno; F2, híbrido corregido frente a la parte vectorial sola (la pregunta que V06 dejó abierta); F3, híbrido corregido con el reranker `BAAI/bge-reranker-v2-m3` frente al híbrido con relleno y el mismo reranker.
- Test de aleatorización pareado bilateral y bootstrap percentil pareado del 95 % (10 000 remuestreos, semilla 0). Todo lo demás es descriptivo.
- Juicios: se parte de los de V12 y el mismo juez (`z-ai/glm-5.3-flash`) puntuó 22 pares nuevos (calificaciones 0/1/2/3: 18 / 4 / 0 / 0; 0.0015 USD). Todo libro devuelto por cualquier sistema tiene juicio.

### Resultados

Contrastes declarados de antemano (nDCG@10, sistema menos referencia):

| Contraste | Sistema | Referencia | nDCG@10 sistema | nDCG@10 referencia | Dif. [IC 95 %] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| F1 | `hybrid_norm_e5_nopad` | `hybrid_norm_e5` | 0.622 | 0.592 | +0.029 [+0.006, +0.058] | 0.021 | 0.063 |
| F2 | `hybrid_norm_e5_nopad` | `vec_e5` | 0.622 | 0.661 | -0.039 [-0.089, +0.010] | 0.134 | 0.268 |
| F3 | `hybrid_norm_e5_bge_nopad` | `hybrid_norm_e5_bge` | 0.660 | 0.688 | -0.028 [-0.071, +0.010] | 0.207 | 0.268 |

Longitud de las listas y libros de relleno que entraron en la fusión (descriptivo):

| Profundidad | Relleno medio por consulta | Máximo | Consultas con relleno |
|---|---|---|---|
| 10 (BM25 solo) | 2.7 | 10 | 18 de 44 |
| 20 (candidatos léxicos del híbrido) | 7.4 | 20 | 21 de 44 |
| 60 (con reranker) | 36.1 | 60 | 36 de 44 |

En todas las consultas, la lista corregida es la lista antigua sin esos libros. Sin relleno, BM25 solo devuelve 7.3 resultados por consulta de media; 18 consultas tienen menos de 10 y 4 no tienen ninguno.

BM25 solo y desglose por idioma (descriptivo):

| Sistema | nDCG@10 [IC 95 %] | nDCG@10 en | nDCG@10 es |
|---|---|---|---|
| `bm25_normalized` (con relleno) | 0.489 [0.395, 0.584] | 0.428 | 0.562 |
| `bm25_normalized_nopad` | 0.474 [0.376, 0.572] | 0.410 | 0.549 |
| `vec_e5` | 0.661 [0.585, 0.732] | 0.591 | 0.745 |
| `hybrid_norm_e5` | 0.592 [0.504, 0.679] | 0.532 | 0.665 |
| `hybrid_norm_e5_nopad` | 0.622 [0.540, 0.700] | 0.551 | 0.707 |
| `hybrid_norm_e5_bge` | 0.688 [0.613, 0.757] | 0.614 | 0.777 |
| `hybrid_norm_e5_bge_nopad` | 0.660 [0.576, 0.739] | 0.562 | 0.778 |

BM25 solo pierde 0.016 al quitar el relleno (diferencia -0.016 [-0.032, -0.004], p sin ajustar 0.030, fuera de la familia contrastada). El híbrido con reranker y sin relleno frente a `vec_e5` queda en -0.001 [-0.042, +0.041] (descriptivo).

### Lectura

1. La corrección elimina un defecto real. El híbrido simple gana unos 0.03 de nDCG@10 (+0.029), diferencia que no se sostiene tras la corrección de Holm (p de Holm 0.063).
2. Incluso sin relleno, el híbrido no supera a la parte vectorial sola en esta colección (0.622 frente a 0.661, intervalo que incluye el cero). La respuesta a la pregunta de V06 sigue siendo que no hay evidencia de que el híbrido sea mejor aquí.
3. Detrás de un reranker, la corrección no ayuda: la estimación puntual es negativa (-0.028) con un intervalo que incluye el cero.

Interpretación del diseñador, no verificada: en un catálogo de 177 libros, 36 libros de relleno son una quinta parte del catálogo, de modo que el relleno actuaba como una forma de mostrar al reranker más parte del catálogo. En un catálogo de decenas de miles de obras el mismo relleno sería ruido. Esto debe comprobarse en el catálogo grande, junto con la cuestión de si el conjunto de candidatos que se entrega al reranker debe completarse con la rama vectorial cuando la lista léxica es corta.

## Parte B: el perfil nube, listo para desplegar

1. **Importaciones perezosas por perfil.** Importar la aplicación tarda 0.25 s en los dos perfiles (antes, 2.16 s) y no carga ninguna biblioteca de modelos. Dos tests en subproceso (`tests/api/test_cloud_profile_imports.py`) arrancan la aplicación en la configuración nube con la base de datos y la API de modelos simuladas y comprueban que todos los endpoints responden con `torch`, `faiss`, `sentence_transformers`, `transformers`, `rank_bm25` y `numpy` ausentes; uno de ellos hace imposible importar esos paquetes.
2. **`requirements-cloud.txt`.** 66 paquetes con versión fijada, cada una tomada de `requirements.txt`, que sigue siendo el fichero completo de `pip freeze` del trabajo local, como exige la regla del proyecto. Un test falla ante cualquier paquete que falte o tenga otra versión. Instalado en un entorno virtual nuevo (127 MB), la aplicación sirvió todos los endpoints. No se ha añadido ninguna dependencia.
3. **Cargador de Postgres.** Calcula los embeddings con el codificador configurado (modelo local o API), toma de él la dimensión del vector y registra la identidad completa (modelo, proveedor, dimensión). La búsqueda vectorial se niega a ejecutarse si la identidad del codificador configurado difiere de la almacenada, y nombra ambas. Una segunda carga no recalcula nada.
4. **Embeddings por lotes.** Método `get_embeddings_batch` en el puerto `EmbeddingsStore`, implementado en los almacenes FAISS y Postgres y usado por MMR. Viajes a la base de datos por llamada, medidos: libros similares con MMR, 22 antes y 3 después; búsqueda con MMR, 35 antes y 3 después; búsqueda simple o libros similares simples, 2 antes y después.
5. **Arranque.** Las categorías salen de una consulta agregada a través del puerto del catálogo (`get_categories`), con caché. En el perfil nube no se lee nada al importar; la primera petición ejecuta 7 sentencias y cada búsqueda posterior 2.
6. **`GET /api/v1/usage`.** Llamadas, tokens, latencia y coste informado por operación desde el arranque. Desactivado salvo con `USAGE_ENDPOINT_ENABLED=true` (responde 404 en caso contrario).
7. **Valores por defecto con `SEARCH_PROFILE=cloud`.** `qwen/qwen3-embedding-8b` a 1024 dimensiones y `qwen/qwen3-reranker-8b`, cada uno con su proveedor fijado y sustituible por variable de entorno. Los valores locales no cambian.
8. **Dos correcciones de robustez halladas en la comprobación en vivo.** Una petición a la API de modelos estuvo abierta más de nueve minutos porque el proveedor enviaba bytes de keep-alive y el tiempo de lectura nunca saltaba; ahora cada intento tiene un plazo duro de 30 segundos. Una conexión muerta a la base de datos bloqueó unos cinco minutos; ahora las conexiones usan TCP keepalives y un enlace muerto falla en unos 25 segundos y se reintenta una vez.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Conservar el relleno porque detrás del reranker su estimación puntual era mayor | Devuelve libros que no coinciden con la consulta, y su beneficio aparente depende de que el catálogo sea diminuto |
| Eliminar del todo el comportamiento antiguo | Los resultados publicados dejarían de poder reproducirse; se conserva tras una opción explícita que solo usa la evaluación |
| Un solo fichero de dependencias para los dos perfiles | El fichero completo trae las bibliotecas de modelos (varios GB instalados), que una función serverless no puede alojar; el fichero reducido se contrasta con el completo mediante un test |
| Importar los adaptadores locales y capturar el error de importación en la nube | Frágil y lento de arrancar; las importaciones perezosas por perfil son explícitas y están probadas |
| Comprobar solo el nombre del modelo entre los vectores almacenados y el codificador de consultas | Dos proveedores del mismo modelo devuelven vectores distintos (V16), de modo que proveedor y dimensión forman parte de la identidad |
| Pedir los embeddings de uno en uno en el perfil nube | Un viaje a la base de datos por libro; el método por lotes lo deja en uno |
| Leer todo el catálogo al arrancar para reunir las categorías | Aceptable con 177 libros, demasiado lento con decenas de miles; una consulta agregada lo sustituye |
| Confiar en el tiempo de lectura del cliente HTTP para la API de modelos | No saltó mientras el proveedor mantenía viva la conexión; un plazo duro por intento sí |

## Cómo se ha validado

- Suite completa: 1522 superados, 16 omitidos (pruebas de integración que necesitan una base de datos de prueba) y 0 fallidos.
- Las 16 pruebas de integración de Postgres se superaron contra una rama temporal de Neon, borrada después. En Neon queda la rama por defecto con los esquemas `catalog_small`, `eval177`, `eval177_hnsw` y el nuevo `eval177_api`.
- La repetición de la ablación de V06 coincide con lo publicado en 44 de 44 listas para los cinco sistemas reproducidos en `BM25_PADDING.md`, y en los once sistemas de la ablación completa.
- Los dos tests en subproceso y el test de las versiones de `requirements-cloud.txt` cubren la Parte B; el entorno reducido de 127 MB se probó además a mano.

## Comprobación en vivo contra Neon

La colección de 177 libros se cargó en un esquema nuevo (`eval177_api`) con embeddings por API (34,647 tokens, 0.00035 USD). La API se ejecutó en local en la configuración nube y se le enviaron las 44 consultas por HTTP, una a una tras una consulta de calentamiento, con 10 resultados. Todos los endpoints respondieron (`/health`, `/ui/`, libro, libros similares con y sin MMR, búsqueda, `/mcp` en `initialize` y `/usage`). Sin búsquedas degradadas. Es una medida descriptiva, sin contraste declarado.

Latencia de `POST /api/v1/search`, desde esta máquina hasta Neon (Frankfurt) y OpenRouter:

| Parte | Mediana ms | p95 ms |
|---|---|---|
| Petición completa (reloj del cliente) | 1755 | 7080 |
| Búsqueda en el servidor | 1753 | 7077 |
| Embedding de la consulta (API) | 494 | 3812 |
| Búsqueda en la base de datos (BM25 y vectorial de Postgres, fusión) | 276 | 762 |
| Reranking (API) | 677 | 922 |

La primera búsqueda del proceso (conexión a Neon y datos del catálogo) tardó 5192 ms. Todo lo que queda fuera de la base de datos y de las dos llamadas a la API ocupa unos 3 ms.

Calidad (nDCG@10 con los juicios de V13, descriptivo; los 41 huecos del top 10 de la pila nube los puntuó el mismo juez, 0.0005 USD):

| Sistema | nDCG@10 [IC 95 %] | Recall@10 | MRR |
|---|---|---|---|
| `cloud_stack` | 0.722 [0.635, 0.799] | 0.473 | 0.902 |
| `vec_e5` | 0.661 [0.585, 0.732] | 0.488 | 0.849 |
| `hybrid_norm_e5_bge` | 0.688 [0.613, 0.757] | 0.527 | 0.851 |
| `hybrid_norm_e5_bge_nopad` | 0.660 [0.576, 0.739] | 0.475 | 0.822 |

| Pareja | Dif. [IC 95 %] | p (sin ajustar) |
|---|---|---|
| `cloud_stack` frente a `hybrid_norm_e5_bge` | +0.034 [-0.027, +0.093] | 0.275 |
| `cloud_stack` frente a `hybrid_norm_e5_bge_nopad` | +0.062 [+0.003, +0.123] | 0.054 |
| `cloud_stack` frente a `vec_e5` | +0.061 [-0.004, +0.127] | 0.079 |

La pila nube difiere de la referencia local en tres puntos a la vez (modelo de embeddings, motor BM25 y reranker), por lo que la diferencia no es atribuible a ninguno de ellos.

Coste: 0.0015 USD por búsqueda de media (0.0660 USD en las 44 búsquedas), con una llamada de embedding y una de reranking por búsqueda. El gasto total de la tarea fue de unos 0.10 USD.

## Decisiones que se derivan

Tomadas por el diseñador el 2026-11-11:

1. Se mantiene la corrección de BM25: devolver libros que no coinciden con la consulta es incorrecto con independencia de su efecto medido.
2. La memoria no debe afirmar que el híbrido supera a la parte vectorial en esta colección.
3. La comparación se repetirá en el catálogo grande, donde se espera que las consultas por título o autor exactos favorezcan a la rama léxica. Es una expectativa, no un resultado.
4. Las llamadas a la API en tiempo de consulta necesitan plazos más cortos que los 30 segundos por cuatro intentos actuales, de modo que una búsqueda se degrade en lugar de esperar (tarea siguiente).
5. El catálogo de 5000 obras no se versiona en el repositorio: el perfil local conserva la colección evaluada de 177 libros y el catálogo multilingüe vive en la nube.

## Límites que debe conocer el tribunal

- Son 44 consultas y 177 libros; con ese tamaño las diferencias inferiores a unos 0.08 de nDCG@10 son difíciles de detectar. Los juicios añadidos proceden de un juez LLM, no de personas.
- La latencia de la comprobación en vivo se midió desde una máquina hasta una región. Su cola la domina la API de embeddings (p95 de la petición completa, unos siete segundos).
- El modelo de embeddings alojado devuelve vectores ligeramente distintos en llamadas repetidas (medido en V16), por lo que los resultados en vivo no son exactamente repetibles.
- No se han verificado las sesiones MCP sobre una función serverless ni `/evaluate` en el perfil nube.
- La primera petición sigue emitiendo varias sentencias pequeñas que podrían fusionarse.
- La explicación de la Parte A sobre el comportamiento detrás del reranker es una interpretación sin verificar.

## Cómo reproducirlo

```
python -m app.evaluation.bm25_padding run      # modelos; el juez solo para los pares sin juicio
python -m app.evaluation.bm25_padding report   # recalcula las tablas desde los rankings guardados

SEARCH_PROFILE=cloud python -m scripts.load_postgres --from data/catalog.db --schema eval177_api
SEARCH_PROFILE=cloud DATABASE_SCHEMA=eval177_api USAGE_ENDPOINT_ENABLED=true uvicorn app.main:app --port 8765
python -m app.evaluation.cloud_live_check run --base-url http://127.0.0.1:8765
python -m app.evaluation.cloud_live_check report   # sin API ni juez
```

- Los dos `report` no necesitan base de datos, modelo, juez ni red. Las cadenas de conexión y la clave de OpenRouter se configuran en el entorno y nunca se versionan.
- Tests: `pytest tests/api/test_cloud_profile_imports.py`, y la suite completa con `pytest`. Las pruebas de integración se omiten sin cadena de conexión de prueba.

## Ficheros

- `app/infrastructure/search/bm25_search_repository.py` (opción `zero_score_padding`), `app/domain/services.py` (listas léxicas cortas o vacías).
- `app/evaluation/bm25_padding.py`, `app/evaluation/cloud_live_check.py`, `app/evaluation/relevance_judgments_extended_v13.json`.
- `data/evaluation/BM25_PADDING.md`, `data/evaluation/CLOUD_LIVE_CHECK.md`, `data/evaluation/results_bm25_padding.json`.
- `requirements-cloud.txt`, `.env.example` (bloque del perfil nube).
- `app/api/v1/usage_endpoints.py`, `app/api/v1/dependencies.py`, `app/main.py`.
- `scripts/load_postgres.py`, puerto `EmbeddingsStore` (`get_embeddings_batch`) y catálogo (`get_categories`).
- Tests: `tests/api/test_cloud_profile_imports.py` (con `tests/api/cloud_profile_driver.py`), `tests/test_requirements_cloud.py` y los de `tests/infrastructure/` y `tests/evaluation/` correspondientes.
