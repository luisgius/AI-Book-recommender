# V12 · Perfil nube: catálogo y búsqueda en Postgres (Neon) y paridad con el perfil local

- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** infraestructura (`app/infrastructure/db/`, `app/infrastructure/search/`, `app/infrastructure/config/`), aplicación (`app/api/v1/`), scripts, evaluación (`app/evaluation/`)

## Qué se ha hecho

- Esquema SQL versionado (`app/infrastructure/db/postgres_schema.sql`) aplicado por una migración idempotente: `works` (con el embedding como `halfvec`), `work_texts` (una fila por idioma de descripción, con un `tsvector` almacenado) y `catalog_info`. Todo vive en un esquema cuyo nombre sale de la configuración, de modo que varios catálogos conviven en la misma base de datos.
- Tres adaptadores que implementan los puertos existentes: `PostgresBookCatalogRepository`, `PostgresLexicalSearchRepository` y `PgVectorSearchRepository`. Devuelven resultados con el `Book` completo, con una consulta por búsqueda.
- Cargador `python -m scripts.load_postgres --from <catálogo sqlite> --schema <nombre>`: se puede reanudar y reutiliza los vectores ya calculados.
- Ajuste `SEARCH_PROFILE` (`local` por defecto, o `cloud`) conectado en la API y visible en `/health`. La fusión RRF sigue en `SearchService` del dominio, sin cambios.
- Medida de paridad `app/evaluation/profile_parity.py`: compara los dos perfiles con los mismos documentos, vectores y fusión. Informe en `data/evaluation/PROFILE_PARITY.md`.

## Por qué

El autor decidió que el sistema debe vivir también en la nube, en Neon (Postgres gestionado), pero "midiendo y probando". El diseño aprobado son dos perfiles de adaptadores sobre los mismos puertos del dominio (`docs/notes/2026-11-11-propuesta-nube-y-mcp.md`, secciones 2, 3 y 6). El perfil local (SQLite, BM25 con `rank-bm25`, FAISS exacto) sigue siendo el sistema evaluado y reproducible. El perfil nube guarda el catálogo y sirve la búsqueda léxica y vectorial desde Postgres.

La nota de propuesta suponía que la búsqueda de texto de Postgres no sería BM25. La comprobación sobre el propio proyecto de Neon mostró otra cosa: este proyecto ofrece PostgreSQL 18.6 en `aws-eu-central-1`, con pooler en modo transacción, y las extensiones `vector` 0.8.6, `unaccent`, `pg_trgm`, `lakebase_text` 0.1.3 (un índice BM25 real), `lakebase_vector` 1.1.2 (índice `lakebase_ann`), `lakebase_tokenizer` 0.2.1 y `pg_search` 0.15.26. La habilidad de Neon instalada en el agente reveló que Neon ya incluye BM25, y se adoptó como implementación léxica. Postgres trae 29 idiomas con lematizador más `simple`; ninguno para chino ni japonés.

## Objetivo

Construir los adaptadores de datos y de búsqueda del perfil nube y medir en qué se diferencian de los locales cuando los documentos, los embeddings y la fusión son idénticos. La pregunta es si lo medido en local se traslada a la nube.

## Diseño de la medida, fijado de antemano

- Colección de 177 libros cargada en Postgres con los mismos embeddings de documento que el brazo local `vec_e5` (`intfloat/multilingual-e5-large-instruct`), 44 consultas (24 en inglés, 20 en español), profundidad 10. Cada consulta se codifica una vez y el mismo vector se entrega a los dos perfiles.
- Fusión RRF con k = 60 y 20 candidatos por rama, en el `SearchService` sin cambios. Solo difieren el almacenamiento y los motores de recuperación.
- Tres contrastes declarados de antemano sobre nDCG@10, con corrección de Holm: léxico (`pg_bm25` frente a `bm25_normalized`), vectorial (`pg_vec_exact` frente a `vec_e5`) e híbrido (`pg_hybrid` frente a `hybrid_norm_e5`). Test de aleatorización pareado bilateral, bootstrap percentil pareado del 95 % (10 000 remuestreos, semilla 0), como en V06. Todo lo demás es descriptivo.
- Juicios: se parte de los juicios extendidos del juez `z-ai/glm-5.3-flash` y ese mismo juez puntuó 68 pares nuevos (`relevance_judgments_extended_cloud.json`). Todo resultado devuelto por cualquier sistema tiene juicio.
- Diseño léxico: una configuración de búsqueda de texto por idioma con lematizador, con `unaccent` delante, más `simple_unaccent` para títulos, autores, categorías e idiomas sin lematizador. Un único índice BM25 sobre todas las filas de texto, con k1 = 1,5 y b = 0,75 como el BM25 local; cada obra conserva su mejor fila. La consulta se interpreta bajo todas las configuraciones en uso, se descartan las palabras vacías de cualquiera de esos idiomas y los términos restantes se combinan con OR. `ts_rank_cd` sigue seleccionable con `PG_LEXICAL_RANKING`.
- Diseño vectorial: coseno sobre `halfvec`. Modo exacto (con los escaneos por índice desactivados en esa sentencia) para evaluar; índice aproximado `lakebase_ann` por defecto, con HNSW seleccionable. Una búsqueda filtrada que se queda corta se repite una vez en modo exacto.
- Conexiones: la API usa la cadena con pooler y el cargador la directa, como recomienda Neon. Cada llamada es un solo mensaje `SET LOCAL ...; <sentencia>` (un viaje de ida y vuelta, y los ajustes no se filtran a otros usuarios de la conexión compartida). Sin sentencias preparadas en el servidor, un pool de como máximo 4 conexiones y ninguna en reposo; la cadena de conexión no se registra nunca.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Supabase en lugar de Neon | Su plan gratuito es de 500 MB y se pausa tras siete días sin actividad; el mismo SQL cuesta 25-30 USD al mes. Neon no tiene cuota mínima y se suspende en reposo |
| Una base de datos vectorial dedicada (Qdrant, Pinecone) | Obliga a un segundo almacén junto al catálogo y no permite filtros SQL sobre las mismas filas; Postgres reúne catálogo, búsqueda léxica y vectorial en un solo sitio |
| Ranking nativo de Postgres (`ts_rank_cd`) | No usa estadísticas del corpus; midió peor (nDCG@10 0.451 frente a 0.492 de `pg_bm25`) y se conserva solo como comparación |
| Fusionar léxico y vectorial dentro de una función SQL | Ahorra un viaje de ida y vuelta (unos 50 ms desde la máquina de prueba), pero la fusión existiría dos veces y la medida de paridad compararía código de fusión en lugar de motores |
| Rellenar con obras sin coincidencia hasta diez resultados, para imitar al adaptador local | Copiaría un defecto; el informe declara las longitudes de lista en su lugar |
| Un índice BM25 por configuración de idioma | Las puntuaciones de índices distintos no son comparables; un solo índice da un solo conjunto de estadísticas del corpus y un solo top K |
| Semántica AND para los términos de la consulta | 32 de las 44 consultas no devolverían nada |
| Un ORM | Los adaptadores necesitan un puñado de sentencias; SQL plano en un fichero versionado es más fácil de leer y de defender |
| Evaluar con el índice aproximado | Los resultados dependerían de los parámetros del índice; las cifras usan búsqueda exacta y el recall del índice se informa aparte |

## Cómo se ha validado

- Suite completa: 1439 superados, 12 omitidos (pruebas de integración que necesitan una cadena de conexión de prueba) y 0 fallidos.
- Las 12 pruebas de integración se ejecutaron contra una rama desechable de Neon, borrada después. En Neon queda solo la rama por defecto, con los esquemas `eval177`, `eval177_hnsw` y `catalog_small`.
- Equivalencia vectorial: la similitud coseno de los mismos pares (consulta, libro) con `halfvec` frente a FAISS en float32 difiere en media 4.2e-06 y como máximo 1.9e-05, sobre 440 pares.
- Dependencias añadidas: `psycopg` 3.3.8, `psycopg-binary` 3.3.8 y `psycopg-pool` 3.3.3. Se probó el paquete Python `pgvector` y se retiró: los vectores se envían como texto y se convierten en el servidor.

## Resultados de paridad

Contrastes declarados de antemano (nDCG@10; diferencia con intervalo del 95 %; p bruto y con Holm):

| Contraste | Sistema | Referencia | nDCG@10 sistema | nDCG@10 referencia | Dif. [IC 95 %] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| léxico | `pg_bm25` | `bm25_normalized` | 0.492 | 0.489 | +0.003 [-0.032, +0.039] | 0.878 | 1.000 |
| vectorial | `pg_vec_exact` | `vec_e5` | 0.662 | 0.662 | +0.000 [+0.000, +0.000] | 1.000 | 1.000 |
| híbrido | `pg_hybrid` | `hybrid_norm_e5` | 0.624 | 0.592 | +0.032 [-0.007, +0.074] | 0.139 | 0.417 |

Todos los sistemas (descriptivo):

| Sistema | nDCG@10 [IC 95 %] | Recall@10 | P@5 | MRR | Judged@10 | Resultados por consulta (consultas < 10) |
|---|---|---|---|---|---|---|
| `bm25_normalized` | 0.489 [0.395, 0.584] | 0.282 | 0.582 | 0.735 | 1.000 | 10.0 (0) |
| `vec_e5` | 0.662 [0.586, 0.732] | 0.492 | 0.750 | 0.849 | 1.000 | 10.0 (0) |
| `hybrid_norm_e5` | 0.592 [0.504, 0.679] | 0.413 | 0.700 | 0.804 | 1.000 | 10.0 (0) |
| `hybrid_norm_e5_matches` | 0.622 [0.540, 0.700] | 0.433 | 0.727 | 0.836 | 1.000 | 10.0 (0) |
| `pg_bm25` | 0.492 [0.394, 0.591] | 0.274 | 0.577 | 0.695 | 0.802 | 8.0 (14) |
| `pg_fts_tsrank` | 0.451 [0.361, 0.539] | 0.264 | 0.605 | 0.740 | 0.802 | 8.0 (14) |
| `pg_vec_exact` | 0.662 [0.586, 0.732] | 0.492 | 0.750 | 0.849 | 1.000 | 10.0 (0) |
| `pg_vec_ann` | 0.662 [0.586, 0.732] | 0.492 | 0.750 | 0.849 | 1.000 | 10.0 (0) |
| `pg_vec_hnsw` | 0.662 [0.586, 0.732] | 0.492 | 0.750 | 0.849 | 1.000 | 10.0 (0) |
| `pg_hybrid` | 0.624 [0.540, 0.705] | 0.429 | 0.709 | 0.809 | 1.000 | 10.0 (0) |
| `bm25_normalized_matches` | 0.474 [0.376, 0.572] | 0.259 | 0.559 | 0.708 | 0.727 | 7.3 (18) |

`hybrid_norm_e5_matches` y `bm25_normalized_matches` son sistemas derivados y descriptivos: el BM25 local sin sus resultados de puntuación cero, solo o fusionado con `vec_e5`.

Sobre Judged@10 (0,802 en los dos sistemas léxicos de Postgres): no hay resultados sin juicio, todo lo devuelto está juzgado (proporción de juzgados entre los devueltos: 1,000). La cifra baja porque Postgres solo devuelve obras que contienen alguna palabra de la consulta (8,0 resultados por consulta de media, 14 consultas con menos de 10 y 2 sin ninguno), mientras que el adaptador local `rank-bm25` siempre devuelve 10 porque rellena la lista con libros de puntuación cero.

Diferencias descriptivas, sin corrección y fuera de la familia contrastada:

| Pareja | nDCG@10 sistema | nDCG@10 referencia | Diferencia [IC 95 %] | p (sin ajustar) |
|---|---|---|---|---|
| `pg_bm25` frente a `bm25_normalized_matches` | 0.492 | 0.474 | +0.019 [-0.013, +0.052] | 0.282 |
| `pg_fts_tsrank` frente a `bm25_normalized` | 0.451 | 0.489 | -0.038 [-0.097, +0.019] | 0.222 |
| `pg_vec_ann` frente a `vec_e5` | 0.662 | 0.662 | +0.000 [+0.000, +0.000] | 1.000 |
| `pg_vec_hnsw` frente a `vec_e5` | 0.662 | 0.662 | +0.000 [+0.000, +0.000] | 1.000 |
| `pg_hybrid` frente a `hybrid_norm_e5_matches` | 0.624 | 0.622 | +0.002 [-0.032, +0.038] | 0.929 |
| `hybrid_norm_e5_matches` frente a `hybrid_norm_e5` | 0.622 | 0.592 | +0.030 [+0.007, +0.059] | 0.020 |

Coincidencia de los rankings (top 10):

| Pareja | Solapamiento medio | Listas idénticas | Conjuntos idénticos |
|---|---|---|---|
| `pg_bm25` frente a `bm25_normalized` | 0.518 | 0 de 44 | 2 de 44 |
| `pg_bm25` frente a `pg_fts_tsrank` | 0.782 | 10 de 44 | 20 de 44 |
| `pg_vec_exact` frente a `vec_e5` | 1.000 | 44 de 44 | 44 de 44 |
| `pg_vec_ann` frente a `pg_vec_exact` | 0.998 | 43 de 44 | 43 de 44 |
| `pg_vec_hnsw` frente a `pg_vec_exact` | 1.000 | 44 de 44 | 44 de 44 |
| `pg_hybrid` frente a `hybrid_norm_e5` | 0.725 | 1 de 44 | 4 de 44 |
| `pg_hybrid` frente a `hybrid_norm_e5_matches` | 0.807 | 6 de 44 | 10 de 44 |

Forma de la consulta léxica (44 consultas):

| Forma | Consultas sin resultados | Consultas con menos de 10 | Mediana de obras que coinciden |
|---|---|---|---|
| AND (`websearch_to_tsquery` en cada configuración) | 32 | 39 | 0.0 |
| OR de los lexemas (la que se usa) | 2 | 14 | 25.5 |

## Latencia, tamaños y capacidad del plan gratuito

Latencia en caliente por consulta, desde la máquina de medida hasta Frankfurt por la cadena con pooler (un viaje por rama de búsqueda). El vector de la consulta está precalculado en ambos perfiles, de modo que el modelo no se cronometra.

| Sistema | Mediana ms | p95 ms |
|---|---|---|
| `bm25_normalized` (local) | 0.07 | 0.09 |
| `vec_e5` (local) | 0.04 | 0.04 |
| `hybrid_norm_e5` (local) | 0.17 | 0.19 |
| `pg_bm25` | 54.54 | 156.70 |
| `pg_fts_tsrank` | 56.07 | 67.86 |
| `pg_vec_exact` | 66.05 | 75.91 |
| `pg_vec_ann` | 66.29 | 109.21 |
| `pg_vec_hnsw` | 95.50 | 157.77 |
| `pg_hybrid` | 131.47 | 303.80 |
| `SELECT 1` (suelo de red) | 47.39 | 56.21 |

Arranque en frío tras suspenderse el cómputo (una sola medida): 966 ms de conexión más 825 ms de la primera búsqueda, 1790 ms en total, frente a 324 + 438 = 762 ms con el cómputo despierto y 140 ms de mediana con conexión y adaptadores reutilizados. La propuesta hablaba de "cientos de milisegundos"; es cercano a un segundo.

Tamaños del esquema `eval177` (177 obras): 3.34 MiB con `lakebase_ann`; `eval177_hnsw`: 2.97 MiB. Con `catalog_small` (5000 obras, 24314 filas de texto, 4.9 por obra, 11 configuraciones de texto en uso) el esquema ocupa 159.79 MiB con `lakebase_ann` y 157.80 MiB con HNSW; el índice `lakebase_ann` pesa 15.00 MiB y el HNSW 13.03 MiB. Recall@10 del índice frente a la búsqueda exacta: 0.998 con `lakebase_ann` (43 de 44 consultas en 1.000) y 0.977 con HNSW (36 de 44).

Límites del plan gratuito de Neon, leídos en su documentación y su CLI: 1 GB de almacenamiento por proyecto, 100 horas de CU por proyecto al mes, suspensión tras 5 minutos de inactividad (no configurable) y sin borrado por inactividad.

Capacidad, medida sobre la carga limpia de `catalog_small`: 32,7 KiB por obra con el índice `lakebase_ann` y 29,7 KiB sin índice aproximado. Caben unas 31 600 obras de este tipo en 1 GiB con `lakebase_ann` y unas 34 800 sin índice aproximado. Son estimaciones, no medidas: unas 42 000 si se prescinde del índice GIN y del texto almacenado de la descripción, y unas 54 000 con registros de una sola descripción. Con 200 000 obras de este tipo no caben en el plan gratuito. La estimación de la nota de propuesta (unos 2,2 GB para 200 000 obras) era demasiado baja: a 32,7 KiB por obra, 200 000 obras necesitan unos 6,2 GiB.

## Lectura de los resultados

1. La búsqueda vectorial exacta de Postgres reproduce la de FAISS lista por lista (44 de 44 listas idénticas). La diferencia de similitud (máximo 1.9e-05) viene del redondeo a media precisión.
2. El BM25 de Postgres y el local no se distinguen en calidad (+0.003, intervalo [-0.032, +0.039]), aunque sus rankings se parecen poco (solapamiento medio del top 10 de 0.518, ninguna lista idéntica). Con 44 consultas, "sin diferencia detectable" no equivale a "equivalentes".
3. El híbrido de Postgres puntúa +0.032 sobre el local, con un intervalo que incluye el cero (p de Holm 0.417), de modo que no se puede afirmar una diferencia. Las dos filas descriptivas explican el origen: el híbrido local cuya rama BM25 descarta los libros de puntuación cero sube a 0.622, es decir +0.030 [+0.007, +0.059] sobre el híbrido local en producción (p sin ajustar 0.020), y `pg_hybrid` menos ese sistema queda en +0.002. La diferencia entre perfiles viene de los libros de relleno que entran en la fusión, no de los motores. Este hallazgo no estaba declarado de antemano y es descriptivo.
   Afecta al hallazgo de V06 de que el híbrido puntuaba menos que la parte vectorial sola: una parte del descenso se debe a ese relleno, aunque el híbrido sin relleno (0.622) sigue por debajo de `vec_e5` (0.662).
4. `ts_rank_cd` queda por debajo de BM25 en nDCG@10 (0.451 frente a 0.492), con un intervalo de la diferencia con `bm25_normalized` que incluye el cero.
5. El índice aproximado no cambia el resultado en esta colección (nDCG@10 0.662 en los tres modos). En `catalog_small` su recall@10 es 0.998 con `lakebase_ann` y 0.977 con HNSW.
6. Con OR la cobertura léxica es aceptable (2 consultas sin resultados); con AND, inservible (32).

Otras cosas aprendidas durante la construcción:
- Poner el plegado de acentos delante del lematizador español debilita el lematizador: "programación" pasa a `programacion` en lugar de la raíz `program`. La insensibilidad a las tildes se mantiene.
- El índice BM25 rechaza k1 por debajo de 1,2 y ignora las palabras repetidas de la consulta.
- `lakebase_ann` sigue haciendo búsqueda plana con 5000 filas y su índice es mayor que el de HNSW.

## Decisiones que se derivan

Tomadas por el diseñador el 2026-11-11, para aplicarlas en tareas posteriores y no en esta:

1. El perfil nube es viable: Postgres reproduce la búsqueda vectorial exacta lista por lista y la calidad del BM25 sin diferencia detectable, de modo que lo medido en local se traslada.
2. El BM25 mediante `lakebase_text` es la implementación léxica de la nube; `ts_rank_cd` queda solo como comparación.
3. El adaptador BM25 local debe dejar de devolver libros de puntuación cero (un defecto que rebaja el híbrido), conservando el comportamiento antiguo tras una opción para que los resultados pasados sigan siendo reproducibles, y el efecto se medirá como contraste declarado de antemano.
4. En el plan gratuito el catálogo desplegado debe ser una selección de unas 30 000 obras. Un catálogo desplegado mayor exige el plan de pago por uso, que es una decisión del autor.
5. Las cifras de evaluación se informan con búsqueda exacta, y el recall del índice aproximado se informa aparte.

## Límites que debe conocer el tribunal

- Son 44 consultas y 177 libros: las diferencias pequeñas no se detectan y los resultados pueden no trasladarse a un catálogo grande. La carga de 5000 obras sirve para tamaños, tiempos y recall del índice, no para calidad de ranking.
- Las latencias proceden de una máquina y una red, y el arranque en frío es una sola medida. Las cifras de Postgres incluyen unos 47 ms de red por viaje; las locales no.
- Los juicios de 68 pares nuevos, como los anteriores del juez LLM, no son notas humanas.
- Las listas léxicas de Postgres son más cortas (8,0 resultados de media, 14 consultas con menos de 10), por lo que Judged@10 es 0,802 aunque todo lo devuelto esté juzgado. Los dos perfiles no devuelven listas de la misma longitud.
- Los rankings léxicos de los dos perfiles difieren mucho (solapamiento del top 10 de 0.518) aunque el nDCG@10 no lo haga.
- La similitud entre libros y la diversificación MMR en el perfil nube necesitan un viaje de ida y vuelta por libro hasta que el puerto del dominio tenga un método por lotes.
- Los datos del catálogo y las estadísticas del corpus BM25 solo se refrescan cuando el cargador reconstruye el índice.
- Al arrancar, el cliente LLM lee todos los libros para reunir las categorías; en un catálogo grande es costoso.
- El chino y el japonés no tienen lematizador y su texto no se divide en palabras, de modo que la recuperación léxica en esos idiomas es pobre. `lakebase_tokenizer` lo resolvería; no se ha hecho.
- Las estimaciones de capacidad de 42 000 y 54 000 obras no están medidas.

## Cómo usarlo y reproducirlo

```
python -m scripts.load_postgres --from data/catalog.db --schema eval177 --embedding-model intfloat/multilingual-e5-large-instruct
python -m scripts.load_postgres --from data/catalog.db --schema eval177_hnsw --embedding-model intfloat/multilingual-e5-large-instruct --vector-index hnsw
python -m app.evaluation.profile_parity run --schema eval177 --hnsw-schema eval177_hnsw
python -m app.evaluation.profile_parity report
python -m scripts.load_postgres --from data/catalog_small/catalog.db --schema catalog_small --embedding-model intfloat/multilingual-e5-large-instruct
python -m app.evaluation.profile_parity scale --schema catalog_small
```

- `report` no necesita base de datos, modelo ni juez; `run` necesita la base de datos, el modelo de embeddings y, salvo con `--no-judge`, el juez.
- Para servir desde la nube: `SEARCH_PROFILE=cloud`, `DATABASE_SCHEMA=<esquema>`, y opcionalmente `PG_LEXICAL_RANKING` (`bm25` o `ts_rank_cd`) y `PG_VECTOR_SEARCH` (`approximate` o `exact`). Las cadenas de conexión (`DATABASE_URL`, `DATABASE_URL_UNPOOLED`) se configuran en el entorno y nunca se versionan.
- Otras operaciones del cargador: `--migrate-only`, `--report` y `--drop`.
- Tests: `pytest tests/infrastructure/test_postgres_rows_and_schema.py tests/infrastructure/test_postgres_search_adapters.py tests/infrastructure/test_load_postgres_script.py tests/infrastructure/test_search_profile_settings.py tests/evaluation/test_profile_parity.py`. Las pruebas de integración se omiten si no hay cadena de conexión de prueba.

## Ficheros

- `app/infrastructure/db/postgres_schema.sql`, `postgres_schema.py`, `postgres_connection.py`, `postgres_rows.py`, `postgres_book_catalog_repository.py`.
- `app/infrastructure/search/postgres_lexical_search_repository.py`, `pgvector_search_repository.py`.
- `app/infrastructure/config/settings.py` (`SearchProfileSettings`), `app/api/v1/dependencies.py`, `app/api/v1/search_endpoints.py`.
- `scripts/load_postgres.py`, `requirements.txt`, `.env.example`.
- `app/evaluation/profile_parity.py`, `app/evaluation/relevance_judgments_extended_cloud.json`.
- `data/evaluation/PROFILE_PARITY.md`, `data/evaluation/results_profile_parity.json`.
- Tests en `tests/infrastructure/` y `tests/evaluation/test_profile_parity.py`.
