# V18 · Ampliación del catálogo desplegado a 50 000 obras en el plan gratuito de Neon, con una carga que no deja filas muertas

- **Fecha:** 2026-11-12 (código en el commit `04569cf`; la carga y las medidas son del mismo día)
- **Estado:** hecha y medida; el sistema desplegado sigue sin evaluarse (ver "Límites que debe conocer el tribunal")
- **Capa:** infraestructura (`app/infrastructure/db/`), interfaz (`app/ui/static/`), scripts, pruebas

Esta tarea continúa V17 (`V17-despliegue-publico.md`), que dejó publicado el perfil nube con 30 000 obras. Los datos y las cifras de esta nota salen de `data/evaluation/DEPLOYMENT_CHECK.md` (versión del commit `04569cf`), del informe de la selección (`data/catalog_deploy/CATALOG_REPORT.md`) y de los cambios de ese commit.

## Qué se ha hecho

- El catálogo de la dirección pública (`https://ai-book-recommender-one.vercel.app`) ha pasado de 30 000 a 50 000 obras. Sigue en el plan gratuito de Neon (el proveedor de Postgres gestionado): la base de datos ocupa 626.4 MiB de un límite de 1 GiB.
- Se ha confirmado el límite del plan antes de cargar nada. El proyecto de Neon está en el plan Free y la API de Neon informa de un límite de almacenamiento (`branch_logical_size_limit`) de 1024 MiB (1 073 741 824 bytes) para este proyecto, que se consultó con `neon projects get twilight-poetry-45310085`. La página de planes de Neon (`https://neon.com/docs/introduction/plans`, actualizada el 2026-11-11) da 1 GB de almacenamiento por proyecto en el plan gratuito y indica que, superado ese tamaño, fallan las escrituras que lo aumentan. El proyecto tiene una sola rama (`production`).
- Se ha seleccionado un catálogo de 50 000 obras que contiene las 30 000 de V17 sin cambios, y se han añadido 20 000 nuevas.
- Se ha cambiado el cargador (`scripts/load_postgres.py`) para que cada fila se escriba una sola vez, de modo que una carga no deja filas muertas (filas que Postgres conserva tras una actualización hasta que se limpian) y no necesita `VACUUM FULL`.
- Se ha cargado en cuatro etapas de 5000 obras, con una guarda de tamaño y midiendo la base de datos tras cada etapa.
- Se ha vuelto a desplegar la aplicación, solo para cambiar un texto de la interfaz, y se ha repetido la comprobación del despliegue (`scripts/deployment_check.py`), con una tabla de latencia en caliente antes y después.

## Por qué

El 2026-11-12 se preguntó al autor si cargar más obras, dado que el plan gratuito tenía hueco más allá de las 30 000 cargadas (423.9 MiB de 1 GiB al terminar V17). Respondió "yeah ofc". Su criterio de coste, citado en V17, sigue vigente: "lo mejor son opciones gratuitas o de muy bajo coste". Por eso la ampliación se hizo sin salir del plan gratuito.

El objetivo de diseño fueron 50 000 obras. Con los bytes por obra medidos en V17, eso son unos 694 MiB, por debajo de un techo que el proyecto se fijó en 800 MiB sobre el límite del plan de 1 GiB. La holgura que queda sirve para los contadores diarios de uso, para la hinchazón de las tablas (bloat, el espacio que Postgres no recupera enseguida) y para las tablas temporales de la siguiente tarea de evaluación. La guarda del cargador se fijó en 780 MiB.

El cargador de V17 insertaba cada fila y luego la actualizaba para añadirle su vector de búsqueda. Cada actualización deja una fila muerta, y esas filas se eliminaban con `VACUUM FULL`. Esa orden reescribe la tabla y necesita espacio temporal para la copia, que a este tamaño podía haber superado el límite del plan. Por eso se cambió el cargador en lugar de repetir el procedimiento de V17.

## Objetivo

1. Ampliar el catálogo desplegado de 30 000 a 50 000 obras dentro del plan gratuito de Neon, con margen bajo el techo de 800 MiB.
2. Conservar las 30 000 obras ya desplegadas, sin reescribirlas ni volver a calcular sus vectores.
3. Hacer que una carga no deje filas muertas y no requiera `VACUUM FULL`.
4. Mantener el sitio público operativo durante la carga.
5. Medir el resultado: tamaño tras cada etapa, coste de los embeddings, latencia en caliente y en frío, y comprobaciones de la base de datos.

## Qué se ha hecho, con detalle

### Selección de 50 000 obras

La selección se obtiene con `python -m scripts.build_catalog select --size 50000` sobre el catálogo de fuentes múltiples (`data/catalog_core/catalog.db`). La regla es la de V17: puntuación igual a ediciones de Wikipedia más descargas de Gutenberg divididas entre 1000, con cuotas por idioma. Para 50 000 obras las cuotas son: inglés 20 000, español 12 500 y 1750 para cada uno de fr, de, it, pt, ca, nl, ru, zh, ja y ar. Dentro de cada idioma van primero las obras cuyo idioma declara una fuente y después las de mayor puntuación (desempate: más idiomas de descripción, luego el identificador). Los puestos de cuota que quedan sin usar los ocupan las mejores obras restantes de cualquier idioma.

**Por qué la selección de 50 000 contiene las 30 000 anteriores sin ninguna opción especial.** Las cuotas solo crecen con el tamaño, el orden de cada idioma es fijo, y los puestos sin usar crecen al menos tanto como las cuotas. Un test lo comprueba (`test_a_larger_selection_contains_every_smaller_one`). En la práctica, los hashes de contenido de las 30 000 obras ya cargadas coincidieron con los de la selección nueva: ninguna se reescribió ni se volvió a incrustar. Las 20 000 restantes son nuevas.

**Cuotas que no se pudieron llenar.** El catálogo de fuentes solo tiene 7231 obras en español, 971 en catalán y 1617 en chino, y las tres se seleccionaron enteras. Por eso las cuotas de esos tres idiomas quedaron sin llenar, y los 6181 puestos libres pasaron a las mejores obras restantes de cualquier idioma: francés 1349, latín 977, inglés 924, alemán 308 y otros idiomas 2623.

Idioma de la obra, antes y después:

| Idioma de la obra | 30 000 (V17) | 50 000 (V18) |
|---|---|---|
| en | 12 000 | 20 924 |
| es | 7231 | 7231 |
| fr | 1050 | 3099 |
| de | 1050 | 2058 |
| it | 1050 | 1750 |
| pt | 1050 | 1750 |
| nl | 1050 | 1750 |
| ru | 1050 | 1750 |
| ja | 1050 | 1750 |
| ar | 1050 | 1750 |
| zh | 1050 | 1617 |
| la | 134 | 977 |
| ca | 971 | 971 |
| otros | 214 (33 idiomas) | 2623 (55 idiomas) |

Fuentes de las obras, antes y después:

| Fuentes de la obra | 30 000 (V17) | 50 000 (V18) |
|---|---|---|
| wikidata | 23 821 | 40 794 |
| gutenberg | 2095 | 4106 |
| gutenberg + wikidata | 2438 | 3198 |
| doab | 1642 | 1897 |
| doab + wikidata | 3 | 4 |
| doab + gutenberg | 1 | 1 |

Otras cifras del informe de la selección de 50 000: 27 639 obras (55.3 %) tienen descripciones en más de un idioma; 4355 descripciones principales son resúmenes generados automáticamente por Project Gutenberg y marcados como tales; cobertura de campos: año 68.3 %, materias 58.9 %, `isbn13` 4.5 %, identificador de Wikidata 88.0 %. Licencias de las descripciones: CC BY-SA 3.0 (74 435), CC BY-SA 4.0 (36 941), catálogo de Project Gutenberg sin licencia explícita (4666) y CC0 1.0 (1905). Los recuentos, la regla y la licencia de cada fuente se guardan también en la base de datos (`catalog_deploy.catalog_info`, clave `catalog_contents`).

### El cambio del cargador y la prueba de que no deja filas muertas

En V17 el cargador insertaba cada obra sin su vector y cada fila de texto sin su `tsvector` (el texto preparado para la búsqueda léxica de Postgres), y rellenaba ambos con `UPDATE`. Sus contadores de escritura de entonces fueron 30 000 actualizaciones de `works` y 98 617 de `work_texts`: una fila muerta por obra y por fila de texto, 13.8 MiB en 5000 obras, que se eliminaban con `VACUUM FULL`.

El cargador nuevo escribe cada fila una vez, ya completa:

- El vector de una obra nueva (o de una obra cuyo texto cambió) se obtiene antes del `INSERT` y va en él, ya sea reutilizado o calculado con el codificador configurado.
- Cada fila de texto se inserta con su `tsvector`, calculado en la misma sentencia.
- Una carga interrumpida se termina ejecutando de nuevo la misma orden. Cada carga marca los índices como obsoletos antes de la primera escritura, para que una carga interrumpida los reconstruya.
- Cada carga termina con un `VACUUM ANALYZE` normal. No se ejecutó `VACUUM FULL`.

**Cómo se ha comprobado.** Los contadores de escritura de Postgres sobre toda la ampliación fueron: `works`, 20 000 inserciones, 0 actualizaciones y 0 borrados; `work_texts`, 33 388 inserciones, 0 actualizaciones y 0 borrados; `n_dead_tup` (filas muertas) 0 en ambas tablas tras la carga. Solo se actualizó `catalog_info` (27 filas de datos del catálogo). Una prueba unitaria (`test_upsert_writes_each_row_once_in_its_final_form`, en `tests/infrastructure/test_postgres_search_adapters.py`) y una prueba de integración (`test_enlarging_a_loaded_catalog_keeps_its_works_and_writes_each_row_once`) cubren el mismo comportamiento.

**Etapas ampliables.** La opción `--limit N` cambió de significado: antes cargaba las primeras N obras por identificador; ahora carga las obras ya almacenadas más las primeras nuevas por identificador, hasta N en total. Así una etapa nunca deja fuera una obra ya cargada, y el sitio vivo conserva todas sus obras y sus vectores. Una carga desde un esquema vacío toma las primeras N obras por identificador, como antes.

### Carga por etapas

Cada ejecución del cargador añade 5000 obras nuevas, conserva todas las ya cargadas y reconstruye los índices. Tamaños tras cada etapa:

| Obras | Base de datos MiB | Esquema MiB | Bytes por obra (esquema) | Índice BM25 MiB | Índice vectorial MiB | Pico de la base de datos MiB | Índice vectorial ausente (s) | Peticiones de embeddings / tokens / USD | Ejecución del cargador (s) |
|---|---|---|---|---|---|---|---|---|---|
| 30 000 | 424.5 | 405.1 | 14 158 | 56.7 | 88.5 | - | - | - | - |
| 35 000 | 473.9 | 454.4 | 13 614 | 60.6 | 103.2 | - | 8.2 | 220 / 863 272 / 0.0086 | 340.1 |
| 40 000 | 523.5 | 504.1 | 13 215 | 65.1 | 117.9 | - | 8.0 | 234 / 868 464 / 0.0087 | 217.0 |
| 45 000 | 574.9 | 555.6 | 12 946 | 69.6 | 132.6 | - | 10.8 | 227 / 886 078 / 0.0089 | 339.2 |
| 50 000 | 626.4 | 607.0 | 12 731 | 74.1 | 147.3 | 636.3 | 9.6 | 231 / 877 430 / 0.0088 | 222.5 |

- **Proyección tras la primera etapa:** 5000 obras añadidas ocuparon 49.4 MiB (10 358 bytes cada una), de modo que 50 000 obras ocuparían unos 622 MiB; se midieron 626.4 MiB. Las 20 000 obras añadidas ocupan 10 585 bytes cada una (201.9 MiB en total), menos que los 14 158 bytes de las 30 000 primeras, porque tienen menos idiomas de descripción por obra.
- **Pico:** la base de datos fue mayor justo antes de la última reconstrucción de índices, con 636.3 MiB (los índices que mantienen las inserciones son unos 10 MiB mayores que los reconstruidos). El cargador solo conserva el registro de tamaño por lote de su última ejecución, de modo que los picos de las etapas 1 a 3 no se guardaron. Lo que consta es que los tamaños finales de esas etapas subieron unos 50 MiB cada una, que cada lote se comprobó contra la guarda de 780 MiB y que la guarda no detuvo ninguna etapa.
- **Embeddings** (`qwen/qwen3-embedding-8b`, 1024 dimensiones, DeepInfra, por OpenRouter): 912 peticiones, 3 495 244 tokens y 0.0350 USD, según OpenRouter, para las 20 000 obras añadidas. Para las 30 000 de V17 fueron 5 603 598 tokens y 0.0560 USD.
- **Idempotencia:** ejecutar de nuevo la misma orden no escribió nada (0 obras nuevas o cambiadas, ninguna petición de embeddings, "Indexes are up to date", 21 s).

### Qué pasó con el sitio público durante la carga

- Ninguna obra cargada se borró ni se reescribió, y las obras nuevas se podían buscar a medida que se confirmaba cada lote.
- Cada etapa reconstruyó los índices, porque las estadísticas de BM25 se fijan al construir el índice.
  - **Índice BM25:** se construyó junto al que estaba en uso y se cambió en una sola transacción (se borra el viejo y se renombra el nuevo), de modo que la búsqueda léxica nunca se quedó sin él. Las construcciones tardaron 4.5, 4.3, 5.5 y 5.4 s.
  - **Índice vectorial:** se eliminó primero, para que la base de datos nunca tuviera a la vez dos índices BM25 y el vectorial, y se reconstruyó al final (3.4, 3.3, 4.4 y 3.9 s). Mientras faltó, las búsquedas vectoriales se hicieron como exploraciones exactas (revisan todas las filas), con resultados correctos pero más lentas: 36.6 s en total en las cuatro etapas. Con 50 000 obras la exploración exacta tarda 264 ms en la base de datos, frente a 4.2 ms con el índice.
- La API trata una búsqueda así como una búsqueda híbrida normal (una exploración exacta no es un fallo, y no se marca como degradada). Durante esos intervalos no se observó ninguna búsqueda pública: lo anterior sobre el comportamiento en esos segundos es una inferencia a partir del diseño y de las medidas de la base de datos, no una observación.
- Antes de reconstruir los índices, el cargador estima el pico de la base de datos y se niega a empezar si superaría la guarda (la reconstrucción de BM25 se supone un 50 % mayor que el índice en uso). En este caso no hizo falta.

### Comprobaciones de la base de datos

Con 50 000 obras: 50 000 obras, 50 000 vectores y 50 000 obras con documentos léxicos (117 947 documentos léxicos, ninguno con `tsvector` vacío); todos los índices válidos. `EXPLAIN ANALYZE` de las sentencias de la propia API (60 candidatos, como en el despliegue): la consulta vectorial usa `Index Scan using works_embedding_ann` (4.2 ms) y la consulta BM25 usa `Index Scan using work_texts_bm25` (18.3 ms).

Cuatro obras añadidas se buscaron por la API pública (`POST /api/v1/search`, 2026-11-12 a las 09:28:57 UTC) y salieron en primer lugar, en modo híbrido:

| Consulta | Idioma | Posición de la obra añadida | Servidor ms |
|---|---|---|---|
| Zuckerman Bound Philip Roth | en | 1 (Zuckerman Bound) | 1115 |
| Une partie de campagne Maupassant | fr | 1 (Une partie de campagne) | 985 |
| Zettels Traum Arno Schmidt | de | 1 (Zettels Traum) | 958 |
| 鹿男あをによし | ja | 1 (鹿男あをによし) | 1923 |

### Nuevo despliegue

Los datos no necesitaban un despliegue nuevo: la función lee el catálogo de la base de datos en cada búsqueda, y los datos del catálogo que guarda en memoria (configuraciones de búsqueda de texto, como máximo 12 textos por obra) no cambiaron. Las cuatro obras anteriores se encontraron con el despliegue del 2026-11-11, antes de cualquier redespliegue.

El redespliegue hizo falta solo para el texto estático "Sobre los datos" de la interfaz (`app/ui/static/index.html`), que decía "unas 30 000 obras" y ahora dice "unas 50 000 obras". Se hizo el 2026-11-12 (despliegue `dpl_D8cAmq5hHoeWrHAThWXniQXroMkE`, construcción de 37 s) desde una exportación limpia del último commit más los cambios de V18 bajo `app/`, porque el directorio de trabajo tenía otros cambios sin confirmar que no debían publicarse. El despliegue de producción anterior (`ai-book-recommender-e1wtlhals`) sigue disponible para volver atrás. Tras el redespliegue se comprobó con `curl` que `/ui/` sirve "unas 50 000 obras"; no se repitió la prueba en un navegador.

### Comprobación del despliegue repetida

`scripts/deployment_check.py` se amplió para guardar las latencias de ejecuciones anteriores y mostrarlas junto a la actual (opción `--catalog-works`, que registra el tamaño del catálogo en cada medida, y la sección "Warm latency before and after the enlargement"). Se ejecutó el 2026-11-12 a las 09:30 UTC con las mismas doce consultas y tres rondas de V17. La salud, la interfaz, las doce consultas, el libro, los similares, las explicaciones, los endpoints desactivados, el límite por cliente y el MCP se comprobaron de nuevo; el MCP se probó después de la comprobación del despliegue (`"A Human Algorithm"` entre los libros similares de `similar_books` es una de las obras añadidas).

## Resultados medidos

Todo lo siguiente sale de `data/evaluation/DEPLOYMENT_CHECK.md`. Son medidas descriptivas desde una máquina en España, sin contraste estadístico ni juicios de relevancia.

### Latencia en caliente, antes y después

36 búsquedas en cada ejecución (12 consultas por 3 rondas, tras una búsqueda de calentamiento); ninguna degradada ni distinta de 200 en ninguna de las dos.

| Parte | 30 000 obras (V17): mediana ms | p95 ms | 50 000 obras (V18): mediana ms | p95 ms |
|---|---|---|---|---|
| Petición completa (reloj del cliente) | 1430.2 | 3276.4 | 1120.7 | 2040.4 |
| Servidor (`latency_ms`) | 1284.3 | 3083.0 | 960.4 | 1918.0 |
| Etapa `embedding` | 384.4 | 1346.5 | 400.1 | 1021.5 |
| Etapa `lexical` | 33.9 | 468.8 | 31.6 | 54.6 |
| Etapa `rerank` | 545.5 | 2530.6 | 479.4 | 675.1 |
| Etapa `vector` | 19.4 | 175.8 | 17.2 | 24.8 |
| Degradadas | 0 de 36 | - | 0 de 36 | - |

Las ejecuciones son del 2026-11-11 a las 22:38 UTC (V17) y del 2026-11-12 a las 09:30 UTC (V18).

**Qué permite concluir esta tabla y qué no.** Con 50 000 obras las búsquedas en caliente no fueron más lentas que con 30 000. Las etapas que dependen de la base de datos se mantuvieron en el mismo orden (mediana vectorial 17.2 ms, léxica 31.6 ms) y los totales fueron menores, sobre todo porque las llamadas a modelos fueron más rápidas ese día; la mediana de la etapa de embeddings fue 15.7 ms mayor. Las dos ejecuciones están separadas por 11 horas y ambas dependen de la carga de los proveedores, así que no es una comparación controlada. La tabla no permite afirmar que la ampliación haya hecho las búsquedas más rápidas ni que el tamaño del catálogo no influya.

La búsqueda con explicaciones tardó 14.1 s (12.6 s de explicaciones), frente a 26.3 s en V17. Es una sola medida de cada ejecución y depende del mismo modo del proveedor del modelo.

### Arranque en frío

| Momento (UTC) | Inactividad | Reloj del cliente | Servidor | Resultado |
|---|---|---|---|---|
| 2026-11-12 09:47:17 | 12.6 min | 12 829.9 ms | 6575.5 ms | búsqueda solo léxica (degradada) |

Neon informó de que el cómputo estaba suspendido desde las 09:39:47 UTC (última actividad a las 09:34:13), y `/health` justo después de la búsqueda informó de 10.8 s de actividad de la instancia, de modo que la instancia de la función y la base de datos estaban frías. La búsqueda se degradó a solo léxica como en la primera observación de V17: la petición de embeddings agotó el plazo dos veces en el proveedor (`POST embeddings: ReadTimeout (after 2 attempts)`), la consulta léxica tardó 21.6 ms y el reranking 673.6 ms. El reloj del cliente marcó 12.8 s frente a 12.6 s en V17. Es una sola observación, y no se leyó el registro de ejecución de Vercel para repartir el tiempo de arranque.

### Uso del presupuesto diario por esta tarea

Contador `usage_budget.daily_usage`, 2026-11-12: las búsquedas con modelos pasaron de 5 a 49 (44 de ellas por estas comprobaciones: 4 búsquedas de obras añadidas, 38 de la comprobación del despliegue, 1 búsqueda MCP y 1 de arranque en frío), y las peticiones al LLM de 0 a 1 (la búsqueda con explicaciones). Los límites son 400 y 60 al día.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Cargar más de 50 000 obras | 50 000 fue el objetivo de diseño: unos 694 MiB con los bytes por obra de V17, bajo el techo propio de 800 MiB sobre el límite de 1 GiB, y deja espacio para los contadores diarios de uso, la hinchazón de las tablas y las tablas temporales de la evaluación de V14. Pasarse de ahí reduce esa holgura. Las fuentes no calculan qué tamaño máximo cabría |
| Mantener el cargador de V17 y su `VACUUM FULL` | Cada obra y cada fila de texto dejaban una fila muerta (13.8 MiB en 5000 obras) y `VACUUM FULL` reescribe la tabla y necesita espacio temporal para la copia, que a este tamaño podía cruzar el límite del plan. El cargador nuevo no deja filas muertas, y no hace falta esa orden |
| Un plan de pago de Neon | Va contra el criterio del autor de preferir opciones gratuitas o de muy bajo coste, y 50 000 obras caben en el plan gratuito. V17 ya había dejado el plan de pago como decisión del autor para un catálogo mayor |
| Sustituir la selección de 30 000 por otra de 50 000 en lugar de ampliarla | Habría obligado a reescribir las obras ya cargadas y a volver a calcular sus vectores, con su coste en peticiones, y habría cambiado los resultados de lo ya desplegado. Con la selección ampliada, las 30 000 obras se conservan con los mismos registros (sus hashes de contenido coincidieron). Esta razón es una deducción de lo anterior, no una medida |
| Reconstruir el índice vectorial junto al que está en uso, como el BM25 | Habría hecho que la base de datos mantuviera a la vez dos índices BM25 y el vectorial. Se eliminó primero el vectorial, a costa de unos 8 a 11 s por etapa de exploración exacta, para que el pico de tamaño no superara la guarda |
| Reconstruir el índice BM25 en su sitio (borrarlo y crearlo de nuevo) | La búsqueda léxica se habría quedado sin índice durante la construcción. Se construye junto al viejo y se cambia en una sola transacción (4.3 a 5.5 s de construcción) |

## Cómo se ha validado

- Medida del despliegue real, repetible con `scripts/deployment_check` (arriba).
- Contadores de escritura y `n_dead_tup` de Postgres sobre toda la carga, comprobaciones de integridad de la base de datos y `EXPLAIN ANALYZE` de las consultas de la API.
- Cuatro obras añadidas, encontradas en primer lugar por la API pública.
- Tests añadidos en el commit: `tests/infrastructure/test_load_postgres_script.py` (etapas que conservan lo ya cargado), `tests/infrastructure/test_postgres_rows_and_schema.py` (reconstrucción de BM25 junto al índice en uso y nombres de índice admitidos), `tests/infrastructure/test_postgres_search_adapters.py` (escritura única de cada fila), `tests/ingestion/test_catalog_selection_and_report.py` (la selección mayor contiene a la menor), `tests/integration/test_postgres_cloud_profile.py` (ampliar un catálogo cargado) y `tests/evaluation/test_deployment_check.py` (tabla de latencia antes y después). Las pruebas de integración se omiten sin cadena de conexión de prueba.

## Criterios de diseño aplicados

1. La ampliación se hace dentro del plan gratuito, bajo un techo propio de 800 MiB y con una guarda en el cargador de 780 MiB, comprobada tras cada lote de 500 obras.
2. Cada fila se escribe una vez en su forma final; una carga no deja filas muertas.
3. Una carga se puede repetir o retomar con la misma orden y no reescribe lo que no ha cambiado.
4. Ampliar un catálogo vivo no puede dejar sin índice la búsqueda léxica.
5. El perfil local sigue siendo el evaluado; esta tarea no cambia el código de búsqueda ni los límites de gasto de V17.

## Límites que debe conocer el tribunal

- **El perfil desplegado no es el evaluado.** Las cifras de calidad de la memoria proceden del perfil local sobre la colección de 177 libros. El sistema desplegado, con 50 000 obras y otros modelos (embeddings y reranker por API), no se ha evaluado todavía; las doce consultas del informe son una inspección de los tres primeros resultados, no una medida de calidad. Esa evaluación es V14.
- **La comparación de latencia no es controlada.** Las ejecuciones están separadas por 11 horas y dependen de la carga de los proveedores de modelos. No muestra que el tamaño del catálogo no influya, y no muestra que la ampliación acelere las búsquedas.
- **Arranque en frío.** Una sola observación después de la ampliación (12.8 s, búsqueda degradada a solo léxica por un fallo del proveedor de embeddings), sin repartir el tiempo con el registro de ejecución. Las observaciones de V17 (12.6 s y 22.3 s) tampoco son una distribución. Conviene abrir la página, o llamar a `/api/v1/health`, un minuto antes de una demostración.
- **Comportamiento durante la carga.** Lo descrito para los segundos sin índice vectorial es una inferencia; no se observó ninguna búsqueda pública en esos intervalos.
- **Cuotas sin llenar.** El español (7231 de 12 500), el catalán (971 de 1750) y el chino (1617 de 1750) no llegan a su cuota, porque el catálogo de fuentes no tiene más obras en esos idiomas.
- **Datos del catálogo.** Cobertura de `isbn13` del 4.5 % y de año del 68.3 %; 4355 descripciones principales son resúmenes generados automáticamente por Project Gutenberg. Las licencias de las descripciones son CC BY-SA 3.0 o 4.0 (con obligación de compartir igual), CC0 1.0 y, para el catálogo de Gutenberg, sin licencia explícita.
- **Consultas sin respuesta.** Las dos consultas del informe sin buena respuesta en el catálogo (un manual del Seat Ibiza y un controlador de entrada de Kubernetes) siguen devolviendo diez resultados no relacionados; el sistema no tiene un umbral de "nada relevante". Algunas consultas en alemán, italiano y portugués devuelven en los primeros puestos obras en otro idioma.
- **Chino y japonés.** V12 documentó que Postgres no trae lematizador para estos idiomas y que su texto no se divide en palabras, por lo que la recuperación léxica en ellos es pobre. El catálogo tiene ahora 1750 obras en japonés y 1617 en chino; no se ha medido esa limitación en este catálogo.
- **Margen de tamaño.** La base de datos ocupa 626.4 MiB, bajo el techo propio de 800 MiB y el límite del plan de 1 GiB; el espacio que queda es el que necesitan los contadores diarios, la hinchazón y la evaluación de V14.
- **Presupuesto compartido, límites por cliente y modelo de embeddings alojado.** Se mantienen como se describen en V17: 400 búsquedas con modelos y 60 peticiones al LLM al día para todos los visitantes, y vectores ligeramente distintos entre llamadas repetidas al proveedor.

## Trabajo pendiente

- **V14**, la evaluación sobre el catálogo desplegado: consultas ampliadas y multilingües, juicios de relevancia y la medida de calidad que hoy falta.
- Repetir la medida del arranque en frío, con el registro de ejecución de Vercel, para tener más de una observación.
- Si se repite la comparación de latencia, hacerlo en ejecuciones más próximas en el tiempo.
- Las tareas pendientes de V17 que no son de esta nota (límite de crédito en la clave de OpenRouter) siguen como estaban.

## Cómo reproducirlo

```
# Seleccionar el catálogo de 50 000 obras (contiene las 30 000 anteriores)
python -m scripts.build_catalog select --size 50000 --from data/catalog_core/catalog.db --out data/catalog_deploy/catalog.db
python -m scripts.build_catalog report --out data/catalog_deploy/catalog.db

# Cargar por etapas (idempotente; cada etapa conserva las obras ya cargadas)
SEARCH_PROFILE=cloud python -m scripts.load_postgres --from data/catalog_deploy/catalog.db --schema catalog_deploy \
    --no-gin --no-text-bodies --embed-workers 4 --max-database-mib 780 --limit 35000
# después --limit 40000, --limit 45000 y una vez más sin --limit;
# tras una pausa o un fallo, se ejecuta de nuevo la misma orden
python -m scripts.load_postgres --schema catalog_deploy --report

# Redesplegar (desde la raíz del repositorio, enlazada con `vercel link`)
vercel deploy --prod --yes

# Comprobar el despliegue (el modo cold, tras 10 minutos de inactividad o más)
python -m scripts.deployment_check run --base-url https://ai-book-recommender-one.vercel.app --catalog-works 50000
python -m scripts.deployment_check cold --base-url https://ai-book-recommender-one.vercel.app --idle-minutes N --catalog-works 50000
python -m scripts.deployment_check report
```

- Cuando el directorio de trabajo tiene cambios sin confirmar que no deben publicarse, se despliega una copia limpia: `git archive HEAD | tar -x -C <directorio vacío>`, se copian los cambios que se quieren y `.vercel/`, y se ejecuta la misma orden de despliegue allí.
- `report` no necesita red ni base de datos.

## Ficheros

- **Infraestructura:** `app/infrastructure/db/postgres_book_catalog_repository.py` (escritura de cada fila en su forma final, vector y `tsvector` en el `INSERT`), `app/infrastructure/db/postgres_schema.py` (reconstrucción del índice BM25 junto al que está en uso y cambio en una transacción), `app/infrastructure/db/postgres_schema.sql` (nombre del índice BM25 como parámetro).
- **Interfaz:** `app/ui/static/index.html` (texto "unas 50 000 obras").
- **Scripts:** `scripts/load_postgres.py` (etapas que conservan lo ya cargado, escritura única, guarda de tamaño, reconstrucción de índices en vivo), `scripts/deployment_check.py` (tabla de latencia antes y después, `--catalog-works`).
- **Pruebas:** `tests/infrastructure/test_load_postgres_script.py`, `tests/infrastructure/test_postgres_rows_and_schema.py`, `tests/infrastructure/test_postgres_search_adapters.py`, `tests/ingestion/test_catalog_selection_and_report.py`, `tests/integration/test_postgres_cloud_profile.py`, `tests/evaluation/test_deployment_check.py`.
- **Informes y datos:** `data/evaluation/DEPLOYMENT_CHECK.md`, `data/evaluation/results_deployment_check.json`, `data/catalog_deploy/CATALOG_REPORT.md` y `data/catalog_deploy/footprint_v18_stages.json` (locales, no versionados).
