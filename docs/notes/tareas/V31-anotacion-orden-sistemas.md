# V31 · Anotación humana que puede ordenar los sistemas: protocolo, hoja ciega y código de análisis

- **Fecha:** 2026-11-14
- **Estado:** fase 1 hecha (preparación). **Nadie ha puesto ninguna nota todavía.** Queda abierto que el autor rellene la hoja y que se ejecute el análisis (véase «Pendiente»)
- **Capa:** evaluación (`app/evaluation/catalog_evaluation/annotation_order.py`, `app/evaluation/catalog_evaluation/cli.py`), datos (`data/evaluation/catalog50k/annotation_order/`), documentación (`docs/notes/`)
- **Commit:** `5468cd0` (protocolo, hoja, clave, instrucciones, código y tests, antes de anotar nada)

## Qué se ha hecho

Esta fase solo prepara la anotación. No se ha hecho ninguna búsqueda nueva ni ninguna llamada a un modelo: todo sale de los ficheros ya guardados de V14.

- Se escribió el protocolo `data/evaluation/catalog50k/annotation_order/PROTOCOL.md`, que fija qué se anota, qué se calcula con las notas y cómo se lee el resultado. Se guardó en git (`5468cd0`) antes de que nadie anotara.
- Se generó la hoja ciega `annotation_order_sheet.csv`, con **607 pares** (consulta, obra) de **56 consultas**, repartidos en dos bloques. Cada par es una obra que alguno de los sistemas S2, S3, S4 o S5 de V14 pone entre sus cinco primeros resultados para esa consulta. La hoja no muestra el sistema, la posición ni ninguna nota de los jueces.
- Se generó la clave `annotation_order_key.json` (para cada par: consulta, necesidad, idioma, estrato, bloque, obra, posición en cada sistema que la devolvió y, si lo hay, el identificador del mismo par en la hoja de 200 pares), que el anotador no abre; `sample.json` (recuentos, estimación de tiempo y cálculo de potencia); e `INSTRUCCIONES.md`, en español, para quien anota.
- Se escribió `app/evaluation/catalog_evaluation/annotation_order.py` y las dos órdenes de `cli.py`: `order-sheet` genera la hoja y sus ficheros, y `order-agreement` ejecuta el análisis declarado sobre la hoja rellenada. Ninguna de las dos usa la red.
- En `annotation.py` se separaron, sin cambiar su resultado, el texto de la obra (`book_cells`) y la lectura de una columna de la hoja (`read_cells`, `read_sheet` con columna). El acuerdo de la hoja de 200 pares recalculado con ese código es idéntico al guardado.
- Tests: `tests/evaluation/test_annotation_order.py`, 18 funciones de test, sin red ni base de datos.

### Las consultas

Son elegibles las consultas de los estratos A y B de V14 escritas en español o en inglés, que el anotador lee sin traducir: 56 (inglés: 24 de A y 6 de B; español: 20 de A y 6 de B). Se reparten en dos bloques:

| Idioma | Disponibles A | Disponibles B | Bloque 1: A | Bloque 1: B | Bloque 2: A | Total |
|---|---|---|---|---|---|---|
| en | 24 | 6 | 9 | 6 | 15 | 30 |
| es | 20 | 6 | 9 | 6 | 11 | 26 |
| Total | 44 | 12 | 18 | 12 | 26 | 56 |

El bloque 1 (obligatorio) toma 30 consultas, repartidas entre los dos idiomas a partes iguales, con todas las de B y el resto de A elegido con semilla 0. Se prefiere B porque las consultas de A son las 44 de `app/evaluation/test_queries.json`, que ya se usaron para elegir modelos y parámetros en V06, V10 y V16. El bloque 2 (opcional) son las otras 26, todas de A.

| Bloque | Consultas | Pares | Pares por consulta (mín., mediana, máx.) | Ya en la hoja de 200 pares | Tiempo a 30-45 s por par |
|---|---|---|---|---|---|
| 1 | 30 | 313 | 8, 10, 14 | 15 | 156 a 235 min (2,6 a 3,9 h) |
| 2 | 26 | 294 | 7, 11,5, 15 | 14 | 147 a 220 min (2,5 a 3,7 h) |
| Total | 56 | 607 | | 29 | |

### Los sistemas y los contrastes

Los cuatro sistemas principales de V14, con sus ficheros de ejecución guardados (`data/evaluation/catalog50k/runs/S2.jsonl` a `S5.jsonl`):

| Sistema | Definición |
|---|---|
| S2 | Vectorial, `qwen/qwen3-embedding-8b` |
| S3 | Híbrido: RRF (k = 60, pesos iguales) de los 60 primeros de BM25 y de S2 |
| S4 | S2 + `qwen/qwen3-reranker-8b` sobre los 30 primeros |
| S5 | S3 + el mismo reranker sobre los 30 primeros (la configuración desplegada en V14) |

Se comparan H1 = S3 − S2 (¿la fusión híbrida añade algo a la búsqueda vectorial sola?) y H3 = S5 − S4 (con reranker, ¿añaden algo los candidatos de BM25?). En V14 los dos fueron negativos con los dos jueces.

### Cómo es la hoja

- Columnas: `pair_id`, `block`, `query`, `need`, `title`, `authors`, `book_language`, `year`, `description`, `grade`, `grade_2`. El texto de la obra sale de la misma función que en la hoja de 200 pares. Respecto a aquella se quita `language` (todas las consultas están en español o inglés) y se añaden el idioma de la obra y su año (vacío en 68 filas del bloque 1 y 66 del bloque 2), que también vieron los jueces. Ninguna fila tiene la descripción vacía.
- Orden: primero el bloque 1 y después el 2; cada bloque se baraja por separado con `random.Random(0)`, por consultas y, dentro de cada consulta, por obras, de modo que el anotador lee una consulta cada vez. Los `pair_id` son `r001` a `r313` (bloque 1) y `r314` a `r607` (bloque 2). Volver a generar la hoja da ficheros idénticos byte a byte, y las filas del bloque 1 no cambian al añadir el bloque 2 (hay un test para cada cosa).
- La escala de 0 a 3 y sus dos reglas tienen la misma redacción que en la hoja anterior, para que las dos anotaciones sean comparables. Cambia la regla de la celda vacía: antes se dejaba vacía si no se podía decidir (16 de 200 quedaron vacías); ahora solo si la obra no se puede valorar con el texto mostrado, y si se duda entre dos notas se elige una. El motivo es que aquí un par vacío sale del nDCG de todos los evaluadores.

### El análisis declarado

Se ejecuta **una sola vez**, cuando el anotador entrega la hoja, nunca entre el bloque 1 y el 2. Calcula, con la función de nDCG de `app/evaluation/evaluation_service.py` y las de `app/evaluation/statistics.py` (10 000 remuestreos, semilla 0):

- nDCG@5 por sistema con las notas de la persona y, sobre las mismas consultas y pares, con las de cada juez;
- H1 y H3 con diferencia media, intervalo del 95 % por bootstrap pareado y p de la prueba de aleatorización pareada, sin Holm; son descriptivos;
- si el signo de H1 y de H3 coincide con el de cada juez, y la tau-b de Kendall entre el orden de los cuatro sistemas de la persona y el de cada juez;
- el acuerdo por par de la persona con cada juez (kappa ponderado cuadrático, acuerdo exacto y a menos de un punto), el acuerdo del anotador consigo mismo en los 29 pares que ya estaban en la hoja de 200, y, si hay segunda persona, su acuerdo con la primera y las mismas cifras con sus notas.

La regla de lectura, escrita antes de anotar: si el signo de la diferencia con la persona es el mismo que con cada uno de los dos jueces, la memoria puede decir que ese resultado se mantiene en esta muestra con las notas de una persona; si se cumple en H1 y en H3, que el orden de los sistemas se mantiene. Si un signo es distinto, o la diferencia es exactamente 0, la conclusión correspondiente de V14 queda como no confirmada por una persona. En los dos casos se informa el resultado completo. Que un signo coincida no significa que la diferencia sea significativa; la tau y el acuerdo por par se informan pero no cambian la regla.

Qué consultas cuentan: en el bloque 1 entra toda consulta con algún par con nota. En el bloque 2 solo las completas, porque al ser opcional una celda vacía puede significar que el anotador se paró. Si las 26 del bloque 2 están completas, el análisis principal usa las 56 consultas y el bloque 1 pasa a tabla secundaria; si no, el principal es el bloque 1 y lo completado del bloque 2 se informa aparte.

## Por qué

Una revisión externa de la memoria (2026-11-13) señaló que la primera anotación del autor, 200 pares sacados al azar y estratificados por la nota del juez principal (V14, adenda del 2026-11-13), no sirve para recalcular ninguna métrica de ningún sistema: no contiene las listas de ningún sistema, sino pares sueltos elegidos por su nota. Por tanto no puede decir si el orden de los sistemas se mantiene con las notas de una persona. La revisión pidió una anotación de los primeros resultados de los sistemas principales sobre un conjunto de consultas, idealmente con dos anotadores.

El acuerdo medido con aquella hoja fue bajo (kappa ponderado cuadrático 0.272 y 0.301 en 184 pares, con el autor dando nota 3 solo 4 veces), y V14 dejó abierto «recalcular las métricas y los contrastes con notas de personas». Esta tarea prepara esa medida. La pregunta es una sola: **¿se mantiene con las notas de una persona el orden de los sistemas que dan los jueces, en particular el signo de H1 y de H3?** No pregunta si los jueces aciertan par a par, aunque el acuerdo por par también se calcula.

El experimento se diseñó después de conocer los resultados de V14 y la anotación de 200 pares. Lo dicen el protocolo y esta nota, y lo dirá el informe.

## Objetivo

Que una persona ponga notas a los cinco primeros resultados de S2, S3, S4 y S5 sobre consultas en español e inglés, sin saber qué sistema devolvió cada obra, y que con ellas se pueda ordenar los sistemas y comparar ese orden con el de los jueces con un análisis y una regla de lectura fijados antes de anotar, de modo que el resultado no pueda elegir la pregunta.

## Por qué dos bloques

Treinta consultas llevan unas media jornada de trabajo y dan poca potencia. Las 56 elegibles dan más potencia para H1 pero duplican el tiempo. Por eso el bloque 1 (30 consultas) es obligatorio y el bloque 2 (26) es opcional. La decisión de hacer el bloque 2 se toma sin ver ningún resultado: el análisis se ejecuta una sola vez al entregar la hoja, y la hoja no muestra el sistema, la posición ni las notas de los jueces. Por eso la elección del conjunto principal no puede depender de lo que salga.

## Por qué no otras alternativas

| Alternativa | Por qué no |
|---|---|
| Reutilizar la hoja de 200 pares | Su muestreo, por nota del juez, no contiene las listas de ningún sistema y no puede ordenar sistemas |
| Consultas en los doce idiomas | El anotador lee español e inglés sin traducir; la anotación anterior en otros idiomas se apoyó en traducción, y con ella no se puede valorar la naturalidad ni los matices de la obra |
| Anotar a profundidad 10 | Daría el doble de pares para el mismo número de consultas. Se prefirió profundidad 5 con más consultas |
| Mostrar las notas de los jueces o los sistemas | Sesgaría al anotador |
| Reclutar anotadores en una plataforma de pago | Coste y tiempo antes de la defensa. La hoja admite una segunda persona (columna `grade_2` o `--second-sheet`) si hay alguien disponible |
| Evaluar en una colección externa con juicios humanos | Es otro catálogo y otros idiomas; queda como trabajo futuro |

## Potencia

La diferencia mínima detectable de una prueba pareada de dos colas con nivel 0,05 y potencia 0,8 es (t_{0,975; n−1} + t_{0,80; n−1}) × s / √n, con s la desviación típica por consulta de las diferencias. Con 30 consultas es 0,529 × s aprox. y con 56, 0,381 × s aprox. Se calculó antes de anotar, con las diferencias de nDCG@5 de cada juez sobre esas consultas (`sample.json`, bloque `power`):

| Consultas | Contraste | Juez | Diferencia media de los jueces | Desviación típica por consulta | Diferencia mínima detectable |
|---|---|---|---|---|---|
| 30 (bloque 1) | H1 (S3 − S2) | `z-ai/glm-5.3-flash` | −0,110 | 0,300 | 0,159 |
| 30 (bloque 1) | H1 (S3 − S2) | `minimax/minimax-m3.1-flash` | −0,078 | 0,245 | 0,130 |
| 30 (bloque 1) | H3 (S5 − S4) | `z-ai/glm-5.3-flash` | −0,005 | 0,081 | 0,043 |
| 30 (bloque 1) | H3 (S5 − S4) | `minimax/minimax-m3.1-flash` | −0,022 | 0,076 | 0,040 |
| 56 (bloques 1 y 2) | H1 (S3 − S2) | `z-ai/glm-5.3-flash` | −0,065 | 0,272 | 0,104 |
| 56 (bloques 1 y 2) | H1 (S3 − S2) | `minimax/minimax-m3.1-flash` | −0,070 | 0,201 | 0,076 |
| 56 (bloques 1 y 2) | H3 (S5 − S4) | `z-ai/glm-5.3-flash` | −0,031 | 0,118 | 0,045 |
| 56 (bloques 1 y 2) | H3 (S5 − S4) | `minimax/minimax-m3.1-flash` | −0,034 | 0,163 | 0,062 |

En los dos conjuntos las diferencias de los propios jueces quedan por debajo de la diferencia mínima detectable, en H1 y en H3. Aunque la persona coincida con los jueces, lo esperable es que sus diferencias no sean significativas. Por eso la p y el intervalo son descriptivos y la regla de lectura se apoya en el signo y en el acuerdo de orden. H3 es frágil en el bloque 1: con los jueces su diferencia es casi nula (−0,005 y −0,022), y un signo que coincide es ahí una prueba débil. Con las 56 consultas es mayor (−0,031 y −0,034).

## Límites

- **Un solo anotador, y es el autor del sistema**, salvo que una segunda persona participe. Conoce los sistemas, las conclusiones de V14 y este protocolo, aunque la hoja no le muestra el sistema, la posición ni las notas de los jueces.
- **Solo español e inglés.** No dice nada de los otros diez idiomas del catálogo.
- **Estrato A.** 18 de las 30 consultas del bloque 1 y las 26 del bloque 2 son del estrato A, que ya se usó para elegir modelos y parámetros; si se hace el bloque 2, 44 de las 56 consultas del análisis principal son de A. Las 12 de B las escribió un LLM (V14).
- **Potencia baja** con 30 y con 56 consultas: las diferencias mínimas detectables de la tabla anterior son mayores que las diferencias que dan los jueces en esas consultas. La lectura descansa en el signo de las diferencias y en el acuerdo de orden, como dice la sección 8 del protocolo.
- **Profundidad 5.** El nDCG@5 de esta hoja, con la ordenación ideal formada con la unión de los cuatro top 5, no es el nDCG@10 de V14 y no se compara con él.
- **Una obra no valorable en el bloque 2 deja fuera su consulta entera.** Es el precio de no confundir una celda vacía con una consulta sin terminar.
- **Los jueces vieron las categorías de la obra; la persona no**, como en la hoja anterior.
- **Las notas del autor pueden tener otra escala.** En la hoja anterior casi no usó la nota 3; la comparación es del orden y del signo, no de los valores.
- **El acuerdo del anotador consigo mismo** mezcla su estabilidad con los cambios de la hoja (idioma y año de la obra, regla de celda vacía más estricta), y 15 o 29 pares son muy pocos.
- **El experimento se diseñó después de conocer los resultados de V14** y de la anotación de 200 pares.

## Pendiente

- **El autor rellena el bloque 1** y, si decide hacerlo, el bloque 2, y guarda la hoja como `annotation_order_sheet_filled.csv`. Opcionalmente una segunda persona anota `grade_2` sobre una copia en blanco.
- **Ejecutar el análisis una vez** con `order-agreement` (véase «Cómo usarlo») y **actualizar la memoria con el resultado, sea cual sea**, aplicando la regla de lectura del protocolo. Hasta entonces la memoria no puede afirmar nada sobre el orden de los sistemas con notas de personas.
- Un segundo anotador sigue abierto, como ya estaba en V14.

## Cómo usarlo

Generar la hoja (ya hecho; lee la copia local del catálogo, las ejecuciones y las cachés de juicios; sin red):

```
.venv/bin/python -m app.evaluation.catalog_evaluation order-sheet
```

Para anotar: abrir `annotation_order_sheet.csv` en una hoja de cálculo, poner una nota de 0 a 3 en `grade` para cada fila siguiendo `INSTRUCCIONES.md`, sin abrir la clave ni los informes, y guardar como CSV UTF-8 con el nombre `annotation_order_sheet_filled.csv` en la misma carpeta. La hoja en blanco se queda como se generó.

Analizar (una sola vez, sin red; antes de calcular nada comprueba que la clave coincide con los ficheros de ejecución, y se detiene si no):

```
.venv/bin/python -m app.evaluation.catalog_evaluation order-agreement --sheet data/evaluation/catalog50k/annotation_order/annotation_order_sheet_filled.csv [--second-sheet <copia de la segunda persona>]
```

Escribe `order_agreement.json` en la misma carpeta (opción `--output` para otra ruta). El informe cuenta las celdas vacías y no válidas por bloque y por consulta y lista las no válidas por `pair_id`.

Tests: `.venv/bin/python -m pytest tests/evaluation/test_annotation_order.py`.

## Ficheros

- Código: `app/evaluation/catalog_evaluation/annotation_order.py`, `app/evaluation/catalog_evaluation/cli.py` (órdenes `order-sheet` y `order-agreement`), `app/evaluation/catalog_evaluation/annotation.py` (`book_cells`, `read_cells`).
- Tests: `tests/evaluation/test_annotation_order.py`.
- Datos (`data/evaluation/catalog50k/annotation_order/`): `PROTOCOL.md`, `INSTRUCCIONES.md`, `annotation_order_sheet.csv`, `annotation_order_key.json`, `sample.json` y, cuando vuelva la hoja, `annotation_order_sheet_filled.csv` y `order_agreement.json`.
- Entradas que lee: `data/evaluation/catalog50k/runs/S2.jsonl` a `S5.jsonl`, las cachés de juicios de V14 y `data/evaluation/catalog50k/annotation/annotation_sheet_filled.csv` (la hoja de 200 pares rellenada).
- Relacionadas: V14 (`V14-evaluacion-catalogo-desplegado.md`, adenda del 2026-11-13), V27 (`V27-hoja-revision-consultas.md`), V29 (`V29-alcance-conclusion-bm25.md`) y `docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`.
