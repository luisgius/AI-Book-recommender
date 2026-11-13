# V27 · Hoja de revisión de las consultas y peticiones generadas por un LLM

- **Fecha:** 2026-11-13
- **Estado:** hoja y herramienta hechas. Cuando se escribió esta nota la revisión no estaba hecha; **el autor la hizo el 2026-11-13** y su resultado está en la sección «Resultado de la revisión (2026-11-13)». Ese mismo día se recalcularon las dos evaluaciones sin las filas dudosas: véase «Análisis de sensibilidad sin las filas dudosas (2026-11-13)», al final. El texto de esta nota anterior a esas secciones se conserva como estaba el día en que se escribió
- **Capa:** evaluación (`app/evaluation/query_review.py`), datos (`data/evaluation/query_review/`), documentación (`docs/notes/`)
- **Commits:** sin guardar en git cuando se escribió esta nota

## Qué se ha hecho

- Se escribió `app/evaluation/query_review.py`, con dos órdenes. `sheet` construye la hoja de revisión y sus instrucciones a partir de los ficheros ya publicados de las dos evaluaciones. `summary` lee la hoja rellenada, comprueba que es válida y cuenta los veredictos por evaluación y estrato.
- Se generó la hoja `data/evaluation/query_review/review_sheet.csv`, con **254 filas**, una por cada consulta o petición que escribió un LLM, y las instrucciones en español `data/evaluation/query_review/INSTRUCCIONES.md`.
- Se escribieron los tests de la herramienta en `tests/evaluation/test_query_review.py` (sin red; solo usan carpetas temporales).

Qué filas tiene la hoja:

| Evaluación | Estrato | Filas | Qué son |
|---|---|---|---|
| catálogo | B | 72 | consultas nuevas en doce idiomas (V14) |
| catálogo | C | 40 | diez necesidades en cuatro idiomas (V14) |
| catálogo | D | 12 | peticiones sin respuesta posible en un catálogo de libros (V14) |
| asistente | E | 30 | peticiones vagas, esperan una pregunta (V22) |
| asistente | F | 20 | peticiones de varios pasos sobre obras del catálogo (V22) |
| asistente | G | 80 | conjunto reservado de V25: 60 válidas sobre temas no literarios y 20 fuera de alcance |

Cada fila lleva el identificador, la evaluación, el estrato, el idioma, el tipo, el texto que recibe el sistema, la necesidad de información (`need`), un campo `context` con lo que hace falta para revisarla (la obra de una consulta de obra conocida, la consulta de B de la que sale una petición de E, las obras de una petición de F, el campo y la traducción al inglés del generador en G) y dos columnas vacías: `verdict` y `comment`. El veredicto es `ok`, `dudosa` o `mal`, o vacío si la fila no se ha revisado. Las `dudosa` y las `mal` se explican en `comment`.

## Por qué

El protocolo de la evaluación del catálogo (`docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`, sección 3) dice de los estratos B, C y D, que escribe un LLM: «Es una amenaza a la validez declarada (sección 11): el autor debe revisar la lista». Las notas de V22 y de V25 piden lo mismo para las peticiones E, F y G del asistente, y dejaron «Revisión por el autor» como pendiente. Nadie lo había hecho.

Antes de esta tarea el autor no tenía una forma práctica de hacerlo: había que abrir varios ficheros JSON de dos carpetas, con campos distintos en cada uno. La hoja pone todas las filas en un solo CSV que se abre en una hoja de cálculo, con lo que hace falta para juzgar cada una, y la herramienta convierte el resultado en recuentos que el texto del TFG puede citar.

## Objetivo

Que el autor pueda revisar con una sola hoja todo lo que escribió un LLM y entró en una de las dos evaluaciones, y que lo revisado quede contado por estrato con la lista de lo dudoso y lo malo y su comentario. La hoja no cambia ninguna consulta ni ninguna petición; si el autor marca algo como `mal`, qué hacer con ello (sustituirlo, repetir una medida) es una decisión posterior y declarada.

## Decisiones de diseño

- **Una fila por consulta o petición, y cada una una sola vez.** Los conjuntos B y D del asistente son los estratos B y D del catálogo, así que se revisan una vez, como filas `catalogo`. E y F son iguales en las dos medidas del asistente (V22 y V25) y aparecen una vez; la herramienta falla si una petición difiere entre las dos medidas o si un mismo identificador nombra dos cosas distintas.
- **El estrato A no está**: son las consultas originales de pruebas, que no escribió un LLM.
- **La hoja nunca se sobrescribe si contiene veredictos o comentarios**: `sheet` se niega a escribir encima de una hoja con trabajo hecho.
- **`summary` rechaza una hoja si falta, sobra o cambió algún `item_id`, o si un veredicto no es `ok`, `dudosa`, `mal` o vacío**, y acepta una hoja a medias. Lee el CSV con o sin marca de orden de bytes (`utf-8-sig`) y con comas o con puntos y coma (así guarda Excel con configuración regional española). La hoja se escribe con la marca de orden de bytes para que Excel lea bien las filas en árabe, chino, japonés y ruso.
- **Las instrucciones dicen qué mirar en cada estrato**, con lo que el protocolo pide a cada uno: idioma correcto, correspondencia con `need` y que pudiera escribirlo una persona (todos); la obra descrita sin palabras de su título (B, obra conocida); que las cuatro versiones pidan exactamente lo mismo (C); que ningún libro pudiera responder (D, y las 20 fuera de alcance de G); que la petición sea vaga pero compatible con `need` (E); que nombre las obras y sea del tipo pedido (F). Para quien no lea el idioma de una fila, basta comprobar que corresponde a `need` con un traductor o con la traducción del generador (G).

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Revisar solo una muestra | El protocolo dice «la lista» y la hoja tiene 254 filas, que son un trabajo razonable para una persona; con una muestra, un problema en un estrato pequeño (D tiene 12, F tiene 20) podría quedar sin ver. Además, la hoja admite filas vacías: el autor puede revisar menos, y el resumen dice cuántas se revisaron |
| Pedir la revisión a otro LLM | No sería la revisión por una persona que piden el protocolo y las notas de V22 y V25. La amenaza que se declara es que un LLM escribió las consultas; que otro LLM las valore no la reduce con la misma garantía y se podría confundir con ella |
| Una hoja ciega, como la de anotación de V14 | La hoja de anotación oculta las notas de los jueces porque hay una nota que podría sesgar a quien anota. Aquí no hay ninguna nota que ocultar: el revisor necesita ver el tipo, la necesidad y el contexto para decidir si la fila cumple lo que se pidió al generador |
| Corregir o sustituir las filas malas en la misma herramienta | Cambiaría las consultas de evaluaciones ya medidas sin que quede declarado. La herramienta solo registra el veredicto; qué se hace con una fila mala se decide después, a la vista del recuento |

## Cómo se ha validado

Los tests de `tests/evaluation/test_query_review.py` comprueban, con ficheros pequeños en carpetas temporales: que la hoja sale completa y es determinista a partir de los ficheros publicados; el orden y el contexto de las filas y que E y F salen una sola vez; que una petición que difiere entre medidas es un error; que una hoja con veredictos no se sobrescribe; los recuentos y la lista de dudosas y malas; que una hoja con marca de orden de bytes o con puntos y coma se lee igual; que se rechazan identificadores cambiados o veredictos desconocidos, una columna ausente, un fichero que no existe y una codificación que no es UTF-8; y que las dos órdenes funcionan de punta a punta en las rutas indicadas.

## Límites

- **(Cuando se escribió la nota; ya hecha, véase el resultado al final.) La revisión no está hecha.** Esta tarea entrega la herramienta y la hoja; no hay ni un veredicto, y no se puede afirmar nada sobre la calidad de las consultas y las peticiones.
- **Revisa una sola persona, el autor**, que además conoce el diseño de la evaluación. No hay segundo revisor ni medida de acuerdo entre revisores.
- **Un veredicto es un juicio, no una medida.** `ok`, `dudosa` y `mal` se definen en las instrucciones, pero el límite entre `ok` y `dudosa` lo pone quien revisa.
- **Quien no lee un idioma de la hoja** (hay doce en B y seis en G, además de los de C y E) solo puede comprobar que el texto corresponde a la necesidad, con un traductor; no valora si suena natural, y la hoja lo dice.

## Pendiente

- **(Hecho el 2026-11-13, véase el resultado al final.) Que el autor rellene la hoja** y ejecute `summary`. El texto del TFG tiene un marcador pendiente para el resultado de esta revisión.
- **Decidir qué se hace con las filas `mal` y `dudosa`** cuando haya veredictos: si alguna cambia lo medido, hay que declararlo y repetir lo que haga falta. Esa decisión sigue sin tomarse: no hay filas `mal` y las 16 `dudosa` están en la sección final, con lo que queda abierto (recalcular sin ellas y segundo revisor). **Recalcular sin ellas: hecho el 2026-11-13, véase «Análisis de sensibilidad sin las filas dudosas (2026-11-13)», al final; no cambia ninguna decisión declarada.** Sigue abierto el segundo revisor.

## Cómo reproducirlo

```
.venv/bin/python -m app.evaluation.query_review sheet
.venv/bin/python -m app.evaluation.query_review summary --sheet data/evaluation/query_review/review_sheet.csv
```

El primero construye la hoja y las instrucciones (se niega a sobrescribir una hoja con veredictos). El segundo comprueba la hoja rellenada y escribe `data/evaluation/query_review/query_review.json`, con las filas, las revisadas y los veredictos por evaluación y estrato, y la lista de las `dudosa` y `mal` con su comentario; puede ejecutarse con la hoja a medias. Tests: `pytest tests/evaluation/test_query_review.py`.

## Ficheros

- Código: `app/evaluation/query_review.py`.
- Tests: `tests/evaluation/test_query_review.py`.
- Datos (`data/evaluation/query_review/`): `review_sheet.csv`, `INSTRUCCIONES.md` y, cuando se ejecute `summary`, `query_review.json`.
- Entradas que lee: `data/evaluation/catalog50k/queries/queries.json` (estratos B, C y D) y `data/evaluation/assistant/sets/` y `data/evaluation/assistant_v25/sets/` (conjuntos E, F y G).
- Protocolos y tareas relacionadas: `docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md` (sección 3), `docs/notes/2026-11-12-protocolo-evaluacion-asistente.md`, V14 (`V14-evaluacion-catalogo-desplegado.md`), V22 (`V22-evaluacion-asistente.md`) y V25 (`V25-revision-ruta-asistente.md`).

## Resultado de la revisión (2026-11-13)

El autor rellenó la hoja el 2026-11-13 y se ejecutó `summary`. Las cifras salen de `data/evaluation/query_review/query_review.json` y de `data/evaluation/query_review/review_sheet.csv`. Se revisaron **254 de 254** filas: **238 `ok`, 16 `dudosa` y 0 `mal`**.

| Evaluación | Estrato | Filas | Revisadas | ok | dudosa | mal |
|---|---|---|---|---|---|---|
| catálogo | B | 72 | 72 | 71 | 1 | 0 |
| catálogo | C | 40 | 40 | 39 | 1 | 0 |
| catálogo | D | 12 | 12 | 12 | 0 | 0 |
| asistente | E | 30 | 30 | 23 | 7 | 0 |
| asistente | F | 20 | 20 | 16 | 4 | 0 |
| asistente | G | 80 | 80 | 77 | 3 | 0 |

Ninguna fila se marcó `mal`: ninguna está en otro idioma, ninguna deja de corresponder a su necesidad y ninguna de D podría responderla un libro. Las 16 `dudosa`, agrupadas por el tipo de duda que el autor anotó en `comment`:

| Tipo de duda | Filas | Identificadores |
|---|---|---|
| Redacción poco natural o traducida literalmente (en `b-zh-6`, suena a reseña y no a búsqueda) | 7 | `b-zh-6`, `c09-fr`, `e-b-ca-1`, `f07`, `g18`, `g50`, `g54` |
| Petición vaga que añade una restricción que la necesidad no tiene (idioma o literatura de un país) | 3 | `e-b-ca-4`, `e-b-de-2`, `e-b-ja-4` |
| Petición vaga ambigua: una palabra suele entenderse como otro género o categoría y podría llevar a otro tipo de libro | 3 | `e-b-fr-5`, `e-b-pt-4`, `e-b-ru-5` |
| Título en inglés dentro de una petición en otro idioma (en `f03` y en `f07`, además, no se nombra al autor) | 2 | `f03`, `f04` |
| Título ambiguo: varias obras se llaman igual y la pregunta por el autor no basta para distinguirlas | 1 | `f20` |

Los tipos son una agrupación de esta nota; el veredicto y el comentario de cada fila están en `query_review.json`.

Límites, dichos sin adornos:

- Revisó **una sola persona, el autor**, que conoce el diseño de la evaluación. No hay segundo revisor ni acuerdo entre revisores.
- La revisión comprueba la forma (idioma, naturalidad, tipo) y la correspondencia con la necesidad. No mide si las consultas son representativas de las de usuarios reales.
- **Cuando se escribió esta sección no se había recalculado ninguna métrica sin las filas dudosas** (se hizo después, el mismo día: véase la sección siguiente). Las cifras publicadas de V14, V22 y V25 siguen siendo las de antes de la revisión; ni esta revisión ni el recálculo las cambian.

Quedaba abierto (el primer punto se hizo después, véase la sección siguiente):

- Recalcular las métricas sin las 16 filas dudosas (o declarar que no se hace) y ver si algo cambia.
- Una revisión por una segunda persona.

Para regenerar el resumen: `.venv/bin/python -m app.evaluation.query_review summary --sheet data/evaluation/query_review/review_sheet.csv`.

## Análisis de sensibilidad sin las filas dudosas (2026-11-13)

### Qué se ha hecho

- Se recalcularon las dos evaluaciones (catálogo V14; asistente V22 y su repetición V25) sin las 16 filas `dudosa`, para ver si las conclusiones dependen de ellas. Se hace con una orden nueva: `.venv/bin/python -m app.evaluation.query_review sensitivity`. Lee la hoja y los ficheros ya guardados; no usa red, ni LLM, ni base de datos.
- Código nuevo: `app/evaluation/review_sensitivity.py`, y la orden `sensitivity` en `app/evaluation/query_review.py`. Un cambio pequeño en `app/evaluation/assistant_evaluation/results.py`: `build_results` acepta ahora `exclude`, la lista de peticiones que se dejan fuera con sus turnos, su conversación y su ejecución del agente directo. Vacía, que es lo que usan los resultados publicados, no cambia nada.
- Ficheros que escribe, en `data/evaluation/query_review/`: `SENSITIVITY.md` (el informe, en inglés, como los demás informes de evaluación) y `sensitivity.json` (las decisiones, y cada cifra recalculada que cambia, con su valor publicado y el nuevo). No escribe nada más.
- Tests: `tests/evaluation/test_review_sensitivity.py`.

### Por qué

El autor lo pidió después de su revisión, para saber si las conclusiones de las evaluaciones dependen de las filas que marcó como dudosas. Hasta entonces solo se sabía cuántas eran y por qué, no si importaban.

### Cómo se protege de errores

- Con nada excluido, el código reproduce **exactamente** los resultados publicados (`results.json` de las tres medidas: en el catálogo, las métricas de A y B, los contrastes, el estrato C con los dos jueces, las conclusiones y las dos reglas de la sección 8; en el asistente, el fichero entero). Lo comprueba un test (`test_no_exclusion_reproduces_the_published_figures_exactly`) y lo comprueba la propia orden antes de informar: si una sola cifra no coincide, se detiene.
- Usa las mismas funciones, el mismo número de remuestreos (10.000), la misma semilla (0), el mismo nivel (0,05) y las mismas familias de Holm que los informes publicados. Rechaza (error) un identificador a excluir que una medida no tenga.

### Resultado

Cifras de `SENSITIVITY.md`; las cifras completas están en ese informe y en `sensitivity.json`.

| Evaluación | Elementos excluidos | Consultas o peticiones antes y después | Cifras recalculadas que cambian |
|---|---|---|---|
| catálogo (V14), estratos A y B | 2 (`b-zh-6`, `c09-fr`; la segunda es del estrato C) | 116 a 115 | 253 de 435 |
| asistente, primera medida (V22) | 12 | H7: 116 a 115; H8 (conjunto E): 30 a 23; H9 (conjunto F): 20 a 16 | 471 de 869 |
| asistente, repetición (V25) | 15 | igual que V22 (G tiene 3 excluidas) | 514 de 1010 |

- **No cambia ninguna decisión declarada** (57 comprobadas): cada contraste declarado mantiene su significación con cada juez y su conclusión, cada umbral de rutas mantiene su resultado y las dos reglas de la sección 8 del protocolo del catálogo mantienen el suyo.
- **Los contrastes que más se mueven** (diferencia de nDCG@10):
  - Catálogo: H4 (S2 menos S1), juez primario, de +0,347 a +0,342 (el que más se mueve; el resto cambia en milésimas).
  - Asistente V22: H9, segundo juez, de +0,095 a +0,144; con el segundo juez el intervalo deja de contener el cero, pero el p de Holm sigue siendo 0,125, así que no es significativo.
  - Asistente V25: H9, segundo juez, de +0,087 a +0,133, con p de Holm de 0,282 a 0,255 (tampoco significativo).
  - H8 (conjunto E, 30 a 23 peticiones): en V22 sigue siendo significativo (p de Holm de 0,002 a 0,016); en V25 sigue sin serlo (de 0,145 a 0,248 con el juez primario).
- **Los umbrales de rutas** no cambian: por ejemplo, en V22 las preguntas innecesarias en A+B pasan de 9,5 % (11/116) a 9,6 % (11/115) con un máximo de 2,0 %, y siguen sin cumplirse; en V25 la medida secundaria (A+B) sigue cumpliéndose y la obligatoria (peticiones G válidas fuera de alcance, 3,3 % a 3,5 % con máximo de 2,0 %) sigue sin cumplirse.
- **Una marca mecánica del informe sí cambia** (no es una decisión declarada): en la tabla «qué cambió» de V25, «segundos medianos del turno del agente: la segunda medida es peor» pasa de sí (10,4 a 11,9) a no (10,3 a 11,0). Es la única de las marcas mecánicas del informe que cambia.
- Las 16 `dudosa` no estaban repartidas por igual: los conjuntos E y F pierden una parte grande (7 de 30 y 4 de 20), así que sus cifras descriptivas se mueven más que las de A y B, que pierden una consulta de 116.

### Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| Editar los informes publicados | Cambiaría resultados ya medidos y declarados. El análisis es aparte: escribe su propio informe y no toca ni los informes ni los `results.json` |
| Excluir solo las filas `mal` | No hay ninguna: no habría excluido nada |
| Quitar del estrato C la necesidad entera en lugar de la versión dudosa | Se pierde más de lo que el autor puso en duda: él marcó una versión (`c09-fr`) y la necesidad se entiende. Se quita la versión y la necesidad conserva las otras (de 60 a 57 pares de versiones en el solape de los 10 primeros) |
| Reescribir las estadísticas para el análisis | Se reutilizan las funciones de los informes publicados, con lo que cualquier diferencia viene de los elementos excluidos y no de otro código; la prueba de reproducción lo comprueba |

### Límites

- **Se decidió después de ver los resultados y no está en ningún protocolo.** Es exploratorio, se informa sea cual sea su dirección y no sustituye a ningún resultado publicado.
- **No todo se puede recalcular.** El informe lista lo que deja como estaba: las tablas por idioma y por tipo, las latencias por etapa, las secciones 9 y 10 del catálogo, la hoja de anotación humana y el acuerdo entre jueces; las cuentas de juicios por necesidad; el gasto y las llamadas por etapa y el cruce con la clave, que salen de un registro que no dice a qué petición sirvió cada entrada; las respuestas que llegaron al tope de tokens (V25); y el inventario de las ejecuciones. La condición 2 de la regla de embeddings usa las medidas operativas publicadas (las 168 consultas).
- **Una sola persona revisó**, el autor, y para los idiomas que no lee se apoyó en una traducción (su declaración del 2026-11-13). Por eso su juicio sobre la naturalidad en esos idiomas es menos fiable, y el recálculo hereda ese límite.
- La exclusión de `b-zh-6` afecta también al asistente, porque sus conjuntos B y D son los estratos B y D del catálogo (mismos identificadores): también sale del conjunto B de V22 y V25, de los umbrales sobre A+B y sobre B, de la estabilidad de rutas y, si estaba, de la muestra del agente directo.

### Queda abierto

- Una revisión por una segunda persona.
