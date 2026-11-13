# V14 · Evaluación del sistema desplegado sobre su catálogo real

- **Fecha:** 2026-11-12
- **Estado:** hecha; los juicios son solo de LLM hasta que el autor rellene la hoja de anotación humana (el 2026-11-13 la rellenó: véase la adenda al final); el resultado sobre el modelo de embeddings se comprobó después en la tarea V23 y se mantiene
- **Capa:** evaluación (`app/evaluation/catalog_evaluation/`), datos (`data/evaluation/catalog50k/`), documentación (`docs/notes/`)
- **Commits:** protocolo en `9abf48e` (antes de cualquier resultado); implementación y resultados en `ac35110`

## Qué se ha hecho

- Se escribió y se guardó en git el protocolo de la evaluación (`docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`) antes de ejecutar nada. Después se ejecutó con 168 consultas en doce idiomas sobre el catálogo desplegado (esquema `catalog_deploy` de Neon, 50,000 obras), con nueve sistemas, un pool de 4,996 pares (consulta, obra) juzgados por dos jueces LLM de proveedores distintos, los seis contrastes declarados con corrección de Holm, la regla de decisión del modelo de embeddings, el análisis de consultas sin respuesta, la comprobación de las explicaciones y la hoja de anotación humana.
- El informe `data/evaluation/CATALOG_EVALUATION.md` se regenera sin conexión a partir de las ejecuciones y los juicios guardados.
- Código nuevo en `app/evaluation/catalog_evaluation/`, un módulo por etapa: `design.py` (todas las constantes que fija el protocolo), `queries.py` y `generation.py` (conjunto de consultas y su generación), `runs.py`, `pipeline.py` y `e2_index.py` (ejecución de los sistemas y el índice local de `baai/bge-m3`), `judging.py` (pool y jueces), `extras.py` (medidas operativas y explicaciones), `analysis.py` (métricas y contrastes, sin red ni base de datos), `annotation.py` (hoja humana), `ledger.py` (registro de gasto con tope), `results.py`, `report.py` y `report_meta.py` (documento de resultados e informe).
- No se modificó el sistema desplegado ni la base de datos de Neon.

## Por qué

Todo lo medido antes (V03 a V16) usa la colección de 177 libros y 44 consultas en inglés y español, sobre el perfil local. El sistema desplegado es otro perfil (Postgres en Neon, modelos por API) y otro catálogo (50,000 obras en más de doce idiomas). La paridad entre perfiles se había medido (V12, V13), pero quedaban tres conclusiones abiertas:

1. En 177 libros la fusión híbrida no superó a la búsqueda vectorial sola (V06, V13; informes `data/evaluation/RETRIEVAL_ABLATION.md` y `data/evaluation/BM25_PADDING.md`). Con 50,000 obras podía cambiar en cualquier sentido.
2. La ganancia del LLM como reordenador no se confirmó en la segunda ejecución (V07, `data/evaluation/LLM_RECOMMENDER.md`).
3. El modelo de embeddings elegido para la nube (`qwen/qwen3-embedding-8b`) no es repetible entre llamadas y tiene una cola de latencia larga (V16, `data/evaluation/API_MODELS.md`, reserva del diseñador número 1). El 2026-11-12 se observaron además dos búsquedas degradadas por tiempo agotado del proveedor. `baai/bge-m3` era la alternativa estable.

## Objetivo

Responder con datos, sobre el catálogo real y con reglas fijadas de antemano, a esas tres preguntas, y dejar un conjunto de consultas, un pool y unos juicios que permitan repetir el análisis sin llamar a ninguna API.

## Diseño fijado de antemano

El protocolo se redactó y se guardó en git (`9abf48e`) antes de generar una sola consulta o ejecutar un solo sistema. Importa porque los resultados no pudieron elegir las preguntas: los sistemas, los seis contrastes, la métrica, la corrección por comparaciones múltiples, las dos reglas de decisión y las amenazas a la validez estaban escritas cuando aún no existía ningún número. Cualquier desviación posterior debía anotarse en el informe con su motivo (están todas listadas más abajo).

- **Colección y sistema.** El catálogo desplegado, sin modificarlo. El sistema es el perfil nube llamando a sus adaptadores directamente, no a la URL pública. El embedding de cada consulta se codifica una sola vez por modelo y ese vector se guarda y se reutiliza en todos los sistemas, para que los contrastes entre sistemas no se mezclen con el ruido del modelo, que se estudia aparte.
- **Sistemas.**

| Id | Sistema |
|---|---|
| S1 | BM25 en Postgres |
| S2 | Vectorial con `qwen/qwen3-embedding-8b`, sobre el índice aproximado del esquema |
| S3 | Híbrido: fusión RRF (k = 60) de los 60 primeros de S1 y de S2 |
| S4 | S2 más el reranker por API `qwen/qwen3-reranker-8b` sobre los 30 primeros |
| S5 | S3 más el mismo reranker sobre los 30 primeros; es la configuración desplegada |
| S6 y S6b | S3 más `deepseek/deepseek-v4.1-flash` como reordenador de la lista de los 30 primeros (V07, variante R1); S6 es la primera ejecución y S6b la segunda, la réplica |
| E1 | Vectorial exacto con `qwen/qwen3-embedding-8b` (Postgres sin el índice aproximado) |
| E2 | Vectorial exacto con `baai/bge-m3` sobre las mismas obras y el mismo texto por obra (índice local) |

  De cada sistema se guardan 100 resultados por consulta, de modo que las métricas se recalculan sin llamar a ninguna API.
- **Consultas.** Cuatro estratos. A: las 44 originales (24 en inglés, 20 en español). B: 72 nuevas, seis en cada uno de doce idiomas (en, es, fr, de, it, pt, ca, nl, ru, zh, ja, ar): tres temáticas, dos descriptivas y una de obra conocida descrita sin palabras de su título. C: diez necesidades de información escritas cada una en español, inglés, francés y alemán (40 consultas, juicios compartidos). D: 12 peticiones que un catálogo de libros no puede atender. La unidad estadística son los estratos A y B (116 consultas independientes); C se informa aparte porque sus cuatro versiones no son independientes, y D solo se usa en el análisis de consultas sin respuesta.
- **Pool y jueces.** El pool es la unión de los diez primeros de todos los sistemas (S1 a S6 con las dos ejecuciones de S6, E1 y E2) por necesidad de información; evita juzgar los 50,000 libros para cada consulta. Cada par se puntúa de 0 a 3 con el texto de instrucción de V05. Juez principal: `z-ai/glm-5.3-flash`. Segundo juez: `minimax/minimax-m3.1-flash`, de otro proveedor distinto del juez principal, del LLM del sistema y de los modelos de recuperación, que juzga el pool entero.
- **Métrica y estadística.** Principal: nDCG@10 (calidad de las diez primeras posiciones, con más peso arriba y ganancia mayor para los grados altos). Secundarias: P@10, MRR@10, Recall@100 relativo al pool y Judged@10 (proporción de los diez primeros que tienen juicio). Intervalos del 95 % por bootstrap sobre consultas y contrastes por prueba de aleatorización pareada bilateral (10,000 remuestreos, semilla 0). Un contraste es la comparación de la diferencia de nDCG@10 entre dos sistemas sobre las mismas consultas. La corrección de Holm ajusta los p de varios contrastes a la vez para que probar seis no multiplique los falsos positivos. Regla: una conclusión se da por sostenida solo si tiene el mismo signo y es significativa con los dos jueces; si discrepan, se informa como no concluyente.
- **Seis contrastes declarados** (familia única de Holm, sobre A y B, con el juez principal y repetidos con el segundo): H1 S3 frente a S2; H2 S5 frente a S3; H3 S5 frente a S4; H4 S2 frente a S1; H5 S6 frente a S5 (con la primera ejecución; la segunda es la réplica, fuera de la familia); H6 E2 frente a E1.
- **Regla del modelo de embeddings** (sección 8 del protocolo). Cada modelo codifica las 168 consultas tres veces con el tiempo límite desplegado. Se propondría cambiar a `baai/bge-m3` si se cumplen las dos condiciones: (1) H6 no muestra a `qwen/qwen3-embedding-8b` significativamente mejor con ninguno de los dos jueces; (2) `baai/bge-m3` es mejor en al menos dos de los tres criterios operativos (latencia, llamadas fallidas o agotadas, consultas cuya lista de diez primeros cambia entre repeticiones) y no es peor en el tercero en más de un 10 %. En cualquier otro caso se mantiene el modelo desplegado.
- **Regla para la fusión híbrida** (último párrafo de la sección 8). El texto del TFG dirá lo que diga H1. La configuración desplegada mantiene BM25 aunque H1 y H3 no sean significativos, porque BM25 es el camino de degradación cuando falla el proveedor de embeddings, y se cambiaría solo si S4 superase a S5 de forma significativa.
- **Descriptivo, sin contraste:** resultados por idioma y por tipo de consulta, solapamiento de S2 con E1 (coste del índice aproximado) y el estrato C.
- **Tope de gasto:** 5 USD en OpenRouter para toda la tarea.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Evaluar a través de la URL pública | Gastaría el presupuesto diario de los visitantes y mediría la red del cliente; se llamó a los adaptadores directamente |
| Un solo juez LLM | No habría forma de saber cuánto depende una conclusión del juez; con dos jueces de proveedores distintos, una conclusión solo se sostiene si ambos coinciden en signo y significación |
| Generar las consultas nuevas a partir de resultados del propio sistema | Favorecería a lo que el sistema ya encuentra; el generador recibe solo la lista de idiomas y materias del catálogo y no ve ninguna salida del sistema, y es de un proveedor que no se usa en ningún otro punto |
| Volver a codificar la consulta en cada sistema | Mezclaría los contrastes entre sistemas con el ruido del modelo de embeddings; el vector de cada consulta se congela y el ruido se mide aparte |
| Guardar los vectores de `baai/bge-m3` en la base de datos de Neon | El protocolo no añade nada a Neon salvo tablas temporales pequeñas; los vectores se guardan en local (`data/catalog_deploy/bge_m3/`, ignorado por git) |
| Dejar que la regla de decisión se aplique mirando los resultados | Las dos reglas (modelo de embeddings y fusión híbrida) estaban escritas antes de ver ningún número |
| Tratar las cuatro versiones del estrato C como consultas independientes | No lo son, comparten necesidad y juicios; se informan aparte |

## Cómo se ha validado

- El informe se regenera sin conexión desde los ficheros guardados con `report --offline`; el documento de resultados se construye sin llamadas a API ni a la base de datos.
- Paridad entre la copia local del catálogo y Neon: 50000 obras en ambos, mismos identificadores, texto embebido igual (`text_hash`) y contenido igual en las 50000.
- Se comprobó con `EXPLAIN` que S2 usa el índice aproximado del esquema y que E1 hace un recorrido secuencial sin ese índice.
- S2 frente a E1: la media de la proporción de los diez primeros de E1 que S2 también devuelve es 1.000 en A y B (1.000 en las 168 consultas); en los cien primeros, 0.995. El índice aproximado no cuesta resultados medibles aquí.
- Los vectores guardados de `baai/bge-m3` se comprobaron con 9 obras recodificadas (coseno 1.000 con los vectores guardados, identificadores alineados).
- Hay tests en `tests/evaluation/test_catalog_evaluation.py`.
- El gasto real notificado por OpenRouter es 2.0229 USD en 14005 llamadas (tope 5.00 USD); el contador acumulado de la clave pasó de 1.3314 a 3.3636 USD (2.0322 USD), que incluye cualquier otro uso de la clave en esa ventana.

## Resultados

Todas las cifras son las del informe `data/evaluation/CATALOG_EVALUATION.md`. Los juicios son de LLM; el juez principal es `z-ai/glm-5.3-flash` y el segundo, `minimax/minimax-m3.1-flash`.

### Conjunto de consultas

| Estrato | Consultas |
|---|---|
| A | 44 |
| B | 72 |
| C | 40 |
| D | 12 |

Los estratos B, C y D los generó `google/gemini-3.8-flash` (temperatura 0.7, razonamiento bajo), de un proveedor que no se usa en ninguna otra parte de la evaluación, a partir de las cuotas de idioma del catálogo y sus 60 materias más frecuentes. Las instrucciones, las salidas y las fechas se guardan en `data/evaluation/catalog50k/queries/generation/`.

Las comprobaciones mecánicas (recuentos, duplicados, idioma de la consulta, palabras del título en las consultas de obra conocida y presencia de la obra en el catálogo) detectaron problemas en la primera pasada, y hubo que corregirlos:

- Tres consultas del estrato C en francés (`c05-fr`, `c08-fr`, `c09-fr`) no se detectaron como francés.
- Dos consultas de obra conocida contenían palabras del título (`b-ca-6`, `b-ar-6`).
- Varias obras conocidas no se encontraban en el catálogo por título y autor (entre ellas `b-ar-6`, `b-ca-6` y `b-zh-6`).
- Se pidieron al generador 10 sustituciones, todas registradas en `checks.json` (hasta tres rondas por consulta); al final quedaron 0 problemas.
- La etapa 1 se ejecutó dos veces: la primera exigía que coincidiera el apellido del autor, lo que rechazaba obras cuyo autor el catálogo guarda transliterado (por ejemplo un título en cirílico con «Fyodor Dostoyevsky»); se corrigió la regla para aceptar la coincidencia exacta del título, se reutilizaron las llamadas en caché y solo se pidieron de nuevo las sustituciones necesarias. Las 18 llamadas de sustitución de la primera ejecución se pagaron y están en el gasto, pero su salida no se usa.
- Veredictos de idioma del conjunto final: 98 correctos y 26 indecisos. Los indecisos son consultas cortas de palabras clave sin palabras funcionales; cada una se leyó a mano y está en su idioma.
- Obras conocidas encontradas por título: 11 de 12. La que no encontró la comprobación mecánica es `b-zh-6`, que pide 红楼梦, y el catálogo la guarda con su título inglés «Dream of the Red Chamber» (Cao Xueqin); se comprobó a mano.
- Ninguna consulta del estrato D se generó como petición de libros y ninguna tuvo que reclasificarse.

### Pool y jueces

El pool tiene 138 necesidades y 4996 pares (por necesidad: mediana 34.0, mínimo 17, máximo 100). Por estrato: A 1452, B 2315, C 839, D 390.

| Juez | Pares puntuados | Llamadas fallidas (reintentadas) | Sin juzgar | Grados 0 / 1 / 2 / 3 |
|---|---|---|---|---|
| `z-ai/glm-5.3-flash` (principal) | 4996 | 7 | 0 | 1639 / 1724 / 856 / 777 |
| `minimax/minimax-m3.1-flash` (segundo) | 4996 | 24 | 0 | 1578 / 1542 / 812 / 1064 |

Acuerdo entre los jueces en los 4996 pares: kappa ponderado cuadrático 0.850 [0.840, 0.859] (bootstrap sobre pares; el kappa mide el acuerdo descontando el azar), acuerdo exacto 0.717, dentro de un grado 0.978, diferencia media (segundo menos principal) +0.118. Kappa binario con grado >= 1: 0.780; con grado >= 2: 0.754. Por idioma de la necesidad el kappa cuadrático va de 0.807 (de) a 0.875 (en).

### Resultados principales (A y B, 116 consultas, nDCG@10 con intervalo del 95 %)

| Sistema | Juez principal | Segundo juez |
|---|---|---|
| S1 | 0.336 [0.287, 0.384] | 0.344 [0.294, 0.393] |
| S2 | 0.683 [0.647, 0.717] | 0.700 [0.662, 0.737] |
| S3 | 0.592 [0.554, 0.628] | 0.601 [0.562, 0.640] |
| S4 | 0.798 [0.772, 0.823] | 0.806 [0.778, 0.832] |
| S5 | 0.769 [0.740, 0.797] | 0.772 [0.744, 0.798] |
| S6 | 0.811 [0.782, 0.838] | 0.803 [0.772, 0.833] |
| S6b | 0.812 [0.783, 0.840] | 0.802 [0.769, 0.832] |
| E1 | 0.683 [0.647, 0.717] | 0.700 [0.662, 0.737] |
| E2 | 0.426 [0.379, 0.470] | 0.460 [0.412, 0.507] |

Judged@10 es 1.000 en todos los sistemas salvo S1 (0.916 con ambos jueces); el informe no explica esa diferencia.

### Los seis contrastes declarados

Diferencia de nDCG@10 (sistema menos referencia) con intervalo del 95 %, bilateral, Holm sobre H1 a H6. Todos los p de Holm de H1 a H4 y H6 son menores de 0.001 con los dos jueces.

| Contraste | Comparación | Juez principal: dif. [IC] | p Holm | Segundo juez: dif. [IC] | p Holm | Resultado |
|---|---|---|---|---|---|---|
| H1 | S3 - S2 | -0.092 [-0.128, -0.055] | <0.001 | -0.099 [-0.130, -0.068] | <0.001 | sostenido |
| H2 | S5 - S3 | +0.178 [+0.146, +0.210] | <0.001 | +0.170 [+0.140, +0.201] | <0.001 | sostenido |
| H3 | S5 - S4 | -0.029 [-0.045, -0.014] | <0.001 | -0.035 [-0.054, -0.014] | <0.001 | sostenido |
| H4 | S2 - S1 | +0.347 [+0.293, +0.401] | <0.001 | +0.357 [+0.304, +0.408] | <0.001 | sostenido |
| H5 | S6 - S5 | +0.042 [+0.012, +0.072] | 0.007 | +0.032 [+0.007, +0.057] | 0.013 | sostenido |
| H6 | E2 - E1 | -0.258 [-0.302, -0.214] | <0.001 | -0.241 [-0.287, -0.195] | <0.001 | sostenido |
| H5-rep | S6b - S5 | +0.043 [+0.013, +0.073] | fuera de la familia | +0.030 [+0.004, +0.056] | fuera de la familia | sostenido (p sin ajustar 0.004 y 0.024) |

### Pregunta 1: fusión híbrida frente a vectorial sola (H1) y candidatos de BM25 con reranker (H3)

- H1: la fusión híbrida es peor que la búsqueda vectorial sola (S3 - S2 = -0.092 con el juez principal y -0.099 con el segundo, ambos con Holm < 0.001). Coincide en el sentido con el hallazgo descriptivo de la ablación de V06 en 177 libros, pero aquí es un contraste declarado de antemano y significativo con los dos jueces. Por tanto, el texto del TFG dirá que la fusión híbrida de este sistema no supera a la vectorial sola.
- H3: con reranker, los candidatos de BM25 empeoran el resultado (S5 - S4 = -0.029 con el juez principal y -0.035 con el segundo, ambos con Holm < 0.001).
- H2 muestra que el reranker mejora mucho al híbrido (S5 - S3 = +0.178 y +0.170), pero el mismo reranker sobre los candidatos vectoriales solos (S4) es mejor aún (0.798 frente a 0.769 con el juez principal; 0.806 frente a 0.772 con el segundo).
- BM25 solo (S1) queda muy por debajo de la vectorial (H4: +0.347 y +0.357 a favor de S2).
- Regla de la configuración desplegada (sección 8 del protocolo, último párrafo): S4 supera a S5 de forma significativa con los dos jueces, de modo que la condición para cambiar el pipeline desplegado, de híbrido más reranker a vectorial más reranker, **se cumple**. Esta tarea no cambió el despliegue; el cambio es una tarea aparte. BM25 sigue siendo el recurso de reserva cuando el proveedor de embeddings falla.

### Pregunta 2: el LLM como reordenador frente al cross-encoder (H5)

- S6 supera a S5 (S6 - S5 = +0.042 con el juez principal, Holm 0.007, y +0.032 con el segundo, Holm 0.013). La segunda ejecución del LLM (S6b) lo replica: +0.043 con el principal y +0.030 con el segundo, con p sin ajustar 0.004 y 0.024 (la réplica queda fuera de la familia de Holm). A diferencia de V07, donde la ganancia no se confirmó en la segunda ejecución, aquí sí se confirma.
- Las dos ejecuciones coinciden poco en cada lista: dieron los mismos diez primeros en 5 de las 116 consultas, y la diferencia absoluta media de nDCG@10 por consulta es 0.041 (juez principal). Es decir, el promedio se mantiene y las listas individuales no.
- Descriptivamente (sin contraste declarado), S4 obtiene 0.798 y S6 0.811 con el juez principal; con el segundo, S4 0.806 y S6 0.803. La latencia mediana de la etapa del LLM es 2675.2 ms en S6 frente a 648.1 ms de la etapa de rerank de S5, medida desde un portátil.
- Se registraron fallos del LLM en las ejecuciones: S6 con 3 listas completadas y 1 respuesta inutilizable; S6b con 1 identificador desconocido, 2 listas completadas y 1 respuesta inutilizable. Ninguna consulta de ningún sistema tuvo que recurrir a la alternativa de «orden de entrada» (0 fallbacks).

### Pregunta 3: el modelo de embeddings (H6 y medidas operativas)

Mediciones operativas (168 consultas codificadas 3 veces por modelo con los tiempos límite desplegados: 3 s por intento, 1 reintento, 5 s por llamada, sin caché, una llamada cada vez por modelo):

| Modelo | Llamadas | Mediana ms | p95 ms | Fallidas o agotadas | Consultas cuyo vector cambió | Consultas cuyo top 10 cambió (de las comparadas) |
|---|---|---|---|---|---|---|
| `qwen/qwen3-embedding-8b` | 504 | 399 | 1977 | 51 | 132 | 116 (167) |
| `baai/bge-m3` | 504 | 398 | 728 | 6 | 94 | 10 (168) |

- **Condición 1** (H6 no muestra a qwen significativamente mejor con ninguno de los dos jueces): **falsa**. E2 - E1 = -0.258 con el juez principal y -0.241 con el segundo, ambos con Holm < 0.001: `qwen/qwen3-embedding-8b` es significativamente mejor con los dos jueces (nDCG@10 0.683 frente a 0.426 con el principal; 0.700 frente a 0.460 con el segundo).
- **Condición 2** (`baai/bge-m3` mejor en al menos dos de los tres criterios y no peor en el tercero en más de un 10 %): **verdadera**. `baai/bge-m3` es mejor en los tres (latencia, llamadas fallidas, cambios del top 10) y en ninguno es más de un 10 % peor.
- **Resultado de la regla: se mantiene `qwen/qwen3-embedding-8b`.** El modelo desplegado no se cambió en esta tarea.

Las dos caras, sin suavizar. En calidad, el modelo desplegado es mejor en esta medición, y por mucho. En lo operativo, es claramente peor: 51 llamadas fallidas o agotadas de 504 frente a 6, percentil 95 de latencia de 1977 ms frente a 728 ms, y la lista de diez primeros cambia entre llamadas repetidas en 116 consultas frente a 10. Para un visitante, un embedding fallido significa que la búsqueda recurre a resultados solo por palabras clave (BM25), que en este catálogo son mucho peores (S1 0.336 frente a S2 0.683 con el juez principal).

**Reserva sobre H6 (resuelta por V23).** La diferencia entre los dos modelos de embeddings es mucho mayor que en la colección de 177 libros (allí 0.614 frente a 0.654, V16; aquí 0.426 frente a 0.683 con el juez principal). El informe da la causa como no verificada: se comprobaron los vectores guardados (9 obras recodificadas, coseno 1.000, identificadores alineados) y que el texto por obra es el del cargador, pero no se investigó por qué la diferencia es mayor en este catálogo. La investigación posterior (tarea V23, `V23-comprobacion-embeddings.md`) no encontró ningún defecto en cómo se construyó o consultó el segundo modelo, de modo que la cifra de H6 se mantiene y la decisión de la regla no cambia.

### Resultados descriptivos que conviene mirar (juez principal, nDCG@10)

**Por idioma.** Cada idioma de B tiene seis consultas; en y es suman 30 y 26 porque incluyen las de A.

| Idioma | S1 | S2 | S4 | S5 | S6 | E2 |
|---|---|---|---|---|---|---|
| ar | 0.283 | 0.676 | 0.772 | 0.779 | 0.889 | 0.328 |
| ca | 0.444 | 0.570 | 0.753 | 0.704 | 0.874 | 0.362 |
| de | 0.359 | 0.665 | 0.807 | 0.759 | 0.778 | 0.523 |
| en | 0.425 | 0.617 | 0.767 | 0.725 | 0.793 | 0.347 |
| es | 0.377 | 0.723 | 0.809 | 0.783 | 0.753 | 0.507 |
| fr | 0.244 | 0.747 | 0.810 | 0.770 | 0.862 | 0.466 |
| it | 0.423 | 0.694 | 0.779 | 0.796 | 0.787 | 0.517 |
| ja | 0.046 | 0.682 | 0.805 | 0.776 | 0.809 | 0.333 |
| nl | 0.435 | 0.657 | 0.830 | 0.775 | 0.824 | 0.471 |
| pt | 0.301 | 0.698 | 0.832 | 0.800 | 0.876 | 0.419 |
| ru | 0.205 | 0.848 | 0.867 | 0.860 | 0.900 | 0.378 |
| zh | 0.000 | 0.758 | 0.834 | 0.834 | 0.851 | 0.504 |

- La búsqueda por palabras clave (S1) falla del todo en chino (0.000) y casi del todo en japonés (0.046); también es baja en ruso (0.205), francés (0.244) y árabe (0.283). El informe no da la causa. La búsqueda vectorial mantiene entre 0.570 (ca) y 0.848 (ru) en todos los idiomas.
- Los seis idiomas de B tienen solo seis consultas por celda: las diferencias entre idiomas son descriptivas y tienen poca precisión. Con el segundo juez las cifras difieren algo en cada celda (tabla completa en el informe) pero el panorama es el mismo.

**Por tipo de consulta.**

| Tipo | Consultas | S1 | S2 | S3 | S4 | S5 | S6 | S6b | E1 | E2 |
|---|---|---|---|---|---|---|---|---|---|---|
| descriptive | 24 | 0.201 | 0.728 | 0.554 | 0.815 | 0.798 | 0.832 | 0.827 | 0.728 | 0.496 |
| known_item | 12 | 0.558 | 0.803 | 0.770 | 0.846 | 0.853 | 0.955 | 0.955 | 0.803 | 0.418 |
| original | 44 | 0.398 | 0.639 | 0.583 | 0.778 | 0.736 | 0.754 | 0.771 | 0.639 | 0.420 |
| thematic | 36 | 0.276 | 0.668 | 0.568 | 0.796 | 0.762 | 0.818 | 0.807 | 0.668 | 0.387 |

**Mismo asunto en cuatro idiomas (estrato C, 10 necesidades, descriptivo).** Solapamiento medio por pares de los diez primeros entre las cuatro versiones de una necesidad: S1 0.065, S2 0.488, S3 0.215, S4 0.567, S5 0.480, S6 0.455, S6b 0.492, E1 0.488, E2 0.402. El nDCG@10 por idioma (es, en, fr, de) de S2 es 0.668, 0.730, 0.618 y 0.660, y el de S5 es 0.738, 0.761, 0.765 y 0.759.

**Coste del índice aproximado.** S2 y E1 obtienen el mismo nDCG@10 (0.683 con el principal) y el solapamiento de sus diez primeros es 1.000.

### Consultas sin respuesta (exploratorio)

- Ninguna consulta del estrato D se reclasificó: el juez principal no dio grado 2 o 3 a ninguna obra del pool de ninguna de las 12 (con el segundo juez tampoco).
- Hay 103 de las 116 consultas de A y B con respuesta (alguna obra con grado >= 2 por el juez principal); las 13 sin respuesta son `q01`, `q04`, `q06`, `q08`, `q09`, `q10`, `q11`, `q12`, `q13`, `q14`, `q30`, `q31` y `q35`.
- Puntuación del reranker del primer resultado de S5: en las respondidas (n = 103) mínimo 0.001, mediana 0.943, media 0.786, máximo 0.999; en las sin respuesta de D (n = 12) mínimo 0.000, mediana 0.001, media 0.013, máximo 0.070.
- Área bajo la curva (AUC) 0.990. El mejor umbral en estos datos (índice J de Youden) es 0.1292, con tasa de verdaderos positivos 0.961, tasa de falsos positivos 0.000 y exactitud 0.965.
- **Advertencia:** el umbral se eligió y se midió con los mismos datos, de modo que no se afirma nada sobre su rendimiento fuera de ellos. Además, las consultas de D las escribió un LLM (estrato generado, no de usuarios reales).

### Explicaciones

Cuarenta consultas de A y B elegidas al azar (semilla 0), los tres primeros resultados de S5, explicación generada por `deepseek/deepseek-v4.1-flash` mediante el cliente desplegado: 120 explicaciones, 120 interpretadas. Precisión de citas antes del guardarraíl (procedimiento de V10): 0.985 sobre 389 citas; explicaciones sin una cita válida: 0.000. En el idioma de la consulta (con el detector de idioma del catálogo): 115 de 120 (2 indecisas).

### Hoja de anotación humana

`data/evaluation/catalog50k/annotation/annotation_sheet.csv` contiene 200 pares tomados de los 3767 pares juzgados de A y B (semilla 0, estratificados por idioma y por nota del juez principal, sin mostrar ninguna nota de los jueces), con instrucciones en español en `INSTRUCCIONES.md`. **Está producida pero sin rellenar: la tiene que rellenar el autor.** Después se calcula el acuerdo con `annotation-agreement --sheet <fichero>` (comando completo en la sección de reproducción). Hasta entonces, ningún acuerdo con personas se ha medido en este catálogo. *(Actualización del 2026-11-13: el autor la rellenó; véase «Adenda 2026-11-13» al final.)*

### Coste y tiempo

Gasto real notificado por OpenRouter: 2.0229 USD en 14005 llamadas (tope 5.00 USD).

| Etapa | Llamadas | Fallidas | USD |
|---|---|---|---|
| 1-queries | 42 | 0 | 0.0851 |
| 2-embed | 2137 | 2 | 0.0897 |
| 2-runs | 672 | 2 | 0.9505 |
| 3-judge | 10023 | 31 | 0.8205 |
| 5-operational | 1008 | 57 | 0.0002 |
| 6-explanations | 123 | 3 | 0.0769 |
| total | 14005 | | 2.0229 |

Tiempo por etapa en segundos (sumado entre ejecuciones reanudadas): consultas 195, embeddings 937, paridad 7, ejecuciones 477, jueces 839, operativa 692, explicaciones 100. Los embeddings congelados de qwen se hicieron mientras DeepInfra iba lento (tiempos agotados de lectura de 30 s repetidos, unos 20 minutos para 168 consultas); por eso la etapa de embedding domina la tabla de latencias del informe (mediana 2327.1 ms; p95 total de S2 33258.0 ms). La medición operativa del mismo modelo con los tiempos límite desplegados, una hora más tarde, da mediana 399 ms y p95 1977 ms. Las latencias se midieron desde un portátil por internet contra un Neon compartido con la web en producción, de forma secuencial; describen ese entorno, no la función desplegada.

## Desviaciones respecto al protocolo

El informe lista diez. Todas están anotadas con su motivo.

1. **Corrige el propio protocolo:** el protocolo llama «HNSW» al índice aproximado de S2, pero el índice del esquema desplegado es `lakebase_ann` (el índice IVF de Neon, con reordenación a precisión completa). S2 se ejecutó sobre ese índice tal como está desplegado, así que la comparación de S2 con E1 mide ese índice.
2. Profundidad de fusión: S3 fusiona los 60 primeros de S1 y de S2, la profundidad que usa el servicio desplegado para una página de 10 resultados con reranker (max(10, 30) x 2); su fichero guarda los 100 primeros de la lista fusionada. S1, S2 y E1 son búsquedas independientes de 100 resultados. S4 reordena los 30 primeros de S2; S5 y S6, los 30 primeros de S3; el resto de cada lista conserva su orden.
3. Las llamadas del reranker y del LLM de S4 a S6 usaron tiempos límite largos de la evaluación (30 s por intento, 3 reintentos y hasta 3 intentos de la llamada completa) en lugar de los 5 s desplegados, para que un proveedor lento no convierta una medición de calidad en una degradación. Los tiempos límite desplegados solo se miden para los embeddings (sección operativa).
4. El juez puntúa cada par contra la descripción de la necesidad en inglés (en el estrato A, la propia consulta) con el prompt de V05 sin cambios (v1.0), cuyo texto llama a la consulta «a short keyword search in Spanish or in English» y nombra solo libros en inglés y español (los demás idiomas aparecen con su código ISO). El prompt se mantuvo sin cambios porque el protocolo lo exige.
5. Interpretación de los criterios de la sección 8: la latencia es mejor si lo son la mediana y el p95, y más de un 10 % peor si cualquiera de los dos supera en más de un 10 %; las llamadas fallidas y las consultas con el top 10 cambiado son recuentos, y un recuento de 0 para qwen haría que cualquier recuento positivo de bge-m3 fuera más de un 10 % peor. Los cambios del top 10 se cuentan sobre las consultas con al menos dos repeticiones correctas, con la búsqueda exacta de cada modelo (E1 en Postgres, E2 en local).
6. La sección 9 usa el juez principal para decidir qué consultas de A y B tienen respuesta y qué consultas de D se reclasifican; los recuentos del segundo juez se muestran al lado.
7. Comprobación de idioma de las explicaciones: la regla de V10 solo distingue español e inglés, así que se aplica en su lugar el detector de idioma del catálogo (`app/ingestion/language_detection.py`, 12 idiomas) al resumen y al razonamiento.
8. La muestra de anotación humana se saca de los pares juzgados del pool de los estratos A y B (la unidad estadística), no de C ni de D.
9. Varias etapas se ejecutaron en parte a la vez en procesos distintos (los ficheros de ejecución mientras se calculaban los vectores de E2; los jueces, la medición operativa y las explicaciones juntos). Cada proceso comprueba el tope contra el registro que leyó al empezar más sus propias llamadas, así que una comprobación podía ir retrasada respecto al gasto de los otros procesos; el total quedó muy por debajo del tope y el registro en disco contiene todas las llamadas.
10. Generador de consultas: `google/gemini-3.8-flash`, temperatura 0.7, razonamiento bajo, salida JSON, una llamada por idioma de B, una para C y una para D; las sustituciones se pidieron en la misma conversación con el motivo (como máximo tres rondas).

## Límites que debe conocer el tribunal

- Los juicios son de dos jueces LLM únicamente, hasta que el autor rellene la hoja humana. El acuerdo con personas se midió en otra colección (177 libros, inglés y español, V05 y V10); en este catálogo y sus doce idiomas está pendiente. Que los dos jueces coincidan (kappa 0.850) no equivale a que coincidan con personas.
- Las consultas de los estratos B, C y D las escribió un LLM, y solo unas pocas las comprobó el agente que implementó la tarea (idioma de las indecisas, la obra de `b-zh-6`), no el autor. Pueden ser más regulares y estar mejor formadas que las de usuarios reales, y las de obra conocida y descriptivas reflejan lo que ese modelo sabe. El protocolo pide que el autor revise la lista; está pendiente.
- El pool lo forman los sistemas evaluados y por tanto los favorece; un sistema futuro no puede compararse con estos juicios sin juzgar sus resultados nuevos. Recall@100 es relativo a este pool.
- Una sola colección y una sola ejecución de cada sistema determinista; S6 se repitió por ser el único con salida variable. Los modelos por API pueden cambiar bajo el mismo identificador; se registran las fechas y los proveedores fijados.
- Las latencias describen un portátil por internet contra un Neon compartido con la web viva, no la función desplegada.
- La diferencia entre E2 y E1 es mucho mayor que en 177 libros. La tarea V23 no encontró ningún defecto en la medida; la explicación que ofrece (consultas que piden conocimiento de la obra que no está en el texto del catálogo) está argumentada con ejemplos, no aislada con un experimento.
- Las 116 consultas de A y B son la unidad estadística; el estrato C (40 consultas, 10 necesidades) y las tablas por idioma y por tipo son descriptivas, con pocas consultas por celda.
- El umbral del análisis de consultas sin respuesta se eligió y se midió con los mismos datos.
- El juez recibe un prompt cuyo texto habla de consultas en español o inglés (desviación 4).

## Pendiente

- **Hoja de anotación humana:** la tiene que rellenar el autor; después se calcula el acuerdo. *(Hecho el 2026-11-13 con un solo anotador; véase la adenda al final. Sigue pendiente un segundo anotador.)*
- **Aplicar el resultado de la regla híbrida al despliegue:** la condición para pasar de híbrido más reranker a vectorial más reranker se cumple (S4 supera a S5 con los dos jueces). No se cambió el despliegue; es una tarea aparte, y BM25 se conserva como recurso cuando falla el proveedor de embeddings.
- **Investigar el resultado de los embeddings y los fallos:** hecha en la tarea V23 (`V23-comprobacion-embeddings.md`, 2026-11-12). No se encontró ningún defecto; la cifra de H6 y la decisión de mantener el modelo desplegado se mantienen. Aplicar su recomendación sobre las llamadas fallidas es la tarea V24.
- **Evaluación del agente y del asistente:** tarea posterior con su propio protocolo; esta evaluación no los mide.
- **Revisión por el autor de las consultas generadas por el LLM.**
- **Qué puede afirmar el texto del TFG:** que, sobre este catálogo, con consultas en parte generadas por un LLM y con juicios de dos jueces LLM, la fusión híbrida no superó a la vectorial sola, que los candidatos de BM25 empeoraron el resultado con reranker, que el reranker mejora mucho al híbrido, que el LLM como reordenador superó al cross-encoder con réplica pero con listas individuales muy distintas entre ejecuciones, y que el modelo desplegado fue mejor en calidad pero claramente peor en lo operativo. **Qué no puede afirmar:** que los juicios coincidan con los de personas, que la causa de la diferencia entre los dos modelos de embeddings esté demostrada (la tarea V23 descartó un defecto de la medida, pero su explicación no se aisló con un experimento), que el umbral de consultas sin respuesta funcione fuera de estos datos, que los resultados valgan para consultas de usuarios reales, ni que un sistema nuevo pueda compararse con estos juicios sin volver a juzgar.

## Cómo reproducirlo

```
.venv/bin/python -m app.evaluation.catalog_evaluation report --offline
.venv/bin/python -m app.evaluation.catalog_evaluation annotation-agreement --sheet <fichero rellenado>
```

El primer comando regenera el informe sin red, desde las ejecuciones, los juicios, las medidas operativas y el registro de gasto guardados. El segundo calcula el acuerdo con el autor cuando la hoja esté rellena. Las demás etapas (consultas, ejecuciones, jueces, medidas operativas y explicaciones) son subcomandos del mismo módulo (`python -m app.evaluation.catalog_evaluation <etapa>`, ver `app/evaluation/catalog_evaluation/cli.py`) y necesitan `OPENROUTER_API_KEY` y acceso de solo lectura a Neon. Tests: `pytest tests/evaluation/test_catalog_evaluation.py`.

## Ficheros

- Evaluación: `app/evaluation/catalog_evaluation/` (`design.py`, `queries.py`, `generation.py`, `runs.py`, `pipeline.py`, `e2_index.py`, `judging.py`, `extras.py`, `analysis.py`, `annotation.py`, `ledger.py`, `results.py`, `report.py`, `report_meta.py`, `cli.py`).
- Tests: `tests/evaluation/test_catalog_evaluation.py`.
- Datos y resultados (`data/evaluation/catalog50k/`): `queries/` (conjunto, comprobaciones y llamadas del generador), `runs/` (ejecuciones de 100 resultados por consulta), `query_embeddings/` (vectores congelados de las consultas), `pool.json`, `judgments/` (cachés de juicios con procedencia), `operational/`, `explanations/`, `annotation/` (hoja, clave e instrucciones), `parity.json`, `spend_ledger.jsonl`, `results.json`, `per_query_metrics.json`. Los vectores de E2 están en `data/catalog_deploy/bge_m3/` (ignorado por git).
- Informe: `data/evaluation/CATALOG_EVALUATION.md`.
- Protocolo: `docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`.
- Resultados anteriores a los que se conecta: `data/evaluation/RETRIEVAL_ABLATION.md`, `data/evaluation/BM25_PADDING.md`, `data/evaluation/LLM_RECOMMENDER.md`, `data/evaluation/API_MODELS.md`.

## Adenda 2026-11-13: la hoja de anotación rellenada por el autor

- **Fecha:** 2026-11-13
- **Estado:** hecha; el acuerdo con los jueces LLM se midió con **un solo anotador, el autor**; no se recalculó ninguna métrica con sus notas

### Qué se hizo

- El autor puso nota de 0 a 3 a la hoja de 200 pares. La hoja rellenada es `data/evaluation/catalog50k/annotation/annotation_sheet_filled.csv`; la hoja en blanco, `annotation_sheet.csv`, se queda como se generó. **184 pares tienen nota y 16 se dejaron vacíos** (las instrucciones permiten dejar vacía la celda si no se puede decidir; esos pares no cuentan).
- Se ejecutó `.venv/bin/python -m app.evaluation.catalog_evaluation annotation-agreement --sheet data/evaluation/catalog50k/annotation/annotation_sheet_filled.csv`, que escribe `data/evaluation/catalog50k/annotation/human_agreement.json`.
- Se regeneró el informe con `report --offline` sobre una copia de los datos y se comparó con el publicado: la única diferencia era la entrada `human_agreement` de `results.json` y las dos líneas de acuerdo de la sección «Human annotation sheet» de `data/evaluation/CATALOG_EVALUATION.md`. Solo entonces se sustituyeron los dos ficheros publicados.
- Arreglo en `read_sheet` (`app/evaluation/catalog_evaluation/annotation.py`): ahora abre la hoja con `utf-8-sig`. La exportación de la hoja de cálculo empieza con una marca de orden de bytes y el comando fallaba con `KeyError: 'pair_id'`. Hay un test (`tests/evaluation/test_catalog_evaluation.py`).

### Cifras (de `human_agreement.json`)

Notas del autor sobre los 184 pares: nota 0, 39; nota 1, 52; nota 2, 89; nota 3, 4. Kappa ponderado cuadrático (intervalo del 95 % por bootstrap sobre pares, 2000 remuestreos, semilla 0):

| Medida | `z-ai/glm-5.3-flash` (principal) | `minimax/minimax-m3.1-flash` (segundo) |
|---|---|---|
| Pares | 184 | 184 |
| Kappa ponderado cuadrático | 0.272 [0.137, 0.396] | 0.301 [0.171, 0.425] |
| Acuerdo exacto | 0.342 | 0.304 |
| Acuerdo a menos de un punto | 0.804 | 0.815 |
| Kappa binario, nota >= 1 (acuerdo) | 0.310 (0.739) | 0.282 (0.745) |
| Kappa binario, nota >= 2 (acuerdo) | 0.177 (0.587) | 0.251 (0.625) |
| Diferencia media con signo (juez menos autor) | -0.065 | +0.152 |
| Notas del juez 0 / 1 / 2 / 3 | 53 / 66 / 31 / 34 | 46 / 58 / 28 / 52 |

Matriz de confusión (filas: nota del autor 0 a 3; columnas: nota del juez 0 a 3):

| Autor | glm 0 | glm 1 | glm 2 | glm 3 | minimax 0 | minimax 1 | minimax 2 | minimax 3 |
|---|---|---|---|---|---|---|---|---|
| 0 | 22 | 12 | 2 | 3 | 19 | 15 | 0 | 5 |
| 1 | 13 | 20 | 8 | 11 | 12 | 17 | 10 | 13 |
| 2 | 18 | 32 | 20 | 19 | 14 | 25 | 18 | 32 |
| 3 | 0 | 2 | 1 | 1 | 1 | 1 | 0 | 2 |

Por idioma el JSON da el kappa de cada uno de los doce idiomas, con entre 6 y 52 pares cada uno (inglés 52, español 43, el resto entre 6 y 11). Con el juez principal el kappa va de -0.37 (catalán, 11 pares) a 0.52 (alemán, 9 pares); en inglés es 0.51 y en español 0.14. Con tan pocos pares, los intervalos por idioma son muy anchos (el de catalán, por ejemplo, va de -0.70 a 0.02) y no permiten comparar idiomas.

### Qué significan

- El acuerdo del autor con los dos jueces es **bajo**: kappa de 0.27 y 0.30, con intervalos que quedan lejos de los valores que suelen tomarse como acuerdo alto, y acuerdo exacto de un tercio de los pares.
- El acuerdo es mejor a menos de un punto (0.80 y 0.82) que exacto: el desacuerdo es sobre todo de un grado, no entre extremos.
- Donde el JSON muestra el desacuerdo es en el uso de las notas altas. El autor puso nota 3 solo 4 veces (2.2 %) y nota 2 en 89 pares (48.4 %); los jueces pusieron nota 3 en 34 pares (glm) y 52 (minimax). De los 89 pares con nota 2 del autor, glm dio 3 a 19 y minimax a 32, y dieron 0 o 1 a 50 (glm) y 39 (minimax). La diferencia media con signo es pequeña (-0.065 y +0.152) porque los desacuerdos van en las dos direcciones y se compensan.
- La distribución de notas de los jueces sobre esta hoja no es la del pool: la muestra se estratificó por la nota del juez principal (véase «Hoja de anotación humana»), de modo que los jueces están representados por diseño en todas las notas.
- El JSON no dice por qué el autor y los jueces discrepan. No se ha comprobado ninguna explicación (por ejemplo, que el autor lea la escala de forma más estricta, o que los jueces valoren la descripción de otra manera); decirlo sería una hipótesis.

### Límites

- **Un solo anotador, y es el autor del sistema.** No hay medida de acuerdo entre personas, así que no se sabe cuánto de la distancia con los jueces es desacuerdo del autor con cualquier otra persona. El autor conoce el sistema que se evalúa, aunque la hoja no mostraba ninguna nota de los jueces.
- **16 de 200 pares sin nota** (8 %); el acuerdo se calcula sobre los 184.
- **No se recalculó ninguna métrica con las notas del autor.** Las tablas de nDCG@10 y los contrastes del informe siguen usando solo los juicios de los dos jueces LLM, y el resto de este documento no cambia.
- Los intervalos son por bootstrap sobre pares; con 184 pares son anchos (unos 0.26 de amplitud).

### Qué queda abierto

- **Un segundo anotador**, para medir el acuerdo entre personas y poder separar el acuerdo del juez con una persona del acuerdo entre personas.
- **Recalcular las métricas y los contrastes con notas de personas**, o al menos comprobar si cambia alguna conclusión al sustituir las notas de los jueces por las del autor en los 184 pares; no se ha hecho.
- **Qué puede afirmar el texto del TFG:** que, sobre este catálogo, el acuerdo de un anotador (el autor) con los dos jueces LLM es bajo (kappa ponderado cuadrático 0.27 y 0.30, 184 pares), con el desacuerdo concentrado en las notas altas (los jueces dan 3 mucho más que el autor). **Qué no puede afirmar:** que los jueces sean fiables ni que no lo sean para este catálogo; que el desacuerdo se deba a un defecto de los jueces y no a las notas del autor; que las conclusiones de la evaluación se mantengan o cambien con notas humanas.
