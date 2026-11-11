# V07 · El LLM como recomendador: anclado a lo recuperado, con el catálogo entero y sin catálogo

- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** infraestructura (`app/infrastructure/llm/`), evaluación (`app/evaluation/`), datos (`data/evaluation/`)

## Qué se ha hecho

- `app/infrastructure/llm/llm_recommender.py`: un adaptador con tres formas de preguntar al modelo. `LLMListwiseReranker` implementa el puerto de dominio `Reranker`, de modo que se conecta a `SearchService` como el cross-encoder: recibe los candidatos fusionados, pide al modelo los diez que recomienda (por identificador, el mejor primero, con una frase de motivo) y devuelve todos los candidatos reordenados. `LLMRecommender.rank_books` hace lo mismo sobre el catálogo entero, sin recuperación. `LLMRecommender.recommend_closed_book` pide libros reales por título y autor, sin catálogo.
- `app/infrastructure/llm/prompts_recommender.py` y `schemas_recommender.py`: dos prompts en inglés (listwise y de libro cerrado), versión `v1.0`, y salida estructurada estricta. La lista de libros va primero y la consulta al final, para que el prefijo sea idéntico entre consultas y el proveedor pueda cachearlo.
- Reglas de anclaje como funciones puras y probadas: un identificador se lee sin espacios ni corchetes y en mayúsculas; los no mostrados y los repetidos se descartan y se cuentan; si vuelven menos de diez válidos, el top 10 se completa con los candidatos restantes en el orden mostrado; una respuesta ilegible cuenta como cero identificadores válidos.
- `app/evaluation/llm_recommender_evaluation.py`: experimento sobre las 44 consultas (24 en inglés, 20 en español) y el catálogo de 177 libros, con diez resultados por consulta. Caché en disco de cada salida del LLM (clave: sistema, ejecución, modelo, versión del prompt y mensajes exactos).
- Relleno de huecos: el juez `z-ai/glm-5.3-flash` puntuó 146 pares nuevos (grados 0 / 1 / 2 / 3: 70 / 54 / 16 / 6) y los escribió en `relevance_judgments_extended_z-ai-glm-5.3-flash.json`, que pasa a 1 702 juicios (637 originales y 1 065 del LLM).
- Informe `data/evaluation/LLM_RECOMMENDER.md` con calidad, contrastes, anclaje, coste, escalado, determinismo y los prompts tal como se usaron.

## Por qué

El proyecto se diseñó en 2024 con una premisa: primero se recupera y el LLM solo explica. En 2026 hay dos alternativas que había que contestar con datos y no con opinión. La primera: los modelos con un millón de tokens de contexto pueden leer entero un catálogo pequeño, de modo que cabe preguntarse para qué recuperar. La segunda: los LLM se usan directamente como recomendadores y la literatura describe que sugieren elementos que no existen en el catálogo (Liao et al., arXiv 2505.03336, leído el 2026-11-11). El autor decidió el 2026-11-11 "aplicar" ambas (`docs/notes/2026-11-11-validacion-proyecto-2026.md`, sección 7, decisiones 5 y 8).

## Objetivo

Medir, sobre la misma colección y los mismos juicios que V06, si un LLM que recomienda mejora al mejor pipeline sin LLM en el ranking, si leer el catálogo entero sustituye a la recuperación y con qué frecuencia recomienda libros ausentes del catálogo cuando no se le ancla.

## Diseño fijado de antemano

El diseño se redactó antes de ejecutar ningún sistema; el informe lo repite.

- R0 `pipeline`: `hybrid_norm_e5_bge` de V06 (BM25 normalizado, e5, RRF y `BAAI/bge-reranker-v2-m3` sobre los 30 primeros). Sin LLM en el ranking.
- R1 `rag_llm_recommender`: los 30 primeros de `hybrid_norm_e5` (los mismos candidatos que ve el reranker bge) reordenados por una llamada listwise al LLM, dentro de `SearchService`.
- R2 `long_context_recommender`: sin recuperación; una llamada con los 177 libros (mismos campos y truncado que R1) devuelve diez identificadores.
- R3 `closed_book_recommender`: sin catálogo; el LLM nombra diez libros reales (título y autor) y se emparejan con el catálogo con una regla fija (títulos normalizados iguales, cortados antes de `:`, `(`, `[` o ` - `, y mismo apellido del primer autor).
- Referencia descriptiva: `hybrid_norm_e5` sin reranker, el generador de candidatos de R1.
- LLM del sistema: `qwen/qwen3.8-flash` por OpenRouter, temperatura 0, razonamiento desactivado, salida JSON estricta, hasta 2048 tokens de salida, prompts `v1.0`. Juez de los pares sin juicio: `z-ai/glm-5.3-flash`, de un proveedor distinto al del LLM del sistema.
- Métrica primaria: nDCG@10 (ganancia 2^grado - 1) con Judged@10 = 1,000 exigido a R0, R1 y R2.
- Dos contrastes declarados de antemano, con corrección de Holm sobre exactamente esos dos: L1 = R1 frente a R0 y L2 = R2 frente a R0. Test de aleatorización pareado bilateral (cambios de signo) sobre el nDCG@10 por consulta, 10 000 remuestreos, semilla 0, e intervalo bootstrap percentil pareado del 95 %.
- Solo descriptivos: Recall@10, P@5, MRR, desglose por idioma, R3, anclaje, coste, escalado y determinismo (R1 y R2 se ejecutan dos veces).

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Dejar que el LLM recomiende títulos libremente y comprobar después si existen | Es lo que mide R3: ninguna de sus 440 recomendaciones estaba en el catálogo |
| Sustituir la recuperación por el catálogo entero en el prompt | Es lo que mide R2: no mejora al pipeline y deja de caber a partir de unos nueve mil libros |
| Decodificación restringida o tokenización de elementos para que el modelo solo pueda emitir elementos del catálogo, como en la literatura | Exige control sobre la decodificación o el entrenamiento del modelo, que un modelo alojado mediante API no da y que el proyecto excluye |
| Puntuar cada candidato con una llamada distinta al LLM (pointwise) | Treinta llamadas por búsqueda en lugar de una |
| Hacer de R1 el predeterminado porque su estimación puntual es la mejor | La evidencia no se sostiene en una segunda ejecución y añadiría una llamada a un modelo alojado a cada búsqueda |
| Omitir el experimento y citar la literatura | La pregunta es sobre este catálogo y estas consultas; lo que la memoria puede defender son las cifras propias del proyecto |

## Cómo se ha validado

- 58 tests nuevos (`tests/infrastructure/llm/test_llm_recommender.py`, `tests/evaluation/test_llm_recommender_evaluation.py`). Suite completa tras la tarea: 1240 superados, 0 fallidos.
- Reproducción de R0: su top 10 es idéntico al de `hybrid_norm_e5_bge` almacenado por V06 y V10 en las 44 consultas (en `hybrid_norm_e5`, también 44 de 44). Con los juicios sin las notas de esta tarea, R0 puntúa 0.701, igual que V10.
- Relleno de huecos: 146 pares juzgados, 0 fallos, 2,29 s por juicio.
- Repetición con `python -m app.evaluation.llm_recommender_evaluation --no-llm`: reprodujo todas las métricas desde las cachés sin ninguna llamada al LLM.
- Gasto total de unos 0,11 USD: 0,108 según OpenRouter por llamada (0,098 del recomendador y 0,010 del juez); el contador de uso de la clave marcó 0,110.

## Resultados

Primera ejecución de cada sistema, 44 consultas, nDCG@10 con intervalo bootstrap del 95 %:

| Sistema | nDCG@10 [IC 95 %] | Recall@10 | P@5 | MRR | Judged@10 |
|---|---|---|---|---|---|
| `pipeline` (R0) | 0.688 [0.613, 0.757] | 0.535 | 0.745 | 0.851 | 1.000 |
| `rag_llm_recommender` (R1) | 0.745 [0.676, 0.810] | 0.529 | 0.809 | 0.861 | 1.000 |
| `long_context_recommender` (R2) | 0.678 [0.590, 0.760] | 0.498 | 0.745 | 0.788 | 1.000 |
| `closed_book_recommender` (R3) | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 |
| `hybrid_norm_e5` (referencia) | 0.592 [0.504, 0.679] | 0.418 | 0.700 | 0.804 | 1.000 |

Contrastes declarados de antemano:

| Contraste | nDCG@10 sistema | nDCG@10 referencia | Dif. [IC 95 %] | p | p (Holm) |
|---|---|---|---|---|---|
| L1: R1 frente a R0 | 0.745 | 0.688 | +0.057 [+0.007, +0.103] | 0.024 | 0.047 |
| L2: R2 frente a R0 | 0.678 | 0.688 | -0.010 [-0.086, +0.060] | 0.805 | 0.805 |

Segunda ejecución, descriptiva y sin corrección: L1 pasa a +0.037 [-0.022, +0.090] (R1 0.725; p 0.203) y L2 a +0.014 [-0.058, +0.080] (R2 0.702; p 0.707).

Por idioma (nDCG@10, descriptivo):

| Sistema | en (n=24) | es (n=20) |
|---|---|---|
| `pipeline` | 0.614 | 0.777 |
| `rag_llm_recommender` | 0.669 | 0.835 |
| `long_context_recommender` | 0.554 | 0.827 |
| `hybrid_norm_e5` | 0.532 | 0.665 |

Anclaje. R1 y R2: ninguno de los 1 725 identificadores devueltos en las cuatro series (430 + 430 + 425 + 440) estaba fuera de la lista dada; hubo un identificador repetido en cada serie de R2. Listas completadas con candidatos restantes: R1, 1 de 44 en cada ejecución; R2, 3 de 44 en la primera y 1 de 44 en la segunda. Respuestas vacías: 1 en cada ejecución de R1, 1 en la primera de R2 y 0 en la segunda. Llamadas fallidas: 0. R3: 0 de 440 recomendaciones en el catálogo (0 de 240 en inglés, 0 de 200 en español), ninguna consulta con un acierto. Seis recomendaciones comparten título con un libro del catálogo pero no autor, y se cuentan como ausentes.

Coste por consulta (R0 en modelos locales tras una consulta de calentamiento; el resto, la llamada al LLM, con ambas ejecuciones juntas en R1 y R2):

| Sistema | Latencia mediana (s) | Latencia p95 (s) | Tokens de entrada (media) | Tokens de salida (media) | Coste por consulta (USD) |
|---|---|---|---|---|---|
| R0 `pipeline` | 2.30 | 2.42 | 0 | 0 | 0 |
| R1 | 6.47 | 11.23 | 3,019 | 380 | 0.000437 |
| R2 | 8.50 | 14.20 | 19,166 | 404 | 0.000526 |
| R3 | 11.39 | 16.40 | 84 | 631 | 0.000309 |

El coste de R2 es bajo porque 1 667 072 de sus tokens de entrada se leyeron de la caché de prompts del proveedor (el 98,8 %): el catálogo va primero en el prompt y es idéntico en todas las consultas. En R1 se leyeron 128 000 tokens de la caché y en R3 ninguno. El tiempo de R1 fuera de la llamada al LLM es de 12,2 ms (mediana).

Escalado de R2 (calculado, no ejecutado): 107,8 tokens de prompt por libro y 93 fijos. El prompt más la salida máxima de 2048 tokens caben en la ventana de 1 000 000 tokens hasta **9 260 libros**.

| Libros | Tokens de prompt por consulta | ¿Cabe? | Coste a precio de lista (USD) | Coste con el catálogo leído de caché (USD) |
|---|---|---|---|---|
| 177 | 19,167 | sí | 0.0031 | 0.0005 |
| 5,000 | 538,891 | sí | 0.0810 | 0.0088 |
| 50,000 | 5,388,067 | no | 0.8084 | 0.0864 |
| 200,000 | 21,551,986 | no | 3.2330 | 0.3450 |
| 1,000,000 | 107,759,558 | no | 16.1641 | 1.7244 |

Determinismo (dos ejecuciones con temperatura 0):

| Sistema | Top 10 idénticos | Mismos libros (cualquier orden) | nDCG@10 ejecución 1 | nDCG@10 ejecución 2 | Diferencia absoluta media por consulta |
|---|---|---|---|---|---|
| R1 | 12 / 44 | 17 / 44 | 0.745 | 0.725 | 0.040 |
| R2 | 5 / 44 | 13 / 44 | 0.678 | 0.702 | 0.051 |

## Lectura de los resultados

1. Apoyo límite a R1. En el contraste declarado, R1 supera a R0 en +0.057 (p de Holm 0.047, intervalo sin ajustar [+0.007, +0.103]): está en el borde del umbral. En la segunda ejecución la diferencia es +0.037 con un intervalo que incluye el cero (p 0.203, sin corrección). Con 44 consultas, la evidencia es débil y no se reproduce con claridad.
2. Sin apoyo para R2. Leer el catálogo entero no mejora al pipeline: -0.010 en la primera ejecución (p de Holm 0.805) y +0.014 en la segunda, ambos con intervalos que incluyen el cero. R2 queda por debajo de R0 en Recall@10 (0.498 frente a 0.535) y en MRR (0.788 frente a 0.851).
3. Anclaje. Cuando se pide al modelo que elija de una lista, no devuelve libros ajenos a ella: 0 identificadores fuera de la lista en 1 725. Cuando no se le da catálogo, ninguna de sus 440 recomendaciones está en el catálogo. La restricción es la que consigue que las recomendaciones sean mostrables; la medida no dice nada sobre si esos libros existen (véanse los límites).
4. Mejora de R1 sobre su propio orden de candidatos y su coste. R1 sube de 0.592 (`hybrid_norm_e5`, el orden de los 30 candidatos) a 0.745, es decir, el LLM reordena mejor que la fusión sola. Esa mejora cuesta en latencia mediana 6.47 s frente a los 2.30 s de R0, con una llamada a un modelo alojado por búsqueda; el 0.745 de R1 se compara con el 0.688 de R0, que ya incluye un reranker.
5. Conclusión. Con esta evidencia, el LLM como recomendador no es una mejora fiable sobre el pipeline: la ventaja de R1 es pequeña y no se reproduce con claridad, R2 no aporta nada y R3 no puede mostrar ningún libro del catálogo.

Nota sobre R0: puntúa 0.688 aquí y 0.701 en V10 con listas de resultados idénticas, porque R1 y R2 hicieron aparecer 76 libros relevantes recién juzgados en 25 consultas, lo que eleva el ranking ideal de todos los sistemas.

## Decisiones que se derivan

Tomadas por el diseñador el 2026-11-11 a partir de estas cifras:

1. Se mantiene la regla de diseño "el LLM solo trabaja sobre libros recuperados del catálogo", que ahora respalda la medición propia del proyecto (R3: ninguna de 440 recomendaciones está en el catálogo; R1 y R2: ningún identificador fuera de la lista dada en 1 725).
2. Se mantiene la recuperación. Leer el catálogo entero no supera al pipeline y deja de caber en la ventana de contexto a unos 9 260 libros, muy por debajo del catálogo que se está construyendo (cientos de miles de obras).
3. El recomendador R1 queda disponible como adaptador opcional del puerto `Reranker`, pero no se hace el predeterminado: su ventaja en el contraste declarado es pequeña, una segunda ejecución no la reproduce, solo 12 de 44 listas se repiten entre ejecuciones y añade unos cuatro segundos y una llamada a un modelo alojado por búsqueda. En esta tarea no se conecta a la API.
4. Debe volver a medirse sobre el catálogo grande y con el modelo de sistema más rápido que pueda elegir la selección velocidad-calidad.

## Límites que debe conocer el tribunal

- Son 44 consultas y 177 libros: las diferencias pequeñas son difíciles de detectar y los resultados pueden no trasladarse a un catálogo de cientos de miles de obras.
- Solo hay dos ejecuciones y el modelo no es determinista a temperatura 0: en R1 se repiten 12 de 44 listas (17 de 44 con los mismos libros en otro orden) y en R2, 5 de 44 (13 de 44). La diferencia entre ejecuciones (0.745 y 0.725 en R1) es del orden de la diferencia que se contrasta.
- El juez es un LLM que puntúa listas elegidas por otro LLM. Son de proveedores distintos, pero no se descarta un sesgo compartido. Las 146 notas nuevas no tienen revisión humana.
- Los libros de R3 no se buscaron en fuentes externas: "no está en el catálogo" no significa "no existe". El catálogo solo tiene 177 libros. Los seis casos con el mismo título eran libros distintos, con una traducción discutible.
- La tabla de escalado de R2 está calculada, no ejecutada.
- Precios y latencias son los del 2026-11-11. Las de R0 incluyen una GPU compartida con otras tareas; las de los demás sistemas incluyen el enrutado de OpenRouter, de modo que son orientativas.

## Cómo reproducirlo

```
python -m app.evaluation.llm_recommender_evaluation
python -m app.evaluation.llm_recommender_evaluation --no-llm
```

El primer comando llama al LLM del sistema y al juez solo para lo que no esté en caché (`data/evaluation/llm_recommender/qwen-qwen3.8-flash__prompt-v1.0.jsonl`, `data/evaluation/llm_judgments/z-ai-glm-5.3-flash__prompt-v1.0.jsonl`); el segundo no llama a ningún LLM y falla si falta alguna salida o juicio. Ambos ejecutan los modelos de recuperación locales (e5, bge), por lo que las latencias se vuelven a medir cada vez. Opciones: `--workers`, `--model`, `--device`, `--n-resamples`, `--seed`, `--output-json`, `--output-md`. Tests: `pytest tests/infrastructure/llm/test_llm_recommender.py tests/evaluation/test_llm_recommender_evaluation.py`.

## Ficheros

- `app/infrastructure/llm/llm_recommender.py`, `prompts_recommender.py`, `schemas_recommender.py`.
- `app/evaluation/llm_recommender_evaluation.py`, `app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json`.
- `data/evaluation/LLM_RECOMMENDER.md`, `results_llm_recommender.json`, `llm_recommender/` (caché del recomendador), `llm_judgments/` (caché del juez).
- `tests/infrastructure/llm/test_llm_recommender.py`, `tests/evaluation/test_llm_recommender_evaluation.py`.
