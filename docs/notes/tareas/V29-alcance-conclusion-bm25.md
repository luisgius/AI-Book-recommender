# V29 · Alcance de la conclusión «los candidatos de BM25 bajan el resultado»

- **Fecha:** 2026-11-14
- **Estado:** hecha. Queda abierta una decisión del autor (véase «Pendiente»)
- **Capa:** evaluación (`app/evaluation/catalog_evaluation/scope.py`, `app/evaluation/known_item/`), datos (`data/evaluation/catalog50k/scope/`, `data/evaluation/known_item/`), documentación (`docs/notes/`)
- **Commits:** `88449eb` (protocolo, muestra y código, antes de medir) y `210236d` (resultados)

## Qué se ha hecho

Dos partes, las dos posteriores a V14 y ninguna declarada en su protocolo.

**Parte A: recálculo post hoc de la evaluación del catálogo sobre subconjuntos de sus consultas.**

- Se escribió `app/evaluation/catalog_evaluation/scope.py` y la orden `scope` en `app/evaluation/catalog_evaluation/cli.py`. Recalcula la tabla de sistemas (nDCG@10) y los contrastes H1 a H6 de V14 con las mismas funciones, los mismos 10 000 remuestreos, la misma semilla (0) y la misma familia de Holm, sobre cuatro subconjuntos de las consultas ya evaluadas. No hace llamadas de red ni a modelos: lee los ficheros de ejecuciones y de juicios guardados. Antes de nada comprueba que el cálculo sin exclusiones reproduce exactamente el `results.json` publicado.
- Los subconjuntos son: (a) solo el estrato B (72 consultas); (b) estratos A y B sin chino ni japonés (104); (c) estrato B sin chino ni japonés (60); (d) todos los estratos salvo A, es decir B, C y D (124).
- Escribe `data/evaluation/catalog50k/scope/SCOPE.md` y `scope.json`.

**Parte B: evaluación de la búsqueda de un libro conocido por título y por autor.**

- Se escribió el paquete `app/evaluation/known_item/` (`design.py`, `sampling.py`, `fusion.py`, `runner.py`, `analysis.py`, `report.py`, `cli.py`) con tres órdenes: `sample`, `run` y `report`.
- Se sacó una muestra de 300 libros del catálogo desplegado (50 000 obras), 25 por cada uno de los doce idiomas de V14, con semilla 29, y se partió en 150 libros de desarrollo y 150 de prueba. De cada libro se construyen tres consultas mecánicas, 900 en total: el título tal como está guardado (T), el título y el primer autor (TA) y el nombre del primer autor solo (A). Para T y TA la única obra relevante es el propio libro; para A lo es toda obra del catálogo cuya lista de autores contiene un autor con el mismo nombre normalizado (de 1 a 238 obras, mediana 3).
- Se midieron los sistemas de V14 sin LLM (S1 a S5) y dos de fusión ponderada, S3w (RRF ponderado de los 60 primeros de S1 y S2) y S5w (S3w más el mismo reranker sobre los 30 primeros). El peso léxico w se eligió en la mitad de desarrollo, sin reranker, entre {0,1; 0,25; 0,75; 0,9}, y se guardó en `weight_selection.json` antes de reordenar la mitad de prueba. Salió w = 0.25.
- Se declararon seis contrastes en la mitad de prueba (C1 y C2 para cada tipo de consulta, con el reranker en los dos lados), con prueba de aleatorización pareada y Holm sobre seis, y una regla de decisión escrita antes de medir.
- Escribe `PROTOCOL.md`, `sample.json`, `results.json`, `KNOWN_ITEM.md`, `deviations.json` y los ficheros de ejecuciones, puntuaciones del reranker y registro de gasto, todos en `data/evaluation/known_item/`.
- Tests: `tests/evaluation/test_catalog_scope.py` (2 funciones de test) y `tests/evaluation/test_known_item.py` (23).

## Por qué

Una revisión independiente de la memoria, fechada el 2026-11-13, objetó que la conclusión «los candidatos de BM25 bajan el resultado» (contrastes H1 y H3 de V14, a partir de los cuales V24 cambió el modo de candidatos por defecto) descansaba sobre:

- consultas sin ninguna petición de título ni de autor, que es donde se espera que la búsqueda léxica ayude;
- un índice léxico que no segmenta el chino ni el japonés;
- una única fusión, con pesos iguales;
- jueces LLM;
- consultas del estrato A que también se habían usado para elegir modelos.

La parte A responde a la tercera de las objeciones sobre datos y a la última: quita el estrato A, quita los dos idiomas sin segmentar y mira qué pasa si se añaden C y D. La parte B responde a la primera, la segunda y la tercera con consultas por título y por autor, la verdad de referencia sacada del propio catálogo y una fusión ponderada además de la de pesos iguales. La objeción sobre los jueces se responde en B porque allí no interviene ningún juez.

## Objetivo

Saber hasta dónde llega la conclusión de V14 con una medida que no dependa de jueces LLM, y dar a la memoria el alcance exacto de esa conclusión.

## Por qué no otras alternativas

| Alternativa | Por qué no |
|---|---|
| Solo reescribir el texto de la memoria sin medir | Dejaría la objeción sin respuesta: la memoria diría lo mismo con más cautela, pero nadie sabría si la cautela es la correcta |
| Consultas de título y autor generadas por un LLM y juzgadas por jueces LLM | La objeción es justamente sobre los jueces. El catálogo da una verdad de referencia exacta para un libro conocido, así que no hace falta juez |
| Consultas escritas por personas | Solo estaba disponible el autor, y repetiría el problema de un único anotador |
| Ajustar el peso de la fusión con las mismas consultas con las que se informa | Favorecería a la fusión. Por eso la muestra se parte en desarrollo y prueba, y el peso se elige en desarrollo |
| Recalcular V14 con una fusión ponderada | Las nuevas listas fusionadas contendrían libros que los jueces nunca juzgaron |

## Resultados

### Parte A: alcance sobre las consultas de V14

Cifras de `data/evaluation/catalog50k/scope/SCOPE.md`, juez principal `z-ai/glm-5.3-flash`, diferencia de nDCG@10 con su intervalo al 95 %.

| Subconjunto | Consultas | H1 (S3 - S2) | H3 (S5 - S4) | Decisiones que cambian |
|---|---|---|---|---|
| V14 completo (A y B) | 116 | -0.092 [-0.128, -0.055] | -0.029 [-0.045, -0.014] | - |
| (a) Estrato B solo | 72 | -0.114 [-0.162, -0.067] | -0.021 [-0.040, -0.004] | ninguna |
| (b) A y B sin chino ni japonés | 104 | -0.098 [-0.139, -0.058] | -0.031 [-0.047, -0.015] | ninguna |
| (c) B sin chino ni japonés | 60 | -0.130 [-0.186, -0.076] | -0.022 [-0.043, -0.002] | ninguna |
| (d) B, C y D | 124 | -0.107 [-0.144, -0.071] | -0.013 [-0.028, +0.003] | cinco |

- En (a), (b) y (c) ninguna decisión declarada cambia: H1 a H6 y la regla del híbrido mantienen su resultado y la regla de embeddings mantiene `qwen/qwen3-embedding-8b`.
- En (d), H1 sigue siendo significativo con los dos jueces. Cambian cinco decisiones: H3 con el juez principal (p de Holm 0.111, deja de ser significativo), la conclusión de H3 con los dos jueces (de «sustained» a «not conclusive»), H5 con el juez principal (p de Holm 0.090), la conclusión de H5 con los dos jueces y la regla del híbrido («S4 significativamente mejor que S5 con los dos jueces» pasa de verdadera a falsa). Con el segundo juez, H3 sigue siendo significativo (diferencia -0.021, p de Holm 0.004).
- Los estratos C y D de (d) no son independientes ni informativos del mismo modo: las cuatro versiones de C comparten juicios y las consultas de D tienen nDCG@10 igual a 0 en todos los sistemas, así que solo diluyen las diferencias. Con menos consultas los intervalos se ensanchan, y perder la significación no es lo mismo que encontrar el efecto contrario.

### Parte B: libro conocido por título y por autor

Cifras de `data/evaluation/known_item/KNOWN_ITEM.md`, mitad de prueba (150 libros, 450 consultas). Peso léxico elegido en la mitad de desarrollo: w = 0.25.

Métrica principal: MRR@10 para T y TA, nDCG@10 para A.

| Sistema | T (MRR@10) | TA (MRR@10) | A (nDCG@10) |
|---|---|---|---|
| S1 (BM25) | 0.919 | 0.967 | 0.898 |
| S2 (vectorial) | 0.973 | 0.993 | 0.816 |
| S3 (RRF, pesos iguales) | 0.972 | 0.986 | 0.925 |
| S4 (S2 + reranker) | 0.979 | 0.997 | 0.894 |
| S5 (S3 + reranker) | 0.980 | 0.997 | 0.960 |
| S3w (w = 0.25) | 0.986 | 0.986 | 0.885 |
| S5w (S3w + reranker) | 0.979 | 0.997 | 0.923 |

Contrastes declarados (reranker en los dos lados, Holm sobre seis):

| Contraste | Métrica | Diferencia | IC 95 % | p Holm | Significativo |
|---|---|---|---|---|---|
| C1-T (S5 - S4) | MRR@10 | +0.001 | [+0.000, +0.003] | 1.000 | no |
| C2-T (S5w - S4) | MRR@10 | +0.000 | [+0.000, +0.000] | 1.000 | no |
| C1-TA (S5 - S4) | MRR@10 | +0.000 | [+0.000, +0.000] | 1.000 | no |
| C2-TA (S5w - S4) | MRR@10 | +0.000 | [+0.000, +0.000] | 1.000 | no |
| C1-A (S5 - S4) | nDCG@10 | +0.067 | [+0.033, +0.107] | <0.001 | sí |
| C2-A (S5w - S4) | nDCG@10 | +0.029 | [+0.008, +0.056] | 0.022 | sí |

Decisión por la regla de la sección 8 del protocolo:

- **T y TA:** ninguna fusión es significativamente mejor que S4. La conclusión de V14 se extiende a estos dos tipos de consulta sobre este catálogo.
- **A:** S5 y S5w son significativamente mejores que S4. La memoria debe limitar la conclusión de V14 a las consultas descriptivas y decir que, para las consultas por nombre de autor, los candidatos léxicos ayudan.
- Sin chino ni japonés (lectura secundaria, 125 consultas por tipo, p sin corregir) el patrón es el mismo: C1-A +0.058 [+0.022, +0.099] y C2-A +0.030 [+0.006, +0.062]; los cuatro contrastes de T y TA siguen en +0.000 o +0.001.
- La fusión ponderada con w = 0.25 gana menos que la de pesos iguales en las consultas A (C2-A +0.029 frente a C1-A +0.067). El peso se eligió con una media de tres números en la que T y TA pesan más que A, y sin reranker; el protocolo lo declara como simplificación.

Gasto real: 0.9695 USD en 1764 llamadas (900 de embeddings, 864 del reranker), con tope de 1.50 USD. Ninguna ejecución terminó en error.

## Límites

- **La parte A es post hoc.** Se decidió después de conocer los resultados de V14, no está en su protocolo, se informa sea cual sea su dirección y no sustituye a ningún resultado publicado.
- **La parte B se diseñó después de conocer los resultados de V14**, a raíz de la revisión. Su protocolo se guardó en git antes de ejecutar ningún sistema, pero no antes de saber qué había salido en V14.
- **Un título exacto tal como está guardado es el caso más favorable para la búsqueda léxica.** Los usuarios reales escriben títulos parciales o con errores, que no se miden.
- **El título y los autores están en el texto que indexan las dos ramas**, así que la rama vectorial también los ve.
- **La muestra es de libros del catálogo, no de peticiones de usuarios.** Dice cómo se encuentra un libro que existe, no con qué frecuencia la gente busca así.
- **La coincidencia de autor es exacta tras normalizar.** «Brothers Grimm» y «Jacob Grimm» son autores distintos.
- **Muchos nombres de autor están en alfabeto latino** aunque la obra sea árabe, china, japonesa o rusa (24 de las 25 consultas A en árabe y las 25 en chino son ASCII), así que esas consultas prueban sobre todo la búsqueda de nombres latinos.
- **En chino y japonés, BM25 trata un título sin espacios como una sola palabra.**
- **El peso w se eligió sin reranker** y puede no ser el mejor para S5w.
- **Se puntúa una vez cada libro de la unión de las tres cabezas** y se ordena con esa puntuación, lo que supone que el reranker puntúa cada par sin mirar los demás libros de la petición. Es lo esperable de un cross-encoder, pero no se comprobó.
- **No se mide el modo de candidatos por defecto con consultas descriptivas escritas por personas.** La parte A sigue dependiendo de los jueces LLM de V14.

### Desviaciones de la ejecución de la parte B

Están en `deviations.json` y en `KNOWN_ITEM.md`.

- La primera ejecución de `run` se paró por el límite de dos horas de la orden en segundo plano que la lanzó, con 882 de las 900 consultas guardadas en el paso de base de datos, y se reanudó a las 13:10 con la misma orden. La reanudación no repitió ninguna consulta guardada ni ningún embedding.
- La conexión a Postgres se cortó siete veces por tiempo de espera; el adaptador abrió una conexión nueva y repitió la consulta. El paso tardó unas dos horas y cuarto en lugar de unos diez minutos.
- Veinticuatro peticiones al reranker agotaron el tiempo de espera y el adaptador las repitió al cabo de un segundo. Ninguna consulta llegó a los tres intentos fallidos.
- Durante la ejecución, antes de leer ningún resultado, se amplió `report.py` (frases de la regla de decisión, límites, desviaciones y gasto por modelo). `analysis.py` y el diseño no se tocaron.

## Tope de gasto

El tope se planeó en 1 USD y se subió a 1.50 USD antes de guardar el protocolo, porque la estimación del reranker (de 1,1 a 1,4 USD) lo superaba. El gasto real, 0.9695 USD, quedó por debajo de 1 USD.

## Pendiente

- **Decisión del autor, no tomada en esta tarea:** si el modo de candidatos por defecto vuelve al híbrido, dado que la fusión gana en las consultas por autor y pierde en las descriptivas. No se cambió el valor por defecto del dominio.
- Redactar en la memoria el alcance de la conclusión de V14 según las dos partes (descriptivas: los candidatos léxicos bajan el resultado; título y título con autor: sin diferencia; autor: ayudan).

## Cómo reproducirlo

```
.venv/bin/python -m app.evaluation.catalog_evaluation scope
.venv/bin/python -m app.evaluation.known_item sample
.venv/bin/python -m app.evaluation.known_item run
.venv/bin/python -m app.evaluation.known_item report
.venv/bin/python -m pytest tests/evaluation/test_catalog_scope.py tests/evaluation/test_known_item.py
```

`scope` y `report` no usan red. `sample` solo lee la copia local del catálogo. `run` llama a Postgres, al servicio de embeddings y al reranker (de pago, con tope), es reanudable y se para al llegar al tope de gasto.

## Ficheros

- Código: `app/evaluation/catalog_evaluation/scope.py`, `app/evaluation/catalog_evaluation/cli.py`, `app/evaluation/known_item/`.
- Tests: `tests/evaluation/test_catalog_scope.py`, `tests/evaluation/test_known_item.py`.
- Datos de la parte A: `data/evaluation/catalog50k/scope/SCOPE.md`, `scope.json`.
- Datos de la parte B (`data/evaluation/known_item/`): `PROTOCOL.md`, `KNOWN_ITEM.md`, `sample.json`, `results.json`, `weight_selection.json`, `deviations.json`, `rerank_estimate.json`, `rerank_scores.jsonl`, `spend_ledger.jsonl`, `query_embeddings/` y `runs/`.
- Relacionadas: `docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`, V14 (`V14-evaluacion-catalogo-desplegado.md`) y V24.
