# Evaluación de búsqueda de un libro conocido por título y autor (V29, parte B)

Protocolo: `data/evaluation/known_item/PROTOCOL.md`. Experimento diseñado después de conocer los resultados de V14, a raíz de una revisión externa del 2026-11-13. La verdad de referencia sale del propio catálogo: no interviene ningún juez LLM ni ninguna valoración humana.

Muestra: 300 libros (semilla 29), tres consultas por libro; mitad de desarrollo y mitad de prueba de 150 libros cada una. Todas las cifras de la tabla principal salen de las 450 consultas de la mitad de prueba.

Peso léxico elegido en la mitad de desarrollo: w = 0.25 (peso vectorial 0.75).

## Mitad de prueba

Consultas T (título tal como está guardado):

| Sistema | Consultas | MRR@10 [IC 95 %] | éxito@1 [IC 95 %] | éxito@10 [IC 95 %] |
|---|---|---|---|---|
| S1 | 150 | 0.919 [0.881, 0.954] | 0.880 [0.827, 0.927] | 0.980 [0.953, 1.000] |
| S2 | 150 | 0.973 [0.952, 0.991] | 0.953 [0.920, 0.987] | 1.000 [1.000, 1.000] |
| S3 | 150 | 0.972 [0.949, 0.990] | 0.953 [0.920, 0.987] | 0.993 [0.980, 1.000] |
| S4 | 150 | 0.979 [0.961, 0.993] | 0.960 [0.927, 0.987] | 1.000 [1.000, 1.000] |
| S5 | 150 | 0.980 [0.963, 0.993] | 0.960 [0.927, 0.987] | 1.000 [1.000, 1.000] |
| S3w | 150 | 0.986 [0.968, 1.000] | 0.980 [0.953, 1.000] | 1.000 [1.000, 1.000] |
| S5w | 150 | 0.979 [0.961, 0.993] | 0.960 [0.927, 0.987] | 1.000 [1.000, 1.000] |

Consultas TA (título y primer autor):

| Sistema | Consultas | MRR@10 [IC 95 %] | éxito@1 [IC 95 %] | éxito@10 [IC 95 %] |
|---|---|---|---|---|
| S1 | 150 | 0.967 [0.940, 0.987] | 0.947 [0.907, 0.980] | 0.987 [0.967, 1.000] |
| S2 | 150 | 0.993 [0.983, 1.000] | 0.987 [0.967, 1.000] | 1.000 [1.000, 1.000] |
| S3 | 150 | 0.986 [0.968, 1.000] | 0.980 [0.953, 1.000] | 1.000 [1.000, 1.000] |
| S4 | 150 | 0.997 [0.990, 1.000] | 0.993 [0.980, 1.000] | 1.000 [1.000, 1.000] |
| S5 | 150 | 0.997 [0.990, 1.000] | 0.993 [0.980, 1.000] | 1.000 [1.000, 1.000] |
| S3w | 150 | 0.986 [0.970, 0.997] | 0.973 [0.947, 0.993] | 1.000 [1.000, 1.000] |
| S5w | 150 | 0.997 [0.990, 1.000] | 0.993 [0.980, 1.000] | 1.000 [1.000, 1.000] |

Consultas A (nombre del primer autor):

| Sistema | Consultas | nDCG@10 [IC 95 %] | éxito@10 [IC 95 %] |
|---|---|---|---|
| S1 | 150 | 0.898 [0.857, 0.935] | 0.960 [0.927, 0.987] |
| S2 | 150 | 0.816 [0.765, 0.866] | 0.907 [0.860, 0.953] |
| S3 | 150 | 0.925 [0.891, 0.955] | 0.980 [0.953, 1.000] |
| S4 | 150 | 0.894 [0.848, 0.935] | 0.927 [0.880, 0.967] |
| S5 | 150 | 0.960 [0.935, 0.981] | 0.987 [0.967, 1.000] |
| S3w | 150 | 0.885 [0.841, 0.926] | 0.940 [0.900, 0.973] |
| S5w | 150 | 0.923 [0.884, 0.957] | 0.953 [0.920, 0.987] |

## Contrastes declarados (mitad de prueba, con reranker en ambos lados, Holm sobre seis)

| Contraste | Tipo | Diferencia de | Métrica | n | Diferencia | IC 95 % | p | p Holm | Significativo |
|---|---|---|---|---|---|---|---|---|---|
| C1-T | T | S5 - S4 | MRR@10 | 150 | +0.001 | [+0.000, +0.003] | 1.000 | 1.000 | no |
| C2-T | T | S5w - S4 | MRR@10 | 150 | +0.000 | [+0.000, +0.000] | 1.000 | 1.000 | no |
| C1-TA | TA | S5 - S4 | MRR@10 | 150 | +0.000 | [+0.000, +0.000] | 1.000 | 1.000 | no |
| C2-TA | TA | S5w - S4 | MRR@10 | 150 | +0.000 | [+0.000, +0.000] | 1.000 | 1.000 | no |
| C1-A | A | S5 - S4 | nDCG@10 | 150 | +0.067 | [+0.033, +0.107] | <0.001 | <0.001 | sí |
| C2-A | A | S5w - S4 | nDCG@10 | 150 | +0.029 | [+0.008, +0.056] | 0.004 | 0.022 | sí |

## Decisión (regla de la sección 8 del protocolo)

- T (título tal como está guardado): ninguna fusión es significativamente mejor que S4; la conclusión de V14 se extiende a este tipo de consulta sobre este catálogo.
- TA (título y primer autor): ninguna fusión es significativamente mejor que S4; la conclusión de V14 se extiende a este tipo de consulta sobre este catálogo.
- A (nombre del primer autor): S5 y S5w son significativamente mejores que S4; la memoria debe limitar su conclusión a las consultas descriptivas y decir que, para este tipo de consulta, los candidatos léxicos ayudan.

## Lectura secundaria sin chino ni japonés (fuera de la familia, p sin corregir)

Estas seis diferencias no forman parte de la familia de contrastes: son descriptivas y sus p no se corrigen.

| Contraste | Diferencia de | n | Diferencia | IC 95 % | p |
|---|---|---|---|---|---|
| C1-T | S5 - S4 | 125 | +0.001 | [+0.000, +0.004] | 1.000 |
| C2-T | S5w - S4 | 125 | +0.000 | [+0.000, +0.000] | 1.000 |
| C1-TA | S5 - S4 | 125 | +0.000 | [+0.000, +0.000] | 1.000 |
| C2-TA | S5w - S4 | 125 | +0.000 | [+0.000, +0.000] | 1.000 |
| C1-A | S5 - S4 | 125 | +0.058 | [+0.022, +0.099] | <0.001 |
| C2-A | S5w - S4 | 125 | +0.030 | [+0.006, +0.062] | 0.034 |

## Mitad de desarrollo (elección de w; sin reranker)

| Ejecución | Objetivo |
|---|---|
| S3w@0.1 | 0.947 |
| S3w@0.25 | 0.951 |
| S3w@0.75 | 0.938 |
| S3w@0.9 | 0.931 |

Consultas T:

| Sistema | Consultas | MRR@10 [IC 95 %] | éxito@1 [IC 95 %] | éxito@10 [IC 95 %] |
|---|---|---|---|---|
| S1 | 150 | 0.898 [0.854, 0.938] | 0.860 [0.800, 0.913] | 0.973 [0.947, 0.993] |
| S2 | 150 | 0.975 [0.951, 0.993] | 0.960 [0.927, 0.987] | 1.000 [1.000, 1.000] |
| S3 | 150 | 0.966 [0.943, 0.984] | 0.933 [0.887, 0.973] | 1.000 [1.000, 1.000] |
| S3w@0.1 | 150 | 0.983 [0.966, 0.997] | 0.973 [0.947, 0.993] | 1.000 [1.000, 1.000] |
| S3w@0.25 | 150 | 0.981 [0.962, 0.997] | 0.967 [0.933, 0.993] | 1.000 [1.000, 1.000] |
| S3w@0.75 | 150 | 0.949 [0.922, 0.973] | 0.907 [0.853, 0.947] | 1.000 [1.000, 1.000] |
| S3w@0.9 | 150 | 0.931 [0.897, 0.961] | 0.887 [0.833, 0.933] | 0.993 [0.980, 1.000] |

Consultas TA:

| Sistema | Consultas | MRR@10 [IC 95 %] | éxito@1 [IC 95 %] | éxito@10 [IC 95 %] |
|---|---|---|---|---|
| S1 | 150 | 0.963 [0.934, 0.986] | 0.947 [0.907, 0.980] | 0.993 [0.980, 1.000] |
| S2 | 150 | 0.988 [0.973, 1.000] | 0.980 [0.953, 1.000] | 1.000 [1.000, 1.000] |
| S3 | 150 | 0.972 [0.950, 0.990] | 0.953 [0.913, 0.987] | 1.000 [1.000, 1.000] |
| S3w@0.1 | 150 | 0.988 [0.973, 1.000] | 0.980 [0.953, 1.000] | 1.000 [1.000, 1.000] |
| S3w@0.25 | 150 | 0.987 [0.970, 1.000] | 0.980 [0.953, 1.000] | 1.000 [1.000, 1.000] |
| S3w@0.75 | 150 | 0.965 [0.938, 0.987] | 0.947 [0.907, 0.980] | 0.993 [0.980, 1.000] |
| S3w@0.9 | 150 | 0.963 [0.935, 0.986] | 0.947 [0.907, 0.980] | 0.993 [0.980, 1.000] |

Consultas A:

| Sistema | Consultas | nDCG@10 [IC 95 %] | éxito@10 [IC 95 %] |
|---|---|---|---|
| S1 | 150 | 0.882 [0.836, 0.922] | 0.947 [0.907, 0.980] |
| S2 | 150 | 0.831 [0.776, 0.881] | 0.893 [0.840, 0.940] |
| S3 | 150 | 0.899 [0.859, 0.933] | 0.967 [0.933, 0.993] |
| S3w@0.1 | 150 | 0.868 [0.819, 0.912] | 0.927 [0.880, 0.967] |
| S3w@0.25 | 150 | 0.886 [0.840, 0.926] | 0.940 [0.900, 0.973] |
| S3w@0.75 | 150 | 0.901 [0.860, 0.937] | 0.953 [0.913, 0.987] |
| S3w@0.9 | 150 | 0.899 [0.858, 0.937] | 0.960 [0.927, 0.987] |

## Errores y gasto

Ejecuciones con error (se guarda el orden base, como en V14): {'S1': 0, 'S2': 0, 'S3': 0, 'S4': 0, 'S5': 0, 'S3w': 0, 'S5w': 0}. Consultas sin ejecución: {'S1': 0, 'S2': 0, 'S3': 0, 'S4': 0, 'S5': 0, 'S3w': 0, 'S5w': 0}.

Gasto real según el registro de gasto: 0.9695 USD en 1764 llamadas (tope 1.50 USD).

| Modelo | Llamadas | Llamadas fallidas | Coste (USD) |
|---|---|---|---|
| qwen/qwen3-embedding-8b | 900 | 0 | 0.0003 |
| qwen/qwen3-reranker-8b | 864 | 0 | 0.9692 |

## Desviaciones

- La primera ejecución de `run` se paró por el límite de dos horas de la orden en segundo plano que la lanzó, con 882 de las 900 consultas guardadas en el paso de base de datos, y se reanudó a las 13:10 con la misma orden; la reanudación no repitió ninguna consulta guardada ni ningún embedding. Motivo: límite de tiempo del entorno que ejecutaba la orden, no un fallo del código; la ejecución es reanudable según la sección 10.
- En el paso de base de datos la conexión a Postgres se cortó siete veces por tiempo de espera; el adaptador abrió una conexión nueva y repitió la consulta, y el paso tardó en total unas dos horas y cuarto en lugar de unos diez minutos. Motivo: fallos de red o del proveedor; son llamadas gratuitas y deterministas, y no se cambió ningún resultado guardado.
- Veinticuatro peticiones al reranker agotaron el tiempo de espera y el adaptador las repitió al cabo de un segundo; ninguna consulta llegó a los tres intentos fallidos, las 450 consultas de prueba tienen orden reordenado y el registro de gasto no tiene llamadas con error. Motivo: fallos del proveedor; la repetición la hace el adaptador de V14 sin cambios.
- Durante la ejecución, antes de leer ningún resultado, se amplió el código del informe (`report.py`) para incluir las frases de la regla de decisión en español, los límites de la sección 9, esta sección y el gasto por modelo; el análisis (`analysis.py`) y el diseño no se tocaron. Motivo: el encargo pide que el informe repita los límites, liste las desviaciones y aplique la regla de decisión en una frase por tipo.

## Límites (sección 9 del protocolo)

- Un título exacto tal como está guardado es el caso más favorable para la búsqueda léxica. Los usuarios reales escriben títulos parciales o con errores, que este experimento no mide.
- El título y los autores forman parte del texto que indexan las dos ramas (sección 2), así que la rama vectorial también ve el título.
- La muestra es de libros del catálogo, no de peticiones de usuarios: dice cómo se encuentra un libro que existe, no con qué frecuencia la gente busca así.
- La coincidencia de autor es exacta tras normalizar: «Brothers Grimm» y «Jacob Grimm» son autores distintos, y una obra con el nombre escrito de otra forma no cuenta como relevante.
- Muchos nombres de autor del catálogo están en alfabeto latino aunque la obra sea árabe, china, japonesa o rusa (por ejemplo, 24 de las 25 consultas A en árabe y las 25 en chino son ASCII). Las consultas A de esos idiomas prueban sobre todo la búsqueda de nombres latinos.
- En chino y japonés, BM25 trata un título sin espacios como una sola palabra: un título exacto puede coincidir entero, pero cualquier variación no.
- El experimento se diseñó después de conocer los resultados de V14, a raíz de una revisión externa del 2026-11-13.
