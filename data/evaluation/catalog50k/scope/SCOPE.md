# Alcance de las conclusiones de la evaluación sobre el catálogo (V29, parte A)

**Análisis post hoc, fuera del protocolo declarado.** Se hizo el 2026-11-14, después de conocer los resultados de V14 y a raíz de una revisión externa del 2026-11-13. El protocolo de V14 (`docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`, commit 9abf48e) no lo declara. Se informa sea cual sea su dirección y no sustituye ni modifica los resultados publicados en `data/evaluation/CATALOG_EVALUATION.md`.

## Qué se recalcula

La tabla de sistemas (nDCG@10) y los contrastes H1 a H6 de V14, con las mismas funciones, el mismo número de remuestreos (10000), la misma semilla (0) y la misma familia de Holm, sobre cuatro subconjuntos de las consultas ya evaluadas. No se hace ninguna llamada de red ni a modelos: todo sale de los ficheros de ejecuciones y de juicios guardados. Antes de nada se comprueba que el cálculo sin exclusiones reproduce exactamente el `results.json` publicado.

- (a) Solo el estrato B: quita las 44 consultas del estrato A, que se usaron para elegir modelos.
- (b) Estratos A y B sin las consultas en chino y japonés, donde el índice BM25 de PostgreSQL no segmenta el texto.
- (c) Estrato B sin chino ni japonés.
- (d) Todos los estratos salvo A: B, C (versiones en cuatro idiomas de diez necesidades, que comparten juicios y no son independientes) y D (consultas sin respuesta en el catálogo; ninguna obra tiene grado relevante, así que su nDCG@10 es 0 en todos los sistemas y solo diluye las diferencias).

La columna «cambia» compara la significación (p corregida por Holm < 0,05) con la del conjunto completo de V14 (estratos A y B, 116 consultas). Con menos consultas los intervalos se ensanchan y la potencia baja, así que perder la significación no es lo mismo que encontrar el efecto contrario.

Comando: `.venv/bin/python -m app.evaluation.catalog_evaluation scope`.

## Resumen

| Subconjunto | Consultas | H1 (juez principal) | H3 (juez principal) | Decisiones que cambian respecto al conjunto completo |
|---|---|---|---|---|
| V14 completo (A y B) | 116 | -0.092 [-0.128, -0.055] | -0.029 [-0.045, -0.014] | - |
| (a) Estrato B solo | 72 | -0.114 [-0.162, -0.067] | -0.021 [-0.040, -0.004] | ninguna |
| (b) Estratos A y B sin chino ni japonés | 104 | -0.098 [-0.139, -0.058] | -0.031 [-0.047, -0.015] | ninguna |
| (c) Estrato B sin chino ni japonés | 60 | -0.130 [-0.186, -0.076] | -0.022 [-0.043, -0.002] | ninguna |
| (d) Todos los estratos salvo A (B, C y D) | 124 | -0.107 [-0.144, -0.071] | -0.013 [-0.028, +0.003] | H3 significant at 0.05, primary judge, H3 conclusion (both judges), H5 significant at 0.05, primary judge, H5 conclusion (both judges), Hybrid rule: S4 significantly better than S5 under both judges |

## nDCG@10 por sistema

Con el juez principal `z-ai/glm-5.3-flash`:

| Subconjunto | S1 | S2 | S3 | S4 | S5 | S6 | S6b | E1 | E2 |
|---|---|---|---|---|---|---|---|---|---|
| V14 completo (A y B) | 0.336 | 0.683 | 0.592 | 0.798 | 0.769 | 0.811 | 0.812 | 0.683 | 0.426 |
| (a) Estrato B solo | 0.298 | 0.711 | 0.597 | 0.811 | 0.789 | 0.846 | 0.838 | 0.711 | 0.429 |
| (b) Estratos A y B sin chino ni japonés | 0.372 | 0.679 | 0.581 | 0.796 | 0.765 | 0.809 | 0.810 | 0.679 | 0.426 |
| (c) Estrato B sin chino ni japonés | 0.353 | 0.709 | 0.579 | 0.809 | 0.786 | 0.849 | 0.839 | 0.709 | 0.431 |
| (d) Todos los estratos salvo A (B, C y D) | 0.287 | 0.643 | 0.536 | 0.743 | 0.730 | 0.756 | 0.763 | 0.643 | 0.417 |

Con el segundo juez `minimax/minimax-m3.1-flash`:

| Subconjunto | S1 | S2 | S3 | S4 | S5 | S6 | S6b | E1 | E2 |
|---|---|---|---|---|---|---|---|---|---|
| V14 completo (A y B) | 0.344 | 0.700 | 0.601 | 0.806 | 0.772 | 0.803 | 0.802 | 0.700 | 0.460 |
| (a) Estrato B solo | 0.308 | 0.716 | 0.598 | 0.819 | 0.792 | 0.835 | 0.829 | 0.716 | 0.453 |
| (b) Estratos A y B sin chino ni japonés | 0.381 | 0.696 | 0.593 | 0.806 | 0.770 | 0.801 | 0.799 | 0.696 | 0.459 |
| (c) Estrato B sin chino ni japonés | 0.366 | 0.711 | 0.583 | 0.822 | 0.793 | 0.838 | 0.830 | 0.711 | 0.450 |
| (d) Todos los estratos salvo A (B, C y D) | 0.295 | 0.641 | 0.535 | 0.747 | 0.727 | 0.760 | 0.764 | 0.641 | 0.434 |

## (a) Estrato B solo

Consultas: 72 (B: 72).

Juez principal:

| Contraste | n | Diferencia | IC 95 % | p Holm | Significativo | Cambia |
|---|---|---|---|---|---|---|
| H1 (S3 - S2) | 72 | -0.114 | [-0.162, -0.067] | <0.001 | sí | no |
| H2 (S5 - S3) | 72 | +0.192 | [+0.154, +0.231] | <0.001 | sí | no |
| H3 (S5 - S4) | 72 | -0.021 | [-0.040, -0.004] | 0.023 | sí | no |
| H4 (S2 - S1) | 72 | +0.413 | [+0.341, +0.482] | <0.001 | sí | no |
| H5 (S6 - S5) | 72 | +0.056 | [+0.028, +0.085] | <0.001 | sí | no |
| H6 (E2 - E1) | 72 | -0.282 | [-0.340, -0.224] | <0.001 | sí | no |
| H5-rep (S6b - S5) | 72 | +0.049 | [+0.018, +0.079] | 0.002 (sin Holm) | sí | no |

Segundo juez:

| Contraste | n | Diferencia | IC 95 % | p Holm | Significativo | Cambia |
|---|---|---|---|---|---|---|
| H1 (S3 - S2) | 72 | -0.118 | [-0.163, -0.076] | <0.001 | sí | no |
| H2 (S5 - S3) | 72 | +0.194 | [+0.155, +0.234] | <0.001 | sí | no |
| H3 (S5 - S4) | 72 | -0.027 | [-0.046, -0.009] | 0.004 | sí | no |
| H4 (S2 - S1) | 72 | +0.408 | [+0.336, +0.477] | <0.001 | sí | no |
| H5 (S6 - S5) | 72 | +0.043 | [+0.017, +0.070] | 0.003 | sí | no |
| H6 (E2 - E1) | 72 | -0.263 | [-0.324, -0.202] | <0.001 | sí | no |
| H5-rep (S6b - S5) | 72 | +0.037 | [+0.007, +0.067] | 0.017 (sin Holm) | sí | no |

Decisiones declaradas:

| Decisión | Conjunto completo | Subconjunto | Cambia |
|---|---|---|---|
| H1 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 72 | no |
| H1 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 72 | no |
| H1 conclusion (both judges) | sustained | sustained | no |
| H2 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 72 | no |
| H2 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 72 | no |
| H2 conclusion (both judges) | sustained | sustained | no |
| H3 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm 0.023, n 72 | no |
| H3 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm 0.004, n 72 | no |
| H3 conclusion (both judges) | sustained | sustained | no |
| H4 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 72 | no |
| H4 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 72 | no |
| H4 conclusion (both judges) | sustained | sustained | no |
| H5 significant at 0.05, primary judge | True p Holm 0.007, n 116 | True p Holm <0.001, n 72 | no |
| H5 significant at 0.05, second judge | True p Holm 0.013, n 116 | True p Holm 0.003, n 72 | no |
| H5 conclusion (both judges) | sustained | sustained | no |
| H6 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 72 | no |
| H6 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 72 | no |
| H6 conclusion (both judges) | sustained | sustained | no |
| H5-rep significant at 0.05, primary judge | True p 0.004, n 116 | True p 0.002, n 72 | no |
| H5-rep significant at 0.05, second judge | True p 0.024, n 116 | True p 0.017, n 72 | no |
| H5-rep conclusion (both judges) | sustained | sustained | no |
| Hybrid rule: S4 significantly better than S5 under both judges | True | True | no |
| Embedding rule, condition 1: H6 does not show qwen significantly better | False | False | no |
| Embedding rule, outcome | keep qwen/qwen3-embedding-8b | keep qwen/qwen3-embedding-8b | no |

## (b) Estratos A y B sin chino ni japonés

Consultas: 104 (A: 44, B: 60).

Juez principal:

| Contraste | n | Diferencia | IC 95 % | p Holm | Significativo | Cambia |
|---|---|---|---|---|---|---|
| H1 (S3 - S2) | 104 | -0.098 | [-0.139, -0.058] | <0.001 | sí | no |
| H2 (S5 - S3) | 104 | +0.184 | [+0.150, +0.218] | <0.001 | sí | no |
| H3 (S5 - S4) | 104 | -0.031 | [-0.047, -0.015] | <0.001 | sí | no |
| H4 (S2 - S1) | 104 | +0.307 | [+0.251, +0.362] | <0.001 | sí | no |
| H5 (S6 - S5) | 104 | +0.044 | [+0.010, +0.075] | 0.010 | sí | no |
| H6 (E2 - E1) | 104 | -0.253 | [-0.302, -0.205] | <0.001 | sí | no |
| H5-rep (S6b - S5) | 104 | +0.045 | [+0.012, +0.077] | 0.008 (sin Holm) | sí | no |

Segundo juez:

| Contraste | n | Diferencia | IC 95 % | p Holm | Significativo | Cambia |
|---|---|---|---|---|---|---|
| H1 (S3 - S2) | 104 | -0.103 | [-0.138, -0.069] | <0.001 | sí | no |
| H2 (S5 - S3) | 104 | +0.177 | [+0.144, +0.209] | <0.001 | sí | no |
| H3 (S5 - S4) | 104 | -0.037 | [-0.058, -0.014] | 0.002 | sí | no |
| H4 (S2 - S1) | 104 | +0.314 | [+0.262, +0.366] | <0.001 | sí | no |
| H5 (S6 - S5) | 104 | +0.032 | [+0.004, +0.058] | 0.025 | sí | no |
| H6 (E2 - E1) | 104 | -0.237 | [-0.288, -0.188] | <0.001 | sí | no |
| H5-rep (S6b - S5) | 104 | +0.029 | [+0.001, +0.057] | 0.046 (sin Holm) | sí | no |

Decisiones declaradas:

| Decisión | Conjunto completo | Subconjunto | Cambia |
|---|---|---|---|
| H1 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 104 | no |
| H1 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 104 | no |
| H1 conclusion (both judges) | sustained | sustained | no |
| H2 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 104 | no |
| H2 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 104 | no |
| H2 conclusion (both judges) | sustained | sustained | no |
| H3 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 104 | no |
| H3 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm 0.002, n 104 | no |
| H3 conclusion (both judges) | sustained | sustained | no |
| H4 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 104 | no |
| H4 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 104 | no |
| H4 conclusion (both judges) | sustained | sustained | no |
| H5 significant at 0.05, primary judge | True p Holm 0.007, n 116 | True p Holm 0.010, n 104 | no |
| H5 significant at 0.05, second judge | True p Holm 0.013, n 116 | True p Holm 0.025, n 104 | no |
| H5 conclusion (both judges) | sustained | sustained | no |
| H6 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 104 | no |
| H6 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 104 | no |
| H6 conclusion (both judges) | sustained | sustained | no |
| H5-rep significant at 0.05, primary judge | True p 0.004, n 116 | True p 0.008, n 104 | no |
| H5-rep significant at 0.05, second judge | True p 0.024, n 116 | True p 0.046, n 104 | no |
| H5-rep conclusion (both judges) | sustained | sustained | no |
| Hybrid rule: S4 significantly better than S5 under both judges | True | True | no |
| Embedding rule, condition 1: H6 does not show qwen significantly better | False | False | no |
| Embedding rule, outcome | keep qwen/qwen3-embedding-8b | keep qwen/qwen3-embedding-8b | no |

## (c) Estrato B sin chino ni japonés

Consultas: 60 (B: 60).

Juez principal:

| Contraste | n | Diferencia | IC 95 % | p Holm | Significativo | Cambia |
|---|---|---|---|---|---|---|
| H1 (S3 - S2) | 60 | -0.130 | [-0.186, -0.076] | <0.001 | sí | no |
| H2 (S5 - S3) | 60 | +0.207 | [+0.168, +0.249] | <0.001 | sí | no |
| H3 (S5 - S4) | 60 | -0.022 | [-0.043, -0.002] | 0.038 | sí | no |
| H4 (S2 - S1) | 60 | +0.356 | [+0.280, +0.428] | <0.001 | sí | no |
| H5 (S6 - S5) | 60 | +0.063 | [+0.030, +0.095] | <0.001 | sí | no |
| H6 (E2 - E1) | 60 | -0.278 | [-0.345, -0.211] | <0.001 | sí | no |
| H5-rep (S6b - S5) | 60 | +0.053 | [+0.017, +0.087] | 0.005 (sin Holm) | sí | no |

Segundo juez:

| Contraste | n | Diferencia | IC 95 % | p Holm | Significativo | Cambia |
|---|---|---|---|---|---|---|
| H1 (S3 - S2) | 60 | -0.128 | [-0.181, -0.080] | <0.001 | sí | no |
| H2 (S5 - S3) | 60 | +0.210 | [+0.169, +0.251] | <0.001 | sí | no |
| H3 (S5 - S4) | 60 | -0.030 | [-0.051, -0.009] | 0.009 | sí | no |
| H4 (S2 - S1) | 60 | +0.345 | [+0.271, +0.418] | <0.001 | sí | no |
| H5 (S6 - S5) | 60 | +0.046 | [+0.016, +0.075] | 0.009 | sí | no |
| H6 (E2 - E1) | 60 | -0.261 | [-0.330, -0.193] | <0.001 | sí | no |
| H5-rep (S6b - S5) | 60 | +0.037 | [+0.004, +0.071] | 0.037 (sin Holm) | sí | no |

Decisiones declaradas:

| Decisión | Conjunto completo | Subconjunto | Cambia |
|---|---|---|---|
| H1 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 60 | no |
| H1 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 60 | no |
| H1 conclusion (both judges) | sustained | sustained | no |
| H2 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 60 | no |
| H2 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 60 | no |
| H2 conclusion (both judges) | sustained | sustained | no |
| H3 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm 0.038, n 60 | no |
| H3 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm 0.009, n 60 | no |
| H3 conclusion (both judges) | sustained | sustained | no |
| H4 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 60 | no |
| H4 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 60 | no |
| H4 conclusion (both judges) | sustained | sustained | no |
| H5 significant at 0.05, primary judge | True p Holm 0.007, n 116 | True p Holm <0.001, n 60 | no |
| H5 significant at 0.05, second judge | True p Holm 0.013, n 116 | True p Holm 0.009, n 60 | no |
| H5 conclusion (both judges) | sustained | sustained | no |
| H6 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 60 | no |
| H6 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 60 | no |
| H6 conclusion (both judges) | sustained | sustained | no |
| H5-rep significant at 0.05, primary judge | True p 0.004, n 116 | True p 0.005, n 60 | no |
| H5-rep significant at 0.05, second judge | True p 0.024, n 116 | True p 0.037, n 60 | no |
| H5-rep conclusion (both judges) | sustained | sustained | no |
| Hybrid rule: S4 significantly better than S5 under both judges | True | True | no |
| Embedding rule, condition 1: H6 does not show qwen significantly better | False | False | no |
| Embedding rule, outcome | keep qwen/qwen3-embedding-8b | keep qwen/qwen3-embedding-8b | no |

## (d) Todos los estratos salvo A (B, C y D)

Consultas: 124 (B: 72, C: 40, D: 12).

Juez principal:

| Contraste | n | Diferencia | IC 95 % | p Holm | Significativo | Cambia |
|---|---|---|---|---|---|---|
| H1 (S3 - S2) | 124 | -0.107 | [-0.144, -0.071] | <0.001 | sí | no |
| H2 (S5 - S3) | 124 | +0.194 | [+0.162, +0.224] | <0.001 | sí | no |
| H3 (S5 - S4) | 124 | -0.013 | [-0.028, +0.003] | 0.111 | no | sí |
| H4 (S2 - S1) | 124 | +0.356 | [+0.302, +0.409] | <0.001 | sí | no |
| H5 (S6 - S5) | 124 | +0.026 | [+0.001, +0.050] | 0.090 | no | sí |
| H6 (E2 - E1) | 124 | -0.226 | [-0.268, -0.185] | <0.001 | sí | no |
| H5-rep (S6b - S5) | 124 | +0.033 | [+0.003, +0.065] | 0.034 (sin Holm) | sí | no |

Segundo juez:

| Contraste | n | Diferencia | IC 95 % | p Holm | Significativo | Cambia |
|---|---|---|---|---|---|---|
| H1 (S3 - S2) | 124 | -0.106 | [-0.139, -0.072] | <0.001 | sí | no |
| H2 (S5 - S3) | 124 | +0.191 | [+0.161, +0.221] | <0.001 | sí | no |
| H3 (S5 - S4) | 124 | -0.021 | [-0.034, -0.008] | 0.004 | sí | no |
| H4 (S2 - S1) | 124 | +0.346 | [+0.292, +0.401] | <0.001 | sí | no |
| H5 (S6 - S5) | 124 | +0.034 | [+0.013, +0.055] | 0.004 | sí | no |
| H6 (E2 - E1) | 124 | -0.208 | [-0.252, -0.165] | <0.001 | sí | no |
| H5-rep (S6b - S5) | 124 | +0.037 | [+0.016, +0.060] | 0.001 (sin Holm) | sí | no |

Decisiones declaradas:

| Decisión | Conjunto completo | Subconjunto | Cambia |
|---|---|---|---|
| H1 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 124 | no |
| H1 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 124 | no |
| H1 conclusion (both judges) | sustained | sustained | no |
| H2 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 124 | no |
| H2 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 124 | no |
| H2 conclusion (both judges) | sustained | sustained | no |
| H3 significant at 0.05, primary judge | True p Holm <0.001, n 116 | False p Holm 0.111, n 124 | sí |
| H3 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm 0.004, n 124 | no |
| H3 conclusion (both judges) | sustained | not conclusive | sí |
| H4 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 124 | no |
| H4 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 124 | no |
| H4 conclusion (both judges) | sustained | sustained | no |
| H5 significant at 0.05, primary judge | True p Holm 0.007, n 116 | False p Holm 0.090, n 124 | sí |
| H5 significant at 0.05, second judge | True p Holm 0.013, n 116 | True p Holm 0.004, n 124 | no |
| H5 conclusion (both judges) | sustained | not conclusive | sí |
| H6 significant at 0.05, primary judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 124 | no |
| H6 significant at 0.05, second judge | True p Holm <0.001, n 116 | True p Holm <0.001, n 124 | no |
| H6 conclusion (both judges) | sustained | sustained | no |
| H5-rep significant at 0.05, primary judge | True p 0.004, n 116 | True p 0.034, n 124 | no |
| H5-rep significant at 0.05, second judge | True p 0.024, n 116 | True p 0.001, n 124 | no |
| H5-rep conclusion (both judges) | sustained | sustained | no |
| Hybrid rule: S4 significantly better than S5 under both judges | True | False | sí |
| Embedding rule, condition 1: H6 does not show qwen significantly better | False | False | no |
| Embedding rule, outcome | keep qwen/qwen3-embedding-8b | keep qwen/qwen3-embedding-8b | no |
