# V03 · Higiene de evaluación

- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** evaluación (`app/evaluation/`), datos (`data/evaluation/`)

## Qué se ha hecho

- `statistics.py`: se añaden `bootstrap_ci`, un bootstrap percentil con semilla
  que remuestrea consultas, y `paired_bootstrap_ci`, su versión pareada para la
  diferencia entre dos sistemas. Solo usan `numpy`. El test de permutación
  pareado y la corrección de Holm ya existían, pero solo los usaba la
  evaluación de la ruta con LLM.
- `evaluation_service.py`: las métricas pasan a funciones de módulo que operan
  sobre listas de identificadores (`ndcg_at_k`, `recall_at_k`,
  `precision_at_k`, `reciprocal_rank`, `judged_fraction_at_k`,
  `condensed_ranking`). Los métodos del servicio las llaman, de modo que el
  comportamiento no cambia. `judged_at_k` vive ahora en ese módulo y las dos
  evaluaciones lo comparten.
- `TestQuery.language` (opcional) y una etiqueta de idioma en las 44 consultas:
  24 en inglés y 20 en español. `q35` ("python") y `q38` ("java") se escriben
  igual en ambos idiomas y se etiquetaron como inglés por convención.
- Módulo nuevo `comparison.py` con `compare_systems`, `describe_judgments`,
  `render_markdown`, `run_system` y una línea de comandos. Por sistema informa
  nDCG@10 (métrica primaria) con intervalo bootstrap del 95 %, Recall@10, P@5,
  MRR, Judged@10, nDCG@10 condensado (se eliminan los documentos sin juicio
  antes de puntuar) y el desglose por idioma de la consulta. Frente a una línea
  base añade la diferencia con intervalo bootstrap pareado, el p-valor de
  permutación y el p-valor ajustado por Holm.
- Salidas: `results_comparison_baseline.json`, `COMPARISON_BASELINE.md` y
  `QRELS_DATASHEET.md`, en `data/evaluation/`. El JSON guarda el SHA-256 de los
  ficheros de consultas y juicios y el commit de git.

## Por qué

Antes de esta tarea la colección tenía 44 consultas, 637 juicios graduados
(0-3) y 177 libros. No había intervalos de confianza, ni desglose por idioma,
ni registro de dónde procedían los juicios. La nota de validación
(`docs/notes/2026-11-11-validacion-proyecto-2026.md`) concluye que sin esto
ninguna comparación entre sistemas es defendible, y que la afirmación de que el
híbrido se distingue del vectorial (p = 0,65) no se sostenía con ese
tamaño de colección.

## Objetivo

Poder comparar los sistemas sin LLM con una sola orden reproducible que
entregue, para cada uno, la métrica primaria con su incertidumbre, la cobertura
de los juicios y el contraste frente a la línea base con corrección por
comparaciones múltiples. Además, dejar escrito qué se sabe y qué no se sabe de
los juicios.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Prueba t de Student o Wilcoxon de rangos con signo | La literatura de recuperación de información que sigue el proyecto (Smucker et al. 2007; Urbano et al. 2019, recogida en la revisión interna del 2026-11-03) recomienda el test de aleatorización pareado, que no supone normalidad |
| Corrección de Bonferroni | Holm controla la misma tasa de error y nunca es menos potente |
| Usar las bibliotecas `ranx` o `ir-measures` | `ranx` no implementa corrección por comparaciones múltiples, y unas cien líneas de `numpy` que se pueden leer y probar se defienden mejor que una dependencia nueva |
| MRR como métrica primaria | nDCG@10 se declara primaria de antemano; MRR se informa como secundaria |
| Deducir de memoria la procedencia de los juicios | Lo que el repositorio no registra se declara desconocido |
| Detección automática del idioma de las consultas | Es poco fiable con consultas de dos palabras; las 44 consultas se etiquetaron leyéndolas |
| Cuatro decimales | Se usan tres, que ya es más precisión de la que sostienen 44 consultas |

## Cómo se ha validado

- Contraste con resultados almacenados. A profundidad 10, `hybrid_rrf` da
  nDCG@10 = 0,531, idéntico consulta por consulta al de
  `results_llm_path_k10.json`. A profundidad 100, los cuatro sistemas
  reproducen `results_v2.json` exactamente (0,4603; 0,5301; 0,5424; 0,5274).
- Tests: 33 tests nuevos en `tests/evaluation/` (`test_bootstrap.py`,
  `test_comparison.py`, `test_collection_loading.py`). La suite completa tras
  la primera oleada de tareas: 771 superados, 0 fallidos.
- Ficha de los juicios: las estadísticas se calculan con código
  (`describe_judgments`). Los 637 juicios se resuelven a libros del catálogo y
  no hay pares repetidos. El conjunto de candidatos (pool) se reproduce con el
  código actual: `pool_candidates.json` es la unión del top 20 de los cuatro
  sistemas sin LLM y `pool_candidates_v2.json`, la del top 50.

## Resultados

Profundidad 10, 44 consultas, intervalos bootstrap del 95 % (10 000
remuestreos, semilla 0). Tablas copiadas de `data/evaluation/COMPARISON_BASELINE.md`
(ejecución sobre el commit `1a1a9ad` con cambios sin confirmar).

| Sistema | nDCG@10 [IC 95 %] | Recall@10 | P@5 | MRR | Judged@10 | nDCG@10 condensado | Dif. nDCG@10 frente a hybrid_rrf [IC 95 %] | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| lexical_only | 0.460 [0.356, 0.564] | 0.441 | 0.423 | 0.571 | 0.507 | 0.522 | -0.071 [-0.132, -0.010] | 0.028 | 0.083 |
| vector_only | 0.530 [0.436, 0.623] | 0.498 | 0.500 | 0.712 | 0.532 | 0.578 | -0.001 [-0.062, +0.057] | 0.979 | 1.000 |
| hybrid_rrf (línea base) | 0.531 [0.436, 0.626] | 0.555 | 0.486 | 0.678 | 0.550 | 0.630 | | | |
| hybrid_rrf_mmr | 0.529 [0.436, 0.622] | 0.555 | 0.473 | 0.684 | 0.550 | 0.633 | -0.002 [-0.008, +0.005] | 0.647 | 1.000 |

| Sistema | nDCG@10 en (n=24) [IC 95 %] | nDCG@10 es (n=20) [IC 95 %] |
|---|---|---|
| lexical_only | 0.396 [0.253, 0.543] | 0.537 [0.389, 0.679] |
| vector_only | 0.495 [0.358, 0.627] | 0.572 [0.445, 0.701] |
| hybrid_rrf | 0.492 [0.362, 0.622] | 0.578 [0.440, 0.709] |
| hybrid_rrf_mmr | 0.498 [0.370, 0.627] | 0.567 [0.430, 0.698] |

Lectura:

1. A profundidad 10, el híbrido y el vectorial no se distinguen (diferencia
   de -0,001; p = 0,979). El p = 0,65 citado antes correspondía a profundidad 100.
2. BM25 solo queda 0,071 por debajo del híbrido. El intervalo sin ajustar
   excluye el cero y el p bruto es 0,028, pero el p ajustado por Holm es
   0,083. La memoria debe citar el valor ajustado.
3. Judged@10 está entre 0,507 y 0,550: cerca de la mitad de los diez primeros
   resultados no tiene juicio y se puntúa como no relevante.
4. El nDCG@10 condensado es una comprobación débil a profundidad 10, porque se
   calcula con las mismas listas de 10 elementos y solo puede subir libros
   juzgados dentro de ellas.

Ficha de los juicios (`QRELS_DATASHEET.md`): se juzga el 8,2 % de la matriz de
44 x 177 pares; juicios por consulta, mínimo 1, mediana 6, máximo 44; grados
0/1/2/3 = 241/251/120/25; cuatro consultas sin ningún libro relevante (`q22`,
`q32`, `q38`, `q39`, todas en inglés); inglés con 339 juicios (231 relevantes)
y español con 298 (165 relevantes). Solo 637 de los 1 518 candidatos del pool
tienen nota: 13 consultas (`q04` a `q16`) tienen el pool completo y las otras
31 tienen 170 de 1 051.

## Cómo usarlo

```
python -m app.evaluation.comparison
```

Tarda unos 7 segundos. Parámetros: `--depth`, `--baseline`, `--n-resamples`,
`--seed`, `--queries-path`, `--judgments-path`, `--output-json`, `--output-md`.
Para ejecutar los tests: `pytest tests/evaluation/`.

## Límites y trabajo pendiente

- La ficha declara "Desconocido: no consta en el repositorio" para: quién o
  qué asignó las notas, número de anotadores y su acuerdo, si el anotador veía
  nombres o posiciones de los sistemas, la fecha (como mucho 2025-12-22, el
  commit que añadió los juicios), las guías de anotación más allá de las cuatro
  etiquetas de una línea, por qué 31 pools quedaron parcialmente anotados, la
  orden exacta que generó `pool_candidates.json` y quién redactó las consultas.
  El autor debe completar lo que recuerde.
- El pool incompleto lo aborda la tarea V05 (juez LLM de relevancia).
- Con 44 consultas la potencia estadística es limitada. Hacen falta más
  consultas; queda aplazado porque requiere intervención de una persona.
- Los resultados corresponden a profundidad 10 y a un árbol de trabajo con
  cambios sin confirmar; no se ha comprobado si cambian con otros índices.

## Ficheros

- `app/evaluation/statistics.py`: `bootstrap_ci`, `paired_bootstrap_ci`.
- `app/evaluation/comparison.py`: comparación, descripción de juicios, CLI.
- `app/evaluation/evaluation_service.py`: métricas como funciones de módulo.
- `app/evaluation/types.py`, `app/evaluation/test_queries.json`: campo `language`.
- `data/evaluation/COMPARISON_BASELINE.md`, `results_comparison_baseline.json`,
  `QRELS_DATASHEET.md`.
- `tests/evaluation/test_bootstrap.py`, `test_comparison.py`,
  `test_collection_loading.py`.
