# V05 · Juez LLM de relevancia con procedencia y medida de acuerdo

- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** evaluación (`app/evaluation/`), infraestructura (`app/infrastructure/llm/`, `app/infrastructure/config/`), datos (`data/evaluation/`)

## Qué se ha hecho

- `prompts_judge.py`: prompt de relevancia versionado
  (`RELEVANCE_JUDGE_PROMPT_VERSION = "v1.0"`) que usa las etiquetas de grado ya
  registradas en el repositorio (docstring de `RelevanceJudgment` en
  `app/evaluation/types.py`) y no añade otras guías. Juicio puntual, un solo
  juez, temperatura 0, razonamiento desactivado, tope de 256 tokens y salida
  estructurada con la justificación antes del grado. El juez solo ve la
  consulta y los campos del catálogo, nunca la nota original, el sistema ni la
  posición.
- `app/infrastructure/llm/relevance_judge.py`: adaptador del juez,
  configurable con `JUDGE_PROVIDER`, `JUDGE_MODEL` y `JUDGE_BASE_URL`
  (`JudgeSettings`), con `LLM_*` como valores por defecto. Temperatura,
  razonamiento y tope de tokens son fijos.
- `app/evaluation/relevance_judge.py`: `JudgmentCache` (caché reanudable, un
  JSON Lines por modelo y versión de prompt en `data/evaluation/llm_judgments/`,
  volcado al disco en cada juicio), `find_unjudged`, `merge_judgments`,
  `agreement_report` y `stratified_sample`. No importa LangChain.
- `merge_judgments` conserva siempre el juicio original. Cada juicio lleva
  `judge` (`original` o `llm`); los del LLM añaden `model`, `prompt_version`,
  `date` y `rationale`. `relevance_judgments.json` no se toca; el conjunto
  fusionado es `relevance_judgments_extended.json`: 637 originales más 405 del
  LLM, 1 042 en total.

## Por qué

Había 637 juicios graduados (0-3) de procedencia desconocida (`QRELS_DATASHEET.md`). Judged@10 estaba entre 0,507 y 0,550 y un libro sin
juicio cuenta como no relevante, de modo que un sistema nuevo que recupere
libros no juzgados queda penalizado (sesgo de pool). La
nota de validación (`docs/notes/2026-11-11-validacion-proyecto-2026.md`)
aplaza la ampliación del catálogo hasta disponer de un juez reproducible.

## Objetivo

Cubrir los huecos del top 10 registrando el origen de cada nota y medir antes
cuánto concuerda el juez con los juicios existentes, sin tocar el original.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Juicio humano de todos los huecos | Es la referencia correcta, pero cada sistema nuevo añade huecos (405 solo con los cuatro sistemas base) y el trabajo se repite; se conserva como trabajo pendiente una muestra humana para validar el juez |
| Juzgar la matriz completa de 44 x 177 (7 788 pares) | Unas 8,6 horas a 4 segundos por par con el modelo local; agrupar el top k de cada sistema es práctica habitual y basta para llegar a Judged@10 = 1 |
| Juez alojado de otra familia de modelos | Es lo que recomienda la literatura; el 2026-11-11 no había clave de API, de modo que queda configurable |
| Varios jueces o prompts de rol | La revisión interna del 2026-11-03 recoge que Thomas et al. (arXiv 2309.10621) no encontraron mejora de acuerdo con ninguno de los dos |
| Juicio por pares o por listas | Las notas puntuales se pueden guardar por par y las reutiliza cualquier sistema futuro |
| Marcos genéricos de evaluación de RAG (RAGAS, DeepEval) | Son jueces LLM sin calibrar; aquí el juez se contrasta con notas existentes antes de usarlo |
| Sobrescribir el fichero de juicios original | El original queda intacto y el conjunto ampliado registra el origen de cada nota |
| Apoyarse solo en listas condensadas para los documentos sin juicio | La tarea V03 mostró que es una comprobación débil a profundidad 10 |

## Cómo se ha validado

- Acuerdo medido antes de rellenar: 200 de los 637 pares, semilla 0,
  estratificados por (grado, idioma de la consulta) y juzgados a ciegas. El
  umbral de aceptación (kappa ponderado cuadrático de 0,4) se fijó de antemano.
- Resultado: 0,729, por encima del umbral, por lo que no se revisó el prompt.
- Ejecución real con `qwen3.8:27b-mlx` en Ollama: unos 4 segundos por juicio;
  el acuerdo (200 pares) tardó unos 13 minutos y el relleno (405 pares) unos
  26, sin fallos.
- Tests: 49 nuevos con jueces falsos. Suite completa tras la segunda oleada:
  885 superados, 0 fallidos.

## Resultados

Acuerdo con los juicios originales (de `JUDGE_AGREEMENT.md`):

| Pares | n | Exacto | A menos de 1 | Kappa ponderado cuadrático [IC 95 %] | Kappa (grado >= 1) | Kappa (grado >= 2) | Media LLM - original |
|---|---|---|---|---|---|---|---|
| todos | 200 | 0.685 | 0.965 | 0.729 [0.645, 0.796] | 0.702 | 0.609 | -0.070 |
| consultas en | 107 | 0.682 | 0.981 | 0.721 [0.585, 0.810] | 0.588 | 0.718 | -0.037 |
| consultas es | 93 | 0.688 | 0.946 | 0.735 [0.618, 0.829] | 0.807 | 0.504 | -0.108 |

Matriz de confusión, todos los pares (filas: original; columnas: LLM):

| Original \ LLM | 0 | 1 | 2 | 3 | Total |
|---|---|---|---|---|---|
| 0 | 68 | 8 | 0 | 0 | 76 |
| 1 | 17 | 52 | 7 | 2 | 78 |
| 2 | 4 | 12 | 13 | 9 | 38 |
| 3 | 0 | 1 | 3 | 4 | 8 |

Patrones de desacuerdo (de un script de análisis puntual, no de los ficheros
generados): el grado 0 es estable (8 de 76 suben a 1, ninguno más). El grado 1
baja a 0 en 17 de 78 pares, por lo general libros del dominio correcto pero del
tema equivocado (un libro de cocina mexicana para "recetas cocina rumana"). El
grado 2 es el menos estable: 13 se mantienen, 12 bajan a 1, 9 suben a 3 y 4
bajan a 0. El juez es algo más estricto con libros sin descripción (-0,17;
n = 58; kappa 0,747). Con libro en otro idioma que la consulta kappa es 0,544
(n = 43) frente a 0,771 si coinciden. Siete pares difieren en dos o más grados.

Relleno: los 405 pares sin juicio del top 10 de los cuatro sistemas sin LLM
recibieron 309 / 55 / 22 / 19 notas de grado 0 / 1 / 2 / 3. Judged@10 pasó de
0,507-0,550 a 1,000.
Efecto en la comparación a profundidad 10 (de `COMPARISON_EXTENDED_QRELS.md`;
44 consultas, línea base `hybrid_rrf`):

| Sistema | nDCG@10 originales [IC 95 %] | nDCG@10 ampliados [IC 95 %] | Dif. ampliada frente a hybrid_rrf [IC 95 %] | p (Holm) ampliado |
|---|---|---|---|---|
| lexical_only | 0.460 [0.356, 0.564] | 0.491 [0.391, 0.591] | -0.097 [-0.149, -0.051] | <0.001 |
| vector_only | 0.530 [0.436, 0.623] | 0.598 [0.524, 0.670] | +0.010 [-0.053, +0.069] | 1.000 |
| hybrid_rrf | 0.531 [0.436, 0.626] | 0.588 [0.500, 0.676] | | |
| hybrid_rrf_mmr | 0.529 [0.436, 0.622] | 0.590 [0.502, 0.677] | +0.002 [-0.006, +0.010] | 1.000 |

Lectura:
1. Con los juicios ampliados, BM25 solo queda significativamente por debajo
   del híbrido (diferencia de -0,097; p ajustado por Holm < 0,001). Con los
   originales el p ajustado era 0,083.
2. El vectorial y el híbrido siguen sin distinguirse (+0,010; p bruto 0,762).
3. Recall@10 baja en `lexical_only` (de 0.441 a 0.415) porque cada consulta
   tiene ahora más libros relevantes conocidos.
4. Las notas del LLM no son humanas: la tabla ampliada es una comprobación de
   robustez frente al sesgo de pool, no una sustituta.

Como contexto, la revisión interna del 2026-11-03 recoge que UMBRELA (arXiv
2406.06519) informa de kappa de Cohen de cuatro niveles entre 0,31 y 0,37 frente
a humanos en colecciones TREC. Las cifras no son comparables: las colecciones
son distintas y aquí las notas de referencia tienen procedencia desconocida.

## Cómo usarlo

```
python -m app.evaluation.relevance_judge agreement --sample 200 --seed 0
python -m app.evaluation.relevance_judge fill --k 10
```

`agreement` escribe `JUDGE_AGREEMENT.md` y `judge_agreement.json`; `fill` escribe
el conjunto ampliado y `COMPARISON_EXTENDED_QRELS.md`. Ambas aceptan `--limit N`
(como mucho N pares nuevos; la caché los conserva) y `fill --limit 0` es un
ensayo en seco. Otro juez: `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_BASE_URL`.
Tests: `pytest tests/evaluation/test_relevance_judge.py tests/infrastructure/llm/test_relevance_judge_adapter.py`.

## Límites que debe conocer el tribunal

- El juez es el mismo modelo que el LLM del sistema (el único local). Los
  cuatro sistemas comparados no usan LLM, pero evaluar la ruta con LLM con
  estos juicios conlleva riesgo de autopreferencia.
- El acuerdo se mide frente a juicios de procedencia desconocida: es acuerdo,
  no exactitud.
- El intervalo de kappa remuestrea pares e ignora que los de una misma consulta
  no son independientes: es optimista.
- La muestra tiene solo 8 pares de grado 3; los subgrupos son pequeños.
- Cada par se juzgó una vez a temperatura 0; el resultado depende de la versión
  de Ollama y de la cuantización del modelo.
- `fill` cubre los cuatro sistemas base; cada sistema nuevo tiene huecos nuevos,
  que la tarea V06 rellena con `find_unjudged`.
- Los jueces alojados (OpenAI, Anthropic) solo tienen tests unitarios;
  `langchain-anthropic` no está instalado.

## Trabajo pendiente

- Muestra de 150 a 200 pares anotada por el autor, para validar el juez frente a una persona.
- Repetir el acuerdo con un juez de otra familia de modelos cuando haya clave de API u otro modelo local (basta `JUDGE_PROVIDER` y `JUDGE_MODEL`).
## Ficheros

- `app/evaluation/relevance_judge.py`, `relevance_judgments_extended.json`.
- `app/infrastructure/llm/relevance_judge.py`, `prompts_judge.py`,
  `schemas_judge.py`; `app/infrastructure/config/settings.py` (`JudgeSettings`).
- `data/evaluation/JUDGE_AGREEMENT.md`, `judge_agreement.json`,
  `COMPARISON_EXTENDED_QRELS.md`, `llm_judgments/` (605 líneas: 200 de la
  muestra de acuerdo y 405 del relleno).
- `tests/evaluation/test_relevance_judge.py`, `tests/infrastructure/llm/test_relevance_judge_adapter.py`.
