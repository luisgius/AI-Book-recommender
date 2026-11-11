# V06 · Ablación unificada de la recuperación

- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** evaluación (`app/evaluation/`), datos (`data/evaluation/`)

## Qué se ha hecho

- `app/evaluation/retrieval_ablation.py`: experimento controlado sobre las 44 consultas (24 en inglés, 20 en español) y el catálogo de 177 libros. El diseño está como datos (`SYSTEMS`, `CONTRASTS`) y el informe lo imprime tal como se declaró.
- Once sistemas con 10 resultados cada uno: BM25 con tokenizador de espacios o normalizador; FAISS con tres modelos de embeddings; cuatro híbridos RRF; y el híbrido con e5 más uno de dos rerankers cross-encoder sobre los 30 primeros. `hybrid_legacy` es el sistema en producción antes de esta tarea.
- Relleno de huecos: el juez de V05 puntuó 514 pares nuevos (grados 0 / 1 / 2 / 3: 291 / 150 / 52 / 21; 7 solo los recuperan las variantes con tildes). `relevance_judgments_extended.json` pasa de 1 042 a 1 556 juicios (637 originales y 919 del LLM); `relevance_judgments.json` no se toca.
- Informe `data/evaluation/RETRIEVAL_ABLATION.md` con comprobación de reproducción frente a V05, tabla de tildes y tabla de coste.

## Por qué

Las tareas V01 a V05 añadieron por separado un tokenizador normalizador, embeddings configurables, un reranker y un juez LLM. Faltaba una medición conjunta que dijera qué componentes mejoran la recuperación en este catálogo antes de fijar la configuración por defecto (`docs/notes/2026-11-11-validacion-proyecto-2026.md`).

## Objetivo

Responder con siete preguntas declaradas de antemano, sobre juicios sin huecos en el top 10, qué componentes ayudan, con intervalos y corrección por comparaciones múltiples; el resto se presenta como descripción.

## Diseño fijado de antemano

El diseño (sistemas, siete contrastes, métrica y corrección de Holm sobre exactamente esos siete) se redactó en el encargo antes de ejecutar ningún sistema; el fichero de salida lo repite.

- Métrica: nDCG@10 (ganancia 2^grado - 1; sin juicio cuenta como no relevante) con Judged@10 = 1,000 en los once sistemas.
- Contrastes (sistema frente a referencia): C1 `bm25_normalized` / `bm25_whitespace`; C2 `vec_e5` / `vec_minilm`; C3 `vec_qwen3` / `vec_minilm`; C4a `hybrid_norm_e5` / `hybrid_legacy`; C4b `hybrid_norm_qwen3` / `hybrid_legacy`; C5 `hybrid_norm_e5_bge` / `hybrid_norm_e5`; C6 `hybrid_norm_e5_mmarco` / `hybrid_norm_e5`.
- Test de aleatorización pareado bilateral (cambios de signo) sobre el nDCG@10 por consulta, 10 000 remuestreos, semilla 0; Holm sobre los siete; intervalo bootstrap percentil pareado del 95 % sobre consultas. Valores fijos: RRF con k = 60 y profundidad de reranking 30.
- Solo descriptivos: tabla de sistemas, por idioma, tildes, coste y contrastes sobre los juicios originales.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Elegir los componentes uno a uno, mirando cada resultado antes de definir la siguiente comparación | Multiplica los caminos posibles a través de los datos e infla los falsos positivos; primero se fijaron los sistemas y los contrastes |
| Contrastar todos los pares de los once sistemas | Son 55 comparaciones y no quedaría potencia tras la corrección con 44 consultas; se eligieron siete preguntas de antemano y el resto se presenta como descriptivo |
| Ajustar los pesos de RRF o la profundidad del reranking con las mismas 44 consultas | Sería ajustar al conjunto de prueba; se usaron valores fijos (k = 60, profundidad 30) |
| Evaluar solo con los juicios originales | Los sistemas nuevos recuperan libros nunca juzgados (Judged@10 cae a 0,39-0,52) y quedarían penalizados por ello |
| Listas profundas (100 resultados) | El producto muestra diez resultados; la profundidad 10 es lo que ve el usuario |
| Informar solo de la media | Cada cifra lleva un intervalo bootstrap y cada contraste un test de aleatorización pareado |

## Cómo se ha validado

- Comprobación de reproducción integrada: `hybrid_legacy`, `bm25_whitespace` y `vec_minilm` reproducen las cifras de V05 (405 pares); todas las filas coinciden.
- El juez que rellenó los 514 pares fue el modelo local `qwen3.8:27b-mlx` (prompt v1.0), el mismo validado en V05 (kappa ponderado cuadrático 0,729 frente a 200 juicios existentes); 0 fallos, 4,84 s por juicio.
- El coordinador repitió la medición con `python -m app.evaluation.retrieval_ablation --no-judge` (sin LLM, 3 minutos): reprodujo todas las cifras desde los juicios en caché y pasó la comprobación de reproducción frente a V05.
- Suite completa tras la tarea: 982 superados, 0 fallidos (incluye `tests/evaluation/test_retrieval_ablation.py`).

## Resultados

Contrastes declarados de antemano, juicios finales (nDCG@10; diferencia con intervalo del 95 %; p bruto y p con corrección de Holm):

| Contraste | Sistema | Referencia | nDCG@10 sistema | nDCG@10 referencia | Dif. [IC 95 %] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| C1 | `bm25_normalized` | `bm25_whitespace` | 0.471 | 0.418 | +0.054 [+0.009, +0.108] | 0.026 | 0.155 |
| C2 | `vec_e5` | `vec_minilm` | 0.621 | 0.523 | +0.098 [+0.006, +0.187] | 0.040 | 0.159 |
| C3 | `vec_qwen3` | `vec_minilm` | 0.592 | 0.523 | +0.069 [-0.007, +0.145] | 0.082 | 0.164 |
| C4a | `hybrid_norm_e5` | `hybrid_legacy` | 0.570 | 0.508 | +0.062 [+0.002, +0.118] | 0.045 | 0.159 |
| C4b | `hybrid_norm_qwen3` | `hybrid_legacy` | 0.580 | 0.508 | +0.072 [+0.010, +0.134] | 0.026 | 0.155 |
| C5 | `hybrid_norm_e5_bge` | `hybrid_norm_e5` | 0.668 | 0.570 | +0.098 [+0.046, +0.155] | 0.001 | 0.008 |
| C6 | `hybrid_norm_e5_mmarco` | `hybrid_norm_e5` | 0.538 | 0.570 | -0.032 [-0.113, +0.045] | 0.437 | 0.437 |

Todos los sistemas (descriptivo):

| Sistema | nDCG@10 [IC 95 %] | Recall@10 | P@5 | MRR |
|---|---|---|---|---|
| `bm25_whitespace` | 0.418 [0.329, 0.509] | 0.277 | 0.518 | 0.646 |
| `bm25_normalized` | 0.471 [0.378, 0.564] | 0.302 | 0.564 | 0.684 |
| `vec_minilm` | 0.523 [0.442, 0.602] | 0.415 | 0.618 | 0.787 |
| `vec_e5` | 0.621 [0.542, 0.696] | 0.481 | 0.709 | 0.796 |
| `vec_qwen3` | 0.592 [0.506, 0.675] | 0.460 | 0.691 | 0.791 |
| `hybrid_legacy` | 0.508 [0.423, 0.593] | 0.392 | 0.605 | 0.768 |
| `hybrid_norm_minilm` | 0.529 [0.440, 0.617] | 0.399 | 0.636 | 0.762 |
| `hybrid_norm_e5` | 0.570 [0.484, 0.651] | 0.427 | 0.686 | 0.794 |
| `hybrid_norm_qwen3` | 0.580 [0.494, 0.663] | 0.444 | 0.645 | 0.765 |
| `hybrid_norm_e5_bge` | 0.668 [0.582, 0.745] | 0.518 | 0.718 | 0.838 |
| `hybrid_norm_e5_mmarco` | 0.538 [0.454, 0.618] | 0.409 | 0.618 | 0.743 |

Por idioma de la consulta (nDCG@10, descriptivo):

| Sistema | en (n=24) [IC 95 %] | es (n=20) [IC 95 %] |
|---|---|---|
| `bm25_whitespace` | 0.385 [0.257, 0.516] | 0.457 [0.322, 0.585] |
| `bm25_normalized` | 0.378 [0.251, 0.508] | 0.584 [0.453, 0.703] |
| `vec_minilm` | 0.499 [0.391, 0.604] | 0.552 [0.430, 0.671] |
| `vec_e5` | 0.538 [0.414, 0.651] | 0.722 [0.648, 0.793] |
| `vec_qwen3` | 0.502 [0.382, 0.621] | 0.700 [0.595, 0.797] |
| `hybrid_legacy` | 0.480 [0.365, 0.595] | 0.541 [0.409, 0.668] |
| `hybrid_norm_minilm` | 0.473 [0.356, 0.592] | 0.596 [0.463, 0.719] |
| `hybrid_norm_e5` | 0.486 [0.362, 0.604] | 0.670 [0.564, 0.767] |
| `hybrid_norm_qwen3` | 0.474 [0.359, 0.585] | 0.708 [0.597, 0.809] |
| `hybrid_norm_e5_bge` | 0.554 [0.427, 0.671] | 0.804 [0.730, 0.869] |
| `hybrid_norm_e5_mmarco` | 0.524 [0.402, 0.639] | 0.554 [0.438, 0.667] |

Robustez a tildes (consultas en español almacenadas sin diacríticos; 14 de las 20 tienen variante con tildes; nDCG@10; diferencia = con tildes menos sin ellas):

| Sistema | es almacenada | es con tildes | Diferencia [IC 95 %] | Top 10 idéntico |
|---|---|---|---|---|
| `bm25_whitespace` | 0.457 | 0.501 | +0.044 [-0.007, +0.109] | 5 de 14 |
| `bm25_normalized` | 0.584 | 0.584 | +0.000 [+0.000, +0.000] | 14 de 14 |
| `vec_minilm` | 0.552 | 0.552 | +0.000 [+0.000, +0.000] | 14 de 14 |
| `vec_e5` | 0.722 | 0.715 | -0.007 [-0.025, +0.005] | 0 de 14 |
| `vec_qwen3` | 0.700 | 0.713 | +0.013 [+0.001, +0.029] | 1 de 14 |
| `hybrid_legacy` | 0.541 | 0.558 | +0.016 [-0.003, +0.048] | 5 de 14 |
| `hybrid_norm_minilm` | 0.596 | 0.596 | +0.000 [+0.000, +0.000] | 14 de 14 |
| `hybrid_norm_e5` | 0.670 | 0.672 | +0.001 [-0.003, +0.005] | 2 de 14 |
| `hybrid_norm_qwen3` | 0.708 | 0.708 | -0.000 [-0.002, +0.002] | 5 de 14 |
| `hybrid_norm_e5_bge` | 0.804 | 0.799 | -0.005 [-0.027, +0.015] | 0 de 14 |
| `hybrid_norm_e5_mmarco` | 0.554 | 0.556 | +0.001 [-0.026, +0.030] | 0 de 14 |

Coste por consulta (`mps`, Apple M5 Pro; tras una consulta de calentamiento):

| Sistema | Mediana ms / consulta | p95 ms / consulta | Consulta de calentamiento ms |
|---|---|---|---|
| `bm25_whitespace` | 0.1 | 0.1 | 0 |
| `bm25_normalized` | 0.1 | 0.1 | 0 |
| `vec_minilm` | 3.6 | 19.0 | 30 |
| `vec_e5` | 11.3 | 26.2 | 651 |
| `vec_qwen3` | 19.3 | 72.8 | 953 |
| `hybrid_legacy` | 3.4 | 3.5 | 4 |
| `hybrid_norm_minilm` | 3.4 | 3.6 | 4 |
| `hybrid_norm_e5` | 11.3 | 12.2 | 12 |
| `hybrid_norm_qwen3` | 19.6 | 22.2 | 21 |
| `hybrid_norm_e5_bge` | 2242.3 | 2392.4 | 2856 |
| `hybrid_norm_e5_mmarco` | 342.4 | 398.5 | 402 |

Sensibilidad con los juicios originales (descriptiva): con solo los 637 juicios originales ningún contraste es significativo tras Holm; C2 pasa a
-0.063 [-0.145, +0.020] y C5 a +0.043 [-0.026, +0.111], y C6 queda en -0.112 [-0.201, -0.028] con p de Holm 0.130.

## Lectura de los resultados

1. Con la corrección de Holm solo se sostiene estadísticamente C5: el reranker `BAAI/bge-reranker-v2-m3` sobre el híbrido con e5 (+0.098, p de Holm 0.008).
2. Hay estimaciones puntuales positivas cuyo intervalo sin ajustar excluye el cero, pero que no sobreviven a la corrección con 44 consultas: el tokenizador normalizador (C1), `multilingual-e5-large-instruct` frente a MiniLM (C2) y los dos híbridos nuevos frente al de producción (C4a y C4b). C3 (embeddings Qwen3) es positivo con un intervalo que incluye el cero.
3. El reranker pequeño de mMARCO no ayuda (C6, estimación puntual negativa).
4. Las mejoras se concentran en las consultas en español. Con `bm25_normalized` frente a `bm25_whitespace` se pasa de 0.457 a 0.584; con `vec_e5` frente a `vec_minilm`, de 0.552 a 0.722; con `hybrid_norm_e5_bge` frente a `hybrid_legacy`, de 0.541 a 0.804. En inglés las diferencias son pequeñas.
5. Hallazgo incómodo: con un modelo de embeddings multilingüe fuerte, el híbrido RRF puntúa menos que la parte vectorial sola (`hybrid_norm_e5` 0.570 frente a `vec_e5` 0.621). La fusión con pesos iguales y un ranking BM25 más débil arrastra el resultado hacia abajo en este catálogo pequeño. No se declaró ningún test para esta comparación, de modo que es descriptiva. El reranker recupera la pérdida y algo más (`hybrid_norm_e5_bge` 0.668).
6. Robustez a tildes: el tokenizador normalizador hace que BM25 sea invariante a las tildes por construcción (top 10 idéntico en 14 de 14), mientras que el de espacios cambia sus resultados (5 de 14 idénticos).
7. Coste: el reranker bge multiplica la latencia mediana por consulta, de 11.3 ms (`hybrid_norm_e5`) a 2242.3 ms (`hybrid_norm_e5_bge`), en GPU. La nota V04 midió unos 10 segundos en CPU con profundidad 30.

## Decisiones que se derivan

Tomadas por el autor y el diseñador el 2026-11-11, para aplicarlas en una tarea posterior y no en esta:

- El reranking con `BAAI/bge-reranker-v2-m3` pasa a formar parte del pipeline por defecto (es la única mejora respaldada), con su latencia por reducir (profundidad, longitud de entrada) y una forma documentada de desactivarlo.
- El modelo de embeddings por defecto pasa a ser `intfloat/multilingual-e5-large-instruct`: el autor exige consultas en cualquier idioma, su estimación puntual es la mejor de los tres en la parte vectorial, tiene licencia MIT y codifica varias veces más rápido que `Qwen/Qwen3-Embedding-0.6B` (la investigación de catálogos del mismo día midió 85,1 textos por segundo en media precisión frente a 8,6 de Qwen3 en el mismo portátil), lo que importa para un catálogo de cientos de miles de libros.
- El tokenizador normalizador pasa a ser el predeterminado en los índices BM25 nuevos.
- El hallazgo sobre la fusión (punto 5) debe reexaminarse con el catálogo mayor, añadiendo el brazo "parte vectorial + reranker" y una fusión ponderada, antes de afirmar que el híbrido es mejor que la parte vectorial.

## Límites que debe conocer el tribunal

- Son 44 consultas: las diferencias de nDCG@10 inferiores a unos 0,08 son difíciles de detectar. Que solo C5 sobreviva a Holm refleja en parte esa falta de potencia y no prueba que los demás componentes no ayuden.
- El catálogo tiene 177 libros; los resultados pueden no trasladarse a uno de cientos de miles.
- 919 de los 1 556 juicios proceden de un juez LLM de la misma familia de modelos que el LLM del sistema usado en otras partes del proyecto. Ninguno de los once sistemas lo usa, pero las notas del LLM no son notas humanas.
- La tabla de sensibilidad con los juicios originales muestra que las conclusiones sobre los sistemas nuevos dependen de las notas rellenadas por el LLM: con solo los originales, C2 cambia de signo (-0.063) y C5 pasa a +0.043 con un intervalo que incluye el cero. El efecto está en parte inducido por el sesgo de pool (los sistemas nuevos recuperan libros sin juicio original), pero no se puede separar de un posible sesgo del juez.
- Los intervalos de confianza de cada contraste por separado no están ajustados por multiplicidad; solo lo están los p de Holm.
- Las latencias se midieron con un servidor LLM local cargado en memoria, en una GPU compartida con otras tareas: son orientativas.
- Los rerankers recuperan 60 candidatos por recuperador en lugar de 20, de modo que C5 y C6 miden la etapa tal como correría en producción, con el conjunto de candidatos mayor incluido.
- Pendiente: el 2026-11-11 el autor decidió dejar de usar el modelo local y usar modelos alojados mediante OpenRouter. Se prevé, como análisis de sensibilidad, volver a puntuar los pares juzgados por el LLM con un juez de otra familia de modelos y repetir los siete contrastes. Aún no se ha hecho.

## Cómo reproducirlo

```
python -m app.evaluation.retrieval_ablation
python -m app.evaluation.retrieval_ablation --no-judge
```

El primer comando juzga con el LLM local los pares que falten (configuración `JUDGE_*`, ver `.env.example`); el segundo no necesita LLM y falla si algún resultado del top 10 carece de juicio. Con los modelos en caché hay que ejecutarlos con conexión: con transformers 4.57.3, `HF_HUB_OFFLINE=1` hace fallar la carga de los tokenizadores de e5 y Qwen3. Los índices de los modelos no predeterminados se construyen en la primera ejecución en `data/indexes_ablation/<slug>/` (ignorado por git). Tests: `pytest tests/evaluation/test_retrieval_ablation.py`.

## Ficheros

- `app/evaluation/retrieval_ablation.py`, `app/evaluation/relevance_judgments_extended.json`.
- `data/evaluation/RETRIEVAL_ABLATION.md`, `results_retrieval_ablation.json`, `llm_judgments/` (caché del juez).
- `tests/evaluation/test_retrieval_ablation.py`.
