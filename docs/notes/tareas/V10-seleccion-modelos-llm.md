# V10 · Selección de los modelos LLM sobre las tareas del proyecto

- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** evaluación (`app/evaluation/`), infraestructura (`app/infrastructure/llm/`, `app/infrastructure/config/`), datos (`data/evaluation/`)

## Qué se ha hecho

- `app/evaluation/llm_model_selection.py`: comparación de cinco modelos alojados en OpenRouter sobre las tres tareas LLM del proyecto (juez de relevancia, explicaciones fundamentadas y comprensión de la consulta). Las dos reglas de selección están codificadas (`select_system_llm`, `select_judge`) y el informe `data/evaluation/LLM_MODEL_SELECTION.md` se genera desde los resultados en bruto, sin llamar a ningún modelo.
- `app/evaluation/judge_sensitivity.py`: vuelve a puntuar con el juez elegido los 919 pares que el juez local de V06 había puntuado, y compara los siete contrastes de V06 bajo ambos jueces (`data/evaluation/RETRIEVAL_ABLATION_JUDGE_SENSITIVITY.md`).
- Valores por defecto: el juez de OpenRouter pasa de `minimax/minimax-m3.1-flash` a `z-ai/glm-5.3-flash`; el LLM del sistema `qwen/qwen3.8-flash` ya estaba como valor por defecto desde V08.
- El registro de uso de V08 guarda ahora también el coste que informa OpenRouter por llamada, y el cliente LLM expone las citas y los filtros en bruto, antes de las protecciones, para poder medirlos.

## Por qué

El 2026-11-11 el autor decidió dejar de usar el modelo local de Ollama (demasiado lento), hacer todo el trabajo con LLM a través de OpenRouter, no usar modelos GPT y usar los mejores modelos chinos baratos. Las clasificaciones públicas de esos modelos que se encontraron ese mismo día en una búsqueda web son fuentes secundarias que se contradicen entre sí, y ninguna mide explicaciones fundamentadas sobre libros en español e inglés. La elección se hizo midiendo los modelos en las tres tareas del proyecto.

## Objetivo

Elegir un LLM del sistema (comprensión de la consulta y explicaciones) y un juez de relevancia de otro proveedor, con una regla escrita antes de ver ningún resultado y aplicada de forma mecánica. Comprobar además si las conclusiones de V06 dependen del juez que rellenó los huecos.

## Diseño y regla de selección fijados de antemano

Candidatos: `qwen/qwen3.8-flash`, `qwen/qwen3.7-flash`, `deepseek/deepseek-v4.1-flash`, `minimax/minimax-m3.1-flash`, `z-ai/glm-5.3-flash` y `z-ai/glm-5.3`. `qwen/qwen3.7-flash` se descartó antes de medir: ninguno de sus endpoints admite salida estructurada con esquema JSON (OpenRouter responde 404). MiniMax y los dos GLM no permiten desactivar el razonamiento y se ejecutaron con esfuerzo bajo; Qwen y DeepSeek, con el razonamiento desactivado.

Tres mediciones:

- **A. Juez.** Concordancia con 200 juicios existentes, la misma muestra con semilla que V05 (`agreement --sample 200 --seed 0`, estratificada por grado e idioma de la consulta), prompt `v1.0`, temperatura 0. Métrica: kappa ponderado cuadrático con intervalo bootstrap.
- **B. Explicaciones.** Las 10 primeras consultas en español y las 10 primeras en inglés por identificador, con los 3 primeros resultados del híbrido de producción: 60 explicaciones por modelo a través del `LangChainLLMClient` real (temperatura 0, hasta 1024 tokens de salida). Se mide la precisión de citas antes de la protección (proporción de citas escritas por el modelo que `app/domain/citation_validation.py` acepta), las explicaciones sin ninguna cita válida y la proporción escrita en el idioma de la consulta.
- **C. Comprensión de la consulta.** Las 44 consultas de prueba por `extract_query_intent`: proporción interpretada y proporción con un filtro de idioma o de año que la consulta no pide (la regla determinista de `filter_guards.py`), medida antes de las protecciones.

Regla del idioma de B: se pliegan las tildes y se cuentan las palabras que son stopwords exclusivas del español o del inglés del tokenizador de BM25 (las que están en ambas listas se ignoran); decide la mayoría y un empate cuenta como no estar en el idioma de la consulta. Se aplica al resumen y al razonamiento, no a los fragmentos citados. Se comprobó a mano en algunos casos, no se validó de forma sistemática.

Reglas de selección:

- **LLM del sistema:** entre los candidatos con el 100 % de llamadas interpretadas en B y C, la mayor precisión de citas antes de la protección. Los candidatos a menos de 0,02 del mejor empatan; el empate se resuelve por la mayor proporción de explicaciones en el idioma de la consulta y después por el menor coste.
- **Juez:** entre los candidatos de un proveedor distinto del LLM del sistema elegido, el mayor kappa ponderado cuadrático. A menos de 0,02 se empata y gana el más barato.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Elegir según clasificaciones públicas | Las que se encontraron se contradicen entre sí y ninguna mide explicaciones fundamentadas de libros en español e inglés |
| Elegir el modelo más barato | El más barato de los candidatos no puede producir salida estructurada y el siguiente más barato falló una llamada en cada tarea |
| Un solo modelo para los dos papeles | Un juez del mismo proveedor que el LLM del sistema corre el riesgo de favorecer su propia salida |
| Decidir después de ver los números | La regla se escribió antes de las ejecuciones y se aplicó mecánicamente; la opinión discrepante del ingeniero se recoge aparte |
| Incluir modelos GPT, Claude o Gemini | Excluidos por decisión del autor |
| Mantener el modelo local como juez | Ya no se usa, y un juez alojado de otro proveedor hace la evaluación reproducible por cualquiera que tenga una clave |

## Cómo se ha validado

- La regla está codificada y se aplica sin intervención (`select_system_llm`, `select_judge`); el informe repite la regla y la decisión con el motivo de cada exclusión.
- Se añadieron 49 tests nuevos (`tests/evaluation/test_llm_model_selection.py`, `test_judge_sensitivity.py` y los del registro de uso y del cliente); la suite completa pasa.
- La reejecución de la ablación sobre los juicios originales reproduce las cifras de V05, y las dos ejecuciones de la ablación devolvieron el mismo top 10 para cada sistema y consulta, de modo que solo cambian los juicios.
- Contraste del gasto: el contador de uso de la clave de OpenRouter pasó de 0,0809 a 0,3924 USD durante la medición (0,3115 USD), frente a 0,32400 USD sumando el coste informado por llamada; el contador se lee justo tras la última llamada y puede ir por detrás.

## Resultados

A. Juez, 200 pares (kappa ponderado cuadrático frente a los juicios originales):

| Modelo | Kappa [IC 95 %] | Exacto | Fallos | s / juicio | Coste real / juicio (USD) |
|---|---|---|---|---|---|
| `qwen/qwen3.8-flash` | 0.752 [0.678, 0.814] | 0.655 | 0 | 2.28 | 0.000084 |
| `deepseek/deepseek-v4.1-flash` | 0.714 [0.629, 0.786] | 0.650 | 0 | 1.02 | 0.000084 |
| `minimax/minimax-m3.1-flash` | 0.739 [0.660, 0.805] | 0.665 | 0 | 1.81 | 0.000085 |
| `z-ai/glm-5.3-flash` | 0.765 [0.704, 0.817] | 0.650 | 0 | 2.03 | 0.000071 |
| `z-ai/glm-5.3` | 0.715 [0.635, 0.784] | 0.670 | 0 | 1.06 | 0.000189 |
| `qwen3.8:27b-mlx` (local, V05, solo referencia) | 0.729 [0.645, 0.796] | 0.685 | 0 | unos 4 | n/d |

B. Explicaciones, 60 por modelo:

| Modelo | Interpretadas | Citas | Precisión de citas | Sin cita válida | En el idioma de la consulta | Latencia mediana (s) | Coste / explicación (USD) |
|---|---|---|---|---|---|---|---|
| `qwen/qwen3.8-flash` | 60/60 | 137 | 0.964 | 0.017 | 1.000 | 6.61 | 0.000283 |
| `deepseek/deepseek-v4.1-flash` | 60/60 | 179 | 0.972 | 0.000 | 0.983 | 2.68 | 0.000409 |
| `minimax/minimax-m3.1-flash` | 60/60 | 236 | 0.958 | 0.017 | 0.633 | 2.98 | 0.000407 |
| `z-ai/glm-5.3-flash` | 60/60 | 170 | 0.959 | 0.000 | 0.950 | 4.35 | 0.000310 |
| `z-ai/glm-5.3` | 59/60 | 166 | 0.970 | 0.000 | 1.000 | 2.85 | 0.000935 |

C. Comprensión de la consulta, 44 por modelo:

| Modelo | Interpretadas | Idioma no pedido | Año no pedido | Idioma o año no pedido | Latencia mediana / consulta (s) | Coste / búsqueda (USD) |
|---|---|---|---|---|---|---|
| `qwen/qwen3.8-flash` | 44/44 | 0.227 | 0.045 | 0.250 | 8.19 | 0.001037 |
| `deepseek/deepseek-v4.1-flash` | 44/44 | 0.386 | 0.023 | 0.386 | 2.81 | 0.001498 |
| `minimax/minimax-m3.1-flash` | 44/44 | 0.341 | 0.023 | 0.341 | 3.73 | 0.001466 |
| `z-ai/glm-5.3-flash` | 43/44 | 0.205 | 0.045 | 0.227 | 5.06 | 0.001249 |
| `z-ai/glm-5.3` | 44/44 | 0.273 | 0.000 | 0.273 | 3.21 | 0.003620 |

Coste por búsqueda: una comprensión de la consulta más tres explicaciones. Los precios de lista son del 2026-11-11 (por ejemplo `qwen/qwen3.8-flash` 0,15 USD de entrada y 0,47 de salida por millón de tokens).

Patrones de fallo:

- MiniMax respondió en español a 22 consultas en inglés; GLM-5.3-Flash a 3 y DeepSeek a 1.
- GLM-5.3 devolvió una explicación con JSON no válido. GLM-5.3-Flash falló una consulta porque sus tokens de razonamiento agotaron el tope de 1024 tokens.
- Las citas no válidas son entre 5 y 10 por modelo; todas son fragmentos que no aparecen en el libro, ninguna cita un campo equivocado.
- Los filtros no pedidos son casi todos una nacionalidad leída como idioma ("cocina española" a `es`, "Italian pasta" a `it`), más "siglo XIX" a 1800-1899 y "Victorian" a 1837-1901.

Concordancia entre jueces: los candidatos concuerdan entre sí (kappa por pares de 0.824 a 0.900) más que ninguno con los juicios originales (0.714 a 0.765).

## Decisión

Aplicación mecánica de la regla:

- **LLM del sistema: `qwen/qwen3.8-flash`.** Los dos GLM quedan excluidos por una llamada fallida cada uno (GLM-5.3-Flash 43/44 consultas, GLM-5.3 59/60 explicaciones). Entre los elegibles, la mejor precisión de citas es 0.972 (DeepSeek); Qwen (0.964) y MiniMax (0.958) están a menos de 0,02, así que los tres empatan. Desempata la proporción en el idioma de la consulta: 1.000, 0.983 y 0.633.
- **Juez: `z-ai/glm-5.3-flash`**, con kappa 0.765. Qwen queda excluido por ser del mismo proveedor que el LLM del sistema. El siguiente candidato elegible, MiniMax (0.739), queda fuera del margen de 0,02, así que no hay empate.

Opinión del ingeniero que ejecutó la tarea, que no forma parte de la regla: Qwen es el candidato más lento (6.6 s por explicación y 8.2 s por consulta, frente a 2.7 s y 2.8 s de DeepSeek). DeepSeek obtuvo la mejor precisión de citas y sus filtros no pedidos adicionales los descartan las protecciones existentes, de modo que, si la latencia interactiva importa, DeepSeek es la alternativa. Para el juez, los intervalos se solapan mucho: GLM-5.3-Flash, Qwen y MiniMax son prácticamente intercambiables.

## Sensibilidad de la ablación V06 al juez

Los 919 pares que el juez local había puntuado se volvieron a puntuar con `z-ai/glm-5.3-flash` (0 fallos, 0,059 USD). Ambos jueces concuerdan en ellos con kappa ponderado cuadrático 0.847 [0.822, 0.871]; la concordancia exacta es 0.812 y el nuevo juez puntúa de media +0.038 más alto.

Los siete contrastes declarados de antemano conservan su dirección y su significación tras la corrección de Holm bajo ambos jueces (7 de 7). C5 (reranker bge) sigue siendo el único significativo (p de Holm 0.008 con el juez antiguo y 0.019 con el nuevo; diferencia +0.098 y +0.100). C4a y C4b pierden su significación sin ajustar con el juez nuevo (p de 0.045 a 0.070 y de 0.026 a 0.075), aunque ninguno era significativo tras Holm. Con ambos jueces, nDCG@10 de `hybrid_norm_e5_bge` es 0.668 y 0.701.

## Límites que debe conocer el tribunal

- Cada modelo se ejecutó una vez. OpenRouter enruta cada llamada a uno de varios endpoints (hardware y cuantización distintos), por lo que una repetición puede diferir.
- Las muestras son pequeñas (60 explicaciones y 44 consultas por modelo): diferencias de unos pocos puntos están dentro del ruido. El intervalo del kappa ignora que los pares de una misma consulta no son independientes y es optimista.
- El juez se compara con juicios de procedencia desconocida (`QRELS_DATASHEET.md`): se mide concordancia, no corrección. Que los candidatos concuerden más entre sí que con los juicios originales es coherente con que esos juicios contengan ruido o una lectura distinta de la escala, pero no lo prueba: los modelos también pueden compartir sesgos.
- Todos los candidatos son de proveedores chinos por decisión del autor; no se dice nada de otros modelos.
- El criterio del idioma de las explicaciones es una heurística de palabras vacías, comprobada a mano en casos sueltos.
- Precios y versiones son los del 2026-11-11 y cambian a menudo. La latencia incluye el enrutamiento de OpenRouter y varía según el endpoint.
- La sensibilidad de V06 reemplaza un juez por otro, ambos LLM; no aporta juicios humanos.
- Gasto total de la tarea: 0,389 USD.

## Cómo reproducirlo

```
python -m app.evaluation.llm_model_selection run      # llama a los modelos (necesita OPENROUTER_API_KEY)
python -m app.evaluation.llm_model_selection report   # regenera el informe, sin llamar a ningún modelo
python -m app.evaluation.judge_sensitivity grade --workers 4
python -m app.evaluation.retrieval_ablation --no-judge --extended-judgments-path app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json --output-json data/evaluation/results_retrieval_ablation_z-ai-glm-5.3-flash.json --output-md ""
python -m app.evaluation.judge_sensitivity report
```

`grade` usa el juez de `JUDGE_PROVIDER` / `JUDGE_MODEL` (ver `.env.example`) y se reanuda desde su caché. Tests: `pytest tests/evaluation/test_llm_model_selection.py tests/evaluation/test_judge_sensitivity.py`.

## Ficheros

- `app/evaluation/llm_model_selection.py`, `app/evaluation/judge_sensitivity.py`.
- `app/infrastructure/llm/usage.py`, `app/infrastructure/llm/langchain_llm_client.py`, `app/infrastructure/config/settings.py`: coste por llamada, citas y filtros en bruto, juez por defecto.
- `data/evaluation/LLM_MODEL_SELECTION.md`, `results_llm_model_selection.json`, `RETRIEVAL_ABLATION_JUDGE_SENSITIVITY.md`, `results_judge_sensitivity.json`, `results_retrieval_ablation_z-ai-glm-5.3-flash.json`.
- `data/evaluation/llm_judgments/` (cachés de los juicios por modelo), `app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json`.
- `tests/evaluation/test_llm_model_selection.py`, `tests/evaluation/test_judge_sensitivity.py`.
