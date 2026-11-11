# V16 · Modelos por API y selección por velocidad y calidad

- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** infraestructura (`app/infrastructure/search/`, `app/infrastructure/llm/`, `app/infrastructure/config/`), aplicación (`app/api/v1/dependencies.py`), evaluación (`app/evaluation/`), datos (`data/evaluation/`)

## Qué se ha hecho

- Se ha separado el codificador de texto del almacén de embeddings con un cambio compatible hacia atrás: `EmbeddingsStoreFaiss` acepta un codificador inyectado y, por defecto, sigue usando sentence-transformers en local. Todos los tests existentes pasaron sin modificarse y el índice entregado se carga como antes. El protocolo `TextEncoder` y las convenciones de consulta y documento de cada modelo viven en `app/infrastructure/search/text_encoding.py`, sin importar ninguna biblioteca de modelos.
- `OpenRouterEmbeddingEncoder`: envía los documentos por lotes con un tope de tamaño de petición, reintenta ante errores transitorios, aplica un tiempo máximo y fija el proveedor. Envía `dimensions` solo a los modelos entrenados para truncarse (los embeddings Qwen3) y aplica a cada modelo la convención de consulta y documento de su ficha.
- `ApiReranker`: implementa el puerto `Reranker` existente, con el mismo contrato y el mismo comportamiento ante fallos que el cross-encoder local. Si falla, `SearchService` conserva el orden fusionado.
- Ajustes `EMBEDDING_PROVIDER` y `RERANKER_PROVIDER` (`local` por defecto, o `openrouter`) en `app/infrastructure/config/settings.py`, y registro de uso para las operaciones `embedding` y `rerank`.
- La identidad guardada con un índice incluye ahora modelo, proveedor y dimensión, de modo que un índice construido con una combinación no puede cargarse con otra. `torch` no se importa cuando solo se usa la vía por API.
- `app/evaluation/api_models_evaluation.py`: medición de los modelos por API frente a los locales, con la regla de selección por velocidad y calidad. El informe `data/evaluation/API_MODELS.md` se genera desde las respuestas guardadas.
- Reaplicación de la misma regla al LLM del sistema y al juez sobre las mediciones de V10 ya guardadas, sin llamadas nuevas. La regla y la decisión de V10 siguen visibles en `data/evaluation/LLM_MODEL_SELECTION.md` como historia, con la nueva regla marcada como sustituta.

## Por qué

El backend del perfil nube no puede cargar ningún modelo en memoria, porque se ejecutará en una función serverless pequeña (`docs/notes/2026-11-11-propuesta-nube-y-mcp.md`, secciones 2 y 3). Los embeddings y el reranking tienen que venir, por tanto, de una API. El autor fijó además: modelos a través de OpenRouter, modelos chinos baratos, sin GPT, "necesitamos modelos equilibrados entre velocidad y calidad" y "medimos y probamos".

En V10 el LLM del sistema se eligió solo por calidad y resultó ser el candidato más lento (6.61 s por explicación). Había que aplicar la misma regla a ese papel.

## Objetivo

Elegir, para el perfil nube, un modelo de embeddings y un reranker por API, y revisar el LLM del sistema y el juez, con una regla escrita antes de ver ningún resultado que cuente la velocidad además de la calidad. Comprobar con la medición si los modelos por API igualan a los locales evaluados y cuantificar su coste, su latencia y su repetibilidad.

## Diseño y regla de selección fijados de antemano

Colección: los 177 libros y las 44 consultas de prueba, profundidad 10, con todos los resultados del top 10 juzgados (Judged@10 = 1.000 en todos los sistemas). Se añadieron 290 notas del juez LLM a los juicios (cifra del informe `API_MODELS.md`); el agente que ejecutó la medida informó de 285 juicios nuevos en la caché. La diferencia de 5 no se ha desglosado; es compatible con pares que ya estaban en la caché por otra tarea, pero no se ha comprobado.3-flash`.

- **Embeddings, lado vectorial con índice exacto.** Referencias locales: `vec_e5` y `vec_qwen3`. Brazos por API: `baai/bge-m3`, `intfloat/multilingual-e5-large` y `qwen/qwen3-embedding-8b` a 1024 dimensiones. Brazos descriptivos: `qwen/qwen3-embedding-8b` a 512, 256 y su dimensión nativa, y `qwen/qwen3-embedding-4b`.
- **Reranker, sobre los mismos 30 candidatos.** Referencia local `hybrid_norm_e5_bge`; brazos por API `qwen/qwen3-reranker-0.6b` y `qwen/qwen3-reranker-8b`. El recomendador LLM de V07 (`rag_llm_recommender`, ejecución 1, sin repetir) entra como candidato de la regla.
- **Proveedores fijados** con `provider.order` y sin alternativas: embeddings en `deepinfra` y rerankers en `primeintellect`, para que los vectores de un brazo salgan de un solo punto de servicio y la latencia no dependa del enrutamiento.
- **Contrastes declarados de antemano:** cinco, con corrección de Holm (tres embeddings por API frente a `vec_e5`, A1 a A3, y dos rerankers por API frente al reranker bge local, A4 y A5). Prueba de aleatorización pareada de dos colas sobre el nDCG@10 por consulta, 10000 remuestreos, semilla 0, e intervalos bootstrap pareados del 95 %.
- **Regla de selección.** Para cada papel, los candidatos a menos de 0,02 del mejor en la métrica primaria de calidad empatan; entre ellos gana la menor latencia mediana; si otro empatado está a menos del 10 % de esa latencia, gana el más barato. Un candidato con cualquier llamada fallida queda excluido.
- **LLM del sistema:** la regla se aplica a las mediciones de V10 ya guardadas. Calidad = precisión de citas antes de la protección; todas las llamadas interpretadas y al menos el 95 % de las explicaciones en el idioma de la consulta.
- **Juez:** proveedor distinto del LLM del sistema, mayor concordancia (kappa), banda de empate de 0,02 y gana el más barato; la velocidad no cuenta.
- Todo lo demás es descriptivo: brazos descriptivos, Recall@10, MRR, columnas por idioma, estabilidad y la comprobación de extremo a extremo.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Alojar los dos modelos locales con el backend | Necesitan unos 6-8 GB de memoria con su entorno de ejecución, lo que descarta una función serverless gratuita |
| Elegir solo por calidad, como en V10 | Eligió el LLM más lento; el autor pidió equilibrio |
| Elegir el modelo más barato de cada papel | Las diferencias de precio por búsqueda son fracciones de céntimo; la latencia y la calidad difieren mucho más |
| Un LLM como reranker en la nube (el recomendador de V07) | Unas once veces más lento que el reranker por API elegido y fuera de la banda de empate en calidad |
| Una segunda cuenta de proveedor de modelos para tener más oferta o rerankers más baratos | Una sola clave cubre el LLM, los embeddings y el reranking, lo que además simplifica la puesta en marcha a quien quiera ejecutar el sistema |
| Mantener `intfloat/multilingual-e5-large-instruct` en la nube por paridad con el perfil local | OpenRouter no sirve la variante instruct; la variante sin instruct fue la más lenta y la más baja de los brazos por API |
| Enviar `dimensions` a todos los modelos | Un proveedor trunca también los modelos no entrenados para ello y los degrada en silencio |
| Fiarse de las tablas de clasificación publicadas | No miden este catálogo, estas consultas ni la latencia del servicio alojado |

## Cómo se ha validado

- La regla está codificada (`select_by_speed_quality` en `app/evaluation/llm_model_selection.py`) y se aplica sin intervención; el informe repite la decisión con el motivo de cada exclusión.
- Se añadieron 93 tests nuevos. La suite completa pasa: 1439 superados, 12 omitidos y 0 fallos.
- La reejecución de la evaluación con `--offline` reprodujo todas las tablas a partir de las respuestas guardadas, sin ninguna llamada. Las referencias locales reproducen el top 10 de la ablación de V06 en las 44 consultas (`vec_e5`, `vec_qwen3`, `hybrid_norm_e5` y `hybrid_norm_e5_bge`).
- Comprobación de extremo a extremo con el `SearchService` real: BM25 y FAISS locales, embedding de la consulta por API y reranker por API.
- Estabilidad: las 44 consultas se repitieron dos veces con cada brazo por API.
- Contraste del gasto: el contador de uso de la clave marcó 0,0589 USD en la última ejecución en vivo; el informe suma el coste informado por cada respuesta.

## Resultados

Embeddings, lado vectorial (nDCG@10 con intervalo del 95 %; latencia mediana de la llamada de consulta):

| Sistema | Dim. | nDCG@10 | Latencia mediana (ms) | p95 (ms) | Llamadas fallidas |
|---|---|---|---|---|---|
| `vec_e5` (local) | 1024 | 0.655 [0.579, 0.726] | 68.4 | 90.9 | 0 |
| `vec_qwen3` (local) | 1024 | 0.606 [0.522, 0.685] | 20.3 | 43.5 | 0 |
| `baai/bge-m3` | 1024 | 0.614 [0.531, 0.693] | 423 | 833 | 3 |
| `intfloat/multilingual-e5-large` | 1024 | 0.591 [0.516, 0.662] | 1,176 | 1,882 | 0 |
| `qwen/qwen3-embedding-8b` | 1024 | 0.654 [0.568, 0.733] | 486 | 2,845 | 0 |
| `qwen/qwen3-embedding-8b` (descriptivo) | 512 | 0.660 [0.569, 0.742] | 543 | 2,788 | 0 |
| `qwen/qwen3-embedding-8b` (descriptivo) | 256 | 0.671 [0.590, 0.745] | 489 | 1,244 | 0 |
| `qwen/qwen3-embedding-8b` (descriptivo) | 4096 | 0.672 [0.587, 0.749] | 602 | 3,054 | 0 |
| `qwen/qwen3-embedding-4b` (descriptivo) | 1024 | 0.653 [0.573, 0.728] | 424 | 1,156 | 0 |

Rerankers sobre los mismos 30 candidatos (los candidatos sin reranker obtienen 0.588 [0.499, 0.675]):

| Sistema | nDCG@10 | Latencia mediana (ms) | p95 (ms) | Coste / 1,000 búsquedas (USD) |
|---|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` (local) | 0.681 [0.607, 0.752] | 4,286 | 4,890 | 0 |
| `qwen/qwen3-reranker-0.6b` | 0.684 [0.601, 0.763] | 441 | 635 | 1.524 |
| `qwen/qwen3-reranker-8b` | 0.758 [0.690, 0.823] | 590 | 730 | 1.524 |
| `qwen/qwen3.8-flash` (recomendador LLM de V07) | 0.736 [0.667, 0.801] | 6,462 | 11,659 | 0.628 |

Contrastes declarados de antemano (diferencia con intervalo del 95 %, p sin ajustar y p de Holm):

| Contraste | Diferencia | p | p (Holm) |
|---|---|---|---|
| A1 `baai/bge-m3` - `vec_e5` | -0.041 [-0.115, +0.028] | 0.284 | 0.851 |
| A2 `intfloat/multilingual-e5-large` - `vec_e5` | -0.064 [-0.130, -0.000] | 0.059 | 0.237 |
| A3 `qwen/qwen3-embedding-8b` (1024) - `vec_e5` | -0.001 [-0.061, +0.059] | 0.978 | 1.000 |
| A4 `qwen/qwen3-reranker-0.6b` - bge local | +0.002 [-0.054, +0.052] | 0.934 | 1.000 |
| A5 `qwen/qwen3-reranker-8b` - bge local | +0.076 [+0.023, +0.133] | 0.009 | 0.044 |

El único resultado con apoyo estadístico es A5: el reranker por API `qwen/qwen3-reranker-8b` supera al `BAAI/bge-reranker-v2-m3` local en esta colección, con una latencia mediana de 590 ms frente a 4,286 ms (alrededor de una séptima parte). Ningún modelo de embeddings por API difiere de forma significativa del e5 local.

Comprobación de extremo a extremo (`e2e_api_stack`, descriptiva):

| Sistema | nDCG@10 | Latencia mediana total (ms) | p95 (ms) | Llamada de embedding (ms) | Recuperación (ms) | Llamada de rerank (ms) | Coste / búsqueda (USD) |
|---|---|---|---|---|---|---|---|
| `e2e_api_stack` | 0.740 [0.665, 0.811] | 1,166 | 4,933 | 557 | 5.3 | 576 | 0.001331 |
| `hybrid_norm_e5_bge` (local) | 0.681 [0.607, 0.752] | 4,378 | - | 68.4 | - | 4,286 | 0 |

Las medianas de cada columna se toman por separado y no suman.

Estabilidad entre dos ejecuciones de las mismas 44 consultas:

| Sistema | Listas top 10 que cambiaron |
|---|---|
| `baai/bge-m3` | 2 / 44 |
| `intfloat/multilingual-e5-large` | 0 / 44 |
| `qwen/qwen3-embedding-8b` a 1024 | 13 / 44 |
| `qwen/qwen3-embedding-8b` a 512 | 17 / 44 |
| `qwen/qwen3-embedding-8b` a 256 y a 4096 | 23 / 44 |
| `qwen/qwen3-reranker-8b` | 0 / 44 |

El modelo `qwen/qwen3-embedding-8b` alojado devuelve vectores ligeramente distintos en llamadas repetidas. Dos proveedores del mismo modelo devuelven vectores distintos para el mismo texto (coseno 0.997), por lo que el proveedor se fija y forma parte de la identidad del índice.

Sistema LLM y juez, sobre las mediciones de V10 (sin llamadas nuevas): `qwen/qwen3.8-flash` obtiene precisión de citas 0.964 y 6.61 s por explicación; `deepseek/deepseek-v4.1-flash`, 0.972 y 2.68 s. `minimax/minimax-m3.1-flash` queda excluido por responder en el idioma de la consulta solo en 0.633 de las explicaciones, y los dos GLM por no tener todas las llamadas interpretadas.

Lo aprendido de las API:

- El parámetro `dimensions` trunca cualquier modelo en un proveedor, también uno no entrenado para ello, por lo que el codificador lo envía solo a los embeddings Qwen3.
- Un lote de documentos se rechaza si la petición completa supera el contexto del endpoint. Eso causó tres llamadas fallidas con `baai/bge-m3` en la primera ejecución y motivó el tope de 20,000 caracteres por petición.
- El e5 alojado trunca en silencio la entrada a 512 tokens.
- Los precios del reranking no figuran en la lista de modelos y se leyeron del coste que informa cada respuesta: 0,20 USD por millón de tokens en ambos rerankers.

Gasto total: 0,356 USD (0,3385 de llamadas a la API y 0,0173 del juez).

## Decisión

Aplicación mecánica de la regla fijada de antemano:

- **Embeddings en la nube: `qwen/qwen3-embedding-8b` a 1024 dimensiones.** Mejor calidad (0.654) y ningún otro candidato dentro de la banda de empate. `baai/bge-m3` queda excluido por sus tres llamadas fallidas.
- **Reranker en la nube: `qwen/qwen3-reranker-8b`.** Mejor calidad (0.758), sin otro candidato dentro de la banda y sin exclusiones.
- **LLM del sistema: `deepseek/deepseek-v4.1-flash`.** Empata en precisión de citas con `qwen/qwen3.8-flash` (0.972 frente a 0.964) y responde en 2.68 s por explicación frente a 6.61 s.
- **Juez: `z-ai/glm-5.3-flash`, sin cambios.**

## Reservas del diseñador

Opinión del diseñador, fechada el 2026-11-11, separada de la regla mecánica y que no la modifica:

1. La regla solo mira la latencia mediana. El modelo de embeddings elegido tiene un percentil 95 de 2,845 ms frente a 833 ms de `baai/bge-m3`, y sus vectores no son repetibles entre llamadas. `baai/bge-m3` queda 0.040 por debajo en la estimación puntual (diferencia no significativa), es estable, lo sirven varios proveedores y puede ejecutarse en local con pesos abiertos, lo que permitiría el mismo modelo en los dos perfiles. Se mantiene el resultado de la regla porque se fijó de antemano. Se repetirá la comparación con el catálogo grande, declarando la latencia en cola y la repetibilidad como criterios.
2. Las tres llamadas fallidas que excluyen a `baai/bge-m3` según la regla vinieron del tamaño de petición del adaptador, no del modelo. La exclusión no cambia la decisión.
3. Las cifras de evaluación deben calcularse siempre a partir de vectores guardados, nunca de llamadas nuevas, por la falta de repetibilidad.
4. Las dimensiones menores (512 o 256) puntuaron lo mismo dentro del ruido en los brazos descriptivos y ahorrarían almacenamiento. Sin embargo, los vectores son una parte pequeña del tamaño guardado por obra en la base de datos de la nube (medido en V12), así que se mantiene 1024.

## Límites que debe conocer el tribunal

- Son 44 consultas y 177 libros: las diferencias de nDCG@10 inferiores a unos 0.08 son difíciles de detectar, y el catálogo de la nube es mucho mayor.
- Las notas añadidas provienen de un juez LLM (`z-ai/glm-5.3-flash`), no de personas.
- La latencia alojada incluye el enrutamiento y varió durante la ejecución: las peticiones de documentos del modelo de embeddings elegido tardaron entre 1.4 y 17.8 segundos.
- Cada brazo se midió una vez, más la repetición de estabilidad, en minutos distintos.
- Las latencias de las referencias locales se midieron mientras otro proceso usaba la CPU y la GPU, y son más altas que en V06.
- Hay un solo proveedor por modelo. La plantilla que el proveedor aplica al reranker Qwen3 no está documentada.
- Precios y disponibilidad son los del 2026-11-11. Los modelos alojados pueden cambiar o desaparecer, por lo que el perfil local con pesos abiertos sigue siendo el sistema evaluado.
- Pendiente de integración (siguiente tarea): el cargador de Postgres sigue construyendo los vectores solo con modelos locales y su comprobación de modelo compara con el identificador simple. Los adaptadores por API aún no reciben un registrador de uso en el cableado.

## Cómo reproducirlo

```
python -m app.evaluation.api_models_evaluation            # llama a lo que no está en caché y juzga los huecos (necesita OPENROUTER_API_KEY)
python -m app.evaluation.api_models_evaluation --offline  # solo cachés; falla si falta algo
python -m app.evaluation.api_models_evaluation --render   # regenera el informe desde el JSON guardado
python -m app.evaluation.llm_model_selection report       # regenera el informe de V10 con la nueva regla, sin llamar a ningún modelo
```

La variante sin conexión recalcula todas las tablas desde `data/evaluation/api_models/responses.jsonl` y carga los modelos locales para reconstruir los candidatos. Para usar los modelos por API en el sistema: `EMBEDDING_PROVIDER=openrouter` y `RERANKER_PROVIDER=openrouter` (ver `.env.example`). Tests: `pytest tests/evaluation/test_api_models_evaluation.py tests/infrastructure/test_openrouter_embedding_encoder.py tests/infrastructure/test_api_reranker.py tests/infrastructure/test_reranker_settings.py tests/infrastructure/test_embedding_model_selection.py`.

## Ficheros

- `app/infrastructure/search/text_encoding.py`, `app/infrastructure/search/openrouter_embedding_encoder.py`, `app/infrastructure/search/api_reranker.py`, `app/infrastructure/search/openrouter_http.py`, `app/infrastructure/search/embeddings_store_faiss.py`: codificador separado del almacén, adaptadores por API e identidad del índice.
- `app/infrastructure/config/settings.py`, `app/infrastructure/llm/usage.py`, `.env.example`: ajustes de proveedor y registro de uso de `embedding` y `rerank`.
- `app/evaluation/api_models_evaluation.py`, `app/evaluation/llm_model_selection.py`: medición y regla de selección.
- `data/evaluation/API_MODELS.md`, `data/evaluation/results_api_models.json`, `data/evaluation/LLM_MODEL_SELECTION.md`, `data/evaluation/api_models/` (respuestas guardadas, tiempos locales y caché de juicios).
- `app/evaluation/relevance_judgments_extended_api_models.json`: juicios ampliados con las 290 notas añadidas.
- `tests/evaluation/test_api_models_evaluation.py`, `tests/infrastructure/test_openrouter_embedding_encoder.py`, `tests/infrastructure/test_api_reranker.py`, `tests/infrastructure/test_reranker_settings.py`, `tests/infrastructure/test_embedding_model_selection.py`.
