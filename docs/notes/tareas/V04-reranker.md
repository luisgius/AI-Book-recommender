# V04 · Puerto `Reranker` y adaptador cross-encoder
- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** dominio (`app/domain/`), infraestructura (`app/infrastructure/search/`, `app/infrastructure/config/`), aplicación (`app/api/v1/`)

## Qué se ha hecho

- Nuevo puerto `Reranker` en `app/domain/ports.py`, con los métodos `rerank`, `get_model_name` e `is_ready`.
- `SearchResult` incorpora el campo opcional `rerank_score`. `final_score` conserva la puntuación de fusión RRF, de modo que ambas son visibles. El orden de la lista lo da `rank`.
- `SearchService` acepta los parámetros opcionales `reranker` y `rerank_depth` (por defecto 30). La canalización con reordenador es: recuperar, fusionar, aplicar filtros, reordenar los primeros `rerank_depth` resultados, MMR opcional y recorte a `max_results`.
  - El texto que se compara con los libros es siempre la consulta original del usuario, nunca una reformulación del LLM ni una variante de la expansión multi-consulta.
  - Los resultados situados más allá de la profundidad mantienen el orden de la fusión, detrás de la cabeza reordenada.
  - La recuperación pide al menos `rerank_depth` candidatos aunque el llamante pida 10.
  - Se aplica en `search`, `search_with_fallback`, `search_with_understanding` y en la búsqueda multi-consulta. No se aplica en `find_similar_books`, que no tiene consulta.
- Tras reordenar, MMR usa `rerank_score` como término de relevancia y solo diversifica la cabeza reordenada. Si ordenara de nuevo por la puntuación RRF, deshacería el trabajo del reordenador.
- Si el reordenador lanza una excepción o devuelve un conjunto de resultados distinto del recibido, se registra un aviso y se conserva el orden de la fusión. La búsqueda no falla por culpa de la tercera etapa.
- Adaptador `CrossEncoderReranker` (`app/infrastructure/search/cross_encoder_reranker.py`) sobre `sentence_transformers.CrossEncoder` (versión 3.3.1 ya instalada, sin actualizar):
  - Carga el modelo en el primer uso. Un fallo de carga se recuerda: `is_ready()` pasa a falso y las llamadas siguientes fallan de inmediato.
  - Como documento usa `Book.get_searchable_text()`, el mismo texto que indexa BM25.
  - Pide siempre la salida cruda del modelo, sin sigmoide, para que las puntuaciones de modelos distintos sean comparables (`bge` aplica una sigmoide por defecto y `mmarco` no).
- Configuración en `RerankerSettings`: `RERANKER_MODEL` (vacío, ausente o `none` desactiva la etapa; es el valor por defecto), `RERANK_DEPTH` (30) y `RERANKER_DEVICE` (opcional).
- Cambios aditivos en la API: cada resultado lleva `rerank_score` (nulo si la etapa está desactivada) y `/health` incluye un bloque `reranker` que no cuenta para el estado global.

## Por qué

- Hasta ahora la canalización terminaba en la fusión RRF, con MMR opcional. La canalización habitual en 2026 tiene una tercera etapa: un cross-encoder que lee la consulta y el documento juntos y reordena los mejores candidatos.
- Es la pieza que falta para medir, en V06, si esa etapa aporta algo sobre este catálogo.
- Problema encontrado durante el trabajo: transformers 4.57.3 contacta con el Hugging Face Hub al cargar un tokenizador de más de 100 000 tokens, incluso con `HF_HUB_OFFLINE=1`. Los dos modelos probados caen en ese caso. Sin conexión, el adaptador carga el modelo desde su carpeta de la caché local, lo que evita esa llamada.

## Objetivo

Disponer de una tercera etapa de reordenación opcional, desactivada por defecto, con una interfaz propia en el dominio y sin alterar el comportamiento cuando no está configurada. No se pretende demostrar en esta tarea que reordenar mejore la calidad de la búsqueda.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Activar un reordenador por defecto ahora | No hay medición sobre este catálogo. Decide V06. |
| Escribir la puntuación del cross-encoder en `final_score` | Los resultados situados más allá de la profundidad conservan puntuaciones de fusión: una misma lista mezclaría dos escalas y el campo cambiaría de significado según la configuración. |
| Reordenar con la consulta reformulada por el LLM | La reformulación puede alejarse de lo que pidió el usuario. El texto original es la necesidad de información. |
| Un LLM como reordenador por listas (estilo RankGPT) | Una llamada al LLM por búsqueda añade segundos y tokens, y hace que la etapa dependa de que el LLM esté disponible. Posible más adelante como brazo de ablación. |
| `Qwen/Qwen3-Reranker-0.6B` | Requiere sentence-transformers 5.4 o posterior, lo que obliga a pasar a transformers 5. Aplazado. |
| Reordenadores de Jina | Licencias no comerciales. |
| Modelos de interacción tardía (ColBERT) | Más componentes, y los multilingües tienen licencias restrictivas o entradas cortas. Un cross-encoder da el mismo tipo de beneficio con un solo modelo. |
| APIs de reordenación alojadas | Coste por llamada y sin reproducibilidad sin conexión. |
| Reordenar todo el catálogo | Es viable con 177 libros, pero no representa una canalización real y el coste crece linealmente con el catálogo. La profundidad es un parámetro para poder variarla en la ablación. |

## Cómo se ha validado

- **Sin regresión con la etapa desactivada.** Con `reranker=None` (valor por defecto) todas las ramas nuevas se omiten. Los cuatro sistemas de referencia reproducen exactamente `data/evaluation/COMPARISON_BASELINE.md` tras el cambio.
- **Tests nuevos:** 65, en `tests/domain/test_reranking.py` (25), `tests/infrastructure/test_cross_encoder_reranker.py` (14), `tests/infrastructure/test_reranker_settings.py` (21) y `tests/api/test_rerank_api.py` (5). Usan un reordenador simulado y un `CrossEncoder` sustituido, por lo que no se descarga ningún modelo. La batería completa tras esta tanda: 885 pasan, 0 fallan.
- **Prueba de humo con modelos reales** sobre el catálogo de 177 libros, cinco consultas y `max_results=10`. Latencia de la etapa de reordenación con profundidad 30, mediana de 25 llamadas, medida mientras un servidor de LLM local usaba la GPU (solo orientativa):

| Modelo | Parámetros | `mps` | `cpu` |
|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 568M | 3137 ms | 10378 ms |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 118M | 589 ms | 1385 ms |

`mps` y `cpu` dieron las mismas puntuaciones con tres decimales.

- **Observaciones cualitativas** de la prueba de humo (no son una medida de calidad):
  - "novelas de misterio ambientadas en la época victoriana": `bge` cambió 9 de las 10 posiciones y metió 3 libros nuevos en el top 10.
  - "machine learning for beginners": cambió 5 posiciones y 1 libro.
  - "history of the roman empire", consulta sin ningún libro relevante en el catálogo: todas las puntuaciones de `bge` fueron negativas (alrededor de -8 o menos) y el orden era ruido ("Learn Python" quedó el primero). La puntuación cruda podría servir más adelante como umbral de "nada relevante".
  - Parte de cada cambio procede del mayor conjunto de candidatos (60 por recuperador en lugar de 20) y no solo del reordenador.

## Cómo usarlo

```bash
# Activar la etapa (se descarga y carga el modelo en la primera búsqueda)
RERANKER_MODEL=BAAI/bge-reranker-v2-m3 RERANK_DEPTH=30 RERANKER_DEVICE=mps
```

Sin `RERANKER_MODEL`, o con el valor `none`, la etapa está desactivada. Las variables se documentan en `.env.example`. Modelos probados: `BAAI/bge-reranker-v2-m3` y `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

## Límites y trabajo pendiente

- Si reordenar mejora nDCG@10 en las 44 consultas de prueba, y qué modelo y profundidad usar, se mide en V06. Esta tarea no aporta ninguna cifra de calidad.
- El código de evaluación todavía no recibe un reordenador.
- La interfaz web no muestra `rerank_score`.
- `max_length` (512) y `batch_size` (32) no se han ajustado.
- No se ha arrancado el servidor de la API con un reordenador real; el cableado está cubierto por tests con sustitutos.
- Las latencias se midieron con la GPU compartida y son solo indicativas.

## Ficheros

- `app/domain/ports.py`: puerto `Reranker`.
- `app/domain/entities.py`: campo `SearchResult.rerank_score`.
- `app/domain/services.py`: `reranker` y `rerank_depth` en `SearchService`, etapa de reordenación y adaptación de MMR.
- `app/infrastructure/search/cross_encoder_reranker.py`: adaptador `CrossEncoderReranker`.
- `app/infrastructure/config/settings.py`: `RerankerSettings`.
- `app/api/v1/dependencies.py`, `app/api/v1/converters.py`, `app/api/v1/schemas.py`, `app/api/v1/search_endpoints.py`: construcción del reordenador, `rerank_score` en la respuesta y bloque `reranker` en `/health`.
- `.env.example`: variables documentadas.
- `tests/domain/test_reranking.py`, `tests/infrastructure/test_cross_encoder_reranker.py`, `tests/infrastructure/test_reranker_settings.py`, `tests/api/test_rerank_api.py`: tests nuevos.
