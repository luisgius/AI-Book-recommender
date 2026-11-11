# V02 · Embeddings configurables, índice normalizado y control modelo-índice
- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** dominio (puerto `EmbeddingsStore`), infraestructura (`app/infrastructure/search/`, `app/infrastructure/config/`), aplicación (`app/api/v1/dependencies.py`), scripts

## Qué se ha hecho

- El modelo de embeddings deja de estar fijado en el código. Se elige con la variable `EMBEDDING_MODEL` (por defecto `all-MiniLM-L6-v2`) y el dispositivo con `EMBEDDING_DEVICE` (`cpu`, `mps` o `cuda`; automático si no se indica).
- El puerto `EmbeddingsStore` incorpora el método `generate_query_embedding`. `SearchService` y la evaluación lo usan para las consultas; `generate_embedding` y `generate_embeddings_batch` quedan para los documentos.
- El adaptador `EmbeddingsStoreFaiss` aplica una tabla explícita de convenciones por modelo (`PROMPT_CONVENTIONS`), contrastada con las fichas de los modelos en Hugging Face:
  - `intfloat/multilingual-e5-large-instruct`: la consulta lleva `Instruct: ...\nQuery: ` y los documentos van sin prefijo.
  - `intfloat/multilingual-e5-small`, `-base` y `-large`: `query: ` para consultas y `passage: ` para documentos.
  - `Qwen/Qwen3-Embedding-0.6B`: prompt almacenado en el modelo con nombre `query` para consultas; documentos sin prefijo.
  - `all-MiniLM-L6-v2` y cualquier modelo desconocido: sin prompt.
- Los índices nuevos guardan vectores normalizados con norma L2 en un `faiss.IndexFlatIP`, de modo que el producto interno es la similitud del coseno. Los índices antiguos con `IndexFlatL2` siguen funcionando: `FaissVectorSearchRepository` lee la métrica del índice y devuelve en ambos casos una puntuación en la que mayor es mejor.
- Junto al índice se escribe el fichero `faiss_metadata.json` con modelo, dimensión, métrica, indicador de normalización, número de vectores y fecha de creación. Cargar un índice cuyo modelo difiere del configurado lanza un error (`IndexModelMismatchError`) que pide reconstruirlo. Un índice sin metadatos se interpreta como `all-MiniLM-L6-v2` con L2, que es el índice incluido en el repositorio.
- `scripts/ingest_and_build.py` acepta `--embedding-model`. Si no se indica `--out-dir` y el modelo no es el de por defecto, el índice se escribe en `data/indexes_ablation/<slug>/`.
- Se corrige una caída del proceso en macOS (ver "Por qué"): `faiss_runtime.py` importa `torch` antes que `faiss` y los dos adaptadores FAISS importan `faiss` a través de ese módulo.

## Por qué

- Antes, `all-MiniLM-L6-v2` (solo inglés) estaba fijado en el código y el índice era `IndexFlatL2`. 61 de los 177 libros del catálogo están en español.
- Nada impedía cargar un índice construido con un modelo y codificar las consultas con otro. El fallo no produce ningún error: las puntuaciones simplemente dejan de tener sentido.
- Los modelos asimétricos codifican de forma distinta consultas y documentos. Con una sola operación de codificación, usar uno de ellos mal degrada la recuperación sin avisar.
- Caída encontrada durante el trabajo: en macOS, importar `faiss` antes que `torch` provocaba un fallo de segmentación al cargar `multilingual-e5-large-instruct`, porque `faiss-cpu` y `torch` incluyen cada uno su propia biblioteca OpenMP.

## Objetivo

Poder cambiar de modelo de embeddings por configuración, con la codificación correcta de consultas y documentos, y con garantía de que el modelo de las consultas coincide con el del índice. Esto es la condición previa para la ablación de V06, que decidirá el modelo por defecto. No se pretende mejorar la calidad de la búsqueda en esta tarea.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Cambiar directamente el modelo por defecto a uno multilingüe | No hay medición sobre este catálogo. Las cifras publicadas de benchmarks no sustituyen una ablación con las consultas del propio sistema (V06). |
| Actualizar a sentence-transformers 6.x | No hace falta para los modelos candidatos y obliga a pasar a transformers 5, una migración mayor sin beneficio para esta tarea. |
| Mantener `IndexFlatL2` | Con vectores normalizados ordena igual, pero el producto interno da directamente el coseno, comparable entre modelos y utilizable más adelante en una fusión por puntuación. |
| Índices aproximados (HNSW, IVF) | Con 177 libros un índice plano es exacto y no tiene parámetros que justificar. |
| Base de datos vectorial (Qdrant, LanceDB, pgvector) | Sin beneficio medible a esta escala y con un servicio más que ejecutar. |
| Un único método con un indicador `is_query` | Un método separado en el puerto hace visible la distinción entre consulta y documento y dificulta el uso incorrecto. |
| Leer los prompts de la configuración de cada modelo | Una tabla explícita en el adaptador es más fácil de explicar y de auditar. |
| APIs de embeddings alojadas | Coste por llamada, sin reproducibilidad sin conexión, y el proyecto prefiere componentes locales. |
| Modelos de embeddings de Jina | Licencias no comerciales. |

## Cómo se ha validado

- **Sin regresión con el modelo por defecto.** Para las 44 consultas de evaluación, los 100 primeros identificadores y sus puntuaciones son idénticos antes y después del cambio. El `md5` de los ficheros del índice incluido no ha cambiado: el índice no se ha reconstruido.
- **Tests nuevos:** 34, en `tests/infrastructure/test_embedding_model_selection.py` (21) y `tests/infrastructure/test_embedding_settings.py` (13). Usan un modelo simulado, por lo que no descargan nada. La batería completa tras esta tanda: 771 pasan, 0 fallan.
- **Compatibilidad de librerías:** no ha hecho falta actualizar nada; sentence-transformers 3.3.1 y transformers 4.57.3 cargan los dos modelos nuevos.
- **Prueba de humo con modelos reales** sobre el catálogo de 177 libros, dispositivo `mps` (Apple M5 Pro). "Construcción" es codificar los 177 libros y construir el índice.

| Modelo | Dimensión | Construcción | Latencia de consulta (mediana) |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 0,40 s | 3,9 ms |
| `intfloat/multilingual-e5-large-instruct` | 1024 | 7,30 s | 11,8 ms |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | 50,11 s | 22,1 ms |

Los tiempos se midieron con la máquina bajo presión de memoria, por lo que son solo orientativos. No se ha medido calidad de recuperación de los modelos nuevos: eso corresponde a V06.

## Cómo usarlo

```bash
# Reconstruir con otro modelo (escribe en data/indexes_ablation/<slug>/)
python -m scripts.ingest_and_build --rebuild --embedding-model intfloat/multilingual-e5-large-instruct

# Servir la API con ese modelo (el índice debe haberse construido con el mismo)
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct EMBEDDING_DEVICE=mps
```

Las variables se documentan en `.env.example`. Si el modelo configurado no coincide con el del índice cargado, se lanza el error de reconstrucción descrito arriba.

## Límites y trabajo pendiente

- `evaluation_job.py` y `llm_path_evaluation.py` siguen construyendo el almacén con el modelo por defecto. Se conectarán a la configuración en V06.
- El error de modelo distinto aparece en la primera petición; no hay comprobación al arrancar la API.
- El ADR `docs/arch_decisions/003-faiss-vector-index.md` sigue describiendo `IndexFlatL2` y debe actualizarse.
- Qué modelo pasa a ser el de por defecto lo decide la ablación de V06, no esta tarea.
- La tabla de prompts solo cubre los modelos listados; otro modelo asimétrico se tratará como simétrico hasta que se añada a la tabla.

## Ficheros

- `app/domain/ports.py`: método `generate_query_embedding` en `EmbeddingsStore`.
- `app/infrastructure/search/embeddings_store_faiss.py`: tabla de prompts, `IndexFlatIP`, metadatos y control modelo-índice.
- `app/infrastructure/search/faiss_vector_search_repository.py`: lectura de la métrica y puntuación común.
- `app/infrastructure/search/faiss_runtime.py`: importación de `torch` antes que `faiss`.
- `app/infrastructure/config/settings.py`: `EMBEDDING_MODEL` y `EMBEDDING_DEVICE`.
- `app/api/v1/dependencies.py`: construcción del almacén con la configuración.
- `scripts/ingest_and_build.py`: opción `--embedding-model` y directorio de ablación.
- `.env.example`: variables documentadas.
- `tests/infrastructure/test_embedding_model_selection.py`, `tests/infrastructure/test_embedding_settings.py`: tests nuevos.
