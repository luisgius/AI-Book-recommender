# Propuesta: dónde debe vivir el sistema y qué aporta MCP

**Fecha:** 2026-11-11
**Estado:** decidida por el autor el 2026-11-11 (ver sección 6)
**Origen:** el autor plantea que el sistema viva en la nube y no en su
portátil (por ejemplo Vercel + Supabase), y pide investigar si conviene usar
MCP o dar al LLM acceso directo a otras bases de datos.

**Método.** Tres investigaciones en paralelo el 2026-11-11 sobre fuentes
primarias (documentación oficial, páginas de precios, especificación de MCP,
catálogos de modelos, llamadas reales a las API públicas). Las cifras marcadas
"est." son aritmética propia, no cifras del proveedor. Lo que no se pudo
verificar está en la sección 7.

---

## 1. Respuesta corta

1. **Sí conviene que viva en la nube**, y el motivo principal no es el tamaño
   sino el tribunal: abrir una URL es más fácil que instalar nada.
2. **"Vercel + Supabase" tal cual no encaja.** Vercel ya ejecuta FastAPI, pero
   una función tiene 2 GB de memoria y los dos modelos del sistema ocupan
   1,1 GB y 2,3 GB. El plan gratuito de Supabase tiene 500 MB y se pausa a los
   7 días sin actividad; 200.000 obras necesitan unos 2,2 GB. La idea funciona
   si se cambian dos piezas: los modelos pasan a usarse por API y la base de
   datos se elige por precio.
3. **MCP sí, pero en un sentido.** Exponer el buscador como servidor MCP es
   barato y actual: cualquier asistente podría consultar el catálogo. Que el
   LLM propio busque en fuentes externas a través de MCP en lugar de tener un
   índice propio no conviene: es más lento, ordena peor y no se puede evaluar
   de forma reproducible.

---

## 2. Qué limita cada pieza

| Pieza | Hecho verificado | Consecuencia |
|---|---|---|
| Vercel | Python con FastAPI es GA desde el 2026-10-15; plan Hobby: 2 GB de memoria, 300 s, uso personal no comercial; no tiene base de datos propia desde diciembre de 2024 | Sirve para la interfaz y para un backend ligero que no cargue modelos |
| Supabase | Gratuito: 500 MB, pausa a los 7 días de poca actividad. Pro: 25 USD/mes con 8 GB; no ofrece BM25 real (solo `tsvector`, PGroonga); su inferencia integrada solo tiene `gte-small`, en inglés | Válido, pero cuesta 25-30 USD/mes con 200.000 obras |
| Neon | El mismo Postgres con pgvector; sin cuota mínima; se suspende a los 5 minutos sin uso y reactiva en cientos de milisegundos | 2-6 USD/mes est. con 200.000 obras; 8-15 USD/mes est. con un millón |
| Postgres como motor léxico | `ts_rank` no usa información global: no hay IDF ni saturación, y la consulta se interpreta como AND | No es BM25. Los resultados de la ablación V06 no se trasladan: hay que volver a medir |
| pgvector | `halfvec` (16 bits) indexa hasta 4.000 dimensiones con HNSW; índice aproximado | Para evaluar hay que usar búsqueda exacta y medir aparte el recall del índice |
| Modelos por API | `multilingual-e5-large-instruct` solo lo sirve un proveedor barato; `bge-reranker-v2-m3` casi nadie. En OpenRouter sí están `baai/bge-m3`, `qwen/qwen3-embedding-8b` (0,01 USD por millón de tokens) y `qwen/qwen3-reranker-0.6b` | Con la clave de OpenRouter que ya existe se cubren LLM, embeddings y reranker, todos modelos chinos; pero no son los modelos medidos en V06 |
| Coste de los embeddings por API | 220 tokens por obra a 0,01 USD por millón | 0,44 USD para 200.000 obras y 2,20 USD para un millón, una sola vez |
| Construcción del catálogo | En el portátil: 3,6 h de GPU para un millón de obras | Con embeddings por API el portátil solo descarga y filtra; los vectores y el índice viven en la base de datos remota |

---

## 3. Arquitectura propuesta: dos perfiles sobre los mismos puertos

La arquitectura hexagonal ya separa el dominio de los adaptadores. Se propone
aprovecharlo con dos juegos de adaptadores, no con una reescritura.

| Puerto | Perfil local (el evaluado) | Perfil nube (el desplegado) |
|---|---|---|
| `BookCatalogRepository` | SQLite | Postgres |
| `LexicalSearchRepository` | BM25 con tokenizador normalizado | Búsqueda de texto de Postgres, una configuración por idioma con plegado de acentos |
| `VectorSearchRepository` | FAISS plano, exacto | pgvector con `halfvec` y HNSW |
| `EmbeddingsStore` | Modelo abierto ejecutado en local | El mismo tipo de modelo por API |
| `Reranker` | Cross-encoder en local | Reranker por API |
| `LLMClient` | OpenRouter (o ninguno) | OpenRouter |
| Entrada | API REST y UI | API REST, UI y servidor MCP |

- **Perfil local:** catálogo pequeño versionado en el repositorio, modelos
  abiertos con versión fijada, resultados exactos. Es el que respalda las
  cifras de la memoria y el que cualquiera puede reproducir sin cuenta.
- **Perfil nube:** catálogo grande (200.000 obras ahora, hasta un millón con
  Open Library), sin modelos en el proceso, de modo que el backend cabe en
  una función de Vercel o en un contenedor mínimo.
- **Medida de paridad:** las mismas consultas por los dos perfiles, con el
  solapamiento de resultados y la diferencia de nDCG@10. Es un resultado
  propio para la memoria: qué se pierde y qué se gana al desplegar.

Coste estimado del perfil nube: 2-6 USD/mes de base de datos con 200.000
obras, 0,44 USD una vez por los embeddings, y por búsqueda unos 0,001 USD del
LLM más 0,0016 USD del reranker. Todo se apaga borrando el proyecto.

### Alternativa más barata y más conservadora

Un solo contenedor que se apaga cuando no se usa (Google Cloud Run, capa
gratuita) con FastAPI, SQLite y FAISS dentro, y los modelos por API. Cuesta
0 USD/mes y conserva exactamente el código evaluado. No escala bien a un
millón de obras (imagen de varios GB, arranque en frío largo) y exige una
cuenta de Google Cloud con tarjeta.

---

## 4. MCP y acceso a otras fuentes

**Estado de MCP.** La revisión vigente es la del 2026-07-28, que hace el
protocolo sin estado. Lo gobierna la Agentic AI Foundation (Linux Foundation)
desde diciembre de 2025. Claude (incluido el plan gratuito, un conector) y
ChatGPT permiten añadir un servidor MCP remoto sin autenticación.

| Diseño | Qué aporta | Coste | Veredicto |
|---|---|---|---|
| (a) Solo índice propio | Lo actual | Ninguno | Base del TFG |
| (b) Índice propio + servidor MCP de solo lectura (`search_books`, `get_book`, `similar_books`) | El catálogo se puede consultar desde Claude, ChatGPT y otros | Un módulo montado en la app FastAPI y una dependencia; sin coste de LLM en el servidor | **Recomendado** |
| (c) Índice propio y, si el reranker indica que no hay nada relevante, consulta en vivo a una fuente externa | Respuesta para libros que no están en el catálogo | Umbral que calibrar, uno o dos adaptadores HTTP, caché; añade 0,2-2,3 s solo cuando se activa | Experimento opcional, después del catálogo grande |
| (d) Agente con herramientas MCP como vía principal | Respuestas libres sobre varias fuentes | Al menos dos turnos de LLM (unos 15 s con el modelo actual), 3,3 veces más tokens de entrada | No |

Motivos para descartar (d):
- Las API de libros en vivo son búsquedas por palabra clave sobre metadatos;
  la mayoría no devuelve descripción en los resultados. No ofrecen búsqueda
  semántica entre idiomas, que es lo que hace el índice propio.
- En la literatura reciente, el refinamiento de la lista por el propio agente
  queda por debajo de un reranker explícito (nDCG@10 43,9 frente a 48,0;
  arXiv 2601.07711).
- Con fuentes en vivo la evaluación no es reproducible (arXiv 2508.06600).
- El texto de fuentes editables entra en el prompt: riesgo de inyección.

Otras vías consideradas:
- **Servidor MCP de Supabase o de Postgres para servir consultas:** el de
  Supabase es una herramienta de desarrollo ("no debe darse a clientes ni a
  usuarios finales"); los de referencia de Postgres y SQLite están archivados.
- **Texto a SQL libre sobre el catálogo:** la comprensión de consulta ya
  convierte "antes de 1950" en filtros tipados, que es la parte segura. No se
  añade SQL libre.
- **Google Books en vivo:** sin clave responde HTTP 429; se descarta.

La regla "el LLM solo recomienda libros que existen en una fuente" se mantiene
en (a), (b) y (c). En (b) el servidor solo devuelve libros del catálogo con su
identificador; lo que el asistente externo diga después queda fuera del
control del sistema y así debe constar en la memoria.

---

## 5. Efecto sobre las tareas

| Tarea | Antes | Con esta propuesta |
|---|---|---|
| V11 catálogo de varias fuentes | En curso | Sin cambios: el pipeline produce los mismos registros para SQLite o para Postgres |
| V12 búsqueda a escala | BM25 escalable y carga bajo demanda en local | Adaptadores de Postgres (catálogo, texto, pgvector); el perfil local se queda en el catálogo pequeño |
| Nueva | — | Adaptadores de embeddings y de reranker por API, y ablación de esos modelos frente a los medidos en V06 |
| Nueva | — | Servidor MCP de solo lectura |
| V13 valores por defecto | Reranker y embeddings multilingües en local | Igual en local; en nube, los modelos por API que gane la ablación |
| V14 evaluación del catálogo nuevo | Sobre el índice local | Sobre búsqueda exacta, más la medida de paridad entre perfiles |
| V09 ejecución en un comando | README, Makefile, demo | Además la URL pública y el despliegue documentado |

---

## 6. Decisiones que debe tomar el autor

1. **Arquitectura:** dos perfiles (recomendado), contenedor único, o seguir
   solo en local.
2. **Base de datos:** Neon (recomendado por coste) o Supabase Pro (siempre
   activa, 25-30 USD/mes). El adaptador es SQL estándar: cambiar después es
   cambiar una cadena de conexión.
3. **Backend:** Vercel (encaja al no cargar modelos) o Cloud Run.
4. **MCP:** servidor de solo lectura ahora; la consulta en vivo como
   experimento posterior.

**Decisiones tomadas (2026-11-11):**

1. Dos perfiles, con la condición de medir y probar cada pieza.
2. Neon: más adecuado para un TFG y más barato. El proyecto existe en la
   región de Fráncfort, en el plan gratuito.
3. Sobre los modelos: deben equilibrar velocidad y calidad. La velocidad
   entra en la regla de selección del LLM, del modelo de embeddings y del
   reranker (tarea V16).
4. MCP: servidor de solo lectura (tarea V15).

Queda por decidir el alojamiento del backend (Vercel o Cloud Run) y, cuando
el catálogo supere lo que admite el plan gratuito de Neon, el paso al plan de
pago por uso.

---

## 7. No verificado en fuente primaria

- Latencias de los modelos por API y arranques en frío: son estimaciones.
- Tamaños del índice HNSW a 1.024 dimensiones: interpolados de medidas a 960
  y 1.536.
- Si la versión de pgvector desplegada en cada proveedor coincide con la
  documentada.
- Disponibilidad de servidores MCP propios en los planes de ChatGPT (la página
  de ayuda devolvió 403).
- Calidad de `qwen/qwen3-embedding-8b`, `baai/bge-m3` y
  `qwen/qwen3-reranker-0.6b` sobre este catálogo: no medida todavía.
- Los precios del reranker en OpenRouter figuran solo en las páginas de cada
  modelo; el JSON público devuelve 0.

## 8. Fuentes (consultadas el 2026-11-11)

- Vercel: https://vercel.com/docs/functions/runtimes/python ,
  https://vercel.com/docs/functions/limitations ,
  https://vercel.com/docs/plans/hobby , https://vercel.com/docs/storage
- Supabase: https://supabase.com/pricing ,
  https://supabase.com/docs/guides/platform/free-project-pausing ,
  https://supabase.com/docs/guides/ai/hybrid-search ,
  https://supabase.com/docs/guides/functions/ai-models ,
  https://supabase.com/docs/guides/ai-tools/mcp.md
- Neon: https://neon.com/pricing , https://neon.com/docs/introduction/scale-to-zero
- pgvector: https://github.com/pgvector/pgvector
- PostgreSQL, búsqueda de texto: https://www.postgresql.org/docs/current/textsearch-controls.html
- Google Cloud Run: https://cloud.google.com/run/pricing
- OpenRouter: https://openrouter.ai/api/v1/embeddings/models ,
  https://openrouter.ai/api/v1/models?output_modalities=rerank
- DeepInfra: https://deepinfra.com/models/embeddings , https://deepinfra.com/models/reranker
- MCP: https://modelcontextprotocol.io/specification/latest ,
  https://modelcontextprotocol.io/specification/2026-07-28/changelog.md ,
  https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation ,
  https://support.claude.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp
- FastMCP: https://gofastmcp.com/integrations/fastapi.md
- Agentic RAG frente a pipeline fijo: https://arxiv.org/abs/2601.07711
- BrowseComp-Plus (reproducibilidad): https://arxiv.org/abs/2508.06600
- Corrective RAG: https://arxiv.org/abs/2401.15884
- Open Library, condiciones de la API: https://openlibrary.org/developers/api
