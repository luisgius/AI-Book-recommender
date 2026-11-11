# Validación del proyecto en noviembre de 2026

**Fecha:** 2026-11-11
**Pregunta:** el TFG se diseñó en 2024. Entre 2024 y 2026 han cambiado las
tecnologías, las prácticas y los costes. ¿Sigue siendo válido el proyecto tal
como está planteado, y qué hay que cambiar sin perder el foco?

**Método.** Se cruzan tres fuentes: (1) el estado real del repositorio a fecha
de hoy (663 tests en verde, catálogo de 177 libros: 116 en inglés y 61 en
español, 55 sin descripción; 44 consultas y 637 juicios de relevancia);
(2) las mediciones propias en `data/evaluation/`; (3) fuentes externas
consultadas el 2026-11-03 y el 2026-11-11. Cada afirmación externa indica su
nivel de verificación: **[P]** leída en fuente primaria, **[S]** fuente
secundaria (blog o resumen), no verificada en origen.

---

## 1. Veredicto

**El núcleo del proyecto sigue siendo válido en 2026 y no hay que pivotar.**
La búsqueda híbrida (léxica + densa, fusionada por rangos) con un LLM que
interpreta la consulta y explica los resultados apoyándose en lo recuperado
es hoy el patrón de referencia, no una idea superada.

**Lo que ha envejecido son cuatro piezas concretas y la vara de medir**, no
la arquitectura:

1. El modelo de embeddings (`all-MiniLM-L6-v2`) es solo inglés y el 34 % del
   catálogo está en español.
2. El tokenizador de BM25 (`text.lower().split()`) no pliega acentos ni
   elimina puntuación ni palabras vacías.
3. Falta la tercera etapa que en 2026 se da por supuesta: un reordenador
   (cross-encoder) tras la fusión.
4. La evaluación no permite afirmar lo que el título promete: con 44
   consultas el híbrido no se distingue del vectorial (p = 0,65) y solo el
   55 % de los diez primeros resultados tiene juicio de relevancia.

Además, 2026 obliga a responder una pregunta que en 2024 no existía: **si un
LLM con contexto largo puede leer el catálogo entero, ¿para qué recuperar?**
El TFG debe contestarla con una medida propia, no con una opinión.

---

## 2. Qué ha cambiado entre 2024 y 2026

| Área | Supuesto de 2024 | Situación en 2026 | Evidencia | Efecto en el TFG |
|---|---|---|---|---|
| Recuperación híbrida | BM25 + vectores + RRF como propuesta del proyecto | Es el patrón por defecto en producción; se añade una etapa de reordenación | Guías de 2026 lo describen como configuración base, con RRF k = 60 [S]; Bruch et al. (TOIS 2023) matizan que la fusión convexa puede superar a RRF [P] | Se mantiene. RRF se defiende como línea base sin ajuste y se cita el matiz |
| Reordenación | No prevista | Cross-encoder multilingüe sobre el top 50-100 es práctica habitual; existen modelos libres (`BAAI/bge-reranker-v2-m3`, `Qwen/Qwen3-Reranker-0.6B`) | Tarjetas de modelo y paper de Qwen3 [P] | **Añadir** puerto `Reranker` y medirlo |
| Embeddings | `all-MiniLM-L6-v2` (2021, inglés) | Modelos multilingües libres muy superiores en español: 0,312 frente a 0,715 nDCG@10 en SpanishPassageRetrievalS2S (`multilingual-e5-large-instruct`) | Resultados oficiales de MTEB [P] | **Cambiar** tras ablación propia |
| Léxico | Tokenizador trivial | El valor está en la normalización (acentos, palabras vacías), no en la velocidad | Documentación de analizadores de Elasticsearch y Snowball [P] | **Corregir** y medir con consultas sin tildes |
| RAG frente a contexto largo | RAG era la única vía práctica | Los modelos de contexto largo compiten en precisión con corpus pequeños y estáticos, a un coste por consulta muy superior; el consenso es híbrido: recuperar un conjunto acotado y razonar sobre él | Comparativas de 2026 [S] | **Añadir** una línea base "catálogo completo en contexto" y comparar calidad, latencia y tokens |
| Coste de los LLM | API de pago cara; modelo local poco capaz | Caída de precios del orden de 600 veces en seis años; vida media del precio de 1,1 años en la gama económica. Los modelos abiertos de ~27B ejecutan en un portátil | Du, "Tiered Super-Moore's Law", arXiv 2603.28576 [P]; medidas propias con Qwen3.8-27B en Ollama | El LLM local pasa a ser la opción por defecto (ya hecho). **Medir** latencia y tokens por consulta |
| Modelo por defecto en API | `gpt-4o-mini` | OpenAI retira el snapshot `gpt-4o-mini-2024-07-18` el 2027-01-14 | Página de deprecaciones de OpenAI [P] | **Cambiar** el valor por defecto del proveedor `openai` |
| Orquestación | LangChain 0.x | LangChain 1.x y LangGraph 1.x son las ramas activas; la 0.3 queda en mantenimiento hasta diciembre de 2026 | Política de versiones de LangChain [P] | Ya migrado (1.4.8 / 1.2.19) |
| Agentes | Opcional, experimental | La recuperación como herramienta de un agente es práctica corriente | Documentación de LangGraph [P] | Ya implementado como grafo explícito (`POST /agent/ask`) |
| LLM como recomendador | Idea emergente | Los LLM que recomiendan sin catálogo inventan ítems; la literatura de 2026 trata el anclaje al catálogo como requisito | Liao et al., arXiv 2505.03336 [P]; revisión "Echoes in the Loop", arXiv 2602.07442 [S] | **Refuerza** el diseño: el LLM nunca propone libros, solo interpreta y explica lo recuperado |
| Evaluación | nDCG y recall sobre juicios propios | Se espera procedencia de los juicios, cobertura (`Judged@k`), intervalos de confianza, test de aleatorización con corrección múltiple y juez LLM validado | UMBRELA, arXiv 2406.06519 [P]; Smucker 2007, Urbano 2019 [P] | **Completar** la higiene de evaluación antes de comparar sistemas |

Las fuentes primarias con fecha de consulta 2026-11-03 están detalladas en
`review/06-research-improvements-2026-11.md` (nota interna, no versionada).
Las del 2026-11-11 figuran en la sección 6.

---

## 3. Qué se mantiene y por qué

- **Arquitectura hexagonal.** Es lo que ha permitido cambiar OpenAI por Ollama
  y LangChain 0.3 por 1.x sin tocar el dominio. Cada cambio propuesto abajo
  entra por un puerto o un adaptador.
- **SQLite + FAISS plano + índices en disco.** A esta escala el índice plano
  es exacto y no añade parámetros que justificar. Una base vectorial gestionada
  no aporta nada medible.
- **RRF como fusión por defecto.** No requiere ajustar pesos sobre las mismas
  consultas con las que se evalúa.
- **LLM solo en dos puntos y siempre condicionado por lo recuperado**
  (interpretar la consulta; explicar con citas a campos del libro). Es
  exactamente la salvaguarda que la literatura de 2026 pide frente a los
  ítems inventados.
- **Sin ajuste fino de modelos.** Sigue siendo la decisión correcta: los
  modelos preentrenados han mejorado más deprisa de lo que un TFG podría
  ajustar.
- **Modo sin LLM.** El sistema funciona como buscador híbrido puro con
  `LLM_PROVIDER=none`. Es la vía más sencilla para que un profesor lo ejecute.

---

## 4. Cambios propuestos

Cada cambio es una tarea con código, tests y una nota propia en
`docs/notes/tareas/`. La lista completa y su estado están en
`docs/notes/tareas/README.md`.

| Tarea | Cambio | Motivo en una línea |
|---|---|---|
| V01 | Normalización léxica para BM25 | El español sin plegado de acentos pierde coincidencias evidentes |
| V02 | Modelo de embeddings configurable, índice de producto interno normalizado y control de coherencia modelo-índice | Poder comparar modelos multilingües sin reescribir nada |
| V03 | Higiene de evaluación: intervalos bootstrap, desglose por idioma, `Judged@k`, ficha de los juicios | Sin esto ninguna comparación es defendible |
| V04 | Puerto `Reranker` y adaptador cross-encoder | Tercera etapa estándar en 2026 |
| V05 | Juez LLM de relevancia con procedencia y medida de acuerdo | Elimina el sesgo de pool que penaliza a los sistemas nuevos |
| V06 | Ablación unificada (léxico, embeddings, fusión, reranker, consultas sin tildes) | Decide con datos qué configuración queda por defecto |
| V07 | Línea base de contexto largo: el LLM lee el catálogo entero | Responde con datos a "¿hace falta recuperar en 2026?" |
| V08 | Actualización de valores por defecto y coste por consulta (latencia, tokens) | `gpt-4o-mini` se retira; el coste es un argumento central en 2026 |
| V09 | Ejecución en un comando: README, Makefile, script de demostración, CI | Que un profesor lo pruebe en cinco minutos |

---

## 5. Perspectivas alternativas consideradas

| Alternativa | Decisión | Motivo |
|---|---|---|
| Sustituir la recuperación por un LLM de contexto largo | No como sistema; sí como línea base (V07) | No escala al crecer el catálogo y multiplica el coste por consulta; pero hay que medirlo para poder afirmarlo |
| Convertir el proyecto en un recomendador conversacional agéntico | No | Cambia el objeto de estudio. El agente existente queda como extensión y no como sistema principal |
| Recomendador generativo (el LLM propone títulos) | No | Inventa ítems fuera del catálogo; exige decodificación restringida o entrenamiento, que el TFG excluye |
| Base vectorial (Qdrant, LanceDB, pgvector) | No | Sin beneficio medible a 177 libros; añade un servicio que el profesor tendría que levantar |
| Recuperación dispersa aprendida (SPLADE) o ColBERT | No | Licencias no comerciales o solo inglés; un cross-encoder da el mismo beneficio con menos piezas |
| Cambiar LangChain por otra librería | No | LangChain 1.x es rama activa; el coste de migrar no compra nada explicable |
| Ampliar ya el catálogo | Aplazado | Invalida los juicios existentes; primero hay que tener un juez reproducible (V05) |

---

## 6. Fuentes consultadas el 2026-11-11

- Du, M. "Tiered Super-Moore's Law: Price Evolution, Production Frontiers, and
  Market Competition in Large Language Model Inference Services".
  arXiv 2603.28576, 30-03-2026. https://arxiv.org/abs/2603.28576 [P: título,
  autor y cifras del resumen]
- Liao, H. et al. "Eliminating Out-of-Domain Recommendations in LLM-based
  Recommender Systems: A Unified View". arXiv 2505.03336 (v2, 22-01-2026).
  https://arxiv.org/abs/2505.03336 [P: título, autores y resumen]
- "Echoes in the Loop: Diagnosing Risks in LLM-Powered Recommender Systems".
  arXiv 2602.07442. https://arxiv.org/pdf/2602.07442 [S: solo resumen de
  búsqueda]
- Índice de precios de BenchLM. https://www.benchlm.ai/stats/llm-pricing [S]
- Pan, T. "Long-Context Models vs. RAG: When the 1M-Token Window Is the Wrong
  Tool", 09-04-2026.
  https://tianpan.co/blog/2026-04-09-long-context-vs-rag-production-decision-framework [S]
- Pan, T. "Hybrid Search in Production: Why BM25 Still Wins on the Queries
  That Matter", 12-04-2026.
  https://tianpan.co/blog/2026-04-12-hybrid-search-production-bm25-dense-embeddings [S]
- "Long context beats RAG on accuracy, so why keep retrieval". HackerNoon.
  https://hackernoon.com/long-context-beats-rag-on-accuracyso-why-keep-retrieval [S]

**Límites de esta validación.** Las afirmaciones marcadas [S] no deben citarse
en la memoria sin leer antes la fuente original. La comparación entre RAG y
contexto largo procede de fuentes secundarias; por eso la tarea V07 la mide
sobre el catálogo propio. No se ha consultado al tutor.

---

## 7. Decisiones del autor (2026-11-11)

Tras leer las secciones 1 a 5, el autor decidió lo siguiente. Estas decisiones
sustituyen a lo que digan las secciones anteriores cuando haya diferencia.

| N.º | Decisión | Consecuencia en las tareas |
|---|---|---|
| 1 | Añadir el reranker | Pasa a formar parte del pipeline por defecto (V13). La ablación V06 lo respalda: es la única mejora que supera la corrección de Holm (+0,098 nDCG@10) |
| 2 | Ampliar el catálogo: más libros y más fuentes, la mayor biblioteca posible; una API de pago solo si el coste es bajo | La ampliación deja de estar aplazada (V11, V12). La investigación del mismo día descartó Google Books como base (sin clave devuelve HTTP 429 con cuota 0 y sus condiciones prohíben construir una base de datos permanente) y las fuentes de pago; no hace falta pagar nada |
| 3 | Los embeddings deben cubrir todos los idiomas, porque la consulta puede llegar en cualquiera | El modelo multilingüe deja de ser una opción de ablación y pasa a ser requisito (V13): `intfloat/multilingual-e5-large-instruct` |
| 4 | Aplicar las medidas léxicas | El tokenizador normalizado es el de los índices nuevos (V13) |
| 5 | Aplicar la comparación RAG frente a contexto largo | V07 |
| 6 | Usar por ahora modelos baratos a través de OpenRouter; el modelo local es demasiado lento y deja de usarse | V08 y V10. Ollama sigue soportado en el código, pero no se usa |
| 7 | No usar modelos GPT; usar modelos chinos baratos | V10 mide cinco candidatos sobre las tareas del proyecto: `qwen/qwen3.8-flash` como LLM del sistema y `z-ai/glm-5.3-flash` como juez |
| 8 | Aplicar el LLM como recomendador | V07 lo incorpora anclado a lo recuperado (el LLM elige entre candidatos del catálogo) y mide cuántas recomendaciones quedan fuera del catálogo cuando no se le ancla |
| 9 | Completar la higiene de evaluación | V14: más consultas y en más idiomas, pool completo juzgado por un juez de otro proveedor que el LLM del sistema, hoja para la validación humana |

**Efecto sobre la sección 3 ("Qué se mantiene").** Se mantienen la
arquitectura hexagonal, SQLite, FAISS plano, RRF como fusión de partida y la
ausencia de ajuste fino. Cambian dos cosas: el catálogo deja de ser pequeño,
de modo que la implementación de BM25 y la carga de todos los libros en
memoria deben escalar (V12); y el LLM por defecto para el trabajo diario pasa
a ser un modelo alojado, con el modo sin LLM como vía de ejecución sin clave.

**Efecto sobre la sección 5.** "Ampliar ya el catálogo" pasa de aplazado a
decidido. El juez reproducible que se pedía como condición existe desde V05 y
se ha contrastado con un segundo juez de otro proveedor en V10.
