# Lista de tareas: actualización del TFG a noviembre de 2026

Origen: `docs/notes/2026-11-11-validacion-proyecto-2026.md` (incluye las
decisiones del autor del 2026-11-11, sección 7) y
`docs/notes/2026-11-11-propuesta-nube-y-mcp.md` (perfil nube y MCP).
Cada tarea tiene código, tests y una nota propia en este directorio con qué se
hizo, por qué, qué alternativas se descartaron, el objetivo y la fecha.

Estado: `[ ]` pendiente, `[~]` en curso, `[x]` hecha y validada, `[-]` aplazada.

## Hechas

- [x] **V01** Normalización léxica para BM25 (acentos, puntuación, palabras vacías)
- [x] **V02** Embeddings configurables, índice normalizado y control modelo-índice
- [x] **V03** Higiene de evaluación (bootstrap, idioma, `Judged@k`, ficha de juicios)
- [x] **V04** Puerto `Reranker` y adaptador cross-encoder
- [x] **V05** Juez LLM de relevancia con procedencia y acuerdo
- [x] **V06** Ablación unificada de la recuperación (colección de 177 libros)
- [x] **V07** El LLM como recomendador: anclado a lo recuperado, con el catálogo entero y sin catálogo
- [x] **V08** Proveedor OpenRouter, modelos por defecto y coste por llamada
- [x] **V10** Selección de los modelos LLM (OpenRouter) sobre las tareas del proyecto
- [x] **V11** Catálogo nuevo, multilingüe y de varias fuentes (Wikidata/Wikipedia, Gutenberg, DOAB, Open Library); pendiente completar los resúmenes de Wikipedia
- [x] **V12** Perfil nube: catálogo y búsqueda en Postgres (Neon) y paridad con el perfil local
- [x] **V15** Servidor MCP de solo lectura sobre el buscador
- [x] **V16** Modelos por API (embeddings y reranker) y regla de selección velocidad-calidad, también para el LLM
- [x] **V13** Integración del perfil nube (sin modelos en el proceso, embeddings por API en Postgres) y corrección del relleno de BM25
- [x] **V17** Despliegue público: catálogo de 30.000 obras en Neon y backend en Vercel, con límites de gasto
- [x] **V18** Ampliación del catálogo desplegado de 30.000 a 50.000 obras en el plan gratuito de Neon, con carga sin filas muertas
- [x] **V09** Ejecución en un comando (`make demo`, `make test`, `make eval`), README para el tribunal, CI y arranque sin clave de API
- [x] **V14** Evaluación del catálogo desplegado (50.000 obras) con protocolo declarado antes de ejecutar: 168 consultas en doce idiomas, dos jueces LLM, seis contrastes; pendiente la hoja de anotación humana
- [x] **V19** Interfaz web para el tribunal y chat del asistente: por qué aparece cada resultado, recorrido por el grafo en cada respuesta, estados de espera y diagramas (incluye V21; nota `V19-interfaz-web-y-chat.md`)
- [x] **V20** Asistente LangGraph como entrada conversacional (enruta, pregunta para aclarar, busca o delega en el agente) y explicaciones en paralelo
- [x] **V23** Comprobación del resultado de bge-m3 (se mantiene) y de los fallos del modelo de embeddings desplegado (causa y mitigación medida)

## En curso


## Pendientes

- [~] **V24** Aplicar al sistema la regla del protocolo: vectorial más reranker, con BM25 solo como respaldo
- [ ] **V22** Evaluación del agente y del asistente con protocolo declarado antes de ejecutar

## Aplazadas (con motivo)

- [-] Verificación de explicaciones oración por oración con NLI: esfuerzo alto; la precisión de citas previa al guardarraíl (V10, V14) cubre una parte.
- [-] Migrar a `uv`: choca con la regla `pip freeze > requirements.txt` del proyecto.
