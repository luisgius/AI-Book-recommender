# Lista de tareas: actualización del TFG a noviembre de 2026

Origen: `docs/notes/2026-11-11-validacion-proyecto-2026.md` (incluye las
decisiones del autor del 2026-11-11, sección 7).
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
- [x] **V08** Proveedor OpenRouter, modelos por defecto y coste por llamada
- [x] **V10** Selección de los modelos LLM (OpenRouter) sobre las tareas del proyecto

## En curso

- [~] **V07** El LLM como recomendador: RAG frente a contexto largo y frente a "sin catálogo"
- [~] **V11** Catálogo nuevo, multilingüe y de varias fuentes (Wikidata/Wikipedia, Gutenberg, DOAB, Open Library)

## Pendientes

- [ ] **V12** Búsqueda a la escala del catálogo nuevo (BM25 escalable, hidratación bajo demanda, índices)
- [ ] **V13** Configuración por defecto según lo medido (tokenizador normalizado, e5 multilingüe, reranker bge con latencia acotada) y catálogo pequeño versionado
- [ ] **V14** Evaluación sobre el catálogo nuevo: consultas ampliadas y multilingües, pool juzgado por el juez LLM, hoja de anotación humana, precisión de citas
- [ ] **V09** Ejecución en un comando (README, Makefile, demo, CI)

## Aplazadas (con motivo)

- [-] Verificación de explicaciones oración por oración con NLI: esfuerzo alto; la precisión de citas previa al guardarraíl (V10, V14) cubre una parte.
- [-] Migrar a `uv`: choca con la regla `pip freeze > requirements.txt` del proyecto.
