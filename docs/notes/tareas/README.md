# Lista de tareas: actualización del TFG a noviembre de 2026

Origen: `docs/notes/2026-11-11-validacion-proyecto-2026.md`.
Cada tarea tiene código, tests y una nota propia en este directorio con qué se
hizo, por qué, qué alternativas se descartaron, el objetivo y la fecha.

Estado: `[ ]` pendiente, `[x]` hecha y validada, `[-]` aplazada.

## Tareas de esta tanda

- [x] **V01** Normalización léxica para BM25 (acentos, puntuación, palabras vacías)
- [x] **V02** Embeddings configurables, índice normalizado y control modelo-índice
- [ ] **V03** Higiene de evaluación (bootstrap, idioma, `Judged@k`, ficha de juicios)
- [ ] **V04** Puerto `Reranker` y adaptador cross-encoder
- [ ] **V05** Juez LLM de relevancia con procedencia y acuerdo
- [ ] **V06** Ablación unificada y elección de la configuración por defecto
- [ ] **V07** Línea base de contexto largo (catálogo completo en el prompt)
- [ ] **V08** Valores por defecto actualizados y coste por consulta
- [ ] **V09** Ejecución en un comando (README, Makefile, demo, CI)

## Aplazadas (con motivo)

- [-] Ampliar a 80-100 consultas: las consultas debe redactarlas y revisarlas una persona.
- [-] Muestra humana de 150-200 juicios para validar al juez LLM: la anota el autor.
- [-] Verificación de explicaciones oración por oración con NLI: esfuerzo alto; depende de V05.
- [-] Ampliar el catálogo con fuentes en español: invalida los juicios actuales.
- [-] Migrar a `uv`: choca con la regla `pip freeze > requirements.txt` del proyecto.
