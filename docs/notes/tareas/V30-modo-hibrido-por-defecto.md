# V30 · El modo de candidatos híbrido vuelve a ser el único valor por defecto

- **Fecha:** 2026-11-14
- **Estado:** hecha en el código y probada; no se desplegó nada. Queda una comprobación del autor (véase «Pendiente»)
- **Capa:** dominio (`app/domain/`), infraestructura (`app/infrastructure/config/`), API (`app/api/`), evaluación (`app/evaluation/`), construcción (`Makefile`), documentación (`README.md`, `.env.example`)
- **Commit:** `62b9a03`

## Qué se ha hecho

El modo de candidatos por defecto del sistema vuelve a ser el híbrido (BM25 y búsqueda vectorial fusionadas con RRF, y después el reranker). V24 lo había cambiado a «solo vectorial» en la aplicación; V30 lo deshace y deja un único valor por defecto. La búsqueda solo vectorial sigue disponible con `SEARCH_CANDIDATES=vector`.

Cambios en el código, según el commit:

- **Un solo valor por defecto en el dominio.** `DEFAULT_CANDIDATES` (`app/domain/value_objects.py`) vale `hybrid` y es el valor por defecto del argumento `candidates` de `SearchService` (`app/domain/services.py`). El comentario de esa constante recoge los motivos y las cifras.
- **Los ajustes de la aplicación lo usan.** `SearchProfileSettings` (`app/infrastructure/config/settings.py`) toma `DEFAULT_CANDIDATES` cuando `SEARCH_CANDIDATES` no está definida, en los dos perfiles. Se quitó la constante propia `DEFAULT_SEARCH_CANDIDATES`, que valía `vector`.
- **Los módulos de evaluación nombran su modo.** Cada `SearchService(` construido en `app/evaluation` pasa `candidates` de forma explícita (`comparison.py`, `evaluation_job.py`, `llm_model_selection.py`, `llm_path_evaluation.py`, `profile_parity.py`, `relevance_judge.py`, `retrieval_ablation.py`), de modo que un informe guardado no cambia si cambia el valor por defecto.
- **La evaluación del asistente queda fijada al modo de sus ejecuciones guardadas.** Como construye el servicio con los ajustes de la aplicación, `CLOUD_ENV` en `app/evaluation/assistant_evaluation/design.py` fija ahora `SEARCH_CANDIDATES` a `vector`, el modo que V22 y V25 registraron como `profile.candidates` en su `system.json`.
- **Descripciones y documentación.** Se ajustaron los textos de la API y del servidor MCP (`app/api/v1/schemas.py`, `search_endpoints.py`, `app/api/mcp/server.py`), `dependencies.py`, el `Makefile`, el `README.md` y `.env.example`.
- **Tests.** Dos tests nuevos o ampliados que fallan si algo se desvía:
  - `tests/infrastructure/test_search_profile_settings.py` comprueba que el valor por defecto de `SearchService`, el de los ajustes sin `SEARCH_CANDIDATES` (perfiles local y nube) y el del servicio que cablea la aplicación coinciden. Falla si el dominio y la aplicación tienen valores por defecto distintos.
  - `tests/evaluation/test_pinned_candidate_modes.py` recorre el código de `app/evaluation` y falla si algún `SearchService(` no nombra su modo (exige al menos 11 construcciones), y comprueba que `CLOUD_ENV` reproduce el modo guardado en `data/evaluation/assistant/system.json` y `data/evaluation/assistant_v25/system.json`.
  - También se actualizaron `tests/domain/test_vector_candidates.py` y `tests/integration/test_mcp_endpoint.py`.

Todo lo demás de V24 se mantiene, en particular la petición de embeddings de la consulta con un segundo proveedor de respaldo (hedge), la caché de vectores de consulta y la restricción de retención de datos.

## Por qué

La decisión la tomó el autor el 2026-11-14, después de V29. Con tres tipos de consulta, el resultado de comparar la fusión híbrida con el reranker (S5) y la búsqueda solo vectorial con el reranker (S4) es:

| Tipo de consulta | Diferencia S5 - S4 | Fuente | Juez |
|---|---|---|---|
| Descriptivas | -0.029 de nDCG@10 (H3, p de Holm <0.001) | `data/evaluation/CATALOG_EVALUATION.md` | Jueces LLM |
| Nombre de autor | +0.067 de nDCG@10, IC 95 % [+0.033, +0.107], p de Holm <0.001 (C1-A) | `data/evaluation/known_item/KNOWN_ITEM.md` (V29) | Ninguno: la verdad de referencia sale del catálogo |
| Título (T y título con autor, TA) | +0.001 y +0.000 de MRR@10 en T, +0.000 en TA, ninguna significativa | `data/evaluation/known_item/KNOWN_ITEM.md` (V29) | Ninguno |

- **Descriptivas.** El híbrido pierde 0.029 frente a solo vectorial con el reranker. Esa cifra descansa en jueces LLM cuyo acuerdo con la anotación del propio autor es bajo: kappa cuadrático ponderado de 0.272 con `z-ai/glm-5.3-flash` y de 0.301 con `minimax/minimax-m3.1-flash` (184 pares cada uno). Entre los dos jueces el kappa es 0.850. Que se parezcan entre sí no dice que se parezcan a una persona.
- **Autor.** El híbrido gana la cantidad de la tabla. Es la medida que no depende de jueces.
- **Título.** No hay diferencia.
- **Código y despliegue.** El modo híbrido es también el que corre en el sitio público, que no se actualizó con V24. Con V30 el código y el despliegue coinciden otra vez.

El autor sopesó una pérdida medida con jueces de poco acuerdo con él en un tipo de consulta frente a una ganancia medida sin jueces en otro, y eligió el híbrido.

## Objetivo

Que haya un solo valor por defecto, que el código y el despliegue ejecuten el mismo modo, y que ningún informe guardado cambie de configuración por este cambio.

## Por qué no otras alternativas

| Alternativa | Por qué no |
|---|---|
| Mantener solo vectorial y limitarse a decir el límite en la memoria | Dejaría por defecto el modo que pierde en el tipo de consulta medido sin jueces, y mantendría el código distinto del despliegue |
| Elegir el modo por consulta con un clasificador o con el LLM | Más piezas en movimiento, no está medido y queda fuera del alcance que queda antes de la defensa. Se deja como trabajo futuro |
| La fusión ponderada con w = 0.25 | En las consultas de autor ganó menos que la de pesos iguales (C2-A +0.029 frente a C1-A +0.067) y no se midió en consultas descriptivas |
| Dos valores por defecto (híbrido en el dominio, vectorial en la aplicación) | Es lo que había tras V24, y una revisión externa criticó exactamente eso |

## Cómo comprobarlo

Desde la raíz del repositorio, sin red ni clave:

```
.venv/bin/python -m pytest tests/infrastructure/test_search_profile_settings.py tests/evaluation/test_pinned_candidate_modes.py tests/domain/test_vector_candidates.py
make test
```

El implementador comunicó que `make test` dio 1878 pasados y 22 omitidos. Con el servicio en marcha, `/api/v1/health` informa del modo en `profile.candidates` y debe decir `hybrid` si `SEARCH_CANDIDATES` no está definida.

## Límites

- **La evaluación del asistente (V22, V25) se midió con candidatos solo vectoriales y no se repitió con el modo híbrido.** Los informes guardados no cambian y dicen qué modo usaron. Lo que el asistente haga con el híbrido no está medido.
- **El balance entre los dos tipos de consulta no se midió con peticiones de usuarios reales.** No se sabe cuál de los dos tipos es más frecuente.
- **La pérdida con consultas descriptivas sigue en pie** y descansa en jueces LLM de acuerdo bajo con el autor, como se explica arriba.
- **No se desplegó nada en esta tarea.**
- **Los límites de V29 siguen valiendo:** título exacto tal como está guardado, muestra de libros del catálogo y no de peticiones, y coincidencia de autor exacta tras normalizar.

## Pendiente

- **Comprobación del autor:** que `SEARCH_CANDIDATES` no esté puesta a `vector` en sus ficheros `.env` locales ni en el entorno del despliegue, porque eso anularía el valor por defecto.
- Redactar en la memoria que el modo por defecto es el híbrido y por qué, con el límite de arriba.
- Medir el asistente y el agente con el modo híbrido, si se quiere cerrar el hueco anterior.

## Ficheros

- **Dominio:** `app/domain/value_objects.py`, `app/domain/services.py`.
- **Infraestructura:** `app/infrastructure/config/settings.py`.
- **API:** `app/api/v1/dependencies.py`, `app/api/v1/schemas.py`, `app/api/v1/search_endpoints.py`, `app/api/mcp/server.py`.
- **Evaluación:** `app/evaluation/assistant_evaluation/design.py`, `comparison.py`, `evaluation_job.py`, `llm_model_selection.py`, `llm_path_evaluation.py`, `profile_parity.py`, `relevance_judge.py`, `retrieval_ablation.py`.
- **Construcción y documentación:** `Makefile`, `README.md`, `.env.example`.
- **Tests:** `tests/infrastructure/test_search_profile_settings.py`, `tests/evaluation/test_pinned_candidate_modes.py`, `tests/domain/test_vector_candidates.py`, `tests/integration/test_mcp_endpoint.py`.
- **Contexto:** `docs/notes/tareas/V24-candidatos-vectoriales-y-embeddings-con-respaldo.md`, `docs/notes/tareas/V29-alcance-conclusion-bm25.md`, `data/evaluation/CATALOG_EVALUATION.md`, `data/evaluation/known_item/KNOWN_ITEM.md`.
