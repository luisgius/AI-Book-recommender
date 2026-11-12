# V09 · Ejecución en un comando, README para el tribunal, integración continua y arranque sin clave de API

- **Fecha:** 2026-11-12
- **Estado:** hecha y validada en local; la integración continua no se ha ejecutado todavía en GitHub
- **Commit:** `1d2adfb`
- **Capa:** aplicación (`app/api/v1/`, `app/ui/`), dominio (`app/domain/services.py`), infraestructura (`app/infrastructure/config/`), empaquetado y herramientas (`Makefile`, `Dockerfile`, `.github/workflows/`, `README.md`)

## Qué se ha hecho

- Se ha hecho el proyecto comprobable por el tribunal en tres niveles de esfuerzo, descritos en la sección siguiente.
- `Makefile` nuevo con cinco objetivos: `make help`, `make demo`, `make test`, `make eval` y `make clean`.
  - `make demo` crea un entorno virtual (`.venv`, con el módulo `venv` estándar y `pip`; no se instala nada en el sistema), instala `requirements.txt`, construye los índices de la colección de 177 libros (`data/catalog.db`) en `data/indexes_demo/` y arranca el servidor en `http://127.0.0.1:8000/ui/`. Arranca la configuración evaluada (BM25 con el tokenizador normalizado, embeddings `intfloat/multilingual-e5-large-instruct`, fusión RRF y reranker `BAAI/bge-reranker-v2-m3`, el sistema `hybrid_norm_e5_bge_nopad` de `data/evaluation/BM25_PADDING.md`). Todas las variables de esa configuración se fijan de forma explícita en el propio `Makefile`, de modo que un fichero `.env` no puede cambiar la demo a otro perfil o modelo.
  - `make test` ejecuta `pytest`. `make eval` regenera los informes de `data/evaluation/` a partir de los juicios y las salidas de modelos guardadas en el repositorio, sin ninguna llamada a una API. `make clean` borra solo lo que `make` creó, apoyándose en ficheros marcador (`.venv/.created-by-make`, `data/indexes_ablation/.created-by-make`).
  - Opciones: `PORT=` (otro puerto), `RERANKER=none` (sin reordenación) y `PYTHON=` (otro intérprete). El `Makefile` acepta Python 3.11 a 3.13 y no 3.14, porque la versión fijada de `faiss-cpu` no tiene paquete para 3.14. Comprueba que el puerto esté libre antes de arrancar y funciona con el Make 3.81 que trae macOS. En Linux instala primero la versión de `torch` solo para CPU, porque el paquete de PyPI para Linux incluye bibliotecas CUDA de varios GB que estos modelos no usan.
- `README.md` reescrito para el tribunal: los tres niveles, el coste de la primera ejecución, la tabla de lo que `make eval` regenera y de lo que no, y los límites.
- `.github/workflows/tests.yml`: flujo de integración continua (CI, ejecución automática de los tests en GitHub en cada `push` y `pull request`). Un solo trabajo, con Python 3.13 y un solo comando, `make test`, el mismo que usa el tribunal. Guarda en caché las descargas de `pip` y el modelo pequeño `all-MiniLM-L6-v2` (90 MB) que cargan unos 60 tests.
- Cambio de comportamiento de la aplicación sin clave de LLM (siguiente sección).
- `Dockerfile`: pasa de Python 3.12 a 3.13 (la versión probada), instala antes la versión de `torch` para CPU y declara en su cabecera que empaqueta la configuración ligera, no la evaluada.
- `.env.example` y `.gitignore` actualizados (`data/indexes_demo/` queda fuera de git).
- Tests nuevos y ajustados: `tests/api/test_without_api_key.py` (composición de la aplicación en un clon sin `.env`, sin clave y sin `LLM_PROVIDER`), más casos en `tests/api/test_search_api.py`, `tests/api/test_ui.py`, `tests/api/test_agent_api.py` y en los tests de configuración y de cableado del LLM.

## Por qué

El autor pidió que todas las tareas estén en código y que se piense en la facilidad para los profesores. Un tribunal no tiene por qué conocer el proyecto, tener una clave de API ni dedicar una tarde a configurarlo. Hasta esta tarea, ejecutar el sistema exigía crear el entorno a mano, construir los índices con un script y, para el LLM, tener un servidor de Ollama o una clave. Además, la aplicación sin clave se comportaba mal (ver más abajo). Lo que no se puede ejecutar sin esfuerzo, el tribunal no lo puede comprobar.

## Objetivo

Que el tribunal pueda verificar el proyecto en tres niveles, de menos a más esfuerzo, y que en ninguno de ellos haga falta una clave de API.

| Nivel | Qué hace falta | Qué prueba |
|---|---|---|
| 1. URL pública | Un navegador, nada que instalar | Que el sistema desplegado (perfil nube, V17 y V18) funciona de verdad y sobre un catálogo de decenas de miles de obras |
| 2. `make demo` | Python 3.11 a 3.13 y `make`; ni clave ni `.env` | Que el perfil local evaluado (el que miden los informes) se construye desde un clon limpio y responde |
| 3. `make test` y `make eval` | Lo mismo | Que los tests pasan y que los informes de evaluación se pueden regenerar y comparar con los publicados, sin llamar a ninguna API |

El nivel 1 demuestra el despliegue, pero no permite inspeccionar el código que se ejecuta. El nivel 2 demuestra que el sistema evaluado se puede construir desde cero, pero sobre 177 libros. El nivel 3 es el único que permite comprobar las cifras de la memoria.

## Comportamiento sin clave de LLM: antes y después

Medido en el commit anterior, sin `.env` y sin nada escuchando en el puerto de Ollama:

- El LLM tomaba en silencio como valor por defecto un servidor local de Ollama que no existía.
- `/health` decía que el LLM estaba configurado (`ollama/qwen3.8:27b-mlx`).
- Las explicaciones llegaban como «Unable to generate explanation. Please try again.» tras reintentos con conexión rechazada, sin ningún aviso.
- La comprensión de la consulta recurría al respaldo en silencio.
- `POST /agent/ask` devolvía HTTP 500.

Ahora:

- `/health` informa del LLM como apagado (`configured: false, provider: none`).
- Una búsqueda que pide explicaciones o comprensión de la consulta devuelve 200 con los resultados y el aviso `llm_unavailable` (constante `NOTICE_LLM_UNAVAILABLE` en `app/domain/services.py`). La interfaz web muestra un texto en español que nombra `OPENROUTER_API_KEY` y desactiva esas dos opciones.
- El endpoint del agente devuelve 503 con un mensaje que nombra `OPENROUTER_API_KEY` y dice que la búsqueda simple sigue funcionando.
- El valor por defecto cambia: con `LLM_PROVIDER` sin definir, el proveedor es OpenRouter si hay `OPENROUTER_API_KEY` y ninguno en caso contrario (función `default_llm_provider` en `settings.py`). Ollama sigue funcionando si se nombra de forma explícita con `LLM_PROVIDER=ollama`.

Por qué cambia el valor por defecto: el LLM de trabajo del proyecto es un modelo alojado a través de OpenRouter (decisión 6 del autor, 2026-11-11: el modelo local era demasiado lento y deja de usarse; Ollama sigue soportado en el código, pero no se usa). Un valor por defecto que apunta a un servidor que el proyecto ya no usa solo produce fallos silenciosos en una máquina nueva. Con el cambio, sin clave el sistema declara qué no puede hacer, y con clave se activa solo.

## Cómo se ha validado

Todas las medidas las tomó el agente que implementó la tarea el 2026-11-12 en el portátil del autor (macOS, Apple silicon, Python 3.13), en un clon nuevo del repositorio sin `.env`. Son una sola medida en una sola máquina, no una distribución.

**Primera ejecución de `make demo`:** unos 95 s hasta que arrancó el servidor; `pip` descargó unos 155 MB; los modelos ya estaban en la caché local de Hugging Face, de modo que no se midió su descarga.

**Con las cachés llenas, desde cero** (`make clean` y después `make demo`): servidor arrancado a los 59.7 s, `/health` respondió 200 a los 66.3 s. La primera búsqueda tardó 4.90 s porque carga el reranker, y la siguiente 2.34 s. Un reinicio con el entorno y los índices ya construidos respondió en 8.5 s. Con `RERANKER=none`, una búsqueda tardó 0.08 s.

**Tamaño de las descargas de la primera vez**, medido en la caché local de Hugging Face:

| Qué | Tamaño |
|---|---|
| `intfloat/multilingual-e5-large-instruct` | 1.14 GB |
| `BAAI/bge-reranker-v2-m3` | 2.29 GB |
| Paquetes de Python (macOS arm64, Python 3.13) | 187 MB |
| Total de `make demo` | unos 3.6 GB |
| `make demo RERANKER=none` | unos 1.3 GB |
| `make eval` (cinco modelos) | 5.2 GB |

El clon ocupó unos 1.2 GB de disco en su punto máximo (`.venv` 1.1 GB, índices 14 MB) y se borró después.

**`make test`:** 1579 superados y 21 omitidos (los omitidos son las pruebas de integración de Postgres) en unos 20 s.

**`make eval`:** 602 s, código de salida 0, sin llamadas a ninguna API ni a Ollama.
- Se regeneraron volviendo a ejecutar los sistemas: `COMPARISON_BASELINE`, `RETRIEVAL_ABLATION` (su comprobación de reproducción informó de que todos los valores coinciden), `LLM_RECOMMENDER`, `API_MODELS` («API calls made in this run: 0») y los dos ficheros `results_llm_path`.
- Se recalcularon las tablas desde resultados guardados: `BM25_PADDING`, `PROFILE_PARITY`, `CLOUD_LIVE_CHECK` y `LLM_MODEL_SELECTION`.
- Comparados campo a campo con los ficheros versionados, solo diferían las fechas, el commit, las latencias, los tiempos de carga de modelos y de construcción de índices, el origen de los índices y el hash del fichero de consultas.

Se comprobó también que 8000 estaba ocupado en la máquina de prueba (el servidor se levantó en el 8010), y que el `Makefile` funciona con el Make 3.81 de macOS.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Exigir una clave de API para la demo | Un profesor tendría que crear una cuenta antes de ver nada. Además, la búsqueda, que es el núcleo del proyecto, no necesita el LLM |
| Mantener Ollama como valor por defecto del LLM | El proyecto ya no lo usa (decisión 6). En una máquina sin Ollama producía fallos silenciosos: `/health` afirmaba que el LLM estaba configurado, las explicaciones fallaban con un texto genérico y el agente daba un 500 |
| Una ruta solo con Docker | Obliga a instalar Docker. Además, la imagen no se ha ejecutado nunca (ver límites) y empaqueta la configuración ligera, no la evaluada |
| Un modelo de embeddings más ligero por defecto en la demo | La demo existe para mostrar el sistema que se evaluó (`multilingual-e5-large-instruct` y el reranker). Con un modelo más ligero, lo que se viera no sería lo medido. Para quien no quiera la descarga mayor existe `RERANKER=none` (1.3 GB en lugar de 3.6 GB), declarado como configuración sin la etapa que la ablación respalda |
| CI con varias versiones de Python | La versión probada es 3.13, y el `Makefile` ya limita el intervalo admitido (3.11 a 3.13) por el paquete de `faiss-cpu`. Más versiones multiplican el tiempo del flujo sin que haya una medida que lo justifique |
| Instalar con el `torch` de PyPI en Linux | Trae bibliotecas CUDA de varios GB que estos modelos, que corren en CPU, no usan |
| Dejar que `make clean` borre todo lo que haya en `.venv` e índices | Podría borrar un entorno o unos índices del propio usuario. Los ficheros marcador limitan el borrado a lo que `make` creó |
| Dejar que `.env` pueda cambiar la configuración de la demo | Un `.env` del autor o de un tribunal cambiaría de forma silenciosa el perfil o el modelo y la demo dejaría de ser el sistema evaluado; por eso `make demo` fija todas las variables |

## Límites que debe conocer el tribunal

- **La integración continua no se ha ejecutado en GitHub**, porque la rama no está publicada. Se comprobó que el YAML se lee bien y que su único comando (`make test`) pasa en el clon. El paso de Linux que instala `torch` para CPU y la imagen de Docker no se han ejecutado nunca. Las versiones de las acciones se compararon con las últimas publicadas el 2026-11-12 (`checkout` v7, `setup-python` v7, `cache` v6).
- Las medidas de tiempo y de tamaño son una sola ejecución en un portátil con Apple silicon. No se han probado Linux ni Windows. La primera ejecución exige una descarga de unos 3.6 GB (1.3 GB con `RERANKER=none`).
- Sin ninguna variable definida (arrancando la aplicación directamente, sin `make demo`), el código ejecuta la configuración ligera anterior (embeddings MiniLM, BM25 con separación por espacios, sin reranker), no la evaluada. Es lo que selecciona `make demo`. No se ha cambiado.
- El `data/catalog.db` versionado, de 177 libros, procede de Google Books, cuyas condiciones, según la nota V11, no permiten una base de datos permanente. El README lo declara; juzgarlo corresponde al autor.
- **Problemas hallados al construir `make eval`.** Se escriben tal como se hallaron. Los cinco se corrigieron el mismo día, 2026-11-12, en el commit `99a941c`; el resultado está en la sección «Actualización: corrección de los problemas de reproducibilidad», más abajo. El punto 3 contenía una afirmación que resultó ser errónea y se corrige en su propio texto.
  1. `RETRIEVAL_ABLATION_JUDGE_SENSITIVITY.md` no se pudo regenerar. Después de producirse, la tarea V07 añadió 146 juicios al fichero del segundo juez, y el comando del informe se detiene. Repetir la ablación del segundo juez sobre el fichero de hoy da las mismas clasificaciones con valores distintos: el contraste C5 sale +0.096 con p de Holm 0.024, frente a los +0.100 y 0.019 publicados, y `hybrid_norm_e5_bge` sale 0.688 frente a 0.701. Las siete conclusiones se mantienen. `make eval` omite estos dos pasos.
  2. `bm25_padding run --no-judge` siempre se detenía («22 top-10 pairs have no judgment») porque contaba los pares sin juicio contra el fichero base de juicios y no contra el fichero que contiene esas 22 calificaciones. `make eval` recalcula ese informe desde los resultados guardados.
  3. `COMPARISON_BASELINE.md` registraba un hash del fichero de consultas (`76c4dcce85e2`) que, según se creyó entonces, ninguna versión versionada tiene; el fichero actual es `60f78e921b0b`. Solo cambió esa línea; las métricas fueron idénticas. **Corrección:** la creencia era errónea. La investigación posterior mostró que `76c4dcce85e2` es el hash del fichero de consultas tal como se versionó en `f6485d5` (tarea V03), el mismo commit que el informe. La tarea V06 (commit `b8d6ee3`) añadió después un campo `text_accented` a 14 consultas en español y no cambió nada más (detalle en la sección de actualización).
  4. Dos modos sin conexión leen la identidad del modelo del entorno, por lo que `make eval` se la pasa de forma explícita.
  5. `app/domain/ports.py` importa un tipo de infraestructura, aunque solo para la comprobación de tipos.
- `make eval` regenera ahora `RETRIEVAL_ABLATION_JUDGE_SENSITIVITY.md` (punto 1, corregido en `99a941c`). No regenera `DEPLOYMENT_CHECK.md` (necesita el despliegue), `JUDGE_AGREEMENT.md` ni `COMPARISON_EXTENDED_QRELS.md` (necesitan el juez LLM), ni `EVALUATION_SUMMARY.md` y `QRELS_DATASHEET.md`, que están escritos a mano.

## Trabajo pendiente

- Hecho: la tarea de seguimiento para los cinco problemas de la sección anterior, completada el mismo día en el commit `99a941c` (ver la sección de actualización).
- Comprobar `make eval` sin ninguna conexión de red, que no está verificado (ver los límites de la sección de actualización).
- Publicar la rama para que el flujo de CI se ejecute en GitHub, y comprobar entonces el paso de Linux.
- Construir y ejecutar la imagen de Docker, que nunca se ha ejecutado.
- Decidir si la configuración por defecto sin variables debe ser la evaluada o seguir siendo la ligera.
- La evaluación sobre el catálogo desplegado (V14) sigue pendiente; su protocolo está declarado en el commit `9abf48e`.

## Cómo reproducirlo

```
git clone https://github.com/luisgius/AI-Book-recommender.git
cd AI-Book-recommender
make demo                     # http://127.0.0.1:8000/ui/ ; Ctrl+C lo detiene
make demo RERANKER=none       # sin el reranker: unos 1.3 GB en lugar de 3.6 GB
make demo PORT=8001           # si el 8000 está ocupado
make test                     # 1579 superados, 21 omitidos, unos 20 s
make eval                     # unos 10 minutos; regenera data/evaluation/ sin API
git diff --stat data/evaluation/   # solo deben cambiar fechas, commit, tiempos y el hash de consultas
make clean                    # borra lo que creó make
```

- Los tests nuevos de esta tarea: `pytest tests/api/test_without_api_key.py tests/api/test_search_api.py tests/api/test_ui.py tests/api/test_agent_api.py`.
- Para activar el LLM, copiar `.env.example` a `.env` y poner `OPENROUTER_API_KEY`. La clave no se versiona.

## Actualización: `make demo` y el LLM

El mismo día se comprobó que `make demo` arrancaba con el LLM activado en un equipo cuyo `.env` tuviera una clave de OpenRouter, en contra de lo que dice esta nota. Desde el commit `4be8a66`, `make demo` se ejecuta sin LLM en cualquier equipo y `make demo LLM=on` lo activa. El detalle está en `V19-interfaz-web-y-chat.md`.

## Actualización: corrección de los problemas de reproducibilidad

Los cinco problemas de la sección de límites se corrigieron el 2026-11-12 en el commit `99a941c`. Esta sección recoge qué se halló y qué resultó de cada uno.

**Informe de sensibilidad al segundo juez (`RETRIEVAL_ABLATION_JUDGE_SENSITIVITY.md`).**
- Causa: la tarea V07 (commit `a3468f5`) añadió 146 calificaciones a `app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json`, todas del juez `z-ai/glm-5.3-flash`, para pares fuera del conjunto de la ablación de V06 y repartidas en 36 consultas. Ninguna de las 919 calificaciones anteriores cambió ni se eliminó. Los valores cambiaron porque el DCG ideal usa todos los juicios relevantes, no solo los del top 10.
- Evidencia: el fichero de hoy, restringido a los 919 pares que calificó el primer juez, es idéntico byte a byte a la versión de la tarea V10 (commit `2d57a6d`), cuyo SHA-256 es el que quedó registrado en la ejecución publicada.
- Corrección: una opción nueva, `--llm-pairs-from`, en `retrieval_ablation --no-judge` conserva los juicios originales y solo las calificaciones del LLM de los pares del fichero indicado. Se detiene si alguno de esos pares no tiene calificación, y registra en los resultados que se fijó el conjunto y el hash del conjunto fijado. `judge_sensitivity` compara los dos jueces sobre los pares del primer juez y se detiene si la ablación no se ejecutó con ese conjunto fijado. El fichero de juicios no se modifica, de modo que los informes que usan las calificaciones más recientes no se ven afectados.
- Resultado: los resultados regenerados coinciden con los publicados en todas las métricas; el contraste C5 vuelve a ser +0.100 con p de Holm 0.019 y `hybrid_norm_e5_bge` vuelve a 0.701. En el informe solo difieren la fecha, el commit, una línea nueva que explica la fijación y los comandos de reproducción. Ambos pasos forman ahora parte de `make eval`.

**Relleno de BM25 (`BM25_PADDING.md`).**
- Causa: la ejecución sin conexión contaba los pares sin juicio contra el fichero base de juicios. Además, habría sobrescrito `relevance_judgments_extended_v13.json` con los juicios base, perdiendo las 22 calificaciones y las 41 añadidas después.
- Corrección: ahora comprueba contra el fichero V13 y nunca lo reescribe.
- Resultado: las 44 consultas devuelven las mismas clasificaciones, con los mismos contrastes y tablas; solo difieren las latencias, las fechas, el commit y el origen de los índices. `make eval` ejecuta ahora este módulo en lugar de volver a presentar los resultados guardados.

**Hash del fichero de consultas.**
- La creencia de que ninguna versión versionada tenía el hash `76c4dcce85e2` era errónea: es el hash del fichero de consultas tal como se versionó en `f6485d5` (tarea V03), el mismo commit que el informe. La tarea V06 (commit `b8d6ee3`) añadió después un campo `text_accented` a 14 consultas en español y no cambió nada más. Solo las ejecuciones de la ablación con acentos leen ese campo; la comparación con la línea base no lo lee.
- Tras regenerar, todas las métricas son idénticas y el informe registra el hash actual, `60f78e921b0b`.

**Identidad del modelo tomada del entorno.**
- Los dos módulos la toman ahora de sus resultados guardados cuando solo vuelven a presentar salidas en caché, y `make eval` ya no pasa esas variables.
- Uno de los dos casos era un defecto real: sin `LLM_REASONING` definida, un clon nuevo habría cambiado en el informe del recomendador la etiqueta del razonamiento del LLM del sistema de «off» a «at the model default».

**Capa de dominio.**
- `app/domain/ports.py` declara ahora dos protocolos pequeños de solo lectura, `ExplanationJudgment` y `JudgmentScore`, en lugar de importar un tipo de infraestructura. El comportamiento no cambia.
- Un test nuevo, `tests/domain/test_architecture.py`, analiza todos los módulos del dominio (incluidos los bloques de comprobación de tipos, las importaciones dentro de funciones y las relativas) y falla si alguno importa de las capas de infraestructura o de la API, de FastAPI, Starlette, LangChain, LangGraph, FAISS o de un controlador de base de datos.

**Validación.**
- Se ejecutó `make eval` en un clon nuevo y se comparó cada informe y cada fichero de resultados regenerado con la versión versionada, campo a campo: no difiere ninguna métrica, recuento, clasificación ni conclusión.
- Batería completa de tests: 1645 superados y 22 omitidos.

**Límites de esta validación.**
- No está verificada una ejecución de `make eval` sin ninguna red. Al forzar las bibliotecas de Hugging Face a modo sin conexión, esta versión de sentence-transformers rechazó los modelos guardados en la caché. Durante la ejecución, Hugging Face recibió peticiones de metadatos de modelos que ya estaban en disco y varias agotaron el tiempo de espera.
- Esa ejecución, que terminó con éxito, duró unos 30 minutos, frente a los 602 s medidos antes. Durante ella el portátil estuvo suspendido de forma intermitente con la tapa cerrada, de modo que la cifra no es una medida limpia.
- El flujo de CI sigue sin haberse ejecutado en GitHub.

## Ficheros

- **Herramientas y empaquetado:** `Makefile`, `Dockerfile`, `.github/workflows/tests.yml`, `.gitignore`, `.env.example`.
- **Documentación:** `README.md`, `docs/notes/tareas/README.md` (V09 pasa a hecha).
- **Dominio:** `app/domain/services.py` (`NOTICE_LLM_UNAVAILABLE` y su uso en la búsqueda simple y en la búsqueda con comprensión).
- **Infraestructura:** `app/infrastructure/config/settings.py` (`default_llm_provider`, valor por defecto de `LLM_PROVIDER`, mensaje del juez).
- **Aplicación:** `app/api/v1/dependencies.py` (mensaje de registro con el LLM apagado), `app/api/v1/schemas.py` (documentación del aviso), `app/api/v1/agent_endpoints.py` (503 con `AGENT_NO_LLM_MESSAGE`), `app/ui/static/app.js` (texto del aviso y opciones desactivadas).
- **Tests:** `tests/api/test_without_api_key.py` (nuevo), `tests/api/test_search_api.py`, `tests/api/test_ui.py`, `tests/api/test_agent_api.py`, `tests/domain/test_usage_budget_in_search.py`, `tests/infrastructure/llm/test_llm_wiring.py`, `test_openrouter_provider.py` y `test_relevance_judge_adapter.py`.
- **Cambiados en `99a941c` (corrección de reproducibilidad):**
  - **Herramientas y documentación:** `Makefile`, `README.md`.
  - **Dominio:** `app/domain/ports.py` (protocolos `ExplanationJudgment` y `JudgmentScore`).
  - **Evaluación:** `app/evaluation/retrieval_ablation.py` (`--llm-pairs-from`), `judge_sensitivity.py`, `bm25_padding.py`, `llm_path_evaluation.py` y `llm_recommender_evaluation.py`.
  - **Informes y resultados regenerados:** `data/evaluation/COMPARISON_BASELINE.md`, `results_comparison_baseline.json`, `RETRIEVAL_ABLATION_JUDGE_SENSITIVITY.md`, `results_judge_sensitivity.json` y `results_retrieval_ablation_z-ai-glm-5.3-flash.json`.
  - **Tests:** `tests/domain/test_architecture.py` (nuevo) y casos añadidos en `tests/evaluation/` para los cinco módulos (`test_bm25_padding.py`, `test_judge_sensitivity.py`, `test_llm_path_evaluation.py`, `test_llm_recommender_evaluation.py`, `test_retrieval_ablation.py`).
