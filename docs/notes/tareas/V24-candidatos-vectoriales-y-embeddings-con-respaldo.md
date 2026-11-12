# V24 · Candidatos vectoriales por defecto y embeddings de la consulta con un segundo proveedor

- **Fecha:** 2026-11-12
- **Estado:** hecha en el código y probada; no se desplegó nada, así que el sitio público sigue con la configuración anterior hasta que se vuelva a desplegar
- **Capa:** dominio (`app/domain/`), infraestructura (`app/infrastructure/search/`, `app/infrastructure/config/`), API (`app/api/`), interfaz web (`app/ui/static/`), construcción (`Makefile`), documentación (`README.md`, `.env.example`) y un script de medición (`scripts/`)
- **Commits:** `0907908` (candidatos vectoriales), `7feb239` (página web), `5f9d757` (`make demo`), `a714ae7` (embeddings con segundo proveedor), `ebeeb1d` (README, `.env.example` y script de medición), `e964682` (nota sobre las peticiones canceladas que pueden facturarse)

## Qué se ha hecho

- **Recuperación.** La configuración de búsqueda pasa de «fusión híbrida más reranker» a «búsqueda vectorial más reranker», con la búsqueda por palabras clave (BM25) solo como respaldo cuando falla la vectorial. El reranker (cross-encoder) es el modelo que reordena los primeros candidatos comparándolos con la consulta. El modo de candidatos es un ajuste nuevo, `SEARCH_CANDIDATES`, con dos valores: `vector` (por defecto, en los dos perfiles) y `hybrid` (la fusión de antes, que sigue disponible).
- **Embeddings de la consulta.** Las llamadas que convierten la consulta en un vector (el embedding de la consulta) tienen ahora un segundo proveedor como «hedge» (una petición de cobertura: si el primer proveedor tarda, se pide lo mismo a otro y vale la primera respuesta), una caché pequeña de vectores de consulta y una restricción de enrutado a endpoints sin retención de datos.
- **Página web, `make demo` y documentación.** La página describe la búsqueda que ejecuta el servidor, `make demo` usa candidatos vectoriales y el README y `.env.example` explican el cambio y dan las cifras que lo motivaron.
- **Medición en vivo** del efecto del hedge, con 600 llamadas reales (ver más abajo).
- No se desplegó nada ni se tocó la base de datos de Neon.

## Por qué

- **La regla declarada de antemano se cumplió.** El protocolo de la evaluación del catálogo desplegado (`docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`, sección 8, último párrafo) fijó, antes de ver ningún resultado, que la configuración desplegada mantendría BM25 en la fusión y que solo se cambiaría si la búsqueda vectorial con reranker (sistema S4) superase de forma significativa a la fusión con reranker (S5). La evaluación (V14, `data/evaluation/CATALOG_EVALUATION.md`) lo dio así: nDCG@10 de 0.798 para S4 frente a 0.769 para S5 con el juez principal (0.806 frente a 0.772 con el segundo), con una diferencia S5 menos S4 de -0.029 y -0.035 y p de Holm menores de 0.001 con los dos jueces. El nDCG@10 mide la calidad de las diez primeras posiciones y la corrección de Holm ajusta los p cuando se contrastan varias hipótesis a la vez. Aplicar la regla era, pues, cumplir lo escrito, no una decisión nueva.
- **La fusión sola también era peor.** Sin reranker, la fusión híbrida obtuvo 0.592 frente a 0.683 de la búsqueda vectorial sola (H1: -0.092 con el juez principal y -0.099 con el segundo, Holm menor de 0.001). Dejarla por defecto contradecía el propio resultado de la memoria.
- **Las llamadas de embedding fallaban y degradaban el sitio.** En V14, 51 de 504 llamadas al modelo desplegado (`qwen/qwen3-embedding-8b`) fallaron por tiempo de lectura agotado. Cuando el embedding de la consulta falla, la búsqueda recurre a BM25, que es el peor sistema medido (0.336). La investigación V23 (`data/evaluation/catalog50k/investigation/FINDINGS.md`, pregunta 2) mostró que los fallos vienen de episodios de lentitud del proveedor fijado (DeepInfra), que un reintento al mismo proveedor falla con él y que Nebius falla de forma independiente. Recomendó pedir a Nebius lo mismo si DeepInfra no responde en 1.0 s, repetir a DeepInfra a los 2.5 s y mantener el plazo de 5 s. V23 midió 0 fallos en 540 llamadas con esa política frente a 3 con la política desplegada.
- **Los vectores de Nebius sirven.** La misma investigación midió que un vector de consulta de Nebius, buscado contra los documentos guardados con DeepInfra, da un nDCG@10 de 0.689 frente a 0.682 con los vectores de consulta del propio DeepInfra. Por eso el hedge no obliga a recodificar el catálogo.

## Objetivo

Que el sistema ejecute por defecto la configuración que la evaluación midió como mejor y que sus llamadas de embedding de consulta fallen menos, sin cambiar lo que ya se publicó (los módulos de evaluación deben seguir reproduciendo sus informes) y sin desplegar todavía.

## Qué cambia en la recuperación y qué no

- **El modo de candidatos es un argumento del constructor del servicio de búsqueda del dominio** (`SearchService`, `app/domain/services.py`; constantes en `app/domain/value_objects.py`). Es información del dominio, que no sabe nada de variables de entorno: la aplicación lo lee de `SEARCH_CANDIDATES` y se lo pasa (`app/api/v1/dependencies.py`).
- **En modo `vector` no se llama a la búsqueda por palabras clave.** Los resultados se etiquetan como procedentes de la búsqueda vectorial (`source` igual a `vector`), la respuesta dice `search_mode: "vector_only"` y fusión `none`, y ningún resultado trae puesto ni puntuación de palabras clave (`lexical_rank` y `lexical_score` vacíos). La puntuación de recuperación es la similitud coseno del vector. Cuando la consulta pasa por la comprensión del LLM y se buscan varias variantes, en modo `vector` sigue sin llamarse a BM25 y las listas de las variantes se siguen fusionando entre sí con RRF.
- **El respaldo no cambia de naturaleza.** Si falla el embedding, si el índice vectorial no está disponible o si se agotó el presupuesto diario de búsquedas con modelos, responde BM25 con la misma marca de degradación y el mismo aviso que antes. Hay tests para los tres casos que comprueban además que la búsqueda por palabras clave se ejecuta de verdad (`tests/domain/test_vector_candidates.py`).
- **El valor por defecto del propio servicio sigue siendo `hybrid`.** Los módulos de evaluación construyen el servicio directamente en 11 sitios, y así sus informes publicados no cambian. Es la aplicación la que fija `vector` a partir de sus ajustes, y el endpoint de salud informa del modo (`profile.candidates`).
- **MCP, agente y asistente.** La herramienta `search_books` del servidor MCP devuelve el modo de búsqueda real en lugar de rotularlo siempre como híbrido, y sus descripciones y las del REST ya no dicen que toda búsqueda sea una fusión. El agente de búsqueda y el asistente llaman al mismo servicio y, por tanto, buscan con candidatos vectoriales.
- **Página web** (`app/ui/static/`). La sección «Cómo funciona» tiene un diagrama y una lista de etapas por modo y muestra la del modo que informa el servidor (un servidor que no lo informa es anterior al ajuste y se trata como híbrido). En modo `vector`, la fila de explicación de cada resultado dice «no se usó» para las palabras clave en lugar de «no lo encontró», la línea de estado cuenta solo candidatos vectoriales y ninguna puntuación se rotula como fusión RRF si el resultado no salió de una fusión.
- **`make demo`** usa candidatos vectoriales; `make demo CANDIDATES=hybrid` fusiona BM25. El `Makefile` avisa de que vectorial más reranker nunca se midió en la colección de 177 libros, donde la búsqueda vectorial sola (0.661) y el sistema fusionado con reranker (0.660) quedaron empatados.
- **README.** Dice que la fusión fue peor y da las cifras de arriba.

### Los informes publicados se siguen regenerando

Después del cambio se ejecutó `make eval` sin clave de API y sin llamadas a ninguna API. Se reescribieron diez informes en Markdown y once resultados en JSON, y se comparó cada valor de los JSON con el de la versión guardada en git. Las únicas diferencias fueron fechas, commits, tiempos y de dónde se cargó el índice; ninguna métrica cambió.

## Qué cambia en los embeddings de la consulta

Un módulo nuevo, `app/infrastructure/search/openrouter_hedge.py`, ejecuta las peticiones cubiertas con un cliente asíncrono en un hilo de fondo. Una petición síncrona que ya está en vuelo no se puede detener, y cancelar una tarea asíncrona es la forma portátil de abandonarla. El codificador (`openrouter_embedding_encoder.py`) construye las peticiones y la llamada sigue siendo bloqueante para el código de dominio.

- **Calendario de peticiones.** A los 0 s al proveedor fijado (DeepInfra); a `EMBEDDING_HEDGE_AFTER_S` (1.0 s) al segundo proveedor (Nebius), o antes si todas las peticiones enviadas hasta ese momento han fallado; a `EMBEDDING_HEDGE_PRIMARY_AGAIN_S` (2.5 s) una petición más al primero. El plazo total es el que ya existía, `API_QUERY_DEADLINE_S` (5 s).
- **Un tramo que falla pone en marcha el siguiente al instante**, sin esperar al temporizador; un error HTTP no consume la espera.
- **Gana la primera respuesta válida** y las demás se cancelan al momento: se cierra su conexión y una respuesta tardía no se lee. Cada petición hace un solo intento; las siguientes son sus reintentos.
- **Contabilidad.** Cada petición enviada queda registrada en el registro de uso: la ganadora con el coste que notifica OpenRouter y las demás como errores sin coste (`cancelled` para las perdedoras). Si fallan todas o vence el plazo se lanza un error con el resultado de cada tramo.
- **Caché de vectores de consulta.** Es una caché LRU (se descartan las entradas menos recientes) acotada, segura entre hilos y propia de cada proceso. Su clave es el modelo, la dimensión y el texto exacto de la consulta con su prefijo, no el proveedor, porque el proveedor solo añade ruido de coseno 0.996 a 0.9998. Una búsqueda repetida en la misma instancia recibe el mismo vector y por tanto la misma lista, en lugar de una de las dos variantes que devuelve DeepInfra. No cubre una instancia nueva ni otra que sirva en paralelo. Los documentos nunca se cubren ni se guardan en caché.
- **Identidad del vector.** El nombre del modelo del codificador sigue siendo el del proveedor fijado, porque los documentos guardados son suyos.
- **Compatibilidad con las evaluaciones.** El constructor del codificador deja el hedge, la caché y la restricción de retención apagados, de modo que las ejecuciones publicadas envían los mismos cuerpos de petición (su caché de respuestas está indexada por el cuerpo). La aplicación los enciende a partir de sus ajustes.

## Ajustes añadidos

| Ajuste | Valores | Valor por defecto |
|---|---|---|
| `SEARCH_CANDIDATES` | `vector` o `hybrid` | `vector` en los dos perfiles |
| `EMBEDDING_HEDGE_PROVIDER` | un proveedor, o `none` (restaura la política anterior) | `nebius` en el perfil nube con el modelo por defecto y el proveedor fijado DeepInfra; en cualquier otro caso, desactivado |
| `EMBEDDING_HEDGE_AFTER_S` | segundos | 1.0 |
| `EMBEDDING_HEDGE_PRIMARY_AGAIN_S` | segundos, o `none` | 2.5 |
| `EMBEDDING_QUERY_CACHE_SIZE` | número de entradas; 0 la desactiva | 256 en el perfil nube; 0 en el resto |
| `EMBEDDING_OPENROUTER_ZDR` | `true` o `false` | `true` para embeddings por API |

El plazo global de la llamada es el ajuste que ya existía, `API_QUERY_DEADLINE_S` (5 s). Los ajustes se validan al arrancar: el segundo proveedor debe diferir del fijado, ninguna espera puede superar el plazo, y el hedge y la restricción de retención solo valen con embeddings por API.

## Medición en vivo

Hecha el 2026-11-12 de 15:17 a 15:26 desde el portátil del autor, con las 100 consultas de evaluación elegidas con semilla, 3 repeticiones por brazo (300 llamadas cada uno), brazos intercalados llamada a llamada y caché desactivada en los dos (`scripts/measure_embedding_hedge.py`). El brazo «política actual» es la desplegada hasta ahora (DeepInfra fijado, un intento de 3 s más un reintento, plazo de 5 s); el brazo «con hedge» es la configuración nueva.

| | Política actual | Con hedge |
|---|---|---|
| Fallos | 3 / 300 (tiempos de lectura agotados, todos entre 15:22:59 y 15:23:18) | 0 / 300 |
| Mediana | 0.564 s | 0.501 s |
| p95 | 1.715 s | 1.557 s |
| p99 / máximo | 4.748 / 5.246 s | 2.348 / 2.950 s |
| Llamadas de más de 3 s | 4 | 0 |
| Respondió | DeepInfra | DeepInfra 287, Nebius 13 |
| Peticiones extra por búsqueda | 0 | 0.213 (segundo proveedor en 61 llamadas, tercera petición en 3; 64 canceladas) |
| Coste notificado en las respuestas | 0.0000981 USD | 0.0000993 USD |

Gasto total en vivo: 0.000216 USD.

**Qué muestra.** Que, en esta ventana de nueve minutos, el hedge no empeoró la mediana ni el p95, evitó los tres fallos y las cuatro llamadas de más de 3 s del brazo de control, y mandó 0.213 peticiones extra por búsqueda a un coste notificado casi igual. La cifra de peticiones extra es algo mayor que la de V23 (unas 0.17).

**Qué no muestra.**
- Los tres fallos del control cayeron en veinte segundos, entre las 15:22:59 y las 15:23:18: fue una racha corta, no un episodio completo de lentitud de DeepInfra. Esta medición no demuestra el comportamiento del hedge durante un episodio de minutos, que es el caso para el que se diseñó; esa cifra (alrededor de un 3 % de fallos en lugar de un 12 %) sigue siendo la estimación de V23, no una observación.
- Son 300 llamadas por brazo desde un portátil, no desde la región de la función desplegada.
- La diferencia de mediana (0.564 s frente a 0.501 s) cabe en el ruido de una ventana tan corta.

**Las peticiones canceladas probablemente se facturan.** El coste notificado en las respuestas fue unos 0.000017 USD menor que lo que subió el contador de uso de la clave, que es más o menos lo que costarían las peticiones canceladas. Por eso el coste registrado puede quedar algo por debajo del real; el código lo dice en un comentario (`e964682`). El importe es despreciable.

## Política de datos

- La documentación de enrutado de proveedores de OpenRouter ofrece una opción que restringe el enrutado a endpoints con retención cero de datos (ZDR, por sus siglas en inglés). El codificador envía `provider.zdr=true` en todas las peticiones de embeddings por API.
- Los endpoints de DeepInfra y de Nebius para este modelo figuran como elegibles en la lista pública del servicio, comprobada el 2026-11-12, y una petición fijada a cada uno con la restricción devolvió 200.
- La página de embeddings de esa misma documentación no menciona la opción y afirma que los embeddings del mismo texto son siempre idénticos, algo que la investigación V23 contradice para DeepInfra.
- **No se leyeron las políticas de privacidad propias de los dos proveedores.** Que el servicio los marque como ZDR es lo único verificado; ver Pendiente.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Mantener la fusión híbrida como configuración por defecto | La regla declarada de antemano pedía cambiarla si S4 superaba a S5, y lo hizo con los dos jueces. La fusión sola también fue peor que la vectorial (H1). Sigue disponible con `SEARCH_CANDIDATES=hybrid` |
| Quitar la búsqueda por palabras clave del todo | Es el único camino de respaldo cuando falla el embedding, el índice vectorial o se agota el presupuesto diario de búsquedas con modelos; sin ella la búsqueda dejaría de responder en esos casos |
| Cambiar el modelo de embeddings a `baai/bge-m3` | La regla de la sección 8 mantuvo `qwen/qwen3-embedding-8b` porque `baai/bge-m3` fue significativamente peor en calidad con los dos jueces (0.426 frente a 0.683 con el principal), y V23 no encontró ningún defecto en esa medida. Además exigiría recalcular 50,000 vectores |
| Fijar Nebius o SiliconFlow como único proveedor | V23 midió fallos del 11.7 % y el 16.1 % en la ronda del episodio y una mediana tres veces mayor |
| Reintentos o plazos más cortos solo en DeepInfra, o duplicar solo en DeepInfra | Las estrategias sobre el mismo proveedor fallan juntas durante un episodio; duplicar solo en DeepInfra dejó un 5.6 % de fallos |
| Las alternativas propias de OpenRouter (`allow_fallbacks`) | Nunca se activan ante la lentitud: un tiempo agotado en el cliente no es un error del que OpenRouter pueda recurrir a otro proveedor |
| Un único plazo más largo (10 s) | Cambiaría los fallos por búsquedas de hasta 5 a 10 s, y V23 no pudo observar sin censura las latencias de un episodio |
| Un tercer proveedor (SiliconFlow) | V23 midió que la variante de tres proveedores rinde igual de bien que la de dos, pero exige un tercer proveedor en el enrutado y en la política de datos |
| Recodificar el catálogo con un proveedor de vectores estables para recuperar la repetibilidad | Renuncia al segundo proveedor del que depende el hedge; la caché por texto recupera la repetibilidad de las búsquedas repetidas sin recodificar |
| Cambiar el valor por defecto del propio servicio de búsqueda a `vector`, en lugar de fijarlo en la aplicación | Los módulos de evaluación lo construyen directamente en 11 sitios y sus informes publicados cambiarían de configuración; así el dominio conserva la configuración publicada y es la aplicación la que elige |

## Cómo se ha validado

- **Tests:** 1784 pasaron y 22 se omitieron.
- **Tests nuevos del modo de candidatos** (`tests/domain/test_vector_candidates.py`): el modo vectorial no llama a BM25 y lo declara; el respaldo funciona con embedding fallido, índice no disponible y presupuesto agotado, y BM25 se ejecuta de verdad; con comprensión de la consulta, los resultados se etiquetan como vectoriales; el modo híbrido sigue fusionando; el valor por defecto del dominio sigue siendo híbrido; un modo desconocido se rechaza. También se ampliaron los de ajustes del perfil, de MCP y de la página web.
- **23 tests del hedge** (`tests/infrastructure/test_openrouter_hedge.py`) sobre transportes falsos: calendario y tiempos, cancelación del perdedor, error del primario que adelanta al segundo, fallo de todos los proveedores con todas las peticiones registradas, plazo que cancela lo que sigue en vuelo, cabecera de clave sin cabeceras identificativas, aciertos de caché, caché por texto exacto y acotada, y documentos que no se cubren ni se guardan en caché.
- **Ninguna prueba llama a la red.** La batería completa se ejecutó además con las conexiones no locales bloqueadas, con 0 intentos. El controlador de pruebas del perfil nube ahora simula las llamadas cubiertas y bloquea cualquier conexión fuera de la máquina.
- **Incidente durante el desarrollo.** Dos ejecuciones de tests enviaron 6 peticiones sin autenticar a la API con claves ficticias; todas fueron rechazadas o fallaron y no se facturó nada. Esos tests usan ahora transportes falsos y el controlador de pruebas bloquea las conexiones externas.
- **`make eval`** sin clave de API: los informes publicados se regeneran con las mismas métricas (ver arriba).
- **Medición en vivo** descrita arriba.

## Límites que debe conocer el tribunal

- **El sitio público sigue con la configuración anterior** (híbrida más reranker, una sola petición de embedding) hasta que se vuelva a desplegar. Nada de lo descrito aquí se ejecuta todavía en la URL pública.
- **Vectorial más reranker se midió sobre el catálogo desplegado de 50,000 obras, no sobre la colección local de 177 libros.** Allí la configuración nueva no se midió; solo se sabe que la búsqueda vectorial sola (0.661) y la fusión con reranker (0.660) quedaron empatadas. Los juicios del catálogo desplegado son de dos jueces LLM y las consultas las generó en parte un LLM (limitaciones de V14).
- **El hedge no se probó durante un episodio lento completo** del primer proveedor ni desde la región del despliegue, incluidas dos peticiones salientes a la vez desde una función serverless. La medición en vivo fue una ventana de nueve minutos desde un portátil, con una racha corta de fallos.
- **No se leyeron las políticas de privacidad de los dos proveedores.** Solo se comprobó la lista de endpoints ZDR del servicio y que una petición fijada a cada uno con la restricción devuelve 200.
- **La página no se probó en un navegador real:** solo se renderizó el diagrama nuevo.
- **La repetibilidad solo mejora dentro de una instancia:** la caché es por proceso, así que una instancia nueva o una en paralelo puede devolver otra variante del vector y reordenar un poco los diez primeros (V23 midió que no cambia la calidad).
- **El coste registrado puede ser algo menor que el real** por las peticiones canceladas que probablemente se facturan.
- **La documentación del servicio sobre embeddings es contradictoria:** afirma que los embeddings del mismo texto son siempre idénticos y V23 lo contradice para DeepInfra.

## Pendiente

- **Desplegar.** Está a la espera de que el autor dé el visto bueno. Necesita las variables de entorno del despliegue: `SEARCH_CANDIDATES` (`vector` es el valor por defecto, basta con no definirla), `EMBEDDING_HEDGE_PROVIDER`, `EMBEDDING_HEDGE_AFTER_S`, `EMBEDDING_HEDGE_PRIMARY_AGAIN_S`, `EMBEDDING_QUERY_CACHE_SIZE` y `EMBEDDING_OPENROUTER_ZDR` (con el perfil nube y el modelo por defecto, los valores por defecto ya activan el hedge, la caché y la restricción de retención; `none` y 0 devuelven la política anterior). Después conviene comprobar `/health` (`profile.candidates`) y medir desde la región de la función.
- **Evaluar el asistente y el agente con esta configuración**, con su propio protocolo (V14 no los midió).
- **Leer las políticas de privacidad de DeepInfra y de Nebius** y contrastarlas con la declaración de privacidad del sitio.
- **Medir vectorial más reranker sobre la colección de 177 libros**, si se quiere cerrar el hueco que señalan el `Makefile` y el README.
- **Observar el hedge durante un episodio lento real** una vez desplegado.

## Cómo reproducirlo

Desde la raíz del repositorio. Los tests y `make eval` no necesitan red ni clave; el script de medición hace llamadas reales y necesita una clave de OpenRouter en un fichero `.env` (no se imprime ni se guarda).

```
.venv/bin/python -m pytest tests/domain/test_vector_candidates.py tests/infrastructure/test_openrouter_hedge.py
make test
make eval
make demo
make demo CANDIDATES=hybrid
.venv/bin/python -m scripts.measure_embedding_hedge --env-file <ruta al .env> --output <fichero de llamadas> --summary <fichero de resumen>
```

`make demo CANDIDATES=hybrid` arranca el sistema fusionado de la colección local. El script de medición admite `--queries`, `--repetitions` y `--max-spend` (tope de gasto) y se detiene antes de superarlo.

## Ficheros

**Dominio**
- `app/domain/value_objects.py`: constantes del modo de candidatos.
- `app/domain/services.py`: `SearchService` con el argumento `candidates` y la propiedad `candidate_mode`.

**Infraestructura**
- `app/infrastructure/config/settings.py`: ajustes nuevos de candidatos, hedge, caché y restricción de retención, con sus validaciones.
- `app/infrastructure/search/openrouter_hedge.py`: peticiones cubiertas (módulo nuevo).
- `app/infrastructure/search/openrouter_embedding_encoder.py`: cobertura de la consulta, caché de vectores y registro de cada petición.
- `app/infrastructure/search/openrouter_http.py`: sin cambios en esta tarea; el hedge reutiliza su comprobación de respuestas (`OpenRouterHttp._check`).
- `app/infrastructure/llm/agents/search_agent.py`: descripción de la herramienta de búsqueda.

**API**
- `app/api/v1/dependencies.py`, `search_endpoints.py`, `schemas.py`, `agent_endpoints.py`: cableado del modo, `/health` y descripciones.
- `app/api/mcp/` (`server.py`, `tools.py`, `schemas.py`): modo real de búsqueda.

**Interfaz web:** `app/ui/static/` (`index.html`, `app.js`, `styles.css`).

**Construcción y documentación:** `Makefile`, `README.md`, `.env.example`.

**Medición:** `scripts/measure_embedding_hedge.py`.

**Tests**
- `tests/domain/test_vector_candidates.py`
- `tests/infrastructure/test_openrouter_hedge.py`
- `tests/infrastructure/test_search_profile_settings.py`
- `tests/api/test_mcp_server.py`, `tests/api/test_ui.py`, `tests/integration/test_mcp_endpoint.py`
- `tests/api/cloud_profile_driver.py`, `tests/infrastructure/test_public_deployment_guards.py`: controlador de pruebas sin red.

**Contexto:** `docs/notes/tareas/V14-evaluacion-catalogo-desplegado.md`, `docs/notes/tareas/V23-comprobacion-embeddings.md`, `data/evaluation/CATALOG_EVALUATION.md`, `data/evaluation/catalog50k/investigation/FINDINGS.md`, `docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md` (sección 8).
