# V19 y V21 · Interfaz web para el tribunal: página reconstruida y chat con el asistente

- **Fecha:** 2026-11-12, en dos pasos: V19 (commit `e5587e7`, página reconstruida) y V21 (commits `ede1d70`, `d0bab6b`, `e745a95`, `3a13dcf` y `4be8a66`, chat del asistente y tres detalles del servidor que la página necesitaba)
- **Estado:** hecha y probada con tests y con comprobaciones en un navegador (Chrome) en un portátil; no desplegada en el sitio público y no probada con el perfil nube
- **Capa:** aplicación (`app/ui/static/`, `app/api/v1/`), dominio (`app/domain/`), infraestructura (`app/infrastructure/llm/`), empaquetado y documentación (`Makefile`, `README.md`)

## Qué se ha hecho

- **Primer paso (V19).** Se reconstruyó la página web del proyecto, que antes funcionaba pero era escueta y no mostraba el agente. La nueva versión tenía dos modos, «Buscar libros» y «Preguntar al agente», estados de espera, mensajes de error en español, una sección «Cómo funciona» con un diagrama y, en cada resultado, una fila que explica por qué aparece.
- **Segundo paso (V21).** El modo «Preguntar al agente» se sustituyó por «Conversar con el asistente», que habla con el asistente de LangGraph descrito en la nota V20 (`POST /api/v1/assistant/chat`). Se añadió el hilo de conversación, las preguntas aclaratorias con respuestas sugeridas, el recorrido del grafo bajo cada respuesta y un segundo diagrama, el del grafo.
- **Tres detalles del servidor** que la página necesitaba (sección «Detalles del servidor añadidos para la página»): la posición de cada resultado en cada etapa de la búsqueda, los libros que devolvió cada llamada del agente y la corrección del texto de explicación cortado.
- **`make demo` arranca ahora sin LLM en todas las máquinas**, y `make demo LLM=on` lo activa. Es una corrección de un comportamiento que la nota V09 describía de otra manera.

La página sigue siendo tres ficheros estáticos: `app/ui/static/index.html`, `app.js` y `styles.css`.

## Por qué

La petición del autor empezó con una pregunta sobre la interfaz: escribió «and about the frontend??» y señaló una lista de repositorios que podrían ayudar. La respuesta del diseñador fue que la página funcionaba, pero era sencilla y no mostraba el agente, y que ninguno de los repositorios de la lista encajaba (un skill de diseño de páginas de aterrizaje y un generador de páginas con desplazamiento en 3D, que sirven para vender un producto y no para enseñar un sistema). Después el autor decidió que LangGraph es el punto de entrada del sistema y que el chat puede hacer preguntas de seguimiento (sus palabras están en la nota V20). Con esas dos decisiones, la página tenía que dejar de mostrar solo la búsqueda y pasar a hablar con el asistente.

El público de la página es el tribunal, no un cliente. Por eso el criterio de diseño fue hacer visible el recorrido de una petición (qué etapa encontró cada libro, qué camino siguió el asistente, cuánto tardó cada nodo), no vender un producto.

## Objetivo

1. Que un examinador, sin conocer el proyecto, pueda probar la búsqueda y el asistente desde un navegador y ver cómo se llega a cada resultado.
2. Que cada afirmación de la pantalla se pueda comprobar contra el servidor: posiciones reales de cada etapa, no estimadas; citas ya verificadas por código; ruta del asistente tal como la devuelve el grafo.
3. Que la página explique por sí misma las esperas, los límites y los fallos, en español, sin dejar al usuario ante una pantalla en blanco.
4. Que el código sea explicable en la memoria y se despliegue como ficheros estáticos.

## Decisiones de diseño

Son decisiones del diseñador, con su motivo. El autor no las ha revisado una a una.

| Decisión | Motivo |
|---|---|
| Tres ficheros estáticos en HTML, CSS y JavaScript sin marco de trabajo, sin paso de compilación, sin CDN y sin ninguna petición a servidores de terceros | El código tiene que poder explicarse en la memoria; se despliega como ficheros estáticos; y las consultas de los visitantes no deben ir a ningún sitio que no sea el servidor de la aplicación |
| Aspecto sobrio y de libro: tipografías del sistema (pila con serifa para títulos y texto de libros, pila sin serifa para los controles), fondo cálido tipo papel, modo oscuro y un solo color de acento | El público es un tribunal y la página debe mostrar el sistema, no distraer de él; las fuentes del sistema evitan cargar nada externo |
| Los datos de la API se insertan siempre como texto, nunca como HTML | Los títulos y descripciones proceden de fuentes que cualquiera puede editar (por ejemplo Wikipedia); un texto con marcas no debe poder inyectar contenido en la página |
| La conversación solo vive en la memoria de la página: no se guarda en el almacenamiento del navegador ni en el servidor | Coincide con el asistente sin estado de V20 y con el rechazo de la memoria de largo plazo de la nota de validación; no se conservan datos de nadie y se pierde al recargar (la página lo dice) |
| Sustituir el modo del agente por el modo del asistente, en lugar de mantener los dos | El asistente ya es la puerta de entrada y enruta al agente cuando conviene; dos modos obligarían al usuario a elegir lo que el grafo decide |

## Qué tiene la página

### Comunes

- Una cabecera con una línea que indica si el servidor responde y si tiene LLM.
- Dos modos como pestañas: «Buscar libros» y «Conversar con el asistente».
- Un cuadro de detalle de libro compartido entre los dos modos (se abre desde el título de un resultado o desde un libro del asistente) con los datos del libro, su fuente y licencia, y un botón para ver libros similares.
- Estados de espera que explican que, si el servidor estaba inactivo, la primera petición puede tardar entre 10 y 20 segundos, y que las explicaciones y el agente tardan más, con un botón para cancelar.
- Mensajes separados, en español, para: sin conexión con el servidor, límite por minuto (HTTP 429), presupuesto diario agotado, sin LLM en el servidor (503), otros errores y búsquedas degradadas (por ejemplo, sin búsqueda por significado). Cada mensaje dice qué ha pasado y qué sigue funcionando.
- Una sección plegable «Cómo funciona» con dos diagramas dibujados en SVG dentro de la propia página (la canalización de búsqueda y el grafo del asistente) y una frase por etapa.
- Una sección plegable «Sobre los datos» con la atribución y las licencias de las fuentes (Wikidata, Wikipedia, Project Gutenberg y DOAB) y la declaración de que la página no carga nada de terceros.
- Accesibilidad: uso completo con teclado y foco visible, enlace para saltar al contenido, mensajes nuevos anunciados a los lectores de pantalla sin mover el foco, etiqueta en cada control, modo oscuro, respeto de la preferencia de movimiento reducido y diseño que se adapta desde 360 px de ancho.

### Modo «Buscar libros»

- Consultas de ejemplo en español, inglés, francés, alemán e italiano.
- Opciones y filtros plegados: interpretar la consulta con el LLM, explicar los primeros resultados, diversificar, idioma, categoría, años y número de resultados.
- Cada resultado muestra título, autores, año e idioma, descripción y, si se pidió, la explicación del LLM con sus citas verificadas.
- Una fila «Por qué aparece» con la posición del libro en la búsqueda por palabras clave, en la búsqueda por significado, tras el reordenador y en la lista final. Los valores exactos se abren a petición. Cuando una etapa no devolvió el libro, la fila lo dice.
- Fuente y licencia del libro, y un botón de libros similares.

### Modo «Conversar con el asistente»

- El hilo de la conversación y un cuadro de mensaje limitado a 500 caracteres, con ejemplos de tres tipos de petición: concreta, vaga (debería provocar una pregunta) y de varios pasos (debería ir al agente).
- Las preguntas aclaratorias aparecen con sus respuestas sugeridas como botones; también se puede escribir una respuesta libre.
- Las respuestas de búsqueda muestran primero los libros recomendados, con la razón y las citas, y después el resto de resultados.
- Bajo cada respuesta hay una línea plegada que, al abrirse, muestra el recorrido por el grafo con el tiempo de cada nodo, si el código forzó la ruta y por qué, la consulta condensada y los filtros, cuántas preguntas aclaratorias se han usado, los pasos del agente con los libros que devolvió cada llamada a una herramienta, y el uso del LLM.
- Un botón «Nueva conversación».
- Sin LLM, o con el presupuesto diario agotado, la página explica en español que el asistente ha hecho una búsqueda simple del último mensaje.

## Detalles del servidor añadidos para la página

### Posición de cada resultado en cada etapa (`d0bab6b`)

`SearchResult` lleva ahora tres campos, `lexical_rank`, `vector_rank` y `rerank_rank` (posición empezando en 1; ausente cuando la etapa no devolvió el libro o no se ejecutó). Los rellena el servicio de dominio (`SearchService`) en las dos rutas de fusión, en el respaldo de solo palabras clave y en el paso de reordenación; con varias variantes de la consulta se conserva la mejor posición del libro. Los cuatro adaptadores de búsqueda (BM25, FAISS, palabras clave en Postgres y pgvector) ya numeraban sus propias listas, de modo que los dos perfiles, local y nube, reciben las posiciones sin tocar los adaptadores. Se exponen en el esquema y el conversor de la API.

Antes de este cambio la página ordenaba cada etapa entre los resultados que había en pantalla y tenía que avisar de que las posiciones eran aproximadas.

### Libros devueltos por cada llamada del agente (`e745a95`)

Cada paso del agente (`AgentStep`) lista ahora los identificadores y títulos de los libros que devolvió su herramienta, leídos de la salida que el agente ya interpretaba; `result_count` pasa a ser su número. Se expone por `/agent/ask` y por `/assistant/chat` con un solo conversor.

### Texto de explicación cortado (`ede1d70`)

El síntoma era un texto de explicación que terminaba en algo como «... Learning' [C». La causa, hallada en el código:

- El campo `reasoning` del esquema de la explicación conservaba los primeros 500 caracteres. Ese corte podía caer dentro de una marca de cita en línea, `[CITE: campo]`, que el modelo escribe para señalar su fuente.
- Las marcas además se guardaban en el texto, y la página solo quitaba las marcas completas, de modo que una marca cortada («[C») o la variante «(CITE:» que a veces escribe el modelo seguían visibles.

La corrección está en el origen. Un módulo nuevo, `app/infrastructure/llm/explanation_text.py`, corta en el último final de frase o, si no es posible, en un límite de palabra con puntos suspensivos, sin cortar nunca dentro de un paréntesis o corchete abierto, y elimina toda marca, completa o cortada. El texto guardado ya no contiene marcas y las citas verificadas se siguen devolviendo por separado. La limpieza propia de la página se retiró.

### `make demo` y el LLM (`4be8a66`)

El `Makefile` y el `README.md` decían que la demo no se veía afectada por un `.env` local, pero la aplicación carga `.env`, de modo que en una máquina con una clave de OpenRouter la demo arrancaba con el LLM encendido. Ahora `make demo` fija `LLM_PROVIDER=none` y no usa el LLM en ninguna máquina; `make demo LLM=on` fija `openrouter` y toma `OPENROUTER_API_KEY` del entorno o del `.env`; cualquier otro valor de `LLM` es un error. Sin clave, `LLM=on` arranca igual con el LLM apagado. El `README.md` también describe ahora el asistente. La nota V09 describía el comportamiento anterior (la clave del `.env` activaba el LLM en la demo); este es el comportamiento actual.

## Cómo se ha comprobado

Todas las comprobaciones son una ejecución en un navegador (Chrome) y un portátil, no una distribución ni una prueba con usuarios.

**Primer paso (Chrome sin pantalla):**
- Anchos de 1280 px y 390 px, en modo claro y oscuro.
- Orden del teclado, flechas en las pestañas, apertura del cuadro de detalle con Intro y cierre con Escape con el foco devuelto a su sitio.
- Sin desplazamiento horizontal a 360 px ni a 1920 px.
- Desplazamiento de diseño no causado por el clic del usuario: unos 0.002 por búsqueda.
- Contraste calculado a partir de los valores de color: texto de al menos 5.8:1 y bordes de controles de al menos 3.4:1, en los dos modos.
- Dos peticiones a un LLM real: una búsqueda con interpretación y tres explicaciones, y una pregunta al agente con 6 llamadas a herramientas.

**Segundo paso (Chrome sin pantalla, dirigido por un pequeño guion del protocolo de herramientas de desarrollo; perfil local sobre la colección de 177 libros):**
- Las cuatro rutas, en vivo: dos preguntas aclaratorias con el tope manteniéndose en 2 de 2; una búsqueda con una recomendación y 2 citas verificadas; una respuesta del agente en 12.5 s con tres llamadas a herramientas, cada una con la lista de sus libros; y una respuesta fuera de ámbito.
- El foco permanece en el cuadro de mensaje tras elegir una opción sugerida.
- Sin LLM, una búsqueda simple con su aviso.
- Un 429 real, obtenido reiniciando el servidor con un límite de 2 peticiones por minuto.
- La cancelación devuelve el texto al cuadro.
- Posiciones en vivo para «Recetas de bacalao»: no está en la lista de palabras clave, es el 19.º por significado y el 5.º tras el reordenador.
- Coste: 9 peticiones al asistente y 0.0040 USD para las 7 cuyo uso se registró.

**Tests:** 1744 superados y 22 omitidos tras unirlo todo (los omitidos son las pruebas de Postgres, que necesitan una base de datos). Los tests de la página (`tests/api/test_ui.py`) comprueban, entre otras cosas, que no se inserta nada como HTML, que no se carga nada de otros servidores, que cada control tiene etiqueta, y que los límites y los nombres de nodo de la página coinciden con los del esquema y el código del grafo del servidor.

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| Un marco como React | Exige paso de compilación y dependencias, y complica explicar el código en la memoria; los tres ficheros estáticos se despliegan tal cual |
| Fuentes externas o bibliotecas desde un CDN | Cada visita enviaría una petición a un tercero; la página no debe enviar datos de los visitantes a ningún sitio. Se usan fuentes del sistema |
| Guardar la conversación en el almacenamiento del navegador | Conservaría en el equipo del visitante conversaciones que el proyecto no necesita; contradice el diseño sin estado del asistente |
| Guardar la conversación en el servidor | Exigiría almacenamiento y datos de usuarios (véase V20: se descartó el guardado de estado en el servidor) |
| Descartar los turnos más antiguos al llegar al límite | Un servidor sin estado cuenta las preguntas aclaratorias sobre lo que recibe; si la página recortara turnos, el contador de preguntas volvería a empezar y el tope de 2 dejaría de cumplirse. A los 50 mensajes la página pide empezar una conversación nueva |
| Quitar las marcas de cita solo en la página | Dejaba el texto guardado con marcas y con cortes, y cualquier otro cliente de la API recibiría el texto con marcas; además la limpieza de la página solo reconocía las marcas completas |
| Calcular en la página posiciones aproximadas de cada etapa | Ordenaba cada etapa solo entre los resultados en pantalla, no en la lista completa de la etapa; la página debía avisar de que las posiciones eran aproximadas. El servidor ya conoce las posiciones reales |
| Reutilizar los repositorios que señaló el autor | Ninguno encajaba: eran para páginas de venta o de efecto visual, no para mostrar un sistema de búsqueda |

## Dónde la construcción se apartó del encargo

1. **Límite de 50 mensajes.** En lugar de descartar turnos antiguos, la página pide una conversación nueva (véase arriba).
2. **Cuadro de mensaje de 500 caracteres.** La API acepta hasta 5000 por mensaje, pero el asistente solo lee los primeros 500 de cada mensaje (véase V20), así que la página se detiene en 500 para que el usuario no escriba texto que el asistente no va a leer.

## Límites que debe conocer el tribunal

- **La página no está desplegada en el sitio público** ni se ha ejecutado con el perfil nube ni con la base de datos Neon.
- **El mensaje de presupuesto diario agotado no se ha provocado de verdad**; una captura de pantalla de ese mensaje es simulada. Los mensajes de 503, 422 y 400 proceden de respuestas simuladas, no de fallos reales del servidor.
- **Solo se probó Chrome**, con un navegador sin pantalla y un portátil. No se ha usado un lector de pantalla real, así que el anuncio de mensajes a tecnologías de apoyo se comprobó en el código y en los atributos, no escuchándolo.
- **El contraste de color se calculó en el primer paso** y no se ha vuelto a medir para los estilos añadidos en el segundo.
- **`make demo` no se ha ejecutado después del cambio.** Se comprobó `make -n demo` con los tres valores de `LLM` (`off`, `on` y uno no válido) y se arrancó el servidor con la misma configuración.
- Las capturas de pantalla se hicieron pero no están en el repositorio.
- La calidad del asistente no se ha medido (véase V20): esta tarea muestra el recorrido del asistente, no demuestra que sus recomendaciones sean mejores que las de la búsqueda simple.
- Todas las cifras de la sección de comprobaciones son de una ejecución en un portátil; el 0.002 de desplazamiento de diseño y los 12.5 s de la respuesta del agente no son medias.

## Trabajo pendiente

- Desplegar la página en el sitio público: queda a la espera de que el autor dé el visto bueno.
- Evaluar el agente y el asistente con un protocolo declarado de antemano (tarea V22).
- Medir de nuevo el contraste de los estilos añadidos en el segundo paso, y probar con un lector de pantalla real y con otros navegadores.
- Probar la página con el perfil nube.

## Cómo reproducirlo

```
make demo                 # http://127.0.0.1:8000/ui/ ; LLM apagado en toda máquina; Ctrl+C lo detiene
make demo LLM=on          # con el LLM de OpenRouter (OPENROUTER_API_KEY del entorno o del .env, que no se versiona)
make test                 # 1744 superados, 22 omitidos tras unirlo todo

pytest tests/api/test_ui.py tests/domain/test_stage_positions.py \
       tests/infrastructure/llm/test_explanation_text.py \
       tests/infrastructure/llm/test_search_agent.py tests/api/test_agent_api.py
```

Con el LLM apagado, la pestaña del asistente responde a cada mensaje con una búsqueda simple y lo avisa; con `LLM=on` y una clave, pregunta, recomienda y enruta al agente. Para ver el recorrido, abrir la línea plegada que hay bajo cada respuesta.

## Ficheros

- **Aplicación (página):** `app/ui/static/index.html`, `app.js`, `styles.css`.
- **Aplicación (API):** `app/api/v1/schemas.py` y `converters.py` (posiciones por etapa y libros de cada paso del agente), `app/api/v1/agent_endpoints.py`.
- **Dominio:** `app/domain/entities.py` (`SearchResult.lexical_rank`, `vector_rank`, `rerank_rank`; `AgentStep.books`, `AgentStepBook`), `app/domain/services.py` (relleno de las posiciones).
- **Infraestructura:** `app/infrastructure/llm/explanation_text.py` (nuevo), `guardrails.py`, `schemas.py`, `agents/search_agent.py`.
- **Empaquetado y documentación:** `Makefile`, `README.md`.
- **Tests:** `tests/api/test_ui.py`, `tests/domain/test_stage_positions.py` (nuevo), `tests/infrastructure/llm/test_explanation_text.py` (nuevo), `tests/infrastructure/llm/test_search_agent.py`, `tests/api/test_agent_api.py`, `tests/api/test_assistant_api.py`.
