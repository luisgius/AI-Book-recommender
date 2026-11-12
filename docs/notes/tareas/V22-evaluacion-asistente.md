# V22 · Evaluación del asistente conversacional y del agente con herramientas

- **Fecha:** 2026-11-12
- **Estado:** hecha; los juicios son solo de LLM hasta que el autor rellene la hoja de anotación humana; de los dos umbrales declarados, uno se cumple y el otro no, y la revisión de la instrucción de ruta (tarea V25) está declarada pero todavía no hecha
- **Capa:** evaluación (`app/evaluation/assistant_evaluation/`), datos (`data/evaluation/assistant/`), documentación (`docs/notes/`)
- **Commits:** protocolo en `0ccd5f2` (antes de cualquier resultado); implementación y resultados en `7e91bf1`; adenda con el siguiente paso en `0a63751` (después de los resultados y antes de cambiar nada)

## Qué se ha hecho

- Se escribió y se guardó en git el protocolo (`docs/notes/2026-11-12-protocolo-evaluacion-asistente.md`) antes de ejecutar nada. Después se midió el asistente de V20 (el grafo de LangGraph que elige entre preguntar, buscar, pasar al agente o rechazar) y el agente con herramientas sobre el catálogo desplegado (esquema `catalog_deploy` de Neon, 50,000 obras), con cinco conjuntos de peticiones, dos jueces LLM, tres contrastes declarados con corrección de Holm y los umbrales de la sección 7 del protocolo.
- El sistema se llamó a través del puerto `ConversationalAssistant` con los adaptadores del perfil nube, sin HTTP, sin la URL pública y sin presupuesto diario. La referencia, que el protocolo llama P, es la recuperación desplegada (vectorial más reranker) sobre el texto de la petición, sin LLM.
- El informe `data/evaluation/ASSISTANT_EVALUATION.md` se regenera sin conexión a partir de las ejecuciones y los juicios guardados.
- Código nuevo en `app/evaluation/assistant_evaluation/`, un módulo por etapa: `design.py` (constantes fijadas por el protocolo), `sets.py` y `generation.py` (conjuntos de peticiones y su generación), `simulated_user.py` (usuario simulado), `system.py` (construcción del sistema evaluado), `runs.py` (ejecuciones), `judging.py` (juicios), `analysis.py` (métricas y contrastes), `results.py`, `report.py` y `cli.py`.
- No se modificó el asistente, el agente ni la base de datos. La conexión a Neon ejecutó cada sentencia con `transaction_read_only=on`.

## Por qué

V20 dejó el asistente construido, con tests y con una comprobación en vivo de diez turnos, pero declaró expresamente que su calidad no se había medido. El autor pidió medirlo el 2026-11-12. El protocolo parte de tres preguntas sin respuesta:

1. Si pasar por el asistente empeora o mejora los resultados de una petición que ya era clara, puesto que el asistente reescribe la consulta antes de buscar.
2. Si preguntar antes de recomendar mejora los resultados cuando la petición es vaga, que es la razón de ser de la pregunta aclaratoria.
3. Si el asistente elige bien la ruta: cuándo pregunta sin necesidad, cuándo rechaza una petición válida y cuándo reconoce una que no puede atender.

## Objetivo

Responder con datos, con reglas fijadas de antemano, a esas tres preguntas; medir además el agente en peticiones de varios pasos; y dejar los conjuntos de peticiones, las ejecuciones y los juicios guardados para repetir el análisis sin llamar a ninguna API.

## Diseño fijado de antemano

El protocolo se guardó en git (`0ccd5f2`) antes de generar una petición o ejecutar un turno. Importa porque los resultados no pudieron elegir las preguntas: los conjuntos, los tres contrastes, las medidas, los dos umbrales y lo que se haría con cada resultado estaban escritos antes de que existiera ningún número.

- **Conjuntos de peticiones.**

| Conjunto | Peticiones | Origen | Ruta esperada |
|---|---|---|---|
| A | 44 | Estrato A de V14, sin cambios (10 de ellas de la categoría `ambiguous`, que se informan aparte) | Búsqueda |
| B | 72 | Estrato B de V14, sin cambios | Búsqueda |
| D | 12 | Estrato D de V14 (peticiones que un catálogo de libros no puede atender) | Fuera de alcance |
| E | 30 | Nuevo: treinta necesidades de B elegidas al azar (semilla 0); de cada una, el generador escribe una petición vaga de una a tres palabras en el mismo idioma, y la necesidad original queda oculta | Pregunta aclaratoria |
| F | 20 | Nuevo: peticiones de varios pasos sobre obras que están en el catálogo (comparar dos obras, «algo como X pero...», datos de una obra concreta), escritas a partir de fichas del catálogo y sin ver resultados del sistema | Agente |

- **Usuario simulado para E.** Un LLM de otro proveedor que el LLM del sistema recibe la necesidad oculta, la pregunta del asistente y sus opciones, y responde con una frase corta coherente con la necesidad (elige una opción si alguna encaja; si no, escribe la suya). No ve resultados. La conversación sigue hasta que el asistente devuelve libros.
- **Juicios.** Los de V14 para las necesidades de A, B y D. Los pares nuevos (resultados y libros recomendados sin juzgar, y todo E y F) los juzgan los dos jueces de V14, `z-ai/glm-5.3-flash` (principal) y `minimax/minimax-m3.1-flash` (segundo), con la misma instrucción y en ficheros aparte. En E se juzga contra la necesidad oculta; en F, contra la petición.
- **Medidas.** Ruta (matriz por conjunto, tasa de preguntas innecesarias y de «fuera de alcance» indebido en A y B, tasa de detección en D, rutas forzadas por código, coincidencia de la ruta al repetir A y B); nDCG@10 de la lista que devuelve el asistente (calidad de las diez primeras posiciones, con más peso arriba); precisión de las recomendaciones (proporción de libros recomendados con nota 2 o 3) frente a la de P en el mismo número de puestos; anclaje (recomendaciones y citas que descarta el nodo de verificación, y libros del agente que no salieron de sus herramientas); idioma de la respuesta; coste y tiempo por turno y ruta.
- **Tres contrastes** (familia única, bilateral, corrección de Holm, con el juez principal y repetida con el segundo; un contraste compara una misma medida entre dos sistemas sobre las mismas peticiones, y la corrección de Holm ajusta los p de varios contrastes a la vez). Una conclusión se sostiene solo si tiene el mismo signo y es significativa con los dos jueces.

| Id | Contraste | Conjunto | Pregunta |
|---|---|---|---|
| H7 | nDCG@10 del asistente frente a P | A y B | ¿Cambia la calidad al pasar por el asistente una petición clara? |
| H8 | nDCG@10 del asistente tras aclarar frente a P sobre la petición vaga | E | ¿Mejora preguntar? |
| H9 | Precisión de las recomendaciones del agente frente a P en los mismos puestos | F | ¿Recomienda el agente mejor que la recuperación sola? |

- **Descriptivo, sin contraste:** en E, P sobre la petición original de B como techo de referencia y el número de turnos; en una muestra de 40 peticiones de A y B (semilla 0), el agente llamado directamente.
- **Qué se haría con cada resultado (sección 7 del protocolo).**
  - Si H7 es negativo y se sostiene, la ruta de búsqueda usaría el texto del usuario, y no la consulta reescrita, cuando la conversación tenga un solo turno (tarea aparte).
  - Si H8 no se sostiene, el TFG diría que la pregunta aclaratoria no mostró mejora medible con este método, sin más.
  - **Dos umbrales previos** a presentar el asistente como entrada principal de la página pública: «fuera de alcance» indebido en A y B no superior al 2 %, y preguntas innecesarias en las peticiones de B no superiores al 10 %. Si no se cumplen, se revisan las instrucciones del nodo de ruta y se repite la medida de ruta, declarando el cambio.
  - H9 no cambia el sistema: informa de cuándo compensa el coste del agente.
- **Amenazas a la validez declaradas:** el usuario de E es simulado, así que la mejora medida en H8 es un límite favorable; las peticiones de E y F las escribe un LLM; los jueces son LLM; el asistente no es determinista, por lo que la ruta se repite una vez para medirlo y la calidad no.
- **Tope de gasto:** 4 USD en OpenRouter.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Medir a través de la URL pública | Gastaría el presupuesto diario de los visitantes; el protocolo exige llamar a los adaptadores del perfil nube directamente |
| Un usuario real (una persona) para las conversaciones de E | El protocolo fija un usuario simulado y declara su límite en lugar de ocultarlo: una persona no responde siempre de forma breve y coherente, así que la mejora de H8 es un límite favorable y no una medida del uso real |
| Puntuar la pregunta aclaratoria como si fuera una lista de resultados | Los jueces califican un libro contra una necesidad, no una pregunta. La lista que da una respuesta es la lista de búsqueda en la ruta de búsqueda, las recomendaciones en orden en la ruta del agente, y ninguna (nDCG@10 = 0) en las rutas de pregunta y fuera de alcance, porque el usuario no recibió libros en ese turno. En E, la calidad se mide tras la conversación, con la lista que se da cuando el asistente por fin busca |
| Dejar las peticiones de E sin comprobar que no regalan la necesidad | Una petición vaga que repite las palabras de la necesidad oculta no sería vaga; se fijaron comprobaciones mecánicas antes de ejecutar ningún sistema |
| Aplicar a H7 la regla de la sección 7 (buscar con el texto del usuario) | La propia medida muestra que la pérdida no viene de la reescritura (véase más abajo): esa regla no quitaría la pérdida |
| Un juez único | Como en V14, una conclusión solo se sostiene si los dos jueces coinciden en signo y significación |

## Resultados

Todas las cifras son las del informe `data/evaluation/ASSISTANT_EVALUATION.md`. Los juicios son de LLM; juez principal `z-ai/glm-5.3-flash`, segundo `minimax/minimax-m3.1-flash`. Las ejecuciones se hicieron el 2026-11-12 entre las 15:45:26 y las 15:55:05, con el código del commit `36802ee`. Modelos, todos por OpenRouter: LLM del sistema `deepseek/deepseek-v4.1-flash` (temperatura 0.0, razonamiento desactivado); embeddings `qwen/qwen3-embedding-8b`; reranker `qwen/qwen3-reranker-8b`; generador de peticiones `google/gemini-3.8-flash`; usuario simulado `mistralai/mistral-small-2603`.

### Los conjuntos de peticiones y cómo se generaron

E y F las escribió `google/gemini-3.8-flash` (temperatura 0.7, razonamiento bajo, versión de prompt v1 para ambos). Los prompts, las salidas y las fechas están en `data/evaluation/assistant/sets/generation/`. A, B y D son los de V14.

**Conjunto E.** La primera pasada produjo 10 problemas detectados por las comprobaciones mecánicas, y hubo 12 sustituciones en hasta tres rondas; al final quedaron 0 problemas. Las comprobaciones que el protocolo no detallaba y que se fijaron en esta tarea: como máximo una palabra de contenido compartida con la consulta original de B, como máximo tres palabras (ocho caracteres en chino y japonés), que no sea la consulta original y, en las necesidades de obra conocida, ni una palabra del título ni el apellido del autor. Dos cosas tuvieron que corregirse:

- La comprobación de palabras compartidas se **amplió después de la primera pasada y antes de cualquier ejecución del sistema**: pasó a contar también las formas flexionadas (mismas cinco primeras letras), porque `славянская мифология` había pasado como petición vaga para una necesidad sobre mitología eslava. Las llamadas del generador se reutilizaron desde la caché y solo se pidieron las sustituciones nuevas. Las comprobaciones de la primera pasada se conservan en `sets/checks_before_prefix_rule.json`.
- La comprobación de idioma heredada de V14 lee el artículo elidido francés (`d'`) como catalán, de modo que tres peticiones francesas correctas (`récits d'aventure`, `roman d'anticipation`, `livre d'histoire`) se devolvieron al generador y se sustituyeron. Las sustituciones también son francesas y son las que se evaluaron.
- Veredictos de idioma del conjunto final: 16 correctos y 14 indecisos (peticiones cortas sin palabras funcionales).

**Conjunto F.** 0 problemas tras la primera pasada y 0 sustituciones; veredictos de idioma: 20 correctos. El diseño de F lo fijó la implementación, no el protocolo: 7 comparaciones de dos obras, 7 peticiones «como X pero...» y 6 preguntas sobre una obra. La primera obra de cada tarea se sacó con semilla 0 entre las obras del catálogo en los doce idiomas de B con una descripción de al menos 300 caracteres; la segunda obra de una comparación comparte autor con la primera (si no, materia, si no, idioma); cada petición se escribe en el idioma del catálogo de la primera obra. Las peticiones de F nombran las obras exactamente con sus títulos del catálogo.

### Pregunta 1: pasar por el asistente empeora una petición clara (H7), y la pérdida viene de la ruta

| Id | Juez | n | Asistente | P | Diferencia [IC 95 %] | p Holm |
|---|---|---|---|---|---|---|
| H7 | principal | 116 | 0.653 | 0.790 | -0.137 [-0.194, -0.084] | <0.001 |
| H7 | segundo | 116 | 0.674 | 0.801 | -0.127 [-0.182, -0.076] | <0.001 |

Resultado: **sostenido** (mismo signo y significativo con los dos jueces). El asistente es peor que la recuperación sola sobre las 116 peticiones de A y B.

La causa no es la consulta reescrita. Solo sobre las 96 peticiones que el asistente buscó (descriptivo), con el juez principal: asistente 0.789 y P 0.789, diferencia +0.000 [-0.003, +0.003], p sin ajustar 0.915. La consulta condensada difirió del texto de la petición en 14 de ellas (cambió el nDCG@10 en 12, diferencia media +0.001) y fue el texto mismo en 82. Con el segundo juez: 0.814 frente a 0.814, diferencia +0.001 [-0.003, +0.005]. Toda la pérdida está en las peticiones que fueron a una pregunta (9) o a «fuera de alcance» (11), que no devuelven libros y puntúan 0.000:

| Grupo (juez principal) | Peticiones | Asistente | P |
|---|---|---|---|
| A `ambiguous` | 10 | 0.091 | 0.769 |
| A otras | 34 | 0.562 | 0.775 |
| B | 72 | 0.774 | 0.800 |
| ruta pregunta | 9 | 0.000 | 0.761 |
| ruta fuera de alcance | 11 | 0.000 | 0.825 |
| ruta búsqueda | 96 | 0.789 | 0.789 |

Consecuencia sobre la regla declarada: la sección 7 mandaba, si H7 era negativo, usar el texto del usuario y no la consulta reescrita. El informe señala que eso **no quitaría la pérdida**; la que la aborda es la revisión de la ruta ligada a los umbrales (adenda, más abajo). Sobre las recomendaciones de la ruta de búsqueda en A y B: 83 respuestas con recomendaciones, precisión (nota 2 o 3) 0.898 frente a 0.793 de P en los mismos puestos con el juez principal, y 0.942 frente a 0.865 con el segundo (descriptivo).

### Pregunta 2: preguntar mejora las peticiones vagas (H8), con un usuario simulado

| Id | Juez | n | Asistente tras la conversación | P sobre la petición vaga | Diferencia [IC 95 %] | p Holm |
|---|---|---|---|---|---|---|
| H8 | principal | 30 | 0.338 | 0.129 | +0.209 [+0.104, +0.326] | 0.002 |
| H8 | segundo | 30 | 0.351 | 0.153 | +0.198 [+0.095, +0.311] | 0.002 |

Resultado: **sostenido**. El asistente preguntó en 14 de las 30 conversaciones (46.7 %, con los dos jueces; 16 conversaciones con 0 preguntas y 14 con 1; todas terminaron respondidas, con ruta final de búsqueda). Descomponiendo, nDCG@10 con el juez principal (entre paréntesis, el segundo):

| Subconjunto | n | Asistente tras la conversación | P sobre la petición vaga | P sobre la petición original de B (referencia) |
|---|---|---|---|---|
| todas | 30 | 0.338 (0.351) | 0.129 (0.153) | 0.775 (0.790) |
| preguntó | 14 | 0.540 (0.537) | 0.091 (0.112) | 0.787 (0.768) |
| no preguntó | 16 | 0.161 (0.188) | 0.161 (0.189) | 0.764 (0.809) |

Dicho sin suavizar: la mejora viene entera de las conversaciones en que preguntó. Donde no preguntó, el asistente busca la petición vaga tal cual y obtiene lo mismo que P (0.161 frente a 0.161 con el juez principal). Y aun tras preguntar (0.540), el asistente queda lejos de P sobre la petición original de B (0.787): la respuesta corta del usuario simulado recupera una parte de la necesidad, no toda.

**Advertencia del propio protocolo, que se repite aquí:** el usuario es simulado y responde siempre de forma breve y coherente con la necesidad oculta, cosa que una persona no garantiza. Por tanto la mejora de H8 es un **límite favorable**, no una estimación de lo que ocurriría con usuarios reales.

### Pregunta 3: las rutas

Matriz de rutas de la primera ejecución (run 1):

| Conjunto | pregunta | búsqueda | agente | fuera de alcance | error | total |
|---|---|---|---|---|---|---|
| A `ambiguous` | 8 | 1 | 0 | 1 | 0 | 10 |
| A otras | 0 | 25 | 0 | 9 | 0 | 34 |
| B | 1 | 70 | 0 | 1 | 0 | 72 |
| D | 0 | 0 | 0 | 12 | 0 | 12 |
| E (primer turno) | 14 | 16 | 0 | 0 | 0 | 30 |
| F | 0 | 1 | 19 | 0 | 0 | 20 |

**Umbrales de la sección 7, aplicados a la primera ejecución de A y B** (la segunda se muestra al lado):

| Medida | Primera ejecución | Segunda ejecución | Máximo | Resultado |
|---|---|---|---|---|
| «Fuera de alcance» indebido en A y B | 11/116 = 9.5 % | 9/116 = 7.8 % | 2.0 % | **no se cumple** |
| Preguntas innecesarias en B | 1/72 = 1.4 % | 0/72 = 0.0 % | 10.0 % | **se cumple** |

El informe resume el conjunto como «no cumplido», porque la condición exige las dos cosas a la vez; el umbral de preguntas sí se cumple y el de «fuera de alcance» indebido no.

- **Peticiones rechazadas por error** (primera ejecución, 11): diez de A (`q03` recetas de cocina rumana, `q25` recetas de cocina mediterránea, `q27` postres caseros, `q28` recetas de pasta italiana, `q31` recetas de mariscos y pescados, `q32` cocina asiática con wok, `q33` comida mexicana, `q37` recetas fáciles, y de informática `q11` guía de visión por computador con OpenCV y `q14` explicación de modelos de lenguaje) y una de B, `b-fr-6` (una escena de *El extranjero* descrita sin su título). Son sobre todo de cocina y recetas, más algunas de informática. Los diez de A son peticiones de libros sobre un tema que el modelo tomó por una petición de otra cosa.
- **Preguntas en A y B:** 9 de 116 (7.8 %) en la primera ejecución; 8 de las 10 peticiones `ambiguous` de A recibieron una pregunta, y una sola petición de B (`b-de-5`, un ensayo filosófico sobre la responsabilidad moral de la ciencia) recibió una.
- **Detección en D:** 12 de 12 fuera de alcance. No hay umbral declarado para esta medida.
- **Agente en A y B:** 0 de 116 (no se pasó al agente ninguna petición de búsqueda).
- **Rutas forzadas por código** (`invalid_clarification` a búsqueda: aclaración sin pregunta o con menos de dos opciones): 3 en E en la primera ejecución y 2 en A en la segunda. Los segundos turnos de E fueron todos búsqueda (14).
- **Estabilidad de la ruta** al repetir A y B: misma ruta en 106 de 116 peticiones (91.4 %); por conjunto, A 36 de 44 y B 70 de 72. Cambios: «fuera de alcance» a búsqueda 4, búsqueda a «fuera de alcance» 2, pregunta a búsqueda 3, búsqueda a pregunta 1 (`b-de-5`, `b-fr-6`, `q02`, `q11`, `q12`, `q13`, `q25`, `q32`, `q40`, `q41`).
- **Qué dice la sección 7 cuando un umbral no se cumple:** que se revisan las instrucciones del nodo de ruta y se repite la medida de ruta, declarando el cambio, antes de que el asistente sea la entrada principal de la página pública. Esta tarea no cambió nada; eso es el paso siguiente (adenda, más abajo). Hasta entonces, el asistente no cumple la condición para presentarse como entrada principal.

### El agente

**En peticiones de varios pasos (F, H9).**

| Id | Juez | n | Agente (asistente) | P en los mismos puestos | Diferencia [IC 95 %] | p Holm |
|---|---|---|---|---|---|---|
| H9 | principal | 20 | 0.925 | 0.795 | +0.130 [+0.025, +0.260] | 0.059 |
| H9 | segundo | 20 | 0.948 | 0.853 | +0.095 [-0.020, +0.230] | 0.186 |

Resultado: **no significativo**. Con el juez principal el intervalo no incluye el cero, pero el p con Holm es 0.059, y con el segundo juez el intervalo sí lo incluye (Holm 0.186). Con 20 peticiones no se puede afirmar que el agente recomiende mejor que la recuperación sola; la sección 7 dice que H9 no cambia el sistema y solo informa del coste.

Descriptivamente, con el juez principal: 19 de las 20 peticiones fueron a la ruta del agente y 1 a búsqueda; las 20 respuestas incluyeron recomendaciones. Precisión del agente 0.921 frente a 0.837 de P en las 19 de la ruta del agente; por tipo, comparar (7) 0.914 frente a 0.771, «como X pero...» (7) 0.871 frente a 0.643, datos de una obra (6) 1.000 frente a 1.000. De las obras que las peticiones nombraban, se recomendaron 20 de 27 (P las tenía en sus primeros puestos en 24). El nDCG@10 de la lista de la respuesta es 0.675 frente a 0.856 de P. El informe no explica esa diferencia.

**Llamado directamente** (40 peticiones de A y B, semilla 0): con el juez principal, 39 de 40 ejecuciones con recomendaciones (0 fallos), precisión 0.795 [0.687, 0.887] frente a 0.698 [0.587, 0.799] de P en los mismos puestos; en las 29 peticiones en que también recomendó el asistente, agente 0.890 y asistente 0.908. Con el segundo juez: 0.781 [0.670, 0.879] frente a 0.753 [0.638, 0.856], y 0.903 frente a 0.948. Sobre esas 40 peticiones el asistente tomó la ruta de búsqueda en 35 y la de «fuera de alcance» en 5. Son descriptivos, sin contraste.

### Anclaje, idioma, coste, tiempo y fallos

**Anclaje.** En la ruta de búsqueda (primera ejecución y todos los turnos de E), el nodo de verificación recibió 312 recomendaciones propuestas, conservó 303 y descartó 9, todas por no tener ninguna cita válida; de 617 citas propuestas, 596 pertenecen a recomendaciones conservadas. En 4 respuestas se descartaron todas las recomendaciones y una respuesta no tuvo salida de la llamada de redacción. En la ruta del agente dentro del asistente: 19 ejecuciones, 52 recomendaciones conservadas y **0 obras recomendadas que ninguna herramienta hubiera devuelto**. El agente llamado directamente: 40 ejecuciones, 182 recomendaciones y también 0 obras fuera de sus herramientas (4 ejecuciones necesitaron el paso de reflexión). El agente no recomendó nada que sus herramientas no hubieran devuelto.

**Idioma de la respuesta** (detector de idioma del catálogo, doce idiomas, 40 letras o más; un texto que no decide se juzga por su escritura en ruso, chino, japonés y árabe, y si no queda «indeciso»). Las respuestas fijas (fuera de alcance, sin resultados, resultados sin recomendaciones) existen solo en español e inglés. Texto escrito por el LLM: 128 correctas, 7 en otro idioma y 19 indecisas. Respuestas fijas: 15 correctas, 14 en otro idioma y 9 indecisas. Por ruta, la de «fuera de alcance» es la peor (5 correctas, 11 en otro idioma, 7 indecisas), porque su texto fijo está en español o inglés. Los desajustes son casi siempre respuestas en inglés a peticiones en otro idioma (por ejemplo neerlandés a inglés, 4).

**Latencia y coste por ruta** (tiempo de pared de `reply()` desde un portátil, cuatro peticiones en curso; coste notificado por OpenRouter para las llamadas del hilo de la petición):

| Ruta | Turnos | Segundos mediana / p95 | USD mediana / p95 | Llamadas al LLM mediana / p95 |
|---|---|---|---|---|
| agente | 19 | 10.4 / 16.3 | 0.00516 / 0.00859 | 5.0 / 5.1 |
| pregunta | 30 | 1.3 / 2.3 | 0.00015 / 0.00024 | 1.0 / 1.0 |
| fuera de alcance | 32 | 0.9 / 1.5 | 0.00008 / 0.00047 | 1.0 / 1.0 |
| búsqueda | 227 | 5.4 / 9.5 | 0.00264 / 0.00348 | 2.0 / 2.0 |
| todos los turnos | 308 | 5.1 / 10.3 | 0.00248 / 0.00399 | 2.0 / 5.0 |
| agente directo | 40 | 11.5 / 18.6 | 0.00559 / 0.01138 | 4.0 / 6.0 |

Mediana de segundos por nodo en la ruta de búsqueda: ruta 1.06, recuperación 1.13, respuesta 2.99 y verificación 0.00; en la del agente: ruta 1.08 y agente 9.22.

**Fallos.** 768 llamadas al LLM del sistema, 5 fallidas; ninguna llamada de embedding ni de reranker falló. Los 5 fallos son la llamada que redacta la recomendación en la ruta de búsqueda llegando a su tope de 1,024 tokens (`LengthFinishReasonError`); en 4 de los 5, el proveedor contó entre 773 y 1,024 tokens como razonamiento aunque el razonamiento está desactivado. Cuando ocurre, la respuesta conserva los resultados y descarta las recomendaciones (aviso `assistant_answer_failed`, 5 veces). No hubo turnos sin respuesta, ni búsquedas degradadas, ni errores del enrutador. Además, 26 peticiones de embedding con respaldo (el proveedor alternativo respondió antes) se cancelaron; no son fallos. El juez principal tuvo 4 llamadas fallidas reintentadas en una pasada posterior.

### Juicios

136 necesidades con pares medidos; 1,826 pares por juez, de los que 1,220 vienen de V14 y 606 se juzgaron aquí; 0 sin juzgar con los dos jueces. Los nDCG@10 de este informe se calculan contra la unión del pool de V14 y los pares juzgados aquí, de modo que **no son comparables con las cifras de V14**, cuyo pool es otro; solo se comparan dentro de este informe.

### Coste y tiempo

Gasto real notificado por OpenRouter: **1.3980 USD en 2,918 llamadas** (tope 4.00 USD). El contador acumulado de la clave se movió 1.4033 USD entre la primera y la última lectura, e incluye cualquier otro uso de la clave en esa ventana.

| Etapa | Llamadas | Fallidas | USD | Segundos |
|---|---|---|---|---|
| 1-queries | 14 | 0 | 0.0428 | 55.6 |
| 2-reference | 357 | 25 | 0.2824 | 125.6 |
| 3-assistant | 1331 | 31 | 0.9655 | 576.8 |
| 4-judge | 1216 | 4 | 0.1072 | 123.8 |

Las 25 y 31 «fallidas» de las etapas 2 y 3 incluyen las peticiones de embedding con respaldo canceladas (25 y 26); las 5 restantes de la etapa 3 son las de redacción. El gasto mayor es el reranker (`qwen/qwen3-reranker-8b`, 528 llamadas, 0.8981 USD de los 1.3980); el LLM del sistema cuesta 0.3488 USD en 768 llamadas.

## Desviaciones respecto al protocolo

El informe las lista todas con su motivo.

1. **Usuario simulado.** El protocolo pide un LLM de otro proveedor que el LLM del sistema; se usó `mistralai/mistral-small-2603` (proveedor que no se usa en ninguna otra parte de la evaluación), temperatura 0.0, con un prompt que ve la necesidad oculta, la conversación y la última pregunta con sus opciones, nunca un resultado. Una respuesta vacía se habría sustituido por la opción elegida o la primera opción (se registra por llamada).
2. **Comprobaciones de E.** El protocolo nombra el generador y el tamaño de las peticiones, pero no las comprobaciones de «no regala la necesidad»; se fijaron aquí (véase «Los conjuntos de peticiones») y la regla de palabras compartidas se amplió a formas flexionadas tras la primera pasada y antes de cualquier ejecución.
3. **Comprobación de idioma de V14** (lee `d'` como catalán): tres peticiones francesas correctas de E se sustituyeron por otras francesas.
4. **Diseño de F**, no fijado por el protocolo: 7, 7 y 6 por tipo, obras sacadas con semilla 0 con descripción de al menos 300 caracteres, segunda obra con el mismo autor (o materia, o idioma).
5. **Definición de la lista que da una respuesta** (necesaria para el nDCG@10 y no detallada en el protocolo): resultados de búsqueda, recomendaciones en orden del agente, y ninguna lista en las rutas de pregunta y fuera de alcance. H7 se muestra además solo sobre las peticiones buscadas, de forma descriptiva.
6. **H9** usa el turno del asistente en cada petición de F sea cual sea la ruta (el sistema tal como funciona; F espera la del agente), y compara la precisión de sus recomendaciones con la de P en el mismo número de puestos; las peticiones sin recomendaciones se dejarían fuera de H9 (aquí no hubo ninguna).
7. **Los dos umbrales** de la sección 7 se aplican a la primera ejecución de A y B; la segunda se muestra al lado.
8. **Profundidad.** El asistente se llamó con los 10 resultados por defecto del endpoint (candidatos vectoriales a profundidad 60) y P con 100 resultados (profundidad 200) para guardar su lista de cien. Se comprobó, solo con la base de datos, que los 30 candidatos que ve el reranker son el mismo conjunto a profundidad 60 y 200 en las 166 consultas de P (166 en el mismo orden; solapamiento mínimo 30 de 30).
9. **Tiempos límite.** El reranker usó los tiempos largos de la evaluación (30 s por intento, 3 reintentos), como en V14, y no los 5 s desplegados, para que un proveedor lento no convierta una medida de calidad en una degradación. Los embeddings usaron el codificador de la aplicación con su configuración desplegada, detrás de una caché de vectores por texto con hasta 3 intentos; una búsqueda de P que volvió sin reranker o degradada se repitió (hasta 3 búsquedas), y un turno del asistente que lanzó una excepción se repitió hasta 3 veces.
10. **Ajustes del LLM.** Los que da `LLMSettings.from_env` con el `.env` del proyecto (OpenRouter, `deepseek/deepseek-v4.1-flash`, temperatura 0, razonamiento desactivado, 1,024 tokens de salida, 180 s de tiempo límite). **No se leyó el entorno de la función desplegada** (V17 registra `LLM_TIMEOUT_S=15` allí): una llamada lenta que el despliegue cortaría a los 15 s se completó aquí.
11. **Sin presupuesto diario.** El servicio de búsqueda se construyó sin presupuesto de uso, de modo que no se leyeron ni se gastaron los contadores del sitio en vivo.
12. **Juicios reutilizados.** Un par ya juzgado en V14 (mismo identificador de necesidad, mismo texto, prompt y juez) se reutilizó; el nDCG@10 se calcula contra la unión del pool de V14 y los pares juzgados aquí.
13. **El agente directo** recibe el texto de la petición, como se lo daría el asistente condensado.
14. **Costes por turno** suman las llamadas al LLM y las de embedding y reranker del hilo de la petición; los embeddings con respaldo corren en otros hilos y solo están en el registro. El gasto lo da el registro, no las sumas por turno.
15. **Etiqueta de etapa.** La clase generadora de V14 etiqueta sus entradas con la etapa `1-queries`; en este registro esa etiqueta es la etapa 1.
16. **Humo previo.** Antes de las ejecuciones completas, cada parte de la etapa 3 se ejecutó con sus dos primeras peticiones como prueba de humo con el mismo código; esos turnos se conservan y forman parte de los datos. Una comprobación aparte de la composición (una búsqueda de P y un turno, unos 0.004 USD) se hizo fuera del directorio de datos y no está en el registro; cae dentro de la ventana del contador de la clave.
17. **Commit.** HEAD pasó de `e964682` a `36802ee` (un commit de documentación de otro agente) durante la tarea; todas las ejecuciones se hicieron en `36802ee`, cuyo código de aplicación es el de `e964682`.

## Límites que debe conocer el tribunal

- Los juicios son de dos jueces LLM únicamente; su acuerdo con personas en este catálogo está pendiente de la hoja humana de V14. Los jueces califican una obra frente a una necesidad: no califican el texto de la respuesta, ni la comparación que pide una petición de F, ni si una pregunta aclaratoria fue útil en sí misma.
- El usuario de E es simulado y responde siempre de forma breve y coherente; la mejora de H8 es un límite favorable.
- Las peticiones de E y F las escribió un LLM, y las de F nombran las obras exactamente con sus títulos del catálogo; pueden ser más regulares que las de usuarios reales.
- Las cifras no son comparables con las de V14, porque el pool de juicios es otro.
- Se midió el sistema con los ajustes del `.env` del proyecto; no se leyó el entorno de la función desplegada.
- El asistente no es determinista: la ruta se repitió una vez en A y B, y la calidad se midió una sola vez.
- E y F son pequeños (30 y 20 peticiones); las tablas por idioma, por tipo y por ruta son descriptivas. Con 20 peticiones, H9 es una prueba pequeña.
- Las latencias se midieron desde un portátil por internet, con cuatro peticiones en curso, contra un Neon compartido con el sitio en vivo y con los tiempos límite largos del reranker; describen ese entorno, no la función desplegada.

## Pendiente

- **Revisión de la instrucción del nodo de ruta y nueva medida (tarea V25).** La adenda `docs/notes/2026-11-12-protocolo-evaluacion-asistente-adenda.md` (commit `0a63751`) declara, antes de tocar nada, cómo se hará: solo se cambian las instrucciones del nodo de ruta, con un criterio (una petición está fuera de alcance por lo que pide, no por su tema) y sin ejemplos tomados de A, B, D, E o F; se corrige a la vez el defecto de las respuestas de redacción que agotan sus tokens. Como la instrucción nueva se escribe conociendo los errores de esta medida, repetirla sobre las mismas peticiones no sería una prueba independiente, de modo que se guarda primero la instrucción y después se genera un conjunto nuevo G (60 peticiones válidas de temas no literarios en seis idiomas y 20 fuera de alcance), que se exige para el umbral del 2 %. La regla de H7 de la sección 7 (usar el texto del usuario) no se aplica, porque V22 mostró que la reescritura no causa la pérdida. Esta nota no anticipa su resultado.
- **Hoja de anotación humana** de V14, que debe rellenar el autor; después se calcula el acuerdo de los jueces con una persona.
- **Revisión por el autor** de las peticiones de E y F generadas por el LLM.
- **Qué puede afirmar el texto del TFG:** que, sobre este catálogo, con peticiones en parte generadas por un LLM y con juicios de dos jueces LLM, pasar por el asistente empeoró la calidad de las peticiones claras (H7, sostenido con los dos jueces), y que esa pérdida viene entera de las peticiones enviadas a una pregunta o a «fuera de alcance» y no de la consulta reescrita; que, con un usuario simulado, preguntar mejoró las peticiones vagas (H8) solo en las conversaciones en que preguntó, como límite favorable; que el umbral de «fuera de alcance» indebido (2 %) no se cumplió (9.5 %) y el de preguntas innecesarias en B (10 %) sí (1.4 %), y que el asistente no cumple la condición para presentarse como entrada principal; que el agente no mostró una ventaja significativa en peticiones de varios pasos (H9) y no recomendó nada que sus herramientas no hubieran devuelto. **Qué no puede afirmar:** que preguntar mejore las búsquedas de usuarios reales, que los juicios coincidan con los de personas, que H9 demuestre que el agente es peor o mejor (no es significativo con 20 peticiones), que las cifras se comparen con las de V14, ni nada sobre el asistente revisado hasta que se ejecute la medida de V25.

## Cómo reproducirlo

```
.venv/bin/python -m app.evaluation.assistant_evaluation report --offline
```

Regenera el informe sin red ni base de datos, desde las ejecuciones, los juicios, los conjuntos y el registro de gasto guardados. Las demás etapas son subcomandos del mismo módulo (`python -m app.evaluation.assistant_evaluation <etapa>`, ver `app/evaluation/assistant_evaluation/cli.py`) y necesitan `OPENROUTER_API_KEY` y acceso de solo lectura a Neon. Tests: `pytest tests/evaluation/test_assistant_evaluation.py`.

## Ficheros

- Evaluación: `app/evaluation/assistant_evaluation/` (`design.py`, `sets.py`, `generation.py`, `simulated_user.py`, `system.py`, `runs.py`, `judging.py`, `analysis.py`, `results.py`, `report.py`, `cli.py`).
- Tests: `tests/evaluation/test_assistant_evaluation.py`.
- Datos y resultados (`data/evaluation/assistant/`): `sets/` (peticiones, comprobaciones y llamadas del generador), `runs/` (`P.jsonl`, `depth_check.jsonl`, `turns.jsonl`, `conversations.jsonl`, `agent_direct.jsonl`, `simulated_user/`), `query_vectors/`, `judgments/` (`pairs.json` y las dos cachés de esta tarea), `system.json`, `spend_ledger.jsonl`, `stage_times.jsonl`, `results.json`, `per_request.json`.
- Informe: `data/evaluation/ASSISTANT_EVALUATION.md`.
- Protocolo: `docs/notes/2026-11-12-protocolo-evaluacion-asistente.md`. Adenda: `docs/notes/2026-11-12-protocolo-evaluacion-asistente-adenda.md`.
- Tareas relacionadas: V14 (`V14-evaluacion-catalogo-desplegado.md`, la evaluación de recuperación de la que parten los conjuntos, los jueces y los juicios) y V20 (`V20-asistente-langgraph.md`, el asistente evaluado).
