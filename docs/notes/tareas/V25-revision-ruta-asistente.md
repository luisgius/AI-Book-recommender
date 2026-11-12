# V25 · Revisión de la instrucción de ruta del asistente y nueva medida

- **Fecha:** 2026-11-12
- **Estado:** hecha; el umbral exigido sobre el conjunto nuevo G **no se cumple** (2 de 60 peticiones válidas rechazadas, 3.3 %, frente a un máximo del 2.0 %), por una petición; el otro umbral exigido se cumple; los juicios son solo de LLM
- **Capa:** asistente (`app/infrastructure/llm/`), cliente del LLM (`app/infrastructure/llm/`), evaluación (`app/evaluation/assistant_evaluation/`), datos (`data/evaluation/assistant_v25/`), documentación (`docs/notes/`)
- **Commits:** adenda con lo declarado en `0a63751` (antes de cambiar nada); instrucción de ruta en `5984e71`; corrección del cliente del LLM en `a09c517`; la repetición de la medida, su informe y esta nota se guardan en un commit posterior, que es el que incorpora el código de evaluación de esta tarea (hasta entonces solo estaba en el árbol de trabajo)

## Resultado

Sobre el conjunto nuevo G, que nadie había visto al escribir la instrucción revisada, el asistente rechazó como «fuera de alcance» 2 de las 60 peticiones válidas: 2/60 = 3.3 %, frente a un máximo declarado del 2.0 % (como máximo una). **El umbral exigido no se cumple**, por muy poco: sobra una petición. El otro umbral exigido, preguntas innecesarias en las peticiones de B, sí se cumple: 0/72 = 0.0 % frente a un máximo del 10.0 %. Las 20 peticiones de G que sí estaban fuera de alcance se detectaron todas (20/20 = 100.0 %; esta medida se informa sin umbral).

Como declaró la adenda, **la instrucción no se ha vuelto a tocar** después de ver este resultado, y se aplica la consecuencia que declaró: el asistente no se presenta como entrada principal de la página pública. La página mantiene la búsqueda de libros como primer modo y el asistente como segundo.

Lo que cambió, en una frase por lado:

- **Mejoró** el rechazo indebido sobre las peticiones originales (de 11 de 116 a 1 de 116 en la primera ejecución), la estabilidad de la ruta, la pérdida en peticiones claras (H7 deja de ser significativa), el paso al agente de todas las peticiones de varios pasos y los fallos de la llamada que redacta la respuesta (de 5 a 1).
- **Empeoró** la pregunta aclaratoria: el asistente pregunta menos ante peticiones vagas, la ganancia de preguntar (H8) deja de ser significativa y la calidad sobre el conjunto de peticiones vagas bajó. El criterio que dejó de rechazar por tema también hizo que el asistente busque más a menudo en lugar de preguntar.

## Qué se ha hecho

- Se cambió **una sola línea** de la instrucción del nodo de ruta (el nodo del grafo de LangGraph que decide entre preguntar, buscar, pasar al agente o rechazar): la línea de `out_of_scope` («fuera de alcance»). El resto de las instrucciones, el grafo y las reglas del código son los mismos. Las instrucciones pasaron de la versión `v1.1` a `v1.2` (`app/infrastructure/llm/prompts_assistant.py`), con un test que fija el criterio.
- Se corrigió en el cliente del LLM (`app/infrastructure/llm/langchain_llm_client.py`) un defecto en cómo se desactiva el razonamiento con el proveedor alojado: con el razonamiento desactivado, el cliente excluye ahora de la ruta al proveedor Novita (véase «El defecto del razonamiento»).
- Se generó un conjunto nuevo, **G**, después de guardar la instrucción en git (véase «El conjunto G»).
- Se repitió la medida completa de V22 (rutas, contrastes H7 a H9, conjuntos A, B, D, E y F) con la instrucción nueva y el cliente corregido, y se midió G. La repetición se guarda aparte, en `data/evaluation/assistant_v25/`; los resultados de la primera medida no se tocaron.
- Se escribió un informe, `data/evaluation/ASSISTANT_EVALUATION_V25.md`, que muestra las dos medidas una junto a otra con la instrucción de cada una.
- Código nuevo en `app/evaluation/assistant_evaluation/`: `set_g.py` (generación y comprobaciones de G), `remeasure.py` (montaje de la repetición) y `report_v25.py` (informe de las dos medidas), además de cambios en los módulos de V22 (`analysis.py`, `cli.py`, `design.py`, `judging.py`, `results.py`, `runs.py`, `system.py`) y en `tests/evaluation/test_assistant_evaluation.py`.

## Por qué

La evaluación V22 declaró dos umbrales antes de ver ningún resultado, como condición para presentar el asistente como entrada principal de la página pública. El de «fuera de alcance» indebido en A y B no se cumplió: 11 de 116 peticiones (9.5 %; 9, es decir 7.8 %, en la segunda ejecución) frente a un máximo del 2 %. Las peticiones rechazadas por error eran sobre todo de cocina y recetas, más algunas de informática. La instrucción definía «fuera de alcance» con ejemplos de tema («tiempo, matemáticas, código») y el modelo tomaba el tema de la petición por su tipo. La pérdida de calidad de H7 (-0.137) venía entera de esas rutas: en las 96 peticiones que el asistente buscó, su resultado era igual al de la referencia.

El protocolo decía que, si un umbral no se cumplía, se revisarían las instrucciones del nodo de ruta y se repetiría la medida declarando el cambio. Esta tarea es ese paso. Además, V22 midió que 5 de 768 llamadas al LLM del sistema (las que redactan la recomendación) agotaban su tope de 1,024 tokens, y en 4 de ellas el modelo había gastado entre 773 y 1,024 tokens en razonamiento aunque estaba desactivado; la adenda pidió corregirlo a la vez.

## Objetivo

Quitar los rechazos por tema sin escribir la instrucción para los casos ya vistos, comprobarlo sobre peticiones que nadie hubiera visto al escribirla, corregir el defecto del razonamiento, y dejar medido qué ganó y qué perdió el asistente con el cambio.

## Lo declarado antes de cambiar nada (adenda `0a63751`)

La adenda se guardó en git a las 16:01, antes del commit de la instrucción (16:12) y antes de generar ninguna petición nueva. Declaró:

- **Qué puede cambiar.** Solo las instrucciones del nodo de ruta, con un criterio y no con una lista de casos: una petición está fuera de alcance por lo que pide, no por su tema. Cualquier tema puede ser el de un libro, así que unas palabras sueltas o un tema escrito en un buscador de libros se tratan como petición de libros sobre ese tema. Queda fuera de alcance lo que pide otra cosa que libros o información sobre libros: un dato del momento, un cálculo, que se escriba un programa o una conversación sin petición. No se añaden ejemplos tomados de las peticiones de A, B, D, E o F. No cambian el grafo, las reglas del código ni el resto de las instrucciones, salvo el defecto del razonamiento.
- **El orden de los pasos, y por qué importa.** Primero se guarda en git la instrucción nueva. Después, y no antes, se genera el conjunto nuevo G. Después se repite la medida completa. El orden importa porque la instrucción se escribió conociendo los errores de la primera medida: repetir la medida solo sobre las mismas peticiones no sería una prueba independiente (la instrucción pudo ajustarse, aunque fuera sin querer, a esas peticiones). G es independiente porque se genera después de fijar la instrucción y sin verla.
- **El conjunto reservado G.** 60 peticiones válidas de temas no literarios (cocina, informática, salud, viajes, economía, ciencia, oficios, deporte y otros), en seis idiomas (es, en, fr, de, it, pt), la mitad como palabras sueltas y la mitad como frases, y 20 peticiones fuera de alcance del tipo descrito. Las escribe el generador de V14, a partir de una lista de temas, sin ver ni la instrucción ni ningún resultado.
- **Dónde se exigen los umbrales.** Los mismos de V22, pero sobre G, el conjunto independiente: «fuera de alcance» indebido en las 60 válidas de G, no más del 2 % (como máximo una); preguntas innecesarias en B, no más del 10 %. La detección en las 20 de G fuera de alcance se informa sin umbral. Sobre A y B se informan los mismos porcentajes como dato secundario, diciendo que la instrucción se escribió conociéndolos.
- **Qué pasa si no se cumplen.** No se vuelve a tocar la instrucción en esta tarea: se informa, y el asistente no se presenta como entrada principal de la página pública.
- **Lo que no se aplica.** La regla de H7 de la sección 7 del protocolo (buscar con el texto del usuario en conversaciones de un turno) no se aplica, porque V22 mostró que la reescritura de la consulta no causa la pérdida.
- **Informe y coste.** Los resultados de la primera medida no se tocan; la repetición se guarda aparte; el informe muestra las dos medidas una junto a otra. Tope de gasto: 2.5 USD.

## La instrucción, antes y después

Solo difiere la línea `out_of_scope`. Instrucción de ruta de V22 (versión `v1.1`, commit `36802ee`, sha256 `2e2cf1bf4920`):

> - "out_of_scope": not something a book catalog can help with (weather, maths, code, chit-chat with no book request).

Instrucción de V25 (versión `v1.2`, commit `5984e71`, sha256 `b8d80f3f6940`):

> - "out_of_scope": the message asks for something other than books or information about books: a fact about the present moment, a calculation, that a program be written, or conversation with no request. Decide by what the message asks for, not by its topic: any topic can be the subject of a book, so a few words or a topic typed into this book search are a request for books about that topic.

La primera nombraba temas («tiempo, matemáticas, código»); la segunda nombra lo que se pide y dice expresamente que el tema no decide.

## Resultados

Todas las cifras son las del informe `data/evaluation/ASSISTANT_EVALUATION_V25.md`. Las ejecuciones de la segunda medida se hicieron el 2026-11-12 entre las 16:19:05 y las 16:30:23. Las 342 llamadas registradas con el commit `a09c517` son las del asistente; las 166 búsquedas de la referencia P (la recuperación desplegada sobre el texto de la petición, sin LLM) son las de la primera medida, copiadas (hechas con `36802ee`; el código de recuperación es el mismo en los dos commits). Los modelos son los de V22. Los juicios son de LLM: juez principal `z-ai/glm-5.3-flash`, segundo `minimax/minimax-m3.1-flash`.

### Umbrales

| Umbral (sección 4 de la adenda) | Valor | Máximo | Resultado |
|---|---|---|---|
| «Fuera de alcance» indebido en las 60 válidas de G | 2/60 = 3.3 % | 2.0 % | **no se cumple** |
| Preguntas en B (primera ejecución) | 0/72 = 0.0 % | 10.0 % | se cumple |

Detección en las 20 de G fuera de alcance (sin umbral): 20/20 = 100.0 %.

Dato secundario sobre A y B (la instrucción se escribió conociendo los errores de la primera medida sobre estas peticiones, de modo que **no es una prueba independiente**):

| Ejecución | Medida | V22 | V25 | Máximo |
|---|---|---|---|---|
| 1 | «fuera de alcance» indebido en A y B | 11/116 = 9.5 % (no se cumple) | 1/116 = 0.9 % (se cumple) | 2.0 % |
| 1 | preguntas en B | 1/72 = 1.4 % (se cumple) | 0/72 = 0.0 % (se cumple) | 10.0 % |
| 2 | «fuera de alcance» indebido en A y B | 9/116 = 7.8 % (no se cumple) | 1/116 = 0.9 % (se cumple) | 2.0 % |
| 2 | preguntas en B | 0/72 = 0.0 % (se cumple) | 0/72 = 0.0 % (se cumple) | 10.0 % |

### Las dos peticiones de G rechazadas por error

Como las da el informe, sin interpretar:

| Petición | Idioma | Campo | Texto | Razón que dio el enrutador |
|---|---|---|---|---|
| g11 | it | hogar y familia | Vorrei un manuale introduttivo per allevare galline in giardino nel modo corretto. | The request is for a practical guide on raising chickens, which is outside the scope of a book catalog. |
| g25 | es | viajes | ruta Islandia coche | The user's message is about a travel itinerary (Iceland by car), not a request for books or book information, so it is out of scope. |

Reparto de las 60 peticiones válidas de G: 58 a búsqueda (96.7 %), 2 a «fuera de alcance» (3.3 %), 0 a pregunta, 0 al agente, 0 errores. Por forma, 1 de las 30 de palabras sueltas y 1 de las 30 frases; por idioma, una en español y una en italiano (diez peticiones por idioma); por campo, una en hogar y familia y una en viajes (seis por campo). Ninguna petición válida recibió una pregunta. Las 20 fuera de alcance, cinco de cada tipo (cálculo, conversación, dato del momento, programa), fueron todas a «fuera de alcance».

### Qué mejoró

**Rechazos indebidos sobre las peticiones originales** (secundario, no independiente). Primera ejecución sobre A y B: de 11 a 1 de 116. La petición que sigue rechazada en la primera ejecución es `q31` (recetas de mariscos y pescados frescos). En A, las peticiones «otras» rechazadas pasaron de 9 a 1 (primera ejecución) y de 8 a 0 (segunda). La detección en D se mantiene en 12 de 12.

**Estabilidad de la ruta** al repetir A y B: misma ruta en 114 de 116 peticiones (98.3 %), frente a 106 de 116 (91.4 %) en V22. Cambios en V25: «fuera de alcance» a búsqueda 1 y búsqueda a «fuera de alcance» 1. Rutas forzadas por código (aclaración inválida convertida en búsqueda): 0 en las dos ejecuciones, frente a 3 en E y 2 en A en V22.

**Pérdida en peticiones claras (H7).** Las peticiones que fueron a pregunta pasaron de 9 a 4 y las que fueron a «fuera de alcance», de 11 a 1; las buscadas, de 96 a 111. Con el juez principal, nDCG@10 (la calidad de las diez primeras posiciones, con más peso arriba) de las peticiones buscadas es 0.790 frente a 0.789 de P, como antes (0.789 frente a 0.789). Por grupo, con el juez principal (asistente / P): A `ambiguous` pasa de 0.091 / 0.769 a 0.462 / 0.769; A otras, de 0.562 / 0.775 a 0.749 / 0.776; B, de 0.774 / 0.800 a 0.801 / 0.800.

**Peticiones de varios pasos (F).** Las 20 van ahora a la ruta del agente (V22: 19 al agente y 1 a búsqueda). Las 20 respuestas incluyen recomendaciones; se recomendaron 20 de las 27 obras que las peticiones nombraban, igual que antes.

**Fallos de la llamada de redacción.** De 5 llamadas que alcanzaron el tope de 1,024 tokens (4 de ellas con tokens de razonamiento) a 1 (sin tokens de razonamiento); avisos `assistant_answer_failed`, de 5 a 1. Ningún turno quedó sin respuesta en ninguna de las dos medidas.

### Qué empeoró

**Pregunta aclaratoria y peticiones vagas (conjunto E, H8).** En E, el asistente preguntó en 8 de las 30 conversaciones, frente a 14 de 30 en V22 (con los dos jueces; 22 conversaciones con 0 preguntas y 8 con 1). En las diez peticiones `ambiguous` de A, las preguntas pasaron de 8 a 4 (primera ejecución; de 6 a 4 en la segunda), y las preguntas en A y B, de 9 a 4 de 116 (7.8 % a 3.4 %). Las cuatro peticiones de A que recibieron pregunta en la primera ejecución son de una sola palabra o muy cortas: `q36` (historia), `q39` (apple), `q41` (learning) y `q43` (classic).

Con el juez principal (entre paréntesis, el segundo), nDCG@10 en E:

| Subconjunto | Medida | n | Asistente tras la conversación | P sobre la petición vaga | P sobre la petición original de B |
|---|---|---|---|---|---|
| todas | V22 | 30 | 0.338 (0.351) | 0.129 (0.153) | 0.775 (0.790) |
| todas | V25 | 30 | 0.206 (0.235) | 0.128 (0.154) | 0.776 (0.793) |
| preguntó | V22 | 14 | 0.540 (0.537) | 0.091 (0.112) | |
| preguntó | V25 | 8 | 0.420 (0.462) | 0.109 (0.133) | |
| no preguntó | V22 | 16 | 0.161 (0.188) | 0.161 (0.189) | |
| no preguntó | V25 | 22 | 0.128 (0.153) | 0.135 (0.161) | |

La calidad sobre el conjunto de peticiones vagas bajó (0.338 a 0.206 con el juez principal; 0.351 a 0.235 con el segundo). La ganancia de preguntar, que en V22 era significativa, ya no lo es (véase la tabla de contrastes). Además, **una petición vaga de E fue rechazada** como «fuera de alcance» en su primer turno (V22: ninguna): la ruta final de E fue búsqueda en 29 conversaciones y «fuera de alcance» en 1. El informe no identifica cuál fue ni da su razón.

Dicho sin suavizar: el criterio que impidió rechazar por tema también hizo que el asistente busque con más facilidad en lugar de preguntar. La instrucción dice que unas palabras sueltas son una petición de libros sobre ese tema; eso quitó rechazos indebidos y quitó también preguntas, y para las peticiones vagas la pregunta era donde estaba la ganancia medida en V22. Sobre A y B las peticiones buscadas puntúan igual que P (nDCG@10), pero sobre las peticiones vagas la calidad bajó.

**Otras caídas que lista el informe** (todas son mecánicas, con el criterio de «peor» que fija el propio informe, y ninguna se contrasta):

- Calidad del agente en H9, media del asistente (precisión de las recomendaciones): de 0.925 a 0.890 con el juez principal y de 0.948 a 0.927 con el segundo. Por tipo con el juez principal (agente / P): comparar 0.914 / 0.771 a 0.829 / 0.781; «como X pero...» 0.871 / 0.643 a 0.857 / 0.593; datos de una obra 1.000 / 1.000 en las dos medidas.
- Precisión de las recomendaciones de la ruta de búsqueda en A y B (asistente frente a P en los mismos puestos): de 0.898 frente a 0.793 a 0.857 frente a 0.774 con el juez principal, y de 0.942 frente a 0.865 a 0.893 frente a 0.828 con el segundo.
- Anclaje: el nodo de verificación descartó 13 de 322 recomendaciones propuestas en la ruta de búsqueda (V22: 9 de 312), todas por no tener ninguna cita válida. En la ruta del agente aparece 1 obra recomendada que ninguna herramienta había devuelto (V22: 0 de 52 conservadas).
- Tiempo del agente: mediana de 10.4 a 11.9 segundos por turno; p95 de 16.3 a 34.5. La mediana de la búsqueda baja de 5.4 a 5.1.

## Contrastes H7 a H9, las dos medidas

Mismo método que en V22: prueba de aleatorización pareada (10,000 cambios de signo, semilla 0), intervalo de confianza del 95 % por bootstrap pareado, corrección de Holm sobre los tres contrastes (un contraste compara una misma medida entre dos sistemas sobre las mismas peticiones, y la corrección de Holm ajusta los p de varios contrastes a la vez), con cada juez. Celda: diferencia [IC 95 %] y p con Holm. Una conclusión se sostiene solo con el mismo signo y significación con los dos jueces.

| Id | Juez | V22 sistema / P | V22 diferencia | V25 sistema / P | V25 diferencia |
|---|---|---|---|---|---|
| H7 | principal | 0.653 / 0.790 | -0.137 [-0.194, -0.084] p <0.001 | 0.756 / 0.790 | -0.034 [-0.068, -0.006] p 0.145 |
| H7 | segundo | 0.674 / 0.801 | -0.127 [-0.182, -0.076] p <0.001 | 0.768 / 0.802 | -0.034 [-0.068, -0.006] p 0.145 |
| H7 | **resultado** | | **sostenido** | | **no significativo** |
| H8 | principal | 0.338 / 0.129 | +0.209 [+0.104, +0.326] p 0.002 | 0.206 / 0.128 | +0.078 [+0.008, +0.167] p 0.145 |
| H8 | segundo | 0.351 / 0.153 | +0.198 [+0.095, +0.311] p 0.002 | 0.235 / 0.154 | +0.082 [+0.005, +0.177] p 0.193 |
| H8 | **resultado** | | **sostenido** | | **no significativo** |
| H9 | principal | 0.925 / 0.795 | +0.130 [+0.025, +0.260] p 0.059 | 0.890 / 0.781 | +0.109 [+0.013, +0.233] p 0.145 |
| H9 | segundo | 0.948 / 0.853 | +0.095 [-0.020, +0.230] p 0.186 | 0.927 / 0.841 | +0.087 [-0.033, +0.230] p 0.282 |
| H9 | **resultado** | | **no significativo** | | **no significativo** |

Lecturas que el propio informe sostiene:

- H7: la pérdida frente a la recuperación sola pasa de -0.137 a -0.034 y deja de ser significativa. Los intervalos de H7 y de H8 con el juez principal no incluyen el cero, pero el p con Holm es 0.145; la regla declarada es la del p con Holm y con los dos jueces, así que ambos son «no significativo».
- H8: la ganancia de preguntar pasa de +0.209 a +0.078 y deja de sostenerse. Esto no demuestra que preguntar no ayude: dice que, con 8 conversaciones en que preguntó y 22 en que no, esta medida ya no lo muestra.
- H9: no significativo en las dos medidas, como en V22; con 20 peticiones es una prueba pequeña.

## El defecto del razonamiento

**Qué enviaba el cliente.** Para desactivar el razonamiento de `deepseek/deepseek-v4.1-flash` por OpenRouter (el servicio que reparte las peticiones entre varios proveedores del mismo modelo), el cliente enviaba `{"provider": {"require_parameters": true}, "reasoning": {"enabled": false}}`. OpenRouter reenvía ese campo. `require_parameters` hace que solo se use un proveedor que admita todos los parámetros enviados.

**Cómo se encontró la causa.** Antes de cambiar la instrucción, se reprodujeron las cinco llamadas de redacción de V22 que alcanzaron su tope (los libros de cada una se leyeron de la base de datos, solo lectura) y después una de ellas se envió, por turnos, a cada proveedor que OpenRouter ofrece para ese modelo. Las llamadas de diagnóstico que respondieron constan en el registro de gasto como etapa `0-diagnosis` (92 llamadas, 0.0649 USD); las que un proveedor rechazó con HTTP 404 o 429 no costaron nada y no figuran. No usan la instrucción de ruta.

**Qué proveedor ignora el interruptor.** Novita mantiene el razonamiento con `{"enabled": false}`, con `{"effort": "none"}` (el valor que OpenRouter documenta para desactivarlo), con los dos a la vez, con `{"max_tokens": 0}` y con el campo de nivel superior `reasoning_effort`, y llega al tope de tokens. Todos los demás proveedores que respondieron usaron 0 tokens de razonamiento. OpenRouter lista a Novita como proveedor que admite `reasoning`, por lo que `require_parameters` no lo excluye, y OpenRouter dirige a él parte de las llamadas.

**La corrección** (commit `a09c517`). Con el razonamiento desactivado, el cliente añade `provider.ignore = ["novita"]`; con el razonamiento activado no se excluye a nadie. El tope de 1,024 tokens no cambia. Hay tests de las dos situaciones.

**Cuántas veces ocurría.** Antes (V22): 5 llamadas de redacción en el tope de 1,024 tokens sobre 768 llamadas al LLM, con tokens de razonamiento de 1,024, 1,024, 773, 0 y 1,024 (cuatro con razonamiento). Después (V25): 1 llamada en el tope sobre 793 llamadas, con 0 tokens de razonamiento. Cuando ocurre, la respuesta conserva los resultados y descarta las recomendaciones (aviso `assistant_answer_failed`: 5 veces antes, 1 después).

**Qué no se probó.**

- El diagnóstico usó una llamada por proveedor, no una muestra repetida.
- No se investigó la llamada que sigue alcanzando el tope en V25: no gastó tokens de razonamiento, así que no es este defecto.
- No se midió el efecto de excluir a Novita sobre la latencia, el coste o la disponibilidad, ni se comprobó con otros modelos u otros proveedores. La latencia p95 del agente subió de 16.3 a 34.5 segundos; el informe no la atribuye a esta exclusión ni a otra causa.
- La instrucción y el cliente cambiaron a la vez en la repetición, y el informe no separa el efecto de cada uno sobre las rutas y las calidades; las rutas dependen de la instrucción y los fallos de redacción dependen del cliente, pero esta medida no las distingue con un experimento aparte.
- No se probó en la función desplegada.

## El conjunto G

**Temas.** La lista de temas está en `data/evaluation/assistant_v25/sets/g_topics.json` (y en `set_g.py`): diez campos (informática, cocina, economía, jardinería, salud, hogar y familia, ciencia, deporte, oficios y viajes), seis peticiones válidas de cada uno.

**Cifras.** 60 peticiones válidas y 20 fuera de alcance. Idiomas: es, en, fr, de, it, pt, con diez válidas por idioma. Forma: 30 palabras sueltas y 30 frases (en cada idioma, cinco y cinco). Las 20 fuera de alcance son cinco por tipo: cálculo, conversación, dato del momento y programa, repartidas entre los seis idiomas. Idiomas y formas se asignaron a los temas en código, con semilla 0.

**Quién las escribió y qué vio.** `google/gemini-3.8-flash` (temperatura 0.7, razonamiento bajo, prompt `v1`, proveedor Google), el 2026-11-12 a las 15:17 UTC, es decir, después del commit de la instrucción (`5984e71`, 16:12 hora local; las 16:17 locales equivalen a las 15:17 UTC). El generador vio solo la lista de temas y las reglas de la tarea; **nunca vio la instrucción de ruta ni ninguna salida del sistema**. El prompt y las salidas en bruto están en `sets/generation/G*.json` y las comprobaciones en `sets/g_checks.json`.

**Qué hubo que corregir.** La adenda no fijaba comprobaciones, de modo que se definieron en esta tarea (véase «Desviaciones»). Tras la primera pasada quedaron 8 problemas y se hicieron 11 sustituciones en hasta tres rondas, hasta 0 problemas. Los problemas fueron: el idioma no era el pedido (`g13`, `g47`, `g50`), o la petición no pedía algo para leer, comprobado sobre la traducción al inglés que da el generador (`g21`, `g34`, `g35`, `g45`, `g47`, `g60`), o no nombraba el tema (`g50`). El informe reconoce que algunas marcas fueron más estrictas de lo necesario: «ouvrage», «Anleitung», «testo» y «publicação» no estaban en la lista de palabras de lectura, y `viajar a Marruecos` se leyó como portugués. Todas las sustituciones están en el informe. Veredictos de idioma de G: 56 correctos y 24 indecisos (palabras clave sin palabras funcionales).

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Añadir a la instrucción ejemplos de las peticiones que fallaron | La adenda lo prohíbe: la instrucción debe ser un criterio, sin ejemplos de A, B, D, E ni F. Una lista de casos tendría el mismo defecto que la de temas y haría que la repetición sobre esas peticiones midiera la memoria de los casos, no el criterio |
| Repetir la medida solo sobre las peticiones originales | La instrucción se escribió conociendo sus errores; no sería una prueba independiente. Por eso G se genera después de guardar la instrucción y sin verla, y los datos de A y B quedan como secundarios |
| Revisar la instrucción una segunda vez al ver el resultado de G | La adenda declaró lo contrario: si G no cumple, la instrucción no se vuelve a tocar en esta tarea y se informa. Ajustarla a G dejaría de dar un conjunto no visto |
| Subir el tope de 1,024 tokens en lugar de excluir al proveedor | La medida mostró que el proveedor razonaba hasta alcanzar el tope; subirlo no quitaría ese gasto y el defecto está en que el interruptor no se respeta. El tope no se cambió |
| Cambiar el grafo, las reglas del código o el resto de las instrucciones | La adenda limita el cambio a la línea `out_of_scope` y al defecto del razonamiento, para que la diferencia entre medidas tenga una sola causa declarada en el lado de la instrucción |
| Aplicar la regla de H7 de la sección 7 (buscar con el texto del usuario) | V22 mostró que la reescritura de la consulta no causa la pérdida (+0.000 en las peticiones buscadas); esa regla no la quitaría |

## Desviaciones respecto a la adenda

El informe las lista todas con su motivo.

1. **Origen del defecto.** La causa del razonamiento se encontró reproduciendo las cinco llamadas de V22 y enviando una de ellas a cada proveedor; esas llamadas son la etapa `0-diagnosis` del registro de gasto. Se hicieron antes de guardar la instrucción y no la usan. El contador de la clave solo se lee alrededor de las etapas lanzadas por la línea de órdenes, de modo que no las cubre.
2. **Reutilización.** La segunda medida copia de la primera los conjuntos A a F, la referencia P (no usa LLM y el código de recuperación es el mismo en los dos commits), la comprobación de profundidad y los vectores de las consultas, y lee las cachés de juicios de la primera; solo se juzgaron los pares no juzgados antes. Las llamadas del usuario simulado no se reutilizaron (dependen de la pregunta del asistente), así que E se jugó de nuevo con el mismo usuario simulado y el mismo prompt.
3. **G se mide solo por su ruta**, como piden los umbrales de la adenda: no se calculó ni P, ni juicios, ni medida de calidad para G. G se ejecutó una vez, no dos.
4. **Comprobaciones de G**, no fijadas por la adenda: idioma (la regla de V14); forma (palabras sueltas: como máximo 4 palabras y sin puntuación final; frase: al menos 5 palabras); sin duplicados dentro de G ni con A a F; y, leído sobre la traducción al inglés que da el generador con cada petición, que una válida comparta una palabra de contenido con su tema y que una frase pida algo para leer (libro, guía, manual y similares), y que una fuera de alcance no mencione libros ni lectura y se lea como su tipo (palabra de tiempo en datos del momento, un dígito en cálculos, una palabra de programación en programas). Algunas marcas fueron más estrictas de lo necesario (véase «El conjunto G»).
5. **Umbrales sobre la primera ejecución.** Los umbrales exigidos son los de la sección 4 de la adenda (G y B, primera ejecución de la segunda medida); las tasas de A y B de las dos ejecuciones son datos secundarios.
6. **Código de evaluación sin guardar en git.** El código que ejecuta y informa la segunda medida (`set_g.py`, `remeasure.py`, `report_v25.py` y las partes de V25 de los demás módulos) no estaba guardado en git cuando se escribió el informe; el sistema medido sí está en los commits registrados por turno, cuyo código de aplicación es el de los dos commits de V25. Se guarda con esta nota.
7. **Etiqueta de etapa.** La clase generadora de V14 etiqueta sus entradas del registro con la etapa `1-queries`; para G esa etiqueta es también la etapa 1.
8. **nDCG@10 con pool propio de cada medida.** Se calcula contra la unión del pool de V14 y los pares juzgados en esa medida, de modo que el valor de P sobre una misma petición puede diferir ligeramente entre las dos columnas cuando la segunda medida juzgó obras nuevas para su necesidad (por ejemplo, P de H7 con el segundo juez: 0.801 y 0.802).
9. **Prueba de humo previa.** Antes de la ejecución completa, cada parte de la etapa 3 se ejecutó con sus dos primeras peticiones con el mismo código; esos turnos forman parte de los datos.

## Límites que debe conocer el tribunal

- Los juicios son de dos jueces LLM únicamente; su acuerdo con personas en este catálogo sigue pendiente de la hoja humana de V14.
- El usuario de E es simulado y responde siempre de forma breve y coherente; cualquier ganancia de preguntar medida con él es un límite favorable.
- Las peticiones de E, F y G las escribió un LLM; pueden ser más regulares que las de usuarios reales. Las de G, además, pasaron comprobaciones mecánicas que sustituyeron 11 peticiones.
- G se ejecutó **una sola vez** y solo se midió su **ruta**; no hay calidad de recuperación ni de respuesta sobre G. Con 60 peticiones válidas, el umbral del 2 % equivale a una sola petición: el resultado (2 de 60) queda a una petición del máximo, y una repetición podría dar otro número porque el asistente no es determinista (en A y B la ruta cambió en 2 de 116 peticiones al repetir).
- E y F son pequeños (30 y 20 peticiones); H8 y H9 se contrastan sobre tan pocas peticiones que un efecto moderado puede no detectarse.
- La instrucción y la corrección del cliente se midieron juntas, sin una medida que separe el efecto de cada una.
- Las cifras de A y B son un dato secundario: la instrucción se escribió conociendo sus errores.
- Se midió con los ajustes del `.env` del proyecto, sin leer el entorno de la función desplegada (como en V22).
- Las latencias se midieron desde un portátil por internet, con cuatro peticiones en curso; no describen la función desplegada.

## Coste y qué se reutilizó

Gasto real de la segunda medida notificado por OpenRouter: **1.3038 USD** (tope 2.50 USD). El contador acumulado de la clave se movió 1.2301 USD entre la primera y la última lectura; no cubre el diagnóstico (etapa `0-diagnosis`). La primera medida costó 1.3980 USD, como referencia.

| Etapa | Llamadas | Fallidas | USD | Segundos |
|---|---|---|---|---|
| 0-diagnosis | 92 | 0 | 0.0649 | n/d |
| 1-queries (conjunto G) | 12 | 0 | 0.0628 | n/d |
| 3-assistant | 1570 | 26 | 1.1660 | 675 |
| 4-judge | 126 | 0 | 0.0101 | 19 |

Las 26 llamadas «fallidas» de la etapa 3 incluyen 25 peticiones de embedding con respaldo canceladas (no son fallos) y la llamada de redacción que alcanzó el tope. El gasto mayor es el reranker (`qwen/qwen3-reranker-8b`, 453 llamadas, 0.7626 USD de los 1.3038). Se reutilizaron los conjuntos A a F, P, la comprobación de profundidad, los vectores de las consultas y 521 de los 1,795 pares de cada juez (junto con 1,211 de V14); solo se juzgaron 63 pares nuevos por juez.

## Pendiente

- **Hoja de anotación humana** de V14, que debe rellenar el autor; después se calcula el acuerdo de los jueces con una persona.
- **Revisión por el autor** de las peticiones de E, F y G generadas por el LLM.
- **Cualquier revisión nueva de la instrucción** necesita un paso nuevo, declarado antes de hacerse, y un conjunto nuevo no visto: G ya se ha medido y no sirve de prueba independiente para una instrucción que se ajuste después de verlo. El paso nuevo tendría que atender el coste ya medido, la caída de las preguntas aclaratorias y las dos peticiones de G rechazadas (`g11`, `g25`).
- **Qué puede afirmar el texto del TFG:** que, con un criterio sobre lo que pide la petición en lugar de su tema, los rechazos por tema se eliminaron en gran parte (de 11 a 1 de 116 en A y B, dato no independiente, y 2 de 60 en el conjunto no visto G), al coste de que el asistente pregunte menos ante peticiones vagas (E: de 14 a 8 de 30; H8 deja de ser significativa y la calidad sobre las vagas baja de 0.338 a 0.206 con el juez principal); que el umbral declarado se incumplió por una petición sobre datos no vistos (2/60 = 3.3 % frente al 2.0 %), por lo que el asistente no se presenta como entrada principal de la página pública (la página mantiene la búsqueda como primer modo y el asistente como segundo); y que si preguntar ayuda sigue mostrado solo por la primera medida y con un usuario simulado. **Qué no puede afirmar:** que la instrucción nueva cumpla la condición para ser la entrada principal; que las cifras de A y B midan la mejora de forma independiente; que preguntar mejore las búsquedas de usuarios reales, ni que no las mejore; que el efecto observado se deba solo a la instrucción (se cambió también el cliente); que los juicios coincidan con los de personas; ni nada sobre calidad de recuperación en G, que no se midió.

## Cómo reproducirlo

```
.venv/bin/python -m app.evaluation.assistant_evaluation report --offline
```

Regenera sin red ni base de datos los dos informes (el de V22, `data/evaluation/ASSISTANT_EVALUATION.md`, y el de V25, `data/evaluation/ASSISTANT_EVALUATION_V25.md`) desde las ejecuciones, los juicios, los conjuntos y los registros de gasto guardados. Las etapas de la repetición son subcomandos del mismo módulo (`python -m app.evaluation.assistant_evaluation <etapa>`, ver `app/evaluation/assistant_evaluation/cli.py`) y necesitan `OPENROUTER_API_KEY` y acceso de solo lectura a Neon. Tests: `pytest tests/evaluation/test_assistant_evaluation.py`, `pytest tests/infrastructure/llm/test_assistant_graph.py` y `pytest tests/infrastructure/llm/test_openrouter_provider.py`.

## Ficheros

- Cambios de sistema: `app/infrastructure/llm/prompts_assistant.py` (versión `v1.2`, línea `out_of_scope`), `app/infrastructure/llm/langchain_llm_client.py` (`PROVIDERS_IGNORING_REASONING_OFF`, `provider.ignore` con el razonamiento desactivado).
- Tests: `tests/infrastructure/llm/test_assistant_graph.py` (criterio de `out_of_scope`), `tests/infrastructure/llm/test_openrouter_provider.py` (exclusión del proveedor solo con el razonamiento desactivado), `tests/evaluation/test_assistant_evaluation.py`.
- Evaluación: `app/evaluation/assistant_evaluation/` (`set_g.py`, `remeasure.py`, `report_v25.py` y las partes de V25 de los demás módulos).
- Datos y resultados (`data/evaluation/assistant_v25/`): `instructions.json`, `sets/` (peticiones con G, `g_topics.json`, `g_checks.json`, `generation/`), `runs/`, `query_vectors/`, `judgments/`, `system.json`, `spend_ledger.jsonl`, `stage_times.jsonl`, `results.json`, `per_request.json`, `assistant.log`.
- Informe: `data/evaluation/ASSISTANT_EVALUATION_V25.md`. Primera medida: `data/evaluation/assistant/` y `data/evaluation/ASSISTANT_EVALUATION.md` (sin cambios).
- Protocolo y adenda: `docs/notes/2026-11-12-protocolo-evaluacion-asistente.md` y `docs/notes/2026-11-12-protocolo-evaluacion-asistente-adenda.md`.
- Tareas relacionadas: V22 (`V22-evaluacion-asistente.md`, la evaluación que falló el umbral), V20 (`V20-asistente-langgraph.md`, el asistente) y V14 (`V14-evaluacion-catalogo-desplegado.md`, el generador, los jueces y los juicios).
