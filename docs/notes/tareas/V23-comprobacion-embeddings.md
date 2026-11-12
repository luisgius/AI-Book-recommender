# V23 · Comprobación del resultado de los embeddings y de sus llamadas fallidas

- **Fecha:** 2026-11-12
- **Estado:** hecha; no cambió la aplicación, el despliegue ni ninguna cifra publicada; aplicar la recomendación es la tarea V24
- **Capa:** evaluación y datos (`data/evaluation/catalog50k/investigation/`); no toca `app/`
- **Commits:** `78616b2`

## Qué se ha hecho

- Se investigaron, en modo de solo lectura, dos resultados de la evaluación del catálogo desplegado (V14, `docs/notes/tareas/V14-evaluacion-catalogo-desplegado.md`): si la puntuación baja del segundo modelo de embeddings (`baai/bge-m3`, llamado E2 en la evaluación) era correcta o fruto de un defecto, y por qué fallaba una de cada diez llamadas al modelo de embeddings desplegado (`qwen/qwen3-embedding-8b`, llamado E1) y qué lo arreglaría.
- Pregunta 1: se hicieron ocho comprobaciones sobre cómo se construyó y se consultó E2, más un análisis de dónde pierde frente a E1. No se encontró ningún defecto.
- Pregunta 2: se analizó el registro de las llamadas de V14, se midió por qué el mismo texto da vectores distintos, se midió si esa diferencia cambia la calidad y se probaron seis políticas de mitigación con llamadas reales en tres rondas. Se recomienda una política concreta.
- El resultado completo está en `data/evaluation/catalog50k/investigation/FINDINGS.md`; esta nota lo resume para quien no haya seguido el trabajo.
- No se cambió ningún código de evaluación, informe, juicio guardado ni ajuste del despliegue. No se recodificó nada ni se pidieron juicios nuevos.

## Por qué

- **La diferencia entre los dos modelos era demasiado grande para darla por buena sin más.** Sobre el catálogo desplegado (50,000 obras en doce idiomas), E2 obtuvo un nDCG@10 de 0.426 frente a 0.683 de E1 con el juez principal (el nDCG@10 mide la calidad de las diez primeras posiciones). En la colección de 177 libros (V16) la diferencia era de 0.614 frente a 0.654. Además la brecha aparecía en todos los idiomas y en todos los tipos de consulta. Eso podía ser real o ser el síntoma de un defecto (vectores mal construidos, ids desalineados, texto truncado), y V14 solo había comprobado nueve obras.
- **La regla de decisión dependía de ese resultado.** La regla del modelo de embeddings (sección 8 del protocolo, `docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`) mantuvo el modelo desplegado porque H6 (el contraste E2 frente a E1) mostraba a E1 significativamente mejor con los dos jueces. Si H6 fuese un artefacto, la decisión estaría mal tomada. Y el TFG cita esa cifra.
- **Las llamadas fallidas degradan la web viva.** Cuando falla el embedding de la consulta, el sitio recurre a la búsqueda solo por palabras clave (BM25), que es el peor sistema medido (nDCG@10 0.336 frente a 0.683, juez principal). V14 midió 51 llamadas fallidas de 504 con el modelo desplegado frente a 6 de 504 con el otro.

## Objetivo

Decidir con datos si la cifra de H6 y la decisión de mantener `qwen/qwen3-embedding-8b` se sostienen, entender la causa de los fallos y de los vectores cambiantes del modelo desplegado, y proponer una corrección medida, sin cambiar nada todavía.

## Pregunta 1: ¿es sólido el resultado de `baai/bge-m3` (E2)?

### Conclusión

No se encontró ningún defecto en cómo se construyó o consultó E2. La cifra publicada de E2 (nDCG@10 0.426 frente a 0.683 de E1) se mantiene. Los vectores guardados son los que devuelve la API, el proveedor sirve el modelo real, el índice y los identificadores están alineados para las 50,000 obras, y `baai/bge-m3` encuentra una obra a partir de su título tan bien como qwen.

La diferencia viene de las consultas que necesitan conocer el libro y ese conocimiento no está en el texto del catálogo: argumento, personajes, lugares («Buendía, Macondo», «Meursault ... Alger»). El modelo qwen de 8,000 millones de parámetros lo lleva consigo y `baai/bge-m3`, de 568 millones, no. El tamaño de la brecha coincide con el banco de pruebas multilingüe publicado: recuperación de MMTEB 70.88 para Qwen3-Embedding-8B frente a 54.60 para BGE-M3 (ficha del modelo Qwen3-Embedding-8B en Hugging Face).

**H6 y la regla de decisión de la sección 8 no cambian.** H6 sigue sostenida con los dos jueces y el resultado de la regla sigue siendo «se mantiene `qwen/qwen3-embedding-8b`». Como no se halló ningún defecto, no se recodificó nada ni se hicieron juicios nuevos.

### Lo que se comprobó y lo que descartó cada comprobación

1. **Texto, prefijos, dimensiones, truncado y proveedor.**
   - El texto por obra es `Book.get_searchable_text()`: título, autores, una descripción y materias (`app/domain/entities.py:138`). No concatena descripciones de varios idiomas; como mucho es bilingüe (descripción en el idioma de la obra y materias, en su mayoría en inglés). Sobre las 50,000 obras, la mediana es de 596 caracteres, el percentil 99 de 1,369 y el máximo de 2,058. Ninguno se acerca al contexto de 8,192 tokens de `baai/bge-m3` en DeepInfra, de modo que el proveedor no pudo truncar.
   - Los documentos pasan por la misma clase `OpenRouterEmbeddingEncoder` para los dos modelos, sin prefijo de documento (el modelo BGE-M3 no lo usa y Qwen3 solo lleva instrucción en las consultas), en peticiones de hasta 32 textos y 20,000 caracteres, fijadas a DeepInfra. `baai/bge-m3` usa su dimensión nativa de 1024 (no se envía `dimensions`); qwen se recorta a 1024 (Matryoshka).
   - Las consultas de `baai/bge-m3` van sin prefijo, como prescribe su ficha para la recuperación densa; las de qwen llevan la instrucción de Qwen3.
   - La similitud es el coseno (`ExactIndex` normaliza ambos lados). Los vectores guardados de `baai/bge-m3` tienen norma unidad (0.9999999 a 1.0000001) y ningún NaN.
   - Proveedor: DeepInfra (`deepinfra/fp32`). OpenRouter lista DeepInfra y Parasail para `baai/bge-m3`.
   - Descarta: truncado, prefijos mal puestos, similitud mal definida.
2. **El proveedor sirve el modelo real** (`q1_reference_check.py`). El ejemplo denso de la ficha de BGE-M3 da similitudes [[0.6265, 0.3477], [0.3499, 0.678]]. A través de OpenRouter, DeepInfra da [[0.626, 0.348], [0.350, 0.6695]] y Parasail [[0.6259, 0.348], [0.3497, 0.6696]]. Los dos proveedores coinciden entre sí y con la ficha hasta la precisión de fp16. Descarta: que el proveedor sirva otro modelo o una versión degradada.
3. **Los vectores guardados son los que devuelve la API y el agrupado en lotes es inocuo** (`q1_batch_vs_single.py`). Se reenviaron los dos primeros lotes originales del fragmento 7 (23 y 27 obras) tal como los agrupó V14 y luego cada obra sola. Coseno con los vectores guardados: por lotes, mínimo 0.9999954; solas, mínimo 0.9999962; solas frente a por lotes, mínimo 0.9999962. No hay correlación con la longitud del texto (r = 0.002). Esto amplía a 50 obras y al camino por lotes la comprobación de nueve obras de V14. Descarta: vectores corruptos al guardarlos o por el agrupado.
4. **Identificadores alineados en todo el conjunto; el índice equivale a la fuerza bruta** (`check_e2_index.py`).
   - Los 25 fragmentos contienen 50,000 identificadores únicos, iguales a los del cargador ordenados (el orden con que `build_vectors` los construyó).
   - Un coseno por fuerza bruta sobre la matriz sin procesar da los mismos diez primeros que el fichero de ejecución de E2 en las 168 consultas (0 discrepancias).
   - Hay un grupo de vectores idénticos, que corresponde al único grupo de textos idénticos del catálogo.
   - Los vectores congelados de las consultas de `baai/bge-m3` coinciden con vectores nuevos: 168 consultas recodificadas, coseno mínimo 0.999998.
   - Descarta: desalineación entre vectores e identificadores, fallos del índice.
5. **Autorrecuperación por título: `baai/bge-m3` no está roto** (`q1_self_retrieval.py`). Se tomaron 200 obras (semilla 0) y el título solo como consulta. `baai/bge-m3` buscó en el índice local de E2 sobre las 50,000 obras; qwen usó la instrucción desplegada con una búsqueda exacta en Neon, de solo lectura y secuencial.

   | Modelo | hit@1 | hit@10 | hit@100 | MRR@100 |
   |---|---|---|---|---|
   | `baai/bge-m3` (índice de E2) | 0.975 | 0.995 | 1.000 | 0.981 |
   | `qwen/qwen3-embedding-8b` (E1) | 0.940 | 1.000 | 1.000 | 0.964 |

   De los 200 rangos, 186 son iguales; `baai/bge-m3` es mejor en 10 y qwen en 4. Un índice desalineado o corrupto fallaría estrepitosamente en esta prueba.
6. **Cortes por longitud de texto y por idioma** (`q1_pool_and_splits.py`). Recall@100 de las obras con grado 2 o 3 del juez principal, sobre las consultas A y B:

   | Corte | n | E1 | E2 |
   |---|---|---|---|
   | texto < 400 caracteres | 212 | 0.858 | 0.481 |
   | 400-800 | 491 | 0.862 | 0.534 |
   | 800-1,200 | 384 | 0.919 | 0.552 |
   | >= 1,200 | 192 | 0.870 | 0.620 |
   | idioma de la obra en | 528 | 0.852 | 0.597 |
   | es | 280 | 0.875 | 0.550 |
   | fr | 105 | 0.933 | 0.410 |
   | de | 58 | 0.897 | 0.241 |
   | otros | 308 | 0.909 | 0.549 |
   | idioma de la obra = idioma de la consulta | 615 | 0.911 | 0.532 |
   | idioma de la obra ≠ idioma de la consulta | 664 | 0.851 | 0.554 |

   La brecha existe en todos los tramos de longitud y es la menor en los textos más largos, así que no es un efecto de texto largo ni de truncado. Es mayor en las obras en francés y alemán. No es menor cuando la obra y la consulta comparten idioma, así que tampoco es un problema de alineación entre idiomas. El segundo juez da el mismo cuadro. Estos cortes son descriptivos y sus celdas son pequeñas (58 obras en alemán).
7. **Otras comprobaciones.** No hay «hubness» (unas pocas obras que aparecen en casi todos los resultados): 1,299 obras distintas en los 168 × 10 resultados de E2 frente a 1,318 en E1, y la obra más frecuente de E2 aparece 7 veces. No hay preferencia por textos cortos: la mediana de longitud de las diez primeras obras de E2 es de 880 caracteres, frente a 774 en E1 y 596 en el catálogo (`q1_length_bias.py`). Las puntuaciones de la ejecución de E1 se recalculan desde los vectores guardados de qwen con un error de 1.2e-5.
8. **Comparación con el método de la medición de 177 libros** (`app/evaluation/api_models_evaluation.py`, V16). Misma clase de codificador, misma función de texto, mismo agrupado de 32 textos y 20,000 caracteres, mismo proveedor fijado, sin prefijo en `baai/bge-m3` y coseno exacto (FAISS sobre vectores normalizados). Solo cambia la biblioteca del índice (FAISS allí, numpy aquí), y la comprobación 4 demuestra que numpy equivale a la fuerza bruta. Las dos mediciones no difieren en el método. Las mismas 44 consultas del estrato A («originales») dan: con 177 libros, `baai/bge-m3` 0.614 y qwen 0.654; con 50,000 obras, E2 0.420 y E1 0.639. El cambio procede, pues, de la colección.

### Dónde pierde `baai/bge-m3`: conocimiento que no está en el texto

Rango de la obra buscada en las 12 consultas de obra conocida (`q1_known_items.py`):

| Consulta | Obra | Rango E1 | Rango E2 |
|---|---|---|---|
| b-en-6 | Jane Eyre | 1 | 14 |
| b-es-6 | Cien años de soledad | 1 | 2,462 |
| b-fr-6 | L'Étranger | 3 | 14,129 |
| b-pt-6 | Memórias Póstumas de Brás Cubas | 2 | 71 |
| b-ru-6 | Мастер и Маргарита | 1 | 13 |
| b-de-6, b-nl-6, b-zh-6, b-ja-6 | (cuatro obras) | 1 | 1 |
| b-it-6 | I promessi sposi | 1 | 2 |
| b-ca-6 | La plaça del Diamant | 18 | 22 |
| b-ar-6 | الأيام | 11 | 2 |

El texto del catálogo de «Cien años de soledad» nunca menciona a Buendía ni Macondo, y el de «L'Étranger» nunca menciona a Meursault ni Argel; la consulta nombra solo eso. Los primeros resultados de `baai/bge-m3` son coincidencias superficiales: «Araby» para «un Arabe», «Edward II» para «Edward Rochester». Hasta «Jane Eyre», cuyo texto sí contiene «Rochester» y «Thornfield Hall», sale la decimocuarta (coseno 0.512 frente a 0.539 del primer resultado). Las consultas descriptivas y temáticas muestran el mismo patrón, más suave.

### La explicación ofrecida y hasta dónde llega

Qwen, mucho mayor, aporta conocimiento del mundo sobre los libros que `baai/bge-m3` no tiene, y las consultas del estudio, escritas por un LLM con ese conocimiento, lo premian. Está respaldada por: la ausencia de defecto en las ocho comprobaciones, la autorrecuperación por título equivalente, el patrón de las consultas de obra conocida y la coincidencia de la brecha con el banco de pruebas publicado. **No está aislada por un experimento.** Por qué la brecha es mucho mayor aquí que con 177 libros se argumenta (hay pocos distractores y encontrar la obra rara vez exige conocimiento ausente del texto; con 50,000 obras en doce idiomas sí lo exige a menudo), no se demuestra.

### Lo que sigue incierto

- No se comparó con una copia local de `BAAI/bge-m3`: pesa 2.2 GB, por encima del límite de descarga de la tarea. La comprobación de la ficha del modelo en dos proveedores hace de sustituto.
- No se puede simular el efecto del tamaño de la colección con estos juicios: cualquier subcolección que conserve el pool devuelve los mismos diez primeros para los dos modelos (comprobado: el nDCG@10 dentro del pool reproduce exactamente 0.683 y 0.426).
- Los cortes de la comprobación 6 son descriptivos y con celdas pequeñas.
- Las consultas las escribió un LLM con conocimiento del mundo; las de obra conocida y las descriptivas premian a un modelo que tenga ese conocimiento, y los usuarios reales pueden formularlas de otro modo.

## Pregunta 2: ¿por qué fallaron 51 de 504 llamadas a qwen y qué lo arregla?

### Conclusión

- **Los fallos.** Los 51 fallos de V14 fueron tiempos de lectura agotados (dos intentos de 3 s que chocaron con el límite de 5 s); no hubo ningún error HTTP. El `qwen/qwen3-embedding-8b` de DeepInfra tiene episodios de lentitud de minutos. Durante un episodio, un reintento o una petición duplicada (un «hedge») al mismo proveedor fallan también, mientras que Nebius y SiliconFlow fallan de forma independiente.
- **Los vectores cambiantes.** El único punto de servicio de DeepInfra devuelve al menos dos variantes discretas del vector para el mismo texto (coseno 0.996-0.998 entre ellas). La variante mayoritaria coincide con los documentos guardados y con las consultas congeladas de E1. Medidas sobre los juicios existentes, las variantes cambian el orden de los diez primeros pero no su calidad.
- **Recomendación.** Mantener DeepInfra en primer lugar y duplicar la petición hacia Nebius: si DeepInfra no ha respondido en 1.0 s, se envía la misma petición a Nebius y se toma la que conteste primero (plazo de 5 s y una segunda petición a DeepInfra a los 2.5 s).

### Evidencia

**1. Qué fueron los fallos de V14** (`q2_analyse_operational.py`; datos en `data/evaluation/catalog50k/operational/`).

| | `qwen/qwen3-embedding-8b` | `baai/bge-m3` |
|---|---|---|
| Fallos | 51 / 504, todos «ReadTimeout (after 2 attempts)», 5.07-5.30 s | 6 / 504: 4 × HTTP 429 «Model busy», 2 × ReadTimeout |
| Cuándo | 39 entre 12:31 y 12:39, 12 entre 12:40 y 12:43 | todos en la repetición 3 |
| Longitud de la consulta (caracteres, mediana) | fallidas 43, correctas 46 | fallidas 72, correctas 46 |
| Latencia de las correctas, mediana / p95 / p99 / máximo | 399 / 1,977 / 4,490 / 4,811 ms | 398 / 728 / 1,432 / 1,712 ms |
| Vectores distintos por consulta (1/2/3) | 36 / 111 / 21 | 74 / 66 / 28 |
| Coseno mínimo entre repeticiones (consulta mediana) | 0.99787 | 0.999999 |

No hay dependencia de la longitud de la consulta y los fallos por idioma son proporcionales al número de llamadas. El registro operativo no guarda el proveedor, pero todas las llamadas estaban fijadas a DeepInfra sin alternativas.

**2. Por qué el mismo texto da vectores distintos** (`q2_determinism.py`, `q2_doc_variant.py`, `q2_generation_metadata.out`). Cuatro consultas se enviaron 15 veces por proveedor con un plazo de 30 s.
- DeepInfra da exactamente 2 variantes distintas bit a bit por texto, con coseno 0.9972-0.9982 entre ellas. La variante minoritaria responde más rápido (mediana de unos 0.3 s frente a 0.8-1.5 s de la mayoritaria), lo que apunta a réplicas o compilaciones distintas detrás del mismo punto de servicio.
- Nebius y SiliconFlow dan un patrón de bits nuevo en casi cada llamada (6-9 distintos en 15 llamadas) pero coinciden consigo mismos y entre sí con coseno 0.9998-0.9999. Es el no determinismo ordinario de bf16/fp8.
- Ambos están a coseno 0.9963-0.9966 de las variantes de DeepInfra: la salida de DeepInfra es la atípica.
- **Qué variante son los datos guardados.** Se recodificaron 8 obras del pool como documentos 8 veces cada una en DeepInfra: 54 de 64 llamadas coinciden con los vectores guardados en Neon (coseno 1.000) y las otras 10 están a 0.9982-0.9993. Nebius está a 0.9965-0.9977 en todas las llamadas. Para 6 consultas, el vector congelado de E1 coincide con la variante mayoritaria de DeepInfra. La proporción de llamadas que devuelven la otra variante cambió a lo largo del día: de 1 a 4 de 15 en una prueba, hasta 7 de 8 para una consulta más tarde.
- OpenRouter no expone nada que distinga las variantes. La respuesta solo lleva `provider: "DeepInfra"`; el registro de generación tiene `provider_name`, latencia y un `upstream_id`, pero no réplica ni cuantización. OpenRouter lista un solo punto de servicio de DeepInfra para este modelo (`deepinfra/us`, cuantización «unknown»), así que no hay identificador que fije una variante.

**El enunciado de la documentación del servicio que estos datos contradicen.** La página de embeddings de OpenRouter afirma que los embeddings del mismo texto «will always be identical» (serán siempre idénticos). Las mediciones lo contradicen para el qwen de DeepInfra.

**3. Las variantes no cambian la calidad** (`q2_variant_quality.py`). Las repeticiones operativas de V14 se dividieron en llamadas que coinciden con la variante guardada (162) y las que no (144), y se puntuaron sus diez primeros con los juicios existentes (A y B):

| | nDCG@10 principal | nDCG@10 segundo | Judged@10 |
|---|---|---|---|
| variante coincidente | 0.679 | 0.695 | 1.000 |
| otra variante | 0.680 | 0.697 | 0.975 |

Pareado sobre 81 consultas, otra menos coincidente: -0.007 con los dos jueces, con diferencia absoluta media de 0.015. Las «116 de 167 consultas cuyo top 10 cambió» de V14 son, por tanto, una reordenación entre obras relevantes: una molestia de reproducibilidad, no una pérdida de calidad.

**4. Vectores de consulta de otros proveedores frente a los documentos de DeepInfra** (`q2_nebius_quality.py`). Búsqueda exacta en Neon con el vector de consulta de Nebius o de SiliconFlow, 60 consultas:
- Nebius: nDCG@10 (A y B, 39 consultas) 0.689 principal y 0.694 segundo, frente a 0.682 y 0.690 de E1 en las mismas consultas.
- SiliconFlow: 0.682 y 0.688.
- Solapamiento de los diez primeros con E1: 0.95 en ambos.

Un vector de consulta de Nebius es, pues, utilizable contra los documentos guardados.

**5. Mitigaciones con llamadas reales.** Cada brazo es una política completa, tal como la ejecutaría una petición web. 60 consultas (semilla 0) × 3 repeticiones = 180 llamadas por brazo, con todos los brazos intercalados llamada a llamada en orden aleatorio, de modo que todos vieron las mismas condiciones de proveedor. «Mismo vector» es el número de consultas cuyos tres vectores coinciden a coseno 0.99999; «variante guardada» es la proporción de vectores iguales a la variante de los documentos guardados. Los tiempos son segundos e incluyen los fallos.

Ronda 1, 13:44-14:16 (`q2_mitigations.py`), que incluyó un episodio de DeepInfra:

| Brazo | Fallos | Mediana | p95 | Peticiones/llamada | Mismo vector | Variante guardada |
|---|---|---|---|---|---|---|
| A0 qwen DeepInfra, desplegado (3 s × 2, 5 s) | 22/180 (12.2 %) | 0.69 | 5.11 | 1.13 | 27/60 | 0.74 |
| A1 qwen Nebius, desplegado | 21/180 (11.7 %) | 1.90 | 5.09 | 1.27 | 1/59 (todos ≥ 0.99984) | 0.00 |
| A2 qwen SiliconFlow (fp8), desplegado | 29/180 (16.1 %) | 1.57 | 5.12 | 1.32 | 2/58 (todos ≥ 0.99978) | 0.00 |
| A3 qwen DeepInfra, 1.5 s × 3, 5 s | 20/180 (11.1 %) | 0.61 | 4.78 | 1.31 | 27/60 | 0.70 |
| A4 qwen DeepInfra, duplicado en DeepInfra (2.ª petición a 1.0 s, máx. 3) | 10/180 (5.6 %) | 0.68 | 5.00 | 1.46 | 23/60 | 0.72 |
| A5 qwen orden [DeepInfra, Nebius], alternativas permitidas | 22/180 (12.2 %) | 0.64 | 5.10 | 1.13 | 28/60 | 0.71 |
| A6 `baai/bge-m3` DeepInfra, desplegado (referencia) | 1/180 (0.6 %, HTTP 429) | 0.46 | 1.00 | 1.01 | 60/60 | 1.00 |

Los fallos se concentran en el tiempo (`q2_failure_timing.out`): entre los brazos de qwen, 5-7 fallos por ventana de 3 minutos antes de las 13:54, 17-26 por ventana entre las 13:54 y las 14:05 y luego 0-7. A0 falló en 2, 20 y 0 llamadas en sus tres repeticiones.
- Las estrategias sobre el mismo proveedor fallan juntas: A0 y A3 comparten 16 de sus 22 y 20 huecos fallidos (se esperarían 2.4 si fuesen independientes); A0 y A4 comparten 8, frente a 1.2 esperados.
- Los demás proveedores fallan de forma aproximadamente independiente de DeepInfra: A0 y A1 comparten 6 huecos fallidos, frente a 2.6 esperados; A0 y A2 comparten 4, frente a 3.5.
- La alternativa propia de OpenRouter (A5) nunca se activó. Una respuesta lenta es un tiempo agotado en el cliente, no un error del que OpenRouter pueda recurrir a otro proveedor, de modo que A5 equivale a A0.

**Estimación, no medición.** A partir de esos huecos, un duplicado DeepInfra más Nebius habría fallado en 6/180 huecos (5 de 62 dentro del episodio, frente a 20 de A0), y con SiliconFlow añadido, en 1/180 (`q2_hedge_estimate_from_round1.out`).

Ronda 2, 14:27-14:36 (`q2_round2.py`), un periodo tranquilo:

| Brazo | Fallos | Mediana | p95 | p99 | Peticiones/llamada | Mismo vector | Variante guardada | Respondió |
|---|---|---|---|---|---|---|---|---|
| B0 DeepInfra, desplegado [control] | 1/180 (0.6 %) | 0.47 | 1.91 | 2.82 | 1.01 | 19/60 | 0.64 | DeepInfra 179 |
| B1 DeepInfra, un intento, 10 s | 0/180 | 0.51 | 2.65 | 3.29 | 1.00 | 26/60 | 0.72 | DeepInfra 180 |
| B2 duplicado DeepInfra 0 s / Nebius 1.0 s / DeepInfra 2.5 s, 5 s | 0/180 | 0.46 | 1.84 | 2.84 | 1.18 | 16/60 | 0.63 | DeepInfra 174, Nebius 6 |
| B3 igual, Nebius a 0.6 s | 0/180 | 0.52 | 1.69 | 2.47 | 1.41 | 17/60 | 0.63 | DeepInfra 167, Nebius 13 |

B1 tuvo 5 llamadas correctas de más de 3 s (3.07-5.34 s), de las cuales la más lenta también la habría perdido el plazo desplegado de 5 s.

Ronda 3, 14:37-14:49 (`q2_round3.py`): 60 consultas × 6 repeticiones = 360 llamadas por brazo. El periodo fue en su mayor parte tranquilo; los únicos 2 fallos del control y 5 de sus 6 llamadas de más de 3 s cayeron en la primera repetición, de modo que fue una racha lenta corta, no un episodio completo.

| Brazo | Fallos | Mediana | p95 | p99 | Peticiones/llamada | Mismo vector (6 rep.) | Variante guardada | Respondió |
|---|---|---|---|---|---|---|---|---|
| B0 DeepInfra, desplegado [control] | 2/360 (0.6 %) | 0.46 | 1.60 | 4.21 | 1.02 | 5/60 | 0.62 | DeepInfra 358 |
| B2 duplicado DeepInfra 0 s / Nebius 1.0 s / DeepInfra 2.5 s, 5 s | 0/360 | 0.50 | 1.57 | 2.51 | 1.17 | 5/60 | 0.59 | DeepInfra 354, Nebius 6 |
| B4 duplicado DeepInfra 0 s / Nebius 1.0 s / SiliconFlow 2.0 s, 5 s | 0/360 | 0.46 | 1.46 | 2.52 | 1.15 | 4/60 | 0.62 | DeepInfra 350, Nebius 9, SiliconFlow 1 |

En las rondas 2 y 3, el duplicado recomendado (B2) falló en 0 de 540 llamadas y el control en 3 de 540. Su mediana se mantuvo al nivel del control, su p99 bajó de 2.8-4.2 s a 2.5-2.8 s y promedió unas 1.17 peticiones por búsqueda. La variante de tres proveedores (B4) lo hizo igual de bien, pero exige un tercer proveedor.

### Recomendación

**Qué cambiar**, en la llamada de embedding de la consulta del perfil nube:
- enviar la petición a DeepInfra como ahora;
- si a 1.0 s no ha llegado respuesta, o DeepInfra devuelve un error, enviar la misma petición a Nebius fijado (`provider: {"order": ["nebius"], "allow_fallbacks": false}`) y usar la primera respuesta correcta;
- enviar una petición más a DeepInfra a los 2.5 s;
- mantener el plazo de 5 s.

Los documentos se quedan como están: los vectores de consulta de Nebius funcionan contra ellos (evidencia 4). Si existe caché de consultas, su clave no debería incluir el proveedor.

**Cifras medidas.**
- Periodos tranquilos (rondas 2 y 3): 0/540 fallos frente a 3/540 de la política desplegada, mediana de 0.46-0.50 s, p95 de 1.57-1.84 s y unas 1.17 peticiones por búsqueda.
- Calidad sobre los juicios existentes: sin cambio.

**Cifra estimada, no medida.** En un episodio: 6/180 fallos frente a 22/180 de la política desplegada, estimado a partir de los huecos de la ronda 1 (alrededor de un 3 % en lugar de un 12 %, o un 0.6 % con SiliconFlow como tercera petición).

**Coste.**
- Unas 0.2 peticiones de embedding más por búsqueda (0.4 si la segunda petición sale a 0.6 s), a 1e-8 USD por token, es decir, despreciable.
- Un proveedor más en el enrutado del despliegue y su política de datos. Los términos de Nebius deberían contrastarse con la declaración de privacidad del sitio.
- Unas 30 líneas en `OpenRouterHttp` y `OpenRouterEmbeddingEncoder` (dos peticiones en vuelo, gana la primera correcta), con tests. La función serverless debe permitir dos peticiones salientes simultáneas.

**Lo que no arregla.**
- La repetibilidad: la misma consulta sigue recibiendo vectores ligeramente distintos, de las dos variantes de DeepInfra o de Nebius. Medido, no cuesta calidad, pero los diez primeros pueden reordenarse entre búsquedas idénticas. Las únicas soluciones son guardar en caché el vector de la consulta por texto (las búsquedas repetidas pasan a ser idénticas; las primeras no cambian) o recodificar todo el catálogo con un proveedor estable a coseno 0.9999 (Nebius) y usar solo ese proveedor, lo que renuncia a la alternativa de la que depende esta recomendación.
- Un episodio que afecte a los dos proveedores a la vez (1 de 180 huecos en la ronda 1) ni las incidencias de todo OpenRouter.

### Lo que no se pudo medir directamente

- **El comportamiento de la política recomendada durante un episodio lento completo.** Las rondas 2 y 3 fueron tranquilas, así que la cifra de episodio (alrededor de un 3 %) está estimada desde la ronda 1, no observada.
- La causa dentro de DeepInfra de los episodios y de las dos variantes: lo medido es coherente con varias réplicas tras un mismo punto de servicio, pero su lado no es observable.
- El comportamiento desde la región de la función desplegada: todas las latencias se midieron desde un portátil.
- Si la política de datos de Nebius es adecuada para el sitio público.

## Alternativas consideradas y por qué no

Las de la tabla de opciones rechazadas de las conclusiones, cada una con su motivo medido.

| Alternativa | Por qué no |
|---|---|
| Fijar Nebius o SiliconFlow como único proveedor | No reduce los fallos (11.7 % y 16.1 % en la ronda 1) y es tres veces más lento en la mediana |
| Intentos más cortos o reintentos solo en DeepInfra (A3) | Sin mejora durante un episodio: las estrategias del mismo proveedor fallan juntas |
| Duplicar solo en DeepInfra (A4) | Reduce los fallos a la mitad, pero sigue en 5.6 % |
| Alternativas de OpenRouter (A5) | Nunca se activan ante la lentitud: un tiempo agotado en el cliente no es un error del que pueda recurrir a otro proveedor |
| Un único plazo más largo (B1) | Cambiaría las alternativas por búsquedas de hasta 5-10 s, y las latencias durante un episodio no se observaron sin censura |

## Cómo se ha validado

- Cada afirmación de las dos preguntas descansa en un script de la carpeta de la investigación y en su salida guardada; las llamadas de cada prueba se conservan con su vector (`*_calls.jsonl`).
- Las pruebas de mitigación intercalan todos los brazos llamada a llamada en orden aleatorio, para que compartan condiciones de proveedor, y se repitieron en tres rondas.
- Las comprobaciones de la pregunta 1 se contrastan entre sí (fuerza bruta frente a índice, vectores guardados frente a llamadas nuevas, ficha del modelo frente a dos proveedores).
- No hay tests nuevos porque no se cambió código de la aplicación.

## Límites que debe conocer el tribunal

- No se comparó con una copia local de `BAAI/bge-m3` (límite de descarga); la comprobación con la ficha en dos proveedores la sustituye.
- La causa de que la brecha sea tan grande aquí está argumentada, no aislada con un experimento; los cortes por longitud e idioma son descriptivos y de celdas pequeñas.
- Las consultas las escribió un LLM con conocimiento del mundo, lo que favorece a un modelo que lo tenga; los usuarios reales pueden preguntar de otro modo.
- La cifra de la política recomendada durante un episodio es una estimación desde la ronda 1; las rondas 2 y 3 fueron tranquilas.
- Todas las latencias se midieron desde un portátil, no desde la región de la función desplegada.
- La causa interna de DeepInfra (episodios y variantes) no es observable; lo que se afirma es coherencia con varias réplicas.
- Los episodios de lentitud dependen del proveedor y del momento: las rondas cubren unas dos horas de un solo día.

## Pendiente

- **Aplicar la recomendación (V24):** tarea aparte, junto con el cambio de la configuración de recuperación desplegada que pidió la regla de la evaluación (pasar de híbrido más reranker a vectorial más reranker, ver V14). Esta tarea no cambió el despliegue.
- **Comprobar la política de datos de Nebius** frente a la declaración de privacidad del sitio antes de usarlo como segundo proveedor.
- Decidir si se añade caché del vector de la consulta por texto para recuperar la repetibilidad de búsquedas repetidas.

## Coste

Gasto real notificado por OpenRouter: 0.0018 USD en 4,899 llamadas registradas (`ledger.jsonl`; el tope era de 1.50 USD). Las 419 llamadas sin coste notificado son llamadas fallidas (tiempos agotados y errores HTTP), para las que OpenRouter no devuelve uso.

## Qué se leyó de la base de datos y cómo

Neon, solo lectura durante toda la tarea, con una sola conexión y sentencias secuenciales:
- 200 sentencias de búsqueda exacta (autorrecuperación por título);
- 120 sentencias de búsqueda exacta (calidad de Nebius y SiliconFlow);
- 9 SELECT de los vectores guardados de las 4,018 obras del pool.

No se escribió nada en Neon ni se creó ninguna tabla.

## Cómo reproducirlo

Desde la raíz del repositorio, con `.venv/bin/python <script>`. Las pruebas de las rondas 2 y 3 necesitan además `PYTHONPATH=data/evaluation/catalog50k/investigation`. Las llamadas a la API necesitan `OPENROUTER_API_KEY` y las lecturas de Neon necesitan acceso de solo lectura. Los scripts de análisis (`*_analyse*.py`) y los `.out` y `.json` guardados permiten revisar los resultados sin llamar a ninguna API.

```
.venv/bin/python data/evaluation/catalog50k/investigation/check_e2_index.py
.venv/bin/python data/evaluation/catalog50k/investigation/q2_analyse_operational.py
PYTHONPATH=data/evaluation/catalog50k/investigation .venv/bin/python data/evaluation/catalog50k/investigation/q2_round3.py
```

## Ficheros

Todos en `data/evaluation/catalog50k/investigation/`.

- Resultado completo: `FINDINGS.md`.
- Cliente y registro de gasto: `orclient.py`, `ledger.jsonl`.
- Pregunta 1: `dump_texts.py`, `check_e2_index.py` (y `.out`), `q1_batch_vs_single.py`, `q1_reference_check.py`, `q1_self_retrieval.py`, `q1_length_bias.py`, `q1_known_items.py`, `q1_pool_and_splits.py`, `fetch_pool_qwen_vectors.py`, con sus `.json`; `endpoints_baai__bge-m3.json`.
- Pregunta 2: `q2_analyse_operational.py`, `q2_determinism.py`, `q2_doc_variant.py`, `q2_variant_quality.py`, `q2_nebius_quality.py`, `q2_mitigations.py`, `q2_mitigations_analyse.py`, `q2_round2.py`, `q2_round2_analyse.py`, `q2_round3.py`, `q2_round3_analyse.py`; salidas `q2_failure_timing.out`, `q2_hedge_estimate_from_round1.out`, `q2_generation_metadata.out`, `q2_round3_timing.out`; llamadas con su vector `q2_mitigations_calls.jsonl`, `q2_round2_calls.jsonl`, `q2_round3_calls.jsonl`; `endpoints_qwen__qwen3-embedding-8b.json`.
- Ignorados por git (regenerables; la carpeta tiene su propio `.gitignore`): `work_texts.jsonl` (44 MB, textos del catálogo) y `pool_qwen_doc_vectors.*` (16 MB, vectores leídos de Neon).
- Contexto: `docs/notes/tareas/V14-evaluacion-catalogo-desplegado.md`, `docs/notes/tareas/V16-modelos-por-api.md`, `data/evaluation/CATALOG_EVALUATION.md`, `docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md` (sección 8).
