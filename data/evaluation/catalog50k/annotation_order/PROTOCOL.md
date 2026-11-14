# Protocolo de la anotación humana del orden de los sistemas (V31)

Fecha: 2026-11-14. Este documento se escribe **antes** de que nadie ponga una
nota, y debe guardarse en git antes de rellenar la hoja. Fija qué se anota,
qué se calcula con las notas y cómo se lee el resultado, de modo que el
resultado no pueda elegir la pregunta. La hoja ya está generada (sección 4),
porque solo necesita los ficheros guardados de V14; no se ha hecho ninguna
búsqueda nueva ni ninguna llamada a un modelo. Cualquier desviación posterior
se anotará en el informe con su motivo.

## 1. Por qué hace falta

La evaluación sobre el catálogo desplegado (V14, protocolo
`docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`, informe
`data/evaluation/CATALOG_EVALUATION.md`) mide la relevancia con dos jueces LLM
en una escala de 0 a 3. El autor anotó después 200 pares sacados al azar y
estratificados por la nota del juez principal (`data/evaluation/catalog50k/annotation/`):
el acuerdo con los jueces fue bajo (kappa ponderado cuadrático 0,272 y 0,301
en 184 pares). Esa muestra, por diseño, no sirve para recalcular ninguna
métrica de los sistemas: no contiene las listas de ningún sistema, sino pares
sueltos elegidos por su nota.

Una revisión externa pidió una anotación que sí pueda decidir: los primeros
resultados de los sistemas principales sobre un conjunto de consultas, para
ordenar los sistemas con notas humanas y comparar ese orden con el que dan
los jueces. La pregunta de este experimento es esa: **¿se mantiene con las
notas de una persona el orden de los sistemas que dan los jueces, en
particular el signo de H1 y de H3?** No pregunta si los jueces aciertan par a
par, aunque eso también se mide (sección 6, d).

Este experimento se diseñó después de conocer los resultados de V14 y la
anotación de 200 pares, a raíz de esa revisión. Se dice aquí y se dirá en el
informe.

## 2. Consultas: dos bloques

Son elegibles las consultas del conjunto confirmatorio de V14 (estratos A y
B) escritas en español o en inglés, los dos idiomas que el anotador lee sin
traducir: 56 consultas (inglés: 24 de A y 6 de B; español: 20 de A y 6 de B).
Quedan fuera los otros diez idiomas de B y los estratos C y D. La hoja las
reparte en dos bloques.

**Bloque 1 (obligatorio): 30 consultas.** Regla (fijada en `select_queries` de
`app/evaluation/catalog_evaluation/annotation_order.py`, semilla 0):

1. Las 30 se reparten entre los dos idiomas tan a partes iguales como lo
   permitan las disponibles (se da una consulta a cada idioma por turno,
   saltando el que no tenga más). Si hubiera 30 o menos elegibles, se tomarían
   todas y se diría cuántas.
2. Dentro de cada idioma se toma primero el estrato B (todas sus consultas, o
   una muestra con la semilla si pasaran de la cuota) y el estrato A completa
   la cuota con una muestra con la semilla (`random.Random(0).sample` sobre los
   identificadores ordenados). Se prefiere B porque las consultas de A son las
   44 de `app/evaluation/test_queries.json`, que también se usaron en tareas
   anteriores para elegir modelos y parámetros (V06, V10, V16).

**Bloque 2 (opcional): las otras 26 consultas elegibles** (`remaining_queries`).
Como el bloque 1 ya tomó todas las de B, las 26 son del estrato A.

Recuento (de `sample.json`):

| Idioma | Disponibles A | Disponibles B | Bloque 1: A | Bloque 1: B | Bloque 2: A | Total |
|---|---|---|---|---|---|---|
| en | 24 | 6 | 9 | 6 | 15 | 30 |
| es | 20 | 6 | 9 | 6 | 11 | 26 |
| Total | 44 | 12 | 18 | 12 | 26 | 56 |

Consultas del bloque 1:

- en, B: `b-en-1` a `b-en-6`.
- en, A: `q04`, `q16`, `q18`, `q24`, `q26`, `q30`, `q32`, `q35`, `q43`.
- es, B: `b-es-1` a `b-es-6`.
- es, A: `q03`, `q05`, `q10`, `q15`, `q25`, `q27`, `q33`, `q36`, `q42`.

Consultas del bloque 2:

- en, A: `q01`, `q06`, `q08`, `q09`, `q11`, `q13`, `q14`, `q20`, `q22`, `q28`,
  `q34`, `q38`, `q39`, `q40`, `q41`.
- es, A: `q02`, `q07`, `q12`, `q17`, `q19`, `q21`, `q23`, `q29`, `q31`, `q37`,
  `q44`.

Las doce de B incluyen las dos de obra conocida (`b-en-6`, `b-es-6`). Entre
las de A hay consultas muy cortas (por ejemplo `q35` «python», `q36`
«historia», `q38` «java», `q39` «apple»).

## 3. Sistemas

Los cuatro sistemas principales de V14, con sus definiciones exactas
(`app/evaluation/catalog_evaluation/design.py`) y sus ficheros de ejecución
guardados (`data/evaluation/catalog50k/runs/S2.jsonl` a `S5.jsonl`). No se hace
ninguna búsqueda nueva.

| Sistema | Definición |
|---|---|
| S2 | Vectorial, `qwen/qwen3-embedding-8b` (1024 d, proveedor fijado `deepinfra`), índice aproximado del esquema |
| S3 | Híbrido: RRF (k = 60, pesos iguales) de los 60 primeros de S1 (BM25) y de S2 |
| S4 | S2 + `qwen/qwen3-reranker-8b` (proveedor fijado `primeintellect`) sobre los 30 primeros |
| S5 | S3 + el mismo reranker sobre los 30 primeros (la configuración desplegada en V14) |

Contrastes que se comparan: H1 = S3 − S2 (¿la fusión híbrida añade algo a la
búsqueda vectorial sola?) y H3 = S5 − S4 (con reranker, ¿añaden algo los
candidatos de BM25?). En V14 los dos fueron negativos y se sostuvieron con los
dos jueces. Ninguna de las 56 consultas tiene error ni recurso en esos cuatro
ficheros. El análisis comprueba, antes de calcular nada, que la clave sigue
coincidiendo con los ficheros de ejecución.

## 4. Pares y hoja

- Pares: por consulta, la unión de los cinco primeros de S2, S3, S4 y S5. Cada
  par (consulta, obra) aparece una sola vez. Los dos jueces juzgaron todos los
  pares en V14 (el pool de V14 son los diez primeros de cada sistema).

| Bloque | Consultas | Pares | Pares por consulta (mín., mediana, máx.) | Ya en la hoja de 200 pares | Tiempo a 30-45 s por par |
|---|---|---|---|---|---|
| 1 | 30 | 313 | 8, 10, 14 | 15 | 156 a 235 min (2,6 a 3,9 h) |
| 2 | 26 | 294 | 7, 11,5, 15 | 14 | 147 a 220 min (2,5 a 3,7 h) |
| Total | 56 | 607 | | 29 | |

- Columnas de `annotation_order_sheet.csv`: `pair_id`, `block`, `query`, `need`,
  `title`, `authors`, `book_language`, `year`, `description`, `grade`,
  `grade_2`. `block` vale 1 o 2.
- La consulta (`query`), la necesidad (`need`, en inglés; en el estrato A es la
  propia consulta) y el título, los autores (separados por «; ») y la
  descripción se muestran exactamente como en la hoja de 200 pares: el texto de
  la obra sale de la misma función (`book_cells` de `annotation.py`). Ninguna
  fila tiene la descripción vacía.
- Cambios respecto a la hoja anterior: la columna `language` de aquella era el
  idioma de la consulta y aquí se quita, porque todas las consultas están en
  español o inglés y se leen directamente; en su lugar hay dos columnas de la
  obra, `book_language` (el código ISO guardado en el catálogo) y `year` (año
  de publicación; vacío en 68 filas del bloque 1 y 66 del bloque 2). El prompt
  de los jueces ya mostraba el idioma y el año de la obra (además de las
  categorías, que la hoja no muestra, como tampoco la anterior), así que la
  persona ve ahora casi lo mismo que vieron los jueces.
- Ciega: la hoja no muestra el sistema, la posición ni ninguna nota de los
  jueces. La clave `annotation_order_key.json` (para cada `pair_id`: consulta,
  necesidad, idioma, estrato, bloque, obra, posición en cada sistema que la
  devolvió y, si lo hay, el identificador del mismo par en la hoja de 200
  pares) está en un fichero aparte que el anotador no abre. `sample.json`
  guarda los recuentos y el cálculo de la sección 7, sin nada de ningún par.
- Orden: primero todas las filas del bloque 1 y después las del bloque 2. Cada
  bloque se baraja por separado con un generador nuevo `random.Random(0)`: sus
  consultas se barajan y, dentro de cada consulta, las obras se ordenan por
  identificador y se barajan con el mismo generador. El anotador lee una
  consulta cada vez. Los `pair_id` siguen el orden de la hoja: `r001` a `r313`
  en el bloque 1 y `r314` a `r607` en el bloque 2. Las filas del bloque 1 son
  idénticas a las de la versión de la hoja de un solo bloque salvo por la
  columna `block`, y volver a generar la hoja da ficheros idénticos byte a byte
  (las dos cosas comprobadas).
- La hoja se escribe en UTF-8 con marca de orden de bytes, para que la hoja de
  cálculo la abra como UTF-8. El lector acepta el fichero devuelto con o sin
  esa marca (`read_cells`, con `utf-8-sig`, igual que `read_sheet` en la hoja
  anterior).
- Solapamiento con la hoja de 200 pares: 15 pares del bloque 1 y 14 del
  bloque 2 ya estaban en aquella (los 29 tienen nota allí). Se quedan en la
  hoja nueva sin su nota anterior; la clave guarda su identificador antiguo
  para el análisis.

## 5. Escala e instrucciones

`INSTRUCCIONES.md` (generado con la hoja) dice al anotador qué hacer. La
escala de 0 a 3, su tabla y sus dos reglas («una obra en otro idioma que la
consulta puede ser relevante, salvo que la consulta pida un idioma»; «si falta
la descripción, valore por el título, los autores y lo que se sepa de la obra
por esos datos») tienen la misma redacción que en la hoja anterior, para que
las dos anotaciones sean comparables. Cambia lo siguiente:

- La lista de datos en que basar la nota incluye ahora el idioma de la obra y
  el año.
- Celda vacía: la hoja anterior decía «si no puede decidir, deje la celda
  vacía» (16 de 200 se quedaron vacías). Ahora: dejarla vacía **solo si la obra
  no se puede valorar con el texto que se muestra**; si se duda entre dos
  notas, se elige una. El motivo es que aquí un par vacío sale del nDCG de
  todos los evaluadores (sección 6).
- Bloques: el bloque 1 es obligatorio y se termina antes de empezar el 2. El
  bloque 2 es opcional: se hace entero o no se hace. En el bloque 2 una
  consulta solo cuenta para comparar los sistemas si todas sus filas tienen
  nota, así que las instrucciones piden terminar cada consulta empezada y
  avisan de que una celda vacía en el bloque 2, aunque sea de una obra que no
  se puede valorar, deja fuera esa consulta.
- La decisión de hacer el bloque 2 se toma sin mirar ningún resultado: el
  análisis se ejecuta una sola vez, al entregar la hoja (sección 6).
- Valorar sin consultar nada sobre los sistemas: no abrir la clave, las
  ejecuciones ni los informes, ni buscar las consultas en la aplicación.
- Los pares repetidos de la hoja anterior se valoran de nuevo sin mirar la
  nota antigua.
- Tiempo, por bloque, en la tabla de la sección 4. La hoja anterior no
  registró tiempos, así que su ritmo no se puede inferir; se usan 30 a 45
  segundos por fila.
- Segunda persona (opcional): la columna `grade_2`, con las mismas
  instrucciones, para todas las filas o para una parte. Para comparar sistemas
  solo cuentan sus consultas completas, en los dos bloques. Para que no vea las
  notas de la primera, anota sobre una copia de la hoja en blanco, que se pasa
  al análisis con `--second-sheet`.

## 6. Análisis declarado

**Se ejecuta una sola vez, cuando el anotador entrega la hoja; nunca entre el
bloque 1 y el bloque 2.**

    .venv/bin/python -m app.evaluation.catalog_evaluation order-agreement --sheet data/evaluation/catalog50k/annotation_order/annotation_order_sheet_filled.csv

Lee la hoja, la clave, los ficheros de ejecución (solo para comprobar la
clave), las cachés de juicios de V14 y la hoja de 200 pares rellenada
(`annotation/annotation_sheet_filled.csv`). Escribe `order_agreement.json` en
esta carpeta.

**Celdas vacías.** Como en el análisis de la hoja anterior, una celda vacía no
cuenta. En concreto: el par sale de todas las medidas de acuerdo; en el nDCG@5
cuenta como no juzgado (ganancia 0 y fuera de la ordenación ideal) **para la
persona y para cada juez por igual**, de modo que los tres evaluadores se
puntúan sobre los mismos pares; una celda con algo distinto de 0, 1, 2 o 3
cuenta como vacía y se lista por `pair_id` para poder corregirla. El comando
informa del número de celdas vacías y no válidas, en total, por bloque y por
consulta.

**Qué consultas entran en las cifras de los sistemas** (`analysis_queries`).
Una consulta está *completa* si todos sus pares tienen nota, *a medias* si
algunos sí y otros no, y *sin empezar* si ninguno.

- Bloque 1: entra toda consulta con algún par con nota. Como el bloque es
  obligatorio, una celda vacía ahí es una obra que no se pudo valorar, no una
  consulta sin terminar.
- Bloque 2: solo entran las consultas completas. Como el bloque es opcional,
  una celda vacía ahí puede significar que el anotador se paró; una consulta a
  medias del bloque 2 no se usa nunca para las cifras de los sistemas.
- **Conjunto principal.** Si todas las consultas del bloque 2 están completas,
  el análisis principal usa las 56 consultas (bloques 1 y 2), y el bloque 1
  solo se informa como tabla secundaria. Si el bloque 2 está vacío o
  incompleto, el análisis principal usa solo el bloque 1, y las consultas
  completas del bloque 2, si las hay, se informan aparte como descriptivas.
- Esta elección no sesga la comparación: depende solo de cuántas consultas
  del bloque 2 se completaron, y el anotador decide si hace el bloque 2 sin
  ver ningún resultado, porque el análisis se ejecuta una sola vez, al
  entregar la hoja, y la hoja no muestra el sistema, la posición ni las notas
  de los jueces. No puede, por tanto, seguir anotando o pararse según le
  convenga a una conclusión.
- Para el segundo anotador se aplica la misma regla de conjunto principal,
  pero solo con consultas completas también en el bloque 1, porque puede
  haber anotado solo una parte de cualquier bloque.

Con el conjunto principal se calcula:

a. **nDCG@5 por sistema**, con las notas de la persona y, sobre las mismas
   consultas, los mismos pares y la misma profundidad, con las de cada juez.
   Ganancia 2^nota − 1 y descuento log2, la función de
   `app/evaluation/evaluation_service.py`. La ordenación ideal se forma con los
   pares con nota de la consulta en esta hoja (la unión de los cuatro top 5),
   no con el pool de V14; por eso estas cifras no se comparan con el nDCG@10 de
   V14. Media sobre las consultas con intervalo del 95 % por bootstrap de
   percentiles (10 000 remuestreos, semilla 0).
b. **H1 = S3 − S2 y H3 = S5 − S4** con las notas de la persona: diferencia media,
   intervalo del 95 % por bootstrap pareado y p de la prueba de aleatorización
   pareada de dos colas, con `app/evaluation/statistics.py` (10 000
   remuestreos, semilla 0), sin corrección de Holm. Son **descriptivas**: la
   potencia es baja (sección 7). Lo mismo con cada juez.
c. **Signo**: si el signo de H1 y el de H3 con la persona coinciden con el de
   cada juez. **Tau de Kendall** (tau-b) entre el orden de los cuatro sistemas
   (por nDCG@5 medio) de la persona y el de cada juez, y entre los dos jueces
   como referencia. Con cuatro sistemas la tau solo toma unos pocos valores
   (sin empates: −1, −2/3, −1/3, 0, 1/3, 2/3 y 1); intercambiar dos sistemas
   vecinos ya la baja a 2/3.

Las mismas cifras (a) a (c) se calculan con la tabla secundaria (bloque 1
solo) o con las consultas descriptivas del bloque 2, según el caso, sin
aplicarles la regla de lectura.

d. **Acuerdo por par** de la persona con cada juez en todos los pares con nota
   de la hoja, de los dos bloques (incluidos los pares con nota de una consulta
   a medias: el acuerdo por par no es una cifra de los sistemas): kappa
   ponderado cuadrático (intervalo del 95 % por bootstrap sobre pares, 2000
   remuestreos, semilla 0), acuerdo exacto y acuerdo a menos de un punto, con
   `agreement_report` como en la hoja anterior, también por idioma de la
   consulta. **Acuerdo del anotador consigo mismo** en los pares que ya estaban
   en la hoja de 200 pares (15 en el bloque 1 y 14 en el bloque 2): nota
   antigua frente a nota nueva, con las mismas medidas; con tan pocos pares es
   solo descriptivo.
e. Si `grade_2` tiene alguna nota: acuerdo entre las dos personas en los pares
   que ambas valoraron (mismas medidas que en d), y (a) a (d) repetidos con
   `grade_2` en lugar de `grade` sobre su propio conjunto principal; además,
   signo de H1 y H3 y tau entre las dos personas sobre las consultas que están
   en el conjunto principal de las dos, con los pares que ambas valoraron.

## 7. Potencia: diferencia mínima detectable

La diferencia mínima detectable de una prueba pareada de dos colas con nivel
0,05 y potencia 0,8, con n consultas, es

    (t_{0,975; n−1} + t_{0,80; n−1}) × s / √n

- con 30 consultas: (2,045 + 0,854) × s / 5,477 ≈ 0,529 × s;
- con 56 consultas: (2,004 + 0,848) × s / 7,483 ≈ 0,381 × s;

donde s es la desviación típica por consulta de las diferencias. Se toma s de
las diferencias de nDCG@5 con cada juez sobre esas mismas consultas y todos
sus pares (todos juzgados), calculadas antes de anotar (`sample.json`, bloque
`power`). Es la aproximación de la prueba t; la prueba de aleatorización tiene
una potencia parecida.

| Consultas | Contraste | Juez | Diferencia media de los jueces | Desviación típica por consulta | Diferencia mínima detectable |
|---|---|---|---|---|---|
| 30 (bloque 1) | H1 (S3 − S2) | `z-ai/glm-5.3-flash` | −0,110 | 0,300 | 0,159 |
| 30 (bloque 1) | H1 (S3 − S2) | `minimax/minimax-m3.1-flash` | −0,078 | 0,245 | 0,130 |
| 30 (bloque 1) | H3 (S5 − S4) | `z-ai/glm-5.3-flash` | −0,005 | 0,081 | 0,043 |
| 30 (bloque 1) | H3 (S5 − S4) | `minimax/minimax-m3.1-flash` | −0,022 | 0,076 | 0,040 |
| 56 (bloques 1 y 2) | H1 (S3 − S2) | `z-ai/glm-5.3-flash` | −0,065 | 0,272 | 0,104 |
| 56 (bloques 1 y 2) | H1 (S3 − S2) | `minimax/minimax-m3.1-flash` | −0,070 | 0,201 | 0,076 |
| 56 (bloques 1 y 2) | H3 (S5 − S4) | `z-ai/glm-5.3-flash` | −0,031 | 0,118 | 0,045 |
| 56 (bloques 1 y 2) | H3 (S5 − S4) | `minimax/minimax-m3.1-flash` | −0,034 | 0,163 | 0,062 |

nDCG@5 medio de los jueces (principal y segundo):

| Sistema | 30 consultas | 56 consultas |
|---|---|---|
| S2 | 0,737 y 0,750 | 0,707 y 0,732 |
| S3 | 0,627 y 0,673 | 0,642 y 0,662 |
| S4 | 0,843 y 0,850 | 0,833 y 0,821 |
| S5 | 0,839 y 0,827 | 0,802 y 0,787 |

Los dos jueces dan el mismo orden en los dos conjuntos: S4, S5, S2, S3.

Qué implica:

- En los dos conjuntos, las diferencias de los propios jueces quedan **por
  debajo** de la diferencia mínima detectable, en H1 y en H3 (con 56 consultas,
  H1 con el segundo juez se acerca: −0,070 frente a 0,076). Aunque la persona
  coincida con los jueces, lo esperable es que sus diferencias no sean
  significativas: por eso la p y el intervalo son descriptivos y la regla de
  lectura (sección 8) se basa en el signo.
- Con 56 consultas la diferencia mínima detectable de H1 baja (de 0,159 y
  0,130 a 0,104 y 0,076). La de H3 no baja, porque las consultas del bloque 2
  varían más (la desviación típica sube de 0,081 y 0,076 a 0,118 y 0,163).
- H3 es frágil en el bloque 1: con los jueces su diferencia es casi nula
  (−0,005 y −0,022). Con las 56 consultas es mayor (−0,031 y −0,034, cerca de
  las de V14 con nDCG@10 en 116 consultas, −0,029 y −0,035), así que su signo
  depende menos de unas pocas notas si se hace el bloque 2. En el bloque 1
  solo, un signo de H3 que coincide es una prueba débil a favor del orden de
  los jueces, y uno que no coincide, una prueba débil en contra.
- La desviación típica con las notas de la persona puede ser distinta de la de
  los jueces (en la hoja anterior el autor casi no usó la nota 3), así que las
  cifras son una orientación, no una garantía.

## 8. Regla de lectura

Escrita antes de anotar. Se aplica al conjunto principal del primer anotador
(sección 6):

- Para cada uno de H1 y H3: si el signo de la diferencia con las notas de la
  persona es el mismo que con cada uno de los dos jueces (sobre los mismos
  pares), la memoria puede decir que ese resultado se mantiene en esta muestra
  con las notas de una persona. Si se cumple en los dos, puede decir que **el
  orden de los sistemas se mantiene en esta muestra con las notas de una
  persona**. En los dos conjuntos posibles los dos jueces dan H1 y H3
  negativos, así que coincidir significa que la persona también los da
  negativos.
- Si un signo es distinto, o la diferencia con la persona es exactamente 0, la
  memoria debe decirlo y tratar la conclusión correspondiente de V14 (H1: la
  fusión híbrida no supera a la vectorial sola; H3: los candidatos de BM25
  empeoran el resultado con reranker) como **no confirmada por una persona**.
- En los dos casos el resultado se informa, con las diferencias, los
  intervalos, las p, la diferencia mínima detectable y qué conjunto fue el
  principal. Que un signo coincida no significa que la diferencia sea
  significativa.
- La tabla secundaria (bloque 1 solo) o las consultas descriptivas del bloque 2
  se informan al lado, pero no cambian la regla.
- La tau de Kendall y el acuerdo por par se informan, pero no cambian la regla.
- Si hay segundo anotador, sus cifras se informan al lado, con su cobertura.

## 9. Límites

- **Un solo anotador, y es el autor del sistema.** Conoce los sistemas, las
  conclusiones de V14 y este protocolo, aunque la hoja no le muestra el sistema,
  la posición ni las notas de los jueces de cada fila. La columna `grade_2` deja
  sitio para una segunda persona, que es opcional.
- **Potencia baja** con 30 consultas y también con 56 (sección 7); los
  resultados son descriptivos.
- **Solo consultas en español e inglés**: no dice nada de los otros diez
  idiomas del catálogo.
- 18 de las 30 consultas del bloque 1 y las 26 del bloque 2 son del estrato A,
  que ya se usó para elegir modelos y parámetros en tareas anteriores; si se
  hace el bloque 2, 44 de las 56 consultas del análisis principal son de A. Las
  12 de B las escribió un LLM (V14).
- **Profundidad 5**: el nDCG@5 de esta hoja, con la ordenación ideal formada
  con la unión de los cuatro top 5, no es la métrica de V14 (nDCG@10 con el pool
  completo). Un signo que coincide a profundidad 5 no dice lo mismo a
  profundidad 10.
- En el bloque 2, una obra que no se puede valorar deja fuera su consulta
  entera (sección 6); es el precio de no confundir una celda vacía con una
  consulta sin terminar.
- El acuerdo del anotador consigo mismo mezcla su estabilidad con los cambios
  de la hoja (idioma y año de la obra, regla de celda vacía más estricta), y
  15 o 29 pares son muy pocos.
- Los jueces vieron también las categorías de la obra; la persona no, como en
  la hoja anterior.
- En la hoja anterior el acuerdo del autor con los jueces fue bajo y el autor
  casi no dio notas de 3; los valores de nDCG@5 con sus notas pueden tener otra
  escala. La comparación es del orden y del signo, no de los valores.
- El experimento se diseñó después de conocer los resultados de V14 y de la
  anotación de 200 pares.

## 10. Ficheros y comandos

- En esta carpeta: `PROTOCOL.md` (este documento), `INSTRUCCIONES.md`,
  `annotation_order_sheet.csv` (la hoja en blanco, que se queda como se
  generó), `annotation_order_key.json` (la clave), `sample.json` (muestra de
  los dos bloques, recuentos, estimación de tiempo y potencia). Cuando vuelva
  la hoja: `annotation_order_sheet_filled.csv` y `order_agreement.json`.
- Código: `app/evaluation/catalog_evaluation/annotation_order.py` y los
  subcomandos `order-sheet` y `order-agreement` de `cli.py`. En
  `annotation.py` se separaron, sin cambiar su resultado, el texto de la obra
  (`book_cells`) y la lectura de una columna de la hoja (`read_cells`,
  `read_sheet` con columna); el acuerdo de la hoja de 200 pares recalculado con
  ese código es idéntico al guardado.
- Tests, sin red: `tests/evaluation/test_annotation_order.py`.
- Generar la hoja (lee la copia local del catálogo, las ejecuciones y las
  cachés de juicios; sin red):

      .venv/bin/python -m app.evaluation.catalog_evaluation order-sheet

- Analizar la hoja rellenada, una sola vez, al entregarla (sin red):

      .venv/bin/python -m app.evaluation.catalog_evaluation order-agreement --sheet data/evaluation/catalog50k/annotation_order/annotation_order_sheet_filled.csv [--second-sheet <copia de la segunda persona>]
