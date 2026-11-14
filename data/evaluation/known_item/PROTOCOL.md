# Protocolo de la evaluación de búsqueda de un libro conocido por título y autor (V29, parte B)

Fecha: 2026-11-14. Este documento se escribe y se guarda en git **antes** de
ejecutar la medición. Fija qué se va a medir, con qué datos, qué contrastes
se van a probar y qué decisión depende de cada resultado, de modo que los
resultados no puedan elegir las preguntas. La muestra de libros ya está
sacada (sección 3), porque solo necesita leer la copia local del catálogo;
no se ha ejecutado ningún sistema sobre ella. Cualquier desviación posterior
se anotará en el informe (`data/evaluation/known_item/KNOWN_ITEM.md`) con su
motivo.

## 1. Por qué hace falta

La evaluación sobre el catálogo desplegado (V14, protocolo
`docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`) concluyó que añadir
los candidatos de BM25 a los vectoriales (RRF con pesos iguales, k = 60) baja
el nDCG@10 (contrastes H1 y H3), y el modo de candidatos por defecto se
cambió por ese resultado (V24). Una revisión externa del 2026-11-13 objetó,
entre otras cosas, que ninguna consulta de V14 pide un libro por su título o
por su autor, que es justo donde se espera que la búsqueda léxica ayude: las
consultas de «obra conocida» se escribieron sin palabras del título.

Este experimento mide ese caso. La verdad de referencia sale del propio
catálogo, así que no interviene ningún juez LLM ni ninguna valoración humana.

Este experimento se diseñó después de conocer los resultados de V14, a raíz
de esa revisión. Se dice aquí y en el informe.

## 2. Colección y sistemas

- Colección: el catálogo desplegado, esquema `catalog_deploy` de Neon, 50 000
  obras, el mismo de V14. No se modifica durante la medición. La muestra y las
  obras relevantes se leen de su copia local (`data/catalog_deploy/catalog.db`,
  la fuente del cargador), que V14 comprobó que tiene los mismos
  identificadores y los mismos textos que Neon.
- Sistemas: los de V14 que no usan un LLM, con sus nombres y definiciones
  exactos (`app/evaluation/catalog_evaluation/design.py`), más dos de fusión
  ponderada:

| Sistema | Definición |
|---|---|
| S1 | BM25 en Postgres (índice `lakebase_bm25` de `catalog_deploy.work_texts`) |
| S2 | Vectorial, `qwen/qwen3-embedding-8b` (1024 d, proveedor fijado `deepinfra`), índice aproximado del esquema |
| S3 | Híbrido: RRF (k = 60, pesos iguales) de los 60 primeros de S1 y de S2 |
| S4 | S2 + `qwen/qwen3-reranker-8b` (proveedor fijado `primeintellect`) sobre los 30 primeros |
| S5 | S3 + el mismo reranker sobre los 30 primeros (la configuración que estaba desplegada en V14) |
| S3w | RRF ponderado (k = 60) de los 60 primeros de S1 y de S2, con peso léxico w y peso vectorial 1 − w |
| S5w | S3w + el mismo reranker sobre los 30 primeros |

- RRF ponderado: la puntuación de un libro es w / (60 + posición en S1) +
  (1 − w) / (60 + posición en S2). Con w = 0,5 da el mismo orden que S3. Los
  empates se resuelven como en S3 (primero el orden léxico). Es una función
  pura del paquete de evaluación (`app/evaluation/known_item/fusion.py`); no
  cambia la fusión del dominio ni sus valores por defecto.
- No se evalúan S6 (el LLM como reordenador), E1 ni E2: el experimento no
  hace llamadas de generación y la pregunta es solo sobre los candidatos
  léxicos.
- Embedding de la consulta congelado, como en V14: cada consulta se codifica
  una vez y ese vector lo usan todos los sistemas.
- Qué indexa cada rama (comprobado en el código): BM25 indexa en
  `work_texts.tsv` el título, los autores y las categorías sin lematizar y la
  descripción lematizada en su idioma
  (`app/infrastructure/search/postgres_lexical_search_repository.py`). El
  vector de cada obra se calcula sobre `Book.get_searchable_text()`, que
  concatena título, autores, descripción y categorías
  (`scripts/load_postgres.py`). El reranker recibe ese mismo texto cortado a
  2000 caracteres (`app/infrastructure/search/api_reranker.py`). Es decir, el
  título y los autores están en el texto que indexan las dos ramas y en el
  que lee el reranker.

## 3. Muestra de libros

Regla (fijada en `app/evaluation/known_item/sampling.py`, semilla 29):

1. Un libro es elegible si su idioma es uno de los doce de V14 (en, es, fr,
   de, it, pt, ca, nl, ru, zh, ja, ar), tiene título, su primer autor nombra a
   una persona y su título normalizado es único en todo el catálogo. Normalizar
   es: forma Unicode NFKC, minúsculas, cada tramo de caracteres que no son
   letras ni dígitos se cambia por un espacio y se recortan los extremos; las
   tildes se conservan. No nombran a una persona los valores «Unknown»,
   «Anonymous», «Various», «Various authors» y «Group of authors», que se
   encontraron listando los nombres de autor del catálogo; un libro con uno de
   ellos como primer autor cuenta como libro sin autor.
2. Por idioma, los elegibles se ordenan por identificador y se barajan con un
   generador aleatorio de semilla 29. Se toman en ese orden, saltando un libro
   cuyo primer autor normalizado ya se tomó (en cualquier idioma), hasta tener
   25 por idioma: 300 libros. El salto hace que las 300 consultas de autor sean
   distintas.
3. Reparto: por idioma, los 25 libros se barajan otra vez con el mismo
   generador; a la mitad de desarrollo van los 13 primeros en los idiomas de
   posición par de la lista y los 12 primeros en los de posición impar. Quedan
   150 libros de desarrollo y 150 de prueba, con 12 o 13 por idioma en cada
   mitad.

Recuento (de `data/evaluation/known_item/sample.json`):

| Idioma | Obras | Elegibles | Sin autor | Título repetido | Saltados por autor repetido | Desarrollo | Prueba |
|---|---|---|---|---|---|---|---|
| en | 20 924 | 19 276 | 920 | 727 | 0 | 13 | 12 |
| es | 7 231 | 5 791 | 1 320 | 120 | 0 | 12 | 13 |
| fr | 3 099 | 2 900 | 121 | 78 | 2 | 13 | 12 |
| de | 2 058 | 1 883 | 125 | 50 | 4 | 12 | 13 |
| it | 1 750 | 1 602 | 115 | 33 | 2 | 13 | 12 |
| pt | 1 750 | 1 630 | 101 | 19 | 3 | 12 | 13 |
| ca | 971 | 729 | 233 | 9 | 2 | 13 | 12 |
| nl | 1 750 | 1 572 | 145 | 33 | 0 | 12 | 13 |
| ru | 1 750 | 1 573 | 137 | 40 | 8 | 13 | 12 |
| zh | 1 617 | 1 067 | 490 | 60 | 4 | 12 | 13 |
| ja | 1 750 | 1 304 | 439 | 7 | 0 | 13 | 12 |
| ar | 1 750 | 1 496 | 236 | 18 | 3 | 12 | 13 |

Además hay 3 600 obras en otros idiomas, que no entran. Un libro en inglés
tiene título vacío.

## 4. Consultas y relevancia

Tres consultas por libro, construidas mecánicamente, 900 en total:

| Tipo | Texto | Obras relevantes |
|---|---|---|
| T | El título tal como está guardado | El propio libro (una sola) |
| TA | El título, un espacio y el primer autor | El propio libro (una sola) |
| A | El nombre del primer autor, solo | Toda obra del catálogo cuya lista de autores contiene un autor con el mismo nombre normalizado |

En las consultas A, el número de obras relevantes va de 1 a 238 (mediana 3;
111 autores tienen una sola obra en el catálogo).

## 5. Métricas

- T y TA: MRR@10, éxito@1 y éxito@10. La métrica principal es MRR@10.
- A: nDCG@10 con relevancia binaria y éxito@10. La métrica principal es
  nDCG@10. Con relevancia binaria, el nDCG ideal tiene en cuenta como mucho
  diez obras relevantes.
- Se usan las funciones de `app/evaluation/evaluation_service.py`, como en V14.
- Intervalos de confianza del 95 % por bootstrap de percentiles sobre las
  consultas, 10 000 remuestreos, semilla 0, como en V14.

## 6. Elección del peso de la fusión ponderada

- Rejilla: w ∈ {0,1; 0,25; 0,75; 0,9}.
- Se elige en la mitad de desarrollo, con S3w (sin reranker), el w que
  maximiza la media de tres números: el MRR@10 medio de las consultas T, el
  MRR@10 medio de las TA y el nDCG@10 medio de las A. Un empate se resuelve a
  favor del w más pequeño.
- El mismo w se usa en S3w y en S5w. Se elige sin reranker para no pagar el
  reranker cuatro veces en la mitad de desarrollo; es una simplificación y
  puede no ser el mejor w para S5w.
- El w elegido se guarda en `weight_selection.json` antes de reordenar ninguna
  consulta de la mitad de prueba, y no se cambia después.
- La mitad de desarrollo se informa aparte (S1, S2, S3 y S3w con cada w, sin
  reranker). Todas las cifras de la tabla principal salen de la mitad de
  prueba: 150 libros, 450 consultas, todos los sistemas.

## 7. Contrastes declarados

Seis contrastes, en la mitad de prueba, con el reranker en los dos lados y la
métrica principal de cada tipo de consulta:

| Contraste | Tipo | Diferencia |
|---|---|---|
| C1-T | T | S5 − S4 (RRF con pesos iguales menos solo vectorial) |
| C2-T | T | S5w − S4 (fusión ponderada menos solo vectorial) |
| C1-TA | TA | S5 − S4 |
| C2-TA | TA | S5w − S4 |
| C1-A | A | S5 − S4 |
| C2-A | A | S5w − S4 |

- Prueba de aleatorización pareada de dos colas y corrección de Holm sobre
  los seis, con el mismo código y los mismos parámetros que V14
  (`app/evaluation/statistics.py`, 10 000 remuestreos, semilla 0). Nivel 0,05.
- Lectura secundaria, fuera de la familia y sin corregir: las mismas seis
  diferencias sin las consultas en chino y japonés, porque el índice BM25 de
  Postgres no segmenta esos textos.
- Las tablas muestran también S1, S2, S3 y S3w, pero sus diferencias son
  descriptivas.

## 8. Regla de decisión

Escrita antes de medir:

- Si, para un tipo de consulta, una fusión (C1 o C2) es significativamente
  mejor que solo vectorial (diferencia positiva y p de Holm < 0,05), la
  memoria debe limitar su conclusión a las consultas descriptivas y decir que,
  para ese tipo de consulta, los candidatos léxicos ayudan.
- Si ningún contraste de ese tipo es significativo, o la fusión es peor, la
  conclusión de V14 se extiende a las consultas por título y por autor sobre
  este catálogo.
- En los dos casos el resultado se informa.

## 9. Límites

- Un título exacto tal como está guardado es el caso más favorable para la
  búsqueda léxica. Los usuarios reales escriben títulos parciales o con
  errores, que este experimento no mide.
- El título y los autores forman parte del texto que indexan las dos ramas
  (sección 2), así que la rama vectorial también ve el título.
- La muestra es de libros del catálogo, no de peticiones de usuarios: dice
  cómo se encuentra un libro que existe, no con qué frecuencia la gente busca
  así.
- La coincidencia de autor es exacta tras normalizar: «Brothers Grimm» y
  «Jacob Grimm» son autores distintos, y una obra con el nombre escrito de
  otra forma no cuenta como relevante.
- Muchos nombres de autor del catálogo están en alfabeto latino aunque la
  obra sea árabe, china, japonesa o rusa (por ejemplo, 24 de las 25 consultas
  A en árabe y las 25 en chino son ASCII). Las consultas A de esos idiomas
  prueban sobre todo la búsqueda de nombres latinos.
- En chino y japonés, BM25 trata un título sin espacios como una sola
  palabra: un título exacto puede coincidir entero, pero cualquier variación
  no.
- El experimento se diseñó después de conocer los resultados de V14, a raíz
  de una revisión externa del 2026-11-13.

## 10. Coste, tope y ejecución

- Las únicas llamadas de pago son los embeddings de las 900 consultas y el
  reranker. No hay generación con LLM ni jueces.
- Embeddings: unos 0,0003 USD (V14 pagó 0,0002 USD por 672 consultas).
- Reranker, solo en la mitad de prueba: S4, S5 y S5w reordenan los 30
  primeros de S2, S3 y S3w. El cross-encoder puntúa cada par (consulta,
  libro) por separado, así que cada libro de la unión de las tres cabezas de
  una consulta se puntúa una sola vez (en peticiones de 30 libros como
  máximo) y cada cabeza se ordena después con esas puntuaciones, con el mismo
  desempate que el adaptador. Al precio observado en V14 (0,0693 USD por
  millón de caracteres enviados), con unos 45 a 50 libros distintos por
  consulta, la estimación es de 1,1 a 1,4 USD.
- Tope: 1,50 USD, con el registro de gasto de V14
  (`app/evaluation/catalog_evaluation/ledger.py`), en
  `data/evaluation/known_item/spend_ledger.jsonl`. Antes de cada llamada de
  pago se comprueba el gasto registrado y la ejecución se para al llegar al
  tope. Además, antes de reordenar, el programa calcula el coste del reranker
  a partir de la longitud de los textos que va a enviar y no empieza si supera
  lo que queda bajo el tope. El tope previsto al encargar el experimento era
  de 1 USD; se subió a 1,50 USD antes de guardar este protocolo porque la
  estimación del reranker lo superaba. Si la estimación hecha antes de
  reordenar supera lo que queda bajo el tope, no se reordena nada y se decide
  de nuevo; no se reordena una parte de la mitad de prueba.
- Una sola puntuación por par: puntuar una vez cada libro de la unión y
  ordenar con esa puntuación las tres cabezas supone que el reranker puntúa
  cada par (consulta, libro) sin mirar los demás libros de la petición. Es lo
  esperable de un cross-encoder, pero no se comprueba en este experimento.
- Orden de la ejecución (`python -m app.evaluation.known_item run`, reanudable):
  embeddings; S1, S2 y sus 60 primeros en Postgres, una consulta cada vez en
  una sola conexión; S3 y S3w con cada w; elección de w en desarrollo;
  estimación; reranker en la mitad de prueba. Después,
  `python -m app.evaluation.known_item report` escribe `results.json` y
  `KNOWN_ITEM.md` a partir de los ficheros guardados, sin red.
- Si el reranker falla tres veces en una consulta, se guarda el orden base y
  el error, como en V14, y el informe cuenta esos casos.
