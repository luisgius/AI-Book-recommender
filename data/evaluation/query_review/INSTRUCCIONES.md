# Revisión de las consultas y peticiones generadas por un LLM

Las consultas de los estratos B, C y D de la evaluación sobre el catálogo
(V14) y las peticiones de los conjuntos E, F y G de la evaluación del
asistente (V22 y V25) las escribió un LLM. El protocolo de V14 lo declara
amenaza a la validez: «el autor debe revisar la lista» (sección 3). El del
asistente cuenta entre sus amenazas que «las peticiones de E y F las escribe
un LLM» (sección 8); G lo escribió el mismo generador (adenda, sección 3).
Esta hoja tiene una fila por cada una, 254 en total:

| Evaluación | Estrato | Filas |
|---|---|---|
| catalogo | B | 72 |
| catalogo | C | 40 |
| catalogo | D | 12 |
| asistente | E | 30 |
| asistente | F | 20 |
| asistente | G | 80 |

No están el estrato A (las consultas originales) ni los conjuntos B y D del
asistente, que son los estratos B y D del catálogo y se revisan una sola vez,
en las filas `catalogo`. E y F son iguales en las dos medidas del asistente
y aparecen una vez.

## Qué hay que hacer

Abra `review_sheet.csv` en una hoja de cálculo y escriba en la columna `verdict` uno
de los tres valores de la tabla; deje la celda vacía si no ha revisado la
fila. En `comment` explique cada `dudosa` o `mal`. No cambie ninguna otra
columna ni el `item_id`, y no borre ni añada filas.

| Veredicto | Significado |
|---|---|
| ok | Cumple lo que el protocolo pide a su estrato (lista de abajo) |
| dudosa | Lo cumple a medias, o lo cumple pero no parece algo que escribiría una persona |
| mal | No lo cumple: otro idioma, no corresponde a `need`, no es del tipo de `type` |

`text` es lo que recibe el sistema. `need` es la necesidad de información,
en inglés, contra la que valoran los jueces (en F, la propia petición; en G,
el tema que recibió el generador). `type` es el tipo que se pidió al
generador y `context` añade lo necesario para revisar algunas filas.

## Qué mirar en cada estrato

En todas: que `text` esté en el idioma de `language`, que corresponda a
`need` y que pudiera escribirlo una persona (el protocolo advierte que las
consultas generadas «pueden ser más regulares que las de usuarios reales»).

Catálogo (`docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`, sección 3):

- **B**: seis consultas por idioma en doce idiomas: tres temáticas
  (`thematic`), dos descriptivas («una novela sobre...», `descriptive`) y
  una de obra conocida (`known_item`), que debe describir la obra de
  `context` sin palabras de su título.
- **C**: diez necesidades escritas en español, inglés, francés y alemán; las
  cuatro versiones van seguidas (`c01-de` a `c01-fr`). Comparten los juicios,
  así que deben pedir exactamente lo mismo.
- **D**: peticiones que un catálogo de libros no puede atender (por ejemplo,
  el horario de un tren). Es `mal` si algún libro podría responderla.

Asistente (`docs/notes/2026-11-12-protocolo-evaluacion-asistente.md`,
sección 3, y su adenda, sección 3):

- **E**: petición vaga de una a tres palabras (en chino y japonés, hasta ocho
  caracteres) en el idioma de una necesidad de B que el asistente no ve
  (`need`; `context` da la consulta de B y `type` su tipo). Se espera que el
  asistente pregunte: debe ser compatible con `need`, pero tan general que
  haga falta preguntar.
- **F**: petición de varios pasos sobre obras del catálogo, escrita a partir
  de sus fichas: comparar dos obras (`compare`), «algo como X pero...»
  (`like_but`) o datos de una obra concreta (`facts`). Debe nombrar las obras
  de `context` y ser del tipo de `type`.
- **G**: 60 válidas (`valid`) sobre un tema no literario (`need`), la mitad
  como palabras sueltas, como en la caja de un buscador (`keywords`), y la
  mitad como una frase que pide algo para leer (`sentence`); y 20 fuera de
  alcance (`out_of_scope`, ids `g-oos`) que piden otra cosa que libros: un
  dato del momento, un cálculo, que se escriba un programa o una
  conversación sin petición (el `tipo` de `context`), sin mencionar libros
  ni lectura. Es `mal` una válida que no trate de su tema o una fuera de
  alcance que un catálogo de libros sí podría atender.

Si no lee el idioma de una fila, compruebe solo que `text` corresponde a
`need` (con un traductor o, en G, con la traducción del propio generador que
da `context`) y no valore si suena natural.

## Después

Guarde la hoja como CSV UTF-8 (en Excel, «CSV UTF-8») y ejecute desde la
raíz del repositorio:

    .venv/bin/python -m app.evaluation.query_review summary --sheet data/evaluation/query_review/review_sheet.csv

El comando rechaza la hoja si falta, sobra o cambió algún `item_id`, o si un
veredicto no es `ok`, `dudosa`, `mal` o vacío. Si la acepta, escribe
`data/evaluation/query_review/query_review.json`: por evaluación y estrato, cuántas filas hay, cuántas se
revisaron, cuántas de cada veredicto y la lista de las `dudosa` y `mal` con
su comentario. Puede ejecutarse con la hoja a medias.
