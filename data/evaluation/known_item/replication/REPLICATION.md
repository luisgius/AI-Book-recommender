# Réplica de la evaluación de búsqueda de un libro conocido (V29, parte B)

## Por qué se repitió

La primera ejecución tardó unas tres horas y media en lugar de minutos. El registro de energía del sistema (`pmset -g log`) mostró después la causa: el Mac entró en reposo por inactividad a las 11:02 y siguió en reposo hasta después de la ejecución, despertando solo unos 45 segundos cada 15 minutos, aunque estaba conectado a la corriente. Ningún resultado de la primera ejecución se perdió ni se cambió.

La primera ejecución sigue siendo el resultado que se informa (`data/evaluation/known_item/KNOWN_ITEM.md`). Esta es una repetición con la máquina despierta (`caffeinate -i`), con la misma muestra, las mismas consultas y la misma división en mitades, que recalcula todo lo demás: embeddings de consulta, S1, S2, fusiones, elección del peso y reranker. Se informa junto a la primera, sea cual sea su resultado.

Carpeta original: `data/evaluation/known_item`. Carpeta de la réplica: `data/evaluation/known_item/replication`.

## Comparación

### Listas guardadas por sistema

Para cada sistema, cuántas de las consultas tienen exactamente la misma lista de ids en las dos ejecuciones (los diez primeros y la lista completa). S4, S5 y S5w solo existen en la mitad de prueba.

| Sistema | Consultas | Top 10 idéntico | Top 10 distinto | Top 10 distinto (prueba) | Lista completa idéntica | Lista completa distinta |
|---|---|---|---|---|---|---|
| S1 | 900 | 900 | 0 | 0 | 900 | 0 |
| S1@60 | 900 | 900 | 0 | 0 | 900 | 0 |
| S2 | 900 | 405 | 495 | 238 | 339 | 561 |
| S2@60 | 900 | 405 | 495 | 238 | 339 | 561 |
| S3 | 900 | 529 | 371 | 183 | 339 | 561 |
| S3w@0.1 | 900 | 420 | 480 | 234 | 339 | 561 |
| S3w@0.25 | 900 | 439 | 461 | 221 | 339 | 561 |
| S3w@0.75 | 900 | 639 | 261 | 132 | 339 | 561 |
| S3w@0.9 | 900 | 723 | 177 | 99 | 339 | 561 |
| S3w | 900 | 439 | 461 | 221 | 339 | 561 |
| S4 | 450 | 350 | 100 | 100 | 181 | 269 |
| S5 | 450 | 375 | 75 | 75 | 181 | 269 |
| S5w | 450 | 363 | 87 | 87 | 181 | 269 |

### Peso elegido en la mitad de desarrollo

Original: w = 0.25. Réplica: w = 0.25. Es el mismo peso.

| Ejecución | Objetivo original | Objetivo réplica |
|---|---|---|
| S3w@0.1 | 0.9465 | 0.9468 |
| S3w@0.25 | 0.9512 | 0.9514 |
| S3w@0.75 | 0.9385 | 0.9379 |
| S3w@0.9 | 0.9313 | 0.9311 |

### Métrica principal en la mitad de prueba

Media de la métrica principal de cada tipo de consulta en las dos ejecuciones, cuántas consultas cambian de valor y el mayor cambio de una consulta (réplica menos original).

| Tipo | Sistema | Métrica | n | Original | Réplica | Consultas con cambio | Mayor cambio |
|---|---|---|---|---|---|---|---|
| T | S1 | MRR@10 | 150 | 0.9191 | 0.9191 | 0 | +0.000 |
| T | S1@60 | MRR@10 | 150 | 0.9191 | 0.9191 | 0 | +0.000 |
| T | S2 | MRR@10 | 150 | 0.9732 | 0.9732 | 0 | +0.000 |
| T | S2@60 | MRR@10 | 150 | 0.9732 | 0.9732 | 0 | +0.000 |
| T | S3 | MRR@10 | 150 | 0.9722 | 0.9722 | 0 | +0.000 |
| T | S3w@0.1 | MRR@10 | 150 | 0.9811 | 0.9811 | 0 | +0.000 |
| T | S3w@0.25 | MRR@10 | 150 | 0.9863 | 0.9863 | 0 | +0.000 |
| T | S3w@0.75 | MRR@10 | 150 | 0.9650 | 0.9639 | 1 | -0.167 |
| T | S3w@0.9 | MRR@10 | 150 | 0.9636 | 0.9636 | 0 | +0.000 |
| T | S3w | MRR@10 | 150 | 0.9863 | 0.9863 | 0 | +0.000 |
| T | S4 | MRR@10 | 150 | 0.9789 | 0.9789 | 0 | +0.000 |
| T | S5 | MRR@10 | 150 | 0.9800 | 0.9800 | 0 | +0.000 |
| T | S5w | MRR@10 | 150 | 0.9789 | 0.9789 | 0 | +0.000 |
| TA | S1 | MRR@10 | 150 | 0.9667 | 0.9667 | 0 | +0.000 |
| TA | S1@60 | MRR@10 | 150 | 0.9667 | 0.9667 | 0 | +0.000 |
| TA | S2 | MRR@10 | 150 | 0.9933 | 0.9900 | 1 | -0.500 |
| TA | S2@60 | MRR@10 | 150 | 0.9933 | 0.9900 | 1 | -0.500 |
| TA | S3 | MRR@10 | 150 | 0.9858 | 0.9858 | 0 | +0.000 |
| TA | S3w@0.1 | MRR@10 | 150 | 0.9900 | 0.9867 | 1 | -0.500 |
| TA | S3w@0.25 | MRR@10 | 150 | 0.9856 | 0.9822 | 1 | -0.500 |
| TA | S3w@0.75 | MRR@10 | 150 | 0.9800 | 0.9800 | 0 | +0.000 |
| TA | S3w@0.9 | MRR@10 | 150 | 0.9733 | 0.9733 | 0 | +0.000 |
| TA | S3w | MRR@10 | 150 | 0.9856 | 0.9822 | 1 | -0.500 |
| TA | S4 | MRR@10 | 150 | 0.9967 | 0.9967 | 0 | +0.000 |
| TA | S5 | MRR@10 | 150 | 0.9967 | 0.9967 | 0 | +0.000 |
| TA | S5w | MRR@10 | 150 | 0.9967 | 0.9967 | 0 | +0.000 |
| A | S1 | nDCG@10 | 150 | 0.8980 | 0.8980 | 0 | +0.000 |
| A | S1@60 | nDCG@10 | 150 | 0.8980 | 0.8980 | 0 | +0.000 |
| A | S2 | nDCG@10 | 150 | 0.8164 | 0.8159 | 9 | +0.185 |
| A | S2@60 | nDCG@10 | 150 | 0.8164 | 0.8159 | 9 | +0.185 |
| A | S3 | nDCG@10 | 150 | 0.9248 | 0.9265 | 5 | +0.333 |
| A | S3w@0.1 | nDCG@10 | 150 | 0.8551 | 0.8572 | 13 | +0.289 |
| A | S3w@0.25 | nDCG@10 | 150 | 0.8853 | 0.8859 | 9 | +0.115 |
| A | S3w@0.75 | nDCG@10 | 150 | 0.9268 | 0.9279 | 3 | +0.131 |
| A | S3w@0.9 | nDCG@10 | 150 | 0.9296 | 0.9281 | 2 | -0.226 |
| A | S3w | nDCG@10 | 150 | 0.8853 | 0.8859 | 9 | +0.115 |
| A | S4 | nDCG@10 | 150 | 0.8937 | 0.8919 | 2 | -0.296 |
| A | S5 | nDCG@10 | 150 | 0.9604 | 0.9604 | 0 | +0.000 |
| A | S5w | nDCG@10 | 150 | 0.9225 | 0.9159 | 1 | -1.000 |

### Contrastes declarados (mitad de prueba, Holm sobre seis)

| Contraste | Diferencia original | p Holm original | Significativo original | Diferencia réplica | p Holm réplica | Significativo réplica | Misma conclusión |
|---|---|---|---|---|---|---|---|
| C1-T | +0.001 | 1.000 | no | +0.001 | 1.000 | no | sí |
| C2-T | +0.000 | 1.000 | no | +0.000 | 1.000 | no | sí |
| C1-TA | +0.000 | 1.000 | no | +0.000 | 1.000 | no | sí |
| C2-TA | +0.000 | 1.000 | no | +0.000 | 1.000 | no | sí |
| C1-A | +0.067 | <0.001 | sí | +0.069 | <0.001 | sí | sí |
| C2-A | +0.029 | 0.022 | sí | +0.024 | 0.020 | sí | sí |

La regla de decisión de la sección 8 da la misma conclusión en los tres tipos de consulta.

## Diferencias y su causa probable

Vectores de consulta: 336 de 900 son idénticos bit a bit; 564 difieren (similitud del coseno mínima 0.991630). Proveedor que sirvió los vectores: deepinfra; el mismo en las dos ejecuciones para 900 de 900 consultas, y para 564 de las 564 cuyo vector difiere. Es decir, el proveedor devolvió vectores ligeramente distintos para el mismo texto.

Puntuaciones del reranker: de las 450 consultas de prueba, 3 tienen algún libro puntuado en las dos ejecuciones con puntuación distinta (mayor cambio 5.1e-09); en 227 se puntuó un conjunto de libros distinto porque las cabezas que se reordenan ya eran distintas.

Consultas con lista completa distinta, por la causa que muestran los ficheros. Una diferencia se atribuye a una causa solo cuando los datos guardados la muestran; si no, se dice «causa no determinada».

| Sistema | Causa | Consultas |
|---|---|---|
| S2 | vector de consulta distinto | 561 |
| S2@60 | vector de consulta distinto | 561 |
| S3 | heredada de una lista de entrada | 561 |
| S3w@0.1 | heredada de una lista de entrada | 561 |
| S3w@0.25 | heredada de una lista de entrada | 561 |
| S3w@0.75 | heredada de una lista de entrada | 561 |
| S3w@0.9 | heredada de una lista de entrada | 561 |
| S3w | heredada de una lista de entrada | 561 |
| S4 | heredada de una lista de entrada | 269 |
| S5 | heredada de una lista de entrada | 269 |
| S5w | heredada de una lista de entrada | 269 |

## Desviaciones, gasto y duración de la réplica

- En el paso de embeddings el proveedor respondió siete veces «Model busy, retry later» (HTTP 429); el adaptador repitió cada petición al cabo de uno a cuatro segundos y las 900 consultas quedaron codificadas sin ningún intento fallido. Motivo: saturación momentánea del proveedor; la repetición la hace el adaptador sin cambios y no cambia ningún resultado.

Gasto según el registro de gasto: réplica 0.9704 USD, original 0.9695 USD (tope 1.50 USD en cada una).

Duración, de la primera a la última entrada del registro de gasto: réplica 15 min (1 invocación); original 3 h 51 min (2 invocaciones, con la máquina en reposo).
