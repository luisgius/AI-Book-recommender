# Hoja de anotación del orden de los sistemas (V31)

Esta hoja contiene 607 pares (consulta, obra) de 56 consultas del
catálogo desplegado: todas las obras que alguno de los sistemas evaluados
pone entre sus cinco primeros resultados para esas consultas. La hoja no
muestra ninguna nota de los jueces, ni qué sistema devolvió la obra, ni en
qué posición. Las filas están agrupadas por consulta: el orden de las
consultas es aleatorio y, dentro de cada consulta, el de las obras también.

## Dos bloques

La columna `block` dice a qué bloque pertenece cada fila.

- **Bloque 1 (obligatorio)**: 313 filas de 30 consultas, las
  primeras de la hoja. Termínelo antes de empezar el bloque 2.
- **Bloque 2 (opcional)**: 294 filas de 26 consultas, después
  del bloque 1. Hágalo entero o no lo haga. En el bloque 2 una consulta solo
  cuenta para comparar los sistemas si todas sus filas tienen nota: si empieza
  una consulta, termínela. Una celda vacía en el bloque 2, aunque sea de una
  obra que no se puede valorar, deja fuera esa consulta. Si deja el bloque a
  medias, las consultas completas se informan aparte y las incompletas no se
  usan.

Decida si hace el bloque 2 sin mirar ningún resultado: el análisis se ejecuta
una sola vez, cuando entregue la hoja, nunca entre un bloque y otro.

## Qué hay que hacer

Abra `annotation_order_sheet.csv` en una hoja de cálculo y escriba en la columna `grade` una
nota de 0 a 3 para cada fila. No cambie ninguna otra columna ni el
`pair_id`. Valore la obra frente a la columna `need` (la necesidad de
información, en inglés); la columna `query` muestra lo que escribió el
usuario. Base la nota solo en la información de la fila (título, autores,
idioma de la obra, año y descripción).

- Valore sin consultar nada sobre los sistemas: no abra la clave
  (`annotation_order_key.json`), ni las ejecuciones, ni los informes de la evaluación, ni busque
  las consultas en la aplicación mientras anota.
- Algunas filas repiten pares de la hoja de 200 pares que ya valoró. Valórelas
  de nuevo, sin mirar la nota que les puso entonces.

## Escala (la misma que la de los jueces, V05)

| Nota | Significado |
|---|---|
| 3 | Perfectamente relevante: es exactamente lo que se pide |
| 2 | Relevante: responde a la necesidad, aunque no del todo |
| 1 | Algo relevante: toca el tema de forma parcial o tangencial |
| 0 | No relevante |

- Una obra en otro idioma que la consulta puede ser relevante, salvo que la
  consulta pida un idioma.
- Si falta la descripción, valore por el título, los autores y lo que se sepa
  de la obra por esos datos.
- Deje la celda vacía solo si la obra no se puede valorar con el texto que se
  muestra: esa fila no cuenta. Si duda entre dos notas, elija una.

## Cuánto tarda

A un ritmo de 30 a 45 segundos por fila: el bloque 1 (313 filas), entre
2,6 y 3,9 horas; el bloque 2 (294 filas), entre 2,5 y
3,7 horas más. Puede hacerse en varias sesiones; guarde la hoja a menudo.

## Segunda persona (opcional)

La columna `grade_2` es para una segunda persona, con estas mismas
instrucciones. Puede rellenarla para todas las filas o solo para una parte;
para comparar los sistemas solo cuentan sus consultas completas (todas las
filas de la consulta con nota), en los dos bloques. Para que no vea las notas
de la primera persona, conviene que anote sobre una copia de la hoja en
blanco y se pase el fichero con `--second-sheet` (abajo).

## Al terminar

Guarde la hoja como CSV UTF-8 con otro nombre (por ejemplo
`annotation_order_sheet_filled.csv`) en esta carpeta y ejecute desde la raíz
del repositorio:

    .venv/bin/python -m app.evaluation.catalog_evaluation order-agreement --sheet data/evaluation/catalog50k/annotation_order/annotation_order_sheet_filled.csv

Si la segunda persona anotó en una copia aparte, añada
`--second-sheet <fichero>`. El comando no usa la red y escribe
`data/evaluation/catalog50k/annotation_order/order_agreement.json`.
