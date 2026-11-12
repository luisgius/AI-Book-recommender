# Hoja de anotación humana (V14)

Esta hoja contiene 200 pares (consulta, obra) del catálogo desplegado,
elegidos al azar con semilla 0 y estratificados por idioma de la consulta y
por la nota del juez principal. La hoja no muestra ninguna nota de los
jueces ni qué sistema devolvió la obra. El orden de las filas es aleatorio.

## Qué hay que hacer

Abra `annotation_sheet.csv` en una hoja de cálculo y escriba en la columna `grade` una
nota de 0 a 3 para cada fila. No cambie ninguna otra columna ni el
`pair_id`. Valore la obra frente a la columna `need` (la necesidad de
información, en inglés); la columna `query` muestra lo que escribió el
usuario. Base la nota solo en la información de la fila (título, autores,
descripción).

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
- Si no puede decidir, deje la celda vacía: esa fila no cuenta.

## Cómo se calcula el acuerdo

Guarde la hoja (CSV, UTF-8) y ejecute desde la raíz del repositorio:

    .venv/bin/python -m app.evaluation.catalog_evaluation annotation-agreement --sheet data/evaluation/catalog50k/annotation/annotation_sheet.csv

El comando compara sus notas con las de cada juez (kappa ponderado
cuadrático con intervalo, acuerdo exacto y diferencia media) y escribe
`data/evaluation/catalog50k/annotation/human_agreement.json`.
