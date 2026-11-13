"""
Human annotation sheet (protocol section 5).

200 (need, work) pairs of the judged pool of strata A and B, drawn at
random with seed 0, stratified by query language and by the primary judge's
grade (proportional allocation, every stratum represented; the same
stratified_sample as V05). The sheet never shows a judge's grade: it holds
the query, the need, the language, the work's title, authors and
description and an empty "grade" column, in a shuffled order (seed 0).
A separate key file maps each pair id to the need and the work.

annotation_agreement() reads a filled sheet and reports the agreement of
the human grades with each judge (quadratic-weighted kappa with a
bootstrap interval, exact agreement, mean signed difference), with
agreement_report of app/evaluation/relevance_judge.py.
"""

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.queries import EvalQuery
from app.evaluation.relevance_judge import agreement_report, stratified_sample

SHEET_COLUMNS = ("pair_id", "query", "need", "language", "title", "authors", "description", "grade")
SHEET_FILE = "annotation_sheet.csv"
KEY_FILE = "annotation_key.json"
INSTRUCTIONS_FILE = "INSTRUCCIONES.md"

INSTRUCTIONS = """# Hoja de anotación humana (V14)

Esta hoja contiene {n} pares (consulta, obra) del catálogo desplegado,
elegidos al azar con semilla 0 y estratificados por idioma de la consulta y
por la nota del juez principal. La hoja no muestra ninguna nota de los
jueces ni qué sistema devolvió la obra. El orden de las filas es aleatorio.

## Qué hay que hacer

Abra `{sheet}` en una hoja de cálculo y escriba en la columna `grade` una
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

    .venv/bin/python -m app.evaluation.catalog_evaluation annotation-agreement --sheet {path}

El comando compara sus notas con las de cada juez (kappa ponderado
cuadrático con intervalo, acuerdo exacto y diferencia media) y escribe
`data/evaluation/catalog50k/annotation/human_agreement.json`.
"""


def population(queries: Sequence[EvalQuery], grades: Mapping[str, Mapping[str, int]]) -> List[Dict[str, Any]]:
    """Judged pool pairs of strata A and B, in a deterministic order (need id, work id)."""
    items = []
    for query in sorted(queries, key=lambda q: q.query_id):
        if query.stratum not in D.STATISTICAL_STRATA:
            continue
        for work_id in sorted(grades.get(query.need_id, {})):
            items.append({"need_id": query.need_id, "query_id": query.query_id, "work_id": work_id,
                          "language": query.language, "grade": grades[query.need_id][work_id]})
    return items


def build_sheet(queries: Sequence[EvalQuery], grades: Mapping[str, Mapping[str, int]], books_by_id: Mapping[str, Any],
                out_dir: Path, n: int = D.ANNOTATION_PAIRS, seed: int = D.SEED) -> Dict[str, Any]:
    """Write the sheet, the key and the instructions; returns a summary (strata sizes, no grades)."""
    items = population(queries, grades)
    sample = stratified_sample(items, n, seed=seed, stratum=lambda item: (item["language"], item["grade"]))
    random.Random(seed).shuffle(sample)
    by_id = {q.query_id: q for q in queries}
    out_dir.mkdir(parents=True, exist_ok=True)
    key = {}
    with (out_dir / SHEET_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for number, item in enumerate(sample, start=1):
            pair_id = f"p{number:03d}"
            book = books_by_id[item["work_id"]]
            query = by_id[item["query_id"]]
            writer.writerow({"pair_id": pair_id, "query": query.text, "need": query.need, "language": query.language,
                             "title": book.title, "authors": "; ".join(book.authors),
                             "description": (book.description or "").strip(), "grade": ""})
            key[pair_id] = {"need_id": item["need_id"], "query_id": item["query_id"], "work_id": item["work_id"]}
    (out_dir / KEY_FILE).write_text(json.dumps(key, indent=2), encoding="utf-8")
    sheet_path = out_dir / SHEET_FILE
    (out_dir / INSTRUCTIONS_FILE).write_text(
        INSTRUCTIONS.format(n=len(sample), sheet=SHEET_FILE, path=sheet_path), encoding="utf-8")
    strata: Dict[str, int] = {}
    for item in sample:
        strata[item["language"]] = strata.get(item["language"], 0) + 1
    return {"pairs": len(sample), "population": len(items), "by_language": strata}


def read_sheet(path: Path) -> Dict[str, int]:
    """{pair_id: grade} of the filled rows (empty or invalid grades are skipped).

    Spreadsheets often export CSV with a byte-order mark; "utf-8-sig" reads
    the file with or without it.
    """
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("grade") or "").strip()
            if raw in ("0", "1", "2", "3"):
                out[row["pair_id"]] = int(raw)
    return out


def annotation_agreement(sheet: Path, key: Mapping[str, Mapping[str, str]],
                         judges: Mapping[str, Mapping[str, Mapping[str, int]]],
                         languages: Mapping[str, str]) -> Dict[str, Any]:
    """Agreement of the human grades with each judge's grades on the same pairs."""
    human = read_sheet(sheet)
    out: Dict[str, Any] = {"filled_rows": len(human), "judges": {}}
    for model, grades in judges.items():
        h, j, langs = [], [], []
        for pair_id, grade in sorted(human.items()):
            pair = key.get(pair_id)
            if pair is None:
                continue
            judge_grade = grades.get(pair["need_id"], {}).get(pair["work_id"])
            if judge_grade is None:
                continue
            h.append(grade)
            j.append(judge_grade)
            langs.append(languages.get(pair["query_id"]))
        report = agreement_report(h, j, langs, n_resamples=2000, seed=D.SEED) if h else {}
        out["judges"][model] = {"pairs": len(h), **report}
    return out
