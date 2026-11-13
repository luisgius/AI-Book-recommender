"""
Manual review of the queries and requests that an LLM wrote for the evaluations.

Which items an LLM wrote, and where the review is asked for:

- Catalog evaluation (V14): strata B, C and D of
  data/evaluation/catalog50k/queries/queries.json. Its protocol
  (docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md, section 3) says
  "Es una amenaza a la validez declarada (sección 11): el autor debe revisar
  la lista". Stratum A holds the original test queries.
- Assistant evaluation (V22 and V25): sets E and F
  (data/evaluation/assistant/sets, the same in both measurements) and the
  held-out set G (data/evaluation/assistant_v25/sets). The protocol lists
  "Las peticiones de E y F las escribe un LLM" among its threats (section 8),
  the addendum has G written by the same generator (section 3), and the task
  notes of V22 and V25 leave "Revisión por el autor" pending. The assistant's
  sets B and D are V14's strata, so they are reviewed once, as catalog rows.

The `sheet` command writes one row per generated item, ordered by evaluation,
stratum and id, with two empty columns the author fills (verdict: ok, dudosa
or mal, empty if not reviewed; comment), and the instructions in Spanish.
The `summary` command checks a filled sheet (same item ids, known verdicts)
and counts its verdicts per evaluation and stratum. The `sensitivity`
command recomputes the evaluations' figures without the items marked dudosa
or mal (app/evaluation/review_sensitivity.py).

    python -m app.evaluation.query_review sheet
    python -m app.evaluation.query_review summary --sheet data/evaluation/query_review/review_sheet.csv
    python -m app.evaluation.query_review sensitivity
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.evaluation.assistant_evaluation import design as AD
from app.evaluation.assistant_evaluation.set_g import VALID
from app.evaluation.assistant_evaluation.sets import Request, load_requests
from app.evaluation.catalog_evaluation import design as CD
from app.evaluation.catalog_evaluation.cli import log
from app.evaluation.catalog_evaluation.queries import EvalQuery, load_queries

OUT_DIR = Path("data/evaluation/query_review")
SHEET_FILE = "review_sheet.csv"
SUMMARY_FILE = "query_review.json"
INSTRUCTIONS_FILE = "INSTRUCCIONES.md"
CATALOG_QUERIES = CD.DATA_DIR / "queries" / "queries.json"
ASSISTANT_DIRS = (AD.DATA_DIR, AD.V25_DIR)  # first measurement (V22) and re-measurement (V25)

CATALOG, ASSISTANT = "catalogo", "asistente"
EVALUATIONS = (CATALOG, ASSISTANT)
GENERATED = {CATALOG: (CD.STRATUM_B, CD.STRATUM_C, CD.STRATUM_D), ASSISTANT: (AD.SET_E, AD.SET_F, AD.SET_G)}
VERDICTS = ("ok", "dudosa", "mal")
FLAGGED = ("dudosa", "mal")
SHEET_COLUMNS: Tuple[str, ...] = ("item_id", "evaluation", "stratum", "language", "type", "text", "need",
                                  "context", "verdict", "comment")

Row = Dict[str, str]


class SheetError(ValueError):
    """The sheet cannot be written or read as it is (the message says why)."""


# =============================================================================
# Rows of the sheet, from the published files
# =============================================================================


def _row(item_id: str, evaluation: str, stratum: str, language: str, kind: str, text: str, need: str,
         context: str) -> Row:
    return {"item_id": item_id, "evaluation": evaluation, "stratum": stratum, "language": language, "type": kind,
            "text": text, "need": need, "context": context, "verdict": "", "comment": ""}


def _catalog_row(query: EvalQuery) -> Row:
    """A query of V14; context names the work a known-item query describes."""
    context = ""
    if query.target_title:
        context = f"obra: {query.target_title} ({query.target_author or 'autor desconocido'})"
    return _row(query.query_id, CATALOG, query.stratum, query.language, query.type, query.text, query.need, context)


def _assistant_row(request: Request, work_labels: Mapping[str, str]) -> Row:
    """A request of E, F or G; context holds what the reviewer needs to check it."""
    if request.set == AD.SET_E:
        context = f"consulta de B {request.need_id}: {request.source_query}"
    elif request.set == AD.SET_F:
        context = "obras: " + "; ".join(work_labels.get(work, work) for work in request.works)
    else:  # G: notes are [field or out-of-scope kind, form, the generator's English translation] (set_g.g_requests)
        field, form, english = request.notes
        shape = f"campo: {field}; forma: {form}" if request.category == VALID else f"tipo: {field}"
        context = f"{shape}; traducción del generador: {english}"
    return _row(request.request_id, ASSISTANT, request.set, request.language, request.category, request.text,
                request.need, context)


def _work_labels(assistant_dir: Path) -> Dict[str, str]:
    """{work_id: "title (authors)"} of the catalog works set F was written from (sets/checks.json)."""
    path = assistant_dir / "sets" / "checks.json"
    if not path.exists():
        return {}
    checks = json.loads(path.read_text(encoding="utf-8"))
    return {work["work_id"]: f"{work['title']} ({', '.join(work.get('authors') or [])})"
            for task in checks.get("f_tasks", []) for work in task.get("works", [])}


def _generated_requests(assistant_dirs: Sequence[Path]) -> List[Request]:
    """Requests of E, F and G of every measurement, each once (E and F are in both)."""
    by_id: Dict[str, Request] = {}
    for directory in assistant_dirs:
        for request in load_requests(directory / "sets" / "requests.json"):
            if request.set not in GENERATED[ASSISTANT]:
                continue
            if request.request_id in by_id and by_id[request.request_id] != request:
                raise SheetError(f"request {request.request_id} is not the same in every assistant measurement")
            by_id[request.request_id] = request
    return list(by_id.values())


def review_items(catalog_queries: Path = CATALOG_QUERIES,
                 assistant_dirs: Sequence[Path] = ASSISTANT_DIRS) -> List[Row]:
    """One row per generated query or request, ordered by evaluation, stratum and id."""
    rows = [_catalog_row(q) for q in load_queries(catalog_queries) if q.stratum in GENERATED[CATALOG]]
    labels: Dict[str, str] = {}
    for directory in assistant_dirs:
        labels.update(_work_labels(directory))
    rows += [_assistant_row(request, labels) for request in _generated_requests(assistant_dirs)]
    repeated = sorted(item_id for item_id, n in Counter(row["item_id"] for row in rows).items() if n > 1)
    if repeated:
        raise SheetError(f"the same id names two different items: {repeated}")
    return sorted(rows, key=lambda row: (EVALUATIONS.index(row["evaluation"]), row["stratum"], row["item_id"]))


def counts(items: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, int]]:
    """{evaluation: {stratum: number of rows}}, in the order of the rows."""
    out: Dict[str, Dict[str, int]] = {}
    for item in items:
        strata = out.setdefault(item["evaluation"], {})
        strata[item["stratum"]] = strata.get(item["stratum"], 0) + 1
    return out


# =============================================================================
# Writing the sheet and the instructions
# =============================================================================

INSTRUCTIONS = """# Revisión de las consultas y peticiones generadas por un LLM

Las consultas de los estratos B, C y D de la evaluación sobre el catálogo
(V14) y las peticiones de los conjuntos E, F y G de la evaluación del
asistente (V22 y V25) las escribió un LLM. El protocolo de V14 lo declara
amenaza a la validez: «el autor debe revisar la lista» (sección 3). El del
asistente cuenta entre sus amenazas que «las peticiones de E y F las escribe
un LLM» (sección 8); G lo escribió el mismo generador (adenda, sección 3).
Esta hoja tiene una fila por cada una, {total} en total:

| Evaluación | Estrato | Filas |
|---|---|---|
{table}

No están el estrato A (las consultas originales) ni los conjuntos B y D del
asistente, que son los estratos B y D del catálogo y se revisan una sola vez,
en las filas `catalogo`. E y F son iguales en las dos medidas del asistente
y aparecen una vez.

## Qué hay que hacer

Abra `{sheet}` en una hoja de cálculo y escriba en la columna `verdict` uno
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

    .venv/bin/python -m app.evaluation.query_review summary --sheet {path}

El comando rechaza la hoja si falta, sobra o cambió algún `item_id`, o si un
veredicto no es `ok`, `dudosa`, `mal` o vacío. Si la acepta, escribe
`{summary}`: por evaluación y estrato, cuántas filas hay, cuántas se
revisaron, cuántas de cada veredicto y la lista de las `dudosa` y `mal` con
su comentario. Puede ejecutarse con la hoja a medias.
"""


def _read_rows(path: Path) -> List[Row]:
    """
    Rows of a sheet as a spreadsheet saves it: UTF-8 with or without a byte
    order mark (hence "utf-8-sig"), separated by commas or, as Excel does in
    a Spanish locale, by semicolons.
    """
    if not path.is_file():
        raise SheetError(f"{path} does not exist")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()
            delimiter = ";" if header.count(";") > header.count(",") else ","
            handle.seek(0)
            reader = csv.DictReader(handle, delimiter=delimiter)
            missing = [c for c in ("item_id", "verdict", "comment") if c not in (reader.fieldnames or [])]
            if missing:
                raise SheetError(f"{path} has no column {missing}; it must keep the columns {list(SHEET_COLUMNS)}")
            return list(reader)
    except UnicodeDecodeError as error:
        raise SheetError(f"{path} is not UTF-8; save it again as 'CSV UTF-8'") from error


def build_sheet(items: Sequence[Mapping[str, str]], out_dir: Path) -> Dict[str, Dict[str, int]]:
    """
    Write the sheet and the instructions into out_dir; returns the rows per
    evaluation and stratum. A sheet that already holds a verdict or a
    comment is never overwritten.
    """
    sheet = out_dir / SHEET_FILE
    if sheet.exists() and any((row.get("verdict") or "").strip() or (row.get("comment") or "").strip()
                              for row in _read_rows(sheet)):
        raise SheetError(f"{sheet} already holds verdicts or comments; move it away before building a new sheet")
    out_dir.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: the byte order mark makes Excel read the Arabic, Chinese, Japanese and Russian rows correctly
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(items)
    by_stratum = counts(items)
    table = "\n".join(f"| {evaluation} | {stratum} | {n} |"
                      for evaluation, strata in by_stratum.items() for stratum, n in strata.items())
    (out_dir / INSTRUCTIONS_FILE).write_text(
        INSTRUCTIONS.format(total=len(items), table=table, sheet=SHEET_FILE, path=sheet.as_posix(),
                            summary=(OUT_DIR / SUMMARY_FILE).as_posix()),
        encoding="utf-8")
    return by_stratum


# =============================================================================
# Reading the filled sheet and summarizing it
# =============================================================================


def _verdict(row: Mapping[str, Optional[str]]) -> str:
    return (row.get("verdict") or "").strip().lower()


def read_sheet(path: Path, items: Sequence[Mapping[str, str]]) -> Dict[str, Row]:
    """
    {item_id: {"verdict", "comment"}} of a filled sheet. Raises SheetError if
    its item ids are not exactly those of `items` or a verdict is not one of
    ok, dudosa, mal or empty (upper case and spaces are accepted).
    """
    rows = _read_rows(path)
    ids = [(row.get("item_id") or "").strip() for row in rows]
    expected = {item["item_id"] for item in items}
    missing = sorted(expected - set(ids))
    unknown = sorted(set(ids) - expected)
    repeated = sorted(item_id for item_id, n in Counter(ids).items() if n > 1)
    if missing or unknown or repeated:
        raise SheetError(f"the item_id column of {path} does not match the published queries and requests "
                         f"(missing: {missing}; unknown or changed: {unknown}; repeated: {repeated}). "
                         "Do not edit item_id, add or delete rows.")
    bad = [f"{item_id}: {row.get('verdict')!r}" for item_id, row in zip(ids, rows)
           if _verdict(row) not in ("",) + VERDICTS]
    if bad:
        raise SheetError(f"unknown verdict in {path} ({'; '.join(bad)}); allowed: {', '.join(VERDICTS)} or empty")
    return {item_id: {"verdict": _verdict(row), "comment": (row.get("comment") or "").strip()}
            for item_id, row in zip(ids, rows)}


def summarize(items: Sequence[Mapping[str, str]], filled: Mapping[str, Mapping[str, str]]) -> Dict[str, Any]:
    """Per evaluation and stratum: items, reviewed, each verdict, and the dudosa/mal ids with their comments."""
    by_evaluation: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in items:
        cell = by_evaluation.setdefault(item["evaluation"], {}).setdefault(
            item["stratum"], {"items": 0, "reviewed": 0, **{v: 0 for v in VERDICTS}, "flagged": []})
        verdict, comment = filled[item["item_id"]]["verdict"], filled[item["item_id"]]["comment"]
        cell["items"] += 1
        if verdict:
            cell["reviewed"] += 1
            cell[verdict] += 1
        if verdict in FLAGGED:
            cell["flagged"].append({"item_id": item["item_id"], "verdict": verdict, "comment": comment})
    cells = [cell for strata in by_evaluation.values() for cell in strata.values()]
    total = {key: sum(cell[key] for cell in cells) for key in ("items", "reviewed", *VERDICTS)}
    return {"total": total, "by_evaluation": by_evaluation}


# =============================================================================
# Command line
# =============================================================================


def cmd_sheet(args: argparse.Namespace) -> int:
    by_stratum = build_sheet(review_items(), args.out_dir)
    log(f"Wrote {args.out_dir / SHEET_FILE} and {INSTRUCTIONS_FILE}: {json.dumps(by_stratum)}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    items = review_items()
    document = {"sheet": args.sheet.as_posix(), **summarize(items, read_sheet(args.sheet, items))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = document["total"]
    log(f"Wrote {args.output}: {total['reviewed']} of {total['items']} reviewed, "
        + ", ".join(f"{v} {total[v]}" for v in VERDICTS))
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    from app.evaluation import review_sensitivity as RS  # imported here: it imports this module

    try:
        document = RS.write_report(args.sheet, args.out_dir)
    except RS.SensitivityError as error:
        raise SystemExit(f"error: {error}") from error
    changes = sum(d["changes"] for e in document["evaluations"].values() for d in e["decisions"]
                  if d["kind"] == RS.DECLARED)
    log(f"Wrote {args.out_dir / RS.JSON_FILE} and {RS.MD_FILE}: {len(document['excluded'])} items excluded, "
        f"{changes} declared decisions change")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation.query_review",
                                     description="Manual review of the LLM-generated queries and requests")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sheet", help="build the review sheet and its instructions from the published files")
    s.add_argument("--out-dir", type=Path, default=OUT_DIR)
    s.set_defaults(func=cmd_sheet)
    m = sub.add_parser("summary", help="check a filled sheet and count its verdicts")
    m.add_argument("--sheet", type=Path, default=OUT_DIR / SHEET_FILE)
    m.add_argument("--output", type=Path, default=OUT_DIR / SUMMARY_FILE)
    m.set_defaults(func=cmd_summary)
    v = sub.add_parser("sensitivity", help="recompute the evaluations' figures without the dudosa and mal items "
                                           "(exploratory, decided after the results; see review_sensitivity.py)")
    v.add_argument("--sheet", type=Path, default=OUT_DIR / SHEET_FILE)
    v.add_argument("--out-dir", type=Path, default=OUT_DIR)
    v.set_defaults(func=cmd_sensitivity)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SheetError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    sys.exit(main())
