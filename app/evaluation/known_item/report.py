"""
results.json and KNOWN_ITEM.md of the known-item evaluation, from the saved
files only (sample, run files, weight selection, ledger). No network.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from app.evaluation.assistant_evaluation.report import _ci, _f, _p, _table
from app.evaluation.catalog_evaluation.ledger import SpendLedger, summarize_ledger
from app.evaluation.catalog_evaluation.runs import load_runs
from app.evaluation.known_item import analysis as KA
from app.evaluation.known_item import design as D
from app.evaluation.known_item.runner import load_sample

DEVIATIONS_FILE = "deviations.json"  # optional: [{"desviacion": ..., "motivo": ...}], written by hand

METRIC_NAMES = {"mrr_at_10": "MRR@10", "success_at_1": "éxito@1", "success_at_10": "éxito@10",
                "ndcg_at_10": "nDCG@10"}


def build_results(data_dir: Path) -> Dict[str, Any]:
    """Every figure of the report (dev and test halves, contrasts, decision, cost)."""
    sample = load_sample(data_dir)
    queries = sample["queries"]
    dev, test = KA.split(queries, D.DEV), KA.split(queries, D.TEST)
    selection = json.loads((data_dir / D.SELECTION_FILE).read_text(encoding="utf-8"))
    dev_systems = list(D.DEV_SYSTEMS) + [D.grid_name(w) for w in D.WEIGHT_GRID]
    runs = load_runs(data_dir, list(dict.fromkeys(dev_systems + list(D.SYSTEM_NAMES))))
    scores = KA.per_query(queries, runs)
    family = KA.contrasts(scores, test)
    errors = {s: sum(1 for q in test if (runs[s].get(q["query_id"]) or {}).get("error")) for s in D.SYSTEM_NAMES}
    missing = {s: sum(1 for q in test if q["query_id"] not in runs[s]) for s in D.SYSTEM_NAMES}
    ledger = SpendLedger(data_dir / D.LEDGER_FILE, D.SPEND_CAP_USD)
    return {
        "sample": {k: sample[k] for k in ("seed", "catalog_works", "languages", "books_per_language", "counts",
                                          "author_query_relevant")},
        "selection": selection,
        "test": {"queries": len(test), "summary": KA.summarize(scores, test, D.SYSTEM_NAMES),
                 "by_language": KA.by_language(scores, test, D.SYSTEM_NAMES),
                 "run_errors": errors, "missing_runs": missing},
        "dev": {"queries": len(dev), "summary": KA.summarize(scores, dev, dev_systems)},
        "contrasts": family,
        "secondary_without_zh_ja": KA.secondary_contrasts(scores, test),
        "decision": KA.decision(family),
        "deviations": _deviations(data_dir, errors, missing),
        "limits": protocol_limits(),
        "spend": summarize_ledger(ledger.entries()),
        "cap_usd": D.SPEND_CAP_USD,
    }


def protocol_limits(path: Path = D.PROTOCOL_PATH) -> List[str]:
    """The bullet points of section 9 (Límites) of the protocol, verbatim (one string per bullet)."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    start = text.find("## 9.")
    end = text.find("## 10.", start)
    if start < 0:
        return []
    bullets: List[str] = []
    for line in text[start:end if end > 0 else None].splitlines()[1:]:
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif line.startswith("  ") and bullets:
            bullets[-1] += " " + line.strip()
    return bullets


def _deviations(data_dir: Path, errors: Mapping[str, int], missing: Mapping[str, int]) -> List[Dict[str, str]]:
    """Deviations written by hand (deviations.json) plus those the run files show (fallbacks, missing runs)."""
    path = data_dir / DEVIATIONS_FILE
    found: List[Dict[str, str]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    for system in D.SYSTEM_NAMES:
        if errors.get(system):
            found.append({"desviacion": f"{errors[system]} consultas de prueba de {system} sin orden reordenado",
                          "motivo": "el reranker falló tres veces; se guarda el orden base y el error, como en V14"})
        if missing.get(system):
            found.append({"desviacion": f"{missing[system]} consultas de prueba sin ejecución de {system}",
                          "motivo": "la ejecución no se completó"})
    return found


OUTCOME_ES = {
    True: "la memoria debe limitar su conclusión a las consultas descriptivas y decir que, para este tipo de "
          "consulta, los candidatos léxicos ayudan",
    False: "la conclusión de V14 se extiende a este tipo de consulta sobre este catálogo",
}


def _summary_table(summary: Mapping[str, Any], query_type: str, systems: List[str]) -> List[str]:
    metrics = D.METRICS[query_type]
    rows = []
    for system in systems:
        row = summary[query_type].get(system)
        if row is None:
            continue
        rows.append([system, str(row["queries"])] + [f"{_f(row[m]['mean'])} {_ci(row[m]['ci'])}" for m in metrics])
    return _table(["Sistema", "Consultas"] + [f"{METRIC_NAMES[m]} [IC 95 %]" for m in metrics], rows)


def build_markdown(results: Mapping[str, Any]) -> str:
    """KNOWN_ITEM.md (Spanish)."""
    sel = results["selection"]
    lines = [
        "# Evaluación de búsqueda de un libro conocido por título y autor (V29, parte B)",
        "",
        f"Protocolo: `{D.PROTOCOL_PATH}`. Experimento diseñado después de conocer los resultados de V14, a raíz de "
        "una revisión externa del 2026-11-13. La verdad de referencia sale del propio catálogo: no interviene "
        "ningún juez LLM ni ninguna valoración humana.",
        "",
        f"Muestra: {results['sample']['books_per_language'] * len(results['sample']['languages'])} libros "
        f"(semilla {results['sample']['seed']}), tres consultas por libro; mitad de desarrollo y mitad de prueba "
        f"de {results['dev']['queries'] // 3} libros cada una. Todas las cifras de la tabla principal salen de las "
        f"{results['test']['queries']} consultas de la mitad de prueba.",
        "",
        f"Peso léxico elegido en la mitad de desarrollo: w = {sel['chosen_weight']} (peso vectorial "
        f"{1 - sel['chosen_weight']:g}).",
        "",
        "## Mitad de prueba",
        "",
    ]
    for qt in D.QUERY_TYPES:
        lines += [f"Consultas {qt} ({D.QUERY_TYPE_NAMES[qt]}):", ""]
        lines += _summary_table(results["test"]["summary"], qt, list(D.SYSTEM_NAMES))
    lines += ["## Contrastes declarados (mitad de prueba, con reranker en ambos lados, Holm sobre seis)", ""]
    rows = [[name, r["query_type"], f"{r['system']} - {r['baseline']}", METRIC_NAMES[r["metric"]], str(r["queries"]),
             _f(r["difference"], signed=True), _ci(r["ci"], signed=True), _p(r["p"]), _p(r["p_holm"]),
             "sí" if r["significant"] else "no"] for name, r in results["contrasts"].items()]
    lines += _table(["Contraste", "Tipo", "Diferencia de", "Métrica", "n", "Diferencia", "IC 95 %", "p", "p Holm",
                     "Significativo"], rows)
    lines += ["## Decisión (regla de la sección 8 del protocolo)", ""]
    for qt, row in results["decision"]["per_type"].items():
        better = row["fusion_significantly_better"]
        which = (f"{' y '.join(better)} {'son' if len(better) > 1 else 'es'} significativamente mejor"
                 f"{'es' if len(better) > 1 else ''} que S4" if better
                 else "ninguna fusión es significativamente mejor que S4")
        lines.append(f"- {qt} ({D.QUERY_TYPE_NAMES[qt]}): {which}; {OUTCOME_ES[bool(better)]}.")
    lines += ["", "## Lectura secundaria sin chino ni japonés (fuera de la familia, p sin corregir)", "",
              "Estas seis diferencias no forman parte de la familia de contrastes: son descriptivas y sus p no se "
              "corrigen.", ""]
    rows = [[name, f"{r['system']} - {r['baseline']}", str(r["queries"]), _f(r["difference"], signed=True),
             _ci(r["ci"], signed=True), _p(r["p"])] for name, r in results["secondary_without_zh_ja"].items()]
    lines += _table(["Contraste", "Diferencia de", "n", "Diferencia", "IC 95 %", "p"], rows)
    lines += ["## Mitad de desarrollo (elección de w; sin reranker)", ""]
    lines += _table(["Ejecución", "Objetivo"], [[k, _f(v)] for k, v in sel["objective"].items()])
    dev_systems = list(D.DEV_SYSTEMS) + [D.grid_name(w) for w in D.WEIGHT_GRID]
    for qt in D.QUERY_TYPES:
        lines += [f"Consultas {qt}:", ""]
        lines += _summary_table(results["dev"]["summary"], qt, dev_systems)
    spend = results["spend"]
    lines += ["## Errores y gasto", "",
              f"Ejecuciones con error (se guarda el orden base, como en V14): {results['test']['run_errors']}. "
              f"Consultas sin ejecución: {results['test']['missing_runs']}.", "",
              f"Gasto real según el registro de gasto: {spend['total_usd']:.4f} USD en {spend['calls']} llamadas "
              f"(tope {results['cap_usd']:.2f} USD).", ""]
    lines += _table(["Modelo", "Llamadas", "Llamadas fallidas", "Coste (USD)"],
                    [[m, str(r["calls"]), str(r["failed_calls"]), f"{r['cost_usd']:.4f}"]
                     for m, r in spend["by_model"].items()])
    lines += ["## Desviaciones", ""]
    lines += ([f"- {d['desviacion']}. Motivo: {d['motivo']}." for d in results["deviations"]]
              or ["Ninguna: la ejecución siguió el protocolo sin cambios."])
    lines += ["", "## Límites (sección 9 del protocolo)", ""]
    lines += [f"- {limit}" for limit in results["limits"]]
    return "\n".join(lines).rstrip() + "\n"


def write_report(data_dir: Path = D.DATA_DIR) -> Dict[str, Any]:
    results = build_results(data_dir)
    (data_dir / D.RESULTS_FILE).write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str),
                                           encoding="utf-8")
    (data_dir / D.REPORT_FILE).write_text(build_markdown(results), encoding="utf-8")
    return results
