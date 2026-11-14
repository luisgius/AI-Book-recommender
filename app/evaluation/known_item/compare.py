"""
Comparison of two runs of the known-item evaluation (V29 replication).

Reads two data folders (the original run and a replication made from the same
sample.json) and says, from the saved files only, how far they agree: per
system and per query whether the stored id lists are identical, the largest
per-query change of each query type's primary metric, the chosen weight, the
six contrasts side by side, and the spend and run time of each run.

When lists differ it looks for the likely cause only where the files show it:
different query vectors from the embedding provider, the same vector giving a
different list (the approximate index), different reranker scores, or a
difference inherited from the lists a system is built from. Anything else is
reported as «causa no determinada». No network, no database.
"""

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from app.evaluation.assistant_evaluation.report import _f, _p, _table
from app.evaluation.catalog_evaluation import design as CD
from app.evaluation.catalog_evaluation.ledger import summarize_ledger
from app.evaluation.catalog_evaluation.runs import embedding_store, load_runs, runs_dir
from app.evaluation.known_item import analysis as KA
from app.evaluation.known_item import design as D
from app.evaluation.known_item.report import METRIC_NAMES
from app.evaluation.known_item.runner import RERANK_SCORES

REPLICATION_JSON = "replication.json"
REPLICATION_MD = "REPLICATION.md"
UNKNOWN = "causa no determinada"

# Which stored lists each system is built from (for "inherited" differences).
INPUTS: Dict[str, Sequence[str]] = {
    D.S3: ("S1@60", "S2@60"),
    **{D.grid_name(w): ("S1@60", "S2@60") for w in D.WEIGHT_GRID},
    D.S3W: ("S1@60", "S2@60"),
    **{name: (base,) for name, base in D.RERANKED.items()},
}
VECTOR_SYSTEMS = (D.S2, "S2@60")
LEXICAL_SYSTEMS = (D.S1, "S1@60")
COMPARED_SYSTEMS: Sequence[str] = (D.S1, "S1@60", D.S2, "S2@60", D.S3, *[D.grid_name(w) for w in D.WEIGHT_GRID],
                                   D.S3W, D.S4, D.S5, D.S5W)


# =============================================================================
# Loading (I/O)
# =============================================================================


def load_folder(data_dir: Path) -> Dict[str, Any]:
    """Everything the comparison reads from one data folder."""
    sample = json.loads((data_dir / D.SAMPLE_FILE).read_text(encoding="utf-8"))
    present = [s for s in COMPARED_SYSTEMS if (runs_dir(data_dir) / f"{s}.jsonl").exists()]
    rerank_path = data_dir / RERANK_SCORES
    rerank = {}
    if rerank_path.exists():
        for line in rerank_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rerank[row["query_id"]] = row["scores"]
    ledger_path = data_dir / D.LEDGER_FILE
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if ledger_path.exists() else []
    results_path = data_dir / D.RESULTS_FILE
    return {
        "folder": str(data_dir),
        "queries": sample["queries"],
        "runs": load_runs(data_dir, present),
        "embeddings": {qid: {"vector": r["vector"], "provider": r.get("provider")}
                       for qid, r in embedding_store(data_dir, CD.QWEN_EMBEDDING).records.items()},
        "rerank": rerank,
        "ledger": ledger,
        "results": json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {},
    }


# =============================================================================
# Pure comparison
# =============================================================================


def _vector(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32)


def compare_embeddings(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Per query, whether the frozen query vectors are bit-identical, the smallest
    cosine similarity among those that are not, and whether the provider that
    served the vector was the same in both runs.
    """
    common = sorted(set(a) & set(b))
    different: List[str] = []
    cosines: List[float] = []
    same_provider = 0
    for qid in common:
        if a[qid].get("provider") == b[qid].get("provider"):
            same_provider += 1
        if a[qid]["vector"] == b[qid]["vector"]:
            continue
        different.append(qid)
        u, v = _vector(a[qid]["vector"]).astype(np.float64), _vector(b[qid]["vector"]).astype(np.float64)
        cosines.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))
    different_same_provider = sum(1 for q in different if a[q].get("provider") == b[q].get("provider"))
    return {"queries": len(common), "identical": len(common) - len(different), "different": different,
            "min_cosine": min(cosines) if cosines else None, "same_provider": same_provider,
            "different_same_provider": different_same_provider,
            "providers": sorted({str(r.get("provider")) for r in list(a.values()) + list(b.values())})}


def compare_rerank(a: Mapping[str, Mapping[str, float]], b: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
    """
    Per query, whether a book scored in both runs got a different score
    (`different`), and how many queries scored a different set of books
    (because the heads to rerank differed).
    """
    common = sorted(set(a) & set(b))
    different: List[str] = []
    other_books = 0
    largest = 0.0
    for qid in common:
        shared = set(a[qid]) & set(b[qid])
        gap = max((abs(a[qid][k] - b[qid][k]) for k in shared), default=0.0)
        if gap > 0:
            different.append(qid)
        if set(a[qid]) != set(b[qid]):
            other_books += 1
        largest = max(largest, gap)
    return {"queries": len(common), "identical": len(common) - len(different), "different": different,
            "different_book_sets": other_books, "largest_score_change": largest}


def _ids(runs: Mapping[str, Mapping[str, Any]], system: str, qid: str) -> List[str]:
    return list(runs[system][qid]["ids"])


def compare_lists(runs_a: Mapping[str, Mapping[str, Any]], runs_b: Mapping[str, Mapping[str, Any]],
                  systems: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Per system: queries in both, identical top 10 and identical full list, and the ids of those that differ."""
    out: Dict[str, Dict[str, Any]] = {}
    for system in systems:
        common = sorted(set(runs_a.get(system, {})) & set(runs_b.get(system, {})))
        top_diff = [q for q in common if _ids(runs_a, system, q)[: D.CUTOFF] != _ids(runs_b, system, q)[: D.CUTOFF]]
        full_diff = [q for q in common if _ids(runs_a, system, q) != _ids(runs_b, system, q)]
        out[system] = {"queries": len(common), "identical_top10": len(common) - len(top_diff),
                       "identical_full": len(common) - len(full_diff), "different_top10": top_diff,
                       "different_full": full_diff}
    return out


def metric_changes(queries: Sequence[Mapping[str, Any]], runs_a: Mapping[str, Mapping[str, Any]],
                   runs_b: Mapping[str, Mapping[str, Any]], systems: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Per query type and system: the largest absolute per-query change of the primary metric and the change of the mean."""
    out: Dict[str, Dict[str, Any]] = {}
    for qt in D.QUERY_TYPES:
        metric = D.PRIMARY_METRIC[qt]
        out[qt] = {}
        for system in systems:
            subset = [q for q in queries if q["query_type"] == qt
                      and q["query_id"] in runs_a.get(system, {}) and q["query_id"] in runs_b.get(system, {})]
            if not subset:
                continue
            sa = KA.per_query(subset, {system: runs_a[system]})[system]
            sb = KA.per_query(subset, {system: runs_b[system]})[system]
            deltas = {q["query_id"]: sb[q["query_id"]][metric] - sa[q["query_id"]][metric] for q in subset}
            worst = max(deltas, key=lambda k: abs(deltas[k]))
            out[qt][system] = {"metric": metric, "queries": len(subset),
                               "largest_change": deltas[worst], "largest_change_query": worst,
                               "queries_changed": sum(1 for d in deltas.values() if d != 0),
                               "mean_a": float(np.mean([sa[k][metric] for k in deltas])),
                               "mean_b": float(np.mean([sb[k][metric] for k in deltas]))}
    return out


def attribute(lists: Mapping[str, Mapping[str, Any]], embeddings: Mapping[str, Any],
              rerank: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Per system, the queries whose full list differs, counted by the cause the
    files support: a different query vector, the same vector with a different
    list (approximate index), a difference inherited from an input list,
    different reranker scores over identical heads, or no determined cause.
    """
    emb_diff, rerank_diff = set(embeddings["different"]), set(rerank["different"])
    out: Dict[str, Dict[str, int]] = {}
    for system, row in lists.items():
        counts: Dict[str, int] = {}
        for qid in row["different_full"]:
            if system in VECTOR_SYSTEMS:
                cause = "vector de consulta distinto" if qid in emb_diff else "mismo vector, lista distinta (índice aproximado)"
            elif system in INPUTS:
                inherited = any(qid in set(lists.get(src, {}).get("different_full", ())) for src in INPUTS[system])
                if inherited:
                    cause = "heredada de una lista de entrada"
                elif system in D.RERANKED and qid in rerank_diff:
                    cause = "puntuaciones del reranker distintas"
                else:
                    cause = UNKNOWN
            else:
                cause = UNKNOWN
            counts[cause] = counts.get(cause, 0) + 1
        out[system] = counts
    return out


def _parse(date: str) -> datetime:
    return datetime.fromisoformat(date)


def run_time(ledger: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Wall-clock time of a run from its ledger: from the first entry (the
    opening key-usage snapshot) to the last, and the number of invocations
    (opening snapshots). The free database step writes no ledger entry, so the
    span is the only duration the ledger can give for a run that was resumed.
    """
    if not ledger:
        return {"invocations": 0, "span_seconds": None}
    invocations = sum(1 for e in ledger if e.get("operation") == "key_usage" and e.get("when") == "start")
    span = (_parse(ledger[-1]["date"]) - _parse(ledger[0]["date"])).total_seconds()
    return {"invocations": invocations, "span_seconds": span, "first": ledger[0]["date"], "last": ledger[-1]["date"]}


def compare(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """The whole comparison of the original run `a` and the replication `b` (pure)."""
    if [q["query_id"] for q in a["queries"]] != [q["query_id"] for q in b["queries"]]:
        raise ValueError("the two folders do not share the same sample")
    queries = a["queries"]
    systems = [s for s in COMPARED_SYSTEMS if s in a["runs"] and s in b["runs"]]
    lists = compare_lists(a["runs"], b["runs"], systems)
    embeddings = compare_embeddings(a["embeddings"], b["embeddings"])
    rerank = compare_rerank(a["rerank"], b["rerank"])
    test = KA.split(queries, D.TEST)
    test_ids = {q["query_id"] for q in test}
    contrasts = {}
    for name in (c.name for c in D.CONTRASTS):
        ra, rb = a["results"].get("contrasts", {}).get(name), b["results"].get("contrasts", {}).get(name)
        contrasts[name] = {"a": ra, "b": rb,
                           "same_decision": None if ra is None or rb is None else ra["significant"] == rb["significant"]}
    decision_a = a["results"].get("decision", {}).get("per_type", {})
    decision_b = b["results"].get("decision", {}).get("per_type", {})
    return {
        "folders": {"a": a["folder"], "b": b["folder"]},
        "queries": len(queries),
        "lists": {s: {k: v for k, v in row.items() if not k.startswith("different_")}
                  | {"different_top10": len(row["different_top10"]), "different_full": len(row["different_full"]),
                     "different_top10_test": sum(1 for q in row["different_top10"] if q in test_ids)}
                  for s, row in lists.items()},
        "different_queries": {s: row["different_full"] for s, row in lists.items()},
        "causes": attribute(lists, embeddings, rerank),
        "embeddings": {k: v for k, v in embeddings.items() if k != "different"} | {"different": len(embeddings["different"])},
        "rerank": {k: v for k, v in rerank.items() if k != "different"} | {"different": len(rerank["different"])},
        "metric_changes_test": metric_changes(test, a["runs"], b["runs"], systems),
        "metric_changes_all": metric_changes(queries, a["runs"], b["runs"], systems),
        "weight": {"a": a["results"].get("selection", {}).get("chosen_weight"),
                   "b": b["results"].get("selection", {}).get("chosen_weight"),
                   "objective_a": a["results"].get("selection", {}).get("objective"),
                   "objective_b": b["results"].get("selection", {}).get("objective")},
        "contrasts": contrasts,
        "decision_same": {qt: (decision_a.get(qt, {}).get("fusion_significantly_better"),
                               decision_b.get(qt, {}).get("fusion_significantly_better")) for qt in D.QUERY_TYPES},
        "spend": {"a": summarize_ledger(a["ledger"])["total_usd"] if a["ledger"] else None,
                  "b": summarize_ledger(b["ledger"])["total_usd"] if b["ledger"] else None},
        "run_time": {"a": run_time(a["ledger"]), "b": run_time(b["ledger"])},
        "deviations_b": b["results"].get("deviations", []),
    }


# =============================================================================
# Markdown (Spanish)
# =============================================================================


def _duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"


def _invocations(n: int) -> str:
    return f"{n} invocación" if n == 1 else f"{n} invocaciones"


def _contrast_cell(row: Optional[Mapping[str, Any]]) -> List[str]:
    if row is None:
        return ["n/a", "n/a", "n/a"]
    return [_f(row["difference"], signed=True), _p(row["p_holm"]), "sí" if row["significant"] else "no"]


def build_markdown(c: Mapping[str, Any]) -> str:
    """REPLICATION.md."""
    t_a, t_b = c["run_time"]["a"], c["run_time"]["b"]
    lines = [
        "# Réplica de la evaluación de búsqueda de un libro conocido (V29, parte B)",
        "",
        "## Por qué se repitió",
        "",
        "La primera ejecución tardó unas tres horas y media en lugar de minutos. El registro de energía del sistema "
        "(`pmset -g log`) mostró después la causa: el Mac entró en reposo por inactividad a las 11:02 y siguió en "
        "reposo hasta después de la ejecución, despertando solo unos 45 segundos cada 15 minutos, aunque estaba "
        "conectado a la corriente. Ningún resultado de la primera ejecución se perdió ni se cambió.",
        "",
        "La primera ejecución sigue siendo el resultado que se informa (`data/evaluation/known_item/KNOWN_ITEM.md`). "
        "Esta es una repetición con la máquina despierta (`caffeinate -i`), con la misma muestra, las mismas "
        "consultas y la misma división en mitades, que recalcula todo lo demás: embeddings de consulta, S1, S2, "
        "fusiones, elección del peso y reranker. Se informa junto a la primera, sea cual sea su resultado.",
        "",
        f"Carpeta original: `{c['folders']['a']}`. Carpeta de la réplica: `{c['folders']['b']}`.",
        "",
        "## Comparación",
        "",
        "### Listas guardadas por sistema",
        "",
        f"Para cada sistema, cuántas de las consultas tienen exactamente la misma lista de ids en las dos "
        f"ejecuciones (los diez primeros y la lista completa). S4, S5 y S5w solo existen en la mitad de prueba.",
        "",
    ]
    lines += _table(["Sistema", "Consultas", "Top 10 idéntico", "Top 10 distinto", "Top 10 distinto (prueba)",
                     "Lista completa idéntica", "Lista completa distinta"],
                    [[s, r["queries"], r["identical_top10"], r["different_top10"], r["different_top10_test"],
                      r["identical_full"], r["different_full"]] for s, r in c["lists"].items()])
    w = c["weight"]
    lines += ["### Peso elegido en la mitad de desarrollo", "",
              f"Original: w = {w['a']}. Réplica: w = {w['b']}. "
              + ("Es el mismo peso." if w["a"] == w["b"] else "El peso elegido cambia."), ""]
    if w["objective_a"] and w["objective_b"]:
        lines += _table(["Ejecución", "Objetivo original", "Objetivo réplica"],
                        [[k, _f(w["objective_a"].get(k), 4), _f(w["objective_b"].get(k), 4)] for k in w["objective_a"]])
    lines += ["### Métrica principal en la mitad de prueba", "",
              "Media de la métrica principal de cada tipo de consulta en las dos ejecuciones, cuántas consultas "
              "cambian de valor y el mayor cambio de una consulta (réplica menos original).", ""]
    rows = []
    for qt, systems in c["metric_changes_test"].items():
        for s, r in systems.items():
            rows.append([qt, s, METRIC_NAMES[r["metric"]], r["queries"], _f(r["mean_a"], 4), _f(r["mean_b"], 4),
                         r["queries_changed"], _f(r["largest_change"], 3, signed=True)])
    lines += _table(["Tipo", "Sistema", "Métrica", "n", "Original", "Réplica", "Consultas con cambio",
                     "Mayor cambio"], rows)
    lines += ["### Contrastes declarados (mitad de prueba, Holm sobre seis)", ""]
    rows = []
    for name, r in c["contrasts"].items():
        rows.append([name] + _contrast_cell(r["a"]) + _contrast_cell(r["b"])
                    + ["sí" if r["same_decision"] else ("no" if r["same_decision"] is False else "n/a")])
    lines += _table(["Contraste", "Diferencia original", "p Holm original", "Significativo original",
                     "Diferencia réplica", "p Holm réplica", "Significativo réplica", "Misma conclusión"], rows)
    changed = [qt for qt, (da, db) in c["decision_same"].items() if da != db]
    lines += [("La regla de decisión de la sección 8 da la misma conclusión en los tres tipos de consulta."
               if not changed else
               f"La regla de decisión cambia de conclusión en: {', '.join(changed)}."), ""]
    e, r = c["embeddings"], c["rerank"]
    lines += ["## Diferencias y su causa probable", "",
              f"Vectores de consulta: {e['identical']} de {e['queries']} son idénticos bit a bit; {e['different']} "
              f"difieren" + (f" (similitud del coseno mínima {e['min_cosine']:.6f})" if e["min_cosine"] is not None
                             else "") + ". "
              f"Proveedor que sirvió los vectores: {', '.join(e['providers'])}; el mismo en las dos ejecuciones para "
              f"{e['same_provider']} de {e['queries']} consultas, y para {e['different_same_provider']} de las "
              f"{e['different']} cuyo vector difiere. Es decir, el proveedor devolvió vectores ligeramente distintos "
              "para el mismo texto." if e["different"] and e["different_same_provider"] == e["different"]
              else f"Vectores de consulta: {e['identical']} de {e['queries']} idénticos.",
              "",
              f"Puntuaciones del reranker: de las {r['queries']} consultas de prueba, {r['different']} tienen algún "
              f"libro puntuado en las dos ejecuciones con puntuación distinta (mayor cambio {r['largest_score_change']:.1e}); "
              f"en {r['different_book_sets']} se puntuó un conjunto de libros distinto porque las cabezas que se "
              "reordenan ya eran distintas.",
              ""]
    causes = {s: v for s, v in c["causes"].items() if v}
    if not causes:
        lines += ["No hay listas distintas: no hay diferencia que explicar.", ""]
    else:
        lines += ["Consultas con lista completa distinta, por la causa que muestran los ficheros. Una diferencia se "
                  "atribuye a una causa solo cuando los datos guardados la muestran; si no, se dice «causa no "
                  "determinada».", ""]
        lines += _table(["Sistema", "Causa", "Consultas"],
                        [[s, cause, n] for s, v in causes.items() for cause, n in v.items()])
    lines += ["## Desviaciones, gasto y duración de la réplica", ""]
    lines += ([f"- {d['desviacion']}. Motivo: {d['motivo']}." for d in c["deviations_b"]]
              or ["Ninguna desviación: la réplica siguió el protocolo sin cambios."])
    spend = c["spend"]
    lines += ["",
              f"Gasto según el registro de gasto: réplica {_f(spend['b'], 4)} USD, original {_f(spend['a'], 4)} USD "
              f"(tope {D.SPEND_CAP_USD:.2f} USD en cada una).",
              "",
              f"Duración, de la primera a la última entrada del registro de gasto: réplica "
              f"{_duration(t_b['span_seconds'])} ({_invocations(t_b['invocations'])}); original "
              f"{_duration(t_a['span_seconds'])} ({_invocations(t_a['invocations'])}, con la máquina en reposo).", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_comparison(original: Path, replication: Path, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Compare the two folders and write replication.json and REPLICATION.md into `out_dir` (the replication's)."""
    out_dir = out_dir or replication
    result = compare(load_folder(original), load_folder(replication))
    (out_dir / REPLICATION_JSON).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                            encoding="utf-8")
    (out_dir / REPLICATION_MD).write_text(build_markdown(result), encoding="utf-8")
    return result
