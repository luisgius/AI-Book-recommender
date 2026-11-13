#!/usr/bin/env python3
"""
Compute the project budget for the thesis from its calculation bases (V28).

    python -m scripts.presupuesto [--bases docs/notes/presupuesto/bases.json] [--out docs/notes/presupuesto/PRESUPUESTO.md]

Reads the bases (bases.json) and the OpenRouter usage export (CSV), computes
each item with decimal.Decimal, rounding half up to cents at every printed
step and computing later steps from the rounded value, prints the table and
writes the Markdown report. No network, standard library only.
"""

import argparse
import csv
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_BASES = Path("docs/notes/presupuesto/bases.json")
DEFAULT_OUT = Path("docs/notes/presupuesto/PRESUPUESTO.md")
CENT = Decimal("0.01")
REQUIRED = [
    "horas", "salario_anual", "horas_anuales", "portatil_precio", "portatil_amortizacion",
    "openrouter_csv", "suscripcion_mensual", "suscripcion_meses", "software_libre", "tipo_cambio",
]


def cents(value: Decimal) -> Decimal:
    """Round half up to cents."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def fmt(value: Decimal) -> str:
    """Format with a space as thousands separator and a decimal point: 4 196.29."""
    return f"{value:,.2f}".replace(",", " ")


def load_bases(path: Path) -> Dict[str, Any]:
    """Read bases.json and fail with a clear message if a base is missing."""
    bases = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in REQUIRED:
        if name not in bases or "valor" not in bases[name]:
            raise ValueError(f"Missing base '{name}' (with a 'valor') in {path}")
    return bases


def csv_total_usd(path: Path) -> Decimal:
    """Sum the Value column (USD) of the OpenRouter usage export."""
    with open(path, newline="", encoding="utf-8") as handle:
        return sum((Decimal(row["Value"]) for row in csv.DictReader(handle)), Decimal("0"))


def compute(bases: Dict[str, Any], root: Path = Path(".")) -> Dict[str, Decimal]:
    """Compute every figure of the budget from the bases (EUR unless the key says usd)."""
    hours = Decimal(str(bases["horas"]["valor"]))
    per_hour = cents(Decimal(bases["salario_anual"]["valor"]) / Decimal(str(bases["horas_anuales"]["valor"])))
    personnel = cents(hours * per_hour)
    laptop_year = cents(Decimal(bases["portatil_precio"]["valor"]) * Decimal(bases["portatil_amortizacion"]["valor"]))
    laptop = cents(laptop_year * hours / Decimal(str(bases["horas_anuales"]["valor"])))
    rate = Decimal(bases["tipo_cambio"]["valor"])
    openrouter_usd_exact = csv_total_usd(root / bases["openrouter_csv"]["valor"])
    openrouter_usd = cents(openrouter_usd_exact)
    openrouter = cents(openrouter_usd_exact / rate)
    months = len(bases["suscripcion_meses"]["valor"])
    subscription_usd = cents(months * Decimal(bases["suscripcion_mensual"]["valor"]))
    subscription = cents(subscription_usd / rate)
    software = cents(Decimal(bases["software_libre"]["valor"]))
    total = personnel + laptop + openrouter + subscription + software
    return {
        "per_hour": per_hour, "personnel": personnel, "laptop_year": laptop_year, "laptop": laptop,
        "openrouter_usd_exact": openrouter_usd_exact, "openrouter_usd": openrouter_usd, "openrouter": openrouter,
        "subscription_usd": subscription_usd, "subscription": subscription, "software": software,
        "total": total,
    }


def items(bases: Dict[str, Any], r: Dict[str, Decimal]) -> List[Tuple[str, str, Decimal]]:
    """The table rows: (item, base of calculation, amount in EUR)."""
    hours = bases["horas"]["valor"]
    annual = bases["horas_anuales"]["valor"]
    months = len(bases["suscripcion_meses"]["valor"])
    return [
        ("Personal (autor)", f"{hours} h x {fmt(r['per_hour'])} EUR/h "
         f"({bases['salario_anual']['valor']} EUR / {annual} h)", r["personnel"]),
        ("Portatil (amortizacion)", f"{bases['portatil_precio']['valor']} EUR x {bases['portatil_amortizacion']['valor']} "
         f"= {fmt(r['laptop_year'])} EUR/anio; x {hours}/{annual} h", r["laptop"]),
        ("API de modelos (OpenRouter)", f"{fmt(r['openrouter_usd'])} USD / {bases['tipo_cambio']['valor']} USD por EUR",
         r["openrouter"]),
        ("Suscripcion a asistente de IA", f"{months} meses x {bases['suscripcion_mensual']['valor']} USD = "
         f"{fmt(r['subscription_usd'])} USD / {bases['tipo_cambio']['valor']}", r["subscription"]),
        ("Software libre y planes gratuitos", "Python, FastAPI, Neon, Vercel, TinyTeX", r["software"]),
    ]


def render(bases: Dict[str, Any], r: Dict[str, Decimal]) -> str:
    """The Spanish Markdown report."""
    lines = [
        "# Presupuesto del proyecto", "",
        f"Calculado por `scripts/presupuesto.py` a partir de `docs/notes/presupuesto/bases.json` "
        f"(bases verificadas el {bases.get('fecha_verificacion', 'n/d')}). Redondeo: mitad hacia arriba a centimos "
        "en cada paso, y los pasos siguientes se calculan con el valor redondeado.", "",
        "| Concepto | Base de calculo | Importe (EUR) |", "|---|---|---:|",
    ]
    for name, base, amount in items(bases, r):
        lines.append(f"| {name} | {base} | {fmt(amount)} |")
    lines += [f"| **Total** | | **{fmt(r['total'])}** |", "",
              "No incluye las horas del tutor ni las cuotas patronales de la Seguridad Social.", "",
              "## Fuentes", ""]
    for key in REQUIRED:
        lines.append(f"- **{key}**: {bases[key]['valor'] if key != 'suscripcion_meses' else ', '.join(bases[key]['valor'])} "
                     f"({bases[key]['unidad']}). {bases[key]['fuente']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Compute the thesis budget.")
    parser.add_argument("--bases", type=Path, default=DEFAULT_BASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    bases = load_bases(args.bases)
    results = compute(bases)
    for name, base, amount in items(bases, results):
        print(f"{name:36} {fmt(amount):>10} EUR   {base}")
    print(f"{'Total':36} {fmt(results['total']):>10} EUR")
    args.out.write_text(render(bases, results), encoding="utf-8")
    print(f"Written {args.out}")


if __name__ == "__main__":
    main()
