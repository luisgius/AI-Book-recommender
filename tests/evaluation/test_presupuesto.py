"""The thesis budget (V28): totals from the bases, the CSV sum and missing bases. Temporary paths only."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from scripts import presupuesto as pr

REPO = Path(__file__).resolve().parents[2]
BASES = REPO / "docs/notes/presupuesto/bases.json"


def test_figures_of_the_budget():
    r = pr.compute(pr.load_bases(BASES), root=REPO)
    assert r["per_hour"] == Decimal("13.39")
    assert r["personnel"] == Decimal("4017.00")
    assert r["laptop_year"] == Decimal("500.00")
    assert r["laptop"] == Decimal("83.33")
    assert r["openrouter_usd"] == Decimal("6.34")
    assert r["openrouter"] == Decimal("5.72")
    assert r["subscription_usd"] == Decimal("100.00")
    assert r["subscription"] == Decimal("90.24")
    assert r["total"] == Decimal("4196.29")


def test_csv_total_is_the_sum_of_the_values():
    csv_path = REPO / pr.load_bases(BASES)["openrouter_csv"]["valor"]
    assert pr.csv_total_usd(csv_path) == Decimal("6.339355")


def test_missing_base_raises_a_clear_error(tmp_path):
    bases = json.loads(BASES.read_text(encoding="utf-8"))
    del bases["tipo_cambio"]
    path = tmp_path / "bases.json"
    path.write_text(json.dumps(bases), encoding="utf-8")
    with pytest.raises(ValueError, match="tipo_cambio"):
        pr.load_bases(path)


def test_report_is_written_and_has_the_total(tmp_path):
    bases = pr.load_bases(BASES)
    text = pr.render(bases, pr.compute(bases, root=REPO))
    out = tmp_path / "PRESUPUESTO.md"
    out.write_text(text, encoding="utf-8")
    assert "4 196.29" in out.read_text(encoding="utf-8")
