"""Guard against malformed committed report CSVs.

Uses only the stdlib ``csv`` module (no pandas) so it runs in the base test
environment. Catches the failure mode where an unquoted comma in a notes field
splits a row into the wrong number of columns -- which would otherwise only
surface when a downstream pandas reader raises a ParserError.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
CSV_FILES = sorted(REPORTS_DIR.glob("*.csv"))


def test_report_csvs_exist():
    assert CSV_FILES, "expected committed CSVs under reports/"


@pytest.mark.parametrize("path", CSV_FILES, ids=lambda p: p.name)
def test_report_csv_has_consistent_columns(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows, f"{path.name} is empty"
    expected = len(rows[0])
    for i, row in enumerate(rows, start=1):
        assert len(row) == expected, (
            f"{path.name}:{i}: expected {expected} fields, got {len(row)} "
            f"(unquoted comma in a field?)"
        )
