#!/usr/bin/env python3
"""Regenerate the Track III six-planner table from committed telemetry.

Reads the per-planner summary CSV (reports/local_global_planning_summary.csv,
the committed source for Table III in reports/world_models_report.tex) and prints
both a readable table and the LaTeX tabular body, so the report table is
reproducible from a committed raw artifact with one command:

    python reports/make_planning_table.py reports/local_global_planning_summary.csv

The raw *per-episode* telemetry that these rows aggregate is produced by results
notebook 08 and lives on Drive (not committed; see the artifact policy). This
script reproduces the summary table, not the per-episode log.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

def _num(v: str, fmt: str) -> str:
    return format(float(v), fmt) if v not in (None, "") else "---"


def _wall(v: str) -> str:
    # Match the report: 2 decimals for sub-10s local planners, 1 for the rest.
    if v in (None, ""):
        return "---"
    f = float(v)
    return f"{f:.2f}" if f < 10 else f"{f:.1f}"


def _calls(v: str) -> str:
    if v in (None, ""):
        return "---"
    n = int(float(v))
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n // 1000}k"
    return str(n)


# (csv column, header, formatter) in report-table order.
COLUMNS = [
    ("planner", "Planner", lambda v: v.replace("_", r"\_") if v else "---"),
    ("success_rate", "succ.", lambda v: _num(v, ".2f")),
    ("normalized_final_distance", "rho", lambda v: _num(v, ".3f")),
    ("wall_time_per_episode_sec", "wall/ep (s)", _wall),
    ("global_forward_calls", "glob. fwd", _calls),
    ("local_forward_calls", "loc. fwd", _calls),
    ("backward_steps", "bwd", _calls),
    ("acceptance_rate", "accept", lambda v: _num(v, ".3f")),
    ("local_global_disagreement", "disagr.", lambda v: _num(v, ".4f")),
]


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def render_plain(rows: list[dict[str, str]]) -> str:
    headers = [h for _, h, _ in COLUMNS]
    table = [headers]
    for row in rows:
        table.append([fmt(row.get(col, "")) for col, _, fmt in COLUMNS])
    widths = [max(len(r[i]) for r in table) for i in range(len(headers))]
    lines = []
    for r, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if r == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(lines)


def render_latex(rows: list[dict[str, str]]) -> str:
    lines = []
    for row in rows:
        cells = [fmt(row.get(col, "")) for col, _, fmt in COLUMNS]
        lines.append("  " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    default = Path(__file__).resolve().parent / "local_global_planning_summary.csv"
    csv_path = Path(argv[1]) if len(argv) > 1 else default
    if not csv_path.is_file():
        print(f"summary CSV not found: {csv_path}", file=sys.stderr)
        return 1
    rows = load_rows(csv_path)
    print("# Track III planner comparison (from", csv_path.name + ")\n")
    print(render_plain(rows))
    print("\n# LaTeX tabular body:\n")
    print(render_latex(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
