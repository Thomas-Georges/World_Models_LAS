#!/usr/bin/env python
"""Aggregate local/global training and planning artifacts into a summary CSV.

Torch-free and safe to run anywhere; missing runs, planners, or metrics simply
produce fewer rows.

Example:
    python scripts/local_global/summarize_runs.py \
        --run-root "$LG_RUN_ROOT" --out "$LG_RUN_ROOT/_summary/summary.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wm_poc.local_global.visualization import (  # noqa: E402
    aggregate_summary,
    write_summary_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", default=None, help="default <run-root>/_summary/summary.csv")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser()
    out = Path(args.out) if args.out else run_root / "_summary" / "summary.csv"
    rows = aggregate_summary(run_root)
    write_summary_csv(rows, out)
    planners = sorted({str(r["planner"]) for r in rows if r.get("planner")})
    print(f"Summarized {len(rows)} rows from {run_root} -> {out}")
    if planners:
        print(f"Planners seen: {', '.join(planners)}")
    elif not rows:
        print("No runs found yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
