#!/usr/bin/env python3
"""Fail if any committed notebook carries an ``error`` output cell.

Result notebooks (06/07/08) are committed *with* their final outputs because the
report reads numbers and figures from them, so a stale traceback left in a cell
would silently undermine the reproducibility claim. This check is the guard:
it scans every ``notebooks/*.ipynb`` and exits non-zero if any cell has an
``output_type == "error"`` output, printing the offending notebook, cell index,
and exception so it can be fixed or the cell re-run/cleared.

Usage:
    python scripts/check_notebook_errors.py [notebooks_dir]

Exit code 0 means no error outputs; 1 means at least one was found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def find_error_outputs(nb_path: Path) -> list[tuple[int, str, str]]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    errors: list[tuple[int, str, str]] = []
    for index, cell in enumerate(nb.get("cells", []), start=1):
        for output in cell.get("outputs", []) or []:
            if output.get("output_type") == "error":
                errors.append((index, output.get("ename", ""), str(output.get("evalue", ""))[:120]))
    return errors


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / "notebooks"
    if not root.exists():
        print(f"notebooks directory not found: {root}", file=sys.stderr)
        return 1

    found = False
    for nb_path in sorted(root.glob("*.ipynb")):
        errors = find_error_outputs(nb_path)
        if errors:
            found = True
            print(f"{nb_path}:")
            for cell_index, ename, evalue in errors:
                print(f"  cell {cell_index}: {ename}: {evalue}")

    if found:
        print("\nFAIL: notebooks contain stale error outputs (see above).", file=sys.stderr)
        return 1
    print("OK: no notebook error outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
