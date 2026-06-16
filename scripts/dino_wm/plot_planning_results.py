#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.visualization import plot_planning_success  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DINO-WM planning summary.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot_planning_success(args.summary_csv.expanduser(), args.out.expanduser())
    print(f"Wrote {args.out.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
