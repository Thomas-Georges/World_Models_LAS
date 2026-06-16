#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.posthoc import collect_summaries, write_summary_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DINO-WM run summaries.")
    parser.add_argument("--root", type=Path, required=True, help="Root containing run folders.")
    parser.add_argument("--out", type=Path, required=True, help="Summary output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_summaries(args.root)
    outputs = write_summary_outputs(rows, args.out)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
