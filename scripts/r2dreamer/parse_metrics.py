#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.r2dreamer.metrics import parse_metrics_to_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert r2dreamer metrics.jsonl to CSV.")
    parser.add_argument("--metrics", type=Path, help="Input metrics.jsonl.")
    parser.add_argument("--out", type=Path, help="Output CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.metrics is None or args.out is None:
        print("Supply --metrics and --out to parse metrics.")
        return 0
    try:
        columns = parse_metrics_to_csv(args.metrics, args.out)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.out} with columns: {', '.join(columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
