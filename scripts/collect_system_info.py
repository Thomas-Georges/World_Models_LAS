#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.system_info import collect_system_info  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write system metadata to JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = collect_system_info(cwd=REPO_ROOT)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote system info to {args.output.expanduser()}")


if __name__ == "__main__":
    main()
