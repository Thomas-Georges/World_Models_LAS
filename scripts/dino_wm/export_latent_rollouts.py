#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DINO-WM latent rollout records.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [
        row
        for row in _records(args.run_dir.expanduser() / "metrics.jsonl")
        if row.get("final_goal_latent_distance") is not None
    ]
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["episode", "planner", "final_goal_latent_distance", "plan_time_seconds", "success"]
    with args.out.expanduser().open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {args.out.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
