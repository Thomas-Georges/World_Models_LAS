#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_value(records: list[dict[str, Any]], key: str) -> float | None:
    for record in reversed(records):
        if key in record:
            return float(record[key])
    return None


def best_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if key in record]
    return max(values) if values else None


def summarize_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_metrics(run_dir / "metrics.jsonl")
    ckpt = run_dir / "latest.pt"
    eval_key = "episode/eval_score"
    train_key = "episode/score"
    return {
        "run_name": run_dir.name,
        "final_eval_score": latest_value(metrics, eval_key),
        "best_eval_score": best_value(metrics, eval_key),
        "final_train_episode_score": latest_value(metrics, train_key),
        "metric_rows": len(metrics),
        "checkpoint_exists": ckpt.is_file(),
        "checkpoint_size_bytes": ckpt.stat().st_size if ckpt.is_file() else 0,
        "has_console_log": (run_dir / "console.log").is_file(),
        "has_metrics_jsonl": (run_dir / "metrics.jsonl").is_file(),
    }


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "run_name",
        "final_eval_score",
        "best_eval_score",
        "final_train_episode_score",
        "metric_rows",
        "checkpoint_exists",
    ]
    lines = [
        "# R2-Dreamer run summary",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(header, "")) for header in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize R2-Dreamer run folders.")
    parser.add_argument("--run-root", type=Path, required=True, help="Root containing run folders.")
    parser.add_argument("--out", type=Path, required=True, help="Output summary CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser()
    rows = [
        summarize_run(path)
        for path in sorted(run_root.iterdir())
        if path.is_dir() and path.name in {"smoke", "source_base", "target_finetune", "target_scratch"}
    ]
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "final_eval_score",
        "best_eval_score",
        "final_train_episode_score",
        "metric_rows",
        "checkpoint_exists",
        "checkpoint_size_bytes",
        "has_console_log",
        "has_metrics_jsonl",
    ]
    with args.out.expanduser().open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.out.expanduser().with_suffix(".md"))
    print(f"Wrote {args.out.expanduser()} and {args.out.expanduser().with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
