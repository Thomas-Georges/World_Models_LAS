from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PREFERRED_COLUMNS = [
    "step",
    "episode/eval_score",
    "episode/eval_length",
    "episode/score",
    "episode/length",
    "fps/fps",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}.")
            records.append(record)
    if not records:
        raise ValueError(f"No metrics records found in {path}.")
    return records


def metric_columns(records: list[dict[str, Any]]) -> list[str]:
    keys = {key for record in records for key in record}
    ordered = [key for key in PREFERRED_COLUMNS if key in keys]
    train_keys = sorted(key for key in keys if key.startswith("train/") and key not in ordered)
    remaining = sorted(keys - set(ordered) - set(train_keys))
    return ordered + train_keys + remaining


def write_csv(records: list[dict[str, Any]], output: Path) -> list[str]:
    columns = metric_columns(records)
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return columns


def parse_metrics_to_csv(metrics: Path, output: Path) -> list[str]:
    records = read_jsonl(metrics)
    return write_csv(records, output)


def score_key(records: list[dict[str, Any]]) -> str:
    if any("episode/eval_score" in record for record in records):
        return "episode/eval_score"
    if any("episode/score" in record for record in records):
        return "episode/score"
    raise ValueError("No episode/eval_score or episode/score metric found.")


def numeric_series(records: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    values: list[float] = []
    for record in records:
        if "step" not in record or key not in record:
            continue
        steps.append(float(record["step"]))
        values.append(float(record[key]))
    if not steps:
        raise ValueError(f"No plottable values for {key}.")
    return steps, values
