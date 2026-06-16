from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from wm_poc.dino_wm.checkpoints import find_latest_checkpoint


# Upstream logs_flash() writes one line per epoch to the run's stdout log:
# "Epoch 3  Training loss: 0.0123                 Validation loss: 0.0456"
EPOCH_LOSS_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+Training loss:\s*(?P<train>[-+0-9.eE]+)"
    r"\s+Validation loss:\s*(?P<val>[-+0-9.eE]+)"
)


def epoch_loss_series(run_dir: str | Path) -> list[dict[str, Any]]:
    """Per-epoch train/val losses parsed from a run's stdout/launcher logs.

    The no-decoder runs do not write metrics.jsonl; the epoch losses only
    exist in the upstream log line emitted at each epoch boundary. Every
    rotated generation of the logs (stdout.log, stdout.log.prev,
    stdout.log.<stamp>.prev, ...) is merged in modification-time order so a
    run relaunched several times keeps its full curve; for a given epoch the
    newest log wins.
    """

    run_dir = Path(run_dir).expanduser()
    candidates = [
        path
        for pattern in ("launcher.log*", "stdout.log*")
        for path in run_dir.glob(pattern)
        if path.is_file()
    ]
    by_epoch: dict[int, dict[str, Any]] = {}
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in EPOCH_LOSS_PATTERN.finditer(text):
            epoch = int(match.group("epoch"))
            by_epoch[epoch] = {
                "epoch": epoch,
                "train_loss": float(match.group("train")),
                "val_loss": float(match.group("val")),
            }
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


TRAINING_KEYS = {
    "epoch",
    "step",
    "wall_seconds",
    "train/loss_pred",
    "val/loss_pred_1step",
    "val/loss_pred_hstep",
    "val/latent_cosine",
    "lr/predictor",
    "lr/action_encoder",
    "gpu/max_memory_gb",
}

PLANNING_KEYS = {
    "episode",
    "success",
    "success_rate",
    "final_goal_latent_distance",
    "final_env_distance",
    "planner",
    "cem_iterations",
    "candidate_count",
    "plan_time_seconds",
    "failure_reason",
}


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _json_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("metrics"), list):
            return [row for row in payload["metrics"] if isinstance(row, dict)]
        if isinstance(payload.get("episodes"), list):
            return [row for row in payload["episodes"] if isinstance(row, dict)]
        return [payload]
    return []


def _normalize_training(record: dict[str, Any]) -> dict[str, Any] | None:
    normalized: dict[str, Any] = {"stage": "train"}
    aliases = {
        "loss_pred": "train/loss_pred",
        "train_loss_pred": "train/loss_pred",
        "val_loss_pred_1step": "val/loss_pred_1step",
        "val_loss_pred_hstep": "val/loss_pred_hstep",
        "latent_cosine": "val/latent_cosine",
        "max_memory_gb": "gpu/max_memory_gb",
    }
    for key, value in record.items():
        target = aliases.get(key, key)
        if target in TRAINING_KEYS:
            normalized[target] = value
    return normalized if len(normalized) > 1 else None


def _normalize_planning(record: dict[str, Any]) -> dict[str, Any] | None:
    normalized: dict[str, Any] = {"stage": "planning"}
    aliases = {
        "is_success": "success",
        "goal_latent_distance": "final_goal_latent_distance",
        "final_goal_distance": "final_goal_latent_distance",
        "env_distance": "final_env_distance",
        "opt_steps": "cem_iterations",
        "samples": "candidate_count",
        "elapsed_seconds": "plan_time_seconds",
    }
    for key, value in record.items():
        target = aliases.get(key, key)
        if target in PLANNING_KEYS:
            normalized[target] = value
    return normalized if len(normalized) > 1 else None


def _candidate_records(run_dir: Path) -> list[dict[str, Any]]:
    files = [
        run_dir / "metrics.jsonl",
        run_dir / "train_metrics.jsonl",
        run_dir / "training_metrics.jsonl",
        run_dir / "planning" / "metrics.jsonl",
        run_dir / "planning" / "planning_metrics.jsonl",
        run_dir / "planning" / "results.jsonl",
        run_dir / "planning" / "results.json",
    ]
    records: list[dict[str, Any]] = []
    for path in files:
        if path.suffix == ".jsonl":
            records.extend(_jsonl_records(path))
        elif path.suffix == ".json":
            records.extend(_json_records(path))
    return records


def parse_training_logs(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir).expanduser()
    records = []
    for record in _candidate_records(root):
        normalized = _normalize_training(record)
        if normalized is not None:
            records.append(normalized)
    return records


def _planning_log_records(run_dir: Path) -> list[dict[str, Any]]:
    """Per-evaluation entries upstream plan.py appends to planning/*/logs.json
    (JSONL despite the suffix), with the final_eval/ prefix stripped."""

    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("planning/*/logs.json")):
        for record in _jsonl_records(path):
            records.append(
                {key.removeprefix("final_eval/"): value for key, value in record.items()}
            )
    return records


def parse_planning_logs(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir).expanduser()
    records = []
    for record in _candidate_records(root) + _planning_log_records(root):
        normalized = _normalize_planning(record)
        if normalized is not None:
            records.append(normalized)
    return records


def write_metrics_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(output)


def _latest(records: list[dict[str, Any]], key: str) -> Any:
    for record in reversed(records):
        if key in record:
            return record[key]
    return None


def _latest_success(records: list[dict[str, Any]]) -> float | None:
    """Success rate of the *most recent* planning run.

    Upstream plan.py appends a fresh ``final_eval/success_rate`` row to
    planning/*/logs.json on every re-plan, so a run first evaluated at one
    ``n_evals`` and later re-planned at another leaves several rows in the same
    file. Take the last one -- NOT ``max()`` across rows -- otherwise an older,
    noisier (often smaller-``n_evals``) run with a higher point estimate masks
    the current result. (Records arrive in file/append order, so the last
    ``success_rate`` is the latest final_eval.)
    """
    latest = _latest(records, "success_rate")
    if latest is not None:
        return float(latest)
    # Episode-level fallback (episodes.jsonl-style): aggregate the per-episode
    # success flags when no final_eval row is present.
    successes = [record.get("success") for record in records if record.get("success") is not None]
    if successes:
        return sum(bool(value) for value in successes) / len(successes)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser()
    resolved = _read_json(root / "resolved_config.json")
    if not resolved:
        try:
            import yaml

            if (root / "resolved_config.yaml").is_file():
                resolved = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
        except Exception:
            resolved = {}

    status = _read_json(root / "status.json")
    train_records = parse_training_logs(root)
    plan_records = parse_planning_logs(root)
    checkpoint = find_latest_checkpoint(root)

    # The no-decoder pipeline writes no metrics.jsonl: epoch losses live in
    # the rotated stdout logs, and planning wall time in planning/status_*.json.
    final_val_loss = _latest(train_records, "val/loss_pred_hstep")
    best_epoch = _latest(train_records, "epoch")
    if final_val_loss is None or best_epoch is None:
        series = epoch_loss_series(root)
        if series:
            if final_val_loss is None:
                final_val_loss = series[-1]["val_loss"]
            if best_epoch is None:
                best_epoch = min(series, key=lambda record: record["val_loss"])["epoch"]

    plan_wall_minutes = _latest(plan_records, "plan_time_seconds")
    if plan_wall_minutes is not None:
        plan_wall_minutes = float(plan_wall_minutes) / 60
    else:
        elapsed = [
            payload.get("elapsed_seconds")
            for payload in (_read_json(path) for path in sorted(root.glob("planning/status_*.json")))
            if payload.get("elapsed_seconds") is not None
        ]
        plan_wall_minutes = max(float(value) for value in elapsed) / 60 if elapsed else None
    mode = "finetune" if resolved.get("finetuning", {}).get("enabled") else "scratch"
    if resolved.get("planner_ablation"):
        mode = "planner_only"
    if "smoke" in root.name:
        mode = "smoke"

    row = {
        "run_name": root.name,
        "config_name": (root / "config.yaml").name if (root / "config.yaml").is_file() else "",
        "env": resolved.get("dataset", {}).get("env"),
        "seed": resolved.get("training", {}).get("seed"),
        "mode": mode,
        "source_checkpoint": resolved.get("finetuning", {}).get("init_from")
        or resolved.get("planner_ablation", {}).get("checkpoint"),
        "max_train_trajectories": resolved.get("dataset", {}).get("max_train_trajectories"),
        "max_val_trajectories": resolved.get("dataset", {}).get("max_val_trajectories"),
        "completed": bool(status.get("completed", False)),
        "timed_out": bool(status.get("timed_out", False)),
        "failed": bool(status.get("failed", False)),
        "train_wall_minutes": status.get("elapsed_seconds", 0) / 60 if status else None,
        "plan_wall_minutes": plan_wall_minutes,
        "best_epoch": best_epoch,
        "final_val_loss_pred_hstep": final_val_loss,
        "best_success_rate": _latest_success(plan_records),
        "final_goal_latent_distance": _latest(plan_records, "final_goal_latent_distance"),
        "checkpoint_path": str(checkpoint) if checkpoint else "",
        "run_dir": str(root),
        "checkpoint_exists": bool(checkpoint and checkpoint.is_file()),
        "config_copied": (root / "config.yaml").is_file(),
        "resolved_config_copied": (root / "resolved_config.yaml").is_file(),
    }
    summary_path = root / "summary.json"
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    tmp.replace(summary_path)
    return row
