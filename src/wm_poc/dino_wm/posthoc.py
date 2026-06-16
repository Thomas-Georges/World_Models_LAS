from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from wm_poc.dino_wm.metrics import summarize_run


SUMMARY_FIELDS = [
    "run_name",
    "config_name",
    "env",
    "seed",
    "mode",
    "source_checkpoint",
    "max_train_trajectories",
    "max_val_trajectories",
    "completed",
    "timed_out",
    "failed",
    "train_wall_minutes",
    "plan_wall_minutes",
    "best_epoch",
    "final_val_loss_pred_hstep",
    "best_success_rate",
    "final_goal_latent_distance",
    "checkpoint_path",
    "run_dir",
]


def _run_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and not path.name.startswith("_") and (path / "status.json").exists()
    ]


def collect_summaries(root: str | Path) -> list[dict[str, Any]]:
    return [summarize_run(path) for path in _run_dirs(Path(root).expanduser())]


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] = SUMMARY_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _best_by_env(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        env = str(row.get("env") or "")
        if not env:
            continue
        current = best.get(env)
        score = row.get("best_success_rate")
        loss = row.get("final_val_loss_pred_hstep")
        if current is None:
            best[env] = row
            continue
        current_score = current.get("best_success_rate")
        current_loss = current.get("final_val_loss_pred_hstep")
        if score is not None and (current_score is None or float(score) > float(current_score)):
            best[env] = row
        elif score is None and loss is not None and (
            current_loss is None or float(loss) < float(current_loss)
        ):
            best[env] = row
    return list(best.values())


def _scratch_vs_finetune(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparable = []
    for row in rows:
        if row.get("mode") in {"scratch", "finetune"}:
            comparable.append(row)
    return comparable


def _planner_ablation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("mode") == "planner_only" or "planner" in str(row.get("run_name", ""))
    ]


def write_summary_outputs(rows: list[dict[str, Any]], out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir).expanduser()
    outputs = {
        "summary_csv": out / "summary.csv",
        "summary_json": out / "summary.json",
        "best_by_env_csv": out / "best_by_env.csv",
        "scratch_vs_finetune_csv": out / "scratch_vs_finetune.csv",
        "planner_ablation_csv": out / "planner_ablation.csv",
    }
    _write_csv(rows, outputs["summary_csv"])
    _write_json(rows, outputs["summary_json"])
    _write_csv(_best_by_env(rows), outputs["best_by_env_csv"])
    _write_csv(_scratch_vs_finetune(rows), outputs["scratch_vs_finetune_csv"])
    _write_csv(_planner_ablation(rows), outputs["planner_ablation_csv"])
    return outputs
