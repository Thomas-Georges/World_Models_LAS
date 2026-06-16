from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wm_poc.dino_wm.configs import get_config_value


def _as_env_flag(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def checkpoint_output_dir(config: dict[str, Any]) -> Path:
    run_name = config.get("run_name")
    if not run_name:
        raise ValueError("DINO-WM config must define run_name before checkpoint resume handling.")
    ckpt_root = Path(str(get_config_value(config, "artifacts.ckpt_root"))).expanduser()
    return ckpt_root / "outputs" / str(run_name)


def latest_checkpoint_path(config: dict[str, Any]) -> Path:
    return checkpoint_output_dir(config) / "checkpoints" / "model_latest.pth"


def _unique_backup_dir(output_dir: Path, timestamp: str) -> Path:
    candidate = output_dir / f"checkpoints_fresh_start_backup_{timestamp}"
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = output_dir / f"checkpoints_fresh_start_backup_{timestamp}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def prepare_training_resume(
    config: dict[str, Any],
    *,
    force_restart: bool | None = None,
    now: datetime | None = None,
) -> dict[str, str]:
    """Prepare upstream DINO-WM checkpoint state before launching train.py.

    The upstream trainer automatically resumes from ``checkpoints/model_latest.pth``
    when it exists. A true fresh start therefore means making that checkpoint
    directory unavailable before the trainer starts. We move it aside rather than
    delete it, because these checkpoints are expensive Drive artifacts.
    """

    output_dir = checkpoint_output_dir(config)
    checkpoint_dir = output_dir / "checkpoints"
    latest = checkpoint_dir / "model_latest.pth"
    resume_enabled = bool(get_config_value(config, "training.resume", True))
    requested_restart = _as_env_flag(os.environ.get("DINO_FORCE_RESTART"))
    if force_restart is not None:
        requested_restart = force_restart

    if checkpoint_dir.exists() and (requested_restart or not resume_enabled):
        timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        output_dir.mkdir(parents=True, exist_ok=True)
        backup_dir = _unique_backup_dir(output_dir, timestamp)
        shutil.move(str(checkpoint_dir), str(backup_dir))
        reason = "DINO_FORCE_RESTART=1" if requested_restart else "training.resume=false"
        return {
            "action": "fresh_start_backup",
            "reason": reason,
            "checkpoint_dir": str(checkpoint_dir),
            "backup_dir": str(backup_dir),
        }

    if latest.is_file() and resume_enabled:
        return {
            "action": "resume",
            "checkpoint_path": str(latest),
        }

    return {
        "action": "fresh_start",
        "checkpoint_path": str(latest),
    }


def latest_epoch_checkpoint(config: dict[str, Any]) -> int:
    """Highest model_<N>.pth epoch saved for this run (0 when none exist)."""

    import re

    checkpoint_dir = checkpoint_output_dir(config) / "checkpoints"
    epochs = [0]
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("model_*.pth"):
            match = re.fullmatch(r"model_(\d+)\.pth", path.name)
            if match:
                epochs.append(int(match.group(1)))
    return max(epochs)


def training_complete(config: dict[str, Any]) -> bool:
    """True when the run's checkpoint tree already holds its final epoch.

    Used by the notebook launch cells to skip completed runs entirely while
    still launching (and therefore resuming) partial ones.
    """

    if get_config_value(config, "finetuning.enabled", False):
        target = int(get_config_value(config, "finetuning.epochs", 0) or 0)
    else:
        target = int(get_config_value(config, "training.epochs", 0) or 0)
    return target > 0 and latest_epoch_checkpoint(config) >= target


def planning_complete(config: dict[str, Any], planner: str | None = None) -> bool:
    """True when this run dir already holds a completed planning evaluation
    matching the config's current n_evals (so raising n_evals re-triggers)."""

    import json

    planner = planner or os.environ.get(
        "DINO_PLANNER", str(get_config_value(config, "planning.planner", "cem"))
    )
    log_root = Path(str(get_config_value(config, "artifacts.log_root"))).expanduser()
    run_dir = log_root / str(config.get("run_name"))
    command_path = run_dir / "planning" / f"command_{planner}.sh"
    status_path = run_dir / "planning" / f"status_{planner}.json"
    if not command_path.is_file() or not status_path.is_file():
        return False
    n_evals = int(get_config_value(config, "planning.n_evals", 0) or 0)
    if f"n_evals={n_evals}" not in command_path.read_text(encoding="utf-8"):
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(status.get("completed"))
