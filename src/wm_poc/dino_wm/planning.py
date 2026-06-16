from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wm_poc.dino_wm.configs import get_config_value


VALID_PLANNERS = {"cem", "gd"}


def validate_planning_config(config: dict[str, Any]) -> None:
    if not get_config_value(config, "planning.enabled", True):
        return
    planner = str(get_config_value(config, "planning.planner", "cem"))
    if planner not in VALID_PLANNERS:
        raise ValueError(f"Unsupported planner {planner!r}. Supported: {sorted(VALID_PLANNERS)}")
    for path in ("planning.n_evals", "planning.opt_steps", "planning.samples", "planning.goal_H"):
        value = int(get_config_value(config, path, 0))
        if value <= 0:
            raise ValueError(f"{path} must be positive.")
    max_wall = float(get_config_value(config, "planning.max_wall_minutes", 60))
    if max_wall > 60:
        raise ValueError("planning.max_wall_minutes must be <= 60.")


def build_planner_overrides(config: dict[str, Any]) -> list[str]:
    planner = str(get_config_value(config, "planning.planner", "cem"))
    overrides = [
        f"planner={get_config_value(config, 'planning.planner', 'cem')}",
        f"n_evals={get_config_value(config, 'planning.n_evals')}",
        f"goal_H={get_config_value(config, 'planning.goal_H')}",
        f"goal_source={get_config_value(config, 'planning.goal_source')}",
        f"planner.opt_steps={get_config_value(config, 'planning.opt_steps')}",
    ]
    if planner in {"cem", "mpc_cem"}:
        overrides.append(f"planner.num_samples={get_config_value(config, 'planning.samples')}")
    return overrides


def _normalize_episode(record: dict[str, Any], planner: str) -> dict[str, Any]:
    success = record.get("success")
    if success is None and "is_success" in record:
        success = record["is_success"]
    normalized = {
        "stage": "planning",
        "episode": record.get("episode", record.get("episode_idx")),
        "success": bool(success) if success is not None else None,
        "success_rate": record.get("success_rate"),
        "final_goal_latent_distance": record.get(
            "final_goal_latent_distance",
            record.get("goal_latent_distance", record.get("final_goal_distance")),
        ),
        "final_env_distance": record.get("final_env_distance", record.get("env_distance")),
        "planner": record.get("planner", planner),
        "cem_iterations": record.get("cem_iterations", record.get("opt_steps")),
        "candidate_count": record.get("candidate_count", record.get("samples")),
        "plan_time_seconds": record.get("plan_time_seconds", record.get("elapsed_seconds")),
        "failure_reason": record.get("failure_reason"),
    }
    return normalized


def parse_planning_outputs(path: str | Path, planner: str = "cem") -> list[dict[str, Any]]:
    root = Path(path).expanduser()
    files: list[Path]
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.glob("*.jsonl")) + sorted(root.glob("*.json"))

    records: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            if file_path.suffix == ".jsonl":
                rows = [json.loads(line) for line in f if line.strip()]
            else:
                payload = json.load(f)
                rows = payload if isinstance(payload, list) else payload.get("episodes", [])
        for row in rows:
            if isinstance(row, dict):
                records.append(_normalize_episode(row, planner))
    return records
