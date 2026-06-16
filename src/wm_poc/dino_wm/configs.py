from __future__ import annotations

import os
import re
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


REQUIRED_SECTIONS = ("dataset", "model", "training", "planning", "artifacts")
OC_ENV_PATTERN = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")


def _require_yaml() -> None:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required for DINO-WM config handling.") from _YAML_IMPORT_ERROR


def _read_yaml(path: Path) -> dict[str, Any]:
    _require_yaml()
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a DINO-WM wrapper config.

    Configs may optionally declare ``extends: base.yaml``. The inheritance is kept
    deliberately small so experiment files can override only the fields they need.
    """

    config_path = Path(path).expanduser()
    data = _read_yaml(config_path)
    extends = data.pop("extends", None)
    if extends is None:
        return data
    base_path = Path(extends).expanduser()
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base = load_config(base_path)
    return _deep_merge(base, data)


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() in {"none", "null"}:
        return None
    _require_yaml()
    try:
        parsed = yaml.safe_load(stripped)
    except Exception:
        return value
    if isinstance(parsed, (dict, list)):
        return value
    return parsed


def _resolve_string(value: str) -> Any:
    full = OC_ENV_PATTERN.fullmatch(value)
    if full:
        env_name, default = full.groups()
        return _parse_scalar(os.environ.get(env_name, default if default is not None else ""))

    def replace(match: re.Match[str]) -> str:
        env_name, default = match.groups()
        replacement = os.environ.get(env_name, default if default is not None else "")
        return "" if replacement is None else replacement

    return os.path.expandvars(OC_ENV_PATTERN.sub(replace, value))


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve supported environment placeholders without mutating ``config``."""

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str):
            return _resolve_string(value)
        return deepcopy(value)

    return resolve(config)


def get_config_value(config: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def set_config_value(config: dict[str, Any], path: str, value: Any) -> None:
    cursor: dict[str, Any] = config
    parts = path.split(".")
    for key in parts[:-1]:
        next_value = cursor.setdefault(key, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set {path}: {key} is not a mapping.")
        cursor = next_value
    cursor[parts[-1]] = value


def _validate_section(config: dict[str, Any], section: str) -> None:
    if section not in config:
        raise ValueError(f"Missing required DINO-WM config section: {section}")
    if not isinstance(config[section], dict):
        raise ValueError(f"Config section {section} must be a mapping.")


def _validate_max_minutes(value: Any, *, limit: int, label: str) -> None:
    try:
        minutes = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if minutes <= 0:
        raise ValueError(f"{label} must be positive.")
    if minutes > limit:
        raise ValueError(f"{label} must be <= {limit}; got {minutes:g}.")


def validate_config(config: dict[str, Any]) -> None:
    for section in REQUIRED_SECTIONS:
        _validate_section(config, section)

    track = config.get("track")
    if track not in {None, "dino_wm"}:
        raise ValueError(f"Expected track=dino_wm, got {track!r}.")

    # Sanity ceiling for the run_train.sh timeout wrapper, not a Colab limit.
    # 600 min gives full stride-1 T4 schedules and notebook reruns headroom;
    # DINO_MAX_WALL_MINUTES can still override the config value at launch time.
    _validate_max_minutes(
        get_config_value(config, "training.max_wall_minutes", 220),
        limit=600,
        label="training.max_wall_minutes",
    )

    planning_enabled = bool(get_config_value(config, "planning.enabled", True))
    if planning_enabled:
        _validate_max_minutes(
            get_config_value(config, "planning.max_wall_minutes", 60),
            limit=60,
            label="planning.max_wall_minutes",
        )

    if get_config_value(config, "features.freeze_encoder", True) is not True:
        raise ValueError("features.freeze_encoder must remain true for main DINO-WM experiments.")
    if get_config_value(config, "finetuning.freeze.visual_encoder", True) is not True:
        raise ValueError("finetuning.freeze.visual_encoder must remain true.")

    env_name = get_config_value(config, "dataset.env")
    if not env_name:
        raise ValueError("dataset.env is required.")

    checksum_mode = get_config_value(config, "dataset.checksum_mode", "metadata")
    if checksum_mode not in {"metadata", "auto", "full"}:
        raise ValueError("dataset.checksum_mode must be one of: metadata, auto, full.")

    train_fraction = float(get_config_value(config, "dataset.train_fraction", 0.9))
    if train_fraction <= 0 or train_fraction >= 1:
        raise ValueError("dataset.train_fraction must be in (0, 1).")

    resume = get_config_value(config, "training.resume", True)
    if not isinstance(resume, bool):
        raise ValueError("training.resume must be a boolean.")

    save_every = int(get_config_value(config, "training.save_every_epochs", 1))
    if save_every <= 0:
        raise ValueError("training.save_every_epochs must be positive.")

    save_every_steps = int(get_config_value(config, "training.save_every_steps", 0))
    if save_every_steps < 0:
        raise ValueError("training.save_every_steps must be >= 0.")

    num_workers = int(get_config_value(config, "training.num_workers", 4))
    if num_workers < 0:
        raise ValueError("training.num_workers must be >= 0.")

    slice_stride = int(get_config_value(config, "dataset.slice_stride", 1))
    if slice_stride < 1:
        raise ValueError("dataset.slice_stride must be >= 1.")

    precompute_batch_size = int(get_config_value(config, "features.precompute_batch_size", 128))
    if precompute_batch_size < 1:
        raise ValueError("features.precompute_batch_size must be >= 1.")

    if get_config_value(config, "execution.require_env_flag", "RUN_DINO_WM") != "RUN_DINO_WM":
        raise ValueError("execution.require_env_flag must be RUN_DINO_WM.")


def make_run_dir(config: dict[str, Any]) -> Path:
    run_name = config.get("run_name")
    if run_name in {None, ""}:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"dino_wm_{stamp}"
        config["run_name"] = run_name

    log_root = Path(str(get_config_value(config, "artifacts.log_root"))).expanduser()
    run_dir = log_root / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("ckpts", "planning", "figures", "videos"):
        (run_dir / subdir).mkdir(exist_ok=True)
    set_config_value(config, "artifacts.run_dir", str(run_dir))
    return run_dir


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    _require_yaml()
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    tmp.replace(path)


def copy_config_artifacts(config_path: Path, run_dir: Path, resolved: dict[str, Any]) -> None:
    run_dir = run_dir.expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path.expanduser(), run_dir / "config.yaml")
    write_yaml(run_dir / "resolved_config.yaml", resolved)
