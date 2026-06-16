from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wm_poc.dino_wm.configs import get_config_value, write_yaml


SUPPORTED_ENVS = {"point_maze", "wall_single", "pusht_noise"}
TRAJECTORY_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle", ".h5", ".hdf5", ".json"}
CHECKSUM_MODES = {"metadata", "auto", "full"}


def _env_dir(root: Path, env: str) -> Path:
    candidates = [root / env, root / "data" / env, root / "datasets" / env]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def validate_dataset_root(root: str | Path, env: str) -> None:
    if env not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported DINO-WM environment {env!r}. Supported: {sorted(SUPPORTED_ENVS)}")
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise FileNotFoundError(f"DINO-WM dataset root does not exist: {root_path}")
    env_path = _env_dir(root_path, env)
    if not env_path.is_dir():
        raise FileNotFoundError(
            f"Missing dataset folder for env={env}: expected one of "
            f"{root_path / env}, {root_path / 'data' / env}, or {root_path / 'datasets' / env}."
        )


def _iter_trajectory_files(env_path: Path) -> list[Path]:
    files = [
        path
        for path in env_path.rglob("*")
        if path.is_file() and path.suffix.lower() in TRAJECTORY_SUFFIXES
    ]
    return sorted(files)


def _checksum(path: Path, *, mode: str) -> str:
    if mode not in CHECKSUM_MODES:
        raise ValueError(f"Unsupported checksum_mode={mode!r}; expected one of {sorted(CHECKSUM_MODES)}.")
    stat = path.stat()
    if mode == "metadata":
        token = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
        return "metadata-token-" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    if stat.st_size > 64 * 1024 * 1024:
        token = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
        return "large-file-token-" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, base: Path, *, checksum_mode: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "relative_path": str(path.relative_to(base)),
        "num_frames": None,
        "checksum": _checksum(path, mode=checksum_mode),
    }
    try:
        record["size_bytes"] = path.stat().st_size
    except OSError:
        record["size_bytes"] = None
    return record


def build_split_manifest(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(get_config_value(config, "dataset.root"))).expanduser()
    env = str(get_config_value(config, "dataset.env"))
    env_path = _env_dir(root, env)
    files = _iter_trajectory_files(env_path) if env_path.is_dir() else []

    seed = int(get_config_value(config, "dataset.split_seed", 0))
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)

    train_fraction = float(get_config_value(config, "dataset.train_fraction", 0.9))
    split_index = int(len(shuffled) * train_fraction)
    train_files = shuffled[:split_index]
    val_files = shuffled[split_index:]

    max_train = int(get_config_value(config, "dataset.max_train_trajectories", len(train_files)))
    max_val = int(get_config_value(config, "dataset.max_val_trajectories", len(val_files)))
    train_files = train_files[:max_train]
    val_files = val_files[:max_val]
    selected = train_files + val_files
    checksum_mode = str(get_config_value(config, "dataset.checksum_mode", "metadata"))

    return {
        "encoder": get_config_value(config, "features.encoder", "dinov2_patch"),
        "encoder_checkpoint": get_config_value(config, "features.encoder_checkpoint"),
        "image_size": int(get_config_value(config, "features.image_size", 224)),
        "env": env,
        "split_seed": seed,
        "train_fraction": train_fraction,
        "max_train_trajectories": max_train,
        "max_val_trajectories": max_val,
        "checksum_mode": checksum_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_root": str(root),
        "dataset_env_dir": str(env_path),
        "cache_dir": str(get_config_value(config, "features.cache_dir")),
        "num_available_files": len(files),
        "num_train_files": len(train_files),
        "num_val_files": len(val_files),
        "train_files": [str(path.relative_to(env_path)) for path in train_files],
        "val_files": [str(path.relative_to(env_path)) for path in val_files],
        "files": [_file_record(path, env_path, checksum_mode=checksum_mode) for path in selected],
    }


def write_split_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    write_yaml(Path(path).expanduser(), manifest)


def validate_feature_cache(cache_dir: str | Path, manifest: dict[str, Any]) -> None:
    cache_path = Path(cache_dir).expanduser()
    if not cache_path.is_dir():
        raise FileNotFoundError(f"Feature cache directory does not exist: {cache_path}")
    missing: list[str] = []
    for item in manifest.get("files", []):
        rel = item.get("relative_path")
        if rel and not (cache_path / rel).exists():
            missing.append(str(rel))
    if missing and not manifest.get("allow_incomplete_cache", False):
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Feature cache is missing {len(missing)} files; first missing: {preview}")
