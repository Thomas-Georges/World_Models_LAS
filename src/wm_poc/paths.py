from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: str | None = None) -> Path:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Environment variable {name} is not set.")
    return Path(value).expanduser().resolve()


def repo_root() -> Path:
    return env_path("WM_POC_REPO", default=".")


def drive_root() -> Path:
    return env_path("WM_POC_DRIVE_ROOT", default="/content/drive/MyDrive/wm_poc")


def data_dir() -> Path:
    return env_path("WM_POC_DATA_DIR", default=str(drive_root() / "data"))


def log_dir() -> Path:
    return env_path("WM_POC_LOG_DIR", default=str(drive_root() / "logs"))


def checkpoint_dir() -> Path:
    return env_path("WM_POC_CKPT_DIR", default=str(drive_root() / "checkpoints"))


def figure_dir() -> Path:
    return env_path(
        "WM_POC_FIGURE_DIR",
        default=os.environ.get("WM_POC_FIG_DIR", str(drive_root() / "figures")),
    )
