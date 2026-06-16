from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def current_git_commit(cwd: Path | None = None) -> str:
    cwd = cwd or Path.cwd()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": "",
        "cuda_available": False,
        "cuda_version": "",
        "gpu_name": "",
    }
    try:
        import torch
    except ImportError:
        return info

    info["torch_version"] = torch.__version__
    cuda_available = bool(torch.cuda.is_available())
    info["cuda_available"] = cuda_available
    info["cuda_version"] = torch.version.cuda or ""
    if cuda_available:
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def collect_system_info(cwd: Path | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "git_commit": current_git_commit(cwd=cwd),
    }
    metadata.update(torch_info())
    return metadata
