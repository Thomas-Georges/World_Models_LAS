#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def collect(r2_repo: Path | None) -> dict[str, object]:
    info: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "project_git_commit": git_commit(Path.cwd()),
        "r2dreamer_git_commit": git_commit(r2_repo) if r2_repo else "",
        "nvidia_smi": "",
        "torch": "",
        "cuda_available": False,
        "cuda_version": "",
        "gpu_name": "",
    }
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        info["nvidia_smi"] = result.stdout
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda or ""
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect small GPU/runtime metadata.")
    parser.add_argument("--r2-repo", type=Path, help="External r2dreamer checkout.")
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info = collect(args.r2_repo)
    text = json.dumps(info, indent=2)
    print(text)
    if args.out:
        args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out.expanduser().write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
