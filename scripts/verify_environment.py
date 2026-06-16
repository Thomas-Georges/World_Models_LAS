#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def print_torch_info(require_torch: bool, cpu_only: bool) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        if require_torch:
            print("ERROR: --require-torch was passed but PyTorch is not installed.")
            return 1
        return 0

    print(f"PyTorch: {torch.__version__}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif not cpu_only:
        print("No GPU detected. This is acceptable for bootstrap checks.")
    return 0


def print_nvidia_smi() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("nvidia-smi: not found")
        return

    result = subprocess.run(
        [nvidia_smi],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print("nvidia-smi output:")
    print(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print lightweight environment diagnostics.")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Do not require GPU availability.",
    )
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help="Fail if PyTorch is not importable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {Path.cwd()}")
    status = print_torch_info(require_torch=args.require_torch, cpu_only=args.cpu_only)
    print_nvidia_smi()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
