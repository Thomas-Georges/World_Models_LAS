#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.configs import load_config, resolve_config, validate_config  # noqa: E402
from wm_poc.dino_wm.mujoco import (  # noqa: E402
    configure_mujoco_runtime_env,
    has_mujoco210_runtime,
    mujoco210_dir,
)


REQUIRED_UPSTREAM_IMPORTS = {
    "accelerate": "accelerate",
    "Cython": "Cython",
    "d4rl": "d4rl",
    "decord": "decord",
    "dm_control": "dm_control",
    "einops": "einops",
    "gym": "gym",
    "hydra": "hydra",
    "mujoco": "mujoco",
    "mujoco_py": "mujoco_py",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "submitit": "submitit",
    "wandb": "wandb",
}

MUJOCO_PY_COMPILE_IMPORTS = {"d4rl", "mujoco_py"}


def _torch_status(require_cuda: bool) -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not importable.")
        return 1
    print(f"PyTorch: {torch.__version__}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    elif require_cuda:
        print("ERROR: CUDA is required for DINO-WM training checks.")
        return 1
    return 0


def _nvidia_smi() -> None:
    binary = shutil.which("nvidia-smi")
    if not binary:
        print("nvidia-smi: not found")
        return
    result = subprocess.run(
        [binary, "--query-gpu=name,memory.total", "--format=csv,noheader"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout.strip())


def _dependency_status() -> int:
    configure_mujoco_runtime_env()
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")
    warnings.filterwarnings("ignore", category=Warning, module=r"wandb\.analytics\.sentry")
    warnings.filterwarnings("ignore", message=r".*sentry_sdk\.Hub is deprecated.*")
    failures: list[tuple[str, str]] = []
    osmesa_missing = not _has_osmesa_header()
    mujoco210_missing = not has_mujoco210_runtime()
    if osmesa_missing:
        failures.append(
            (
                "system:osmesa",
                "missing GL/osmesa.h; run python scripts/dino_wm/install_colab_deps.py",
            )
        )
    if mujoco210_missing:
        failures.append(
            (
                "system:mujoco210",
                f"missing MuJoCo 2.1 runtime libraries under {mujoco210_dir()}",
            )
        )
    for display_name, module_name in REQUIRED_UPSTREAM_IMPORTS.items():
        if (osmesa_missing or mujoco210_missing) and module_name in MUJOCO_PY_COMPILE_IMPORTS:
            continue
        if module_name in MUJOCO_PY_COMPILE_IMPORTS:
            reason = _subprocess_import_failure(module_name)
            if reason:
                failures.append((display_name, reason))
        else:
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                failures.append((display_name, f"{type(exc).__name__}: {exc}"))
    for package_name, max_major in (("numpy", 1), ("Cython", 0)):
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if _major_version(version) > max_major:
            failures.append(
                (
                    package_name,
                    f"version {version}; DINO-WM PointMaze uses legacy Gym/D4RL/mujoco_py and needs "
                    f"{package_name} major version <= {max_major}",
                )
            )
    if failures:
        print("ERROR: missing or broken DINO-WM Python packages:")
        for name, reason in failures:
            print(f"  {name}: {reason}")
        print("Run: python scripts/dino_wm/install_colab_deps.py")
        return 1
    print("DINO-WM Python package imports OK.")
    return 0


def _subprocess_import_failure(module_name: str) -> str | None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode == 0:
        return None
    output = result.stdout.strip().splitlines()
    return output[-1] if output else f"import failed with exit code {result.returncode}"


def _has_osmesa_header() -> bool:
    if platform.system() != "Linux":
        return True
    return any(
        Path(root, "GL", "osmesa.h").is_file()
        for root in ("/usr/include", "/usr/local/include")
    )


def _major_version(version: str) -> int:
    head = version.split(".", maxsplit=1)[0]
    try:
        return int(head)
    except ValueError:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify DINO-WM environment.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/dino_wm/base.yaml")
    parser.add_argument("--allow-cpu", action="store_true", help="Do not fail when CUDA is absent.")
    parser.add_argument(
        "--allow-missing-upstream",
        action="store_true",
        help="Do not fail when the upstream DINO-WM repo is absent.",
    )
    parser.add_argument("--skip-dependency-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    validate_config(config)
    print(f"Python: {sys.version}")
    print(f"Wrapper config OK: {args.config}")
    status = _torch_status(require_cuda=not args.allow_cpu)
    _nvidia_smi()
    if not args.skip_dependency_check:
        status = max(status, _dependency_status())

    upstream = Path(str(config.get("external_repo", "external_repos/dino_wm"))).expanduser()
    if not upstream.is_absolute():
        upstream = REPO_ROOT / upstream
    print(f"DINO_WM_REPO: {upstream}")
    missing = [path for path in (upstream / "train.py", upstream / "plan.py") if not path.is_file()]
    if missing:
        print("ERROR: missing upstream files:")
        for path in missing:
            print(f"  {path}")
        if not args.allow_missing_upstream:
            status = 1
    else:
        print("Upstream train.py and plan.py found.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
