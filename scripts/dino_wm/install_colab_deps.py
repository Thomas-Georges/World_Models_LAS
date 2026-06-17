#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import shlex
import shutil
import subprocess
import sys
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.mujoco import (  # noqa: E402
    configure_mujoco_runtime_env,
    ensure_mujoco210,
    has_mujoco210_runtime,
    mujoco210_dir,
)


REQUIRED_IMPORTS = {
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
    "psutil": "psutil",
    "submitit": "submitit",
    "wandb": "wandb",
}

PIP_PACKAGES = [
    "numpy==1.26.4",
    "Cython==0.29.37",
    "hydra-core==1.3.2",
    "hydra-submitit-launcher==1.2.0",
    "omegaconf==2.3.0",
    "accelerate==0.26.1",
    "wandb==0.17.9",
    "decord==0.6.0",
    "einops==0.4.1",
    "gym==0.23.1",
    "dm-control==1.0.27",
    "mujoco==3.2.7",
    "mujoco-py==2.1.2.14",
    "d4rl==1.1",
    "pybullet==3.2.7",
    "submitit==1.5.1",
    "psutil>=5.9.0",
]

APT_PACKAGES = [
    "build-essential",
    "libosmesa6-dev",
    "libgl1-mesa-dev",
    "libglfw3",
    "libglew-dev",
    "patchelf",
]

MUJOCO_PY_COMPILE_IMPORTS = {"d4rl", "mujoco_py"}


def import_failures() -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    osmesa_missing = not has_osmesa_header()
    mujoco210_missing = not has_mujoco210_runtime()
    if osmesa_missing:
        failures.append(
            (
                "system:osmesa",
                "missing GL/osmesa.h; install apt package libosmesa6-dev before importing mujoco_py",
            )
        )
    if mujoco210_missing:
        failures.append(
            (
                "system:mujoco210",
                f"missing MuJoCo 2.1 runtime libraries under {mujoco210_dir()}",
            )
        )
    for display_name, module_name in REQUIRED_IMPORTS.items():
        if (osmesa_missing or mujoco210_missing) and module_name in MUJOCO_PY_COMPILE_IMPORTS:
            continue
        if module_name in MUJOCO_PY_COMPILE_IMPORTS:
            reason = subprocess_import_failure(module_name)
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
    return failures


def _major_version(version: str) -> int:
    head = version.split(".", maxsplit=1)[0]
    try:
        return int(head)
    except ValueError:
        return 0


def subprocess_import_failure(module_name: str) -> str | None:
    env = os.environ.copy()
    command = [sys.executable, "-c", f"import {module_name}"]
    result = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode == 0:
        return None
    output = result.stdout.strip().splitlines()
    detail = output[-1] if output else f"import failed with exit code {result.returncode}"
    return detail


def has_osmesa_header() -> bool:
    if platform.system() != "Linux":
        return True
    return any(
        Path(root, "GL", "osmesa.h").is_file()
        for root in ("/usr/include", "/usr/local/include")
    )


def apt_get_binary() -> str | None:
    return shutil.which("apt-get")


def apt_command(apt_get: str) -> list[str]:
    if hasattr(os, "geteuid") and os.geteuid() != 0 and shutil.which("sudo"):
        return ["sudo", apt_get]
    return [apt_get]


def ensure_system_packages(*, dry_run: bool = False, quiet: bool = False) -> None:
    if platform.system() != "Linux":
        return
    apt_get = apt_get_binary()
    if not apt_get:
        print("apt-get not found; cannot install MuJoCo system packages automatically.")
        return
    if has_osmesa_header():
        return

    command_prefix = apt_command(apt_get)
    update_command = [*command_prefix]
    install_command = [*command_prefix]
    if quiet:
        update_command.append("-qq")
        install_command.append("-qq")
    update_command.append("update")
    install_command.extend(["install", "-y", *APT_PACKAGES])
    print("$", shlex.join(update_command))
    print("$", shlex.join(install_command))
    if dry_run:
        return
    subprocess.run(update_command, check=True)
    subprocess.run(install_command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install minimal DINO-WM runtime dependencies for Colab.")
    parser.add_argument("--check-only", action="store_true", help="Only report missing imports.")
    parser.add_argument("--dry-run", action="store_true", help="Print the pip command without executing it.")
    parser.add_argument("--quiet", action="store_true", help="Pass -q to pip.")
    return parser.parse_args()


def configure_warning_filters() -> None:
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")
    warnings.filterwarnings("ignore", category=Warning, module=r"wandb\.analytics\.sentry")
    warnings.filterwarnings("ignore", message=r".*sentry_sdk\.Hub is deprecated.*")


def _torchvision_spec_for_torch(torch_version: str) -> str | None:
    """torchvision pin matching a torch version: minor tracks torch (2.N -> 0.(N+15)).

    e.g. torch 2.0 -> torchvision 0.15.0, 2.6 -> 0.21.0, 2.7 -> 0.22.0. Returns
    None for versions outside the known torch 2.x mapping.
    """
    base = torch_version.split("+", 1)[0]
    parts = base.split(".")
    if len(parts) < 2:
        return None
    try:
        major, minor = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if major != 2:
        return None
    return f"torchvision==0.{minor + 15}.0"


def matching_torchvision_spec() -> str | None:
    """Pin for the torchvision release matching the installed torch.

    Set `DINO_TORCHVISION_SPEC` to override the derived pin (e.g. for an unusual
    torch build, `DINO_TORCHVISION_SPEC=torchvision==0.21.0+cu124`).
    """
    override = os.environ.get("DINO_TORCHVISION_SPEC")
    if override:
        return override
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    return _torchvision_spec_for_torch(torch_version)


def ensure_torchvision_matches_torch(
    *, check_only: bool = False, dry_run: bool = False, quiet: bool = False
) -> None:
    """Repair a torch/torchvision mismatch left by the dependency install.

    The legacy pins above can shift Colab's preinstalled torch, leaving its
    torchvision compiled against a different build (`RuntimeError: operator
    torchvision::nms does not exist`), which breaks the DINO encoder import in the
    latent precompute/training. Reinstall the torchvision that matches the
    installed torch, without touching torch itself.
    """
    reason = subprocess_import_failure("torchvision")
    if reason is None:
        return
    spec = matching_torchvision_spec()
    if spec is None:
        print(
            f"WARNING: torchvision fails to import ({reason}) and no matching version "
            "could be derived from the installed torch. Set DINO_TORCHVISION_SPEC to a "
            "torchvision matching your torch and rerun.",
            file=sys.stderr,
        )
        return
    print(f"torch/torchvision mismatch detected ({reason}).")
    command = [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall"]
    if quiet:
        command.append("-q")
    command.append(spec)
    print("Repairing torch/torchvision pair:", shlex.join(command))
    if check_only or dry_run:
        return
    subprocess.run(command, check=True)
    after = subprocess_import_failure("torchvision")
    if after is None:
        print(f"torchvision repaired ({spec}).")
    else:
        print(f"ERROR: torchvision still fails to import after reinstall: {after}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    configure_warning_filters()
    configure_mujoco_runtime_env()
    if not args.check_only:
        ensure_system_packages(dry_run=args.dry_run, quiet=args.quiet)
        ensure_mujoco210(dry_run=args.dry_run)
        configure_mujoco_runtime_env()
    failures = import_failures()
    if not failures:
        print("DINO-WM Python dependencies found.")
        ensure_torchvision_matches_torch(
            check_only=args.check_only, dry_run=args.dry_run, quiet=args.quiet
        )
        return 0

    print("Missing or broken DINO-WM Python dependencies:")
    for name, reason in failures:
        print(f"  {name}: {reason}")
    command = [sys.executable, "-m", "pip", "install"]
    if args.quiet:
        command.append("-q")
    command.extend(PIP_PACKAGES)
    print("$", shlex.join(command))

    if args.check_only:
        ensure_torchvision_matches_torch(check_only=True)
        return 1
    if args.dry_run:
        ensure_torchvision_matches_torch(dry_run=True)
        return 0

    subprocess.run(command, check=True)
    still_failing = import_failures()
    if still_failing:
        print("ERROR: imports still missing or broken after install:", file=sys.stderr)
        for name, reason in still_failing:
            print(f"  {name}: {reason}", file=sys.stderr)
        return 1
    # The legacy install above can desync Colab's torch/torchvision pair.
    ensure_torchvision_matches_torch(quiet=args.quiet)
    print("DINO-WM Python dependencies installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
