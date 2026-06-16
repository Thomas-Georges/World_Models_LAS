from __future__ import annotations

import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


MUJOCO210_URL = "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
REQUIRED_MUJOCO210_LIBS = (
    "libglfw.so.3",
    "libglew.so",
    "libglewegl.so",
    "libglewosmesa.so",
    "libmujoco210.so",
)


def mujoco210_dir() -> Path:
    configured = os.environ.get("DINO_MUJOCO210_DIR") or os.environ.get("MUJOCO210_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".mujoco" / "mujoco210"


def configure_mujoco_runtime_env() -> Path:
    mujoco_dir = mujoco210_dir()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", str(mujoco_dir))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    _prepend_path("LD_LIBRARY_PATH", mujoco_dir / "bin")
    nvidia_lib = Path("/usr/lib/nvidia")
    if nvidia_lib.is_dir():
        _prepend_path("LD_LIBRARY_PATH", nvidia_lib)
    return mujoco_dir


def ensure_mujoco210(*, dry_run: bool = False) -> Path:
    target = mujoco210_dir()
    if has_mujoco210_runtime(target):
        return target
    if platform.system() != "Linux":
        print(f"MuJoCo 2.1 binary setup skipped on {platform.system()}: {target}")
        return target
    url = os.environ.get("DINO_MUJOCO210_URL", MUJOCO210_URL)
    if dry_run:
        print(f"Would download MuJoCo 2.1 from {url} to {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wm_poc_mujoco210_") as tmpdir:
        tmp_path = Path(tmpdir)
        archive = tmp_path / "mujoco210-linux-x86_64.tar.gz"
        print(f"Downloading MuJoCo 2.1 to {archive}")
        urllib.request.urlretrieve(url, archive)
        extract_root = tmp_path / "extract"
        extract_root.mkdir()
        _safe_extract_tar(archive, extract_root)
        extracted = extract_root / "mujoco210"
        if not (extracted / "bin").is_dir():
            raise RuntimeError(f"MuJoCo archive did not contain expected mujoco210/bin: {url}")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))
    return target


def has_mujoco210_runtime(path: Path | None = None) -> bool:
    target = mujoco210_dir() if path is None else path
    return all((target / "bin" / name).is_file() for name in REQUIRED_MUJOCO210_LIBS)


def _prepend_path(name: str, path: Path) -> None:
    value = str(path)
    existing = os.environ.get(name, "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if value not in parts:
        os.environ[name] = os.pathsep.join([value, *parts])


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
        tar.extractall(destination)
