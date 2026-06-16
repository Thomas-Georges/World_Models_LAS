"""Runtime-environment setup for the in-process DINO-WM global model.

Loading the DINO-WM world model imports the upstream ``plan.py``, whose
module-level imports pull in ``mujoco_py``. ``mujoco_py`` refuses to load
unless ``LD_LIBRARY_PATH`` contains ``<mujoco210>/bin`` (and the upstream
checkpoint needs ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` to unpickle whole
modules). The DINO-WM track configures this in
``scripts/dino_wm/mujoco_runtime_env.sh`` before launching ``plan.py``; the
local/global planner loads the model in-process, so it must do the same.

Because the dynamic linker reads ``LD_LIBRARY_PATH`` at process start,
:func:`setup_mujoco_runtime` re-execs the interpreter once after updating it.
This module is intentionally torch-free so it can run before the heavy imports
(and be unit-tested without torch).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence

REEXEC_FLAG = "_WM_POC_MUJOCO_REEXEC"


def setup_mujoco_runtime(
    *,
    environ: dict[str, str] | None = None,
    isdir: Callable[[str], bool] | None = None,
    execv: Callable[[str, Sequence[str]], None] | None = None,
    argv: Sequence[str] | None = None,
    executable: str | None = None,
    log: Callable[[str], None] = print,
) -> bool:
    """Set mujoco_py's runtime env and re-exec once if ``LD_LIBRARY_PATH`` changed.

    Mirrors ``scripts/dino_wm/mujoco_runtime_env.sh`` for the in-process model
    load. Returns ``True`` when a re-exec was triggered. It is a no-op
    (returns ``False``) when the mujoco directories are absent (e.g. local CPU
    boxes), when the paths are already present, or when a previous call already
    re-exec'd (guarded by ``REEXEC_FLAG``), so it never loops. The mujoco/torch
    env vars are applied (via ``setdefault``) on every call regardless.
    """
    environ = os.environ if environ is None else environ
    isdir = os.path.isdir if isdir is None else isdir
    execv = os.execv if execv is None else execv
    argv = sys.argv if argv is None else argv
    executable = sys.executable if executable is None else executable

    home = environ.get("HOME", "/root")
    mujoco_dir = (
        environ.get("DINO_MUJOCO210_DIR")
        or environ.get("MUJOCO210_DIR")
        or f"{home}/.mujoco/mujoco210"
    )
    # Set unconditionally: MUJOCO_PY_MUJOCO_PATH locates the install, and the
    # TORCH flag lets torch.load unpickle the upstream whole-module checkpoint.
    environ.setdefault("MUJOCO_PY_MUJOCO_PATH", mujoco_dir)
    environ.setdefault("MUJOCO_GL", "egl")
    environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    candidates = [os.path.join(mujoco_dir, "bin"), "/usr/lib/nvidia", "/usr/lib64-nvidia"]
    current = environ.get("LD_LIBRARY_PATH", "")
    parts = current.split(":") if current else []
    additions = [p for p in candidates if isdir(p) and p not in parts]
    if not additions or environ.get(REEXEC_FLAG) == "1":
        return False

    environ["LD_LIBRARY_PATH"] = ":".join(additions + parts)
    environ[REEXEC_FLAG] = "1"
    log(
        "[mujoco] LD_LIBRARY_PATH += "
        + ":".join(additions)
        + "; re-exec so the dynamic linker can load mujoco_py"
    )
    execv(executable, [executable, *argv])
    return True
