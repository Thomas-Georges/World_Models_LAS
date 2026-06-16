"""Startup shim that delegates to ``wm_poc.dino_wm.import_bootstrap``.

This file exists only as a convenience for the subprocess launch path: the
DINO-WM runtime prepends this directory to ``PYTHONPATH`` (see
``scripts/dino_wm/mujoco_runtime_env.sh``) so the interpreter runs it at startup
and upstream ``datasets``/``models``/``planning`` are pinned before any import.

Correctness does **not** depend on this shim running. A ``sitecustomize`` is a
process-wide singleton and may be shadowed by another one earlier on
``sys.path``; the same runtime therefore also prepends the upstream repo to
``PYTHONPATH`` (so path ordering already favours it), and launchers may call
``enable_dino_wm_imports`` explicitly. When this shim *does* run it simply reuses
that one code path. On failure it logs a single line to stderr rather than
silently swallowing the error (a silent ``except: pass`` previously hid real
bootstrap failures).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _main() -> None:
    repo_value = os.environ.get("DINO_WM_REPO")
    if not repo_value:
        return

    # Make ``wm_poc`` importable from this checkout if it is not already, so the
    # shim works before an editable install is on the path.
    repo_root = Path(__file__).resolve().parents[3]  # python_startup/dino_wm/scripts/<root>
    src_dir = repo_root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from wm_poc.dino_wm.import_bootstrap import enable_dino_wm_imports

    enable_dino_wm_imports(repo_value)


try:
    _main()
except Exception as exc:  # pragma: no cover - a startup hook must not hard-fail
    print(f"[dino_wm sitecustomize] DINO-WM import bootstrap skipped: {exc!r}", file=sys.stderr)
