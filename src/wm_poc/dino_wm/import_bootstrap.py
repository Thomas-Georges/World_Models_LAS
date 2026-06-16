"""Explicit, testable bootstrap for upstream DINO-WM imports.

Upstream DINO-WM uses plain top-level package names (``datasets``, ``models``,
``planning``) that collide with widely-installed packages -- most importantly
Hugging Face ``datasets``. This module puts a DINO-WM checkout first on
``sys.path`` and pins those top-level packages to the checkout, so
``import datasets`` resolves to the upstream copy regardless of what else is
installed in the environment.

Why this exists separately from ``sitecustomize``. A ``sitecustomize`` startup
hook is a process-wide singleton: Python imports the *first* ``sitecustomize``
it finds on ``sys.path`` and ignores the rest. In environments that ship their
own (e.g. an ``/opt/.../sitecustomize.py`` ahead of this repo on the path) the
repo hook never runs, and upstream ``datasets`` is silently shadowed. Relying on
``sitecustomize`` therefore makes correctness depend on path ordering we do not
control. :func:`enable_dino_wm_imports` is the explicit alternative: call it from
the process that launches upstream DINO-WM code, before importing any upstream
module. The ``sitecustomize`` shim under
``scripts/dino_wm/python_startup`` and the shell helper in
``scripts/dino_wm/mujoco_runtime_env.sh`` both reduce to this function; neither
is required for correctness, they are conveniences for the subprocess launch
path.

Upstream ``env`` is intentionally *not* eager-pinned: importing it pulls in
Gym/D4RL/MuJoCo, which is slow and unwanted in helper processes. It still
resolves to the upstream checkout on demand because the checkout is first on
``sys.path``.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

DEFAULT_EAGER_PACKAGES: tuple[str, ...] = ("datasets", "models", "planning")


def _is_inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        return False
    return True


def _remove_conflicting_modules(package_name: str, package_dir: Path) -> None:
    """Drop any already-imported ``package_name`` that is not the upstream one."""
    existing = sys.modules.get(package_name)
    if existing is not None:
        origin_value = getattr(existing, "__file__", "")
        if origin_value and _is_inside(Path(str(origin_value)), package_dir):
            return

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]


def _load_upstream_package(package_name: str, package_dir: Path) -> bool:
    """Import ``package_name`` from ``package_dir``, pinning it in ``sys.modules``.

    Returns ``True`` if the package was loaded, ``False`` if the directory is not
    an importable package. Propagates exceptions raised while executing the
    package's ``__init__`` (these are real failures, not "package missing").
    """
    init_path = package_dir / "__init__.py"
    if not init_path.is_file():
        return False

    _remove_conflicting_modules(package_name, package_dir)
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        return False

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(package_name, None)
        raise
    return True


def enable_dino_wm_imports(
    repo: str | Path, *, eager_packages: tuple[str, ...] = DEFAULT_EAGER_PACKAGES
) -> Path:
    """Put the upstream DINO-WM ``repo`` first on ``sys.path`` and pin its packages.

    Idempotent: re-running moves the repo back to the front of ``sys.path`` and
    re-pins the eager packages. Raises :class:`FileNotFoundError` if ``repo`` is
    not a directory (an explicit caller error worth surfacing); warns -- rather
    than failing -- if an individual eager package is absent from the checkout,
    so a partial upstream tree degrades gracefully.

    Returns the resolved repository path.
    """
    repo_path = Path(repo).expanduser()
    if not repo_path.is_absolute():
        repo_path = Path.cwd() / repo_path
    if not repo_path.is_dir():
        raise FileNotFoundError(f"DINO-WM repo not found or not a directory: {repo_path}")

    repo_str = str(repo_path)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)

    for package_name in eager_packages:
        package_dir = repo_path / package_name
        if not _load_upstream_package(package_name, package_dir):
            warnings.warn(
                f"DINO-WM bootstrap: upstream package {package_name!r} not found "
                f"under {repo_path}; skipping eager pin.",
                stacklevel=2,
            )
    return repo_path
