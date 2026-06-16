from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


PATCH_MARKER = "WM_POC_DINO_MIXED_PRECISION_PATCH"
ACCELERATOR_ENV_EXPR = 'os.environ.get("DINO_MIXED_PRECISION", "no")'


_ACCELERATOR_PATTERNS = (
    'Accelerator(log_with="wandb")',
    "Accelerator(log_with='wandb')",
)


def _ensure_os_import(source: str) -> str:
    if re.search(r"^import os(?:\s|$)", source, flags=re.MULTILINE):
        return source

    match = re.search(r"^(import [^\n]+\n)", source, flags=re.MULTILINE)
    if match is None:
        return f"import os  # {PATCH_MARKER}\n{source}"
    return source[: match.start()] + f"import os  # {PATCH_MARKER}\n" + source[match.start() :]


def patch_train_source(source: str) -> tuple[str, bool]:
    if PATCH_MARKER in source or "DINO_MIXED_PRECISION" in source:
        return source, False

    patched = source
    for old in _ACCELERATOR_PATTERNS:
        if old not in patched:
            continue
        patched = _ensure_os_import(patched)
        patched = patched.replace(
            old,
            f'Accelerator(log_with="wandb", mixed_precision={ACCELERATOR_ENV_EXPR})  # {PATCH_MARKER}',
            1,
        )
        return patched, True

    raise ValueError(
        "Could not apply DINO-WM mixed-precision patch; "
        "missing expected Accelerator(log_with=\"wandb\") constructor."
    )


def patch_train_file(train_path: Path) -> bool:
    train_path = train_path.expanduser()
    source = train_path.read_text(encoding="utf-8")
    patched, changed = patch_train_source(source)
    if not changed:
        return False

    backup_dir = train_path.parent / ".wm_poc_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"train.py.mixed_precision.{stamp}"
    shutil.copy2(train_path, backup_path)
    train_path.write_text(patched, encoding="utf-8")
    return True
