"""Run the upstream DINO-WM validation forward under torch.no_grad().

Upstream ``Trainer.val()`` calls ``self.model(obs, act)`` with autograd
enabled and never backpropagates, so each validation batch builds a full
forward graph and the previous batch's graph is still alive while the next
forward runs (the loop variables are only rebound mid-iteration). That keeps
two full batch-32 graphs resident on top of optimizer state — harmless on the
authors' 80 GB H100s, an instant CUDA OOM at the first epoch boundary on a
16 GB T4. Wrapping the call in ``no_grad`` removes the graphs entirely and
speeds validation up.

The decoder-only diagnostics inside ``if self.cfg.has_decoder and plot:`` are
left untouched; the wrapper configs run with ``has_decoder=false``.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


PATCH_MARKER = "WM_POC_DINO_VAL_NO_GRAD_PATCH"

_ANCHOR = """            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )
"""

_REPLACEMENT = f"""            obs, act, state = data
            plot = i == 0
            self.model.eval()
            with torch.no_grad():  # {PATCH_MARKER}
                z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                    obs, act
                )
"""


def patch_train_source(source: str) -> tuple[str, bool]:
    if PATCH_MARKER in source:
        return source, False
    if _ANCHOR not in source:
        raise ValueError(
            "Could not apply DINO-WM val no-grad patch; validation loop anchor "
            "not found in train.py."
        )
    return source.replace(_ANCHOR, _REPLACEMENT, 1), True


def patch_train_file(train_path: Path) -> bool:
    train_path = train_path.expanduser()
    source = train_path.read_text(encoding="utf-8")
    patched, changed = patch_train_source(source)
    if not changed:
        return False

    backup_dir = train_path.parent / ".wm_poc_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"train.py.val_no_grad.{stamp}"
    shutil.copy2(train_path, backup_path)
    train_path.write_text(patched, encoding="utf-8")
    return True
