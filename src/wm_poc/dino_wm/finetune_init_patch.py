"""Teach upstream DINO-WM train.py to initialize from a source checkpoint.

Upstream has no fine-tuning support: its Hydra schema has no ``finetuning``
section and ``init_models`` only knows how to resume a run's own
``model_latest.pth``. The wrapper passes ``++finetuning.*`` overrides (append
syntax, so Hydra accepts the new keys) and this patch adds the consumer: a
hook at the end of ``init_models`` that, for a fresh fine-tune run, loads the
predictor / action encoder / proprio encoder (optionally decoder) weights
from ``finetuning.init_from``.

Resume still wins: the hook is skipped when the run already restored its own
epoch checkpoint, and a rolling step checkpoint is loaded after model init,
overwriting these weights with the resumed state.

Learning rates and epoch counts need no upstream support — the wrapper maps
``finetuning.{predictor_lr,action_encoder_lr,epochs}`` onto the plain
``training.*`` overrides at command-build time.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


PATCH_MARKER = "WM_POC_DINO_FINETUNE_INIT_PATCH"

_ANCHOR = """            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )

    def init_optimizers(self):
"""

_REPLACEMENT = f"""            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )
        self._wm_poc_apply_finetune_init()  # {PATCH_MARKER}

    def _wm_poc_load_finetune_ckpt(self, path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch without weights_only
            return torch.load(path, map_location="cpu")

    def _wm_poc_load_finetune_component(self, ckpt, name, module, strict):
        if module is None:
            return
        source = ckpt.get(name)
        if source is None:
            log.warning("Fine-tune source checkpoint has no %s; leaving fresh init.", name)
            return
        state = source.state_dict() if hasattr(source, "state_dict") else source
        self.accelerator.unwrap_model(module).load_state_dict(state, strict=strict)
        log.info("Fine-tune initialized %s from the source checkpoint.", name)

    def _wm_poc_apply_finetune_init(self):
        if not bool(OmegaConf.select(self.cfg, "finetuning.enabled", default=False)):
            return
        if int(getattr(self, "epoch", 0) or 0) > 0:
            log.info(
                "Fine-tune init skipped; run already resumed at epoch %s.", self.epoch
            )
            return
        init_from = OmegaConf.select(self.cfg, "finetuning.init_from", default=None)
        if init_from in (None, "", "null", "None"):
            raise ValueError("finetuning.enabled=true requires finetuning.init_from.")
        ckpt = self._wm_poc_load_finetune_ckpt(str(init_from))
        strict = bool(OmegaConf.select(self.cfg, "finetuning.strict", default=True))
        if bool(OmegaConf.select(self.cfg, "finetuning.load_predictor", default=True)):
            self._wm_poc_load_finetune_component(ckpt, "predictor", self.predictor, strict)
        if bool(OmegaConf.select(self.cfg, "finetuning.load_action_encoder", default=True)):
            self._wm_poc_load_finetune_component(
                ckpt, "action_encoder", self.action_encoder, strict
            )
            self._wm_poc_load_finetune_component(
                ckpt, "proprio_encoder", self.proprio_encoder, strict
            )
        if bool(OmegaConf.select(self.cfg, "finetuning.load_decoder", default=False)):
            self._wm_poc_load_finetune_component(ckpt, "decoder", self.decoder, strict)
        if not bool(OmegaConf.select(self.cfg, "finetuning.reset_epoch", default=True)):
            self.epoch = int(ckpt.get("epoch", 0))
        log.info("Fine-tune initialization complete from %s", init_from)

    def init_optimizers(self):
"""


def patch_train_source(source: str) -> tuple[str, bool]:
    if PATCH_MARKER in source:
        return source, False
    if _ANCHOR not in source:
        raise ValueError(
            "Could not apply DINO-WM fine-tune init patch; init_models tail anchor "
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
    backup_path = backup_dir / f"train.py.finetune_init.{stamp}"
    shutil.copy2(train_path, backup_path)
    train_path.write_text(patched, encoding="utf-8")
    return True
