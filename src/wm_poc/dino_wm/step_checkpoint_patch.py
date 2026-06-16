from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


PATCH_MARKER = "WM_POC_STEP_CHECKPOINTING_STATE_DICT_PATCH"
LEGACY_PATCH_MARKER = "WM_POC_STEP_CHECKPOINTING_PATCH"
PATCH_MARKERS = (PATCH_MARKER, LEGACY_PATCH_MARKER)
STEP_CHECKPOINT_FORMAT = "wm_poc_dino_step_state_dict_v1"


STEP_CHECKPOINT_HELPERS = f'''    def _move_to_cpu(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {{key: self._move_to_cpu(item) for key, item in value.items()}}
        if isinstance(value, list):
            return [self._move_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_to_cpu(item) for item in value)
        return value

    def _cpu_state_dict(self, obj):
        unwrapped = self.accelerator.unwrap_model(obj)
        return self._move_to_cpu(unwrapped.state_dict())

    def _cpu_optimizer_state_dict(self, optimizer):
        return self._move_to_cpu(optimizer.state_dict())

    def _rng_state(self):
        state = {{
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return self._move_to_cpu(state)

    def _restore_rng_state(self, rng_state):
        if not isinstance(rng_state, dict):
            return
        try:
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"])
            cuda_states = rng_state.get("cuda")
            if cuda_states is not None and torch.cuda.is_available() and len(cuda_states) > 0:
                torch.cuda.set_rng_state_all(cuda_states)
        except Exception as exc:
            log.warning("Could not restore DINO-WM step checkpoint RNG state: %s", exc)

    def _load_step_ckpt_payload(self, filename):
        try:
            ckpt = torch.load(filename, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(filename, map_location="cpu")
        if not isinstance(ckpt, dict) or ckpt.get("format") != "{STEP_CHECKPOINT_FORMAT}":
            raise ValueError(
                f"Expected {STEP_CHECKPOINT_FORMAT} at {{filename}}, "
                f"got keys: {{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}}"
            )
        return ckpt

    def save_step_ckpt(self, batch_index):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt = {{
                "format": "{STEP_CHECKPOINT_FORMAT}",
                "epoch": int(self.epoch),
                "batch_index": int(batch_index),
                "resume_batch_index": int(batch_index),
                "global_step": int(getattr(self, "train_step", 0)),
                "epoch_log": self._move_to_cpu(getattr(self, "epoch_log", OrderedDict())),
                "rng_state": self._rng_state(),
            }}
            if self.train_encoder and self.encoder is not None:
                ckpt["encoder"] = self._cpu_state_dict(self.encoder)
                ckpt["encoder_optimizer"] = self._cpu_optimizer_state_dict(self.encoder_optimizer)
            if self.cfg.has_predictor and self.train_predictor:
                ckpt["predictor"] = self._cpu_state_dict(self.predictor)
                ckpt["action_encoder"] = self._cpu_state_dict(self.action_encoder)
                ckpt["proprio_encoder"] = self._cpu_state_dict(self.proprio_encoder)
                ckpt["predictor_optimizer"] = self._cpu_optimizer_state_dict(
                    self.predictor_optimizer
                )
                ckpt["action_encoder_optimizer"] = self._cpu_optimizer_state_dict(
                    self.action_encoder_optimizer
                )
            if self.cfg.has_decoder and self.train_decoder and self.decoder is not None:
                ckpt["decoder"] = self._cpu_state_dict(self.decoder)
                ckpt["decoder_optimizer"] = self._cpu_optimizer_state_dict(self.decoder_optimizer)
            tmp_path = Path("checkpoints") / "model_latest_step.pth.tmp"
            final_path = Path("checkpoints") / "model_latest_step.pth"
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, final_path)
            log.info(
                "Saved rolling DINO-WM state_dict step checkpoint at epoch %s "
                "train_step %s batch %s to %s",
                self.epoch,
                getattr(self, "train_step", 0),
                batch_index,
                final_path,
            )
        self.accelerator.wait_for_everyone()

    def clear_step_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            for path in (
                Path("checkpoints") / "model_latest_step.pth",
                Path("checkpoints") / "model_latest_step.pth.tmp",
            ):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        self.accelerator.wait_for_everyone()

    def load_step_ckpt(self, filename):
        ckpt = self._load_step_ckpt_payload(filename)
        if "encoder" in ckpt and self.encoder is not None:
            self.accelerator.unwrap_model(self.encoder).load_state_dict(ckpt["encoder"])
        if "predictor" in ckpt and self.predictor is not None:
            self.accelerator.unwrap_model(self.predictor).load_state_dict(ckpt["predictor"])
        if "action_encoder" in ckpt and self.action_encoder is not None:
            self.accelerator.unwrap_model(self.action_encoder).load_state_dict(
                ckpt["action_encoder"]
            )
        if "proprio_encoder" in ckpt and self.proprio_encoder is not None:
            self.accelerator.unwrap_model(self.proprio_encoder).load_state_dict(
                ckpt["proprio_encoder"]
            )
        if "decoder" in ckpt and self.decoder is not None:
            self.accelerator.unwrap_model(self.decoder).load_state_dict(ckpt["decoder"])
        if "encoder_optimizer" in ckpt and hasattr(self, "encoder_optimizer"):
            self.encoder_optimizer.load_state_dict(ckpt["encoder_optimizer"])
        if "predictor_optimizer" in ckpt and hasattr(self, "predictor_optimizer"):
            self.predictor_optimizer.load_state_dict(ckpt["predictor_optimizer"])
        if "action_encoder_optimizer" in ckpt and hasattr(self, "action_encoder_optimizer"):
            self.action_encoder_optimizer.load_state_dict(ckpt["action_encoder_optimizer"])
        if "decoder_optimizer" in ckpt and hasattr(self, "decoder_optimizer"):
            self.decoder_optimizer.load_state_dict(ckpt["decoder_optimizer"])
        self.epoch = int(ckpt["epoch"])
        self.train_step = int(ckpt.get("global_step", 0))
        self.resume_batch_index = int(
            ckpt.get("resume_batch_index", ckpt.get("batch_index", -1))
        )
        self.epoch_log = ckpt.get("epoch_log", OrderedDict())
        self._restore_rng_state(ckpt.get("rng_state"))
        log.info(
            "Loaded DINO-WM state_dict step checkpoint from %s at epoch %s "
            "train_step %s; skipping through train batch %s",
            filename,
            self.epoch,
            self.train_step,
            self.resume_batch_index,
        )

'''


REPLACEMENTS = (
    (
        """import os
import time
""",
        f"""import os
import random  # {PATCH_MARKER}
import time
""",
    ),
    (
        """        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0
""",
        f"""        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0
        self.train_step = 0  # {PATCH_MARKER}
        self.resume_batch_index = -1
        self.epoch_log = OrderedDict()
        self._step_ckpt_path = None
""",
    ),
    (
        """        self.init_models()
        self.init_optimizers()

        self.epoch_log = OrderedDict()
""",
        """        self.init_models()
        self.init_optimizers()
        if self._step_ckpt_path is not None:
            self.load_step_ckpt(self._step_ckpt_path)
""",
    ),
    (
        """    def save_ckpt(self):
""",
        STEP_CHECKPOINT_HELPERS + """    def save_ckpt(self):
""",
    ),
    (
        """        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")
""",
        """        step_model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest_step.pth"
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if step_model_ckpt.exists():
            self._step_ckpt_path = step_model_ckpt
            log.info(
                "Found DINO-WM state_dict step checkpoint; loading after initialization: %s",
                step_model_ckpt,
            )
        elif model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")
""",
    ),
    (
        """        init_epoch = self.epoch + 1  # epoch starts from 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
""",
        f"""        if getattr(self, "resume_batch_index", -1) >= 0:
            init_epoch = self.epoch  # {PATCH_MARKER}
            log.info(
                "Resuming inside epoch %s after train batch %s",
                self.epoch,
                self.resume_batch_index,
            )
        else:
            init_epoch = self.epoch + 1  # epoch starts from 1
        for epoch in range(init_epoch, self.total_epochs + 1):
""",
    ),
    (
        """                ckpt_path, model_name, model_epoch = self.save_ckpt()
""",
        """                ckpt_path, model_name, model_epoch = self.save_ckpt()
                self.clear_step_ckpt()
""",
    ),
    (
        """    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
""",
        f"""    def train(self):
        resume_batch_index = int(getattr(self, "resume_batch_index", -1))  # {PATCH_MARKER}
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {{self.epoch}} Train")
        ):
            if resume_batch_index >= 0 and i <= resume_batch_index:
                continue
            obs, act, state = data
""",
    ),
    (
        """            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()
""",
        """            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            self.train_step = int(getattr(self, "train_step", 0)) + 1
            self.resume_batch_index = i
            save_every_steps = int(
                OmegaConf.select(self.cfg, "training.save_every_steps", default=0) or 0
            )
            if save_every_steps > 0 and self.train_step % save_every_steps == 0:
                self.save_step_ckpt(i)

            loss = self.accelerator.gather_for_metrics(loss).mean()
""",
    ),
    (
        """            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def val(self):
""",
        """            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

        if resume_batch_index >= 0:
            log.info("Finished resumed DINO-WM epoch %s; cleared train batch skip state.", self.epoch)
        self.resume_batch_index = -1

    def val(self):
""",
    ),
)


def _has_any_patch_marker(source: str) -> bool:
    return any(marker in source for marker in PATCH_MARKERS)


def patch_train_source(source: str) -> tuple[str, bool]:
    if PATCH_MARKER in source:
        return source, False
    if LEGACY_PATCH_MARKER in source:
        raise ValueError(
            "DINO-WM train.py contains the legacy step checkpointing patch. "
            "Restore it from backup before applying the state_dict patch."
        )

    patched = source
    missing = []
    for old, new in REPLACEMENTS:
        if old not in patched:
            missing.append(old.splitlines()[0].strip())
            continue
        patched = patched.replace(old, new, 1)

    if missing:
        details = ", ".join(missing)
        raise ValueError(f"Could not apply DINO-WM step checkpointing patch; missing anchors: {details}")
    return patched, True


def patch_train_file(train_path: Path) -> bool:
    train_path = train_path.expanduser()
    source = train_path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        return False
    if LEGACY_PATCH_MARKER in source:
        restore_train_file(train_path)
        source = train_path.read_text(encoding="utf-8")

    patched, changed = patch_train_source(source)
    if not changed:
        return False

    backup_dir = train_path.parent / ".wm_poc_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"train.py.step_checkpointing.{stamp}"
    shutil.copy2(train_path, backup_path)
    train_path.write_text(patched, encoding="utf-8")
    return True


def restore_train_file(train_path: Path) -> bool:
    train_path = train_path.expanduser()
    source = train_path.read_text(encoding="utf-8")
    if not _has_any_patch_marker(source):
        return False

    backup_dir = train_path.parent / ".wm_poc_backups"
    backups = sorted(backup_dir.glob("train.py.step_checkpointing.*"), reverse=True)
    for backup_path in backups:
        candidate = backup_path.read_text(encoding="utf-8")
        if not _has_any_patch_marker(candidate):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            patched_backup = backup_dir / f"train.py.patched_before_restore.{stamp}"
            shutil.copy2(train_path, patched_backup)
            train_path.write_text(candidate, encoding="utf-8")
            return True

    raise FileNotFoundError(
        f"Could not restore {train_path}: no unpatched backup found in {backup_dir}. "
        "Re-run setup with a fresh upstream DINO-WM checkout or restore train.py from upstream git."
    )
