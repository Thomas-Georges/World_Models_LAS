from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from wm_poc.dino_wm.checkpoints import find_latest_checkpoint
from wm_poc.dino_wm.commands import (
    build_plan_command,
    build_precompute_command,
    build_train_command,
    latent_cache_dir,
    latent_training_enabled,
    render_command,
)
from wm_poc.dino_wm.configs import load_config, resolve_config, validate_config
from wm_poc.dino_wm.data import build_split_manifest, validate_dataset_root
from wm_poc.dino_wm.latent_cache_patch import (
    LATENT_DATASET_MODULE_NAME,
    LATENT_DATASET_MODULE_SOURCE,
    MODEL_PATCH_MARKER,
    PRECOMPUTE_SCRIPT_NAME,
    PRECOMPUTE_SCRIPT_SOURCE,
    install_latent_support,
    patch_model_file,
    patch_model_source,
)
from wm_poc.dino_wm.mujoco import REQUIRED_MUJOCO210_LIBS, configure_mujoco_runtime_env, has_mujoco210_runtime
from wm_poc.dino_wm.mixed_precision_patch import (
    PATCH_MARKER as MIXED_PRECISION_PATCH_MARKER,
    patch_train_file as patch_mixed_precision_train_file,
    patch_train_source as patch_mixed_precision_train_source,
)
from wm_poc.dino_wm.notebook_monitor import (
    _command_for_stage,
    _log_freshness,
    _read_new_matches,
    _read_status,
    _tail_lines,
    run_dino_with_live_display,
)
from wm_poc.dino_wm.resume import latest_checkpoint_path, prepare_training_resume
from wm_poc.dino_wm.step_checkpoint_patch import (
    PATCH_MARKER,
    STEP_CHECKPOINT_FORMAT,
    patch_train_file,
    patch_train_source,
    restore_train_file,
)


def _upstream_like_train_source() -> str:
    return """import os
import time
import logging
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self):
        self.cfg = None
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0
        self._keys_to_save = [
            "epoch",
        ]
        self.init_models()
        self.init_optimizers()

        self.epoch_log = OrderedDict()

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    ckpt[k] = self.__dict__[k]
            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        pass

    def init_models(self):
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

    def init_optimizers(self):
        pass

    def run(self):
        init_epoch = self.epoch + 1  # epoch starts from 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                if ckpt_path is not None:
                    pass

    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()
            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def val(self):
        pass
"""


def _upstream_like_mixed_precision_source() -> str:
    return """import time
from accelerate import Accelerator

class Trainer:
    def __init__(self):
        self.accelerator = Accelerator(log_with="wandb")
"""


def _upstream_like_model_source() -> str:
    return '''import torch
import torch.nn as nn
from einops import rearrange, repeat

class VWorldModel(nn.Module):
    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}
'''


def test_dino_wm_smoke_config_builds_train_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    config = resolve_config(load_config("configs/dino_wm/smoke_pointmaze.yaml"))
    validate_config(config)

    rendered = render_command(build_train_command(config))

    assert "/tmp/dino_wm/train.py" in rendered
    assert "env=point_maze" in rendered
    assert "frameskip=5" in rendered
    assert "num_hist=3" in rendered
    assert "training.seed=0" in rendered
    assert "training.batch_size=4" in rendered
    assert "training.epochs=1" in rendered
    assert "training.save_every_x_epoch=1" in rendered
    assert "env.dataset.n_rollout=20" in rendered
    assert "env.dataset.split_ratio=0.800000" in rendered


def test_dino_wm_oom_safe_config_renders_memory_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.delenv("DINO_SAVE_EVERY_STEPS", raising=False)
    config = resolve_config(load_config("configs/dino_wm/pointmaze_oom_safe.yaml"))
    validate_config(config)

    rendered = render_command(build_train_command(config))

    assert "training.batch_size=2" in rendered
    assert "training.epochs=2" in rendered
    assert "training.save_every_x_epoch=2" in rendered
    assert "env.dataset.n_rollout=72" in rendered
    assert "env.dataset.split_ratio=0.888889" in rendered
    assert "has_decoder=false" in rendered
    assert "model.train_decoder=false" in rendered
    assert "++training.num_reconstruct_samples=0" in rendered
    assert "env.num_workers=0" in rendered
    assert "++training.save_every_steps" not in rendered


def test_dino_wm_full_nodecoder_config_renders_bf16_profile_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.delenv("DINO_SAVE_EVERY_STEPS", raising=False)
    monkeypatch.delenv("DINO_SAVE_EVERY_EPOCHS", raising=False)
    config = resolve_config(load_config("configs/dino_wm/pointmaze_full_nodecoder_bf16.yaml"))
    validate_config(config)

    rendered = render_command(build_train_command(config))

    assert "pointmaze_full_nodecoder_bf16_a100_b32_seed0" in rendered
    assert "training.batch_size=32" in rendered
    assert "training.epochs=10" in rendered
    assert "training.save_every_x_epoch=1" in rendered
    assert "has_decoder=false" in rendered
    assert "model.train_decoder=false" in rendered
    assert "++training.num_reconstruct_samples=0" in rendered
    # Disconnect insurance: rolling intra-epoch checkpoint every 2000 steps.
    assert "++training.save_every_steps=2000" in rendered
    # Throughput fixes: dataloader workers stay on and training reads the
    # precomputed latent cache instead of re-encoding images online.
    assert "env.num_workers=4" in rendered
    assert "env.num_workers=0" not in rendered
    assert (
        "env.dataset._target_=wm_poc_latent_dataset.load_point_maze_latent_slice_train_val"
        in rendered
    )
    assert "+env.dataset.latent_cache_dir=" in rendered
    assert "+env.dataset.slice_stride=1" in rendered


def test_dino_wm_train_command_respects_save_every_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.setenv("DINO_SAVE_EVERY_EPOCHS", "2")
    monkeypatch.setenv("DINO_SAVE_EVERY_STEPS", "250")
    config = resolve_config(load_config("configs/dino_wm/pointmaze_scratch_a100.yaml"))
    validate_config(config)

    rendered = render_command(build_train_command(config))

    assert "training.epochs=60" in rendered
    assert "training.save_every_x_epoch=2" in rendered
    assert "++training.save_every_steps=250" in rendered


def test_dino_wm_train_command_disables_default_step_checkpoint_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.delenv("DINO_SAVE_EVERY_STEPS", raising=False)
    config = resolve_config(load_config("configs/dino_wm/pointmaze_scratch_a100.yaml"))
    validate_config(config)

    rendered = render_command(build_train_command(config))

    assert "++training.save_every_steps" not in rendered


def test_dino_wm_mixed_precision_patch_is_idempotent(tmp_path: Path) -> None:
    source = _upstream_like_mixed_precision_source()
    train_path = tmp_path / "train.py"
    train_path.write_text(source, encoding="utf-8")

    changed = patch_mixed_precision_train_file(train_path)
    patched = train_path.read_text(encoding="utf-8")
    second, changed_again = patch_mixed_precision_train_source(patched)

    assert changed
    assert not changed_again
    assert second == patched
    assert MIXED_PRECISION_PATCH_MARKER in patched
    assert 'import os' in patched
    assert 'mixed_precision=os.environ.get("DINO_MIXED_PRECISION", "no")' in patched


def test_dino_wm_resume_keeps_latest_checkpoint_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DINO_FORCE_RESTART", raising=False)
    config = resolve_config(load_config("configs/dino_wm/pointmaze_scratch_a100.yaml"))
    config["artifacts"]["ckpt_root"] = str(tmp_path)
    latest = latest_checkpoint_path(config)
    latest.parent.mkdir(parents=True)
    latest.write_text("fake checkpoint", encoding="utf-8")

    state = prepare_training_resume(config)

    assert state["action"] == "resume"
    assert state["checkpoint_path"] == str(latest)
    assert latest.is_file()


def test_dino_wm_force_restart_moves_existing_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINO_FORCE_RESTART", "1")
    config = resolve_config(load_config("configs/dino_wm/pointmaze_scratch_a100.yaml"))
    config["artifacts"]["ckpt_root"] = str(tmp_path)
    latest = latest_checkpoint_path(config)
    latest.parent.mkdir(parents=True)
    latest.write_text("fake checkpoint", encoding="utf-8")

    state = prepare_training_resume(config)

    assert state["action"] == "fresh_start_backup"
    assert not latest.parent.exists()
    backup_dir = Path(state["backup_dir"])
    assert (backup_dir / "model_latest.pth").is_file()


def test_dino_wm_latest_checkpoint_prefers_upstream_model_latest(tmp_path: Path) -> None:
    older = tmp_path / "checkpoints" / "model_1.pth"
    latest = tmp_path / "checkpoints" / "model_latest.pth"
    latest.parent.mkdir(parents=True)
    older.write_text("older", encoding="utf-8")
    latest.write_text("latest", encoding="utf-8")

    assert find_latest_checkpoint(tmp_path) == latest


def test_dino_wm_step_checkpoint_patch_is_idempotent(tmp_path: Path) -> None:
    source = _upstream_like_train_source()
    train_path = tmp_path / "train.py"
    train_path.write_text(source, encoding="utf-8")

    changed = patch_train_file(train_path)
    patched = train_path.read_text(encoding="utf-8")
    second, changed_again = patch_train_source(patched)

    assert changed
    assert not changed_again
    assert second == patched
    assert PATCH_MARKER in patched
    assert STEP_CHECKPOINT_FORMAT in patched
    assert "self.save_step_ckpt(i)" in patched
    assert "self.save_ckpt(latest_only=True)" not in patched
    assert "model_latest_step.pth" in patched
    assert "os.replace(tmp_path, final_path)" in patched
    assert "model_step_" not in patched
    assert "for epoch in range(init_epoch, self.total_epochs + 1):" in patched
    assert "i <= resume_batch_index" in patched


def test_dino_wm_step_checkpoint_restore_uses_pre_patch_backup(tmp_path: Path) -> None:
    source = _upstream_like_train_source()
    train_path = tmp_path / "train.py"
    train_path.write_text(source, encoding="utf-8")
    assert patch_train_file(train_path)

    restored = restore_train_file(train_path)

    assert restored
    assert train_path.read_text(encoding="utf-8") == source
    backups = list((tmp_path / ".wm_poc_backups").glob("train.py.patched_before_restore.*"))
    assert len(backups) == 1


def test_dino_wm_step_checkpoint_patch_generated_source_compiles(tmp_path: Path) -> None:
    train_path = tmp_path / "train.py"
    train_path.write_text(_upstream_like_train_source(), encoding="utf-8")
    assert patch_train_file(train_path)

    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(train_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr


def test_dino_wm_plan_command_supports_planner_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.setenv("DINO_PLANNER", "gd")
    config = resolve_config(load_config("configs/dino_wm/planner_cem_vs_gd_pointmaze.yaml"))
    validate_config(config)

    rendered = render_command(
        build_plan_command(config, "/tmp/ckpts/outputs/source_run/checkpoints/model_latest.pth")
    )

    assert "/tmp/dino_wm/plan.py" in rendered
    assert "planner=gd" in rendered
    # the model reference follows the checkpoint, not the planner config
    assert "model_name=source_run" in rendered
    assert "ckpt_base_path=/tmp/ckpts" in rendered
    assert "model_epoch=latest" in rendered


def test_dino_wm_config_rejects_over_budget_training() -> None:
    config = resolve_config(load_config("configs/dino_wm/smoke_pointmaze.yaml"))
    config["training"]["max_wall_minutes"] = 600  # full stride-1 / rerun schedules are allowed
    validate_config(config)

    config["training"]["max_wall_minutes"] = 601
    with pytest.raises(ValueError, match="training.max_wall_minutes"):
        validate_config(config)


def test_dino_wm_split_manifest_uses_capped_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    env_dir = data_root / "point_maze"
    env_dir.mkdir(parents=True)
    for index in range(10):
        (env_dir / f"traj_{index:02d}.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("DINO_WM_DATA_ROOT", str(data_root))
    config = resolve_config(load_config("configs/dino_wm/smoke_pointmaze.yaml"))
    validate_dataset_root(data_root, "point_maze")
    manifest = build_split_manifest(config)

    assert manifest["env"] == "point_maze"
    assert manifest["num_available_files"] == 10
    assert manifest["num_train_files"] <= 16
    assert manifest["num_val_files"] <= 4
    assert manifest["files"]
    assert manifest["checksum_mode"] == "metadata"
    assert all(item["checksum"].startswith("metadata-token-") for item in manifest["files"])


def test_dino_wm_config_rejects_invalid_checksum_mode() -> None:
    config = resolve_config(load_config("configs/dino_wm/smoke_pointmaze.yaml"))
    config["dataset"]["checksum_mode"] = "slow"

    with pytest.raises(ValueError, match="dataset.checksum_mode"):
        validate_config(config)


def test_dino_wm_mujoco_runtime_env_sets_legacy_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mujoco_dir = tmp_path / "mujoco210"
    (mujoco_dir / "bin").mkdir(parents=True)
    monkeypatch.setenv("DINO_MUJOCO210_DIR", str(mujoco_dir))
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    monkeypatch.delenv("MUJOCO_PY_MUJOCO_PATH", raising=False)
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    selected = configure_mujoco_runtime_env()

    assert selected == mujoco_dir
    assert Path(os.environ["MUJOCO_PY_MUJOCO_PATH"]) == mujoco_dir
    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert str(mujoco_dir / "bin") in os.environ["LD_LIBRARY_PATH"].split(":")


def test_dino_wm_mujoco_runtime_requires_legacy_libraries(tmp_path: Path) -> None:
    mujoco_dir = tmp_path / "mujoco210"
    bin_dir = mujoco_dir / "bin"
    bin_dir.mkdir(parents=True)

    assert not has_mujoco210_runtime(mujoco_dir)

    for name in REQUIRED_MUJOCO210_LIBS:
        (bin_dir / name).write_text("", encoding="utf-8")

    assert has_mujoco210_runtime(mujoco_dir)


def _make_upstream_and_shadow(tmp_path: Path) -> tuple[Path, Path]:
    """Build a fake upstream DINO-WM checkout and a conflicting shadow tree.

    The shadow provides top-level ``datasets``/``env`` packages (standing in for
    e.g. Hugging Face ``datasets``) that must lose to the upstream checkout.
    Upstream ``env`` records that it was imported, so a test can assert it stays
    lazy (never imported eagerly).
    """
    upstream = tmp_path / "upstream_dino_wm"
    upstream_datasets = upstream / "datasets"
    upstream_datasets.mkdir(parents=True)
    (upstream_datasets / "__init__.py").write_text("SOURCE = 'upstream'\n", encoding="utf-8")
    (upstream_datasets / "point_maze_dset.py").write_text("VALUE = 'point_maze'\n", encoding="utf-8")
    upstream_env = upstream / "env"
    upstream_env.mkdir()
    (upstream_env / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(tmp_path / 'env_imported.txt')!r}).write_text('imported', encoding='utf-8')\n"
        "SOURCE = 'upstream_env'\n",
        encoding="utf-8",
    )

    shadow = tmp_path / "site_packages"
    shadow_datasets = shadow / "datasets"
    shadow_datasets.mkdir(parents=True)
    (shadow_datasets / "__init__.py").write_text("SOURCE = 'shadow'\n", encoding="utf-8")
    shadow_env = shadow / "env"
    shadow_env.mkdir()
    (shadow_env / "__init__.py").write_text("SOURCE = 'shadow_env'\n", encoding="utf-8")
    return upstream, shadow


def test_dino_wm_import_bootstrap_prefers_upstream_datasets(tmp_path: Path) -> None:
    """The explicit bootstrap pins upstream ``datasets`` over a shadow on the path.

    This is the canonical mechanism: it does not depend on which ``sitecustomize``
    the interpreter happens to load, only on calling ``enable_dino_wm_imports``.
    """
    upstream, shadow = _make_upstream_and_shadow(tmp_path)
    src = str(Path("src").resolve())

    env = os.environ.copy()
    env.pop("DINO_WM_REPO", None)
    # Shadow 'datasets' is importable; the upstream repo is NOT on the path at
    # all -- the bootstrap call is what must put it first.
    env["PYTHONPATH"] = os.pathsep.join([src, str(shadow)])
    code = (
        "from wm_poc.dino_wm.import_bootstrap import enable_dino_wm_imports;"
        f"enable_dino_wm_imports({str(upstream)!r});"
        "import datasets, datasets.point_maze_dset as point_maze;"
        "print(datasets.SOURCE); print(point_maze.VALUE); print(datasets.__file__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    source, value, package_file = result.stdout.strip().splitlines()
    assert source == "upstream"
    assert value == "point_maze"
    assert Path(package_file).parent == upstream / "datasets"
    # ``env`` must not be imported by the bootstrap (it would pull in MuJoCo/D4RL).
    assert not (tmp_path / "env_imported.txt").exists()

    # An explicit ``import env`` still resolves to upstream (checkout is first on
    # sys.path) -- only the eager pinning skips it.
    code_env = (
        "from wm_poc.dino_wm.import_bootstrap import enable_dino_wm_imports;"
        f"enable_dino_wm_imports({str(upstream)!r});"
        "import env; print(env.SOURCE); print(env.__file__)"
    )
    result_env = subprocess.run(
        [sys.executable, "-c", code_env],
        check=True,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env_source, env_file = result_env.stdout.strip().splitlines()
    assert env_source == "upstream_env"
    assert Path(env_file).parent == upstream / "env"


def test_dino_wm_import_bootstrap_rejects_missing_repo(tmp_path: Path) -> None:
    from wm_poc.dino_wm.import_bootstrap import enable_dino_wm_imports

    with pytest.raises(FileNotFoundError):
        enable_dino_wm_imports(tmp_path / "does_not_exist")


def test_dino_wm_sitecustomize_delegates_to_bootstrap(tmp_path: Path) -> None:
    """The startup shim, when it runs, reuses the explicit bootstrap.

    Executed deterministically via ``runpy`` so the test does not depend on the
    interpreter selecting this repo's ``sitecustomize`` over another one.
    """
    upstream, shadow = _make_upstream_and_shadow(tmp_path)
    sitecustomize = str(Path("scripts/dino_wm/python_startup/sitecustomize.py").resolve())

    env = os.environ.copy()
    env["DINO_WM_REPO"] = str(upstream)
    env["PYTHONPATH"] = str(shadow)  # conflicting top-level 'datasets'
    code = (
        "import runpy;"
        f"runpy.run_path({sitecustomize!r}, run_name='sitecustomize');"
        "import datasets, datasets.point_maze_dset as point_maze;"
        "print(datasets.SOURCE); print(point_maze.VALUE)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    source, value = result.stdout.strip().splitlines()
    assert source == "upstream"
    assert value == "point_maze"
    assert not (tmp_path / "env_imported.txt").exists()


def test_dino_wm_monitor_reads_new_matches_and_truncation(tmp_path: Path) -> None:
    log = tmp_path / "stdout.log"
    pattern = re.compile(r"Epoch [0-9]+|Traceback")

    log.write_text("setup\nEpoch 1 Training loss: 2.4\n", encoding="utf-8")
    offset, matches = _read_new_matches(log, pattern, 0)
    assert matches == ["Epoch 1 Training loss: 2.4"]

    with log.open("a", encoding="utf-8") as f:
        f.write("ignored\nTraceback (most recent call last):\n")
    offset, matches = _read_new_matches(log, pattern, offset)
    assert matches == ["Traceback (most recent call last):"]

    log.write_text("Epoch 0 Training loss: 3.0\n", encoding="utf-8")
    offset, matches = _read_new_matches(log, pattern, offset)
    assert matches == ["Epoch 0 Training loss: 3.0"]
    assert offset == log.stat().st_size


def test_dino_wm_monitor_reads_tqdm_carriage_return_updates(tmp_path: Path) -> None:
    log = tmp_path / "stderr.log"
    pattern = re.compile(r"Epoch [0-9]+ (Train|Valid):")

    log.write_text(
        "\rEpoch 4 Train:   0%|          | 0/4 [00:00<?, ?it/s]"
        "\rEpoch 4 Train:  25%|##5       | 1/4 [01:00<03:00, 60.0s/it]",
        encoding="utf-8",
    )
    offset, matches = _read_new_matches(log, pattern, 0)

    assert matches == [
        "Epoch 4 Train:   0%|          | 0/4 [00:00<?, ?it/s]",
        "Epoch 4 Train:  25%|##5       | 1/4 [01:00<03:00, 60.0s/it]",
    ]
    assert offset == log.stat().st_size


def test_dino_wm_monitor_tail_lines(tmp_path: Path) -> None:
    log = tmp_path / "launcher.log"
    log.write_text("\n".join(f"line {index}" for index in range(5)), encoding="utf-8")

    assert _tail_lines(log, 2) == ["line 3", "line 4"]
    assert _tail_lines(tmp_path / "missing.log", 2) == []


def test_dino_wm_monitor_ignores_stale_status_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "smoke_pointmaze_seed0"
    planning_dir = run_dir / "planning"
    planning_dir.mkdir(parents=True)
    status = run_dir / "status.json"
    plan_status = planning_dir / "status_cem.json"
    status.write_text('{"completed": true, "elapsed_seconds": 60, "return_code": 0}', encoding="utf-8")
    plan_status.write_text('{"failed": true, "elapsed_seconds": 30, "return_code": 1}', encoding="utf-8")
    old_time = 1_000_000_000
    os.utime(status, (old_time, old_time))
    os.utime(plan_status, (old_time, old_time))

    assert _read_status(run_dir, min_mtime=old_time + 10) == []

    fresh_time = old_time + 20
    os.utime(status, (fresh_time, fresh_time))
    lines = _read_status(run_dir, min_mtime=old_time + 10)

    assert len(lines) == 1
    assert lines[0].startswith("status.json: completed")


def test_dino_wm_monitor_reports_log_freshness(tmp_path: Path) -> None:
    log = tmp_path / "stdout.log"
    log.write_text("hello\n", encoding="utf-8")
    os.utime(log, (100, 100))

    lines = _log_freshness([log, tmp_path / "missing.log"], now=160)

    assert lines[0] == "stdout.log: 6 bytes, updated 1.0m ago"
    assert lines[1] == "missing.log: missing"


def test_dino_wm_monitor_builds_stage_commands() -> None:
    assert _command_for_stage("smoke", "configs/dino_wm/smoke_pointmaze.yaml") == [
        "bash",
        "scripts/dino_wm/run_smoke.sh",
    ]
    assert _command_for_stage(
        "experiment",
        "configs/dino_wm/pointmaze_scratch_a100.yaml",
        skip_cache=True,
        skip_plan=True,
    ) == [
        "bash",
        "scripts/dino_wm/run_experiment.sh",
        "--config",
        "configs/dino_wm/pointmaze_scratch_a100.yaml",
        "--skip-cache",
        "--skip-plan",
    ]
    assert _command_for_stage(
        "plan",
        "configs/dino_wm/pointmaze_scratch_a100.yaml",
        checkpoint="/tmp/model.pt",
    ) == [
        "bash",
        "scripts/dino_wm/run_plan.sh",
        "--config",
        "configs/dino_wm/pointmaze_scratch_a100.yaml",
        "--checkpoint",
        "/tmp/model.pt",
    ]


def test_dino_wm_monitor_rejects_invalid_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="monitor_interval"):
        run_dino_with_live_display("smoke", repo_dir=tmp_path, monitor_interval=0)


def test_dino_wm_diagnostic_configs_keep_online_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    for name in ("smoke_pointmaze.yaml", "pointmaze_oom_safe.yaml"):
        config = resolve_config(load_config(f"configs/dino_wm/{name}"))
        validate_config(config)
        assert not latent_training_enabled(config)
        rendered = render_command(build_train_command(config))
        assert "wm_poc_latent_dataset" not in rendered
        assert "latent_cache_dir" not in rendered


def test_dino_wm_latent_cache_dir_is_encoder_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DINO_WM_FEATURE_CACHE", raising=False)
    config = resolve_config(load_config("configs/dino_wm/pointmaze_full_nodecoder_bf16.yaml"))

    assert latent_training_enabled(config)
    assert latent_cache_dir(config) == "/content/wm_poc_latent_cache/point_maze/dinov2_vits14_img224"


def test_dino_wm_precompute_command_targets_installed_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.delenv("DINO_WM_FEATURE_CACHE", raising=False)
    config = resolve_config(load_config("configs/dino_wm/pointmaze_full_nodecoder_bf16.yaml"))
    validate_config(config)

    argv = build_precompute_command(config)
    rendered = render_command(argv)

    assert "/tmp/dino_wm/wm_poc_precompute_latents.py" in rendered
    assert "--cache-dir /content/wm_poc_latent_cache/point_maze/dinov2_vits14_img224" in rendered
    assert "--n-rollout 2200" in rendered
    assert "--img-size 224" in rendered
    assert "--encoder-name dinov2_vits14" in rendered
    assert "--batch-size 128" in rendered


def test_dino_wm_latent_support_sources_are_valid_python() -> None:
    compile(LATENT_DATASET_MODULE_SOURCE, LATENT_DATASET_MODULE_NAME, "exec")
    compile(PRECOMPUTE_SCRIPT_SOURCE, PRECOMPUTE_SCRIPT_NAME, "exec")
    assert "def load_point_maze_latent_slice_train_val(" in LATENT_DATASET_MODULE_SOURCE
    assert "class LatentTrajSlicerDataset" in LATENT_DATASET_MODULE_SOURCE
    assert "wm_poc_latent_manifest.json" in LATENT_DATASET_MODULE_SOURCE
    assert "def main():" in PRECOMPUTE_SCRIPT_SOURCE
    assert "wm_poc_latent_manifest.json" in PRECOMPUTE_SCRIPT_SOURCE


def test_dino_wm_install_latent_support_is_idempotent(tmp_path: Path) -> None:
    first = install_latent_support(tmp_path)
    second = install_latent_support(tmp_path)

    assert sorted(first) == sorted([LATENT_DATASET_MODULE_NAME, PRECOMPUTE_SCRIPT_NAME])
    assert second == []
    assert (tmp_path / LATENT_DATASET_MODULE_NAME).is_file()
    assert (tmp_path / PRECOMPUTE_SCRIPT_NAME).is_file()


def test_dino_wm_latent_bypass_patch_is_idempotent(tmp_path: Path) -> None:
    model_path = tmp_path / "visual_world_model.py"
    model_path.write_text(_upstream_like_model_source(), encoding="utf-8")

    changed = patch_model_file(model_path)
    patched = model_path.read_text(encoding="utf-8")
    second, changed_again = patch_model_source(patched)

    assert changed
    assert not changed_again
    assert second == patched
    assert MODEL_PATCH_MARKER in patched
    assert "if visual.ndim == 4:" in patched
    # The online path must survive for image inputs (planning, smoke runs).
    assert 'visual = self.encoder_transform(visual)' in patched
    assert list((tmp_path / ".wm_poc_backups").glob("visual_world_model.py.latent_bypass.*"))
    compile(patched, "visual_world_model.py", "exec")


def test_dino_wm_latent_bypass_patch_requires_anchor() -> None:
    with pytest.raises(ValueError, match="latent bypass"):
        patch_model_source("def encode_obs(self, obs):\n    return obs\n")


def test_dino_wm_rotate_stale_logs(tmp_path: Path) -> None:
    from wm_poc.dino_wm.notebook_monitor import _rotate_stale_logs

    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"
    empty_log = tmp_path / "empty.log"
    stdout_log.write_text("Epoch 1 Train: 5%\n", encoding="utf-8")
    empty_log.write_text("", encoding="utf-8")

    rotated = _rotate_stale_logs((stdout_log, stderr_log, empty_log))

    assert rotated == ["stdout.log"]
    assert not stdout_log.exists()
    generations = list(tmp_path.glob("stdout.log.*.prev"))
    assert len(generations) == 1
    assert generations[0].read_text(encoding="utf-8") == "Epoch 1 Train: 5%\n"
    # empty and missing logs stay put
    assert empty_log.exists()

    # a second rotation keeps every generation (training curves merge them)
    stdout_log.write_text("fresh run\n", encoding="utf-8")
    assert _rotate_stale_logs((stdout_log,)) == ["stdout.log"]
    contents = {
        path.read_text(encoding="utf-8") for path in tmp_path.glob("stdout.log.*.prev")
    }
    assert contents == {"Epoch 1 Train: 5%\n", "fresh run\n"}


def test_dino_wm_stale_pid_scan_parses_pgrep(monkeypatch: pytest.MonkeyPatch) -> None:
    from wm_poc.dino_wm import notebook_monitor

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        class Result:
            stdout = f"123\n{os.getpid()}\nnot-a-pid\n123\n"
        return Result()

    monkeypatch.setattr(notebook_monitor.subprocess, "run", fake_run)

    pids = notebook_monitor._stale_dino_pids()

    assert pids == [123]
    assert all(argv[:2] == ["pgrep", "-f"] for argv in calls)
    patterns = {argv[2] for argv in calls}
    assert "dino_wm/train.py" in patterns
    assert "wm_poc_precompute_latents.py" in patterns


def test_dino_wm_stale_processes_killed_or_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from wm_poc.dino_wm import notebook_monitor

    monkeypatch.setattr(notebook_monitor, "_stale_dino_pids", lambda: [4242])
    killed: list[int] = []
    monkeypatch.setattr(
        notebook_monitor, "_terminate_stale_dino_processes", lambda: killed.append(4242) or [4242]
    )

    monkeypatch.setenv("DINO_KILL_STALE", "1")
    lines = notebook_monitor._handle_stale_processes()
    assert killed == [4242]
    assert any("4242" in line for line in lines)

    monkeypatch.setenv("DINO_KILL_STALE", "0")
    with pytest.raises(RuntimeError, match="Stale DINO-WM process"):
        notebook_monitor._handle_stale_processes()

    monkeypatch.setattr(notebook_monitor, "_stale_dino_pids", lambda: [])
    assert notebook_monitor._handle_stale_processes() == []


def test_dino_wm_latent_sources_warn_about_drive_cache() -> None:
    assert "WARNING: latent cache is on Google Drive" in PRECOMPUTE_SCRIPT_SOURCE
    assert "WARNING: reading DINO latents from Google Drive" in LATENT_DATASET_MODULE_SOURCE


def test_dino_wm_t4_profile_halves_steps_with_stride(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.delenv("DINO_NUM_WORKERS", raising=False)
    monkeypatch.delenv("DINO_SAVE_EVERY_STEPS", raising=False)
    monkeypatch.delenv("DINO_SAVE_EVERY_EPOCHS", raising=False)
    config = resolve_config(load_config("configs/dino_wm/pointmaze_full_nodecoder_t4.yaml"))
    validate_config(config)

    rendered = render_command(build_train_command(config))

    assert "pointmaze_full_nodecoder_t4_fp16_b32_stride2_seed0" in rendered
    assert "+env.dataset.slice_stride=2" in rendered
    assert "env.num_workers=2" in rendered
    assert "training.batch_size=32" in rendered
    assert "has_decoder=false" in rendered
    # T4 steps are ~0.76 s, so the rolling checkpoint lands every ~6 min.
    assert "++training.save_every_steps=500" in rendered
    assert latent_training_enabled(config)


def test_dino_wm_train_command_respects_num_workers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.setenv("DINO_NUM_WORKERS", "8")
    config = resolve_config(load_config("configs/dino_wm/pointmaze_full_nodecoder_bf16.yaml"))
    validate_config(config)

    rendered = render_command(build_train_command(config))

    assert "env.num_workers=8" in rendered


def test_dino_wm_latent_smoke_exercises_full_run_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.delenv("DINO_WM_FEATURE_CACHE", raising=False)
    monkeypatch.delenv("DINO_NUM_WORKERS", raising=False)
    config = resolve_config(load_config("configs/dino_wm/smoke_pointmaze_latent.yaml"))
    validate_config(config)

    assert latent_training_enabled(config)
    rendered = render_command(build_train_command(config))

    # Same training shape as the full run, scaled down.
    assert "smoke_pointmaze_latent_seed0" in rendered
    assert "env.dataset.n_rollout=20" in rendered
    assert "training.epochs=1" in rendered
    assert "training.batch_size=4" in rendered
    assert "has_decoder=false" in rendered
    assert (
        "env.dataset._target_=wm_poc_latent_dataset.load_point_maze_latent_slice_train_val"
        in rendered
    )
    assert "+env.dataset.slice_stride=1" in rendered
    # Tiny planning eval stays enabled to crash-test the planner path.
    assert bool(config["planning"]["enabled"]) is True

    precompute = render_command(build_precompute_command(config))
    assert "--n-rollout 20" in precompute


def test_dino_wm_step_and_mixed_precision_patches_compose(tmp_path: Path) -> None:
    # run_train.sh applies the step-checkpoint patch first, then mixed
    # precision; both must land on the same train.py and the result must
    # still be valid Python.
    combined_source = _upstream_like_train_source().replace(
        "        self.cfg = None\n",
        '        self.accelerator = Accelerator(log_with="wandb")\n        self.cfg = None\n',
    )
    train_path = tmp_path / "train.py"
    train_path.write_text(combined_source, encoding="utf-8")

    assert patch_train_file(train_path)  # step checkpointing
    assert patch_mixed_precision_train_file(train_path)

    patched = train_path.read_text(encoding="utf-8")
    assert PATCH_MARKER in patched
    assert MIXED_PRECISION_PATCH_MARKER in patched
    assert "save_step_ckpt" in patched
    compile(patched, "train.py", "exec")


def test_dino_wm_latent_dataset_propagates_transform_for_planning() -> None:
    # plan.py builds Preprocessor(transform=dset.transform) to process raw env
    # renders; the latent dataset must store the image transform even though
    # it never applies it when serving cached latents.
    assert "transform=transform," in LATENT_DATASET_MODULE_SOURCE
    assert "Preprocessor" in LATENT_DATASET_MODULE_SOURCE  # the why, documented


def _upstream_like_val_source() -> str:
    return '''import torch
from tqdm import tqdm

class Trainer:
    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            self.model.train()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

    def val(self):
        self.model.eval()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            loss = self.accelerator.gather_for_metrics(loss).mean()
'''


def test_dino_wm_val_no_grad_patch_targets_only_validation(tmp_path: Path) -> None:
    from wm_poc.dino_wm.val_no_grad_patch import (
        PATCH_MARKER as VAL_NO_GRAD_MARKER,
        patch_train_file as patch_val_no_grad_file,
        patch_train_source as patch_val_no_grad_source,
    )

    train_path = tmp_path / "train.py"
    train_path.write_text(_upstream_like_val_source(), encoding="utf-8")

    changed = patch_val_no_grad_file(train_path)
    patched = train_path.read_text(encoding="utf-8")
    second, changed_again = patch_val_no_grad_source(patched)

    assert changed
    assert not changed_again
    assert second == patched
    assert VAL_NO_GRAD_MARKER in patched
    compile(patched, "train.py", "exec")
    # the validation forward is wrapped; the training forward is untouched
    assert "with torch.no_grad():" in patched
    train_section = patched.split("def val")[0]
    assert "with torch.no_grad():" not in train_section
    assert list((tmp_path / ".wm_poc_backups").glob("train.py.val_no_grad.*"))


def test_dino_wm_val_no_grad_patch_requires_anchor() -> None:
    from wm_poc.dino_wm.val_no_grad_patch import patch_train_source as patch_val_no_grad_source

    with pytest.raises(ValueError, match="val no-grad"):
        patch_val_no_grad_source("def val(self):\n    return None\n")


def test_dino_wm_lowdata_configs_train_nodecoder_on_latents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    for name in ("pointmaze_lowdata_scratch_a100.yaml", "pointmaze_lowdata_finetune_a100.yaml"):
        config = resolve_config(load_config(f"configs/dino_wm/{name}"))
        validate_config(config)
        assert latent_training_enabled(config), name
        rendered = render_command(build_train_command(config))
        assert "has_decoder=false" in rendered, name
        assert "model.train_decoder=false" in rendered, name
        assert "wm_poc_latent_dataset.load_point_maze_latent_slice_train_val" in rendered, name
        assert "env.dataset.n_rollout=360" in rendered, name


def test_dino_wm_decoder_scratch_config_stays_on_image_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    config = resolve_config(load_config("configs/dino_wm/pointmaze_scratch_a100.yaml"))
    validate_config(config)

    assert not latent_training_enabled(config)
    rendered = render_command(build_train_command(config))
    assert "wm_poc_latent_dataset" not in rendered
    # the no-decoder child must keep the latent path despite the parent
    child = resolve_config(load_config("configs/dino_wm/pointmaze_full_nodecoder_bf16.yaml"))
    assert latent_training_enabled(child)


def test_dino_wm_latent_training_with_decoder_fails_at_build_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    config = resolve_config(load_config("configs/dino_wm/pointmaze_full_nodecoder_bf16.yaml"))
    config["upstream"]["train_overrides"] = []  # decoder stays on

    with pytest.raises(ValueError, match="decoder"):
        build_train_command(config)


def test_dino_wm_finetune_command_appends_new_hydra_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.setenv("DINO_POINTMAZE_SOURCE_CKPT", "/tmp/source/model_latest.pth")
    config = resolve_config(load_config("configs/dino_wm/pointmaze_lowdata_finetune_a100.yaml"))
    validate_config(config)

    rendered = render_command(build_train_command(config))

    # Upstream's schema has no finetuning section, so every key must use ++.
    assert "++finetuning.enabled=true" in rendered
    assert "++finetuning.init_from=/tmp/source/model_latest.pth" in rendered
    assert "++finetuning.load_predictor=true" in rendered
    assert "++finetuning.load_action_encoder=true" in rendered
    assert "++finetuning.load_decoder=false" in rendered
    assert " finetuning.enabled=true" not in rendered
    # lr/epochs are mapped onto plain training.* keys, not finetuning.*
    assert "training.predictor_lr=1e-05" in rendered
    assert "training.epochs=20" in rendered
    assert "finetuning.predictor_lr" not in rendered
    assert "finetuning.epochs" not in rendered
    assert "finetuning.warmup_epochs_action_encoder_only" not in rendered


def test_dino_wm_finetune_init_patch_is_idempotent(tmp_path: Path) -> None:
    from wm_poc.dino_wm.finetune_init_patch import (
        PATCH_MARKER as FT_MARKER,
        patch_train_file as patch_ft_file,
        patch_train_source as patch_ft_source,
    )

    source = """import torch
from omegaconf import OmegaConf

class Trainer:
    def init_models(self):
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )

    def init_optimizers(self):
        pass
"""
    train_path = tmp_path / "train.py"
    train_path.write_text(source, encoding="utf-8")

    changed = patch_ft_file(train_path)
    patched = train_path.read_text(encoding="utf-8")
    second, changed_again = patch_ft_source(patched)

    assert changed
    assert not changed_again
    assert second == patched
    assert FT_MARKER in patched
    assert "_wm_poc_apply_finetune_init" in patched
    compile(patched, "train.py", "exec")
    assert list((tmp_path / ".wm_poc_backups").glob("train.py.finetune_init.*"))

    with pytest.raises(ValueError, match="fine-tune init"):
        patch_ft_source("def init_optimizers(self):\n    pass\n")


def test_dino_wm_plan_command_targets_checkpoint_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.delenv("DINO_PLANNER", raising=False)
    config = resolve_config(load_config("configs/dino_wm/pointmaze_scratch_a100.yaml"))
    validate_config(config)

    checkpoint = (
        "/drive/ckpts/outputs/pointmaze_full_nodecoder_t4_fp16_b32_stride2_seed0/"
        "checkpoints/model_latest.pth"
    )
    rendered = render_command(build_plan_command(config, checkpoint))

    # The checkpoint, not the config run_name, decides which model is loaded.
    assert "model_name=pointmaze_full_nodecoder_t4_fp16_b32_stride2_seed0" in rendered
    assert "ckpt_base_path=/drive/ckpts" in rendered
    assert "model_epoch=latest" in rendered
    assert "model_name=pointmaze_scratch_a100_seed0" not in rendered

    # Numbered epoch checkpoints resolve to their epoch.
    rendered_epoch = render_command(
        build_plan_command(config, "/drive/ckpts/outputs/some_run/checkpoints/model_7.pth")
    )
    assert "model_name=some_run" in rendered_epoch
    assert "model_epoch=7" in rendered_epoch

    # Without a checkpoint, the config-derived reference is unchanged.
    rendered_default = render_command(build_plan_command(config, ""))
    assert "model_name=pointmaze_scratch_a100_seed0" in rendered_default
    assert "model_epoch=latest" in rendered_default


def test_dino_wm_plan_command_rejects_unusable_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    config = resolve_config(load_config("configs/dino_wm/pointmaze_scratch_a100.yaml"))

    with pytest.raises(ValueError, match="rolling intra-epoch"):
        build_plan_command(
            config, "/drive/ckpts/outputs/run/checkpoints/model_latest_step.pth"
        )
    with pytest.raises(ValueError, match="outputs"):
        build_plan_command(config, "/somewhere/else/model_latest.pth")
    with pytest.raises(ValueError, match="model_<epoch>"):
        build_plan_command(config, "/drive/ckpts/outputs/run/checkpoints/weights.bin")


def test_dino_wm_precompute_skip_handles_clamped_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import json

    monkeypatch.setenv("DINO_WM_FEATURE_CACHE", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "precompute_latents_script", "scripts/dino_wm/precompute_latents.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    config = mod.resolve_config(mod.load_config("configs/dino_wm/pointmaze_full_nodecoder_t4.yaml"))
    cache = Path(mod.latent_cache_dir(config))
    cache.mkdir(parents=True)
    manifest = cache / "wm_poc_latent_manifest.json"

    # A 360-episode cache from a low-data session does not cover the full config.
    manifest.write_text(json.dumps({"num_episodes": 360}), encoding="utf-8")
    assert not mod._cache_covers_required(config)

    # The full config requests 2200 rollouts but the raw dataset has 2000
    # episodes; once all of them are encoded the cache is complete.
    manifest.write_text(
        json.dumps({"num_episodes": 2000, "dataset_episodes": 2000}), encoding="utf-8"
    )
    assert mod._cache_covers_required(config)

    manifest.write_text(json.dumps({"num_episodes": 2200}), encoding="utf-8")
    assert mod._cache_covers_required(config)

    # The precompute script records the dataset size for this check.
    assert '"dataset_episodes"' in PRECOMPUTE_SCRIPT_SOURCE


def test_dino_wm_plan_command_treats_blank_checkpoint_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unset checkpoint reaches build_plan_command as "" or "." (empty
    # string through argparse's Path type); both must fall back to the
    # config-derived model reference instead of being parsed as a path.
    monkeypatch.setenv("DINO_WM_REPO", "/tmp/dino_wm")
    monkeypatch.delenv("DINO_PLANNER", raising=False)
    config = resolve_config(load_config("configs/dino_wm/smoke_pointmaze_latent.yaml"))
    validate_config(config)

    for blank in ("", ".", " "):
        rendered = render_command(build_plan_command(config, blank))
        assert "model_name=smoke_pointmaze_latent_seed0" in rendered, repr(blank)
        assert "model_epoch=latest" in rendered, repr(blank)


def test_dino_wm_epoch_loss_series_parses_upstream_stdout(tmp_path: Path) -> None:
    from wm_poc.dino_wm.metrics import epoch_loss_series

    stdout = tmp_path / "stdout.log"
    stdout.write_text(
        "[2026-06-12 04:20:39,429][__main__][INFO] - dataloader batch size: 32\n"
        "[2026-06-12 04:31:02,101][__main__][INFO] - Epoch 1  Training loss: 0.0123          "
        "      Validation loss: 0.0456\n"
        "[2026-06-12 04:59:10,000][__main__][INFO] - Epoch 2  Training loss: 0.0100  "
        "        Validation loss: 0.0400\n"
        # a resumed run re-logs epoch 2 with the value that should win
        "[2026-06-12 05:20:00,000][__main__][INFO] - Epoch 2  Training loss: 0.0090  "
        "        Validation loss: 0.0390\n",
        encoding="utf-8",
    )

    series = epoch_loss_series(tmp_path)

    assert [record["epoch"] for record in series] == [1, 2]
    assert series[0]["val_loss"] == pytest.approx(0.0456)
    assert series[1]["train_loss"] == pytest.approx(0.0090)
    assert epoch_loss_series(tmp_path / "missing") == []


def test_dino_wm_training_curves_fall_back_to_stdout(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from wm_poc.dino_wm.visualization import plot_training_loss_curves

    run_dir = tmp_path / "pointmaze_run"
    run_dir.mkdir()
    (run_dir / "stdout.log").write_text(
        "[ts][__main__][INFO] - Epoch 1  Training loss: 0.0123        Validation loss: 0.0456\n"
        "[ts][__main__][INFO] - Epoch 2  Training loss: 0.0100        Validation loss: 0.0400\n",
        encoding="utf-8",
    )
    out = tmp_path / "curves.png"

    plot_training_loss_curves([run_dir], out)

    assert out.is_file() and out.stat().st_size > 0


def test_dino_wm_epoch_loss_series_merges_rotated_logs(tmp_path: Path) -> None:
    from wm_poc.dino_wm.metrics import epoch_loss_series
    import os as _os

    line = "[ts][__main__][INFO] - Epoch {e}  Training loss: {t}        Validation loss: {v}\n"
    old = tmp_path / "stdout.log.20260611T010101Z.prev"
    old.write_text(line.format(e=1, t=0.05, v=0.09) + line.format(e=2, t=0.04, v=0.08), encoding="utf-8")
    new = tmp_path / "stdout.log"
    # the resumed attempt re-logs epoch 2 with the value that should win
    new.write_text(line.format(e=2, t=0.03, v=0.07) + line.format(e=3, t=0.02, v=0.06), encoding="utf-8")
    _os.utime(old, (1000, 1000))
    _os.utime(new, (2000, 2000))

    series = epoch_loss_series(tmp_path)

    assert [record["epoch"] for record in series] == [1, 2, 3]
    assert series[0]["val_loss"] == pytest.approx(0.09)
    assert series[1]["val_loss"] == pytest.approx(0.07)
    assert series[2]["val_loss"] == pytest.approx(0.06)


def test_dino_wm_summarize_run_reads_pipeline_artifacts(tmp_path: Path) -> None:
    from wm_poc.dino_wm.metrics import summarize_run

    run_dir = tmp_path / "pointmaze_full_nodecoder_t4_fp16_b32_stride2_seed0"
    (run_dir / "planning" / "cem").mkdir(parents=True)
    (run_dir / "status.json").write_text(
        '{"completed": true, "elapsed_seconds": 1800}', encoding="utf-8"
    )
    (run_dir / "stdout.log").write_text(
        "[ts][__main__][INFO] - Epoch 1  Training loss: 0.0123        Validation loss: 0.0456\n"
        "[ts][__main__][INFO] - Epoch 2  Training loss: 0.0100        Validation loss: 0.0400\n"
        "[ts][__main__][INFO] - Epoch 3  Training loss: 0.0095        Validation loss: 0.0410\n",
        encoding="utf-8",
    )
    # upstream plan.py appends JSONL entries with final_eval/ prefixes
    (run_dir / "planning" / "cem" / "logs.json").write_text(
        '{"final_eval/success_rate": 0.42, "final_eval/mean_visual_dist": 1.5}\n',
        encoding="utf-8",
    )
    (run_dir / "planning" / "status_cem.json").write_text(
        '{"completed": true, "elapsed_seconds": 1200}', encoding="utf-8"
    )

    row = summarize_run(run_dir)

    assert row["completed"] is True
    assert row["train_wall_minutes"] == pytest.approx(30.0)
    assert row["final_val_loss_pred_hstep"] == pytest.approx(0.0410)  # last epoch
    assert row["best_epoch"] == 2  # lowest validation loss
    assert row["best_success_rate"] == pytest.approx(0.42)
    assert row["plan_wall_minutes"] == pytest.approx(20.0)


def test_dino_wm_summarize_run_uses_latest_replan_not_max(tmp_path: Path) -> None:
    """A re-plan appends a new final_eval row to the same logs.json; the summary
    must reflect the *latest* run, not the max across reruns. Regression for the
    50->200 eval staleness bug where an older 50-eval estimate (0.34) masked the
    current 200-eval result (0.275)."""
    from wm_poc.dino_wm.metrics import summarize_run

    run_dir = tmp_path / "pointmaze_lowdata_finetune_a100_seed0"
    (run_dir / "planning" / "cem").mkdir(parents=True)
    (run_dir / "status.json").write_text(
        '{"completed": true, "elapsed_seconds": 600}', encoding="utf-8"
    )
    # First a 50-eval run (higher, noisier), then a 200-eval re-plan appended.
    (run_dir / "planning" / "cem" / "logs.json").write_text(
        '{"plan_0/success_rate": 0.30, "step": 10}\n'
        '{"final_eval/success_rate": 0.34, "final_eval/mean_visual_dist": 0.55}\n'
        '{"plan_0/success_rate": 0.27, "step": 10}\n'
        '{"final_eval/success_rate": 0.275, "final_eval/mean_visual_dist": 0.55}\n',
        encoding="utf-8",
    )

    row = summarize_run(run_dir)

    # latest final_eval, NOT max(0.34, 0.275)
    assert row["best_success_rate"] == pytest.approx(0.275)


def test_dino_wm_chart_rows_exclude_diagnostics_and_label_runs() -> None:
    from wm_poc.dino_wm.visualization import (
        display_label,
        prepare_planning_rows,
        prepare_scratch_finetune_rows,
    )

    rows = [
        {"run_name": "pointmaze_full_nodecoder_t4_fp16_b32_stride2_seed0", "mode": "scratch",
         "best_success_rate": "0.26", "final_val_loss_pred_hstep": ""},
        {"run_name": "pointmaze_lowdata_finetune_a100_seed0", "mode": "finetune",
         "best_success_rate": "0.34", "final_val_loss_pred_hstep": "0.0142"},
        {"run_name": "pointmaze_lowdata_scratch_a100_seed0", "mode": "scratch",
         "best_success_rate": "0.12", "final_val_loss_pred_hstep": "0.0231"},
        {"run_name": "pointmaze_oom_safe_seed0", "mode": "scratch",
         "best_success_rate": "", "final_val_loss_pred_hstep": "1.4"},
        {"run_name": "smoke_pointmaze_latent_seed0", "mode": "smoke",
         "best_success_rate": "0.0", "final_val_loss_pred_hstep": "0.5"},
    ]

    planning = prepare_planning_rows(rows)
    assert [label for label, _ in planning] == [
        "low-data scratch", "full data (T4, stride-2)", "low-data fine-tune",
    ]  # sorted ascending by success rate; smoke/diagnostics dropped

    comparison = prepare_scratch_finetune_rows(rows)
    assert [(label, mode) for label, _, mode in comparison] == [
        ("low-data fine-tune", "finetune"), ("low-data scratch", "scratch"),
    ]  # OOM diagnostic and smoke excluded, missing-loss rows dropped

    assert display_label("pointmaze_full_nodecoder_bf16_a100_b32_seed0") == "full data"
    assert display_label("custom_run_seed3") == "custom run"


def test_dino_wm_evaluator_video_patch_records_without_decoder(tmp_path: Path) -> None:
    from wm_poc.dino_wm.evaluator_video_patch import (
        PATCH_MARKER as VIDEO_MARKER,
        patch_evaluator_file,
        patch_evaluator_source,
    )

    source = '''import torch

class PlanEvaluator:
    def eval_actions(self, actions, action_len=None, filename="output", save_video=False):
        # plot trajs
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes
'''
    path = tmp_path / "evaluator.py"
    path.write_text(source, encoding="utf-8")

    changed = patch_evaluator_file(path)
    patched = path.read_text(encoding="utf-8")
    second, changed_again = patch_evaluator_source(patched)

    assert changed
    assert not changed_again
    assert second == patched
    assert VIDEO_MARKER in patched
    assert "torch.zeros" in patched  # honest black imagined panel
    assert patched.count("_plot_rollout_compare") == 2
    compile(patched, "evaluator.py", "exec")
    assert list((tmp_path / ".wm_poc_backups").glob("evaluator.py.video_no_decoder.*"))

    with pytest.raises(ValueError, match="evaluator video"):
        patch_evaluator_source("def eval_actions(self):\n    pass\n")


def test_dino_wm_run_state_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from wm_poc.dino_wm.resume import latest_epoch_checkpoint, planning_complete, training_complete

    monkeypatch.setenv("DINO_CKPT_ROOT", str(tmp_path / "ckpts"))
    monkeypatch.setenv("DINO_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.delenv("DINO_PLANNER", raising=False)
    config = resolve_config(load_config("configs/dino_wm/pointmaze_lowdata_scratch_a100.yaml"))
    run_name = str(config["run_name"])

    ckpt_dir = tmp_path / "ckpts" / "outputs" / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    assert latest_epoch_checkpoint(config) == 0
    assert not training_complete(config)

    (ckpt_dir / "model_12.pth").write_text("x", encoding="utf-8")
    (ckpt_dir / "model_latest.pth").write_text("x", encoding="utf-8")
    assert latest_epoch_checkpoint(config) == 12
    assert not training_complete(config)  # config trains 30 epochs

    (ckpt_dir / "model_30.pth").write_text("x", encoding="utf-8")
    assert training_complete(config)

    # the fine-tune config gates on finetuning.epochs instead
    monkeypatch.setenv("DINO_POINTMAZE_SOURCE_CKPT", "/x/model_latest.pth")
    ft = resolve_config(load_config("configs/dino_wm/pointmaze_lowdata_finetune_a100.yaml"))
    ft_dir = tmp_path / "ckpts" / "outputs" / str(ft["run_name"]) / "checkpoints"
    ft_dir.mkdir(parents=True)
    (ft_dir / "model_20.pth").write_text("x", encoding="utf-8")
    assert training_complete(ft)

    # planning gate: needs a completed status AND a command matching n_evals=200
    plan_dir = tmp_path / "logs" / run_name / "planning"
    plan_dir.mkdir(parents=True)
    assert not planning_complete(config)
    (plan_dir / "command_cem.sh").write_text("plan.py n_evals=50 planner=cem", encoding="utf-8")
    (plan_dir / "status_cem.json").write_text(json.dumps({"completed": True}), encoding="utf-8")
    assert not planning_complete(config)  # stale 50-eval run does not count
    (plan_dir / "command_cem.sh").write_text("plan.py n_evals=200 planner=cem", encoding="utf-8")
    assert planning_complete(config)
    (plan_dir / "status_cem.json").write_text(json.dumps({"completed": False, "failed": True}), encoding="utf-8")
    assert not planning_complete(config)
