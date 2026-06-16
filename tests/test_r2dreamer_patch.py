from pathlib import Path

from wm_poc.r2dreamer.patching import (
    DMC_RENDER_PATCH_MARKER,
    PATCH_MARKER,
    SAVE_FINALLY_PATTERN,
    SERIAL_ENV_PATCH_MARKER,
    TRAINER_CHECKPOINT_PATCH_MARKER,
    TRAINER_PROGRESS_PATCH_MARKER,
    patch_dmc_rendering,
    patch_serial_envs,
    patch_train_py,
    patch_trainer_interval_checkpoints,
    verify_dmc_render_patch,
    verify_patch,
    verify_serial_env_patch,
    verify_trainer_checkpoint_patch,
)


FAKE_TRAIN = '''import pathlib
import torch
import tools

def main(config):
    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")
'''

FAKE_DMC = '''import gymnasium as gym
import numpy as np


class DeepMindControl(gym.Env):
    metadata = {}

    def __init__(self):
        self._size = (64, 64)
        self._camera = 0

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
'''

FAKE_TRAINER = '''import torch

import tools


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat

    def eval(self, agent, train_step):
        pass

    def begin(self, agent):
        step = 0
        while step < self.steps:
            if self._should_eval(step) and self.eval_episode_num > 0 and self.eval_envs is not None:
                self.eval(agent, step)
            step += 1
'''

FAKE_ENVS_INIT = '''import atexit

from . import parallel, wrappers


def make_envs(config):
    suite = config.task.split("_", 1)[0]

    if suite == "isaaclab":
        return _make_isaaclab_envs(config)

    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = (
        parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
        if config.eval_episode_num > 0
        else None
    )
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space
'''


def test_patch_train_py_is_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "r2dreamer"
    repo.mkdir()
    train_py = repo / "train.py"
    train_py.write_text(FAKE_TRAIN, encoding="utf-8")

    assert patch_train_py(repo) == "patched"
    patched_once = train_py.read_text(encoding="utf-8")
    assert PATCH_MARKER in patched_once
    assert "wm_poc_meta" in patched_once
    assert "finally:" not in patched_once
    assert (repo / "train.py.before_wm_poc_checkpoint_patch").is_file()

    assert patch_train_py(repo) == "already_patched"
    assert train_py.read_text(encoding="utf-8") == patched_once
    verify_patch(repo)


def test_patch_train_py_updates_old_finally_save(tmp_path: Path) -> None:
    repo = tmp_path / "r2dreamer"
    repo.mkdir()
    train_py = repo / "train.py"
    train_py.write_text(f"# {PATCH_MARKER}\n{SAVE_FINALLY_PATTERN}", encoding="utf-8")

    assert patch_train_py(repo) == "updated_save_on_success_only"
    patched = train_py.read_text(encoding="utf-8")
    assert "finally:" not in patched
    assert "wm_poc_meta" in patched
    assert (repo / "train.py.before_wm_poc_checkpoint_patch").is_file()


def test_patch_dmc_rendering_is_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "r2dreamer"
    envs = repo / "envs"
    envs.mkdir(parents=True)
    dmc_py = envs / "dmc.py"
    dmc_py.write_text(FAKE_DMC, encoding="utf-8")

    assert patch_dmc_rendering(repo) == "patched"
    patched_once = dmc_py.read_text(encoding="utf-8")
    assert DMC_RENDER_PATCH_MARKER in patched_once
    assert "WM_POC_DMC_DISABLE_IMAGE_RENDER" in patched_once
    assert "np.zeros(tuple(self._size) + (3,), dtype=np.uint8)" in patched_once
    assert (envs / "dmc.py.before_wm_poc_dmc_render_patch").is_file()

    assert patch_dmc_rendering(repo) == "already_patched"
    assert dmc_py.read_text(encoding="utf-8") == patched_once
    verify_dmc_render_patch(repo)


def test_patch_trainer_interval_checkpoints_is_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "r2dreamer"
    repo.mkdir()
    trainer_py = repo / "trainer.py"
    trainer_py.write_text(FAKE_TRAINER, encoding="utf-8")

    assert patch_trainer_interval_checkpoints(repo) == "patched"
    patched_once = trainer_py.read_text(encoding="utf-8")
    assert TRAINER_CHECKPOINT_PATCH_MARKER in patched_once
    assert TRAINER_PROGRESS_PATCH_MARKER in patched_once
    assert "checkpoint_every" in patched_once
    assert "checkpoint_keep" in patched_once
    assert "progress_every" in patched_once
    assert "_log_progress(step)" in patched_once
    assert '"includes_optimizer": False' in patched_once
    assert "_save_interval_checkpoint(agent, step)" in patched_once
    assert (repo / "trainer.py.before_wm_poc_interval_checkpoint_patch").is_file()

    assert patch_trainer_interval_checkpoints(repo) == "already_patched"
    assert trainer_py.read_text(encoding="utf-8") == patched_once
    verify_trainer_checkpoint_patch(repo)


def test_patch_trainer_interval_checkpoints_upgrades_progress_heartbeat(tmp_path: Path) -> None:
    repo = tmp_path / "r2dreamer"
    repo.mkdir()
    trainer_py = repo / "trainer.py"
    trainer_py.write_text(FAKE_TRAINER, encoding="utf-8")

    assert patch_trainer_interval_checkpoints(repo) == "patched"
    old_patched = trainer_py.read_text(encoding="utf-8")
    old_patched = old_patched.replace("import os\n", "")
    old_patched = old_patched.replace("import time\n", "")
    old_patched = old_patched.replace(
        '        self.progress_every = int(config.get("progress_every", os.environ.get("WM_POC_R2_PROGRESS_EVERY", 100)))\n'
        "        self._wm_poc_start_time = time.time()\n"
        "        self._wm_poc_last_progress_step = -1\n",
        "",
    )
    start = old_patched.index(f"    # {TRAINER_PROGRESS_PATCH_MARKER}")
    end = old_patched.index("    # BEGIN WM_POC_INTERVAL_CHECKPOINTS", start)
    old_patched = old_patched[:start] + old_patched[end:]
    old_patched = old_patched.replace("            self._log_progress(step)\n", "")
    trainer_py.write_text(old_patched, encoding="utf-8")

    assert patch_trainer_interval_checkpoints(repo) == "updated_progress_heartbeat"
    upgraded = trainer_py.read_text(encoding="utf-8")
    assert TRAINER_PROGRESS_PATCH_MARKER in upgraded
    assert upgraded.count(TRAINER_CHECKPOINT_PATCH_MARKER) == 1
    assert "progress_every" in upgraded
    assert "_log_progress(step)" in upgraded
    verify_trainer_checkpoint_patch(repo)


def test_patch_serial_envs_is_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "r2dreamer"
    envs = repo / "envs"
    envs.mkdir(parents=True)
    init_py = envs / "__init__.py"
    init_py.write_text(FAKE_ENVS_INIT, encoding="utf-8")

    assert patch_serial_envs(repo) == "patched"
    patched_once = init_py.read_text(encoding="utf-8")
    assert SERIAL_ENV_PATCH_MARKER in patched_once
    assert "WM_POC_R2_SERIAL_ENVS" in patched_once
    assert "WMPOCSerialEnv" in patched_once
    assert "_wm_poc_make_env_group(env_constructor, config.env_num, config.device)" in patched_once
    assert (envs / "__init__.py.before_wm_poc_serial_env_patch").is_file()

    assert patch_serial_envs(repo) == "already_patched"
    assert init_py.read_text(encoding="utf-8") == patched_once
    verify_serial_env_patch(repo)
