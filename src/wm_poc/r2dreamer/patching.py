from __future__ import annotations

import difflib
import py_compile
import shutil
from pathlib import Path


PATCH_MARKER = "BEGIN WM_POC_CHECKPOINT_LOADING"
BACKUP_SUFFIX = ".before_wm_poc_checkpoint_patch"
DMC_RENDER_PATCH_MARKER = "BEGIN WM_POC_DMC_RENDER_GUARD"
DMC_RENDER_BACKUP_SUFFIX = ".before_wm_poc_dmc_render_patch"
TRAINER_CHECKPOINT_PATCH_MARKER = "BEGIN WM_POC_INTERVAL_CHECKPOINTS"
TRAINER_PROGRESS_PATCH_MARKER = "BEGIN WM_POC_PROGRESS_HEARTBEAT"
TRAINER_RESUME_PATCH_MARKER = "BEGIN WM_POC_RESUME_CHECKPOINT"
TRAINER_BACKUP_SUFFIX = ".before_wm_poc_interval_checkpoint_patch"
SERIAL_ENV_PATCH_MARKER = "BEGIN WM_POC_SERIAL_ENVS"
SERIAL_ENV_BACKUP_SUFFIX = ".before_wm_poc_serial_env_patch"

AGENT_ANCHOR = """    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)
"""

CHECKPOINT_LOADING_BLOCK = """
    # BEGIN WM_POC_CHECKPOINT_LOADING
    def _wm_poc_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    pretrained = config.get("pretrained", None)
    pretrained_strict = _wm_poc_bool(config.get("pretrained_strict", True), default=True)
    load_optimizer = _wm_poc_bool(config.get("load_optimizer", False), default=False)
    resume_enabled = _wm_poc_bool(config.get("resume", True), default=True)

    # Mid-run resume: continue the SAME run from its rolling checkpoint when one
    # exists. This takes priority over `pretrained` so an interrupted source or
    # fine-tune run picks up where it stopped (model + optimizer + step) instead
    # of restarting. Delete logdir/latest.pt (or set +resume=false) to force a
    # clean restart.
    agent._wm_poc_resume_step = 0
    wm_poc_resume_path = pathlib.Path(logdir) / "latest.pt"
    if resume_enabled and wm_poc_resume_path.is_file():
        print(f"[wm_poc] Resuming run from {wm_poc_resume_path}")
        resume_ckpt = torch.load(wm_poc_resume_path, map_location=config.device)
        if "agent_state_dict" not in resume_ckpt:
            raise KeyError(
                f"Resume checkpoint {wm_poc_resume_path} has no 'agent_state_dict'. "
                f"Available keys: {list(resume_ckpt.keys())}"
            )
        agent.load_state_dict(resume_ckpt["agent_state_dict"], strict=False)
        if "optims_state_dict" in resume_ckpt:
            tools.recursively_load_optim_state_dict(agent, resume_ckpt["optims_state_dict"])
            print("[wm_poc] Restored optimizer state for resume.")
        else:
            print("[wm_poc] No optimizer state in resume checkpoint; optimizer warm-restarts.")
        agent._wm_poc_resume_step = int(resume_ckpt.get("wm_poc_meta", {}).get("step", 0))
        print(f"[wm_poc] Resuming at step {agent._wm_poc_resume_step}.")
    elif pretrained:
        pretrained_path = pathlib.Path(str(pretrained)).expanduser()
        print(f"[wm_poc] Loading pretrained checkpoint: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=config.device)

        if "agent_state_dict" not in ckpt:
            raise KeyError(
                f"Checkpoint {pretrained_path} does not contain 'agent_state_dict'. "
                f"Available keys: {list(ckpt.keys())}"
            )

        load_result = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
        if hasattr(load_result, "missing_keys"):
            missing = list(load_result.missing_keys)
            unexpected = list(load_result.unexpected_keys)
        else:
            missing = list(load_result[0])
            unexpected = list(load_result[1])
        print(f"[wm_poc] Loaded agent weights from {pretrained_path}")
        print(f"[wm_poc] Missing keys: {len(missing)}")
        print(f"[wm_poc] Unexpected keys: {len(unexpected)}")
        if pretrained_strict and (missing or unexpected):
            raise RuntimeError(
                "[wm_poc] Strict checkpoint loading failed with "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys."
            )

        if load_optimizer:
            if "optims_state_dict" not in ckpt:
                msg = f"[wm_poc] No optimizer state found in {pretrained_path}"
                if pretrained_strict:
                    raise KeyError(msg)
                print(msg)
            else:
                tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])
                print("[wm_poc] Loaded optimizer states.")
    # END WM_POC_CHECKPOINT_LOADING
"""

SAVE_PATTERN = """    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")
"""

SAVE_FINALLY_PATTERN = """    try:
        policy_trainer.begin(agent)
    finally:
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "wm_poc_meta": {
                "pretrained": str(config.get("pretrained", None)),
                "pretrained_strict": pretrained_strict,
                "load_optimizer": load_optimizer,
                "env": str(config.env),
                "model": str(config.model),
                "seed": str(config.seed),
            },
        }
        torch.save(items_to_save, logdir / "latest.pt")
        print(f"[wm_poc] Saved checkpoint to {logdir / 'latest.pt'}")
"""

SAVE_REPLACEMENT = """    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "wm_poc_meta": {
            "pretrained": str(config.get("pretrained", None)),
            "pretrained_strict": pretrained_strict,
            "load_optimizer": load_optimizer,
            "step": int(getattr(policy_trainer, "steps", 0)),
            "env": str(config.env),
            "model": str(config.model),
            "seed": str(config.seed),
        },
    }
    torch.save(items_to_save, logdir / "latest.pt")
    print(f"[wm_poc] Saved checkpoint to {logdir / 'latest.pt'}")
"""

DMC_IMPORT_ANCHOR = """import gymnasium as gym
import numpy as np
"""

DMC_IMPORT_REPLACEMENT = """import os

import gymnasium as gym
import numpy as np


# BEGIN WM_POC_DMC_RENDER_GUARD
def _wm_poc_bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _wm_poc_disable_dmc_image_render():
    return _wm_poc_bool_env("WM_POC_DMC_DISABLE_IMAGE_RENDER", default=False)
# END WM_POC_DMC_RENDER_GUARD
"""

DMC_RENDER_PATTERN = """    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
"""

DMC_RENDER_REPLACEMENT = """    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        if _wm_poc_disable_dmc_image_render():
            return np.zeros(tuple(self._size) + (3,), dtype=np.uint8)
        return self._env.physics.render(*self._size, camera_id=self._camera)
"""

TRAINER_IMPORT_ANCHOR = """import torch

import tools
"""

TRAINER_IMPORT_REPLACEMENT = """import os
import pathlib
import time

import torch

import tools
"""

TRAINER_INIT_ANCHOR = """        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
"""

TRAINER_INIT_REPLACEMENT = """        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.logdir = pathlib.Path(logdir)
        self.checkpoint_every = int(config.get("checkpoint_every", 0))
        self.checkpoint_keep = int(config.get("checkpoint_keep", 0))
        self._wm_poc_saved_checkpoints = []
        self._wm_poc_last_checkpoint_step = 0
        self.progress_every = int(config.get("progress_every", os.environ.get("WM_POC_R2_PROGRESS_EVERY", 100)))
        self._wm_poc_start_time = time.time()
        self._wm_poc_last_progress_step = -1
"""

TRAINER_METHOD_ANCHOR = """    def eval(self, agent, train_step):
"""

TRAINER_PROGRESS_METHOD_INSERT = """    # BEGIN WM_POC_PROGRESS_HEARTBEAT
    def _log_progress(self, step):
        step = int(step)
        total = max(int(self.steps), 1)
        if self.progress_every <= 0:
            return
        if (
            self._wm_poc_last_progress_step >= 0
            and step - self._wm_poc_last_progress_step < self.progress_every
            and step < total
        ):
            return
        self._wm_poc_last_progress_step = step
        elapsed = max(time.time() - self._wm_poc_start_time, 1e-6)
        pct = 100.0 * min(step, total) / total
        rate = step / elapsed if step > 0 else 0.0
        width = 24
        filled = int(width * min(max(pct, 0.0), 100.0) / 100.0)
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"[wm_poc] progress [{bar}] {step:09d}/{total:09d} "
            f"({pct:5.1f}%) elapsed={elapsed / 60.0:.1f}m rate={rate:.1f} steps/s",
            flush=True,
        )
    # END WM_POC_PROGRESS_HEARTBEAT

"""

TRAINER_RESUME_METHOD_INSERT = """    # BEGIN WM_POC_RESUME_CHECKPOINT
    def _save_resume_checkpoint(self, agent, step):
        step = int(step)
        if self.checkpoint_every <= 0 or step <= 0:
            return
        self.logdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "wm_poc_meta": {
                "kind": "resume",
                "step": step,
                "includes_optimizer": True,
            },
        }
        tmp_path = self.logdir / "latest.pt.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, self.logdir / "latest.pt")
        print(f"[wm_poc] Saved resume checkpoint to {self.logdir / 'latest.pt'} (step {step})", flush=True)
    # END WM_POC_RESUME_CHECKPOINT

"""

# Resume the same run from its rolling latest.pt: start the loop at the saved
# step instead of 0. train.py stores the step on the agent before begin().
TRAINER_RESUME_EVAL_LINE = "                self._save_resume_checkpoint(agent, step)\n"
TRAINER_RESUME_LOOP_PATTERN = "        while step < self.steps:\n"
TRAINER_RESUME_LOOP_REPLACEMENT = (
    '        step = getattr(agent, "_wm_poc_resume_step", step)\n'
    "        while step < self.steps:\n"
)

TRAINER_METHOD_INSERT = TRAINER_PROGRESS_METHOD_INSERT + """    # BEGIN WM_POC_INTERVAL_CHECKPOINTS
    def _save_interval_checkpoint(self, agent, step):
        step = int(step)
        if self.checkpoint_every <= 0 or step <= 0:
            return
        if step - self._wm_poc_last_checkpoint_step < self.checkpoint_every:
            return
        checkpoint_dir = self.logdir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"step_{step:09d}.pt"
        if path in self._wm_poc_saved_checkpoints:
            return
        payload = {
            "agent_state_dict": agent.state_dict(),
            "wm_poc_meta": {
                "kind": "interval",
                "step": step,
                "checkpoint_every": self.checkpoint_every,
                "checkpoint_keep": self.checkpoint_keep,
                "includes_optimizer": False,
            },
        }
        torch.save(payload, path)
        self._wm_poc_saved_checkpoints.append(path)
        self._wm_poc_last_checkpoint_step = step
        print(f"[wm_poc] Saved interval checkpoint to {path}")
        if self.checkpoint_keep > 0:
            checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
            for old in checkpoints[:-self.checkpoint_keep]:
                try:
                    old.unlink()
                except FileNotFoundError:
                    pass
    # END WM_POC_INTERVAL_CHECKPOINTS

""" + TRAINER_RESUME_METHOD_INSERT + """    def eval(self, agent, train_step):
"""

TRAINER_LOOP_PATTERN = """        while step < self.steps:
"""

TRAINER_LOOP_REPLACEMENT = """        while step < self.steps:
            self._log_progress(step)
"""

TRAINER_EVAL_PATTERN = """            if self._should_eval(step) and self.eval_episode_num > 0 and self.eval_envs is not None:
                self.eval(agent, step)
"""

TRAINER_EVAL_REPLACEMENT = """            if self._should_eval(step) and self.eval_episode_num > 0 and self.eval_envs is not None:
                self.eval(agent, step)
                self._save_interval_checkpoint(agent, step)
                self._save_resume_checkpoint(agent, step)
"""

SERIAL_ENV_IMPORT_ANCHOR = """import atexit

from . import parallel, wrappers
"""

SERIAL_ENV_IMPORT_REPLACEMENT = """import atexit
import os

import numpy as np
import torch
from tensordict import TensorDict

import tools
from . import parallel, wrappers


# BEGIN WM_POC_SERIAL_ENVS
def _wm_poc_bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _wm_poc_use_serial_envs():
    return _wm_poc_bool_env("WM_POC_R2_SERIAL_ENVS", default=False)


class WMPOCSerialEnv:
    def __init__(self, constructor, env_num, device):
        self.envs = [constructor(i)() for i in range(env_num)]
        self.device = device

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def env_num(self):
        return len(self.envs)

    def lift_dim(self, td):
        for key in td.keys():
            if td[key].ndim == 1:
                td[key] = td[key].unsqueeze(-1)
        return td

    def step(self, action, done):
        action_np = tools.to_np(action)
        done_cpu = done.detach().cpu() if isinstance(done, torch.Tensor) else done
        new_o, new_r, new_d = [], [], []
        for env, a, d in zip(self.envs, action_np, done_cpu):
            if bool(d):
                new_o.append(env.reset())
                new_r.append(0.0)
                new_d.append(False)
            else:
                o, r, d, _ = env.step(a)
                new_o.append(o)
                new_r.append(r)
                new_d.append(d)
        obs_stacked = {k: np.stack([o[k] for o in new_o]) for k in new_o[0].keys()}
        obs_tensors = {k: torch.as_tensor(v, device="cpu") for k, v in obs_stacked.items()}
        rew_stacked = torch.as_tensor(new_r, dtype=torch.float32, device="cpu")
        td = TensorDict({**obs_tensors, "reward": rew_stacked}, batch_size=(self.env_num,), device="cpu")
        if torch.cuda.is_available():
            td = td.pin_memory()
        done = torch.as_tensor(new_d, device="cpu")
        return self.lift_dim(td), done

    def close(self):
        for env in self.envs:
            close = getattr(env, "close", None)
            if close is not None:
                close()


def _wm_poc_make_env_group(constructor, env_num, device):
    if _wm_poc_use_serial_envs():
        print(f"[wm_poc] Using serial envs for {env_num} DMC worker(s).", flush=True)
        return WMPOCSerialEnv(constructor, env_num, device)
    return parallel.ParallelEnv(constructor, env_num, device)
# END WM_POC_SERIAL_ENVS
"""

SERIAL_ENV_PATTERN = """    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = (
        parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
        if config.eval_episode_num > 0
        else None
    )
"""

SERIAL_ENV_REPLACEMENT = """    train_envs = _wm_poc_make_env_group(env_constructor, config.env_num, config.device)
    eval_envs = (
        _wm_poc_make_env_group(env_constructor, config.eval_episode_num, config.device)
        if config.eval_episode_num > 0
        else None
    )
"""


def train_py_path(r2_repo: Path) -> Path:
    return r2_repo.expanduser() / "train.py"


def dmc_py_path(r2_repo: Path) -> Path:
    return r2_repo.expanduser() / "envs" / "dmc.py"


def trainer_py_path(r2_repo: Path) -> Path:
    return r2_repo.expanduser() / "trainer.py"


def envs_init_py_path(r2_repo: Path) -> Path:
    return r2_repo.expanduser() / "envs" / "__init__.py"


def patch_train_py(r2_repo: Path) -> str:
    train_py = train_py_path(r2_repo)
    if not train_py.is_file():
        raise FileNotFoundError(f"Missing r2dreamer train.py: {train_py}")

    text = train_py.read_text(encoding="utf-8")
    upgrade = False
    if PATCH_MARKER in text:
        if SAVE_FINALLY_PATTERN in text:
            backup = train_py.with_name(train_py.name + BACKUP_SUFFIX)
            if not backup.exists():
                shutil.copy2(train_py, backup)
            text = text.replace(SAVE_FINALLY_PATTERN, SAVE_REPLACEMENT, 1)
            train_py.write_text(text, encoding="utf-8")
            return "updated_save_on_success_only"
        if "_wm_poc_resume_step" in text:
            return "already_patched"
        # Stale patch predating mid-run resume: re-apply the current patch from
        # the pristine backup so the resume load/save logic is added.
        backup = train_py.with_name(train_py.name + BACKUP_SUFFIX)
        if not backup.is_file():
            return "already_patched"
        shutil.copy2(backup, train_py)
        text = train_py.read_text(encoding="utf-8")
        upgrade = True

    if AGENT_ANCHOR not in text:
        raise RuntimeError("Could not find Dreamer agent creation anchor in train.py.")
    if SAVE_PATTERN not in text:
        raise RuntimeError("Could not find latest.pt save block in train.py.")

    backup = train_py.with_name(train_py.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(train_py, backup)

    text = text.replace(AGENT_ANCHOR, AGENT_ANCHOR + CHECKPOINT_LOADING_BLOCK, 1)
    text = text.replace(SAVE_PATTERN, SAVE_REPLACEMENT, 1)
    train_py.write_text(text, encoding="utf-8")
    return "updated_resume_support" if upgrade else "patched"


def patch_dmc_rendering(r2_repo: Path) -> str:
    dmc_py = dmc_py_path(r2_repo)
    if not dmc_py.is_file():
        raise FileNotFoundError(f"Missing r2dreamer envs/dmc.py: {dmc_py}")

    text = dmc_py.read_text(encoding="utf-8")
    if DMC_RENDER_PATCH_MARKER in text:
        return "already_patched"

    if DMC_IMPORT_ANCHOR not in text:
        raise RuntimeError("Could not find import anchor in envs/dmc.py.")
    if DMC_RENDER_PATTERN not in text:
        raise RuntimeError("Could not find render block in envs/dmc.py.")

    backup = dmc_py.with_name(dmc_py.name + DMC_RENDER_BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(dmc_py, backup)

    text = text.replace(DMC_IMPORT_ANCHOR, DMC_IMPORT_REPLACEMENT, 1)
    text = text.replace(DMC_RENDER_PATTERN, DMC_RENDER_REPLACEMENT, 1)
    dmc_py.write_text(text, encoding="utf-8")
    return "patched"


def _insert_trainer_resume(text: str) -> str:
    """Idempotently add the mid-run resume save/load wiring to a patched trainer.py.

    The fresh patch already gets the resume method and save call from
    TRAINER_METHOD_INSERT / TRAINER_EVAL_REPLACEMENT, so for that path this only
    adds the loop start line. For an older patched checkout that predates resume,
    all three pieces are inserted.
    """
    if TRAINER_RESUME_PATCH_MARKER not in text:
        eval_anchor = "    def eval(self, agent, train_step):\n"
        if eval_anchor not in text:
            raise RuntimeError("Could not find trainer.py eval anchor for resume checkpoint.")
        text = text.replace(eval_anchor, TRAINER_RESUME_METHOD_INSERT + eval_anchor, 1)
    if TRAINER_RESUME_EVAL_LINE not in text:
        interval_call = "                self._save_interval_checkpoint(agent, step)\n"
        if interval_call not in text:
            raise RuntimeError("Could not find trainer.py interval-save call for resume checkpoint.")
        text = text.replace(interval_call, interval_call + TRAINER_RESUME_EVAL_LINE, 1)
    if 'getattr(agent, "_wm_poc_resume_step"' not in text:
        if TRAINER_RESUME_LOOP_PATTERN not in text:
            raise RuntimeError("Could not find trainer.py loop anchor for resume checkpoint.")
        text = text.replace(TRAINER_RESUME_LOOP_PATTERN, TRAINER_RESUME_LOOP_REPLACEMENT, 1)
    return text


def patch_trainer_interval_checkpoints(r2_repo: Path) -> str:
    trainer_py = trainer_py_path(r2_repo)
    if not trainer_py.is_file():
        raise FileNotFoundError(f"Missing r2dreamer trainer.py: {trainer_py}")

    text = trainer_py.read_text(encoding="utf-8")
    if TRAINER_CHECKPOINT_PATCH_MARKER in text:
        needs_progress = TRAINER_PROGRESS_PATCH_MARKER not in text
        needs_resume = TRAINER_RESUME_PATCH_MARKER not in text
        if not needs_progress and not needs_resume:
            return "already_patched"

        backup = trainer_py.with_name(trainer_py.name + TRAINER_BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(trainer_py, backup)

        original = text
        status = "already_patched"
        if needs_progress:
            if "import os\n" not in text:
                text = text.replace("import pathlib\n", "import os\nimport pathlib\n", 1)
            if "import time\n" not in text:
                text = text.replace("import pathlib\n", "import pathlib\nimport time\n", 1)
            init_anchor = "        self._wm_poc_last_checkpoint_step = 0\n"
            init_insert = (
                init_anchor
                + '        self.progress_every = int(config.get("progress_every", os.environ.get("WM_POC_R2_PROGRESS_EVERY", 100)))\n'
                + "        self._wm_poc_start_time = time.time()\n"
                + "        self._wm_poc_last_progress_step = -1\n"
            )
            if "self.progress_every" not in text:
                if init_anchor not in text:
                    raise RuntimeError("Could not find patched trainer init anchor for progress heartbeat.")
                text = text.replace(init_anchor, init_insert, 1)
            checkpoint_anchor = "    # BEGIN WM_POC_INTERVAL_CHECKPOINTS\n"
            if checkpoint_anchor not in text:
                raise RuntimeError("Could not find trainer.py checkpoint anchor for progress heartbeat.")
            text = text.replace(checkpoint_anchor, TRAINER_PROGRESS_METHOD_INSERT + checkpoint_anchor, 1)
            if TRAINER_LOOP_REPLACEMENT not in text:
                if TRAINER_LOOP_PATTERN not in text:
                    raise RuntimeError("Could not find trainer.py loop anchor for progress heartbeat.")
                text = text.replace(TRAINER_LOOP_PATTERN, TRAINER_LOOP_REPLACEMENT, 1)
            status = "updated_progress_heartbeat"
        if needs_resume:
            text = _insert_trainer_resume(text)
            if status == "already_patched":
                status = "updated_resume_checkpoint"
        if text != original:
            trainer_py.write_text(text, encoding="utf-8")
        return status

    missing_anchors = [
        name
        for name, anchor in {
            "import": TRAINER_IMPORT_ANCHOR,
            "init": TRAINER_INIT_ANCHOR,
            "method": TRAINER_METHOD_ANCHOR,
            "loop": TRAINER_LOOP_PATTERN,
            "eval": TRAINER_EVAL_PATTERN,
        }.items()
        if anchor not in text
    ]
    if missing_anchors:
        raise RuntimeError(
            "Could not find trainer.py anchors for interval checkpoint patch: "
            + ", ".join(missing_anchors)
        )

    backup = trainer_py.with_name(trainer_py.name + TRAINER_BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(trainer_py, backup)

    text = text.replace(TRAINER_IMPORT_ANCHOR, TRAINER_IMPORT_REPLACEMENT, 1)
    text = text.replace(TRAINER_INIT_ANCHOR, TRAINER_INIT_REPLACEMENT, 1)
    text = text.replace(TRAINER_METHOD_ANCHOR, TRAINER_METHOD_INSERT, 1)
    text = text.replace(TRAINER_LOOP_PATTERN, TRAINER_LOOP_REPLACEMENT, 1)
    text = text.replace(TRAINER_EVAL_PATTERN, TRAINER_EVAL_REPLACEMENT, 1)
    text = _insert_trainer_resume(text)
    trainer_py.write_text(text, encoding="utf-8")
    return "patched"


def patch_serial_envs(r2_repo: Path) -> str:
    envs_init = envs_init_py_path(r2_repo)
    if not envs_init.is_file():
        raise FileNotFoundError(f"Missing r2dreamer envs/__init__.py: {envs_init}")

    text = envs_init.read_text(encoding="utf-8")
    if SERIAL_ENV_PATCH_MARKER in text:
        if "[wm_poc] Using serial envs" not in text:
            backup = envs_init.with_name(envs_init.name + SERIAL_ENV_BACKUP_SUFFIX)
            if not backup.exists():
                shutil.copy2(envs_init, backup)
            text = text.replace(
                "    if _wm_poc_use_serial_envs():\n"
                "        return WMPOCSerialEnv(constructor, env_num, device)\n",
                "    if _wm_poc_use_serial_envs():\n"
                "        print(f\"[wm_poc] Using serial envs for {env_num} DMC worker(s).\", flush=True)\n"
                "        return WMPOCSerialEnv(constructor, env_num, device)\n",
                1,
            )
            envs_init.write_text(text, encoding="utf-8")
            return "updated_serial_env_logging"
        return "already_patched"

    if SERIAL_ENV_IMPORT_ANCHOR not in text:
        raise RuntimeError("Could not find import anchor in envs/__init__.py.")
    if SERIAL_ENV_PATTERN not in text:
        raise RuntimeError("Could not find ParallelEnv construction block in envs/__init__.py.")

    backup = envs_init.with_name(envs_init.name + SERIAL_ENV_BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(envs_init, backup)

    text = text.replace(SERIAL_ENV_IMPORT_ANCHOR, SERIAL_ENV_IMPORT_REPLACEMENT, 1)
    text = text.replace(SERIAL_ENV_PATTERN, SERIAL_ENV_REPLACEMENT, 1)
    envs_init.write_text(text, encoding="utf-8")
    return "patched"


def verify_patch(r2_repo: Path, compile_file: bool = True) -> list[str]:
    train_py = train_py_path(r2_repo)
    if not train_py.is_file():
        raise FileNotFoundError(f"Missing r2dreamer train.py: {train_py}")
    text = train_py.read_text(encoding="utf-8")
    required = [PATCH_MARKER, "pretrained", "wm_poc_meta", "latest.pt", "_wm_poc_resume_step"]
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"Patch verification failed; missing tokens: {missing}")
    if SAVE_FINALLY_PATTERN in text:
        raise RuntimeError("Patch verification failed; latest.pt is still saved in a finally block.")
    if compile_file:
        py_compile.compile(str(train_py), doraise=True)
    return required


def verify_trainer_checkpoint_patch(r2_repo: Path, compile_file: bool = True) -> list[str]:
    trainer_py = trainer_py_path(r2_repo)
    if not trainer_py.is_file():
        raise FileNotFoundError(f"Missing r2dreamer trainer.py: {trainer_py}")
    text = trainer_py.read_text(encoding="utf-8")
    required = [
        TRAINER_CHECKPOINT_PATCH_MARKER,
        TRAINER_PROGRESS_PATCH_MARKER,
        TRAINER_RESUME_PATCH_MARKER,
        "checkpoint_every",
        "checkpoint_keep",
        "progress_every",
        "_log_progress(step)",
        "checkpoints",
        "includes_optimizer",
        "_save_interval_checkpoint(agent, step)",
        "_save_resume_checkpoint(agent, step)",
        'getattr(agent, "_wm_poc_resume_step"',
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"Trainer checkpoint patch verification failed; missing tokens: {missing}")
    if compile_file:
        py_compile.compile(str(trainer_py), doraise=True)
    return required


def verify_dmc_render_patch(r2_repo: Path, compile_file: bool = True) -> list[str]:
    dmc_py = dmc_py_path(r2_repo)
    if not dmc_py.is_file():
        raise FileNotFoundError(f"Missing r2dreamer envs/dmc.py: {dmc_py}")
    text = dmc_py.read_text(encoding="utf-8")
    required = [
        DMC_RENDER_PATCH_MARKER,
        "WM_POC_DMC_DISABLE_IMAGE_RENDER",
        "np.zeros(tuple(self._size) + (3,), dtype=np.uint8)",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"DMC render patch verification failed; missing tokens: {missing}")
    if compile_file:
        py_compile.compile(str(dmc_py), doraise=True)
    return required


def verify_serial_env_patch(r2_repo: Path, compile_file: bool = True) -> list[str]:
    envs_init = envs_init_py_path(r2_repo)
    if not envs_init.is_file():
        raise FileNotFoundError(f"Missing r2dreamer envs/__init__.py: {envs_init}")
    text = envs_init.read_text(encoding="utf-8")
    required = [
        SERIAL_ENV_PATCH_MARKER,
        "WM_POC_R2_SERIAL_ENVS",
        "WMPOCSerialEnv",
        "_wm_poc_make_env_group(env_constructor, config.env_num, config.device)",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"Serial env patch verification failed; missing tokens: {missing}")
    if compile_file:
        py_compile.compile(str(envs_init), doraise=True)
    return required


def backup_diff(r2_repo: Path, context: int = 3) -> str:
    train_py = train_py_path(r2_repo)
    backup = train_py.with_name(train_py.name + BACKUP_SUFFIX)
    if not backup.exists():
        return ""
    before = backup.read_text(encoding="utf-8").splitlines(keepends=True)
    after = train_py.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=str(backup),
            tofile=str(train_py),
            n=context,
        )
    )


def trainer_backup_diff(r2_repo: Path, context: int = 3) -> str:
    trainer_py = trainer_py_path(r2_repo)
    backup = trainer_py.with_name(trainer_py.name + TRAINER_BACKUP_SUFFIX)
    if not backup.exists():
        return ""
    before = backup.read_text(encoding="utf-8").splitlines(keepends=True)
    after = trainer_py.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=str(backup),
            tofile=str(trainer_py),
            n=context,
        )
    )


def dmc_backup_diff(r2_repo: Path, context: int = 3) -> str:
    dmc_py = dmc_py_path(r2_repo)
    backup = dmc_py.with_name(dmc_py.name + DMC_RENDER_BACKUP_SUFFIX)
    if not backup.exists():
        return ""
    before = backup.read_text(encoding="utf-8").splitlines(keepends=True)
    after = dmc_py.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=str(backup),
            tofile=str(dmc_py),
            n=context,
        )
    )


def serial_env_backup_diff(r2_repo: Path, context: int = 3) -> str:
    envs_init = envs_init_py_path(r2_repo)
    backup = envs_init.with_name(envs_init.name + SERIAL_ENV_BACKUP_SUFFIX)
    if not backup.exists():
        return ""
    before = backup.read_text(encoding="utf-8").splitlines(keepends=True)
    after = envs_init.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=str(backup),
            tofile=str(envs_init),
            n=context,
        )
    )
