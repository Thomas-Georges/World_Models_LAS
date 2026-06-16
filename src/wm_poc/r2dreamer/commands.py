from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


REPO_DEFAULT = "/content/wm-prediction"
DRIVE_ROOT_DEFAULT = "/content/drive/MyDrive/wm_poc"
R2_REMOTE_DEFAULT = "https://github.com/NM512/r2dreamer.git"


def load_experiment_config(path: Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read R2-Dreamer wrapper configs.") from (
            _YAML_IMPORT_ERROR
        )
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return data


def nested(config: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def bool_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower()


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def env_or_config(env_name: str, config: dict[str, Any], config_path: str, default: Any) -> str:
    return str(os.environ.get(env_name, nested(config, config_path, default)))


def path_defaults() -> dict[str, str]:
    drive_root = os.environ.get("WM_POC_DRIVE_ROOT", DRIVE_ROOT_DEFAULT)
    log_dir = os.environ.get("WM_POC_LOG_DIR", f"{drive_root}/logs")
    figure_base = os.environ.get(
        "WM_POC_FIGURE_DIR",
        os.environ.get("WM_POC_FIG_DIR", f"{drive_root}/figures"),
    )
    fig_dir = os.environ.get("R2_FIGURE_DIR", f"{figure_base}/r2dreamer")
    return {
        "repo": os.environ.get("WM_POC_REPO", REPO_DEFAULT),
        "drive_root": drive_root,
        "log_root": os.environ.get("R2_LOG_ROOT", f"{log_dir}/r2dreamer"),
        "figure_dir": fig_dir,
        "external_repos": os.environ.get("WM_POC_EXTERNAL_REPOS", "/content/external_repos"),
    }


def r2dreamer_repo(config: dict[str, Any]) -> str:
    paths = path_defaults()
    return os.environ.get(
        "R2DREAMER_REPO",
        f"{paths['external_repos']}/r2dreamer",
    )


def render_train_command(
    *,
    r2_repo: str,
    logdir: str,
    env_name: str,
    task: str,
    model: str,
    rep_loss: str,
    compile_model: str,
    seed: str,
    steps: str,
    batch_size: str,
    batch_length: str,
    train_ratio: str,
    eval_every: str,
    eval_episodes: str,
    env_num: str,
    disable_dmc_image_render: str,
    checkpoint_every: str,
    checkpoint_keep: str,
    progress_every: str,
    mujoco_gl: str = "egl",
    mujoco_egl_device_id: str = "0",
    serial_envs: str = "false",
    pretrained: str | None = None,
    pretrained_strict: str = "true",
    load_optimizer: str = "false",
) -> str:
    parts = [
        "python3 train.py",
        f"logdir={shlex.quote(logdir)}",
        f"env={shlex.quote(env_name)}",
        f"env.task={shlex.quote(task)}",
        f"model={shlex.quote(model)}",
        f"model.rep_loss={shlex.quote(rep_loss)}",
        f"model.compile={shlex.quote(compile_model)}",
        f"seed={shlex.quote(seed)}",
        f"env.env_num={shlex.quote(env_num)}",
        f"env.steps={shlex.quote(steps)}",
        f"env.train_ratio={shlex.quote(train_ratio)}",
        f"env.eval_episode_num={shlex.quote(eval_episodes)}",
        f"batch_size={shlex.quote(batch_size)}",
        f"batch_length={shlex.quote(batch_length)}",
        f"trainer.steps={shlex.quote(steps)}",
        f"trainer.eval_every={shlex.quote(eval_every)}",
        f"trainer.eval_episode_num={shlex.quote(eval_episodes)}",
        f"trainer.train_ratio={shlex.quote(train_ratio)}",
        f"+trainer.checkpoint_every={shlex.quote(checkpoint_every)}",
        f"+trainer.checkpoint_keep={shlex.quote(checkpoint_keep)}",
        f"+trainer.progress_every={shlex.quote(progress_every)}",
    ]
    if pretrained is not None:
        parts.extend(
            [
                f"+pretrained={shlex.quote(pretrained)}",
                f"+pretrained_strict={shlex.quote(pretrained_strict)}",
                f"+load_optimizer={shlex.quote(load_optimizer)}",
            ]
        )

    body = " \\\n  ".join(parts)
    return "\n".join(
        [
            f"cd {shlex.quote(r2_repo)}",
            "export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}",
            "export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}",
            "export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/xdg-runtime}",
            'mkdir -p "${XDG_RUNTIME_DIR}"',
            'chmod 700 "${XDG_RUNTIME_DIR}" 2>/dev/null || true',
            f"export MUJOCO_GL=${{MUJOCO_GL:-{shlex.quote(mujoco_gl)}}}",
            'if [[ "${MUJOCO_GL}" == "egl" || "${MUJOCO_GL}" == "osmesa" ]]; then',
            '  export PYOPENGL_PLATFORM="${MUJOCO_GL}"',
            "else",
            "  unset PYOPENGL_PLATFORM",
            "fi",
            "export MUJOCO_EGL_DEVICE_ID="
            f"${{MUJOCO_EGL_DEVICE_ID:-{shlex.quote(mujoco_egl_device_id)}}}",
            f"export WM_POC_DMC_DISABLE_IMAGE_RENDER={shlex.quote(disable_dmc_image_render)}",
            "export WM_POC_R2_PROGRESS_EVERY="
            f"${{WM_POC_R2_PROGRESS_EVERY:-{shlex.quote(progress_every)}}}",
            f"export WM_POC_R2_SERIAL_ENVS={shlex.quote(serial_envs)}",
            body,
        ]
    )


def build_all_commands(config: dict[str, Any]) -> dict[str, str]:
    paths = path_defaults()
    log_root = paths["log_root"]
    r2_repo = r2dreamer_repo(config)

    env_name = env_or_config("R2_ENV", config, "environment.env", "dmc_proprio")
    source_task = env_or_config("R2_SOURCE_TASK", config, "environment.source_task", "dmc_walker_walk")
    target_task = env_or_config("R2_TARGET_TASK", config, "environment.target_task", "dmc_walker_run")
    disable_image_default = "true" if env_name == "dmc_proprio" else "false"
    disable_dmc_image_render = bool_string(
        is_truthy(
            env_or_config(
                "R2_DISABLE_DMC_IMAGE_RENDER",
                config,
                "environment.disable_image_render",
                disable_image_default,
            )
        )
    )
    mujoco_gl = env_or_config("R2_MUJOCO_GL", config, "rendering.mujoco_gl", "egl")
    mujoco_egl_device_id = env_or_config(
        "R2_MUJOCO_EGL_DEVICE_ID",
        config,
        "rendering.mujoco_egl_device_id",
        0,
    )
    serial_envs = bool_string(env_or_config("R2_SERIAL_ENVS", config, "environment.serial_envs", False))
    model = env_or_config("R2_MODEL", config, "algorithm.model", "size12M")
    rep_loss = env_or_config("R2_REP_LOSS", config, "algorithm.rep_loss", "dreamer")
    compile_model = bool_string(env_or_config("R2_COMPILE", config, "algorithm.compile", True))
    seed = env_or_config("R2_SEED", config, "training.seed", 0)
    batch_size = env_or_config("R2_BATCH_SIZE", config, "training.batch_size", 16)
    batch_length = env_or_config("R2_BATCH_LENGTH", config, "training.batch_length", 64)
    train_ratio = env_or_config("R2_TRAIN_RATIO", config, "training.train_ratio", 16)
    eval_every = env_or_config("R2_EVAL_EVERY", config, "training.eval_every", 10000)
    eval_episodes = env_or_config("R2_EVAL_EPISODES", config, "training.eval_episodes", 2)
    env_num = env_or_config("R2_ENV_NUM", config, "training.env_num", 4)
    save_eval_checkpoints = is_truthy(
        env_or_config("R2_SAVE_EVAL_CHECKPOINTS", config, "training.save_eval_checkpoints", True)
    )
    checkpoint_every_default = eval_every if save_eval_checkpoints else 0
    checkpoint_every = env_or_config(
        "R2_CHECKPOINT_EVERY",
        config,
        "training.checkpoint_every",
        checkpoint_every_default,
    )
    checkpoint_keep = env_or_config("R2_CHECKPOINT_KEEP", config, "training.checkpoint_keep", 12)
    progress_every = env_or_config("R2_PROGRESS_EVERY", config, "training.progress_every", 100)
    source_steps = env_or_config("R2_SOURCE_STEPS", config, "training.source_steps", 100000)
    target_steps = env_or_config("R2_TARGET_STEPS", config, "training.target_steps", 50000)

    smoke_steps = env_or_config("R2_SMOKE_STEPS", config, "smoke.steps", 2000)
    smoke_batch_size = env_or_config("R2_SMOKE_BATCH_SIZE", config, "smoke.batch_size", 4)
    smoke_batch_length = env_or_config("R2_SMOKE_BATCH_LENGTH", config, "smoke.batch_length", 16)
    smoke_train_ratio = env_or_config("R2_SMOKE_TRAIN_RATIO", config, "smoke.train_ratio", 4)
    smoke_eval_every = env_or_config("R2_SMOKE_EVAL_EVERY", config, "smoke.eval_every", 1000)
    smoke_eval_episodes = env_or_config("R2_SMOKE_EVAL_EPISODES", config, "smoke.eval_episodes", 0)
    smoke_env_num = env_or_config("R2_SMOKE_ENV_NUM", config, "smoke.env_num", 1)
    smoke_checkpoint_every = env_or_config("R2_SMOKE_CHECKPOINT_EVERY", config, "smoke.checkpoint_every", 0)
    smoke_checkpoint_keep = env_or_config("R2_SMOKE_CHECKPOINT_KEEP", config, "smoke.checkpoint_keep", 1)
    smoke_progress_every = env_or_config("R2_SMOKE_PROGRESS_EVERY", config, "smoke.progress_every", 100)
    smoke_compile_model = bool_string(env_or_config("R2_SMOKE_COMPILE", config, "smoke.compile", False))
    smoke_task = str(nested(config, "environment.task", source_task))

    pretrained_strict = bool_string(
        env_or_config("R2_PRETRAINED_STRICT", config, "finetuning.pretrained_strict", True)
    )
    load_optimizer = bool_string(
        env_or_config("R2_LOAD_OPTIMIZER", config, "finetuning.load_optimizer", False)
    )
    source_checkpoint = os.environ.get("R2_SOURCE_CKPT", f"{log_root}/source_base/latest.pt")

    common = {
        "r2_repo": r2_repo,
        "env_name": env_name,
        "model": model,
        "rep_loss": rep_loss,
        "seed": seed,
        "disable_dmc_image_render": disable_dmc_image_render,
        "mujoco_gl": mujoco_gl,
        "mujoco_egl_device_id": mujoco_egl_device_id,
        "serial_envs": serial_envs,
    }
    return {
        "smoke": render_train_command(
            **common,
            logdir=f"{log_root}/smoke",
            task=smoke_task,
            steps=smoke_steps,
            batch_size=smoke_batch_size,
            batch_length=smoke_batch_length,
            train_ratio=smoke_train_ratio,
            eval_every=smoke_eval_every,
            eval_episodes=smoke_eval_episodes,
            env_num=smoke_env_num,
            checkpoint_every=smoke_checkpoint_every,
            checkpoint_keep=smoke_checkpoint_keep,
            progress_every=smoke_progress_every,
            compile_model=smoke_compile_model,
        ),
        "source_base": render_train_command(
            **common,
            logdir=f"{log_root}/source_base",
            task=source_task,
            steps=source_steps,
            batch_size=batch_size,
            batch_length=batch_length,
            train_ratio=train_ratio,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            env_num=env_num,
            checkpoint_every=checkpoint_every,
            checkpoint_keep=checkpoint_keep,
            progress_every=progress_every,
            compile_model=compile_model,
        ),
        "target_finetune": render_train_command(
            **common,
            logdir=f"{log_root}/target_finetune",
            task=target_task,
            steps=target_steps,
            batch_size=batch_size,
            batch_length=batch_length,
            train_ratio=train_ratio,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            env_num=env_num,
            checkpoint_every=checkpoint_every,
            checkpoint_keep=checkpoint_keep,
            progress_every=progress_every,
            compile_model=compile_model,
            pretrained=source_checkpoint,
            pretrained_strict=pretrained_strict,
            load_optimizer=load_optimizer,
        ),
        "target_scratch": render_train_command(
            **common,
            logdir=f"{log_root}/target_scratch",
            task=target_task,
            steps=target_steps,
            batch_size=batch_size,
            batch_length=batch_length,
            train_ratio=train_ratio,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            env_num=env_num,
            checkpoint_every=checkpoint_every,
            checkpoint_keep=checkpoint_keep,
            progress_every=progress_every,
            compile_model=compile_model,
        ),
    }


def format_commands(commands: dict[str, str], run: str | None = None) -> str:
    selected = {run: commands[run]} if run else commands
    chunks = []
    for name, command in selected.items():
        chunks.append(f"# --- {name} ---\n{command}")
    return "\n\n".join(chunks) + "\n"
