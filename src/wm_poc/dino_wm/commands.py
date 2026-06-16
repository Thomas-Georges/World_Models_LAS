from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

from wm_poc.dino_wm.configs import get_config_value


def _as_bool_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower()


def _repo_path(config: dict[str, Any]) -> Path:
    repo = Path(str(config.get("external_repo", "external_repos/dino_wm"))).expanduser()
    return repo if repo.is_absolute() else (Path.cwd() / repo)


def _run_dir(config: dict[str, Any]) -> str:
    configured = get_config_value(config, "artifacts.run_dir")
    if configured:
        return str(configured)
    return str(Path(str(get_config_value(config, "artifacts.log_root"))) / str(config.get("run_name")))


def _dataset_env_path(config: dict[str, Any]) -> str:
    root = Path(str(get_config_value(config, "dataset.root"))).expanduser()
    env_name = str(get_config_value(config, "dataset.env"))
    return str(root / env_name)


def _upstream_env_name(config: dict[str, Any]) -> str:
    env_name = str(get_config_value(config, "dataset.env"))
    return {
        "point_maze": "point_maze",
        "pusht_noise": "pusht",
        "wall_single": "wall",
    }.get(env_name, env_name)


# Upstream Hydra dataset targets that can serve precomputed DINO latents.
LATENT_DATASET_TARGETS = {
    "point_maze": "wm_poc_latent_dataset.load_point_maze_latent_slice_train_val",
}


def _upstream_encoder_name(config: dict[str, Any]) -> str:
    encoder = str(get_config_value(config, "features.encoder", "dinov2_patch"))
    return {"dinov2_patch": "dinov2_vits14"}.get(encoder, encoder)


def latent_training_enabled(config: dict[str, Any]) -> bool:
    if not bool(get_config_value(config, "features.cache_enabled", False)):
        return False
    return _upstream_env_name(config) in LATENT_DATASET_TARGETS


def latent_cache_dir(config: dict[str, Any]) -> str:
    cache_root = Path(str(get_config_value(config, "features.cache_dir"))).expanduser()
    env_name = str(get_config_value(config, "dataset.env"))
    img_size = int(get_config_value(config, "features.image_size", 224))
    return str(cache_root / env_name / f"{_upstream_encoder_name(config)}_img{img_size}")


def _decoder_disabled(config: dict[str, Any]) -> bool:
    overrides = get_config_value(config, "upstream.train_overrides", []) or []
    return any(str(item).strip().lower() == "has_decoder=false" for item in overrides)


def _validate_latent_training(config: dict[str, Any]) -> None:
    """Latent-cache training serves DINO patch latents instead of images, so
    the upstream decoder (which reconstructs pixels against image targets)
    cannot run. Fail at command-build time with the fix instead of crashing
    in mse_loss on the GPU."""

    if latent_training_enabled(config) and not _decoder_disabled(config):
        raise ValueError(
            "features.cache_enabled=true serves precomputed latents, but this config keeps the "
            "upstream decoder on (has_decoder defaults to true) and the decoder needs image "
            "targets. Add upstream.train_overrides [has_decoder=false, model.train_decoder=false] "
            "like the no-decoder configs, or set features.cache_enabled=false to train on images."
        )


def _max_rollouts(config: dict[str, Any]) -> int:
    train = int(get_config_value(config, "dataset.max_train_trajectories"))
    val = int(get_config_value(config, "dataset.max_val_trajectories"))
    return max(train + val, 1)


def _upstream_split_ratio(config: dict[str, Any]) -> float:
    train = int(get_config_value(config, "dataset.max_train_trajectories"))
    total = _max_rollouts(config)
    return train / total


def _upstream_model_name(config: dict[str, Any]) -> str:
    return str(config.get("run_name"))


def _append_if_present(argv: list[str], key: str, value: Any) -> None:
    if value is not None:
        argv.append(f"{key}={value}")


def _effective_epochs(config: dict[str, Any]) -> Any:
    if get_config_value(config, "finetuning.enabled", False):
        return get_config_value(config, "finetuning.epochs", get_config_value(config, "training.epochs"))
    return get_config_value(config, "training.epochs")


def _effective_save_every(config: dict[str, Any]) -> int:
    epochs = int(_effective_epochs(config))
    configured = int(
        os.environ.get(
            "DINO_SAVE_EVERY_EPOCHS",
            get_config_value(config, "training.save_every_epochs", epochs),
        )
    )
    return max(1, min(configured, epochs))


def _effective_save_every_steps(config: dict[str, Any]) -> int:
    configured = int(
        os.environ.get(
            "DINO_SAVE_EVERY_STEPS",
            get_config_value(config, "training.save_every_steps", 0),
        )
    )
    return max(0, configured)


def _effective_num_workers(config: dict[str, Any]) -> int:
    configured = int(
        os.environ.get(
            "DINO_NUM_WORKERS",
            get_config_value(config, "training.num_workers", 4),
        )
    )
    return max(0, configured)


def _effective_lr(config: dict[str, Any], component: str) -> Any:
    if get_config_value(config, "finetuning.enabled", False):
        value = get_config_value(config, f"finetuning.{component}_lr")
        if value is not None:
            return value
    return get_config_value(config, f"training.{component}_lr")


def _append_common_train_overrides(argv: list[str], config: dict[str, Any]) -> None:
    run_name = str(config.get("run_name"))
    argv.extend(
        [
            "--config-name",
            str(get_config_value(config, "upstream.train_config_name", "train.yaml")),
            f"env={_upstream_env_name(config)}",
            f"frameskip={get_config_value(config, 'model.frameskip')}",
            f"num_hist={get_config_value(config, 'model.num_hist')}",
            f"ckpt_base_path={get_config_value(config, 'artifacts.ckpt_root')}",
            f"hydra.run.dir={get_config_value(config, 'artifacts.ckpt_root')}/outputs/{run_name}",
            f"hydra.sweep.dir={get_config_value(config, 'artifacts.ckpt_root')}/outputs/{run_name}",
            f"env.dataset.data_path={_dataset_env_path(config)}",
            f"env.dataset.n_rollout={_max_rollouts(config)}",
            f"env.dataset.split_ratio={_upstream_split_ratio(config):.6f}",
            f"env.num_workers={_effective_num_workers(config)}",
            f"training.seed={get_config_value(config, 'training.seed')}",
            f"training.batch_size={get_config_value(config, 'training.batch_size')}",
            f"training.epochs={_effective_epochs(config)}",
            f"training.save_every_x_epoch={_effective_save_every(config)}",
            f"img_size={get_config_value(config, 'features.image_size')}",
            f"model.train_encoder={_as_bool_string(not get_config_value(config, 'features.freeze_encoder', True))}",
            f"action_emb_dim={get_config_value(config, 'model.action_emb_dim')}",
            f"training.predictor_lr={_effective_lr(config, 'predictor')}",
            f"training.action_encoder_lr={_effective_lr(config, 'action_encoder')}",
            f"training.decoder_lr={_effective_lr(config, 'decoder')}",
            "plan_settings.plan_cfg_path=null",
        ]
    )
    if latent_training_enabled(config):
        argv.extend(
            [
                f"env.dataset._target_={LATENT_DATASET_TARGETS[_upstream_env_name(config)]}",
                f"+env.dataset.latent_cache_dir={latent_cache_dir(config)}",
                f"+env.dataset.slice_stride={int(get_config_value(config, 'dataset.slice_stride', 1))}",
            ]
        )
    save_every_steps = _effective_save_every_steps(config)
    if save_every_steps > 0:
        argv.append(f"++training.save_every_steps={save_every_steps}")
    for override in get_config_value(config, "upstream.train_overrides", []) or []:
        argv.append(str(override))


def build_train_command(config: dict[str, Any]) -> list[str]:
    _validate_latent_training(config)
    repo = _repo_path(config)
    argv = [sys.executable, str(repo / "train.py")]
    _append_common_train_overrides(argv, config)

    if get_config_value(config, "finetuning.enabled", False):
        # Upstream's Hydra schema has no finetuning section: the keys are
        # appended (++) and consumed by the fine-tune init patch installed by
        # patch_finetune_loading.py. Fine-tune learning rates and epochs are
        # already mapped onto the plain training.* overrides above.
        argv.extend(
            [
                "++finetuning.enabled=true",
                f"++finetuning.init_from={get_config_value(config, 'finetuning.init_from')}",
                f"++finetuning.strict={_as_bool_string(get_config_value(config, 'finetuning.strict', True))}",
                f"++finetuning.reset_epoch={_as_bool_string(get_config_value(config, 'finetuning.reset_epoch', True))}",
                f"++finetuning.load_predictor={_as_bool_string(get_config_value(config, 'finetuning.load.predictor', True))}",
                f"++finetuning.load_action_encoder={_as_bool_string(get_config_value(config, 'finetuning.load.action_encoder', True))}",
                f"++finetuning.load_decoder={_as_bool_string(get_config_value(config, 'finetuning.load.decoder', False))}",
            ]
        )
    return argv


def _model_ref_from_checkpoint(checkpoint_path: str) -> tuple[str, str, str]:
    """Resolve (ckpt_base_path, model_name, model_epoch) for upstream plan.py
    from a concrete checkpoint file.

    plan.py never takes a checkpoint path directly: it loads
    ``<ckpt_base_path>/outputs/<model_name>/hydra.yaml`` and
    ``checkpoints/model_<model_epoch>.pth``, so the checkpoint must live in
    that layout and the reference must point at the run that produced it.
    """

    path = Path(checkpoint_path).expanduser()
    stem = path.stem
    if stem == "model_latest_step":
        raise ValueError(
            "model_latest_step.pth is a rolling intra-epoch state_dict checkpoint that "
            "plan.py cannot load; plan from the epoch checkpoint model_latest.pth instead."
        )
    if not stem.startswith("model_") or path.suffix not in {".pth", ".pt"}:
        raise ValueError(
            f"Expected an upstream model_<epoch>.pth checkpoint, got: {path.name}"
        )
    if path.parent.name != "checkpoints" or path.parent.parent.parent.name != "outputs":
        raise ValueError(
            "plan.py can only load checkpoints from "
            f"<ckpt_base>/outputs/<run_name>/checkpoints/; got: {path}"
        )
    model_name = path.parent.parent.name
    ckpt_base = path.parent.parent.parent.parent
    model_epoch = stem[len("model_") :]
    return str(ckpt_base), model_name, model_epoch


def build_plan_command(config: dict[str, Any], checkpoint_path: str) -> list[str]:
    repo = _repo_path(config)
    planner = os.environ.get("DINO_PLANNER", str(get_config_value(config, "planning.planner", "cem")))
    output_dir = Path(_run_dir(config)) / "planning" / planner
    # An unset checkpoint can reach us as "" or "." (empty string through
    # argparse's Path type); both mean "use the config-derived reference".
    if checkpoint_path and str(checkpoint_path).strip() not in {"", "."}:
        ckpt_base, model_name, model_epoch = _model_ref_from_checkpoint(checkpoint_path)
    else:
        ckpt_base = str(get_config_value(config, "artifacts.ckpt_root"))
        model_name = _upstream_model_name(config)
        model_epoch = "latest"
    argv = [
        sys.executable,
        str(repo / "plan.py"),
        "--config-name",
        str(get_config_value(config, "upstream.plan_config_name", "plan.yaml")),
        f"model_name={model_name}",
        f"ckpt_base_path={ckpt_base}",
        f"model_epoch={model_epoch}",
        f"hydra.run.dir={output_dir}",
        f"hydra.sweep.dir={output_dir}",
        f"n_evals={get_config_value(config, 'planning.n_evals')}",
        f"planner={planner}",
        f"goal_H={get_config_value(config, 'planning.goal_H')}",
        f"goal_source={get_config_value(config, 'planning.goal_source')}",
        f"planner.opt_steps={get_config_value(config, 'planning.opt_steps')}",
        f"seed={get_config_value(config, 'training.seed')}",
    ]
    if planner in {"cem", "mpc_cem"}:
        argv.append(f"planner.num_samples={get_config_value(config, 'planning.samples')}")
    for override in get_config_value(config, "upstream.plan_overrides", []) or []:
        argv.append(str(override))
    return argv


def build_precompute_command(config: dict[str, Any]) -> list[str]:
    repo = _repo_path(config)
    argv = [
        sys.executable,
        str(repo / "wm_poc_precompute_latents.py"),
        "--data-path",
        _dataset_env_path(config),
        "--cache-dir",
        latent_cache_dir(config),
        "--n-rollout",
        str(_max_rollouts(config)),
        "--img-size",
        str(get_config_value(config, "features.image_size", 224)),
        "--encoder-name",
        _upstream_encoder_name(config),
        "--batch-size",
        str(int(get_config_value(config, "features.precompute_batch_size", 128))),
    ]
    for override in get_config_value(config, "upstream.precompute_overrides", []) or []:
        argv.append(str(override))
    return argv


def render_command(argv: list[str]) -> str:
    return shlex.join(argv)


def write_command_file(argv: list[str], path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_command(argv) + "\n", encoding="utf-8")
