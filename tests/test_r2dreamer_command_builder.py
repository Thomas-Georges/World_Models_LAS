from pathlib import Path

from wm_poc.r2dreamer.commands import build_all_commands, format_commands, load_experiment_config


def test_build_commands_contains_expected_hydra_overrides(monkeypatch) -> None:
    monkeypatch.delenv("R2_LOG_ROOT", raising=False)
    monkeypatch.delenv("R2_SOURCE_CKPT", raising=False)
    config = load_experiment_config(Path("configs/r2dreamer/three_way_walker_walk_to_run.yaml"))

    rendered = format_commands(build_all_commands(config))

    assert "python3 train.py" in rendered
    assert "env=dmc_proprio" in rendered
    assert "env.task=dmc_walker_walk" in rendered
    assert "env.task=dmc_walker_run" in rendered
    assert "model.rep_loss=r2dreamer" in rendered
    assert "model.compile=true" in rendered
    assert "+pretrained=" in rendered
    assert "trainer.steps=" in rendered
    assert "env.steps=" in rendered
    assert "env.train_ratio=" in rendered
    assert "trainer.steps=510000" in rendered
    assert "trainer.steps=250000" in rendered
    assert "trainer.train_ratio=64" in rendered
    assert "+trainer.checkpoint_every=25000" in rendered
    assert "+trainer.checkpoint_keep=8" in rendered
    assert "+trainer.progress_every=100" in rendered
    assert "WM_POC_R2_PROGRESS_EVERY=${WM_POC_R2_PROGRESS_EVERY:-100}" in rendered
    assert "TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}" in rendered
    assert "SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}" in rendered
    assert "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/xdg-runtime}" in rendered
    assert 'export PYOPENGL_PLATFORM="${MUJOCO_GL}"' in rendered


def test_smoke_command_keeps_env_worker_count_tiny(monkeypatch) -> None:
    monkeypatch.delenv("R2_SMOKE_ENV_NUM", raising=False)
    monkeypatch.delenv("R2_SMOKE_EVAL_EPISODES", raising=False)
    config = load_experiment_config(Path("configs/r2dreamer/three_way_walker_walk_to_run.yaml"))

    command = build_all_commands(config)["smoke"]

    assert "env.env_num=1" in command
    assert "env.eval_episode_num=0" in command
    assert "trainer.eval_episode_num=0" in command
    assert "model.compile=false" in command
    assert "WM_POC_DMC_DISABLE_IMAGE_RENDER=true" in command
    assert "WM_POC_R2_SERIAL_ENVS=false" in command
    assert "+trainer.checkpoint_every=0" in command
    assert "+trainer.progress_every=100" in command


def test_default_three_way_config_is_scaled_r2() -> None:
    config = load_experiment_config(Path("configs/r2dreamer/three_way_walker_walk_to_run.yaml"))

    assert config["algorithm"]["rep_loss"] == "r2dreamer"
    assert int(config["training"]["source_steps"]) >= 500000
    assert int(config["training"]["target_steps"]) >= 250000
    assert int(config["training"]["train_ratio"]) >= 64
    assert int(config["training"]["eval_episodes"]) >= 5


def test_debug_config_preserves_tiny_values() -> None:
    config = load_experiment_config(Path("configs/r2dreamer/debug_walker_walk_to_run.yaml"))

    assert config["build_scale"]["tier"] == "debug"
    assert int(config["training"]["source_steps"]) == 100000
    assert int(config["training"]["target_steps"]) == 50000
    assert int(config["training"]["train_ratio"]) == 16


def test_command_builder_respects_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("R2_TARGET_STEPS", "1234")
    monkeypatch.setenv("R2_ENV_NUM", "2")
    config = load_experiment_config(Path("configs/r2dreamer/three_way_walker_walk_to_run.yaml"))

    command = build_all_commands(config)["target_scratch"]

    assert "trainer.steps=1234" in command
    assert "env.steps=1234" in command
    assert "env.env_num=2" in command


def test_command_builder_respects_checkpoint_overrides(monkeypatch) -> None:
    monkeypatch.setenv("R2_CHECKPOINT_EVERY", "5000")
    monkeypatch.setenv("R2_CHECKPOINT_KEEP", "3")
    monkeypatch.setenv("R2_PROGRESS_EVERY", "250")
    config = load_experiment_config(Path("configs/r2dreamer/three_way_walker_walk_to_run.yaml"))

    command = build_all_commands(config)["source_base"]

    assert "+trainer.checkpoint_every=5000" in command
    assert "+trainer.checkpoint_keep=3" in command
    assert "+trainer.progress_every=250" in command


def test_command_builder_respects_compile_overrides(monkeypatch) -> None:
    monkeypatch.setenv("R2_COMPILE", "false")
    monkeypatch.setenv("R2_SMOKE_COMPILE", "true")
    config = load_experiment_config(Path("configs/r2dreamer/three_way_walker_walk_to_run.yaml"))

    commands = build_all_commands(config)

    assert "model.compile=false" in commands["source_base"]
    assert "model.compile=true" in commands["smoke"]


def test_command_builder_uses_r2_source_ckpt_override(monkeypatch) -> None:
    monkeypatch.setenv("R2_SOURCE_CKPT", "/tmp/custom_source/latest.pt")
    config = load_experiment_config(Path("configs/r2dreamer/three_way_walker_walk_to_run.yaml"))

    command = build_all_commands(config)["target_finetune"]

    assert "+pretrained=/tmp/custom_source/latest.pt" in command


def test_finetuning_flags_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("R2_PRETRAINED_STRICT", "false")
    monkeypatch.setenv("R2_LOAD_OPTIMIZER", "true")
    config = load_experiment_config(Path("configs/r2dreamer/three_way_walker_walk_to_run.yaml"))

    command = build_all_commands(config)["target_finetune"]

    assert "+pretrained_strict=false" in command
    assert "+load_optimizer=true" in command


def test_dmc_vision_keeps_image_rendering_enabled(monkeypatch) -> None:
    monkeypatch.delenv("R2_DISABLE_DMC_IMAGE_RENDER", raising=False)
    config = load_experiment_config(Path("configs/r2dreamer/dreamer_dmc_vision_12m.yaml"))

    command = build_all_commands(config)["source_base"]

    assert "WM_POC_DMC_DISABLE_IMAGE_RENDER=false" in command


def test_rendering_defaults_are_read_from_config(monkeypatch) -> None:
    monkeypatch.delenv("R2_MUJOCO_GL", raising=False)
    monkeypatch.delenv("R2_MUJOCO_EGL_DEVICE_ID", raising=False)
    config = load_experiment_config(
        Path("configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml")
    )

    command = build_all_commands(config)["source_base"]

    assert "export MUJOCO_GL=${MUJOCO_GL:-osmesa}" in command
    assert 'export PYOPENGL_PLATFORM="${MUJOCO_GL}"' in command
    assert "export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}" in command
    assert "WM_POC_DMC_DISABLE_IMAGE_RENDER=false" in command
    assert "WM_POC_R2_SERIAL_ENVS=false" in command


def test_stale_internal_render_flag_does_not_disable_dmc_vision(monkeypatch) -> None:
    monkeypatch.setenv("WM_POC_DMC_DISABLE_IMAGE_RENDER", "true")
    monkeypatch.delenv("R2_DISABLE_DMC_IMAGE_RENDER", raising=False)
    config = load_experiment_config(
        Path("configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml")
    )

    command = build_all_commands(config)["source_base"]

    assert "WM_POC_DMC_DISABLE_IMAGE_RENDER=false" in command


def test_public_render_disable_override_is_still_respected(monkeypatch) -> None:
    monkeypatch.setenv("R2_DISABLE_DMC_IMAGE_RENDER", "true")
    config = load_experiment_config(
        Path("configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml")
    )

    command = build_all_commands(config)["source_base"]

    assert "WM_POC_DMC_DISABLE_IMAGE_RENDER=true" in command


def test_a100_vision_config_is_image_r2_25m() -> None:
    config = load_experiment_config(
        Path("configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml")
    )

    assert config["experiment_name"] == "walker_walk_to_run_a100_r2_vision25m"
    assert config["environment"]["env"] == "dmc_vision"
    assert config["environment"]["disable_image_render"] is False
    assert config["environment"]["serial_envs"] is False
    assert config["algorithm"]["rep_loss"] == "r2dreamer"
    assert config["algorithm"]["model"] == "size25M"
    assert int(config["training"]["source_steps"]) == 800000
    assert int(config["training"]["target_steps"]) == 400000
    assert int(config["training"]["train_ratio"]) == 224
    assert int(config["training"]["eval_episodes"]) == 5
    assert int(config["training"]["checkpoint_keep"]) == 6
    assert int(config["training"]["progress_every"]) == 100
