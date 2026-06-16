from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from wm_poc.local_global.configs import (  # noqa: E402
    latent_cache_dir,
    load_local_global_config,
    resolve_run_dir,
    typed_config,
    validate_local_global_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "local_global"
ALL_CONFIGS = [
    "base.yaml",
    "smoke_synthetic.yaml",
    "smoke_pointmaze.yaml",
    "pointmaze_surrogate_t4.yaml",
    "pointmaze_surrogate_a100.yaml",
]


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_all_configs_load_and_validate(name):
    config = load_local_global_config(CONFIG_DIR / name)
    typed = typed_config(config)
    assert typed.planner.step_action_dim == typed.planner.action_dim * typed.global_model.frameskip
    assert len(typed.planner.step_action_low) == typed.planner.step_action_dim


def test_env_placeholder_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("LG_RUN_ROOT", str(tmp_path / "runs"))
    config = load_local_global_config(CONFIG_DIR / "base.yaml")
    assert config["paths"]["run_root"] == str(tmp_path / "runs")


def test_latent_cache_dir_derivation(monkeypatch):
    monkeypatch.delenv("LG_LATENT_CACHE_DIR", raising=False)
    config = load_local_global_config(CONFIG_DIR / "base.yaml")
    derived = latent_cache_dir(config)
    assert derived.name == "dinov2_vits14_img224"
    assert derived.parent.name == "point_maze"


def test_latent_cache_dir_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LG_LATENT_CACHE_DIR", str(tmp_path / "cache"))
    config = load_local_global_config(CONFIG_DIR / "base.yaml")
    assert latent_cache_dir(config) == tmp_path / "cache"


def test_resolve_run_dir_creates_layout(tmp_path):
    config = load_local_global_config(CONFIG_DIR / "smoke_synthetic.yaml")
    run_dir = resolve_run_dir(config, tmp_path / "my_run")
    for subdir in ("transition_data", "checkpoints", "metrics", "planning", "figures"):
        assert (run_dir / subdir).is_dir()
    assert config["paths"]["run_dir"] == str(run_dir)


def test_auto_run_name(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_RUN_ROOT", str(tmp_path))
    config = load_local_global_config(CONFIG_DIR / "base.yaml")
    config["run_name"] = ""
    run_dir = resolve_run_dir(config)
    assert run_dir.name.startswith("local_global_point_maze_")


def _minimal_valid():
    return load_local_global_config(CONFIG_DIR / "smoke_synthetic.yaml")


def test_validate_rejects_wrong_track():
    config = _minimal_valid()
    config["track"] = "dino_wm"
    with pytest.raises(ValueError, match="track"):
        validate_local_global_config(config)


def test_validate_rejects_bad_bounds():
    config = _minimal_valid()
    config["planning"]["action_low"] = [-1.0]
    with pytest.raises(ValueError, match="action_low"):
        validate_local_global_config(config)


def test_validate_rejects_unknown_planner():
    config = _minimal_valid()
    config["evaluation"]["planners"] = ["global_cem", "made_up"]
    with pytest.raises(ValueError, match="made_up"):
        validate_local_global_config(config)


def test_validate_rejects_bad_projection():
    config = _minimal_valid()
    config["local_model"]["projection"] = "flatten_everything"
    with pytest.raises(ValueError, match="projection"):
        validate_local_global_config(config)
