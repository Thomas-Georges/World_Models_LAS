import os
from pathlib import Path

import pytest

from wm_poc.r2dreamer.commands import load_experiment_config
from wm_poc.r2dreamer.posthoc import configure_headless_mujoco
from wm_poc.r2dreamer.visualization import (
    default_run_specs,
    pca_projection,
    resolve_log_root,
    visualization_paths,
)


def test_default_visualization_run_specs_use_source_and_target_tasks() -> None:
    config = load_experiment_config(
        Path("configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml")
    )
    specs = default_run_specs(config, Path("/tmp/r2-log-root"))

    assert [spec.name for spec in specs] == ["source_base", "target_finetune", "target_scratch"]
    assert specs[0].task == "dmc_walker_walk"
    assert specs[1].task == "dmc_walker_run"
    assert specs[2].checkpoint == Path("/tmp/r2-log-root/target_scratch/latest.pt")


def test_visualization_paths_follow_drive_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WM_POC_DRIVE_ROOT", "/drive/wm_poc")
    monkeypatch.delenv("R2_FIGURE_DIR", raising=False)
    monkeypatch.delenv("R2_VIDEO_DIR", raising=False)
    monkeypatch.delenv("R2_REPORT_DIR", raising=False)

    paths = visualization_paths(
        run_root=Path("/drive/wm_poc/logs/r2dreamer/example"),
        run_name="example",
    )

    assert paths.video_dir == Path("/drive/wm_poc/videos/r2dreamer/example/rollouts")
    assert paths.figure_dir == Path("/drive/wm_poc/figures/r2dreamer/example/visualizations")
    assert paths.report_dir == Path("/drive/wm_poc/reports/r2dreamer/example/visualizations")
    assert paths.latent_dir == paths.report_dir / "latents"


def test_resolve_log_root_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_LOG_ROOT", "/tmp/custom-run")

    assert resolve_log_root() == Path("/tmp/custom-run")


def test_pca_projection_returns_requested_dimensions() -> None:
    np = pytest.importorskip("numpy")
    matrix = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 1.0],
        ]
    )

    result = pca_projection(matrix, dimensions=3)

    assert result["projected"].shape == (4, 3)
    assert result["components"].shape == (3, 3)
    assert result["explained_variance_ratio"].shape == (3,)
    assert result["explained_variance_ratio"][0] >= result["explained_variance_ratio"][1]


def test_configure_headless_mujoco_overrides_stale_pyopengl_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")

    selected = configure_headless_mujoco("osmesa")

    assert selected == "osmesa"
    assert os.environ["MUJOCO_GL"] == "osmesa"
    assert os.environ["PYOPENGL_PLATFORM"] == "osmesa"
