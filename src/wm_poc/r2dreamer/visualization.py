from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from wm_poc.r2dreamer.commands import DRIVE_ROOT_DEFAULT, nested


DEFAULT_VISUALIZATION_RUN_NAME = "walker_walk_to_run_t4_r2_proprio_12m_seed0"
DEFAULT_R2DREAMER_REPO = "/content/external_repos/r2dreamer"
RUN_NAMES = ("source_base", "target_finetune", "target_scratch")


@dataclass(frozen=True)
class R2RunSpec:
    name: str
    label: str
    task: str
    checkpoint: Path


@dataclass(frozen=True)
class VisualizationPaths:
    run_name: str
    run_root: Path
    video_dir: Path
    figure_dir: Path
    report_dir: Path
    latent_dir: Path


def default_run_name(config: dict[str, Any] | None = None) -> str:
    if "R2_VIS_RUN_NAME" in os.environ:
        return os.environ["R2_VIS_RUN_NAME"]
    if config:
        value = nested(config, "experiment_name")
        if value:
            return str(value)
    return DEFAULT_VISUALIZATION_RUN_NAME


def resolve_drive_root() -> Path:
    return Path(os.environ.get("WM_POC_DRIVE_ROOT", DRIVE_ROOT_DEFAULT)).expanduser()


def resolve_r2dreamer_repo(value: Path | str | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser()
    return Path(os.environ.get("R2DREAMER_REPO", DEFAULT_R2DREAMER_REPO)).expanduser()


def resolve_log_root(
    *,
    config: dict[str, Any] | None = None,
    value: Path | str | None = None,
) -> Path:
    if value is not None:
        return Path(value).expanduser()
    if "R2_LOG_ROOT" in os.environ:
        return Path(os.environ["R2_LOG_ROOT"]).expanduser()
    run_name = default_run_name(config)
    return resolve_drive_root() / "logs" / "r2dreamer" / run_name


def visualization_paths(
    *,
    run_root: Path,
    config: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> VisualizationPaths:
    name = run_name or default_run_name(config) or run_root.name
    drive = resolve_drive_root()

    if "R2_VIDEO_DIR" in os.environ:
        video_dir = Path(os.environ["R2_VIDEO_DIR"]).expanduser()
    else:
        video_dir = drive / "videos" / "r2dreamer" / name / "rollouts"

    if "R2_FIGURE_DIR" in os.environ:
        figure_base = Path(os.environ["R2_FIGURE_DIR"]).expanduser()
    else:
        figure_base = drive / "figures" / "r2dreamer" / name
    figure_dir = (
        figure_base if figure_base.name == "visualizations" else figure_base / "visualizations"
    )

    if "R2_REPORT_DIR" in os.environ:
        report_base = Path(os.environ["R2_REPORT_DIR"]).expanduser()
    else:
        report_base = drive / "reports" / "r2dreamer" / name
    report_dir = (
        report_base if report_base.name == "visualizations" else report_base / "visualizations"
    )

    return VisualizationPaths(
        run_name=name,
        run_root=run_root,
        video_dir=video_dir,
        figure_dir=figure_dir,
        report_dir=report_dir,
        latent_dir=report_dir / "latents",
    )


def ensure_visualization_dirs(paths: VisualizationPaths) -> None:
    for path in (paths.video_dir, paths.figure_dir, paths.report_dir, paths.latent_dir):
        path.mkdir(parents=True, exist_ok=True)


def default_run_specs(config: dict[str, Any], log_root: Path) -> list[R2RunSpec]:
    source_task = str(nested(config, "environment.source_task", "dmc_walker_walk"))
    target_task = str(nested(config, "environment.target_task", "dmc_walker_run"))
    return [
        R2RunSpec(
            name="source_base",
            label=f"Source: {source_task}",
            task=source_task,
            checkpoint=log_root / "source_base" / "latest.pt",
        ),
        R2RunSpec(
            name="target_finetune",
            label=f"Target fine-tune: {target_task}",
            task=target_task,
            checkpoint=log_root / "target_finetune" / "latest.pt",
        ),
        R2RunSpec(
            name="target_scratch",
            label=f"Target scratch: {target_task}",
            task=target_task,
            checkpoint=log_root / "target_scratch" / "latest.pt",
        ),
    ]


def select_run_specs(specs: Iterable[R2RunSpec], selected: str) -> list[R2RunSpec]:
    specs_by_name = {spec.name: spec for spec in specs}
    if selected == "all":
        return [specs_by_name[name] for name in RUN_NAMES if name in specs_by_name]
    if selected not in specs_by_name:
        choices = ", ".join(["all", *specs_by_name])
        raise ValueError(f"Unknown run {selected!r}. Expected one of: {choices}")
    return [specs_by_name[selected]]


def pca_projection(matrix: Any, dimensions: int = 3) -> dict[str, Any]:
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1.")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for latent PCA plotting.") from exc

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {values.shape}.")
    if values.shape[0] < 2:
        raise ValueError("At least two rows are required for PCA.")
    if values.shape[1] < 1:
        raise ValueError("At least one feature column is required for PCA.")
    if not np.isfinite(values).all():
        raise ValueError("PCA input contains NaN or infinite values.")

    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    used = min(dimensions, vh.shape[0])
    components = vh[:used]
    projected = centered @ components.T

    if used < dimensions:
        pad = np.zeros((projected.shape[0], dimensions - used), dtype=projected.dtype)
        projected = np.concatenate([projected, pad], axis=1)
        components = np.concatenate(
            [components, np.zeros((dimensions - used, values.shape[1]), dtype=components.dtype)],
            axis=0,
        )

    variances = singular_values**2 / max(values.shape[0] - 1, 1)
    total = float(variances.sum())
    ratios = variances[:used] / total if total > 0 else np.zeros(used, dtype=np.float64)
    if used < dimensions:
        ratios = np.concatenate([ratios, np.zeros(dimensions - used, dtype=np.float64)])

    return {
        "projected": projected,
        "components": components,
        "explained_variance_ratio": ratios,
        "mean": values.mean(axis=0),
    }
