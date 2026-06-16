#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.r2dreamer.commands import load_experiment_config  # noqa: E402
from wm_poc.r2dreamer.visualization import (  # noqa: E402
    pca_projection,
    resolve_log_root,
    visualization_paths,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml"


@dataclass
class LatentSet:
    path: Path
    run_name: str
    label: str
    features: Any
    rewards: Any
    episode: Any
    step: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2D and 3D PCA projections of R2-Dreamer latent trajectories."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Wrapper YAML config."
    )
    parser.add_argument(
        "--log-root", type=Path, help="Run root used to infer default output paths."
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Latent NPZ file from extract_latent_trajectories.py. Can be repeated.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Plot label matching each --input. Defaults to NPZ metadata labels.",
    )
    parser.add_argument(
        "--output-prefix", type=Path, help="Output path prefix without _2d/_3d suffix."
    )
    parser.add_argument(
        "--color-by",
        choices=("run", "step", "reward", "episode"),
        default=None,
        help="Color points by run, step, reward, or episode. Defaults to run for comparisons.",
    )
    parser.add_argument(
        "--max-points-per-run",
        type=int,
        default=5000,
        help="Evenly subsample each run before PCA and plotting.",
    )
    return parser.parse_args()


def load_latent(path: Path, label: str | None = None) -> LatentSet:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required to load latent trajectory files.") from exc

    data = np.load(path.expanduser(), allow_pickle=False)
    metadata: dict[str, Any] = {}
    if "metadata_json" in data:
        metadata_value = data["metadata_json"]
        raw_metadata = metadata_value.item() if metadata_value.shape == () else metadata_value
        metadata = json.loads(str(raw_metadata))
    run_name = str(metadata.get("run_name", path.stem.replace("_latents", "")))
    return LatentSet(
        path=path,
        run_name=run_name,
        label=label or str(metadata.get("label", run_name)),
        features=np.asarray(data["features"], dtype=float),
        rewards=np.asarray(data["rewards"], dtype=float),
        episode=np.asarray(data["episode"], dtype=int),
        step=np.asarray(data["step"], dtype=int),
    )


def default_inputs(latent_dir: Path) -> list[Path]:
    preferred = [
        latent_dir / "target_finetune_latents.npz",
        latent_dir / "target_scratch_latents.npz",
    ]
    if all(path.is_file() for path in preferred):
        return preferred
    return sorted(latent_dir.glob("*_latents.npz"))


def subsample_indices(length: int, max_points: int) -> Any:
    import numpy as np

    if max_points <= 0 or length <= max_points:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=int))


def output_prefix(args: argparse.Namespace, paths: Any, inputs: list[Path]) -> Path:
    if args.output_prefix:
        return args.output_prefix.expanduser()
    stems = {path.stem.replace("_latents", "") for path in inputs}
    if {"target_finetune", "target_scratch"}.issubset(stems):
        return paths.figure_dir / "target_finetune_vs_scratch_latent_pca"
    if len(inputs) == 1 and next(iter(stems)).startswith("target"):
        return paths.figure_dir / "target_latent_pca"
    return paths.figure_dir / "latent_pca"


def color_values(latents: LatentSet, color_by: str, run_index: int) -> Any:
    import numpy as np

    if color_by == "run":
        return np.full(latents.features.shape[0], run_index)
    if color_by == "step":
        return latents.step
    if color_by == "reward":
        return latents.rewards[: latents.features.shape[0]]
    if color_by == "episode":
        return latents.episode
    raise ValueError(color_by)


def plot_projection(
    *,
    projected_sets: list[tuple[LatentSet, Any]],
    color_by: str,
    output: Path,
    is_3d: bool,
    explained: Any,
) -> None:
    import matplotlib.pyplot as plt

    if is_3d:
        fig = plt.figure(figsize=(8.0, 6.2))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(7.8, 5.4))

    colors = ["#2f6f9f", "#bb4e4e", "#2f8f5b", "#8a62b3", "#b8832f"]
    for run_index, (latents, projected) in enumerate(projected_sets):
        if color_by == "run":
            color = colors[run_index % len(colors)]
            for episode_id in sorted(set(latents.episode.tolist())):
                mask = latents.episode == episode_id
                label = latents.label if episode_id == int(latents.episode.min()) else None
                if is_3d:
                    ax.plot(
                        projected[mask, 0],
                        projected[mask, 1],
                        projected[mask, 2],
                        color=color,
                        alpha=0.82,
                        linewidth=1.5,
                        label=label,
                    )
                else:
                    ax.plot(
                        projected[mask, 0],
                        projected[mask, 1],
                        color=color,
                        alpha=0.82,
                        linewidth=1.5,
                        label=label,
                    )
        else:
            values = color_values(latents, color_by, run_index)
            if is_3d:
                points = ax.scatter(
                    projected[:, 0],
                    projected[:, 1],
                    projected[:, 2],
                    c=values,
                    cmap="viridis",
                    s=9,
                    alpha=0.8,
                    label=latents.label,
                )
            else:
                points = ax.scatter(
                    projected[:, 0],
                    projected[:, 1],
                    c=values,
                    cmap="viridis",
                    s=9,
                    alpha=0.8,
                    label=latents.label,
                )
            fig.colorbar(points, ax=ax, shrink=0.75, label=color_by)

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    if is_3d:
        ax.set_zlabel(f"PC3 ({explained[2] * 100:.1f}%)")
    ax.set_title("R2-Dreamer Latent Trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    log_root = resolve_log_root(config=config, value=args.log_root)
    paths = visualization_paths(run_root=log_root, config=config)

    input_paths = args.input or default_inputs(paths.latent_dir)
    if not input_paths:
        raise FileNotFoundError(f"No latent NPZ files found in {paths.latent_dir}.")
    if args.label and len(args.label) != len(input_paths):
        raise SystemExit("--label must be passed once per --input.")

    labels = args.label or [None] * len(input_paths)
    latent_sets = [load_latent(path, label) for path, label in zip(input_paths, labels)]

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for latent PCA plotting.") from exc

    sampled_sets: list[LatentSet] = []
    matrices = []
    for latents in latent_sets:
        idx = subsample_indices(latents.features.shape[0], args.max_points_per_run)
        sampled = LatentSet(
            path=latents.path,
            run_name=latents.run_name,
            label=latents.label,
            features=latents.features[idx],
            rewards=latents.rewards[idx],
            episode=latents.episode[idx],
            step=latents.step[idx],
        )
        sampled_sets.append(sampled)
        matrices.append(sampled.features)

    all_features = np.concatenate(matrices, axis=0)
    pca = pca_projection(all_features, dimensions=3)
    projected = pca["projected"]
    splits = np.cumsum([latents.features.shape[0] for latents in sampled_sets[:-1]])
    projected_chunks = np.split(projected, splits)
    projected_sets = list(zip(sampled_sets, projected_chunks))

    color_by = args.color_by or ("run" if len(sampled_sets) > 1 else "step")
    prefix = output_prefix(args, paths, input_paths)
    output_2d = prefix.with_name(prefix.name + "_2d.png")
    output_3d = prefix.with_name(prefix.name + "_3d.png")
    plot_projection(
        projected_sets=[(latents, points[:, :2]) for latents, points in projected_sets],
        color_by=color_by,
        output=output_2d,
        is_3d=False,
        explained=pca["explained_variance_ratio"],
    )
    plot_projection(
        projected_sets=projected_sets,
        color_by=color_by,
        output=output_3d,
        is_3d=True,
        explained=pca["explained_variance_ratio"],
    )
    report = {
        "inputs": [str(path) for path in input_paths],
        "outputs": [str(output_2d), str(output_3d)],
        "color_by": color_by,
        "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
        "points_per_run": {
            latents.run_name: int(latents.features.shape[0]) for latents in sampled_sets
        },
    }
    report_path = prefix.with_name(prefix.name + "_summary.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_2d}")
    print(f"Wrote {output_3d}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
