#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.r2dreamer.commands import load_experiment_config, nested  # noqa: E402
from wm_poc.r2dreamer.posthoc import (  # noqa: E402
    RolloutData,
    compose_upstream_config,
    configure_headless_mujoco,
    load_agent_and_env,
    run_policy_rollouts,
    save_video,
)
from wm_poc.r2dreamer.patching import patch_dmc_rendering  # noqa: E402
from wm_poc.r2dreamer.visualization import pca_projection  # noqa: E402
from wm_poc.r2dreamer.visualization import (  # noqa: E402
    RUN_NAMES,
    R2RunSpec,
    default_run_specs,
    ensure_visualization_dirs,
    resolve_log_root,
    resolve_r2dreamer_repo,
    visualization_paths,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render post-hoc policy rollout videos from R2-Dreamer checkpoints."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Wrapper YAML config."
    )
    parser.add_argument(
        "--r2-repo", type=Path, help="Path to the upstream NM512/r2dreamer checkout."
    )
    parser.add_argument(
        "--log-root", type=Path, help="Run root containing source_base/target_* folders."
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for MP4 rollout videos.")
    parser.add_argument(
        "--run",
        choices=("all", *RUN_NAMES),
        default="all",
        help="Which default checkpoint to render.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Render one custom checkpoint.")
    parser.add_argument("--task", help="Task for --checkpoint, for example dmc_walker_run.")
    parser.add_argument("--name", default="custom", help="Output stem for --checkpoint.")
    parser.add_argument("--env", dest="env_name", help="Upstream env config, e.g. dmc_proprio.")
    parser.add_argument("--model", help="Upstream model config, e.g. size12M.")
    parser.add_argument("--rep-loss", help="Representation loss used by the checkpoint.")
    parser.add_argument("--seed", type=int, help="Evaluation seed.")
    parser.add_argument("--device", help="Torch device. Defaults to R2_DEVICE or auto-detected.")
    parser.add_argument(
        "--mujoco-gl",
        choices=("auto", "egl", "osmesa", "glfw"),
        default=os.environ.get("R2_MUJOCO_GL", "auto"),
        help="MuJoCo OpenGL backend. Prefer osmesa on Colab; use egl only after validating it.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per checkpoint.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frames per second.")
    parser.add_argument(
        "--render-mode",
        choices=("auto", "camera", "trace"),
        default=os.environ.get("R2_RENDER_MODE", "auto"),
        help=(
            "camera writes environment RGB frames; trace avoids MuJoCo camera rendering and "
            "writes latent/reward/action diagnostics."
        ),
    )
    parser.add_argument(
        "--trace-max-frames",
        type=int,
        default=300,
        help="Maximum frames to render for trace-mode diagnostic videos.",
    )
    parser.add_argument(
        "--skip-missing", action="store_true", help="Skip missing default checkpoints."
    )
    parser.add_argument(
        "--no-strict", action="store_true", help="Allow non-strict checkpoint loading."
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional upstream Hydra override. Can be passed more than once.",
    )
    return parser.parse_args()


def default_device() -> str:
    if "R2_DEVICE" in os.environ:
        return os.environ["R2_DEVICE"]
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def custom_specs(args: argparse.Namespace) -> list[R2RunSpec] | None:
    if args.checkpoint is None:
        return None
    if not args.task:
        raise SystemExit("--checkpoint requires --task.")
    return [
        R2RunSpec(
            name=args.name,
            label=args.name,
            task=args.task,
            checkpoint=args.checkpoint,
        )
    ]


def video_name(run_name: str, episode_count: int) -> str:
    if episode_count == 1:
        return f"{run_name}_policy_rollout.mp4"
    return f"{run_name}_policy_rollouts_{episode_count}eps.mp4"


def looks_like_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").is_dir()


def resolve_render_mode(requested: str, env_name: str) -> str:
    if requested != "auto":
        return requested
    if env_name == "dmc_proprio" and looks_like_colab():
        return "trace"
    return "camera"


def save_trace_video(
    rollout: RolloutData,
    path: Path,
    *,
    fps: int,
    max_frames: int,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib and NumPy are required for trace-mode rollout videos."
        ) from exc

    if not rollout.features:
        raise ValueError("Trace-mode video requires collected latent features.")

    features = np.asarray(rollout.features, dtype=np.float64)
    actions = np.asarray(rollout.actions, dtype=np.float64)
    rewards = np.asarray(rollout.rewards, dtype=np.float64)[: features.shape[0]]
    if features.shape[0] < 2:
        projected = np.zeros((features.shape[0], 2), dtype=np.float64)
    else:
        projected = pca_projection(features, dimensions=2)["projected"]
    cumulative = np.cumsum(rewards)
    action_norm = np.linalg.norm(actions, axis=1) if actions.size else np.zeros(features.shape[0])

    total = features.shape[0]
    if max_frames > 0 and total > max_frames:
        frame_indices = np.unique(np.linspace(0, total - 1, max_frames, dtype=int))
    else:
        frame_indices = np.arange(total)

    frames = []
    x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
    y_min, y_max = projected[:, 1].min(), projected[:, 1].max()
    x_pad = max((x_max - x_min) * 0.08, 1e-6)
    y_pad = max((y_max - y_min) * 0.08, 1e-6)
    for idx in frame_indices:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), dpi=120)
        ax_path, ax_metrics = axes
        ax_path.plot(projected[: idx + 1, 0], projected[: idx + 1, 1], color="#2f6f9f", lw=2)
        ax_path.scatter(projected[idx, 0], projected[idx, 1], color="#bb4e4e", s=30)
        ax_path.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_path.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_path.set_title("RSSM feature PCA")
        ax_path.set_xlabel("PC1")
        ax_path.set_ylabel("PC2")
        ax_path.grid(True, alpha=0.25)

        steps = np.arange(idx + 1)
        ax_metrics.plot(steps, cumulative[: idx + 1], color="#2f8f5b", label="return")
        ax_metrics.set_xlabel("step")
        ax_metrics.set_title(title)
        ax_metrics.grid(True, alpha=0.25)
        ax2 = ax_metrics.twinx()
        ax2.plot(steps, action_norm[: idx + 1], color="#b8832f", alpha=0.8, label="|action|")
        ax2.set_ylabel("|action|")
        ax_metrics.set_ylabel("return")
        ax_metrics.set_xlim(0, max(total - 1, 1))
        fig.tight_layout()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    save_video(frames, path, fps=fps)


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    log_root = resolve_log_root(config=config, value=args.log_root)
    paths = visualization_paths(run_root=log_root, config=config)
    ensure_visualization_dirs(paths)
    output_dir = args.output_dir.expanduser() if args.output_dir else paths.video_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env_name = args.env_name or str(nested(config, "environment.env", "dmc_proprio"))
    model = args.model or str(nested(config, "algorithm.model", "size12M"))
    rep_loss = args.rep_loss or str(nested(config, "algorithm.rep_loss", "r2dreamer"))
    seed = args.seed if args.seed is not None else int(nested(config, "training.seed", 0))
    device = args.device or default_device()
    r2_repo = resolve_r2dreamer_repo(args.r2_repo)
    mujoco_gl = configure_headless_mujoco(args.mujoco_gl)
    render_mode = resolve_render_mode(args.render_mode, env_name)

    if render_mode == "trace":
        if env_name != "dmc_proprio":
            raise RuntimeError("Trace render mode is only safe for dmc_proprio checkpoints.")
        patch_status = patch_dmc_rendering(r2_repo)
        print(f"DMC render patch status: {patch_status}", flush=True)
        os.environ["WM_POC_DMC_DISABLE_IMAGE_RENDER"] = "true"
    else:
        os.environ["WM_POC_DMC_DISABLE_IMAGE_RENDER"] = "false"
    print(f"Using MUJOCO_GL={mujoco_gl}", flush=True)
    print(f"Using PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}", flush=True)
    print(f"Using render_mode={render_mode}", flush=True)

    specs = custom_specs(args)
    if specs is None:
        specs = default_run_specs(config, log_root)
        specs = [spec for spec in specs if args.run == "all" or spec.name == args.run]

    summaries: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.checkpoint.expanduser().is_file():
            message = f"Missing checkpoint for {spec.name}: {spec.checkpoint}"
            if args.skip_missing:
                print(message)
                continue
            raise FileNotFoundError(message)

        print(f"Rendering {spec.name}: {spec.checkpoint}")
        upstream_config = compose_upstream_config(
            r2_repo=r2_repo,
            env_name=env_name,
            task=spec.task,
            model=model,
            rep_loss=rep_loss,
            seed=seed,
            device=device,
            extra_overrides=args.extra_override,
        )
        loaded = load_agent_and_env(
            r2_repo=r2_repo,
            checkpoint=spec.checkpoint,
            config=upstream_config,
            strict=not args.no_strict,
        )
        rollout = run_policy_rollouts(
            loaded=loaded,
            episodes=args.episodes,
            max_steps=args.max_steps,
            collect_frames=render_mode == "camera",
            collect_latents=render_mode == "trace",
        )
        video_path = output_dir / video_name(spec.name, args.episodes)
        if render_mode == "trace":
            save_trace_video(
                rollout,
                video_path,
                fps=args.fps,
                max_frames=args.trace_max_frames,
                title=f"{spec.name}: {spec.task}",
            )
        else:
            save_video(rollout.frames, video_path, fps=args.fps)
        summary = {
            "run_name": spec.name,
            "label": spec.label,
            "task": spec.task,
            "checkpoint": str(spec.checkpoint),
            "video": str(video_path),
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "steps_collected": rollout.length,
            "return_total": rollout.total_return,
            "render_mode": render_mode,
            "load_info": loaded.load_info,
        }
        summaries.append(summary)
        print(f"Wrote {video_path}")

    summary_path = paths.report_dir / "policy_rollout_videos.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
