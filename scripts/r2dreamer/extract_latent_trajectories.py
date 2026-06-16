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
    compose_upstream_config,
    configure_headless_mujoco,
    load_agent_and_env,
    run_policy_rollouts,
    save_latents_npz,
)
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
        description="Extract recurrent latent trajectories from R2-Dreamer policy rollouts."
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
    parser.add_argument(
        "--output-dir", type=Path, help="Directory for latent trajectory NPZ files."
    )
    parser.add_argument(
        "--run",
        choices=("all", *RUN_NAMES),
        default="all",
        help="Which default checkpoint to evaluate.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Extract one custom checkpoint.")
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
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per checkpoint.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
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


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    log_root = resolve_log_root(config=config, value=args.log_root)
    paths = visualization_paths(run_root=log_root, config=config)
    ensure_visualization_dirs(paths)
    output_dir = args.output_dir.expanduser() if args.output_dir else paths.latent_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env_name = args.env_name or str(nested(config, "environment.env", "dmc_proprio"))
    model = args.model or str(nested(config, "algorithm.model", "size12M"))
    rep_loss = args.rep_loss or str(nested(config, "algorithm.rep_loss", "r2dreamer"))
    seed = args.seed if args.seed is not None else int(nested(config, "training.seed", 0))
    device = args.device or default_device()
    r2_repo = resolve_r2dreamer_repo(args.r2_repo)
    mujoco_gl = configure_headless_mujoco(args.mujoco_gl)
    print(f"Using MUJOCO_GL={mujoco_gl}", flush=True)
    print(f"Using PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}", flush=True)

    if env_name == "dmc_proprio" and "WM_POC_DMC_DISABLE_IMAGE_RENDER" not in os.environ:
        os.environ["WM_POC_DMC_DISABLE_IMAGE_RENDER"] = "true"

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

        print(f"Extracting latents for {spec.name}: {spec.checkpoint}")
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
            collect_frames=False,
            collect_latents=True,
        )
        latent_path = output_dir / f"{spec.name}_latents.npz"
        metadata = {
            "run_name": spec.name,
            "label": spec.label,
            "task": spec.task,
            "checkpoint": str(spec.checkpoint),
            "env": env_name,
            "model": model,
            "rep_loss": rep_loss,
            "seed": seed,
            "device": device,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "load_info": loaded.load_info,
        }
        save_latents_npz(rollout=rollout, path=latent_path, metadata=metadata)
        summary = {
            **metadata,
            "latent_path": str(latent_path),
            "steps_collected": rollout.length,
            "return_total": rollout.total_return,
        }
        summaries.append(summary)
        print(f"Wrote {latent_path}")

    summary_path = paths.report_dir / "latent_trajectories.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
