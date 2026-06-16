#!/usr/bin/env python
"""Evaluate local/global planners on offline latent goal-reaching episodes.

Episodes are sampled from held-out validation episodes of the latent cache;
the global model acts as MPC simulator and scorer (see
``wm_poc/local_global/eval.py`` for the caveats). Writes
``planning/<planner>/{episodes.jsonl,summary.json,traces/}`` under the run dir.

Smoke runs (``--smoke``) shrink every optimizer and are always allowed; full
runs are gated on ``RUN_LOCAL_GLOBAL=1``.

Example:
    python scripts/local_global/run_planning_eval.py \
        --config configs/local_global/pointmaze_surrogate_a100.yaml \
        --run-dir "$LG_RUN_ROOT/pointmaze_local_v1" \
        --planners global_cem local_adam hybrid_cem_local_refine_global_rescore \
        --num-episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wm_poc.dino_wm.configs import get_config_value, set_config_value  # noqa: E402
from wm_poc.local_global.configs import (  # noqa: E402
    RUN_GATE_ENV,
    VALID_PLANNERS,
    load_local_global_config,
    resolve_run_dir,
    save_resolved_config,
)
from wm_poc.local_global.runtime import setup_mujoco_runtime  # noqa: E402

SMOKE_OVERRIDES = (
    ("planning.cem_population", 16),
    ("planning.cem_elites", 4),
    ("planning.cem_iters", 2),
    ("planning.gd_iters", 5),
    ("evaluation.num_episodes", 2),
    ("training.max_episodes", 8),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["global_cem", "local_adam", "hybrid_cem_local_refine_global_rescore"],
        choices=list(VALID_PLANNERS),
    )
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument(
        "--max-wall-minutes",
        type=float,
        default=None,
        help="wall-clock cap (default evaluation.max_wall_minutes, 110)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None, help="local surrogate checkpoint override")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_local_global_config(args.config)
    if args.smoke:
        for key, value in SMOKE_OVERRIDES:
            set_config_value(config, key, value)
    print(f"Config: {args.config} | task={config['task']} | planners={args.planners}")
    if args.dry_run:
        print("Dry run: not evaluating.")
        return 0
    if not args.smoke and os.environ.get(RUN_GATE_ENV) != "1":
        print(
            f"Planning evaluation is disabled for safety. Set {RUN_GATE_ENV}=1 to launch, "
            "or pass --smoke for a tiny run."
        )
        return 0

    # The DINO-WM global model is loaded in-process, which imports the upstream
    # plan.py -> mujoco_py. Configure mujoco's runtime env (and re-exec once so
    # the linker sees LD_LIBRARY_PATH) BEFORE importing torch/eval. No-op for
    # the synthetic model and on boxes without a mujoco install.
    if str(get_config_value(config, "global_model.source", "dino_wm")) == "dino_wm":
        setup_mujoco_runtime()

    import time

    import torch

    from wm_poc.local_global.datasets import ensure_synthetic_task_data
    from wm_poc.local_global.eval import evaluate_planner

    ensure_synthetic_task_data(config)
    # Honor the config's device (like training does): --device > config.device >
    # auto. The synthetic smoke pins device: cpu, so it must not be forced onto
    # cuda; the real run leaves device: auto -> cuda on a GPU.
    requested = args.device or str(get_config_value(config, "device", "auto"))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    run_dir = resolve_run_dir(config, args.run_dir)
    save_resolved_config(config, run_dir)

    # The same experiment runs on any GPU; slower cards just hit this wall
    # clock sooner. A capped planner writes summary_partial.json (not done),
    # so re-running the eval finishes the remaining planners/episodes.
    max_wall_minutes = float(
        args.max_wall_minutes
        if args.max_wall_minutes is not None
        else get_config_value(config, "evaluation.max_wall_minutes", 110)
    )
    deadline = time.perf_counter() + max_wall_minutes * 60

    summaries = []
    for planner_name in args.planners:
        if time.perf_counter() > deadline:
            print(
                f"Wall-clock limit ({max_wall_minutes:g} min) reached; remaining planners "
                f"not started: re-run this command to continue."
            )
            break
        print(f"--- evaluating {planner_name} (device={device}) ---")
        summary = evaluate_planner(
            config,
            planner_name,
            run_dir,
            num_episodes=args.num_episodes,
            device=device,
            checkpoint=args.checkpoint,
            deadline=deadline,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    print(f"Wrote planner outputs under {run_dir / 'planning'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
