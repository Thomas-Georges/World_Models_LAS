#!/usr/bin/env python
"""Export a local/global transition-dataset manifest from a DINO latent cache.

Verifies the latent cache + action tensors line up, materializes the episode
split and window index, and writes ``manifest.json`` / ``dataset_stats.json``
under the run's ``transition_data`` directory. For the synthetic task it can
generate the tiny latent cache first (CPU-only smoke path).

Example:
    python scripts/local_global/export_transitions.py \
        --config configs/local_global/smoke_pointmaze.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wm_poc.dino_wm.configs import get_config_value  # noqa: E402
from wm_poc.local_global.configs import (  # noqa: E402
    action_data_dir,
    latent_cache_dir,
    load_local_global_config,
    resolve_run_dir,
    save_resolved_config,
)
from wm_poc.local_global.datasets import (  # noqa: E402
    LATENT_MANIFEST_NAME,
    LatentTrajectoryStore,
    ensure_synthetic_task_data,
    export_transition_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="local_global YAML config")
    parser.add_argument("--run-dir", default=None, help="run directory (default from config)")
    parser.add_argument("--out", default=None, help="output dir (default <run-dir>/transition_data)")
    parser.add_argument("--max-windows", type=int, default=0, help="cap windows (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="verify inputs, write nothing")
    args = parser.parse_args()

    config = load_local_global_config(args.config)
    cache_dir = latent_cache_dir(config)
    action_dir = action_data_dir(config)
    print(f"Task:          {config['task']}")
    print(f"Latent cache:  {cache_dir}")
    print(f"Action data:   {action_dir}")

    if args.dry_run:
        manifest = cache_dir / LATENT_MANIFEST_NAME
        print(f"Cache manifest present: {manifest.is_file()} ({manifest})")
        if not manifest.is_file() and config.get("task") != "synthetic":
            print(
                "To create it: python scripts/dino_wm/precompute_latents.py "
                "--config configs/dino_wm/pointmaze_full_nodecoder_bf16.yaml --no-dry-run"
            )
        print("Dry run: no files written.")
        return 0

    if ensure_synthetic_task_data(config):
        print(f"Generated synthetic latent task under {cache_dir.parent}")
    try:
        store = LatentTrajectoryStore(
            cache_dir,
            action_dir,
            max_episodes=int(get_config_value(config, "training.max_episodes", 0)),
        )
    except FileNotFoundError as exc:
        print(f"Cannot export transitions: {exc}")
        print(
            "Create the cache first with scripts/dino_wm/precompute_latents.py "
            "(see the --dry-run output of this script for the exact command)."
        )
        return 1
    run_dir = resolve_run_dir(config, args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "transition_data"
    manifest = export_transition_manifest(
        store,
        out_dir,
        context_len=int(get_config_value(config, "local_model.context_len", 2)),
        rollout_steps=int(get_config_value(config, "local_model.rollout_steps", 3)),
        frameskip=int(get_config_value(config, "global_model.frameskip", 1)),
        val_fraction=float(get_config_value(config, "training.val_fraction", 0.1)),
        split_seed=int(get_config_value(config, "training.split_seed", 42)),
        max_windows=args.max_windows,
    )
    save_resolved_config(config, run_dir)
    print(
        f"Exported manifest: {out_dir / 'manifest.json'} "
        f"({manifest['num_train_windows']} train / {manifest['num_val_windows']} val windows, "
        f"{manifest['num_episodes']} episodes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
