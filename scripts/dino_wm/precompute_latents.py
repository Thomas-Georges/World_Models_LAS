#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.commands import (  # noqa: E402
    build_precompute_command,
    latent_cache_dir,
    latent_training_enabled,
    render_command,
)
from wm_poc.dino_wm.configs import get_config_value, load_config, resolve_config, validate_config  # noqa: E402
from wm_poc.dino_wm.data import build_split_manifest, write_split_manifest  # noqa: E402
from wm_poc.dino_wm.latent_cache_patch import LATENT_MANIFEST_NAME, install_latent_support  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen DINO-WM latents.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _upstream_repo(config: dict) -> Path:
    repo = Path(str(config.get("external_repo", "external_repos/dino_wm"))).expanduser()
    return repo if repo.is_absolute() else (REPO_ROOT / repo)


def _required_rollouts(config: dict) -> int:
    train = int(get_config_value(config, "dataset.max_train_trajectories"))
    val = int(get_config_value(config, "dataset.max_val_trajectories"))
    return train + val


def _cache_covers_required(config: dict) -> bool:
    manifest_path = Path(latent_cache_dir(config)) / LATENT_MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    covered = int(manifest.get("num_episodes", 0))
    if covered >= _required_rollouts(config):
        return True
    # A request larger than the raw dataset clamps to the dataset size; once
    # every available episode is encoded the cache cannot grow further.
    dataset_episodes = manifest.get("dataset_episodes")
    return dataset_episodes is not None and covered >= int(dataset_episodes)


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    validate_config(config)

    if not latent_training_enabled(config):
        env_name = get_config_value(config, "dataset.env")
        print(
            "Latent caching is disabled for this config "
            f"(features.cache_enabled is false or env {env_name!r} is unsupported); nothing to do."
        )
        return 0

    cache_dir = Path(latent_cache_dir(config))
    wrapper_manifest_path = (
        Path(str(get_config_value(config, "features.cache_dir"))).expanduser()
        / str(get_config_value(config, "dataset.env"))
        / "feature_cache_manifest.yaml"
    )
    command = build_precompute_command(config)
    if args.force:
        command.append("--force")

    manifest = build_split_manifest(config)
    manifest["complete"] = False
    manifest["dry_run"] = bool(args.dry_run)
    manifest["command"] = render_command(command)
    manifest["latent_cache_dir"] = str(cache_dir)
    manifest["allow_incomplete_cache"] = bool(get_config_value(config, "features.allow_incomplete_cache", False))

    if args.dry_run:
        print("Dry run only. Set --no-dry-run and RUN_DINO_WM=1 to execute latent precompute.")
        print(render_command(command))
        return 0
    if not args.force and _cache_covers_required(config):
        print(f"Latent cache already covers this config: {cache_dir / LATENT_MANIFEST_NAME}. Use --force to regenerate.")
        return 0
    if os.environ.get("RUN_DINO_WM") != "1":
        print("Refusing to execute because RUN_DINO_WM is not 1.", file=sys.stderr)
        return 2

    installed = install_latent_support(_upstream_repo(config))
    for name in installed:
        print(f"Installed DINO-WM latent support file: {name}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(render_command(command), flush=True)
    result = subprocess.run(command, check=False)
    manifest["complete"] = result.returncode == 0
    write_split_manifest(manifest, wrapper_manifest_path)
    print(f"Wrote {wrapper_manifest_path}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
