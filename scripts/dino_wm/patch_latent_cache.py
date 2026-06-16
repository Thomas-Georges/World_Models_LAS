#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.configs import load_config, resolve_config  # noqa: E402
from wm_poc.dino_wm.latent_cache_patch import (  # noqa: E402
    LATENT_DATASET_MODULE_NAME,
    MODEL_PATCH_MARKER,
    PRECOMPUTE_SCRIPT_NAME,
    install_latent_support,
    patch_model_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install DINO-WM latent cache support into the upstream checkout."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/dino_wm/base.yaml")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    upstream = Path(str(config.get("external_repo", "external_repos/dino_wm"))).expanduser()
    if not upstream.is_absolute():
        upstream = REPO_ROOT / upstream
    model_path = upstream / "models" / "visual_world_model.py"
    if not model_path.is_file():
        print(f"Upstream DINO-WM visual_world_model.py is missing: {model_path}")
        return 1

    if args.verify_only:
        missing = [
            name
            for name in (LATENT_DATASET_MODULE_NAME, PRECOMPUTE_SCRIPT_NAME)
            if not (upstream / name).is_file()
        ]
        if missing:
            print(f"DINO-WM latent support files are missing: {', '.join(missing)}")
            return 1
        if MODEL_PATCH_MARKER not in model_path.read_text(encoding="utf-8"):
            print(f"DINO-WM latent bypass patch is not applied: {model_path}")
            return 1
        print(f"DINO-WM latent cache support verified: {upstream}")
        return 0

    installed = install_latent_support(upstream)
    for name in installed:
        print(f"Installed DINO-WM latent support file: {upstream / name}")
    if not installed:
        print(f"DINO-WM latent support files already up to date: {upstream}")

    if patch_model_file(model_path):
        print(f"Applied DINO-WM latent bypass patch: {model_path}")
    else:
        print(f"DINO-WM latent bypass patch already applied: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
