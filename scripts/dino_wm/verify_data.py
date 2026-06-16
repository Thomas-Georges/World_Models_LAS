#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.configs import get_config_value, load_config, resolve_config, validate_config  # noqa: E402
from wm_poc.dino_wm.data import (  # noqa: E402
    build_split_manifest,
    validate_dataset_root,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DINO-WM dataset layout.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    validate_config(config)
    root = get_config_value(config, "dataset.root")
    env = get_config_value(config, "dataset.env")
    validate_dataset_root(root, env)
    print(f"Dataset root: {root}", flush=True)
    print(f"Environment: {env}", flush=True)
    print(
        f"Building split manifest with checksum_mode={get_config_value(config, 'dataset.checksum_mode', 'metadata')}...",
        flush=True,
    )
    manifest = build_split_manifest(config)
    print(f"Available trajectory-like files: {manifest['num_available_files']}", flush=True)
    print(f"Selected train files: {manifest['num_train_files']}", flush=True)
    print(f"Selected val files: {manifest['num_val_files']}", flush=True)

    if args.write_manifest:
        run_name = str(config.get("run_name"))
        output = Path(str(get_config_value(config, "artifacts.log_root"))) / run_name / "split_manifest.yaml"
        write_split_manifest(manifest, output)
        print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
