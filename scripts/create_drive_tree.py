#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DRIVE_SUBDIRS = [
    "data/dino_wm",
    "data/dmc",
    "data/metaworld",
    "data/robomimic_optional",
    "data/libero_optional",
    "checkpoints/r2dreamer",
    "checkpoints/dino_wm",
    "checkpoints/local_global",
    "logs/r2dreamer",
    "logs/dino_wm",
    "logs/local_global",
    "logs/system",
    "figures/r2dreamer",
    "figures/dino_wm",
    "figures/local_global",
    "tensorboard/r2dreamer",
    "tensorboard/dino_wm",
    "tensorboard/local_global",
    "videos/r2dreamer",
    "videos/dino_wm",
    "videos/local_global",
    "external_repos/r2dreamer",
    "external_repos/dino_wm",
    "external_repos/jepa_wms_optional",
    "reports",
]


def create_drive_tree(drive_root: Path) -> None:
    drive_root = drive_root.expanduser().resolve()
    for subdir in DRIVE_SUBDIRS:
        path = drive_root / subdir
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        status = "existing" if existed else "created"
        print(f"{status}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the WM POC Google Drive folder tree.")
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/wm_poc"),
        help="Root directory for persistent Drive artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_drive_tree(args.drive_root)


if __name__ == "__main__":
    main()
