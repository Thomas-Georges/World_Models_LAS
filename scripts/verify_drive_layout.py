#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from create_drive_tree import DRIVE_SUBDIRS
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Run this script from the repository root or scripts directory.") from exc


def verify_drive_layout(drive_root: Path, dry_run: bool) -> int:
    drive_root = drive_root.expanduser().resolve()
    missing: list[Path] = []

    print(f"Drive root: {drive_root}")
    for subdir in DRIVE_SUBDIRS:
        path = drive_root / subdir
        if dry_run:
            action = "exists" if path.exists() else "would check/create"
            print(f"{action}: {path}")
        elif path.is_dir():
            print(f"ok: {path}")
        else:
            print(f"missing: {path}")
            missing.append(path)

    if dry_run:
        return 0
    if missing:
        print(f"ERROR: {len(missing)} expected directories are missing.")
        return 1
    print("Drive layout verified.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the WM POC Drive folder tree.")
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/wm_poc"),
        help="Root directory for persistent Drive artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expected checks without failing for missing directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return verify_drive_layout(args.drive_root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
