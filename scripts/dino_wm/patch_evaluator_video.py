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
from wm_poc.dino_wm.evaluator_video_patch import PATCH_MARKER, patch_evaluator_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch upstream DINO-WM planning/evaluator.py to record videos for decoder-free models."
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
    evaluator_path = upstream / "planning" / "evaluator.py"
    if not evaluator_path.is_file():
        print(f"Upstream DINO-WM planning/evaluator.py is missing: {evaluator_path}")
        return 1

    if args.verify_only:
        if PATCH_MARKER not in evaluator_path.read_text(encoding="utf-8"):
            print(f"DINO-WM evaluator video patch is not applied: {evaluator_path}")
            return 1
        print(f"DINO-WM evaluator video patch verified: {evaluator_path}")
        return 0

    if patch_evaluator_file(evaluator_path):
        print(f"Applied DINO-WM evaluator video patch: {evaluator_path}")
    else:
        print(f"DINO-WM evaluator video patch already applied: {evaluator_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
