#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.r2dreamer.patching import (  # noqa: E402
    patch_dmc_rendering,
    patch_serial_envs,
    patch_train_py,
    patch_trainer_interval_checkpoints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch external r2dreamer for WM POC runs.")
    parser.add_argument("--r2-repo", type=Path, required=True, help="Path to r2dreamer checkout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_status = patch_train_py(args.r2_repo)
    dmc_status = patch_dmc_rendering(args.r2_repo)
    trainer_status = patch_trainer_interval_checkpoints(args.r2_repo)
    serial_env_status = patch_serial_envs(args.r2_repo)
    print(f"checkpoint patch status: {checkpoint_status}")
    print(f"dmc render patch status: {dmc_status}")
    print(f"interval checkpoint patch status: {trainer_status}")
    print(f"serial env patch status: {serial_env_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
