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
    backup_diff,
    dmc_backup_diff,
    serial_env_backup_diff,
    trainer_backup_diff,
    verify_dmc_render_patch,
    verify_patch,
    verify_serial_env_patch,
    verify_trainer_checkpoint_patch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify external r2dreamer WM POC patches.")
    parser.add_argument("--r2-repo", type=Path, required=True, help="Path to r2dreamer checkout.")
    parser.add_argument("--show-diff", action="store_true", help="Print diff against backup if present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokens = verify_patch(args.r2_repo)
    dmc_tokens = verify_dmc_render_patch(args.r2_repo)
    trainer_tokens = verify_trainer_checkpoint_patch(args.r2_repo)
    serial_env_tokens = verify_serial_env_patch(args.r2_repo)
    print("checkpoint patch verified:", ", ".join(tokens))
    print("dmc render patch verified:", ", ".join(dmc_tokens))
    print("interval checkpoint patch verified:", ", ".join(trainer_tokens))
    print("serial env patch verified:", ", ".join(serial_env_tokens))
    if args.show_diff:
        diff = backup_diff(args.r2_repo)
        dmc_diff = dmc_backup_diff(args.r2_repo)
        trainer_diff = trainer_backup_diff(args.r2_repo)
        serial_env_diff = serial_env_backup_diff(args.r2_repo)
        print(diff if diff else "No checkpoint backup diff available.")
        print(dmc_diff if dmc_diff else "No DMC render backup diff available.")
        print(trainer_diff if trainer_diff else "No trainer checkpoint backup diff available.")
        print(serial_env_diff if serial_env_diff else "No serial env backup diff available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
