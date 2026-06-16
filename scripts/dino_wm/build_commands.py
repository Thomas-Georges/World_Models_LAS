#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.commands import (  # noqa: E402
    build_plan_command,
    build_precompute_command,
    build_train_command,
    render_command,
    write_command_file,
)
from wm_poc.dino_wm.configs import load_config, resolve_config, validate_config  # noqa: E402
from wm_poc.dino_wm.planning import validate_planning_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render DINO-WM upstream commands.")
    parser.add_argument("--config", type=Path, required=True, help="Wrapper YAML config.")
    parser.add_argument("--stage", choices=["train", "plan", "precompute"], required=True)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path for planning.")
    parser.add_argument("--print", action="store_true", dest="print_command", help="Print command.")
    parser.add_argument("--write-command", type=Path, help="Optional command.sh destination.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    validate_config(config)

    if args.stage == "train":
        argv = build_train_command(config)
    elif args.stage == "plan":
        validate_planning_config(config)
        checkpoint_path = "" if args.checkpoint is None else str(args.checkpoint.expanduser())
        argv = build_plan_command(config, checkpoint_path)
    else:
        argv = build_precompute_command(config)

    if args.write_command:
        write_command_file(argv, args.write_command)
    if args.print_command or not args.write_command:
        print(render_command(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
