#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.r2dreamer.commands import build_all_commands, format_commands, load_experiment_config  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/r2dreamer/three_way_walker_walk_to_run.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R2-Dreamer training commands.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Wrapper YAML config.")
    parser.add_argument(
        "--run",
        choices=["smoke", "source_base", "target_finetune", "target_scratch"],
        help="Only print or execute one run command.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and exit.")
    parser.add_argument("--print-only", action="store_true", help="Print commands and exit.")
    parser.add_argument("--execute", action="store_true", help="Execute the selected command.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    commands = build_all_commands(config)
    rendered = format_commands(commands, run=args.run)

    if args.execute:
        if args.run is None:
            raise SystemExit("--execute requires --run.")
        if os.environ.get("RUN_TRAINING") != "1":
            print(rendered)
            print("Refusing to execute because RUN_TRAINING is not 1.", file=sys.stderr)
            return 2
        return subprocess.run(["bash", "-lc", commands[args.run]], check=False).returncode

    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
