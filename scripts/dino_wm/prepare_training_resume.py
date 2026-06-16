#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.configs import load_config, resolve_config, validate_config  # noqa: E402
from wm_poc.dino_wm.resume import prepare_training_resume  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DINO-WM checkpoint resume state.")
    parser.add_argument("--config", type=Path, required=True, help="Wrapper YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    validate_config(config)
    state = prepare_training_resume(config)
    action = state["action"]
    if action == "resume":
        print(f"Resume enabled; loading latest DINO-WM checkpoint: {state['checkpoint_path']}")
    elif action == "fresh_start_backup":
        print(
            "Fresh DINO-WM start requested; moved existing checkpoints "
            f"from {state['checkpoint_dir']} to {state['backup_dir']} "
            f"({state['reason']})."
        )
    else:
        print(f"No latest DINO-WM checkpoint found; starting from scratch: {state['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
