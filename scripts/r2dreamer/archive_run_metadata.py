#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write small run metadata files into a logdir.")
    parser.add_argument("--logdir", type=Path, required=True, help="Run log directory.")
    parser.add_argument("--command-file", type=Path, help="Optional shell command file to copy.")
    parser.add_argument("--r2-repo", type=Path, help="External r2dreamer checkout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logdir = args.logdir.expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": git_commit(Path.cwd()),
        "r2dreamer_git_commit": git_commit(args.r2_repo) if args.r2_repo else "",
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("WM_POC_") or key.startswith("R2") or key in {"MUJOCO_GL"}
        },
    }
    (logdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.command_file:
        command_text = args.command_file.expanduser().read_text(encoding="utf-8")
        (logdir / "command.sh").write_text(command_text, encoding="utf-8")
    print(f"Wrote metadata to {logdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
