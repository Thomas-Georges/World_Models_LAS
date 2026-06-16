#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.manifests import create_run_manifest, write_json_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an empty run manifest template.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file path.")
    parser.add_argument("--track", required=True, help="Experiment track name.")
    parser.add_argument("--run-name", required=True, help="Human-readable run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_run_manifest(track=args.track, run_name=args.run_name, cwd=REPO_ROOT)
    write_json_manifest(manifest, args.output)
    print(f"Wrote run manifest to {args.output.expanduser()}")


if __name__ == "__main__":
    main()
