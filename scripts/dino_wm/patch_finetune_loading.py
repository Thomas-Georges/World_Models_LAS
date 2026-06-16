#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.configs import load_config, resolve_config  # noqa: E402
from wm_poc.dino_wm.finetune_init_patch import PATCH_MARKER, patch_train_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch upstream DINO-WM train.py to initialize fine-tune runs from a source checkpoint."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/dino_wm/base.yaml")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "patches/dino_wm/patch_manifest.json")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    upstream = Path(str(config.get("external_repo", "external_repos/dino_wm"))).expanduser()
    if not upstream.is_absolute():
        upstream = REPO_ROOT / upstream
    train_path = upstream / "train.py"
    if not train_path.is_file():
        print(f"Upstream DINO-WM train.py is missing: {train_path}")
        return 1

    if args.verify_only:
        if PATCH_MARKER not in train_path.read_text(encoding="utf-8"):
            print(f"DINO-WM fine-tune init patch is not applied: {train_path}")
            return 1
        print(f"DINO-WM fine-tune init patch verified: {train_path}")
        return 0

    changed = patch_train_file(train_path)
    if changed:
        print(f"Applied DINO-WM fine-tune init patch: {train_path}")
    else:
        print(f"DINO-WM fine-tune init patch already applied: {train_path}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "upstream_patch",
        "patched_files": [str(train_path)],
        "marker": PATCH_MARKER,
        "notes": [
            "init_models now ends with _wm_poc_apply_finetune_init(), which loads",
            "predictor/action_encoder/proprio_encoder (optionally decoder) weights",
            "from ++finetuning.init_from for fresh fine-tune runs; resume wins.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
