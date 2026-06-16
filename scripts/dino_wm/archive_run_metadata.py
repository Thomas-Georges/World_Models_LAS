#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.commands import (  # noqa: E402
    build_plan_command,
    build_precompute_command,
    build_train_command,
    write_command_file,
)
from wm_poc.dino_wm.configs import (  # noqa: E402
    copy_config_artifacts,
    get_config_value,
    load_config,
    make_run_dir,
    resolve_config,
    validate_config,
)
from wm_poc.dino_wm.data import build_split_manifest, write_split_manifest  # noqa: E402


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_dirty(path: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else False


def _torch_metadata() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": "",
        "cuda_available": False,
        "gpu_name": "",
        "gpu_count": 0,
    }
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["gpu_count"] = int(torch.cuda.device_count())
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def _safe_env() -> dict[str, str]:
    blocked = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")
    allowed_prefixes = ("DINO_", "WM_POC_", "RUN_DINO_WM")
    result = {}
    for key, value in os.environ.items():
        if not key.startswith(allowed_prefixes):
            continue
        if any(marker in key.upper() for marker in blocked):
            continue
        result[key] = value
    return result


def _metadata(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    upstream = Path(str(config.get("external_repo", ""))).expanduser()
    if upstream and not upstream.is_absolute():
        upstream = REPO_ROOT / upstream
    data = {
        "run_name": config.get("run_name"),
        "track": "dino_wm",
        "project_git_commit": _git_commit(REPO_ROOT),
        "project_git_dirty": _git_dirty(REPO_ROOT),
        "upstream_dino_wm_commit": _git_commit(upstream) if upstream.is_dir() else "",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "config_path": str(config_path),
        "dataset_root": get_config_value(config, "dataset.root"),
        "artifact_roots": {
            "log_root": get_config_value(config, "artifacts.log_root"),
            "ckpt_root": get_config_value(config, "artifacts.ckpt_root"),
            "figure_dir": get_config_value(config, "artifacts.figure_dir"),
            "video_dir": get_config_value(config, "artifacts.video_dir"),
        },
        "environment": _safe_env(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    data.update(_torch_metadata())
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _command_for_stage(config: dict[str, Any], stage: str, checkpoint: Path | None) -> list[str]:
    if stage in {"train", "finetune"}:
        return build_train_command(config)
    if stage == "plan":
        # An empty --checkpoint argument arrives as Path("") == Path("."),
        # which means "no checkpoint given".
        if checkpoint is None or str(checkpoint) in {"", "."}:
            checkpoint_path = ""
        else:
            checkpoint_path = str(checkpoint.expanduser())
        return build_plan_command(config, checkpoint_path)
    if stage == "precompute":
        return build_precompute_command(config)
    raise ValueError(f"Unknown stage: {stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DINO-WM run metadata files.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["train", "finetune", "plan", "precompute"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--print-run-dir", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))
    validate_config(config)
    run_dir = make_run_dir(config)
    copy_config_artifacts(args.config, run_dir, config)

    argv = _command_for_stage(config, args.stage, args.checkpoint)
    command_path = run_dir / "command.sh"
    metadata_path = run_dir / "metadata.json"
    if args.stage == "plan" and command_path.exists():
        planner = os.environ.get("DINO_PLANNER", str(get_config_value(config, "planning.planner", "cem")))
        command_path = run_dir / "planning" / f"command_{planner}.sh"
        metadata_path = run_dir / "planning" / f"metadata_{planner}.json"
    write_command_file(argv, command_path)
    _write_json(metadata_path, _metadata(config, args.config))
    split_manifest = build_split_manifest(config)
    write_split_manifest(split_manifest, run_dir / "split_manifest.yaml")

    if args.stage == "finetune":
        _write_json(
            run_dir / "finetune_manifest.json",
            {
                "enabled": True,
                "init_from": get_config_value(config, "finetuning.init_from"),
                "load": get_config_value(config, "finetuning.load", {}),
                "strict": get_config_value(config, "finetuning.strict", True),
                "reset_epoch": get_config_value(config, "finetuning.reset_epoch", True),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending_upstream_load",
            },
        )

    if args.print_run_dir:
        print(run_dir)
    else:
        print(f"Wrote DINO-WM metadata to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
