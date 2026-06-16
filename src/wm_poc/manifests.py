from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from wm_poc.system_info import current_git_commit


RUN_MANIFEST_TEMPLATE: dict[str, str] = {
    "run_name": "",
    "track": "",
    "date": "",
    "git_commit": "",
    "external_repo": "",
    "external_repo_commit": "",
    "gpu": "",
    "environment": "",
    "task_source": "",
    "task_target": "",
    "model_config": "",
    "checkpoint_input": "",
    "checkpoint_output": "",
    "logdir": "",
    "notes": "",
}


def create_run_manifest(track: str = "", run_name: str = "", cwd: Path | None = None) -> dict[str, str]:
    manifest = dict(RUN_MANIFEST_TEMPLATE)
    manifest["run_name"] = run_name
    manifest["track"] = track
    manifest["date"] = date.today().isoformat()
    manifest["git_commit"] = current_git_commit(cwd=cwd)
    return manifest


def write_json_manifest(manifest: dict[str, str], output: Path) -> None:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
