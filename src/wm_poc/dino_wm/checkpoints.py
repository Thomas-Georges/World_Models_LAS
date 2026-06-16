from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def _checkpoint_files(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    return sorted(
        [
            path
            for path in run_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
        ],
        key=lambda path: path.stat().st_mtime,
    )


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    root = Path(run_dir).expanduser()
    for preferred in (
        "latest.pt",
        "latest.pth",
        "model_latest.pth",
        "model_latest.pt",
        "checkpoint_latest.pt",
    ):
        matches = list(root.rglob(preferred))
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)
    files = _checkpoint_files(root)
    return files[-1] if files else None


def _metric_from_name(path: Path, metric: str) -> float | None:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", metric).strip("_")
    pattern = re.compile(rf"{re.escape(normalized)}[_=-]([-+]?\d+(?:\.\d+)?)")
    match = pattern.search(path.stem)
    if match:
        return float(match.group(1))
    return None


def find_best_checkpoint(run_dir: str | Path, metric: str) -> Path | None:
    root = Path(run_dir).expanduser()
    metrics_path = root / "metrics.jsonl"
    best: tuple[float, Path] | None = None
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if metric not in record or "checkpoint_path" not in record:
                    continue
                checkpoint = Path(str(record["checkpoint_path"]))
                if not checkpoint.is_absolute():
                    checkpoint = root / checkpoint
                if checkpoint.is_file():
                    value = float(record[metric])
                    if best is None or value < best[0]:
                        best = (value, checkpoint)
    if best is not None:
        return best[1]

    candidates = []
    for path in _checkpoint_files(root):
        value = _metric_from_name(path, metric)
        if value is not None:
            candidates.append((value, path))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return find_latest_checkpoint(root)


def copy_checkpoint_to_artifact_root(src: Path, dst_root: Path, run_name: str) -> Path:
    src = src.expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {src}")
    destination_dir = dst_root.expanduser() / run_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / src.name
    shutil.copy2(src, destination)
    return destination


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


def _infer_epoch_step(path: Path) -> dict[str, int | None]:
    text = path.stem
    epoch_match = re.search(r"epoch[_=-]?(\d+)", text)
    step_match = re.search(r"step[_=-]?(\d+)", text)
    return {
        "epoch": int(epoch_match.group(1)) if epoch_match else None,
        "step": int(step_match.group(1)) if step_match else None,
    }


def write_checkpoint_manifest(
    checkpoint_path: Path,
    manifest_path: Path,
    extra: dict[str, Any],
) -> None:
    checkpoint = checkpoint_path.expanduser()
    manifest = {
        "source_run_name": extra.get("source_run_name", checkpoint.parent.name),
        "source_checkpoint_path": str(extra.get("source_checkpoint_path", checkpoint)),
        "copied_checkpoint_path": str(checkpoint),
        "metric_used_for_selection": extra.get("metric_used_for_selection"),
        "used_for_finetuning": bool(extra.get("used_for_finetuning", False)),
        "source_config_path": extra.get("source_config_path"),
        "upstream_dino_wm_commit": extra.get("upstream_dino_wm_commit", ""),
        "project_git_commit": extra.get("project_git_commit", _git_commit(Path.cwd())),
    }
    manifest.update(_infer_epoch_step(checkpoint))
    manifest.update({key: value for key, value in extra.items() if key not in manifest})

    manifest_path = manifest_path.expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
