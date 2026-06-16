#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.data import SUPPORTED_ENVS, validate_dataset_root  # noqa: E402


OSF_DATASETS_API = "https://api.osf.io/v2/nodes/bmw48/files/osfstorage/678ab2bbd34740f8d0283952/"
OSF_PROJECT_URL = "https://osf.io/bmw48/"
DATASET_ARCHIVES = {
    "point_maze": "point_maze.zip",
    "pusht_noise": "pusht_noise.zip",
    "wall_single": "wall_single.zip",
}
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "DINO_WM_DATA_ROOT",
        str(Path(os.environ.get("WM_POC_DATA_DIR", "/content/drive/MyDrive/wm_poc/data")) / "dino_wm"),
    )
)
PROGRESS_INTERVAL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ArchiveRecord:
    name: str
    materialized_path: str
    size_bytes: int | None
    download_url: str


def human_size(size: int | None) -> str:
    if size is None:
        return "unknown size"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def read_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "World_Models_LAS-dino-wm-data/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_osf_files(url: str) -> list[ArchiveRecord]:
    records: list[ArchiveRecord] = []
    next_url: str | None = url
    while next_url:
        payload = read_json(next_url)
        for item in payload.get("data", []):
            attributes = item.get("attributes", {})
            if attributes.get("kind") != "file":
                continue
            download_url = item.get("links", {}).get("download")
            if not download_url:
                continue
            records.append(
                ArchiveRecord(
                    name=str(attributes.get("name")),
                    materialized_path=str(attributes.get("materialized_path") or attributes.get("name")),
                    size_bytes=attributes.get("size"),
                    download_url=str(download_url),
                )
            )
        next_url = payload.get("links", {}).get("next")
    return records


def select_datasets(values: list[str]) -> list[str]:
    requested = values or ["point_maze"]
    if "all" in requested:
        return list(DATASET_ARCHIVES)
    selected: list[str] = []
    for dataset in requested:
        if dataset not in DATASET_ARCHIVES:
            choices = ", ".join(sorted(DATASET_ARCHIVES) + ["all"])
            raise ValueError(f"Unsupported dataset {dataset!r}. Choose one of: {choices}")
        if dataset not in selected:
            selected.append(dataset)
    return selected


def archive_index() -> dict[str, ArchiveRecord]:
    archives = iter_osf_files(OSF_DATASETS_API)
    return {archive.name: archive for archive in archives}


def is_dataset_present(data_root: Path, dataset: str) -> bool:
    try:
        validate_dataset_root(data_root, dataset)
    except FileNotFoundError:
        return False
    return True


def download_file(url: str, output: Path, expected_size: int | None, force: bool) -> None:
    if output.is_file() and not force:
        actual_size = output.stat().st_size
        if expected_size is None or actual_size == expected_size:
            print(f"Using existing archive: {output} ({human_size(actual_size)})")
            return
        print(f"Existing archive size mismatch; redownloading {output.name}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(output.suffix + ".part")
    if part.exists():
        part.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "World_Models_LAS-dino-wm-data/1.0"})
    started = time.monotonic()
    copied = 0
    next_progress = PROGRESS_INTERVAL_BYTES
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            copied += len(chunk)
            if copied >= next_progress:
                elapsed = max(time.monotonic() - started, 0.001)
                rate = copied / elapsed
                print(f"  downloaded {human_size(copied)} at {human_size(int(rate))}/s")
                next_progress += PROGRESS_INTERVAL_BYTES

    if expected_size is not None and copied != expected_size:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded size mismatch for {output.name}: expected {expected_size} bytes, got {copied} bytes."
        )
    part.replace(output)
    print(f"Downloaded archive: {output} ({human_size(copied)})")


def safe_extract_zip(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe path in archive {archive.name}: {member.filename}")
            destination = (target / member.filename).resolve()
            if target != destination and target not in destination.parents:
                raise RuntimeError(f"Archive member escapes target directory: {member.filename}")
        zf.extractall(target)


def manifest_path(data_root: Path) -> Path:
    return data_root / "_manifests" / "dino_wm_dataset_downloads.json"


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"source_project": OSF_PROJECT_URL, "datasets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(
    data_root: Path,
    dataset: str,
    archive: ArchiveRecord,
    archive_path: Path,
    archive_kept: bool,
) -> None:
    path = manifest_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(path)
    manifest.setdefault("datasets", {})[dataset] = {
        "source_project": OSF_PROJECT_URL,
        "source_api": OSF_DATASETS_API,
        "archive_name": archive.name,
        "materialized_path": archive.materialized_path,
        "download_url": archive.download_url,
        "size_bytes": archive.size_bytes,
        "archive_path": str(archive_path),
        "archive_kept": archive_kept,
        "data_root": str(data_root),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {path}")


def print_archive_plan(selected: list[str], archives: dict[str, ArchiveRecord], data_root: Path) -> None:
    print(f"Official DINO-WM source: {OSF_PROJECT_URL}")
    print(f"Data root: {data_root}")
    for dataset in selected:
        archive_name = DATASET_ARCHIVES[dataset]
        archive = archives.get(archive_name)
        if not archive:
            print(f"- {dataset}: missing expected OSF archive {archive_name}")
            continue
        status = "present" if is_dataset_present(data_root, dataset) else "missing"
        print(f"- {dataset}: {archive.name}, {human_size(archive.size_bytes)}, {status}")


def download_dataset(
    *,
    dataset: str,
    archive: ArchiveRecord,
    data_root: Path,
    download_dir: Path,
    force: bool,
    keep_archives: bool,
) -> None:
    if dataset not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported DINO-WM environment {dataset!r}")

    if is_dataset_present(data_root, dataset) and not force:
        print(f"Dataset already present for {dataset}: {data_root}")
        return

    archive_path = download_dir / archive.name
    download_file(archive.download_url, archive_path, archive.size_bytes, force=force)
    print(f"Extracting {archive_path.name} into {data_root}")
    safe_extract_zip(archive_path, data_root)
    validate_dataset_root(data_root, dataset)
    if not keep_archives:
        archive_path.unlink(missing_ok=True)
        print(f"Removed archive after extraction: {archive_path}")
    write_manifest(data_root, dataset, archive, archive_path, archive_kept=keep_archives)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official DINO-WM datasets from OSF.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset to download. May be repeated. Default: point_maze. Use 'all' for all supported archives.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually download and extract archives.")
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract even if data exists.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep zip archives after extraction.")
    parser.add_argument("--list", action="store_true", help="List official archives without downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = select_datasets(args.dataset)
    data_root = args.data_root.expanduser()
    download_dir = (args.download_dir or (data_root / "_downloads")).expanduser()
    archives = archive_index()

    print_archive_plan(selected, archives, data_root)
    if args.list:
        return 0
    if not args.execute:
        joined = " ".join(f"--dataset {dataset}" for dataset in selected)
        print(
            "Dry run only. Add --execute to download: "
            f"python scripts/dino_wm/download_data.py {joined} --data-root {data_root} --execute"
        )
        return 0

    for dataset in selected:
        archive_name = DATASET_ARCHIVES[dataset]
        archive = archives.get(archive_name)
        if archive is None:
            raise RuntimeError(f"Expected OSF archive not found: {archive_name}")
        download_dataset(
            dataset=dataset,
            archive=archive,
            data_root=data_root,
            download_dir=download_dir,
            force=args.force,
            keep_archives=args.keep_archives,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
