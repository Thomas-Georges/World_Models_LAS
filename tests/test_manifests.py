import json
from pathlib import Path

from wm_poc.manifests import RUN_MANIFEST_TEMPLATE, create_run_manifest, write_json_manifest


def test_create_run_manifest_sets_required_fields() -> None:
    manifest = create_run_manifest(track="local_global", run_name="debug-run")

    assert set(manifest) == set(RUN_MANIFEST_TEMPLATE)
    assert manifest["track"] == "local_global"
    assert manifest["run_name"] == "debug-run"
    assert manifest["date"]


def test_write_json_manifest(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "manifest.json"
    manifest = create_run_manifest(track="r2dreamer", run_name="smoke")

    write_json_manifest(manifest, output)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == manifest
