from pathlib import Path

import pytest

from wm_poc import paths


def test_env_path_respects_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = tmp_path / "repo"
    monkeypatch.setenv("WM_POC_REPO", str(expected))

    assert paths.repo_root() == expected.resolve()


def test_env_path_raises_without_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WM_POC_REQUIRED", raising=False)

    with pytest.raises(RuntimeError, match="WM_POC_REQUIRED"):
        paths.env_path("WM_POC_REQUIRED")


def test_default_paths_return_path_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "WM_POC_REPO",
        "WM_POC_DRIVE_ROOT",
        "WM_POC_DATA_DIR",
        "WM_POC_LOG_DIR",
        "WM_POC_CKPT_DIR",
        "WM_POC_FIG_DIR",
        "WM_POC_FIGURE_DIR",
    ]:
        monkeypatch.delenv(name, raising=False)

    assert isinstance(paths.repo_root(), Path)
    assert isinstance(paths.drive_root(), Path)
    assert isinstance(paths.data_dir(), Path)
    assert isinstance(paths.log_dir(), Path)
    assert isinstance(paths.checkpoint_dir(), Path)
    assert isinstance(paths.figure_dir(), Path)
