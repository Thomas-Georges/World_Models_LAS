from __future__ import annotations

import re

import pytest

from wm_poc.r2dreamer.notebook_monitor import _read_new_matches, _tail_lines, run_r2_with_live_display


def test_read_new_matches_tracks_offset_and_truncation(tmp_path):
    log = tmp_path / "console.log"
    pattern = re.compile(r"\[wm_poc\] progress|RuntimeError")

    log.write_text("Create envs.\n[wm_poc] progress 0001\n", encoding="utf-8")
    offset, matches = _read_new_matches(log, pattern, 0)
    assert matches == ["[wm_poc] progress 0001"]

    with log.open("a", encoding="utf-8") as f:
        f.write("ignored\nRuntimeError: failed\n")
    offset, matches = _read_new_matches(log, pattern, offset)
    assert matches == ["RuntimeError: failed"]

    log.write_text("[wm_poc] progress 0000\n", encoding="utf-8")
    offset, matches = _read_new_matches(log, pattern, offset)
    assert matches == ["[wm_poc] progress 0000"]
    assert offset == log.stat().st_size


def test_tail_lines_returns_requested_suffix(tmp_path):
    log = tmp_path / "launcher.log"
    log.write_text("\n".join(f"line {i}" for i in range(5)), encoding="utf-8")

    assert _tail_lines(log, 3) == ["line 2", "line 3", "line 4"]
    assert _tail_lines(tmp_path / "missing.log", 3) == []


def test_run_monitor_rejects_invalid_interval(tmp_path):
    with pytest.raises(ValueError, match="monitor_interval"):
        run_r2_with_live_display("smoke", repo_dir=tmp_path, monitor_interval=0)
