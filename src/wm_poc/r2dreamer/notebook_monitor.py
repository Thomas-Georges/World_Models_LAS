from __future__ import annotations

import html
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Iterable


VALID_RUNS = {"smoke", "source_base", "target_finetune", "target_scratch"}
DEFAULT_CONFIG = "configs/r2dreamer/three_way_walker_walk_to_run.yaml"
DEFAULT_PATTERN = (
    r"\[wm_poc\] progress|\[wm_poc\] Using serial envs|Saved checkpoint|"
    r"Saved interval checkpoint|Logdir|Create envs|Simulate agent|Encoder|"
    r"Optimizer has|Compiling update function|Evaluating|Traceback|RuntimeError|"
    r"Error executing job|ERROR|Exception|Lost connection|Segmentation fault"
)


def _tail_lines(path: Path, count: int) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-count:]


def _read_new_matches(path: Path, pattern: re.Pattern[str], offset: int) -> tuple[int, list[str]]:
    if not path.is_file():
        return offset, []
    size = path.stat().st_size
    if size < offset:
        offset = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        lines = f.readlines()
        offset = f.tell()
    matches = [line.rstrip("\n") for line in lines if pattern.search(line)]
    return offset, matches


def _active_pid(pid_file: Path) -> int | None:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _kill_process_group(proc: subprocess.Popen[object]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait()


def _display_panel(text: str, handle: object | None = None) -> object | None:
    try:
        from IPython.display import HTML, display
    except ImportError:  # pragma: no cover - exercised outside notebooks.
        print(text, flush=True)
        return None

    body = (
        "<pre style="
        "'white-space: pre-wrap; font-size: 13px; line-height: 1.35; "
        "background: #111827; color: #e5e7eb; padding: 12px; "
        "border-radius: 6px; border: 1px solid #374151;'>"
        f"{html.escape(text)}"
        "</pre>"
    )
    if handle is None:
        return display(HTML(body), display_id=True)
    handle.update(HTML(body))  # type: ignore[attr-defined]
    return handle


def _format_panel(
    *,
    run_name: str,
    config: str,
    run_dir: Path,
    launcher_log: Path,
    console_log: Path,
    pid: int,
    state: str,
    started_at: float,
    latest: str,
    history: Iterable[str],
    launcher_tail: Iterable[str] = (),
    status: int | None = None,
) -> str:
    elapsed = time.monotonic() - started_at
    lines = [
        f"R2-Dreamer {run_name}: {state}",
        f"Config: {config}",
        f"Run dir: {run_dir}",
        f"Launcher log: {launcher_log}",
        f"Console log: {console_log}",
        f"PID: {pid}",
        f"Elapsed: {elapsed / 60:.1f}m",
    ]
    if status is not None:
        lines.append(f"Exit status: {status}")
    lines.extend(["", f"Latest: {latest}", "", "Recent monitored lines:"])
    recent = list(history)
    lines.extend(recent if recent else ["  waiting for matching trainer output..."])
    launcher = list(launcher_tail)
    if launcher:
        lines.extend(["", "Recent launcher output:", *launcher])
    return "\n".join(lines)


def run_r2_with_live_display(
    run_name: str,
    *,
    repo_dir: str | Path | None = None,
    config: str | None = None,
    monitor_interval: int | float | None = None,
    history_lines: int = 16,
    pattern: str | None = None,
) -> int:
    """Run one R2-Dreamer preset and update a notebook display while it runs."""

    if run_name not in VALID_RUNS:
        expected = ", ".join(sorted(VALID_RUNS))
        raise ValueError(f"Unknown R2-Dreamer run {run_name!r}; expected one of: {expected}.")

    repo = Path(repo_dir or os.environ.get("WM_POC_REPO", "/content/World_Models_LAS"))
    if not repo.is_dir():
        raise FileNotFoundError(f"Repository directory does not exist: {repo}")

    config_path = config or os.environ.get("R2_CONFIG", DEFAULT_CONFIG)
    drive_root = os.environ.get("WM_POC_DRIVE_ROOT", "/content/drive/MyDrive/wm_poc")
    log_dir = os.environ.get("WM_POC_LOG_DIR", f"{drive_root}/logs")
    log_root = Path(os.environ.get("R2_LOG_ROOT", f"{log_dir}/r2dreamer"))
    run_dir = log_root / run_name
    pid_file = run_dir / "launcher.pid"
    launcher_log = run_dir / "launcher.log"
    console_log = run_dir / "console.log"
    interval_value = (
        monitor_interval if monitor_interval is not None else os.environ.get("R2_MONITOR_INTERVAL", 15)
    )
    interval = float(interval_value)
    if interval <= 0:
        raise ValueError("monitor_interval must be greater than 0 seconds.")
    compiled_pattern = re.compile(pattern or os.environ.get("R2_TAIL_PATTERN", DEFAULT_PATTERN))

    active_pid = _active_pid(pid_file)
    if active_pid is not None:
        raise RuntimeError(
            f"Run already appears active: {run_name} (PID {active_pid}). "
            f"Console log: {console_log}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    launcher_log.unlink(missing_ok=True)
    console_log.unlink(missing_ok=True)

    command = [
        sys.executable,
        "scripts/r2dreamer/build_commands.py",
        "--config",
        config_path,
        "--run",
        run_name,
        "--execute",
    ]
    env = os.environ.copy()
    env["RUN_TRAINING"] = "1"

    launcher_handle = launcher_log.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        command,
        cwd=repo,
        env=env,
        stdout=launcher_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    pid_file.write_text(str(proc.pid), encoding="utf-8")

    started_at = time.monotonic()
    history: Deque[str] = deque(maxlen=history_lines)
    latest = "waiting for console.log..."
    offset = 0
    display_handle: object | None = None

    try:
        while proc.poll() is None:
            offset, matches = _read_new_matches(console_log, compiled_pattern, offset)
            for match in matches:
                history.append(match)
                latest = match
            if not console_log.exists() and not matches:
                latest = "waiting for console.log..."

            launcher_tail = [] if console_log.exists() else _tail_lines(launcher_log, 8)
            panel = _format_panel(
                run_name=run_name,
                config=config_path,
                run_dir=run_dir,
                launcher_log=launcher_log,
                console_log=console_log,
                pid=proc.pid,
                state="running",
                started_at=started_at,
                latest=latest,
                history=history,
                launcher_tail=launcher_tail,
            )
            display_handle = _display_panel(panel, display_handle)
            time.sleep(interval)

        status = proc.wait()
        offset, matches = _read_new_matches(console_log, compiled_pattern, offset)
        for match in matches:
            history.append(match)
            latest = match
        launcher_tail = _tail_lines(launcher_log, 12) if status else []
        panel = _format_panel(
            run_name=run_name,
            config=config_path,
            run_dir=run_dir,
            launcher_log=launcher_log,
            console_log=console_log,
            pid=proc.pid,
            state="finished" if status == 0 else "failed",
            started_at=started_at,
            latest=latest,
            history=history,
            launcher_tail=launcher_tail,
            status=status,
        )
        _display_panel(panel, display_handle)
        if status:
            raise subprocess.CalledProcessError(status, command)
        return status
    except KeyboardInterrupt:
        latest = "interrupted; stopping training subprocess..."
        panel = _format_panel(
            run_name=run_name,
            config=config_path,
            run_dir=run_dir,
            launcher_log=launcher_log,
            console_log=console_log,
            pid=proc.pid,
            state="stopping",
            started_at=started_at,
            latest=latest,
            history=history,
        )
        _display_panel(panel, display_handle)
        _kill_process_group(proc)
        raise
    finally:
        launcher_handle.close()
        if pid_file.exists():
            try:
                if int(pid_file.read_text(encoding="utf-8").strip()) == proc.pid:
                    pid_file.unlink()
            except ValueError:
                pid_file.unlink()
