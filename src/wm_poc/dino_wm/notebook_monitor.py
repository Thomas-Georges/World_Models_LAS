from __future__ import annotations

import html
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, Sequence

from wm_poc.dino_wm.configs import get_config_value, load_config, resolve_config


DEFAULT_CONFIG = "configs/dino_wm/smoke_pointmaze.yaml"
DEFAULT_PATTERN = (
    r"DINO-WM Python package imports OK|Dataset root:|Environment:|Available trajectory|"
    r"Building split manifest|Selected train files|Selected val files|split_manifest|Skipping latent precompute|"
    r"Latent caching is disabled|Precomputing DINO latents|Encoded latents for episode|Latent precompute complete|"
    r"Latent cache already covers|Installed DINO-WM latent support|latent bypass patch|"
    r"Latent cache dir:|WARNING: latent cache|WARNING: reading DINO latents|val no-grad patch|"
    r"Loaded [0-9]+ rollouts|dataloader batch size|Train encoder|Model emb_dim|"
    r"Epoch [0-9]+ (Train|Valid):|"
    r"Epoch [0-9]+.*Training loss|Validation loss|Saved model|Saved rolling latest|"
    r"Planning result saved dir|success_rate|final_goal_latent_distance|"
    r"summary_csv:|DINO-WM .* command failed|Traceback|RuntimeError|Error executing job|"
    r"ModuleNotFoundError|DependencyNotInstalled|CompileError|ImportError|"
    r"UnpicklingError|Killed|Segmentation fault|timeout|Timed out|ERROR"
)
VALID_STAGES = {"smoke", "experiment", "train", "finetune", "plan"}


def _tail_lines(path: Path, count: int) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]


def _read_new_matches(path: Path, pattern: re.Pattern[str], offset: int) -> tuple[int, list[str]]:
    if not path.is_file():
        return offset, []
    size = path.stat().st_size
    if size < offset:
        offset = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        text = f.read()
        offset = f.tell()
    lines = text.splitlines()
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


def _force_kill_process_group(proc: subprocess.Popen[object]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _kill_process_group(proc: subprocess.Popen[object]) -> None:
    if proc.poll() is not None:
        return
    # A second KeyboardInterrupt while waiting for SIGTERM must escalate to
    # SIGKILL instead of orphaning children (wandb traps SIGTERM and a leaked
    # trainer keeps holding GPU memory against the next launch).
    try:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _force_kill_process_group(proc)
    except KeyboardInterrupt:
        _force_kill_process_group(proc)
        raise


# Command-line substrings of DINO-WM workers that can outlive an interrupted
# notebook cell and keep holding GPU memory.
STALE_PROCESS_PATTERNS = (
    "dino_wm/train.py",
    "dino_wm/plan.py",
    "wm_poc_precompute_latents.py",
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stale_dino_pids() -> list[int]:
    pids: set[int] = set()
    for pattern in STALE_PROCESS_PATTERNS:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return []
        for token in result.stdout.split():
            try:
                pid = int(token)
            except ValueError:
                continue
            if pid != os.getpid():
                pids.add(pid)
    return sorted(pids)


def _terminate_stale_dino_processes() -> list[int]:
    pids = _stale_dino_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    if pids:
        deadline = time.time() + 10
        while time.time() < deadline and any(_pid_alive(pid) for pid in pids):
            time.sleep(0.5)
    return pids


def _handle_stale_processes() -> list[str]:
    """Detect leftover DINO-WM workers before launching a new run.

    Returns monitor lines describing what happened. Kills the leftovers when
    DINO_KILL_STALE is unset or "1"; raises otherwise so the user can decide.
    """

    pids = _stale_dino_pids()
    if not pids:
        return []
    if os.environ.get("DINO_KILL_STALE", "1") == "1":
        killed = _terminate_stale_dino_processes()
        return [
            "Killed stale DINO-WM process(es) from a previous interrupted run: "
            + ", ".join(str(pid) for pid in killed)
        ]
    raise RuntimeError(
        f"Stale DINO-WM process(es) still running: {pids}. They hold GPU memory and will "
        "slow down or OOM this run. Leave DINO_KILL_STALE unset (or =1) to clean them up "
        "automatically, kill them manually, or restart the Colab runtime."
    )


def _rotate_stale_logs(paths: Iterable[Path]) -> list[str]:
    """Move non-empty logs from a previous run aside so the live panel and log
    freshness only reflect the run being launched.

    Rotations are timestamped instead of overwriting a single ``.prev`` file:
    epoch loss lines from every attempt stay recoverable for training-curve
    plots (metrics.epoch_loss_series merges all generations)."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rotated: list[str] = []
    for path in paths:
        try:
            if not path.is_file() or path.stat().st_size == 0:
                continue
            previous = path.with_name(f"{path.name}.{stamp}.prev")
            index = 1
            while previous.exists():
                previous = path.with_name(f"{path.name}.{stamp}_{index}.prev")
                index += 1
            path.replace(previous)
            rotated.append(path.name)
        except OSError:
            continue
    return rotated


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


def _config_run_dir(repo: Path, config_path: str | Path) -> tuple[dict[str, object], Path]:
    config = resolve_config(load_config(repo / config_path if not Path(config_path).is_absolute() else config_path))
    run_name = str(config.get("run_name") or "dino_wm_run")
    log_root = Path(str(get_config_value(config, "artifacts.log_root"))).expanduser()
    return config, log_root / run_name


def _command_for_stage(
    stage: str,
    config: str,
    *,
    checkpoint: str | Path | None = None,
    skip_cache: bool = False,
    skip_plan: bool = False,
) -> list[str]:
    if stage == "smoke":
        return ["bash", "scripts/dino_wm/run_smoke.sh"]
    if stage == "experiment":
        command = ["bash", "scripts/dino_wm/run_experiment.sh", "--config", config]
        if skip_cache:
            command.append("--skip-cache")
        if skip_plan:
            command.append("--skip-plan")
        if checkpoint:
            command.extend(["--checkpoint", str(checkpoint)])
        return command
    if stage == "train":
        return ["bash", "scripts/dino_wm/run_train.sh", "--config", config]
    if stage == "finetune":
        return ["bash", "scripts/dino_wm/run_finetune.sh", "--config", config]
    if stage == "plan":
        command = ["bash", "scripts/dino_wm/run_plan.sh", "--config", config]
        if checkpoint:
            command.extend(["--checkpoint", str(checkpoint)])
        return command
    expected = ", ".join(sorted(VALID_STAGES))
    raise ValueError(f"Unknown DINO-WM stage {stage!r}; expected one of: {expected}.")


def _format_age(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def _log_freshness(paths: Iterable[Path], *, now: float | None = None) -> list[str]:
    current = time.time() if now is None else now
    lines = []
    for path in paths:
        if not path.is_file():
            lines.append(f"{path.name}: missing")
            continue
        stat = path.stat()
        lines.append(f"{path.name}: {stat.st_size} bytes, updated {_format_age(current - stat.st_mtime)} ago")
    return lines


def _read_status(run_dir: Path, *, min_mtime: float | None = None) -> list[str]:
    status_paths = [run_dir / "status.json", *sorted((run_dir / "planning").glob("status_*.json"))]
    lines = []
    for path in status_paths:
        if not path.is_file():
            continue
        stat = path.stat()
        if min_mtime is not None and stat.st_mtime < min_mtime:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        label = path.relative_to(run_dir)
        state = "completed" if payload.get("completed") else "failed" if payload.get("failed") else "running"
        elapsed = payload.get("elapsed_seconds")
        suffix = f", elapsed={float(elapsed) / 60:.1f}m" if elapsed is not None else ""
        return_code = payload.get("return_code")
        code = f", return_code={return_code}" if return_code is not None else ""
        lines.append(f"{label}: {state}{suffix}{code}, updated {_format_age(time.time() - stat.st_mtime)} ago")
    return lines


def _gpu_status() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "GPU: unavailable"
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        return "GPU: unavailable"
    first = text.splitlines()[0]
    name, used, total, util = [part.strip() for part in first.split(",", maxsplit=3)]
    return f"GPU: {name}, {used}/{total} MiB, util={util}%"


def _format_panel(
    *,
    stage: str,
    config: str,
    command: Sequence[str],
    run_dir: Path,
    launcher_log: Path,
    stdout_log: Path,
    stderr_log: Path,
    pid: int,
    state: str,
    started_at: float,
    latest: str,
    history: Iterable[str],
    status_lines: Iterable[str] = (),
    log_status_lines: Iterable[str] = (),
    launcher_tail: Iterable[str] = (),
    return_code: int | None = None,
) -> str:
    elapsed = time.monotonic() - started_at
    lines = [
        f"DINO-WM {stage}: {state}",
        f"Config: {config}",
        f"Command: {' '.join(command)}",
        f"Run dir: {run_dir}",
        f"Launcher log: {launcher_log}",
        f"stdout log: {stdout_log}",
        f"stderr log: {stderr_log}",
        f"PID: {pid}",
        f"Elapsed: {elapsed / 60:.1f}m",
        _gpu_status(),
    ]
    if return_code is not None:
        lines.append(f"Exit status: {return_code}")
    status = list(status_lines)
    if status:
        lines.extend(["", "Status files:", *status])
    log_status = list(log_status_lines)
    if log_status:
        lines.extend(["", "Log freshness:", *log_status])
    lines.extend(["", f"Latest: {latest}", "", "Recent monitored lines:"])
    recent = list(history)
    lines.extend(recent if recent else ["  waiting for matching DINO-WM output..."])
    launcher = list(launcher_tail)
    if launcher:
        lines.extend(["", "Recent launcher output:", *launcher])
    return "\n".join(lines)


def run_dino_with_live_display(
    stage: str = "smoke",
    *,
    repo_dir: str | Path | None = None,
    config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    monitor_interval: int | float | None = None,
    history_lines: int = 24,
    pattern: str | None = None,
    skip_cache: bool = False,
    skip_plan: bool = False,
) -> int:
    """Run a DINO-WM wrapper command and update a notebook display while it runs."""

    interval_value = (
        monitor_interval if monitor_interval is not None else os.environ.get("DINO_MONITOR_INTERVAL", 15)
    )
    interval = float(interval_value)
    if interval <= 0:
        raise ValueError("monitor_interval must be greater than 0 seconds.")

    repo = Path(repo_dir or os.environ.get("WM_POC_REPO", "/content/World_Models_LAS"))
    if not repo.is_dir():
        raise FileNotFoundError(f"Repository directory does not exist: {repo}")

    config_path = str(config or os.environ.get("DINO_CONFIG", DEFAULT_CONFIG))
    _, run_dir = _config_run_dir(repo, config_path)
    pid_file = run_dir / "live_monitor.pid"
    launcher_log = run_dir / "launcher.log"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"

    active_pid = _active_pid(pid_file)
    if active_pid is not None:
        raise RuntimeError(
            f"DINO-WM run already appears active for {run_dir.name} (PID {active_pid}). "
            f"stdout log: {stdout_log}"
        )

    command = _command_for_stage(
        stage,
        config_path,
        checkpoint=checkpoint,
        skip_cache=skip_cache,
        skip_plan=skip_plan,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    launcher_log.unlink(missing_ok=True)
    stale_lines = _handle_stale_processes()
    rotated = _rotate_stale_logs((stdout_log, stderr_log))
    if rotated:
        stale_lines.append(
            "Rotated stale logs from a previous run to *.prev: " + ", ".join(rotated)
        )

    compiled_pattern = re.compile(pattern or os.environ.get("DINO_TAIL_PATTERN", DEFAULT_PATTERN))
    offsets = {
        launcher_log: 0,
        stdout_log: stdout_log.stat().st_size if stdout_log.exists() else 0,
        stderr_log: stderr_log.stat().st_size if stderr_log.exists() else 0,
    }
    history: Deque[str] = deque(maxlen=history_lines)
    for line in stale_lines:
        history.append(line)
    latest = stale_lines[-1] if stale_lines else "starting DINO-WM wrapper..."

    env = os.environ.copy()
    env["RUN_DINO_WM"] = "1"
    env["DINO_CONFIG"] = config_path

    launcher_handle = launcher_log.open("w", encoding="utf-8", buffering=1)
    started_wall_time = time.time()
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
    display_handle: object | None = None

    try:
        while proc.poll() is None:
            for log_path in (launcher_log, stdout_log, stderr_log):
                offsets[log_path], matches = _read_new_matches(log_path, compiled_pattern, offsets[log_path])
                for match in matches:
                    tagged = f"{log_path.name}: {match}"
                    history.append(tagged)
                    latest = tagged

            panel = _format_panel(
                stage=stage,
                config=config_path,
                command=command,
                run_dir=run_dir,
                launcher_log=launcher_log,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                pid=proc.pid,
                state="running",
                started_at=started_at,
                latest=latest,
                history=history,
                status_lines=_read_status(run_dir, min_mtime=started_wall_time),
                log_status_lines=_log_freshness((launcher_log, stdout_log, stderr_log)),
                launcher_tail=[] if stdout_log.exists() or stderr_log.exists() else _tail_lines(launcher_log, 10),
            )
            display_handle = _display_panel(panel, display_handle)
            time.sleep(interval)

        return_code = proc.wait()
        for log_path in (launcher_log, stdout_log, stderr_log):
            offsets[log_path], matches = _read_new_matches(log_path, compiled_pattern, offsets[log_path])
            for match in matches:
                tagged = f"{log_path.name}: {match}"
                history.append(tagged)
                latest = tagged
        panel = _format_panel(
            stage=stage,
            config=config_path,
            command=command,
            run_dir=run_dir,
            launcher_log=launcher_log,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            pid=proc.pid,
            state="finished" if return_code == 0 else "failed",
            started_at=started_at,
            latest=latest,
            history=history,
            status_lines=_read_status(run_dir, min_mtime=started_wall_time),
            log_status_lines=_log_freshness((launcher_log, stdout_log, stderr_log)),
            launcher_tail=_tail_lines(launcher_log, 16) if return_code else [],
            return_code=return_code,
        )
        _display_panel(panel, display_handle)
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        return return_code
    except KeyboardInterrupt:
        latest = "interrupted; stopping DINO-WM subprocess group..."
        panel = _format_panel(
            stage=stage,
            config=config_path,
            command=command,
            run_dir=run_dir,
            launcher_log=launcher_log,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            pid=proc.pid,
            state="stopping",
            started_at=started_at,
            latest=latest,
            history=history,
            status_lines=_read_status(run_dir, min_mtime=started_wall_time),
            log_status_lines=_log_freshness((launcher_log, stdout_log, stderr_log)),
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
