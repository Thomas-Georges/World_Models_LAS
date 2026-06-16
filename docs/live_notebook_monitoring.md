# Live notebook monitoring pattern

This note describes the pattern used to monitor long R2-Dreamer runs from a single Colab cell without waiting for Colab to flush all `%%bash` output at the end.

The same pattern is useful for any long-running training, evaluation, data-processing, or simulation job that emits progress to stdout or a log file.

## Problem

Colab can buffer output from long foreground `%%bash` cells. Even if the shell command prints progress with `flush=True` or line-buffered tools, the notebook UI may still show a large dump only after the process exits.

That makes progress bars, checkpoint messages, and failure traces much less useful for monitoring.

## Architecture

Split the workflow into two roles:

1. The long-running job writes structured progress lines to a log file.
2. The notebook kernel runs a lightweight Python monitor that polls that log file and updates an IPython display panel.

For this repository, the relevant files are:

- `src/wm_poc/r2dreamer/patching.py`: patches the upstream trainer to emit progress heartbeat lines.
- `src/wm_poc/r2dreamer/notebook_monitor.py`: starts the run, polls logs, and updates the notebook display.
- `scripts/r2dreamer/run_with_live_monitor.sh`: shell fallback for terminal use.
- `src/wm_poc/dino_wm/notebook_monitor.py`: starts DINO-WM wrapper scripts, polls launcher/stdout/stderr logs on Drive, and updates the notebook display.

## Progress heartbeat

The trainer is patched to print progress lines such as:

```text
[wm_poc] progress [#####-------------------] 000100000/000800000 ( 12.5%) elapsed=45.2m rate=36.9 steps/s
```

Important details:

- The line has a stable prefix: `[wm_poc] progress`.
- The print uses `flush=True`.
- The cadence is controlled by `R2_PROGRESS_EVERY`.
- The monitor can filter for that prefix rather than parsing every log line.

For other projects, make the long-running code emit a stable progress line at a predictable cadence.

## Python monitor

The notebook monitor does not rely on `%%bash` output. It starts the training process with `subprocess.Popen`, writes launcher output to `launcher.log`, and then watches the trainer's `console.log`.

Key implementation points:

- Clear stale logs before starting, so old progress does not appear in a new run.
- Write a PID file, so a second monitor can detect an already-running job.
- Track the current file offset and read only new log content.
- Keep a short history of matching lines for the display panel.
- Stop the subprocess group on notebook interruption.

The display is updated with an IPython display handle:

```python
from IPython.display import HTML, display

handle = display(HTML("<pre>starting...</pre>"), display_id=True)
handle.update(HTML("<pre>latest progress...</pre>"))
```

That is the main difference from a shell monitor. Colab may buffer shell stdout, but Python display updates from the notebook kernel are shown much closer to real time.

## Minimal reusable example

This is a small version of the pattern for another project:

```python
from pathlib import Path
from IPython.display import HTML, display
import html
import subprocess
import time


def show(text: str, handle=None):
    body = (
        "<pre style='white-space: pre-wrap; font-size: 13px; "
        "background: #111827; color: #e5e7eb; padding: 12px; "
        "border-radius: 6px;'>"
        f"{html.escape(text)}"
        "</pre>"
    )
    if handle is None:
        return display(HTML(body), display_id=True)
    handle.update(HTML(body))
    return handle


log_path = Path("run.log")
log_path.unlink(missing_ok=True)

with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
    proc = subprocess.Popen(
        ["python", "your_long_job.py"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    handle = show("starting...")
    offset = 0
    recent: list[str] = []

    try:
        while proc.poll() is None:
            if log_path.exists():
                with log_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    lines = f.readlines()
                    offset = f.tell()

                for line in lines:
                    line = line.rstrip("\n")
                    if "progress" in line or "ERROR" in line:
                        recent.append(line)
                        recent = recent[-16:]

            panel = "\n".join(recent) if recent else "waiting for progress..."
            handle = show(panel, handle)
            time.sleep(5)

        status = proc.wait()
        handle = show(f"finished with status {status}\n\n" + "\n".join(recent[-16:]), handle)
        if status:
            raise subprocess.CalledProcessError(status, proc.args)
    except KeyboardInterrupt:
        proc.terminate()
        raise
```

## Tuning knobs

There are two separate update frequencies:

- Job logging cadence: how often the long-running process emits progress.
- Monitor refresh cadence: how often the notebook display polls and refreshes.

In the R2-Dreamer notebook:

```python
import os
from wm_poc.r2dreamer.notebook_monitor import run_r2_with_live_display

os.environ["R2_PROGRESS_EVERY"] = "500"
run_r2_with_live_display("source_base", monitor_interval=30)
```

In the DINO-WM notebook:

```python
import os
from wm_poc.dino_wm.notebook_monitor import run_dino_with_live_display

os.environ["RUN_DINO_WM"] = "1"
run_dino_with_live_display("smoke", config="configs/dino_wm/smoke_pointmaze.yaml", monitor_interval=15)
```

DINO-WM does not currently patch upstream training to emit a custom heartbeat. Instead, the monitor watches the wrapper launcher log plus the run directory's `stdout.log` and `stderr.log`, filtering for epoch, checkpoint, planning, status, and error lines.

Use a smaller `R2_PROGRESS_EVERY` or monitor interval for chatty debugging. Use a larger value for long stable runs.

## When to use this

Use this pattern when:

- A notebook cell must both start and monitor a long job.
- The notebook environment buffers shell output.
- You cannot run a second concurrent notebook cell.
- You want a compact status panel instead of thousands of output lines.
- You need the monitor to stop the job cleanly when the notebook cell is interrupted.

For normal terminals, `tail -f` or a shell monitor is still fine. The Python display monitor is mainly a notebook reliability workaround.
