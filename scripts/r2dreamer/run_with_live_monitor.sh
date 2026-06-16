#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-${R2_RUN:-source_base}}"
CONFIG="${R2_CONFIG:-configs/r2dreamer/three_way_walker_walk_to_run.yaml}"
DRIVE_ROOT="${WM_POC_DRIVE_ROOT:-/content/drive/MyDrive/wm_poc}"
LOG_DIR="${WM_POC_LOG_DIR:-${DRIVE_ROOT}/logs}"
LOG_ROOT="${R2_LOG_ROOT:-${LOG_DIR}/r2dreamer}"
RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
PID_FILE="${RUN_DIR}/launcher.pid"
LAUNCHER_LOG="${RUN_DIR}/launcher.log"
CONSOLE_LOG="${RUN_DIR}/console.log"
MONITOR_INTERVAL="${R2_MONITOR_INTERVAL:-15}"
PATTERN="${R2_TAIL_PATTERN:-\\[wm_poc\\] progress|\\[wm_poc\\] Using serial envs|Saved checkpoint|Saved interval checkpoint|Logdir|Create envs|Simulate agent|Encoder|Optimizer has|Compiling update function|Evaluating|Traceback|RuntimeError|Error executing job|ERROR|Exception|Lost connection|Segmentation fault}"

case "${RUN_NAME}" in
  smoke|source_base|target_finetune|target_scratch) ;;
  *)
    echo "Unknown R2-Dreamer run: ${RUN_NAME}" >&2
    echo "Expected one of: smoke, source_base, target_finetune, target_scratch" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Run already appears active: ${RUN_NAME}" >&2
    echo "PID: ${old_pid}" >&2
    echo "Use: tail -f ${CONSOLE_LOG}" >&2
    exit 1
  fi
fi

repo_dir="$(pwd)"
rm -f "${LAUNCHER_LOG}" "${CONSOLE_LOG}"

echo "Starting R2-Dreamer run: ${RUN_NAME}"
echo "Config: ${CONFIG}"
echo "Run dir: ${RUN_DIR}"
echo "Launcher log: ${LAUNCHER_LOG}"
echo "Console log: ${CONSOLE_LOG}"
echo "Filter: ${PATTERN}"
echo "Monitor interval: ${MONITOR_INTERVAL}s"

bash -lc '
  set -euo pipefail
  cd "$1"
  export RUN_TRAINING=1
  python scripts/r2dreamer/build_commands.py --config "$2" --run "$3" --execute
' bash "${repo_dir}" "${CONFIG}" "${RUN_NAME}" >"${LAUNCHER_LOG}" 2>&1 &

run_pid="$!"
echo "${run_pid}" >"${PID_FILE}"
echo "PID: ${run_pid}"

run_is_active() {
  jobs -pr | grep -qx "${run_pid}"
}

stop_run() {
  echo "Stopping monitored run: ${RUN_NAME}" >&2
  if run_is_active; then
    kill "${run_pid}" 2>/dev/null || true
    wait "${run_pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  exit 130
}

trap stop_run INT TERM

while [[ ! -f "${CONSOLE_LOG}" ]] && run_is_active; do
  echo "Waiting for console.log; latest launcher output:"
  tail -n 20 "${LAUNCHER_LOG}" 2>/dev/null || true
  sleep 5
done

if [[ -f "${CONSOLE_LOG}" ]]; then
  echo "Monitoring live progress. Interrupting this cell stops the monitored run."
else
  echo "console.log was not created before the run exited."
fi

while run_is_active; do
  timestamp="$(date '+%H:%M:%S')"
  if [[ -f "${CONSOLE_LOG}" ]]; then
    latest="$(grep -E "${PATTERN}" "${CONSOLE_LOG}" 2>/dev/null | tail -n 1 || true)"
    if [[ -n "${latest}" ]]; then
      echo "[wm_poc-monitor ${timestamp}] ${latest}"
    else
      echo "[wm_poc-monitor ${timestamp}] waiting for matching trainer output..."
    fi
  else
    echo "[wm_poc-monitor ${timestamp}] waiting for console.log..."
  fi
  sleep "${MONITOR_INTERVAL}"
done

set +e
wait "${run_pid}"
status="$?"
set -e

rm -f "${PID_FILE}"

echo "Run finished with status ${status}: ${RUN_NAME}"
if [[ -f "${CONSOLE_LOG}" ]]; then
  echo "Recent monitored lines:"
  grep -E "${PATTERN}" "${CONSOLE_LOG}" 2>/dev/null | tail -n 20 || true
fi
echo "Final launcher output:"
tail -n 40 "${LAUNCHER_LOG}" 2>/dev/null || true
exit "${status}"
