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
    echo "Run already appears active: ${RUN_NAME}"
    echo "PID: ${old_pid}"
    echo "Launcher log: ${LAUNCHER_LOG}"
    echo "Console log: ${RUN_DIR}/console.log"
    exit 0
  fi
fi

repo_dir="$(pwd)"
echo "Starting R2-Dreamer run in background: ${RUN_NAME}"
echo "Config: ${CONFIG}"
echo "Run dir: ${RUN_DIR}"
echo "Launcher log: ${LAUNCHER_LOG}"

nohup bash -lc '
  set -euo pipefail
  cd "$1"
  export RUN_TRAINING=1
  python scripts/r2dreamer/build_commands.py --config "$2" --run "$3" --execute
' bash "${repo_dir}" "${CONFIG}" "${RUN_NAME}" >"${LAUNCHER_LOG}" 2>&1 &

pid="$!"
echo "${pid}" >"${PID_FILE}"
echo "PID: ${pid}"
echo "Monitor with:"
echo "  bash scripts/r2dreamer/tail_run_progress.sh ${RUN_NAME}"
