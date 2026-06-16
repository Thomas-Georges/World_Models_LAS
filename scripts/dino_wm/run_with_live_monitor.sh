#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${DINO_MONITOR_DIR:-.}"
if [[ "${1:-}" == "--out-dir" ]]; then
  OUT_DIR="$2"
  shift 2
fi
if [[ "${1:-}" != "--" ]]; then
  echo "Usage: $0 [--out-dir DIR] -- command ..." >&2
  exit 2
fi
shift

mkdir -p "${OUT_DIR}"
MONITOR_FILE="${OUT_DIR}/gpu_monitor.csv"
echo "timestamp,index,name,memory_used_mb,memory_total_mb,utilization_gpu" >"${MONITOR_FILE}"

monitor() {
  while true; do
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >>"${MONITOR_FILE}" || true
    fi
    sleep 30
  done
}

monitor &
MONITOR_PID="$!"
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

"$@"
