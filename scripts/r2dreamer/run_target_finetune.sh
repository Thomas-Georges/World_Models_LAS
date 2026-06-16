#!/usr/bin/env bash
set -euo pipefail

CONFIG="${R2_CONFIG:-configs/r2dreamer/three_way_walker_walk_to_run.yaml}"
DRIVE_ROOT="${WM_POC_DRIVE_ROOT:-/content/drive/MyDrive/wm_poc}"
LOG_DIR="${WM_POC_LOG_DIR:-${DRIVE_ROOT}/logs}"
R2_LOG_ROOT="${R2_LOG_ROOT:-${LOG_DIR}/r2dreamer}"
SOURCE_CKPT="${R2_SOURCE_CKPT:-${R2_LOG_ROOT}/source_base/latest.pt}"

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "Missing source checkpoint: ${SOURCE_CKPT}" >&2
  echo "Run source_base first or set R2_SOURCE_CKPT." >&2
  if [[ "${RUN_TRAINING:-0}" == "1" ]]; then
    exit 1
  fi
fi

if [[ "${RUN_TRAINING:-0}" != "1" ]]; then
  python scripts/r2dreamer/build_commands.py --config "${CONFIG}" --run target_finetune --print-only
  echo "Set RUN_TRAINING=1 to execute target_finetune training."
  exit 0
fi

python scripts/r2dreamer/build_commands.py --config "${CONFIG}" --run target_finetune --execute
