#!/usr/bin/env bash
set -euo pipefail

CONFIG="${R2_CONFIG:-configs/r2dreamer/three_way_walker_walk_to_run.yaml}"

if [[ "${RUN_TRAINING:-0}" != "1" ]]; then
  python scripts/r2dreamer/build_commands.py --config "${CONFIG}" --run target_scratch --print-only
  echo "Set RUN_TRAINING=1 to execute target_scratch training."
  exit 0
fi

python scripts/r2dreamer/build_commands.py --config "${CONFIG}" --run target_scratch --execute
