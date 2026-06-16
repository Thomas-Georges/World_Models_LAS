#!/usr/bin/env bash
set -euo pipefail

CONFIG="${R2_CONFIG:-configs/r2dreamer/three_way_walker_walk_to_run.yaml}"

if [[ "${RUN_TRAINING:-0}" != "1" ]]; then
  python scripts/r2dreamer/build_commands.py --config "${CONFIG}" --run source_base --print-only
  echo "Set RUN_TRAINING=1 to execute source_base training."
  exit 0
fi

python scripts/r2dreamer/build_commands.py --config "${CONFIG}" --run source_base --execute
