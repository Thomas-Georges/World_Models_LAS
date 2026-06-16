#!/usr/bin/env bash
# End-to-end local/global smoke on the synthetic point-mass task (CPU, ~1-2 min):
# export transitions -> train a tiny surrogate -> run planner smokes -> summarize.
# Artifacts land under ${LG_SMOKE_ROOT:-runs/local_global} (gitignored).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python3}"
export LG_SMOKE_ROOT="${LG_SMOKE_ROOT:-$REPO_ROOT/runs/local_global}"
CONFIG="configs/local_global/smoke_synthetic.yaml"
RUN_DIR="$LG_SMOKE_ROOT/smoke_synthetic"

echo "== local/global smoke: $CONFIG -> $RUN_DIR =="
# The smoke is a fresh end-to-end check; clear smoke-owned artifacts so an
# interrupted earlier attempt (half-written synthetic cache) cannot wedge it.
rm -rf "$LG_SMOKE_ROOT/_synthetic" "$RUN_DIR"
"$PYTHON" scripts/local_global/export_transitions.py --config "$CONFIG" --run-dir "$RUN_DIR" --dry-run
"$PYTHON" scripts/local_global/export_transitions.py --config "$CONFIG" --run-dir "$RUN_DIR"
"$PYTHON" scripts/local_global/train_local_surrogate.py --config "$CONFIG" --run-dir "$RUN_DIR" --smoke
"$PYTHON" scripts/local_global/run_planning_eval.py --config "$CONFIG" --run-dir "$RUN_DIR" --smoke \
  --planners global_cem local_gd local_adam local_cem hybrid_cem_local_refine hybrid_cem_local_refine_global_rescore
"$PYTHON" scripts/local_global/summarize_runs.py --run-root "$LG_SMOKE_ROOT"

echo "== smoke OK: summary at $LG_SMOKE_ROOT/_summary/summary.csv =="
