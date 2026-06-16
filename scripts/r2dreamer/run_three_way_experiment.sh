#!/usr/bin/env bash
set -euo pipefail

SKIP_EXISTING=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

CONFIG="${R2_CONFIG:-configs/r2dreamer/three_way_walker_walk_to_run.yaml}"
DRIVE_ROOT="${WM_POC_DRIVE_ROOT:-/content/drive/MyDrive/wm_poc}"
LOG_DIR="${WM_POC_LOG_DIR:-${DRIVE_ROOT}/logs}"
FIG_BASE="${WM_POC_FIGURE_DIR:-${WM_POC_FIG_DIR:-${DRIVE_ROOT}/figures}}"
FIG_ROOT="${R2_FIGURE_DIR:-${FIG_BASE}/r2dreamer}"
R2_LOG_ROOT="${R2_LOG_ROOT:-${LOG_DIR}/r2dreamer}"
R2DREAMER_REPO="${R2DREAMER_REPO:-/content/external_repos/r2dreamer}"

echo "Config: ${CONFIG}"
echo "R2DREAMER_REPO: ${R2DREAMER_REPO}"
echo "R2_LOG_ROOT: ${R2_LOG_ROOT}"
echo "R2_FIGURE_DIR: ${FIG_ROOT}"

if [[ "${RUN_TRAINING:-0}" != "1" ]]; then
  python scripts/r2dreamer/build_commands.py --config "${CONFIG}" --print-only
  echo "Set RUN_TRAINING=1 to execute the three-way experiment."
  exit 0
fi

if [[ ! -d "${R2DREAMER_REPO}/.git" ]]; then
  echo "Missing r2dreamer repo at ${R2DREAMER_REPO}" >&2
  exit 1
fi

python scripts/r2dreamer/verify_r2dreamer_patch.py --r2-repo "${R2DREAMER_REPO}"

if [[ "${SKIP_EXISTING}" -eq 1 && -f "${R2_LOG_ROOT}/source_base/latest.pt" ]]; then
  echo "Skipping source_base because checkpoint exists."
else
  bash scripts/r2dreamer/run_source_base.sh
fi

python scripts/r2dreamer/verify_checkpoint.py \
  --checkpoint "${R2_LOG_ROOT}/source_base/latest.pt"

bash scripts/r2dreamer/run_target_finetune.sh
bash scripts/r2dreamer/run_target_scratch.sh

python scripts/r2dreamer/summarize_runs.py \
  --run-root "${R2_LOG_ROOT}" \
  --out "${R2_LOG_ROOT}/summary.csv"

python scripts/r2dreamer/plot_finetune_vs_scratch.py \
  --finetune "${R2_LOG_ROOT}/target_finetune/metrics.jsonl" \
  --scratch "${R2_LOG_ROOT}/target_scratch/metrics.jsonl" \
  --out "${FIG_ROOT}/finetune_vs_scratch.png"
