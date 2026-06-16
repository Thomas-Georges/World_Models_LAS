#!/usr/bin/env bash
set -euo pipefail

CONFIG="${DINO_CONFIG:-configs/dino_wm/pointmaze_scratch_a100.yaml}"
SKIP_CACHE="false"
SKIP_PLAN="false"
CHECKPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --skip-cache)
      SKIP_CACHE="true"
      shift
      ;;
    --skip-plan)
      SKIP_PLAN="true"
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${RUN_DINO_WM:-0}" != "1" ]]; then
  echo "Dry run only. Set RUN_DINO_WM=1 to execute."
  python scripts/dino_wm/build_commands.py --config "${CONFIG}" --stage train --print || true
  if [[ -n "${CHECKPOINT}" ]]; then
    python scripts/dino_wm/build_commands.py --config "${CONFIG}" --stage plan --checkpoint "${CHECKPOINT}" --print || true
  fi
  exit 0
fi

if [[ "${DINO_INSTALL_DEPS:-1}" == "1" ]]; then
  python scripts/dino_wm/install_colab_deps.py --quiet
else
  echo "Skipping DINO-WM dependency install because DINO_INSTALL_DEPS=${DINO_INSTALL_DEPS}."
fi

source scripts/dino_wm/mujoco_runtime_env.sh
python scripts/dino_wm/verify_dino_wm_env.py
python scripts/dino_wm/verify_data.py --config "${CONFIG}" --write-manifest
if [[ "${SKIP_CACHE}" != "true" ]]; then
  # Installs wm_poc_precompute_latents.py / wm_poc_latent_dataset.py into the
  # upstream checkout, then encodes any missing episodes. The precompute
  # script is a no-op for configs with features.cache_enabled=false.
  python scripts/dino_wm/patch_latent_cache.py --config "${CONFIG}"
  python scripts/dino_wm/precompute_latents.py --config "${CONFIG}" --no-dry-run
fi

MODE="$(python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from wm_poc.dino_wm.configs import get_config_value, load_config, resolve_config
config = resolve_config(load_config(sys.argv[1]))
if get_config_value(config, "planner_ablation.checkpoint"):
    print("planner_only")
elif get_config_value(config, "finetuning.enabled", False):
    print("finetune")
else:
    print("scratch")
PY
)"

if [[ "${MODE}" == "planner_only" ]]; then
  CHECKPOINT="${CHECKPOINT:-$(python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from wm_poc.dino_wm.configs import get_config_value, load_config, resolve_config
config = resolve_config(load_config(sys.argv[1]))
value = get_config_value(config, "planner_ablation.checkpoint", "")
print("" if value is None else value)
PY
)}"
elif [[ "${MODE}" == "finetune" ]]; then
  bash scripts/dino_wm/run_finetune.sh --config "${CONFIG}"
else
  bash scripts/dino_wm/run_train.sh --config "${CONFIG}"
fi

if [[ "${SKIP_PLAN}" != "true" ]]; then
  if [[ "${MODE}" == "planner_only" ]]; then
    python - "${CONFIG}" <<'PY' | while IFS= read -r planner; do
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from wm_poc.dino_wm.configs import get_config_value, load_config, resolve_config
config = resolve_config(load_config(sys.argv[1]))
for item in get_config_value(config, "planner_ablation.planners", ["cem"]):
    print(item)
PY
      DINO_PLANNER="${planner}" bash scripts/dino_wm/run_plan.sh --config "${CONFIG}" --checkpoint "${CHECKPOINT}"
    done
    exit 0
  fi
  if [[ -n "${CHECKPOINT}" ]]; then
    bash scripts/dino_wm/run_plan.sh --config "${CONFIG}" --checkpoint "${CHECKPOINT}"
  else
    bash scripts/dino_wm/run_plan.sh --config "${CONFIG}"
  fi
fi
