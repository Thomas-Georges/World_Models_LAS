#!/usr/bin/env bash
set -euo pipefail

CONFIG="${DINO_CONFIG:-configs/dino_wm/pointmaze_lowdata_finetune_a100.yaml}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
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
  python scripts/dino_wm/build_commands.py --config "${CONFIG}" --stage train --print
  exit 0
fi

python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from wm_poc.dino_wm.configs import get_config_value, load_config, resolve_config, validate_config
config = resolve_config(load_config(sys.argv[1]))
validate_config(config)
if not get_config_value(config, "finetuning.enabled", False):
    raise SystemExit("finetuning.enabled must be true for run_finetune.sh")
checkpoint = get_config_value(config, "finetuning.init_from")
if checkpoint in {None, "", "null"}:
    raise SystemExit("finetuning.init_from is required.")
if not Path(str(checkpoint)).expanduser().is_file():
    raise SystemExit(f"finetuning.init_from does not exist: {checkpoint}")
PY

DINO_STAGE_NAME="finetune" DINO_CONFIG="${CONFIG}" bash scripts/dino_wm/run_train.sh --config "${CONFIG}"
