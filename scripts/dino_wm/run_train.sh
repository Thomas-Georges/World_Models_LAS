#!/usr/bin/env bash
set -euo pipefail

CONFIG="${DINO_CONFIG:-configs/dino_wm/pointmaze_scratch_a100.yaml}"
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

STAGE="${DINO_STAGE_NAME:-train}"

if [[ "${RUN_DINO_WM:-0}" != "1" ]]; then
  echo "Dry run only. Set RUN_DINO_WM=1 to execute."
  python scripts/dino_wm/build_commands.py --config "${CONFIG}" --stage train --print
  exit 0
fi

source scripts/dino_wm/mujoco_runtime_env.sh
UPSTREAM_REPO="$(python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from wm_poc.dino_wm.configs import load_config, resolve_config
config = resolve_config(load_config(sys.argv[1]))
repo = Path(str(config.get("external_repo", "external_repos/dino_wm"))).expanduser()
if not repo.is_absolute():
    repo = Path.cwd() / repo
print(repo)
PY
)"
wm_poc_enable_dino_wm_imports "${UPSTREAM_REPO}"
if [[ "${DINO_PATCH_STEP_CHECKPOINTING:-0}" == "1" ]]; then
  python scripts/dino_wm/patch_step_checkpointing.py --config "${CONFIG}"
else
  python scripts/dino_wm/patch_step_checkpointing.py --config "${CONFIG}" --restore
fi
python scripts/dino_wm/patch_mixed_precision.py --config "${CONFIG}"
python scripts/dino_wm/patch_val_no_grad.py --config "${CONFIG}"
python scripts/dino_wm/patch_finetune_loading.py --config "${CONFIG}"
python scripts/dino_wm/patch_latent_cache.py --config "${CONFIG}"
python scripts/dino_wm/prepare_training_resume.py --config "${CONFIG}"
RUN_DIR="$(python scripts/dino_wm/archive_run_metadata.py --config "${CONFIG}" --stage "${STAGE}" --print-run-dir)"
COMMAND="$(cat "${RUN_DIR}/command.sh")"
DINO_MAX_WALL_MINUTES="${DINO_MAX_WALL_MINUTES:-$(python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from wm_poc.dino_wm.configs import get_config_value, load_config, resolve_config
config = resolve_config(load_config(sys.argv[1]))
print(get_config_value(config, "training.max_wall_minutes", 220))
PY
)}"

STARTED_AT="$(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"
START_SECONDS="$(date +%s)"

export WANDB_MODE="${WANDB_MODE:-offline}"

set +e
if command -v timeout >/dev/null 2>&1; then
  timeout --signal=INT "${DINO_MAX_WALL_MINUTES}m" bash -lc "${COMMAND}" >"${RUN_DIR}/stdout.log" 2>"${RUN_DIR}/stderr.log"
  RC=$?
else
  echo "timeout command not found; running without timeout wrapper." >"${RUN_DIR}/stderr.log"
  bash -lc "${COMMAND}" >"${RUN_DIR}/stdout.log" 2>>"${RUN_DIR}/stderr.log"
  RC=$?
fi
set -e

ENDED_AT="$(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"
END_SECONDS="$(date +%s)"
ELAPSED_SECONDS="$((END_SECONDS - START_SECONDS))"
TIMED_OUT="false"
if [[ "${RC}" == "124" || "${RC}" == "130" ]]; then
  TIMED_OUT="true"
fi

python - "${RUN_DIR}" "${STAGE}" "${RC}" "${TIMED_OUT}" "${STARTED_AT}" "${ENDED_AT}" "${ELAPSED_SECONDS}" "${DINO_MAX_WALL_MINUTES}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
stage = sys.argv[2]
return_code = int(sys.argv[3])
timed_out = sys.argv[4] == "true"
payload = {
    "run_name": run_dir.name,
    "stage": stage,
    "completed": return_code == 0 and not timed_out,
    "failed": return_code != 0 and not timed_out,
    "timed_out": timed_out,
    "return_code": return_code,
    "started_at": sys.argv[5],
    "ended_at": sys.argv[6],
    "elapsed_seconds": int(sys.argv[7]),
    "max_wall_minutes": float(sys.argv[8]),
}
(run_dir / "status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

python scripts/dino_wm/summarize_runs.py --root "$(dirname "${RUN_DIR}")" --out "$(dirname "${RUN_DIR}")/_summary" || true
if [[ "${RC}" != "0" ]]; then
  echo "DINO-WM train command failed with exit code ${RC}." >&2
  echo "stdout log: ${RUN_DIR}/stdout.log" >&2
  tail -n 80 "${RUN_DIR}/stdout.log" >&2 || true
  echo "stderr log: ${RUN_DIR}/stderr.log" >&2
  tail -n 120 "${RUN_DIR}/stderr.log" >&2 || true
fi
exit "${RC}"
