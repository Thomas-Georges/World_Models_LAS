#!/usr/bin/env bash
set -euo pipefail

CONFIG="${DINO_CONFIG:-configs/dino_wm/pointmaze_scratch_a100.yaml}"
CHECKPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
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

if [[ -z "${CHECKPOINT}" ]]; then
  # Auto-resolve the run's latest epoch checkpoint from the checkpoint tree
  # (ckpt_root/outputs/<run_name>/checkpoints/), not the log tree.
  CHECKPOINT="$(python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from wm_poc.dino_wm.checkpoints import find_latest_checkpoint
from wm_poc.dino_wm.configs import load_config, resolve_config
from wm_poc.dino_wm.resume import checkpoint_output_dir
config = resolve_config(load_config(sys.argv[1]))
checkpoint = find_latest_checkpoint(checkpoint_output_dir(config))
print(checkpoint or "")
PY
)"
fi

if [[ "${RUN_DINO_WM:-0}" != "1" ]]; then
  echo "Dry run only. Set RUN_DINO_WM=1 to execute."
  if [[ -n "${CHECKPOINT}" ]]; then
    python scripts/dino_wm/build_commands.py --config "${CONFIG}" --stage plan --checkpoint "${CHECKPOINT}" --print
  else
    python scripts/dino_wm/build_commands.py --config "${CONFIG}" --stage plan --print
  fi
  exit 0
fi

# Upstream plan.py imports hydra/omegaconf; a fresh Colab session has none of
# the upstream deps, so install them here (no-op when already present) the same
# way run_experiment.sh does -- otherwise standalone planning fails with
# "ModuleNotFoundError: No module named 'hydra'". Set DINO_INSTALL_DEPS=0 to skip.
if [[ "${DINO_INSTALL_DEPS:-1}" == "1" ]]; then
  python scripts/dino_wm/install_colab_deps.py --quiet
else
  echo "Skipping DINO-WM dependency install because DINO_INSTALL_DEPS=${DINO_INSTALL_DEPS}."
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
# A latent-trained checkpoint's saved config references the latent dataset,
# and the cache lives on session-local disk. Rebuild any missing coverage
# before planning; both steps are no-ops when the cache already covers the
# config or latent training is disabled.
python scripts/dino_wm/patch_latent_cache.py --config "${CONFIG}"
python scripts/dino_wm/patch_evaluator_video.py --config "${CONFIG}"
python scripts/dino_wm/precompute_latents.py --config "${CONFIG}" --no-dry-run
RUN_DIR="$(python scripts/dino_wm/archive_run_metadata.py --config "${CONFIG}" --stage plan --checkpoint "${CHECKPOINT}" --print-run-dir)"
PLANNER_NAME="${DINO_PLANNER:-$(python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from wm_poc.dino_wm.configs import get_config_value, load_config, resolve_config
config = resolve_config(load_config(sys.argv[1]))
print(get_config_value(config, "planning.planner", "cem"))
PY
)}"
if [[ -f "${RUN_DIR}/planning/command_${PLANNER_NAME}.sh" ]]; then
  COMMAND="$(cat "${RUN_DIR}/planning/command_${PLANNER_NAME}.sh")"
else
  COMMAND="$(cat "${RUN_DIR}/command.sh")"
fi
DINO_MAX_WALL_MINUTES="${DINO_MAX_WALL_MINUTES:-$(python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from wm_poc.dino_wm.configs import get_config_value, load_config, resolve_config
config = resolve_config(load_config(sys.argv[1]))
print(get_config_value(config, "planning.max_wall_minutes", 60))
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
  timeout --signal=INT "${DINO_MAX_WALL_MINUTES}m" bash -lc "${COMMAND}" >>"${RUN_DIR}/stdout.log" 2>>"${RUN_DIR}/stderr.log"
  RC=$?
else
  echo "timeout command not found; running without timeout wrapper." >>"${RUN_DIR}/stderr.log"
  bash -lc "${COMMAND}" >>"${RUN_DIR}/stdout.log" 2>>"${RUN_DIR}/stderr.log"
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

python - "${RUN_DIR}" plan "${RC}" "${TIMED_OUT}" "${STARTED_AT}" "${ENDED_AT}" "${ELAPSED_SECONDS}" "${DINO_MAX_WALL_MINUTES}" "${PLANNER_NAME}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
return_code = int(sys.argv[3])
timed_out = sys.argv[4] == "true"
planner = sys.argv[9]
payload = {
    "run_name": run_dir.name,
    "stage": sys.argv[2],
    "completed": return_code == 0 and not timed_out,
    "failed": return_code != 0 and not timed_out,
    "timed_out": timed_out,
    "return_code": return_code,
    "started_at": sys.argv[5],
    "ended_at": sys.argv[6],
    "elapsed_seconds": int(sys.argv[7]),
    "max_wall_minutes": float(sys.argv[8]),
    "planner": planner,
}
status_path = run_dir / "planning" / f"status_{planner}.json"
status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if not (run_dir / "status.json").exists():
    (run_dir / "status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

python scripts/dino_wm/summarize_runs.py --root "$(dirname "${RUN_DIR}")" --out "$(dirname "${RUN_DIR}")/_summary" || true
if [[ "${RC}" != "0" ]]; then
  echo "DINO-WM plan command failed with exit code ${RC}." >&2
  echo "stdout log: ${RUN_DIR}/stdout.log" >&2
  tail -n 80 "${RUN_DIR}/stdout.log" >&2 || true
  echo "stderr log: ${RUN_DIR}/stderr.log" >&2
  tail -n 120 "${RUN_DIR}/stderr.log" >&2 || true
fi
exit "${RC}"
