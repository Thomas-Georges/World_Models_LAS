#!/usr/bin/env bash
set -euo pipefail

CONFIG="${DINO_CONFIG:-configs/dino_wm/smoke_pointmaze.yaml}"

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

if [[ "${RUN_DINO_WM:-0}" != "1" ]]; then
  echo "Dry run only. Set RUN_DINO_WM=1 to execute."
  python scripts/dino_wm/build_commands.py --config "${CONFIG}" --stage train --print
  exit 0
fi

if [[ ! -f "${UPSTREAM_REPO}/train.py" || ! -f "${UPSTREAM_REPO}/plan.py" ]]; then
  echo "ERROR: upstream DINO-WM repo is missing train.py or plan.py: ${UPSTREAM_REPO}" >&2
  echo "Run this setup command first:" >&2
  echo "  DINO_WM_REPO=${UPSTREAM_REPO} bash scripts/dino_wm/setup_dino_wm.sh" >&2
  echo "Or set DINO_AUTO_SETUP=1 to let run_smoke.sh run setup_dino_wm.sh automatically." >&2
  if [[ "${DINO_AUTO_SETUP:-0}" == "1" ]]; then
    DINO_WM_REPO="${UPSTREAM_REPO}" bash scripts/dino_wm/setup_dino_wm.sh
  else
    exit 1
  fi
fi

bash scripts/dino_wm/run_experiment.sh --config "${CONFIG}"
