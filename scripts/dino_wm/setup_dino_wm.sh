#!/usr/bin/env bash
set -euo pipefail

REPORT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --report-dir)
      REPORT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

DINO_WM_REMOTE_URL="${DINO_WM_REMOTE_URL:-https://github.com/gaoyuezhou/dino_wm.git}"
DINO_WM_REPO="${DINO_WM_REPO:-/content/drive/MyDrive/wm_poc/external_repos/dino_wm}"

if [[ ! -d "${DINO_WM_REPO}/.git" ]]; then
  if [[ -e "${DINO_WM_REPO}" ]] && [[ -n "$(find "${DINO_WM_REPO}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: ${DINO_WM_REPO} exists but is not a git checkout." >&2
    echo "Move it aside or set DINO_WM_REPO to a clean path before setup." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${DINO_WM_REPO}")"
  git clone "${DINO_WM_REMOTE_URL}" "${DINO_WM_REPO}"
else
  if [[ -n "$(git -C "${DINO_WM_REPO}" status --short)" ]]; then
    echo "Upstream DINO-WM repo has local changes; leaving it untouched."
  else
    echo "Upstream DINO-WM repo exists: ${DINO_WM_REPO}"
  fi
fi

# Pin to a specific upstream commit. Defaults to the locked SHA in
# external_revisions.lock (the report-run revision); export DINO_WM_COMMIT=<sha>
# or DINO_WM_COMMIT=main to override for exploratory work.
if [[ -z "${DINO_WM_COMMIT:-}" ]]; then
  DINO_WM_COMMIT="$(python3 "$(dirname "${BASH_SOURCE[0]}")/../read_lock.py" dino_wm commit 2>/dev/null || true)"
fi
DINO_WM_COMMIT="${DINO_WM_COMMIT:-}"
if [[ -n "${DINO_WM_COMMIT}" && "${DINO_WM_COMMIT}" != "main" ]]; then
  if [[ -n "$(git -C "${DINO_WM_REPO}" status --short)" ]]; then
    echo "DINO-WM repo has local changes; not switching to commit ${DINO_WM_COMMIT}." >&2
  else
    git -C "${DINO_WM_REPO}" fetch --quiet origin "${DINO_WM_COMMIT}" 2>/dev/null \
      || git -C "${DINO_WM_REPO}" fetch --quiet --all
    git -C "${DINO_WM_REPO}" checkout --quiet "${DINO_WM_COMMIT}"
    echo "Checked out DINO-WM commit ${DINO_WM_COMMIT}"
  fi
fi

if [[ ! -f "${DINO_WM_REPO}/train.py" || ! -f "${DINO_WM_REPO}/plan.py" ]]; then
  echo "ERROR: upstream DINO-WM checkout is missing train.py or plan.py: ${DINO_WM_REPO}" >&2
  echo "Check DINO_WM_REMOTE_URL or set DINO_WM_REPO to the upstream DINO-WM repository." >&2
  exit 1
fi

if [[ "${DINO_PATCH_STEP_CHECKPOINTING:-0}" == "1" ]]; then
  python scripts/dino_wm/patch_step_checkpointing.py
else
  python scripts/dino_wm/patch_step_checkpointing.py --restore
fi
python scripts/dino_wm/patch_mixed_precision.py

if [[ "${DINO_INSTALL_DEPS:-1}" == "1" ]]; then
  python scripts/dino_wm/install_colab_deps.py --quiet
else
  echo "Skipping DINO-WM dependency install because DINO_INSTALL_DEPS=${DINO_INSTALL_DEPS}."
fi

source scripts/dino_wm/mujoco_runtime_env.sh
python scripts/verify_environment.py --cpu-only
if [[ "${DINO_INSTALL_DEPS:-1}" == "1" ]]; then
  python scripts/dino_wm/verify_dino_wm_env.py --allow-cpu || true
else
  python scripts/dino_wm/verify_dino_wm_env.py --allow-cpu --skip-dependency-check || true
fi

if [[ -n "${REPORT_DIR}" ]]; then
  mkdir -p "${REPORT_DIR}"
  python - "${REPORT_DIR}" "${DINO_WM_REPO}" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

report_dir = Path(sys.argv[1])
repo = Path(sys.argv[2])
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=repo,
    check=False,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
).stdout.strip() if repo.is_dir() else ""
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "dino_wm_repo": str(repo),
    "dino_wm_commit": commit,
}
(report_dir / "setup_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
fi
