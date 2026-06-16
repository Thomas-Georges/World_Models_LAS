#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${R2DREAMER_REPO:-/content/external_repos/r2dreamer}"
REMOTE="${R2DREAMER_REMOTE:-https://github.com/NM512/r2dreamer.git}"
# Default to the locked R2-Dreamer SHA in external_revisions.lock; the
# R2DREAMER_COMMIT env var (or --commit below) overrides for exploratory work.
COMMIT="${R2DREAMER_COMMIT:-}"
if [[ -z "${COMMIT}" ]]; then
  COMMIT="$(python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/read_lock.py" r2dreamer commit 2>/dev/null || true)"
fi
COMMIT="${COMMIT:-main}"
EXTRAS="${R2_EXTRAS:-dmc}"
ALLOW_UNSUPPORTED_PYTHON=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --commit)
      COMMIT="$2"
      shift 2
      ;;
    --extras)
      EXTRAS="$2"
      shift 2
      ;;
    --allow-unsupported-python)
      ALLOW_UNSUPPORTED_PYTHON=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/clone_r2dreamer.sh" \
  --target-dir "${TARGET_DIR}" \
  --remote "${REMOTE}" \
  --commit "${COMMIT}"

install_args=(
  --r2-repo "${TARGET_DIR}"
  --extras "${EXTRAS}"
)
if [[ "${ALLOW_UNSUPPORTED_PYTHON}" -eq 1 ]]; then
  install_args+=(--allow-unsupported-python)
fi

bash "${SCRIPT_DIR}/install_r2dreamer.sh" "${install_args[@]}"

cat <<EOF

R2-Dreamer setup complete.

Recommended MuJoCo headless variables:
  export MUJOCO_GL=osmesa
  export PYOPENGL_PLATFORM=osmesa
  export MUJOCO_EGL_DEVICE_ID=0

Use EGL only after confirming it is stable in this Colab runtime:
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export R2_ALLOW_EGL=1

Current r2dreamer commit:
  $(git -C "${TARGET_DIR}" rev-parse HEAD)

No training was run.
EOF
