#!/usr/bin/env bash
set -euo pipefail

R2_REPO="${R2DREAMER_REPO:-/content/external_repos/r2dreamer}"
EXTRAS="${R2_EXTRAS:-dmc}"
ALLOW_UNSUPPORTED_PYTHON=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --r2-repo|--target-dir)
      R2_REPO="$2"
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

if [[ ! -f "${R2_REPO}/pyproject.toml" ]]; then
  echo "ERROR: Missing pyproject.toml in ${R2_REPO}" >&2
  exit 1
fi

set +e
python - <<'PY'
import sys
print("Python:", sys.version)
supported = sys.version_info[:2] == (3, 11)
if not supported:
    print("WARNING: upstream r2dreamer requires Python >=3.11,<3.12.")
    print("This interpreter is outside that range; switch to a Python 3.11 runtime.")
    sys.exit(42)
PY
status=$?
set -e
if [[ "${status}" -eq 42 && "${ALLOW_UNSUPPORTED_PYTHON}" -ne 1 ]]; then
  echo "Refusing to install on unsupported Python. Use a Python 3.11 runtime (Colab: Runtime -> Change runtime type). Pass --allow-unsupported-python only to force an unsupported install." >&2
  exit 42
elif [[ "${status}" -ne 0 && "${status}" -ne 42 ]]; then
  exit "${status}"
fi

python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except Exception as exc:
    print("torch check failed:", repr(exc))
PY

cd "${R2_REPO}"
pip_args=()
if [[ "${ALLOW_UNSUPPORTED_PYTHON}" -eq 1 ]]; then
  pip_args+=(--ignore-requires-python)
fi
if [[ -n "${EXTRAS}" ]]; then
  python -m pip install "${pip_args[@]}" -e ".[${EXTRAS}]"
else
  python -m pip install "${pip_args[@]}" -e .
fi

python - <<'PY'
import importlib.metadata as md
for name in ["torch", "hydra-core", "mujoco", "dm_control", "r2dreamer"]:
    try:
        print(f"{name}: {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}: not installed")
PY
