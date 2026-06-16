#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_ROOT="${WM_POC_EXTERNAL_REPOS:-/content/drive/MyDrive/wm_poc/external_repos}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --external-root)
      EXTERNAL_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

clone_or_report() {
  local name="$1"
  local url="$2"
  local target="${EXTERNAL_ROOT}/${name}"

  if [[ -d "${target}/.git" ]]; then
    echo "Repository already exists: ${target}"
    git -C "${target}" rev-parse HEAD
    return 0
  fi

  if [[ -e "${target}" ]] && [[ -n "$(find "${target}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: ${target} exists but is not a git repository." >&2
    return 1
  fi

  echo "Cloning ${url} into ${target}"
  git clone "${url}" "${target}"
}

mkdir -p "${EXTERNAL_ROOT}"

clone_or_report "r2dreamer" "https://github.com/NM512/r2dreamer.git"
clone_or_report "dino_wm" "https://github.com/gaoyuezhou/dino_wm.git"

# Optional reference repo for later manual work:
# clone_or_report "jepa_wms_optional" "https://github.com/facebookresearch/jepa-wms.git"

echo "External repo clone step complete. No dependencies were installed and no training was run."
