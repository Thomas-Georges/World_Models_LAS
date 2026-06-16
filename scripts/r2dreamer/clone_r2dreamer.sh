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
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "${TARGET_DIR}")"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "Updating existing r2dreamer repo at ${TARGET_DIR}"
  git -C "${TARGET_DIR}" fetch --all --tags
elif [[ -e "${TARGET_DIR}" ]] && [[ -n "$(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: ${TARGET_DIR} exists but is not an empty directory or git repo." >&2
  exit 1
else
  echo "Cloning ${REMOTE} into ${TARGET_DIR}"
  git clone "${REMOTE}" "${TARGET_DIR}"
fi

git -C "${TARGET_DIR}" checkout "${COMMIT}"

echo "remote URL: $(git -C "${TARGET_DIR}" remote get-url origin)"
echo "current branch: $(git -C "${TARGET_DIR}" branch --show-current || true)"
echo "current commit: $(git -C "${TARGET_DIR}" rev-parse HEAD)"
echo "status:"
git -C "${TARGET_DIR}" status --short
