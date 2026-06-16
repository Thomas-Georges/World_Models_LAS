#!/usr/bin/env bash
set -euo pipefail

exec python scripts/dino_wm/download_data.py "$@"
