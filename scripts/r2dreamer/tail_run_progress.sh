#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-${R2_RUN:-source_base}}"
DRIVE_ROOT="${WM_POC_DRIVE_ROOT:-/content/drive/MyDrive/wm_poc}"
LOG_DIR="${WM_POC_LOG_DIR:-${DRIVE_ROOT}/logs}"
LOG_ROOT="${R2_LOG_ROOT:-${LOG_DIR}/r2dreamer}"
RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
CONSOLE_LOG="${RUN_DIR}/console.log"
LAUNCHER_LOG="${RUN_DIR}/launcher.log"
TAIL_LINES="${R2_TAIL_LINES:-80}"
PATTERN="${R2_TAIL_PATTERN:-\\[wm_poc\\] progress|\\[wm_poc\\] Using serial envs|Saved checkpoint|Saved interval checkpoint|Logdir|Create envs|Simulate agent|Encoder|Optimizer has|Compiling update function|Evaluating|Traceback|RuntimeError|Error executing job|ERROR|Exception|Lost connection|Segmentation fault}"

echo "Run dir: ${RUN_DIR}"
echo "Console log: ${CONSOLE_LOG}"
echo "Launcher log: ${LAUNCHER_LOG}"
echo "Filter: ${PATTERN}"

while [[ ! -f "${CONSOLE_LOG}" ]]; do
  if [[ -f "${LAUNCHER_LOG}" ]]; then
    echo "Waiting for console.log; latest launcher output:"
    tail -n 20 "${LAUNCHER_LOG}" || true
  else
    echo "Waiting for console.log..."
  fi
  sleep 5
done

echo "Tailing progress. Interrupt this cell to stop monitoring; the background run keeps going."
if grep --help 2>/dev/null | grep -q -- "--line-buffered"; then
  tail -n "${TAIL_LINES}" -F "${CONSOLE_LOG}" | grep --line-buffered -E "${PATTERN}"
else
  tail -n "${TAIL_LINES}" -F "${CONSOLE_LOG}" | grep -E "${PATTERN}"
fi
