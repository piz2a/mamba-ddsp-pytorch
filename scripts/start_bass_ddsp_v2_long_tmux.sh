#!/usr/bin/env bash
set -euo pipefail

cd /workspace

SESSION="${SESSION:-bass_ddsp_v2_long}"
LOG_DIR="${LOG_DIR:-runs/tmux_logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${SESSION}_${STAMP}.log"
MAX_TIME="${MAX_TIME:-10h}"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "Attach with: tmux attach -t ${SESSION}"
  exit 1
fi

COMMAND="cd /workspace && WANDB=1 timeout ${MAX_TIME} bash scripts/train_bass_ddsp_v2_full.sh 2>&1 | tee ${LOG_FILE}"
tmux new-session -d -s "${SESSION}" "${COMMAND}"

echo "Started tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: /workspace/${LOG_FILE}"
