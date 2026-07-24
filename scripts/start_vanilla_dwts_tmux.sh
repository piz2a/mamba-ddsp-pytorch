#!/usr/bin/env bash
set -euo pipefail

cd /workspace

SESSION="${SESSION:-vanilla_dwts_full}"
LOG_DIR="${LOG_DIR:-runs/tmux_logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${SESSION}_${STAMP}.log"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "Attach with: tmux attach -t ${SESSION}"
  exit 1
fi

COMMAND="cd /workspace && bash scripts/train_vanilla_dwts_full.sh 2>&1 | tee ${LOG_FILE}"
tmux new-session -d -s "${SESSION}" "${COMMAND}"

echo "Started tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: /workspace/${LOG_FILE}"
