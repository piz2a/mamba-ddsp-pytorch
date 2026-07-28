#!/usr/bin/env bash
set -euo pipefail

cd /workspace

DEVICE="${DEVICE:-cuda:4}"
RUN_NAME="vanilla_dwts_riff_resume_to_200k_20260726_071231"
COMPARE_DIR="runs/model_comparison_branchbalance_final_dwts_resumed_20260726_071231"
OLD_DWTS_RUN="runs/vanilla_dwts_riff_20260722_040632"
BASS_RUN="runs/bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513"
VANILLA_DDSP_RUN="runs/vanilla_ddsp_riff_20260720_090532"
STEP_OFFSET=32270
TARGET_STEPS=200000

python -u -m bass_ddsp.train \
  --config "${OLD_DWTS_RUN}/config.yaml" \
  --name "${RUN_NAME}" \
  --root runs \
  --steps "${TARGET_STEPS}" \
  --step-offset "${STEP_OFFSET}" \
  --batch 4 \
  --device "${DEVICE}" \
  --init-state "${OLD_DWTS_RUN}/state.pth" \
  --wandb \
  --wandb-project bass-ddsp-v2 \
  --wandb-name "${RUN_NAME}"

python -u -m bass_ddsp.compare_three_models \
  --model "Bass-DDSP fixed resume=${BASS_RUN}" \
  --model "Vanilla DWTS resumed=runs/${RUN_NAME}" \
  --model "Vanilla DDSP=${VANILLA_DDSP_RUN}" \
  --out-dir "${COMPARE_DIR}" \
  --seed 20260723 \
  --num-samples 32 \
  --num-plots 4 \
  --pitch-source labels \
  --device "${DEVICE}"

echo "Done."
echo "DWTS resumed run: /workspace/runs/${RUN_NAME}"
echo "Final comparison: /workspace/${COMPARE_DIR}"
