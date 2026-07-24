#!/usr/bin/env bash
set -euo pipefail

cd /workspace

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:2}"
BATCH="${BATCH:-4}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-bass-ddsp-v2}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

SINGLE_STEPS="${SINGLE_STEPS:-50000}"
RIFF_STEPS="${RIFF_STEPS:-200000}"

START_LR="${START_LR:-1e-3}"
STOP_LR="${STOP_LR:-1e-4}"
DECAY_OVER="${DECAY_OVER:-200000}"

SINGLE_RUN="vanilla_dwts_single_note_${STAMP}"
RIFF_RUN="vanilla_dwts_riff_${STAMP}"

echo "Device: ${DEVICE}"
echo "Batch: ${BATCH}"
echo "Single-note DWTS run: ${SINGLE_RUN} (${SINGLE_STEPS} steps)"
echo "Riff DWTS run:        ${RIFF_RUN} (${RIFF_STEPS} steps)"

WANDB_FLAGS=()
if [[ "${WANDB}" == "1" || "${WANDB}" == "true" ]]; then
  WANDB_FLAGS+=(--wandb --wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_FLAGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
fi

python -m bass_ddsp.train \
  --config configs/vanilla_dwts_single_note.yaml \
  --name "${SINGLE_RUN}" \
  --root runs \
  --steps "${SINGLE_STEPS}" \
  --batch "${BATCH}" \
  --start-lr "${START_LR}" \
  --stop-lr "${STOP_LR}" \
  --decay-over "${DECAY_OVER}" \
  --device "${DEVICE}" \
  "${WANDB_FLAGS[@]}" \
  --wandb-name "${SINGLE_RUN}"

python -m bass_ddsp.train \
  --config configs/vanilla_dwts_riff.yaml \
  --name "${RIFF_RUN}" \
  --root runs \
  --steps "${RIFF_STEPS}" \
  --batch "${BATCH}" \
  --start-lr "${START_LR}" \
  --stop-lr "${STOP_LR}" \
  --decay-over "${DECAY_OVER}" \
  --device "${DEVICE}" \
  --init-state "runs/${SINGLE_RUN}/state.pth" \
  "${WANDB_FLAGS[@]}" \
  --wandb-name "${RIFF_RUN}"

python -m bass_ddsp.export_branch_debug \
  --run "runs/${RIFF_RUN}" \
  --out-dir "runs/${RIFF_RUN}/branch_debug_random3_label_pitch" \
  --seed 98765 \
  --num-samples 3 \
  --pitch-source labels \
  --device "${DEVICE}"

python -m bass_ddsp.visualize_debug_controls \
  --run "runs/${RIFF_RUN}" \
  --out-dir "runs/${RIFF_RUN}/control_debug_random3_label_pitch" \
  --seed 98765 \
  --num-samples 3 \
  --pitch-source labels \
  --device "${DEVICE}"

echo "Done."
echo "Single-note DWTS run: /workspace/runs/${SINGLE_RUN}"
echo "Riff DWTS run:        /workspace/runs/${RIFF_RUN}"
echo "Branch debug:         /workspace/runs/${RIFF_RUN}/branch_debug_random3_label_pitch"
echo "Control debug:        /workspace/runs/${RIFF_RUN}/control_debug_random3_label_pitch"
