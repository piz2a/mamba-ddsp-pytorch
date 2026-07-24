#!/usr/bin/env bash
set -euo pipefail

cd /workspace

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:0}"
BATCH="${BATCH:-4}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-bass-ddsp-v2}"
WANDB_ENTITY="${WANDB_ENTITY:-piz2a-snu}"

ORIGINAL_SINGLE_RUN="${ORIGINAL_SINGLE_RUN:-bass_ddsp_v2_branchbalance_single_note_20260723_052406}"
ORIGINAL_SINGLE_STATE="${ORIGINAL_SINGLE_STATE:-runs/${ORIGINAL_SINGLE_RUN}/state.pth}"

TARGET_SINGLE_TOTAL_STEPS="${TARGET_SINGLE_TOTAL_STEPS:-50000}"
COMPLETED_SINGLE_STEPS="${COMPLETED_SINGLE_STEPS:-11768}"
SINGLE_STEPS="${SINGLE_STEPS:-$((TARGET_SINGLE_TOTAL_STEPS - COMPLETED_SINGLE_STEPS))}"
RIFF_STEPS="${RIFF_STEPS:-200000}"

# Continue the original exponential LR schedule instead of restarting at 1e-3.
# The crashed run's last logged LR at step 11767 was approximately 8.733e-4.
SINGLE_START_LR="${SINGLE_START_LR:-8.733030963920689e-4}"
SINGLE_STOP_LR="${SINGLE_STOP_LR:-5.623413251903491e-4}"
SINGLE_DECAY_OVER="${SINGLE_DECAY_OVER:-${SINGLE_STEPS}}"

RIFF_START_LR="${RIFF_START_LR:-1e-3}"
RIFF_STOP_LR="${RIFF_STOP_LR:-1e-4}"
RIFF_DECAY_OVER="${RIFF_DECAY_OVER:-200000}"

SINGLE_RUN="${SINGLE_RUN:-bass_ddsp_v2_branchbalance_single_note_20260723_052406_resume_to_50k_${STAMP}}"
RIFF_RUN="${RIFF_RUN:-bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_${STAMP}}"
COMPARE_DIR="${COMPARE_DIR:-runs/model_comparison_branchbalance_fixedresume_dwts_vanilla_${STAMP}}"

if [[ ! -f "${ORIGINAL_SINGLE_STATE}" ]]; then
  echo "Missing checkpoint: ${ORIGINAL_SINGLE_STATE}" >&2
  exit 1
fi

echo "Device: ${DEVICE}"
echo "Batch: ${BATCH}"
echo "Original single-note checkpoint: ${ORIGINAL_SINGLE_STATE}"
echo "Single-note continuation run: ${SINGLE_RUN} (${SINGLE_STEPS} steps)"
echo "Equivalent total single-note updates: ${TARGET_SINGLE_TOTAL_STEPS}"
echo "Riff run: ${RIFF_RUN} (${RIFF_STEPS} steps)"
echo "Comparison dir: ${COMPARE_DIR}"
echo "W&B enabled: ${WANDB}"

WANDB_FLAGS=()
if [[ "${WANDB}" == "1" || "${WANDB}" == "true" ]]; then
  WANDB_FLAGS+=(--wandb --wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_FLAGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
fi

python -m bass_ddsp.train \
  --config configs/bass_ddsp_v2_branchbalance_single_note.yaml \
  --name "${SINGLE_RUN}" \
  --root runs \
  --steps "${SINGLE_STEPS}" \
  --batch "${BATCH}" \
  --start-lr "${SINGLE_START_LR}" \
  --stop-lr "${SINGLE_STOP_LR}" \
  --decay-over "${SINGLE_DECAY_OVER}" \
  --device "${DEVICE}" \
  --init-state "${ORIGINAL_SINGLE_STATE}" \
  "${WANDB_FLAGS[@]}" \
  --wandb-name "${SINGLE_RUN}"

python -m bass_ddsp.train \
  --config configs/bass_ddsp_v2_branchbalance_riff.yaml \
  --name "${RIFF_RUN}" \
  --root runs \
  --steps "${RIFF_STEPS}" \
  --batch "${BATCH}" \
  --start-lr "${RIFF_START_LR}" \
  --stop-lr "${RIFF_STOP_LR}" \
  --decay-over "${RIFF_DECAY_OVER}" \
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

python -m bass_ddsp.synthesize_bend_slide \
  --run "runs/${RIFF_RUN}" \
  --out-dir "runs/${RIFF_RUN}/bend_slide_debug" \
  --device "${DEVICE}"

python -m bass_ddsp.compare_three_models \
  --model "Bass-DDSP fixed resume=runs/${RIFF_RUN}" \
  --model "Vanilla DWTS=runs/vanilla_dwts_riff_20260722_040632" \
  --model "Vanilla DDSP=runs/vanilla_ddsp_riff_20260720_090532" \
  --out-dir "${COMPARE_DIR}" \
  --seed 20260723 \
  --num-samples 32 \
  --num-plots 4 \
  --pitch-source labels \
  --device "${DEVICE}"

echo "Done."
echo "Single-note continuation: /workspace/runs/${SINGLE_RUN}"
echo "Riff run:                 /workspace/runs/${RIFF_RUN}"
echo "Branch debug:             /workspace/runs/${RIFF_RUN}/branch_debug_random3_label_pitch"
echo "Control debug:            /workspace/runs/${RIFF_RUN}/control_debug_random3_label_pitch"
echo "Bend/slide debug:         /workspace/runs/${RIFF_RUN}/bend_slide_debug"
echo "Comparison:               /workspace/${COMPARE_DIR}"
