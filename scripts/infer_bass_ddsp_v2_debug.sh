#!/usr/bin/env bash
set -euo pipefail

cd /workspace

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 runs/<run-name> [device]"
  exit 2
fi

RUN="$1"
DEVICE="${2:-${DEVICE:-cuda:7}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${RUN}/debug_${STAMP}}"

python -m bass_ddsp.export_branch_debug \
  --run "${RUN}" \
  --out-dir "${OUT_ROOT}/branch_random3_label_pitch" \
  --seed "${SEED:-98765}" \
  --num-samples "${NUM_SAMPLES:-3}" \
  --pitch-source labels \
  --device "${DEVICE}"

python -m bass_ddsp.visualize_transient_styles \
  --run "${RUN}" \
  --out-dir "${OUT_ROOT}/transient_styles" \
  --device "${DEVICE}"

python -m bass_ddsp.synthesize_bend_slide \
  --run "${RUN}" \
  --out-dir "${OUT_ROOT}/bend_slide" \
  --device "${DEVICE}"

echo "Debug outputs: /workspace/${OUT_ROOT}"
