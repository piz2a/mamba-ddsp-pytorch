#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script must run on macOS." >&2
  exit 1
fi
if [[ -z "${ONNXRUNTIME_ROOT:-}" ]]; then
  echo "Set ONNXRUNTIME_ROOT to an ONNX Runtime macOS C/C++ package." >&2
  exit 1
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
npm --prefix "$root/ui" ci
npm --prefix "$root/ui" run build
cmake -S "$root" -B "$root/build-mac" \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
cmake --build "$root/build-mac" --config Release -j"$(sysctl -n hw.logicalcpu)"

echo "Built artifacts under $root/build-mac/ScatToBass_artefacts/Release"
