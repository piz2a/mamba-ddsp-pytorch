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
juce_root="${JUCE_ROOT:-$HOME/dev/JUCE}"
if [[ ! -f "$juce_root/CMakeLists.txt" ]]; then
  echo "Set JUCE_ROOT to the JUCE source tree (default: $HOME/dev/JUCE)." >&2
  exit 1
fi

npm --prefix "$root/ui" ci
npm --prefix "$root/ui" run build

for config in Debug Release; do
  build_dir="$root/build-mac-$config"
  PKG_CONFIG_LIBDIR=/usr/lib/pkgconfig cmake --fresh -S "$root" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE="$config" \
    -DJUCE_ROOT="$juce_root" \
    -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
  cmake --build "$build_dir" -j"$(sysctl -n hw.logicalcpu)"
  echo "Built $config artifacts under $build_dir/ScatToBass_artefacts/$config"
done
