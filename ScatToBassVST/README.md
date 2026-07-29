# Scat to Bass VST

`ScatToBassVST` is the first-party native prototype that connects the causal
voice-control frontend to the cached-DCT Bass-DDSP model. Audio and inference
run in C++; React is used only for the JUCE WebView editor.

## Signal Flow

```text
host audio
  -> mono + host-rate/16 kHz resampling
  -> user-adjustable RMS noise gate (-80 to 0 dBFS)
  -> aubio Complex onset detector (512 window, 128 hop)
  -> tiny CREPE ONNX (1024 frame, 256 hop)
  -> causal monophonic note state
  -> style selection / slap pitch rule
  -> learned onset-envelope ONNX
  -> recurrent Bass-DDSP controller ONNX
  -> cached wavetable sustain + filtered noise + cached-DCT transient
  -> host-rate resampling
```

The audio thread only moves blocks through lock-free FIFOs. ONNX Runtime,
aubio, resampling, and synthesis execute on a high-priority worker thread.

## One-Knob Styles

| Knob label | Bass-DDSP articulation |
|---|---|
| Finger | `FS_NO` |
| Muted | `MU_NO` |
| Pick | `PK_NO` |
| Slap Auto | `ST_NO` below MIDI 40; `SP_NO` at or above MIDI 40 |
| Slap Pop | `SP_NO` |
| Slap Thumb | `ST_NO` |
| Dead Note | `FS_DN` |

The model itself has six articulation classes. `Slap Auto` is the seventh
user-facing mode and deterministically dispatches to one of the two slap
classes. The React editor plots F0, periodicity, note gate, onset/offset
events, articulation, note age, and native inference time. The detected F0 is
sent directly to Bass-DDSP at the default octave setting; the automatable
Octave knob provides integer shifts from `-2` to `+2`.

## Model Provenance

Run `python tools/export_models.py` to regenerate `Models/`. The manifest pins
the Bass-DDSP synthesis tables to the successful 2026-07-24 checkpoint:

```text
runs/bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513/state.pth
SHA-256 9efada41772b2d81277773cc771c2361c0744e50f27d2643cfb00d97e4bf2832
```

The exporter also validates PyTorch/ONNX numerical agreement for tiny CREPE
and the onset-envelope model.

## Linux Build

```bash
cd /workspace/ScatToBassVST
npm --prefix ui ci
npm --prefix ui run build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/ScatToBassModelSmoke
./build/ScatToBassEngineSmoke
```

Artifacts:

```text
build/ScatToBass_artefacts/Release/VST3/Scat to Bass.vst3
build/ScatToBass_artefacts/Release/Standalone/Scat to Bass
```

## macOS Build

A macOS binary cannot be cross-built or code-signed from this Linux
container. On the Mac, install Xcode command-line tools, CMake, and Node, then
download an ONNX Runtime 1.18.1 C/C++ package matching the Mac architecture.

```bash
cd /path/to/workspace/ScatToBassVST
ONNXRUNTIME_ROOT=/path/to/onnxruntime-osx-arm64 \
  ./tools/build_macos.sh
```

`JUCE_ROOT` defaults to `~/dev/JUCE` and can be overridden in the environment.
The script builds Debug and Release VST3, AU, and standalone targets in
`build-mac-Debug` and `build-mac-Release`. Each bundle contains aubio and ONNX
Runtime under `Contents/Frameworks` and is ad-hoc signed for local testing.
Distribution still requires Developer ID signing, notarization, and DAW
validation.

## Verification Status

- React/Vite production build: pass.
- Linux VST3 and standalone Release build: pass.
- Three embedded ONNX graphs: finite-output pass.
- Threaded onset-to-synth engine: onset, gate, and finite-audio pass.
- Synthetic engine output RMS in the smoke test: `0.217451`.
- macOS build and audio-device testing: pending on macOS hardware.

The native synthesis code follows the trained branch parameters and cached
tables, but it remains a production-oriented C++ port that requires listening
tests against the Python demo before a release claim.

## Licensing

aubio is GPL licensed. Statically or dynamically distributing a plugin linked
against aubio has GPL compatibility implications for the combined work.
Resolve that before commercial distribution, or replace/license the onset
detector appropriately. JUCE and ONNX Runtime have their own distribution
requirements.

See [docs/LATENCY_REPORT.md](docs/LATENCY_REPORT.md) for the measured onset
and inference latency.
