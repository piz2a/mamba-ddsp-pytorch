# macOS Build Report

Date: 2026-07-30  
Host: Apple Silicon (`arm64`), macOS 26.5.2, AppleClang 17.0.0, Xcode 26.0.1  
JUCE: `/Users/ziho/dev/JUCE` (JUCE 8.0.13)  
ONNX Runtime: macOS arm64 1.24.4

## Outcome

Debug and Release builds completed for all three JUCE formats:

| Configuration | VST3 | Audio Unit | Standalone |
|---|---|---|---|
| Debug | `build-mac-Debug/ScatToBass_artefacts/Debug/VST3/Scat to Bass.vst3` | `build-mac-Debug/ScatToBass_artefacts/Debug/AU/Scat to Bass.component` | `build-mac-Debug/ScatToBass_artefacts/Debug/Standalone/Scat to Bass.app` |
| Release | `build-mac-Release/ScatToBass_artefacts/Release/VST3/Scat to Bass.vst3` | `build-mac-Release/ScatToBass_artefacts/Release/AU/Scat to Bass.component` | `build-mac-Release/ScatToBass_artefacts/Release/Standalone/Scat to Bass.app` |

Use the Release build for DAW testing. Logic Pro loads the AU component; Logic
does not host VST3 plugins. The VST3 build is for VST3-compatible hosts.

## Inputs restored from the Ubuntu workspace

The clone intentionally ignores generated models and the aubio source tree.
They were copied from the original workspace at
`snu124:/home/ahnjiho/mamba-ddsp-pytorch`:

- `Models/bass_ddsp_controller.onnx`
- `Models/torchcrepe_tiny.onnx`
- `Models/onset_envelope.onnx`
- `Models/sustain_wavetables.f32`
- `Models/transient_bank.f32`
- `../aubio`

All five model/table SHA-256 values match `Models/manifest.json`.

The React editor was regenerated with `npm ci` and `npm run build`.

## Source and build changes

- `CMakeLists.txt`
  - Gives VST3, AU, and standalone bundles a bundle-relative
    `@loader_path/../Frameworks` runtime search path.
  - Copies the exact versioned ONNX Runtime dylib and aubio dylib into every
    bundle.
  - Ad-hoc signs each nested dylib and then the completed bundle for local use.
- `tools/build_macos.sh`
  - Defaults JUCE to `~/dev/JUCE`, with a `JUCE_ROOT` override.
  - Builds both Debug and Release in separate build directories.
  - isolates aubio from optional Homebrew codec libraries, keeping the plugin
    self-contained.
  - starts with a fresh CMake configuration on every invocation.
- `README.md`
  - Documents the new two-configuration build and bundled dependencies.
- `.gitignore`
  - Excludes the generated `build-mac-*` trees.

Homebrew `pkgconf` 3.0.4 was installed because aubio's CMake setup requires
`pkg-config`. During that install Homebrew's automatic cleanup removed an
unneeded Homebrew `openjdk` 24.0.2 keg. The system still has Zulu JDK 17 at
`/Library/Java/JavaVirtualMachines/zulu-17.jdk`.

## Verification

Both Debug and Release passed:

- Embedded model smoke test: all CREPE, Bass-DDSP controller, and onset-envelope
  outputs finite.
- Threaded engine smoke test: onset and active note observed; finite synthesized
  audio with RMS `0.238754`.
- Architecture inspection: plugin executables are native `arm64`.
- Runtime linkage inspection: only bundled `@rpath/libaubio.5.4.8.dylib` and
  `@rpath/libonnxruntime.1.24.4.dylib`; no Homebrew or source-tree paths.
- Strict recursive code-sign verification for VST3, AU, and standalone bundles.
- VST3 and AU property-list validation.

The Release AU metadata is:

- Bundle identifier: `com.jihoaudio.scattobass`
- Type: `aufx`
- Subtype: `Stbs`
- Manufacturer: `Jiho`

The binaries target macOS 26.0 or later because they were built against the
current macOS 26 SDK/toolchain. They are intended for this Apple Silicon Mac.

## Install and test

Copy the Release AU for Logic Pro:

```bash
mkdir -p "$HOME/Library/Audio/Plug-Ins/Components"
ditto \
  "build-mac-Release/ScatToBass_artefacts/Release/AU/Scat to Bass.component" \
  "$HOME/Library/Audio/Plug-Ins/Components/Scat to Bass.component"
killall AudioComponentRegistrar 2>/dev/null || true
auval -v aufx Stbs Jiho
```

Then restart Logic Pro and insert **Jihoaudio → Scat to Bass** as an Audio FX.

For a VST3 host:

```bash
mkdir -p "$HOME/Library/Audio/Plug-Ins/VST3"
ditto \
  "build-mac-Release/ScatToBass_artefacts/Release/VST3/Scat to Bass.vst3" \
  "$HOME/Library/Audio/Plug-Ins/VST3/Scat to Bass.vst3"
```

These are ad-hoc signed development builds. Sharing them with other machines
requires a suitable deployment target, Developer ID signing, and notarization.

## UI and noise-gate follow-up

The following improvements were added and both plugin configurations were
rebuilt on 2026-07-30:

- All WebView text is unselectable through global `user-select: none` and
  `-webkit-user-select: none` CSS.
- Added an automatable **Noise Gate** knob with a range of `-80.0` to `0.0`
  dBFS, 0.5 dB host steps, and a default of `-45.0` dBFS.
- The parameter is stored in JUCE's `AudioProcessorValueTreeState`, so DAW
  automation and saved sessions restore it.
- Vertical drag, mouse wheel, arrow keys, and double-click reset are supported.
- Removed the old first-512-ms noise calibration. The gate now responds
  immediately to the knob's absolute RMS threshold, including after playhead
  jumps.
- aubio's internal silence filter is kept permissive at `-90 dB`; the explicit
  RMS gate is the single user-controlled silence threshold.
- The engine smoke input now starts at 400 ms, inside the former calibration
  window, and passes in Debug and Release. This specifically guards against
  learning an early vocal as room noise.

The React production build and TypeScript check passed. The in-app preview
browser was unavailable during this follow-up, so final visual interaction
should be confirmed in the rebuilt standalone app or Logic Pro.

## F0 octave correction and control

The VST and offline demo previously applied a fixed `-12` semitone/`0.5x`
mapping to detected vocal F0. That fixed octave-down mapping was removed, so
the new octave `0` default sends detected F0 directly to the Bass-DDSP decoder.

The VST now exposes an automatable, session-persistent **Octave** knob with
integer values from `-2` through `+2`. The Debug and Release plugins were
rebuilt and re-signed. The revised native smoke test measures a mapped F0 of
`109.174 Hz` for its 110 Hz test input, rather than the former ~55 Hz result,
and produces finite output with RMS `0.238754`.

The demo pipeline likewise defaults to zero semitone shift and accepts an
octave shift from `-2` through `+2`. Its notebook exposes an `IntSlider` next
to the articulation control and reads the selected octave at each synthesis
run. Notebook JSON, Python syntax, and the exact `55/110/220 Hz` mapping for
octaves `-1/0/+1` were validated. Full demo synthesis was not run on this Mac
because its Python environment does not currently contain `soundfile` or the
training-time model dependencies.
