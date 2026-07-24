# Vocal-To-Controls Extractor Plan

## Goal

Build a fast laboratory for inspecting voice/scat recordings before fixing the final real-time voice-to-Bass-DDSP control extractor.

The output should eventually match the Bass-DDSP control interface:

- `f0(t)`
- `loudness(t)`
- `articulation_id(t)`
- `gate(t)`
- `onset_strength(t)`
- `offset(t)`
- `note_age(t)`
- `periodicity(t)`

The first implementation target is analysis and visualization, not final classification.

## Master Frame Grid

- Sample rate: `16 kHz`
- Frame interval: `16 ms`
- Hop size: `256 samples`
- Analysis window: `32 ms`
- Window size: `512 samples`

This matches the Bass-DDSP control rate and makes Silero VAD easy to align because Silero emits one probability every `512` samples, i.e. `32 ms`.

## Current Control Definitions

### `f0(t)`

- Extract with TorchCREPE.
- Use the same 16 ms hop as the control grid.
- Keep raw Hz for inspection.
- Later mapping to bass register can be handled by a deterministic pitch mapper.

### `loudness(t)`

- Use frame RMS in dB.
- Convert to z-score:
  - preferably normalize over active/gated frames,
  - fall back to all frames if VAD finds no active frames.
- This should describe vocal intensity, not note onset.

### `gate(t)`

- Use causal Silero VAD probability.
- Apply hysteresis:
  - open threshold > close threshold,
  - this prevents fast flickering near the decision boundary.
- Interpret as "voice/scat is active enough to synthesize bass."

### `offset(t)`

- Derive from causal VAD falling edges.
- It is a pulse when the hysteresis gate changes from `1` to `0`.
- This is a note/control-state boundary cue, not a spectral offset detector.

### `note_age(t)`

- Causal counter.
- Reset to `0` on each onset.
- Increment by `16 ms` per active frame.
- Hold at `0` while the gate is inactive.
- This is real-time safe because it does not require knowing the future note length.

### `periodicity(t)`

- Use TorchCREPE periodicity/confidence.
- Clip to `[0, 1]`.
- This helps distinguish voiced, pitched syllable regions from noisy consonants.

### `onset_strength(t)`

Not finalized yet. It should represent consonant/attack strength, not volume increase.

Candidate signals:

- HPSS percussive energy.
- Spectral flux, but only as a candidate because it can also spike at offsets.
- High-frequency energy ratio and spectral tilt.
- ContentVec boundary novelty, using drops in cosine similarity between adjacent content embeddings.
- Causal VAD rising edge as a boundary prior.

Practical combination for the lab:

- Visualize all candidates separately.
- Build a provisional candidate:
  - require or strongly weight VAD rising-edge neighborhoods,
  - use HPSS + high-frequency ratio + ContentVec novelty,
  - downweight frames where VAD is closing or already inactive.
- Clip final candidate to `[0, 1]`.

Do not treat `dE/dt` alone as onset strength. A vowel can become louder within the same note, but that is not a new bass pluck.

## `articulation_id(t)` Strategy

The final articulation extractor should not freely change articulation every frame.

Use a causal note-state machine:

1. Detect note start from VAD rising edge plus consonant/onset candidates.
2. Open a short classification window after onset, e.g. first `80-160 ms`.
3. Feed causal features into a small classifier:
   - spectral tilt,
   - high-frequency energy ratio,
   - periodicity,
   - ZCR,
   - onset candidate features,
   - optional ContentVec embeddings or low-dimensional ContentVec projections.
4. Classifier architecture candidate:
   - feature tensor -> causal GRU (`H=128`) -> linear -> softmax over articulation classes.
5. Latch the predicted articulation once confidence is sufficient.
6. Hold the latched `articulation_id(t)` until `offset(t)`.
7. Reset to unknown/default when gate is inactive.

This solves the core problem: the first few consonant frames determine the pluck/expression, while later vowel frames sustain the same bass note instead of causing unstable articulation changes.

## Laboratory Notebook

Notebook:

- `/workspace/learn/scat_feature_extraction_colab.ipynb`

It should support quick local testing:

- set `AUDIO_PATH` manually, or
- place audio files in `/workspace/learn/voice_inputs/` and auto-load the newest one.

The lab should visualize:

- waveform,
- spectrogram,
- TorchCREPE `f0(t)`,
- TorchCREPE periodicity,
- RMS z-score loudness,
- causal VAD probability,
- gate,
- onset pulse,
- offset pulse,
- note age,
- HPSS onset candidate,
- spectral flux candidate,
- high-frequency ratio / spectral tilt,
- optional ContentVec novelty.

## Current Assumptions

- The final system is causal or near-causal.
- `note_age(t)` is allowed; `note_progress(t)` is not allowed.
- `articulation_id(t)` should be note-latched, not continuously free-running.
- `onset_strength(t)` and `articulation_id(t)` are still research targets and must be evaluated on actual voice recordings.
