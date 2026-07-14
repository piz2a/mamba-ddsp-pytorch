# Bass-DDSP v2 Architecture Report

Date: 2026-07-14

This report documents the current implemented Bass-DDSP v2 architecture in this repository. It focuses on implementation details that are easy to miss from the high-level plan: tensor contracts, hidden dimensions, feature scaling, recurrent wiring, DSP branch internals, dataset-control assumptions, and current limitations.

Relevant implementation files:

- `ddsp/model.py`
- `ddsp/core.py`
- `idmt_bass.py`
- `train.py`
- `config_idmt_bass_v2_single_note.yaml`
- `config_idmt_bass_v2_riff.yaml`

## Executive Summary

The current model is not a classifier and not a neural vocoder. It is a conditional differentiable synthesizer:

```text
controls per frame
  -> ArticulationEncoder
  -> separate F0 / loudness / z MLPs
  -> Mamba or GRU temporal model
  -> decoder MLP
  -> harmonic parameters + noise-filter parameters + transient gain
  -> harmonic synth + filtered-noise synth + learned transient bank
  -> summed audio
```

The current Bass-DDSP v2 checkpoint has three connected synthesis branches:

- `sustain`: harmonic additive DDSP branch.
- `noise`: filtered noise branch.
- `transient`: learned articulation-conditioned waveform-bank branch.

However, the loss currently supervises only the final summed audio. There is no explicit target for each branch. As observed in branch diagnostics, the model currently routes almost all energy through `sustain`, while `transient` and `noise` remain small.

## Current Config Snapshot

The main riff config is `config_idmt_bass_v2_riff.yaml`.

| Item | Value |
|---|---:|
| Sampling rate | `16000 Hz` |
| Riff signal length | `32768 samples` |
| Riff duration | `2.048 s` |
| Block size | `256 samples` |
| Frame rate | `62.5 frames/s` |
| Riff frame count | `128` |
| Single-note signal length | `32000 samples` |
| Single-note duration | `2.0 s` |
| Single-note frame count | `125` |
| Hidden size | `256` |
| z size | `64` |
| Articulation embedding size | `24` |
| Harmonic count | `100` |
| Noise bands | `65` |
| Recurrent type | `mamba` |
| Mamba `d_state` | `16` |
| Mamba `d_conv` | `4` |
| Mamba expand | `2` |
| Transient duration | `0.20 s` |
| Reverb | disabled |

The current active expression set is intentionally restricted:

```yaml
include_expression_styles: ["NO", "DN", "HA"]
```

This excludes F0-dependent subclasses such as bend, slide, and vibrato subclasses from the expression control path.

## Input Tensor Contract

For `bass_ddsp_v2`, the training batch contains 9 tensors:

```text
audio, pitch, loudness, articulation, onset, offset, gate, note_age, note_progress
```

After the training loop reshapes them:

| Tensor | Shape | Meaning |
|---|---|---|
| `audio` | `(B, N)` | Target waveform |
| `pitch` | `(B, T, 1)` | F0 in raw Hz before model-internal scaling |
| `loudness` | `(B, T, 1)` | Normalized loudness feature |
| `articulation` | `(B, T)` | Integer articulation ID |
| `onset` | `(B, T, 1)` | Soft onset pulse |
| `offset` | `(B, T, 1)` | Soft offset pulse |
| `gate` | `(B, T, 1)` | Active note mask |
| `note_age` | `(B, T, 1)` | Seconds since current note start, clipped |
| `note_progress` | `(B, T, 1)` | Fractional progress through current interval |

For riff training:

```text
N = 32768
T = 32768 / 256 = 128
```

For single-note training:

```text
N = 32000
T = 32000 / 256 = 125
```

## Articulation Label Strategy

The model does not use independent `P(t)` and `E(t)` embeddings in the v2 path. It uses observed articulation combinations:

```text
articulation = pluck + "_" + expression
```

For the current restricted IDMT subset, the observed classes are:

```text
FS_NO
MU_NO
PK_NO
SP_NO
ST_NO
FS_DN
FS_HA
```

This is implemented as one categorical input, not two independent categorical inputs.

Important detail: `articulation_id` is not predicted by the model. It is an input condition. There is no classification loss or cross-entropy loss in the current Bass-DDSP decoder training.

## Articulation Encoder

The v2 model uses `ArticulationEncoder`, not the older `StyleEncoder`.

Input:

```text
articulation_id(t) + control vector(t)
```

The control vector has 5 channels when `use_note_shape_controls: true`:

```text
[onset, offset, gate, note_age, note_progress]
```

The encoder path is:

```text
articulation_id
  -> embedding, 24 dims

[embedding, onset, offset, gate, note_age, note_progress]
  -> MLP, hidden 256
  -> Linear to z_size 64
  -> LayerNorm
  -> LeakyReLU
  -> z(t)
```

So the actual per-frame latent vector `z(t)` is not learned as a free latent variable. It is deterministically produced from known labels and event controls.

## F0 Scaling Detail

The model uses pitch in two different ways:

1. Raw Hz is used by the oscillator.
2. Log-MIDI-normalized pitch is used by the neural network.

The neural pitch transform is:

```text
midi = 69 + 12 * log2(f0_hz / 440)
pitch_control = (midi - midi_min) / (midi_max - midi_min)
pitch_control = clamp(pitch_control, 0, 1)
```

Current range:

```yaml
f0_min_hz: 30.0
f0_max_hz: 330.0
```

This means:

- The MLP/Mamba sees normalized pitch.
- The additive harmonic synthesizer still receives Hz.

This separation matters. If the oscillator received normalized pitch, synthesis would be wrong. If the neural network received raw Hz, the pitch input scale would be poorly conditioned.

## Loudness Scaling Detail

Loudness is extracted with `extract_loudness()` in `ddsp/core.py`.

The extraction is:

```text
STFT magnitude
-> log magnitude
-> A-weighting by frequency
-> mean over frequency bins
-> one scalar per frame
```

During training, loudness is normalized:

```text
loudness = (loudness - dataset_mean) / dataset_std
```

The normalization statistics are computed from the training dataset at the start of training. The model receives this normalized loudness as a conditioning signal, but the current synthesizer does not deterministically force the final waveform RMS to match loudness. Loudness only influences the neural decoder through learned weights.

This is why the model can still produce a reconstruction with weaker or flatter loudness than the target.

## Separate Input MLPs and Relative Capacity

The model processes pitch, loudness, and z through separate MLPs before concatenating them. Current dimensions:

| Input | Input dim | MLP output dim |
|---|---:|---:|
| pitch control | `1` | `64` |
| loudness | `1` | `64` |
| z | `64` | `256` |

Total recurrent input size:

```text
64 + 64 + 256 = 384
```

This means the conditioning path allocates more representation capacity to articulation/event-derived `z` than to pitch or loudness individually. This does not force dependence, but it does give the model more channel capacity for articulation/style information.

Each branch MLP is a 3-layer MLP using:

```text
Linear
LayerNorm
LeakyReLU
```

repeated three times.

## Temporal Model

The current config uses Mamba:

```yaml
recurrent_type: mamba
mamba_d_state: 16
mamba_d_conv: 4
mamba_expand: 2
```

The wrapper does:

```text
384-dim condition input
  -> Linear projection to 256
  -> Mamba(d_model=256)
  -> LayerNorm
```

There is also a GRU alternative in the code. The GRU path uses:

```text
GRU(input_size=384, hidden_size=256, batch_first=True)
```

## Decoder Head

After the recurrent model, the decoder does not use only the recurrent state. It concatenates skip features:

```text
[recurrent_output, pitch_control, loudness, z]
```

Current dimension:

```text
256 + 1 + 1 + 64 = 322
```

This 322-dim tensor goes through:

```text
out_mlp: MLP(322 -> 256), 3 layers
```

The output hidden representation then feeds the synthesis parameter projections:

```text
harmonic projection: 256 -> n_harmonic + 1 = 101
noise projection:    256 -> n_bands = 65
transient gain:      256 -> 1
```

## Harmonic Sustain Branch

The sustain branch is an additive harmonic synthesizer.

The harmonic projection produces:

```text
total_amp: 1 channel
harmonic_distribution: 100 channels
```

Processing:

```text
hidden
  -> Linear(256, 101)
  -> scale_function
  -> split total amplitude and harmonic amplitudes
  -> remove harmonics above Nyquist
  -> normalize harmonic amplitudes to sum to 1
  -> multiply by total amplitude
  -> upsample from frame rate to audio rate
  -> synthesize sinusoidal harmonics using raw Hz F0
  -> multiply by gate
```

The phase equation in the implementation is:

```text
omega[n] = cumulative_sum(2 * pi * f0_hz[n] / sample_rate)
```

This means raw `f0_hz` is internally converted to normalized cycles per sample during oscillator phase accumulation.

Important detail: the harmonic branch is gated after synthesis. If `gate` is 0, harmonic output is forced to 0.

## Filtered Noise Branch

The noise branch predicts a time-varying filter magnitude, then filters white noise.

Processing:

```text
hidden
  -> Linear(256, 65)
  -> subtract 5
  -> scale_function
  -> convert magnitude response to impulse response
  -> generate uniform random white noise per block
  -> FFT convolution
  -> reshape to audio
  -> multiply by gate
```

The `-5` before `scale_function` suppresses initial noise energy. This is inherited from the original DDSP-style code path. It is one reason the noise branch starts weak and may remain underused unless losses force it to matter.

Important detail: this branch is stochastic. Running the same checkpoint twice can produce slightly different noise audio because fresh random noise is generated on every forward pass.

## Transient Branch

The transient branch is not a filtered-noise burst. It is a learned waveform-bank lookup.

Current shape:

```text
transient_bank: (n_articulation, transient_samples)
```

For the current config:

```text
n_articulation = 7
transient_seconds = 0.20
sample_rate = 16000
transient_samples = 3200
transient_bank shape = (7, 3200)
```

Processing:

```text
hidden
  -> Linear(256, 1)
  -> scale_function
  -> upsample to audio-rate transient gain

articulation_id(t)
  -> repeat to audio rate

note_age(t)
  -> repeat to audio rate
  -> sample_index = note_age_seconds * sample_rate
  -> clamp to transient bank length

transient_bank[articulation_id, sample_index]
  -> multiply by quadratic decay envelope
  -> multiply by gate
  -> multiply by learned transient gain
```

The envelope is:

```text
envelope = clamp(1 - note_age / transient_seconds, 0, 1)^2
```

Important details:

- The transient waveform is tied to articulation class.
- It is addressed by `note_age`, not directly by the `onset` pulse.
- The `onset` pulse only affects the transient indirectly through `z(t)` and the recurrent hidden state.
- If `note_age` is wrong or does not reset cleanly at note boundaries, this branch will not fire correctly.
- Because the branch has no direct loss target, the model can ignore it.

## Final Sum and Reverb

For Bass-DDSP v2:

```text
signal = sustain + noise + transient
```

If `use_reverb` is true, this summed signal is passed through a learnable convolutional reverb module. In the current configs:

```yaml
use_reverb: false
```

So current diagnostics satisfy:

```text
signal.wav = sustain.wav + noise.wav + transient.wav
```

The reverb module still exists in the model class. It uses a learned noise impulse response with a learned exponential decay and wet parameter. It is not room-specific convolution reverb in the current training configs.

## Branch Output Logging

The model stores the last generated branches:

```python
model.last_branch_outputs = {
    "transient": transient,
    "sustain": harmonic,
    "noise": noise,
    "signal": signal,
}
```

This is what `export_branch_debug.py` uses to write branch-separated WAV files.

Branch diagnostics from the 500-step riff checkpoint showed:

| Sample | Sustain RMS vs Signal | Transient RMS vs Signal | Noise RMS vs Signal |
|---|---:|---:|---:|
| `sample_00_idx_0267` | `100.00%` | `0.15%` | `0.24%` |
| `sample_01_idx_0190` | `99.85%` | `3.48%` | `3.46%` |
| `sample_02_idx_0870` | `99.96%` | `1.59%` | `0.59%` |

Interpretation: all three branches are connected, but the trained checkpoint currently relies almost entirely on the harmonic sustain branch.

## Dataset Processing Details That Affect Architecture

The model behavior is strongly shaped by the online dataset generator.

### Silence Trimming

Each note is trimmed using frame RMS. The threshold is based on:

- peak-relative threshold,
- edge-noise percentile,
- noise margin,
- separate onset threshold.

Current riff config:

```yaml
trim_top_db: 35.0
trim_onset_top_db: 25.0
trim_noise_percentile: 20.0
trim_noise_margin_db: 12.0
trim_frame_size: 512
trim_hop_size: 128
trim_pad_seconds: 0.012
```

### Edge Fades

After trimming, each loaded note gets a short edge fade:

```yaml
edge_fade_seconds: 0.004
```

If a note is cropped shorter than its trimmed source, a release fade is applied:

```yaml
release_fade_seconds: 0.035
```

### Random Note Length

For riff training, note segment length is sampled uniformly in samples between:

```yaml
min_note_seconds: 0.28
max_note_seconds: 1.10
```

Important detail: when the source note is longer than the sampled duration, the current code takes the beginning of the note:

```text
segment = audio[:target]
```

It does not randomly crop from the middle. This preserves attacks but limits sustain variation.

### Riff Concatenation

Riff generation repeats:

```text
choose random note
trim/load/crop note
append with equal-power overlap-add crossfade
stop after signal_length is filled
```

The crossfade is equal-power:

```text
fade_out = cos(theta)
fade_in  = sin(theta)
```

Current riff crossfade range:

```yaml
min_crossfade_seconds: 0.030
max_crossfade_seconds: 0.075
```

### Label Boundary During Crossfade

For a new note appended with crossfade, the label boundary is placed at the crossfade midpoint:

```text
label_start = transition_start + crossfade // 2
```

This means during the first half of the crossfade, the previous note label remains active. During the second half, the new note label becomes active.

This is why you can sometimes hear residual energy from one note while the control label has already switched or is about to switch. It is an intentional compromise, not a physical string interaction model.

### Onset and Offset Pulses

Onset and offset are not hard one-frame impulses. They are triangular pulses:

```text
pulse = max(0, 1 - distance / event_width)
```

Current width:

```yaml
event_width_seconds: 0.032
```

### Gate, Note Age, Note Progress

For each non-overlapping labeled interval:

```text
gate = 1 inside interval, 0 outside
note_age = seconds since interval start, clipped to note_age_clip_seconds
note_progress = (current_frame - start) / interval_duration
```

Current clip:

```yaml
note_age_clip_seconds: 1.0
```

The transient branch depends directly on `note_age`. The harmonic and noise branches are directly multiplied by `gate`.

### Single-Note Padding Detail

In `IDMTBassNoteDataset`, a trimmed note is placed at the beginning of a fixed-length buffer and the rest is zero-padded.

Important detail: `label_pitch` and `articulation` are filled across all frames, even after the note ends. The `gate` becomes 0 after the active note interval. This means the model must learn to respect `gate` for silence.

The short single-note training run showed that the model can learn this gating behavior reasonably quickly.

## Randomness Detail

The dataset uses a per-index random generator.

If `seed` is set:

```text
rng = Random(seed + idx * 1000003)
```

This makes each dataset index deterministic.

If `seed` is not set:

```text
seed = torch random int
rng = Random(seed + idx * 1000003)
```

This makes online examples change across epochs and dataloader visits.

The current configs leave `seed:` empty, so training examples are online-randomized.

## Training Loss Is Output Reconstruction Only

The current loss is:

```text
loss =
  multiscale_spectral_loss
  + rms_loss_weight * log_frame_rms_loss
  + onset_loss_weight * onset_region_spectral_loss
  + transient_loss_weight * highpass_onset_loss
```

There is no:

- articulation classification loss,
- onset classification loss,
- pitch prediction loss,
- loudness prediction loss,
- branch-specific target loss,
- adversarial loss,
- perceptual embedding loss.

All labels are conditions. The model is trained only to reconstruct audio.

## Real-Time Path Exists, But Is Not Fully Validated

The model has `realtime_forward()`.

Relevant state buffers:

- GRU hidden cache for the GRU path.
- Mamba convolution and SSM state cache for the Mamba path.
- Harmonic phase buffer for phase continuity.

The real-time path is architecturally plausible, but it has not been extensively validated in the current experiments. Specific caveats:

- The noise branch is stochastic per forward call.
- The transient branch depends on externally correct `note_age`.
- The riff generator/training pipeline is offline and does not prove streaming behavior.
- Mamba `step()` path exists but has not been stress-tested for long real-time inference.

## Details From the Original Plan That Are Not Implemented Yet

These were discussed conceptually, but they are not implemented in the current Bass-DDSP v2 model:

| Planned idea | Current status |
|---|---|
| `string_id(t)` input | Not implemented |
| `fret_id` or fret position input | Not implemented |
| Brightness/centroid input | Not implemented |
| Periodicity/tonalness input | Not implemented |
| Per-string EQ/body filter | Not implemented |
| Differentiable wavetable sustain | Not implemented |
| Explicit branch target decomposition | Not implemented |
| Scat/vocal encoder | Not implemented |
| Scat-to-articulation classifier | Not implemented |
| Manifold projection | Not implemented |

String and fret are parsed from IDMT filenames in the dataset metadata, but they are not currently emitted as model input tensors.

## Important Current Weakness

The architecture has three branches, but the training objective does not force specialization.

The model can reduce the final audio loss by using the harmonic branch for most content. This is currently happening. Three implementation details make this especially likely:

1. The harmonic branch is expressive enough to explain much of the periodic bass signal.
2. The noise branch is initialized/suppressed with `proj_noise - 5`.
3. The transient branch has no direct target and only receives gradient through the final summed audio.

Therefore, the current architecture is a useful baseline but not yet a reliable transient-assisted bass synthesizer.

## Recommended Next Architecture Changes

The next changes should be architectural or loss-level, not just longer training:

1. Add branch-specific supervision or constraints.
   - Example: force high-pass onset energy into transient/noise branches.
   - Example: penalize sustain energy inside the first few milliseconds of attacks.

2. Make transient branch more explicitly onset-triggered.
   - Current transient indexing uses `note_age`.
   - It should also have direct onset-triggered amplitude or envelope control.

3. Add deterministic loudness gain after synthesis.
   - The model currently receives loudness but is not forced to match it.
   - A learned residual gain plus deterministic RMS envelope matching may improve dynamics.

4. Add string conditioning before increasing model size.
   - IDMT filenames contain string and fret.
   - `string_id` is likely more useful than `fret_id` because `f0` already captures pitch, but string affects timbre.

5. Consider pretraining single-note reconstruction longer before riff training.
   - Single-note diagnostics are cleaner.
   - Riff training should come after attack, decay, and branch balance are credible.

## Minimal Current Architecture Diagram

```text
Per-frame controls:
  pitch_hz, loudness, articulation_id, onset, offset, gate, note_age, note_progress

Pitch path:
  pitch_hz
    -> log-MIDI normalize to [0, 1]
    -> pitch MLP, 64 dims

Loudness path:
  loudness
    -> dataset z-score normalize
    -> loudness MLP, 64 dims

Articulation path:
  articulation_id -> embedding, 24 dims
  [embedding, onset, offset, gate, note_age, note_progress]
    -> ArticulationEncoder
    -> z(t), 64 dims
    -> z MLP, 256 dims

Temporal path:
  [pitch_mlp, loudness_mlp, z_mlp], 384 dims
    -> Mamba or GRU
    -> recurrent output, 256 dims

Decoder:
  [recurrent output, pitch_control, loudness, z], 322 dims
    -> output MLP, 256 dims

Synthesis:
  hidden -> harmonic params -> additive harmonic synth -> sustain
  hidden -> noise bands -> filtered white noise -> noise
  hidden + articulation + note_age -> transient bank lookup -> transient

Final:
  signal = sustain + noise + transient
```
