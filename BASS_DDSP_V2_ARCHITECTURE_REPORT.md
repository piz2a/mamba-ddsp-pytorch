# Bass-DDSP v2 Architecture Report

Date: 2026-07-19

This report documents the active first-party Bass-DDSP implementation under `/workspace/bass_ddsp`. Cloned repositories under `/workspace/ddsp_pytorch`, `/workspace/ddsp-guitar`, `/workspace/diff-wave-synth`, and `/workspace/mamba` are reference-only.

## Current Goal

Bass-DDSP v2 is trained as a label-controlled bass synthesizer before scat integration. The current target is:

```text
IDMT controls per frame -> Bass-DDSP v2 -> bass waveform
```

The later scat system should map vocal/scat features into the same controls, but the bass synthesizer must first produce credible single-note attacks, sustains, releases, and generated riffs.

## Active Control Contract

`T` is the number of control frames. It is not the block size.

```text
B = batch size
T = control frames
C = channels
block_size = audio samples per control frame
N = T * block_size output samples
```

The model receives:

| Tensor | Shape | Meaning |
|---|---:|---|
| `pitch` | `(B, T, 1)` | F0 in Hz from IDMT labels by default |
| `loudness` | `(B, T, 1)` | dataset-normalized frame loudness |
| `articulation` | `(B, T)` | observed IDMT articulation class |
| `onset_strength` | `(B, T, 1)` | continuous `[0, 1]` HPSS percussive onset/residual strength |
| `offset` | `(B, T, 1)` | note-end pulse |
| `gate` | `(B, T, 1)` | active note interval mask |
| `note_age` | `(B, T, 1)` | seconds since current onset, causal |
| `periodicity` | `(B, T, 1)` | articulation prior mixed with CREPE confidence |

`note_progress` is not used because it requires knowing the future offset. `string_id`, `fret_id`, and `centroid` are also excluded from the initial input contract.

## Dataset Path

`bass_ddsp/dataset.py` provides:

- `IDMTBassNoteDataset`: isolated note reconstruction stage.
- `IDMTBassRiffDataset`: online generated riff stage.

The riff dataset trims note silence, crops note durations, and joins adjacent notes with equal-power overlap-add crossfades. Labels are assigned to non-overlapping note intervals, with boundaries placed at crossfade midpoints.

F0-dependent expression subclasses are excluded from the articulation path:

```text
BEQ, BES, SLD, SLU, VIF, VIS
```

The active articulation labels are observed combinations such as:

```text
FS_NO, MU_NO, PK_NO, SP_NO, ST_NO, FS_DN, FS_HA
```

## HPSS And Periodicity

`onset_strength(t)` is measured from `librosa.decompose.hpss(D)` percussive magnitude, normalized to `[0, 1]`, and max-combined with the existing event pulse fallback.

`periodicity(t)` is:

```text
periodicity = articulation_prior * ((1 - mix) + mix * CREPE_confidence)
```

Defaults:

```yaml
periodicity_crepe_mix: 0.35
periodicity_mu_prior: 0.25
periodicity_dn_prior: 0.45
periodicity_crepe_device: cpu
```

CREPE is used only for confidence here. Training F0 is still label-derived.

## Model Summary

The active model is `bass_ddsp.model.BassDDSPV2`.

```text
pitch Hz -> log-MIDI normalized pitch feature -> pitch MLP
loudness z-score -> loudness MLP
articulation + onset_strength + offset + gate + note_age + periodicity
  -> ArticulationEncoder -> z(t) -> z MLP
[pitch features, loudness features, z features]
  -> GRU
  -> decoder MLP
  -> transient / sustain / noise branch heads
```

The default recurrent model is now GRU, following the original DDSP baseline. The Mamba wrapper remains available for experiments but is not selected by the v2 configs.

## Sustain Branch

The additive harmonic sustain branch has been removed from the active first-party model path.

The active sustain is DWTS-style learned wavetable synthesis:

```text
hidden -> wavetable attention over K learned tables
pitch Hz -> phase accumulator -> audio-rate table lookup
attention -> audio-rate table mix
hidden -> sustain amplitude
loudness -> deterministic loudness_gain
periodicity -> DDSP-SFX harmonic gate
note_age -> attention mix toward fundamental over time
```

The sustain output is:

```text
sustain = wavetable_mix
          * learned_amplitude
          * gate
          * short_fade_in
          * loudness_gain(t)
          * harmonic_gate(periodicity)
```

The DDSP-SFX-style harmonic indicator is:

```text
H = sigmoid(a * (periodicity - b))
harmonic_gate = h_floor + (1 - h_floor) * H
```

Defaults:

```yaml
n_wavetables: 16
wavetable_length: 512
sustain_fade_seconds: 0.006
harmonic_indicator_a: 10.0
harmonic_indicator_b: 0.7
harmonic_gate_floor: 0.15
```

The short fade-in is the current click-prevention mechanism for sustain note starts. The note-age sustain effect does not multiply sustain energy directly; it shifts wavetable attention toward the initialized fundamental table as a causal high-frequency-first decay proxy.

## Transient Branch

The transient branch is retained as a style + velocity residual, not an exact transient reconstruction target.

The active config uses `transient_type: dct_bank`:

```text
articulation_id -> style-specific DCT coefficient row
DCT coefficient row @ partial IDCT basis -> 300 ms waveform prototype
sample-accurate note_age -> waveform index
hidden -> transient gain
```

The final transient branch is:

```text
transient = transient_raw
            * gate
            * onset_strength
            * exp(-A_transient * note_age)
            * branch_gain
```

This branch is available at every frame, but its audible energy is controlled by continuous `onset_strength` and causal note decay. It is not restricted to one binary onset frame.

## Noise Branch

The noise branch predicts filtered noise:

```text
hidden -> noise-band magnitudes -> impulse response
white noise -> FFT convolution -> noise_raw
```

The final noise branch is:

```text
noise = noise_raw
        * gate
        * onset_strength
        * exp(-A_noise * note_age)
        * branch_gain
```

The noise source is stochastic per forward pass.

## Branch Decay And Loudness

The branch decay rates are positive and trainable by default:

```yaml
sustain_age_mix_rate: 1.0
noise_decay_rate: 10.0
transient_decay_rate: 18.0
learnable_decay_rates: true
```

`loudness(t)` is now deterministic in the sustain path:

```text
loudness_gain = clamp(10 ** (normalized_loudness * loudness_gain_db_per_std / 20),
                      loudness_gain_min,
                      loudness_gain_max)
```

Defaults:

```yaml
loudness_gain_db_per_std: 8.0
loudness_gain_min: 0.15
loudness_gain_max: 4.0
```

This is not exact RMS matching. It is deterministic gain conditioning so the model cannot completely ignore the requested loudness envelope, while still allowing learned synthesis dynamics.

## Loss Function

The current training loss is intentionally simplified:

```text
loss = multi_scale_stft_loss(target_audio, reconstructed_audio)
```

The old log-RMS, onset-weighted, high-pass transient, and direct transient-branch auxiliary terms have been removed from the active optimization path. They are not used by default configs.

This means branch specialization is not guaranteed by the loss. Branch diagnostics are required after training.

## Diagnostics

Training logs scalar metrics to TensorBoard and optionally W&B:

```bash
python -m bass_ddsp.train ... --wandb --wandb-project bass-ddsp-v2
```

Branch export writes:

- `target.wav`
- `signal.wav`
- `sustain.wav`
- `noise.wav`
- `transient.wav`
- `sustain_attention.png`
- `sustain_loudness_gain.npy`
- `sustain_harmonic_gate.npy`
- `branch_metrics.csv`

Transient-style visualization writes:

- `raw_waveform_bank.png`
- `raw_waveform_bank_overlay.png`
- one transient WAV per articulation label

Bend/slide synthesis writes one-note test cases:

- `steady.wav`
- `bend_up.wav`
- `bend_down.wav`
- `slide_up_down.wav`
- matching `*_controls.npz`

## Smoke Result

Current code was smoke-tested with:

```text
runs/bass_ddsp_v2_dwts_100step_20260719_164834
```

It trained 100 single-note steps on `cuda:7` and produced finite outputs.

Observed branch metrics on 3 random debug samples:

| Sample | Signal/Target RMS | Sustain vs Signal | Transient vs Signal | Noise vs Signal |
|---|---:|---:|---:|---:|
| `sample_00_idx_0276` | `0.243` | `99.31%` | `12.09%` | `0.35%` |
| `sample_01_idx_0029` | `0.114` | `99.94%` | `2.57%` | `0.15%` |
| `sample_02_idx_0171` | `0.230` | `99.18%` | `10.08%` | `0.27%` |

Interpretation:

- The DWTS sustain branch is connected and learning; it is not silent.
- The reconstruction remains too quiet after 100 steps.
- Sustain dominates the summed signal, so branch balance is still weak.
- The transient branch produces measurable energy but is still mostly an onset residual.
- Longer training is required before judging audio quality.

## Current Training Scripts

Full staged training:

```bash
cd /workspace && WANDB=1 DEVICE=cuda:7 ./scripts/train_bass_ddsp_v2_full.sh
```

Detached 10-hour tmux launch:

```bash
cd /workspace && SESSION=bass_ddsp_v2_long DEVICE=cuda:7 ./scripts/start_bass_ddsp_v2_long_tmux.sh
```

Diagnostics from a trained run:

```bash
cd /workspace && ./scripts/infer_bass_ddsp_v2_debug.sh runs/<run_name> cuda:7
```

## Current Blocker

W&B is installed, but the container is not logged in:

```text
wandb api_key: null
```

To start remote-trackable long training, run one of:

```bash
wandb login
```

or:

```bash
export WANDB_API_KEY=<your-key>
```

Then run the tmux launcher above.

## Next Direction

The next technical issue is not branch connectivity. It is loudness/energy calibration and branch specialization.

Recommended next experiments:

1. Run a longer single-note stage with W&B enabled and inspect `signal/target RMS`, `sustain_attention`, and transient bank plots.
2. If the model stays quiet, increase deterministic loudness authority by raising `loudness_gain_db_per_std` or the learned sustain amplitude initialization.
3. If sustain keeps absorbing attack energy, reintroduce HPSS-derived branch supervision in a controlled way after the simplified baseline is measured.
4. Only start riff training after isolated notes have believable attack and decay.
