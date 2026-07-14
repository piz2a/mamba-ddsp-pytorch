# Scat-to-Bass: DDSP-Based, Transient-Assisted Bass Synthesizer from Scat Recordings
is a sequantial model (Scat-to-bass classifier + Bass-DDSP).

Currently working on Bass-DDSP Part.

## Summary

Bass-DDSP is a bass-specific synthesis path for IDMT-SMT-BASS. The goal is
to solve the smooth, violin-like behavior of the baseline DDSP model before
connecting the later scat-to-bass classifier.

The model is trained first on isolated labeled notes, then on generated riffs.
It uses observed IDMT articulation classes such as `FS_NO`, `PK_NO`, `SP_NO`,
and `FS_HA` instead of assuming plucking style and expression style are fully independent.

## Architecture

Frame-level controls:

- `gate(t)`: $\{0,1\}^{(B,T,1)} := $ `(f0(t) > 0).float()`
* Category A
  + When training Bass-DDSP: Extracted from bass audio
  + When training scat-to-bass classifier: Extracted from vocal audio
  - `f0(t)`: $[0,1]^{(B,T,1)}$: converted log-scale from $[28Hz, 330Hz]$ (comprehensive) (original IDMT dataset: $[35Hz, 240Hz]$)
  - deterministic `loudness(t)`: z-score normalized $\mathbb{R}^{(B,T,1)}$.'

* Category B:
  + When training Bass-DDSP: Extracted from bass dataset
  + When training scat-to-bass classifier: Predicted with vocal encoder

  - observed `articulation_id(t)` (8 categories: `FS_NO`, `MU_NO`, `PK_NO`, `SP_NO`, `ST_NO`, `FS_NO`, `FS_HA`, `FS_DN`. BE, SL, VI are excluded since they are assumed to be fully controlled by `f0(t)`.) Not used directly. Further embedded to `articulation_emb(t)`: $\mathbb{R}^{(B,T,d_{art})}$ where $d_{art}=$
  - `string_id(t)`: 4 categories: $\{1, 2, 3, 4\}^{(B,T)}$. Not used directly. Further embedded to `string_emb(t)`: $\mathbb{R}^{(B,T,d_{str})}$ where $d_{str}=$
  - `onset(t)`, `offset(t)`: $\{0,1\}^{(B,T,1)}$
  - `note_age(t)`: $[0, t_{max}]^{(B,T,1)}$ where $t_{max}$ denotes the maximum number of single-note samples observed in the dataset (should find the exact constant value)

* Category C:
  + When training Bass-DDSP: Extracted from bass audio
  + When training scat-to-bass classifier: Predicted with vocal encoder
  - `periodicity(t)`: $[0,1]^{(B,T,1)}$ confidence of `f0`
  - `centroid(t)`: $[0,1)^{(B,T,1)}$: spectral centroid of each time frame. scaled same as `f0(t)`.


Synthesis branches:

- `transient`: articulation-conditioned learned waveform burst, active only
  near note onset.
- `sustain`: harmonic DDSP branch for pitched string sustain.
- `noise`: filtered-noise branch for pluck, fret, and release energy.

Final audio is `transient + sustain + noise`. Reverb is disabled by default for
debugging so attack and loudness problems stay visible.

## Training Path

1. Train `config_idmt_bass_v2_single_note.yaml` until isolated note attacks and
   decays are believable.
2. Reuse the same model path for generated riff training once single notes are
   sharp enough.
3. Train the scat/vocal classifier independently.
4. Map scat syllable probabilities to observed articulation classes and
   continuous controls.

## Diagnostics

Use the dataset visualizer before training:

```bash
python visualize_idmt_riff.py \
  --config config_idmt_bass_v2_single_note.yaml \
  --out-dir debug/idmt_bass_v2_single_note \
  --seed 1234 \
  --pitch-source labels
```

Use a unique run name when training:

```bash
python train.py \
  --config config_idmt_bass_v2_single_note.yaml \
  --name idmt_bass_v2_single_note_001 \
  --steps 1000 \
  --batch 4
```

Render reconstruction diagnostics after training:

```bash
python visualize_ddsp_run.py \
  --run runs/idmt_bass_v2_single_note_001 \
  --out-dir runs/idmt_bass_v2_single_note_001/visuals \
  --seed 4321 \
  --pitch-source labels
```
