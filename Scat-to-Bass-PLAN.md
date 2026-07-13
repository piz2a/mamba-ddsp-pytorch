# Scat-to-Bass: DDSP-Based, Transient-Assisted Bass Synthesizer from Scat Recordings

## Summary

Bass-DDSP v2 is a bass-specific synthesis path for IDMT-SMT-BASS. The goal is
to solve the smooth, violin-like behavior of the baseline DDSP model before
connecting the later scat-to-bass classifier.

The model is trained first on isolated labeled notes, then on generated riffs.
It uses observed IDMT articulation classes such as `FS_NO`, `PK_NO`, `SP_NO`,
and `FS_HA` instead of assuming plucking style and expression style are fully
independent.

## Architecture

Frame-level controls:

- `f0(t)` from labels by default.
- deterministic `loudness(t)`.
- observed `articulation_id(t)`.
- `gate(t)`, `onset(t)`, `offset(t)`, `note_age(t)`, and `note_progress(t)`.

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
