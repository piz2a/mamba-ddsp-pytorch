# Scat-to-Bass Final Demo

Open `scat_to_bass_demo.ipynb` in the workspace Python environment. The
notebook keeps only the user-facing steps: upload/select audio, configure the
articulation and octave controls, load the models, and synthesize.

The pipeline is:

1. Decode the recording at 16 kHz.
2. Extract TorchCREPE F0/periodicity and causal aubio Complex onsets.
3. Resolve onset, offset, gate, and note age with the existing monophonic
   state machine and calibrated noise gate.
4. Apply the selected `-2` to `+2` octave shift to vocal F0 and latch one
   articulation per note. Octave `0` sends detected F0 directly to Bass-DDSP.
5. In `Slap auto`, select `ST_NO` below MIDI 40 (82.41 Hz) and `SP_NO` at or
   above MIDI 40. This is a deterministic arrangement convention inferred
   from the IDMT slap riffs, not an acoustic slap classifier.
6. Predict a bass-like 320 ms onset-strength envelope using the trained
   articulation + F0 + loudness onset-envelope model.
7. Synthesize with the final Bass-DDSP checkpoint. Its IDCT waveform bank is
   computed once at model load and cached for every reconstruction.

Every run creates a unique directory under `demo/outputs/`. It contains raw
floating-point input/output WAVs, listening audio, the three branch WAVs, and
`run.json` with configuration, timing, and latency metadata. Existing results
are never overwritten.

The notebook reports two different timing concepts:

- **Observed wall time** for control extraction, mapping, synthesis, and total
  offline processing, including synthesis real-time factor (RTF).
- **Algorithmic timing** from the 32 ms analysis frame, 16 ms hop, and aubio's
  accepted-event decision delay. This excludes microphone, audio-driver, and
  host block scheduling latency. The 80 ms causal release/offset rules are not
  added as mandatory look-ahead.

The first 500 ms of a recording should contain only ambient background noise;
that segment calibrates the deterministic noise gate.
