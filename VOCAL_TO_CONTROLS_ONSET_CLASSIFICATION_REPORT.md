# Vocal-to-Controls: Onset Detection and Articulation Classification

## 1. Purpose and Status

This document is the authoritative design report for the causal vocal front end
of the scat-to-bass system. It replaces:

- `SCAT_ONSET_STRENGTH_PREPROCESSING_REPORT.md`
- `vocal-to-controls-extractor-PLAN.md`

The intended output is a synchronized control tensor for Bass-DDSP:

```text
(B, T, C)

f0(t)
loudness(t)
periodicity(t)
gate(t)
onset(t)
offset(t)
note_age(t)
articulation_id(t)
onset_strength(t)
```

The current implementation is a laboratory prototype:

| Component | Status |
|---|---|
| TorchCREPE F0 and periodicity | Implemented |
| RMS loudness | Implemented |
| Aubio Complex onset events | Implemented |
| Deterministic noise gate | Implemented |
| Monophonic onset/offset state machine | Implemented |
| Note age | Implemented |
| Acoustically aligned latency visualization | Not implemented |
| Learned articulation classifier | Not implemented |
| Final vocal-to-bass onset-strength mapping | Not integrated |
| User-specific adaptation | Not implemented |

No trained vocal articulation classifier exists. The function
`_articulation_id()` in `vocal_controls.py` is only a deterministic laboratory
placeholder. It must not be described as a trained classifier or as validated
articulation recognition.

## 2. Frame and Signal Contract

Current laboratory settings:

| Setting | Value |
|---|---:|
| Sample rate | 16 kHz |
| Control hop | 16 ms / 256 samples |
| Analysis frame | 32 ms / 512 samples |
| Control topology | Monophonic |

Every control at index `t` must ultimately describe the same acoustic time.
This requires two timestamps:

```text
acoustic_timestamp: when the event occurred in the waveform
availability_timestamp: when a causal detector produced the result
```

The evaluation plot must use acoustic timestamps. A real-time implementation
must buffer faster control paths to a common availability deadline.

## 3. What the AVP Classification Paper Actually Shows

Reference:

> Delgado et al., "Deep Embeddings for Robust User-Based Amateur Vocal
> Percussion Classification," 2022,
> `papers/2204.04646v1-AVP-Classification.pdf`.

The paper separates Vocal Percussion Transcription into onset detection and
classification, but studies classification only. Its classifier receives
already segmented vocal-percussion events. The authors explicitly leave onset
detection for future work. It therefore does not provide the missing causal
onset/offset system for this project.

Relevant details:

- The combined AVP-LVT set contains 5,714 boxemes before augmentation.
- Pitch shifting and time stretching expand it to 62,854 examples.
- Inputs are log-mel spectrograms with 64 mel bands.
- The hop is 12 ms and each example has 48 time steps, approximately 0.56 s.
- The authors test analysis frame sizes of 23, 46, and 93 ms; 46 ms performs
  best in their reported experiment.
- The embedding CNN has four convolutional blocks followed by an embedding
  layer and a classification layer.
- Syllable-level supervision produces the strongest and most stable
  embeddings.
- User-specific methods outperform the user-agnostic speech-recognition
  baseline.
- A cited earlier study found that starting classification analysis 23 ms
  after onset improved real-time discrimination.

The approximately 0.56 s input duration is suitable for classifying isolated
boxemes after segmentation. It must not be interpreted as an acceptable
real-time latency for scat-to-bass. The transferable findings are:

1. User-specific adaptation is important for amateur vocalizations.
2. Onset-plus-coda or syllable supervision can be more informative than only
   supervising a high-level instrument class.
3. The earliest transient is important, but a very short post-onset delay may
   expose more discriminative consonant information.
4. Background-noise conditions materially affect generalization.

## 4. Onset Detection

### 4.1 Aubio Novelty Is Not an Onset Strength

The laboratory computes two aubio methods:

- `aubio.onset("complex")`
- `aubio.onset("hfc")`

It also plots the corresponding `aubio.specdesc` novelty functions.

These outputs have different meanings:

| Output | Meaning |
|---|---|
| Complex/HFC novelty | Continuous change descriptor with many non-event peaks |
| `aubio.onset` return | Sparse accepted event marker |
| `get_last_s()` | Delay-compensated estimate of the acoustic event time |
| Detector callback frame | Time when the result becomes causally available |

A novelty peak is not a probability and is not a reliable transient-amplitude
control. Aubio's onset object applies peak picking, adaptive thresholding,
silence rejection, minimum inter-onset interval, and delay compensation after
the descriptor calculation.

### 4.2 Selected Detector

Complex is the selected onset detector. In the current voice experiments it is
less sensitive to irrelevant silent/background events than HFC. HFC remains a
diagnostic plot only.

Known weaknesses:

- Fast consonant changes can be missed.
- Vowel-only emphasis is difficult because there may be no clear consonant
  boundary.
- Soft `/l/` attacks, such as repeated "la," are poorly detected.
- Users may currently need to emphasize note boundaries.
- A consonant followed by its vowel can occasionally be split into two notes.

The last failure is unresolved. It needs a consonant-vowel grouping rule or a
learned boundary model, not just a larger global retrigger hold.

### 4.3 Noise-Gate Masking

The noise gate is calibrated independently for each recording:

```text
noise profile      = first 500 ms
noise peak         = maximum frame RMS in that profile
gate threshold     = noise peak + 10 dB
raw gate           = frame RMS >= gate threshold
```

The assumption is that the first 500 ms contains only ambient noise. If speech
appears in this interval, the estimated threshold may become unusably high.
A production UI must expose calibration and warn when this assumption fails.

The gate has a configurable 80 ms causal release hold. This is a state rule,
not processing latency:

```text
if raw gate was recently open:
    keep effective gate open
else:
    close effective gate now
```

It prevents the gate from closing during aubio's observed decision time.

The gate only masks aubio's accepted output:

```text
onset_candidate(t) = aubio_complex_callback(t) AND effective_noise_gate(t)
```

It does not mask aubio's audio input. It also never creates an onset from its
own rising edge. This distinction fixed the earlier failure where every short
gate opening could become a note.

The callback frame, not the backdated `get_last_s()` frame, is used for causal
masking. The backdated timestamp is retained for acoustic-alignment plots.

### 4.4 Re-trigger Filtering

Accepted onset candidates pass through a configurable 80 ms causal re-trigger
hold. It suppresses duplicate detector events that occur too close to the
previous accepted onset.

This hold is not algorithmic latency. The first onset is emitted immediately;
only later candidates inside the hold are rejected.

## 5. Monophonic Onset/Offset State Machine

The detector has two states:

```text
INACTIVE
ACTIVE
```

The valid transitions are:

| Current state | Condition | Output | Next state |
|---|---|---|---|
| Inactive | Accepted onset and gate open | onset | Active |
| Inactive | Offset evidence | nothing | Inactive |
| Active | Effective gate closes | offset immediately | Inactive |
| Active | Periodicity < 0.35 after guard | offset | Inactive |
| Active | New accepted onset | offset + onset on same frame | Active |

The same-frame offset/onset transition is required for consonant-separated
notes sung over continuous voicing. It terminates the previous monophonic note
before opening the next one.

The state machine prevents unmatched repeated offsets. In a completed
recording that returns to silence, onset and offset event counts should match.
An active note at the end of a live stream is not an error; stream termination
must explicitly flush it.

### 5.1 Offset Controls

Current configurable rules:

- Periodicity threshold: `0.35`.
- Low-periodicity offset guard: `80 ms` after onset.
- Effective noise-gate closure: immediate offset.
- New onset during an active note: immediate offset and onset together.

The 80 ms guard is causal. It means the model intentionally plays the first
80 ms of a newly opened note before low periodicity is allowed to terminate
it. It does not wait for future evidence about an earlier frame.

Likewise, the 80 ms noise-gate release is causal state memory. It does not add
80 ms to processing latency. Once the effective gate becomes false, offset is
known and emitted immediately.

## 6. Note Age

`note_age(t)` is a causal counter:

```text
on onset:       age = 0
while active:   age += 16 ms per frame
while inactive: age = 0
```

It requires no future note-off time. `note_progress(t)` is excluded because it
would require knowing the future duration of the note.

## 7. Onset Strength for Bass-DDSP

### 7.1 Bass Training Target

Bass-DDSP did not learn a one-frame Dirac delta. In
`bass_ddsp/dataset.py`, its target is:

```text
HPSS percussive energy
    -> peak normalization
    -> max with annotated triangular onset floor
```

The triangular fallback has a configured width of 32 ms on a 16 ms frame
grid, so the annotation affects multiple nearby frames. The learned target is
a short continuous bass attack envelope, not merely an event Boolean.

### 7.2 Why Vocal HPSS Is Not the Final Mapping

Vocal HPSS and bass HPSS are not distributionally equivalent:

- Voice recordings contain ambient noise that causes unstable percussive
  energy.
- A vocal consonant has different spectral and temporal structure from a
  plucked bass string.
- Aubio novelty also contains non-onset spikes and must not be used directly
  as transient amplitude.

Therefore, the final decision is:

```text
detected onset + predicted articulation
    -> bass-domain onset-envelope lookup/model
    -> onset_strength(t) expected by Bass-DDSP
```

The envelope is instantiated at each accepted onset and reset/truncated at
offset or retrigger.

The existing `onset_envelope_net` was trained on onset-aligned
`IDMTBassNoteDataset` examples. The best ablation used articulation, F0, and
loudness with validation MSE approximately `0.06049`, but all tested control
sets performed very similarly. This does not yet prove that the extra
continuous controls produce perceptually meaningful envelope changes.

The conservative interpretation is:

- Articulation-conditioned templates are the baseline.
- F0/loudness modulation remains optional until perceptual evaluation shows a
  benefit.
- The generated envelopes must be listened to through Bass-DDSP, not judged
  only by pointwise MSE.

### 7.3 Current Implementation Mismatch

`vocal_controls.py` still assigns vocal HPSS percussive energy directly to
`onset_strength(t)`. This remains useful as a diagnostic comparison but
contradicts the final lookup-model decision. It must be replaced before the
vocal front end is integrated with Bass-DDSP.

## 8. Articulation Classification

### 8.1 Current Status

Articulation classification has not been implemented.

There is currently:

- no labeled scat-to-articulation dataset,
- no trained GRU/CNN classifier,
- no validated mapping from Korean scat syllables to the six bass classes,
- no confidence calibration,
- no user-adaptation procedure.

The deterministic `_articulation_id()` function in `vocal_controls.py`
selects a class from hand-arranged feature values. It is a visualization
placeholder and must not be used as experimental evidence.

### 8.2 Target Classes

The current Bass-DDSP label set is:

```text
FS_NO
MU_NO
PK_NO
SP_NO
ST_NO
FS_DN
```

These are observed articulation classes, not independent plucking and
expression factors.

### 8.3 Candidate Causal Classifier

The classifier should operate only after an accepted onset:

```text
accepted onset
    -> short causal post-onset feature sequence
    -> articulation classifier
    -> one articulation_id
    -> latch class until offset/retrigger
```

Candidate inputs:

- log-mel or compact spectral representation,
- high-frequency energy ratio,
- spectral tilt,
- ZCR,
- periodicity,
- optional low-dimensional speech/content embedding.

The model should not classify every frame independently. It should produce one
note-level decision and latch it for the active note.

A causal GRU or compact temporal CNN is a reasonable first baseline. The
window length is not finalized. The AVP paper's 0.56 s isolated-boxeme input
is too long for this application. Initial experiments should compare short
post-onset contexts such as:

```text
23 ms delayed start from onset
64 ms context
96 ms context
128 ms context
```

This is an experiment proposal, not an implemented architecture decision.
Accuracy must be reported jointly with classification availability latency.

### 8.4 Label Strategy

The AVP result suggests that direct high-level class supervision may discard
important phonetic structure. Two strategies should be compared:

1. Direct six-class bass articulation supervision.
2. Multi-task or hierarchical supervision:
   - consonant/onset category,
   - vowel/coda category,
   - final bass articulation.

The second strategy may help distinguish syllables such as "땅", "땡", and
"뜽", while still producing one bass articulation class.

### 8.5 User Adaptation

Amateur vocalization is highly speaker-specific. A practical adaptation
session should let a user sing or imitate a small set of known riffs, ideally
aligned with examples from `IDMT-SMT-BASS-SINGLE-TRACKS`.

Possible adaptation methods:

- calibrate a small classification head while freezing the feature encoder,
- prototype/class-centroid adaptation in embedding space,
- few-shot nearest-neighbor classification,
- per-user threshold and confidence calibration.

This requires a concrete recording and annotation protocol before model
training begins.

## 9. ContentVec Real-Time Feasibility

### 9.1 Evaluated Model

Checkpoint:

`contentvec/checkpoints/checkpoint_best_legacy_100.pt`

Benchmark:

`scripts/benchmark_contentvec_realtime.py`

Raw results:

- `contentvec_latency_benchmark.json`
- `contentvec_latency_benchmark_cpu.json`

The checkpoint is a legacy ContentVec-100 representation model:

| Property | Value |
|---|---:|
| Parameters | 94,595,200 |
| Checkpoint size | 1.33 GB |
| Transformer layers | 12 |
| Embedding dimension | 768 |
| Feature stride | 20 ms / 320 samples |
| Convolutional receptive field | 25 ms / 400 samples |
| Positional convolution | 128 feature frames / 2.56 s total span |
| Self-attention | Unrestricted/bidirectional |

The symmetric positional convolution alone uses approximately 1.28 s of
future context around an interior frame. More importantly, unrestricted
self-attention allows every output frame to depend on every later frame in
the supplied chunk. The downloaded model therefore has no fixed finite
streaming lookahead independent of chunk size.

### 9.2 Throughput Results

Test audio: `learn/voice_inputs/Scat 1.wav`.

GPU: NVIDIA GeForce RTX 3080 Ti, FP32.

CPU: Intel Xeon Gold 6226R at 2.90 GHz.

| Input chunk | Median GPU time | GPU RTF | Median CPU time | CPU RTF |
|---:|---:|---:|---:|---:|
| 32 ms | 10.70 ms | 0.334 | 24.81 ms | 0.775 |
| 64 ms | 9.64 ms | 0.151 | 25.83 ms | 0.404 |
| 80 ms | 17.84 ms | 0.223 | 27.14 ms | 0.339 |
| 100 ms | 15.00 ms | 0.150 | 27.05 ms | 0.271 |
| 250 ms | 14.63 ms | 0.059 | 30.71 ms | 0.123 |
| 500 ms | 15.64 ms | 0.031 | 43.34 ms | 0.087 |
| 1 s | 17.83 ms | 0.018 | 53.02 ms | 0.053 |

RTF below one means the model can process the chunk faster than the chunk's
duration on this machine. It does not prove causal correctness. The CPU
32 ms result also leaves only about 7 ms for every other VST operation before
the block deadline, so it is not a comfortable production budget.

Checkpoint loading takes approximately 2.6-2.9 s. Measured peak CUDA tensor
allocation was approximately 489 MB, while the checkpoint on disk is much
larger because it contains additional training state.

These measurements do not include:

- audio-driver and VST callback overhead,
- the articulation classifier,
- Bass-DDSP synthesis,
- device transfer and synchronization in a complete plugin,
- contention with other audio software.

### 9.3 Chunk Consistency

Independent chunks were compared with the corresponding frames from one
4-second full-context inference using cosine similarity:

| Independent chunk | Mean cosine | 10th percentile | Minimum |
|---:|---:|---:|---:|
| 250 ms | 0.320 | 0.131 | 0.008 |
| 500 ms | 0.526 | 0.243 | -0.011 |
| 1 s | 0.701 | 0.446 | 0.196 |

This is substantial representation drift. Naively resetting ContentVec every
small VST block does not reproduce the representations used during normal
full-context inference.

The model also emits features every 20 ms, while the Bass-DDSP control grid is
16 ms. A resampling/alignment policy would be required even after solving
causality.

### 9.4 Decision

The downloaded ContentVec model is computationally fast enough for offline or
buffered inference on this machine, but it is not a drop-in causal VST
encoder. It must not be included when claiming the current approximately
65 ms causal control latency.

Valid future uses:

1. Use ContentVec offline as a teacher and distill its useful articulation
   information into a small causal student.
2. Retrain or fine-tune with causal/chunked attention and causal positional
   encoding.
3. Replace it with a purpose-built streaming speech encoder.
4. Use it only for offline annotation, analysis, or user-adaptation feature
   preparation.

A rolling past-only window followed by selecting the last embedding is
technically causal with respect to input samples, but it is inefficient and
places every prediction at a boundary condition the model was not trained
for. It requires task-level validation and should not be assumed equivalent
to ordinary ContentVec inference.

The preferred first articulation baseline remains compact causal features plus
a small GRU/temporal CNN. ContentVec should be revisited only as an offline
teacher or after a causal conversion experiment.

## 10. Latency Contract

### 10.1 Three Different Concepts

Do not conflate:

1. **Algorithmic latency:** future audio required before a result is known.
2. **Causal state duration:** a rule such as an 80 ms release or guard.
3. **Execution time:** wall-clock compute required by the Python/VST code.

The two 80 ms state rules add no algorithmic lookahead. They use only current
and past frames.

### 10.2 Current Known Latencies

Code inspection gives:

| Process | Current latency interpretation |
|---|---:|
| Note-age/state-machine update | 0 ms lookahead |
| 80 ms onset re-trigger hold | 0 ms lookahead |
| 80 ms low-periodicity offset guard | 0 ms lookahead |
| 80 ms gate release | 0 ms lookahead |
| Centered 32 ms RMS/STFT features | 16 ms lookahead |
| TorchCREPE 1024-sample centered frame | approximately 32 ms lookahead |
| Aubio Complex onset | measured approximately 50-65 ms decision latency |
| Librosa HPSS | noncausal time-context operation; not VST-ready |
| Legacy ContentVec-100 | bidirectional/full-context; not VST-ready |
| Articulation classifier | unknown; not implemented |
| Onset-envelope lookup | expected causal, execution time not measured |

Excluding HPSS and the unimplemented classifier, the largest currently known
algorithmic delay is Aubio Complex at approximately 65 ms.

This is not yet the final end-to-end VST latency because:

- the classifier latency is unknown,
- Python execution time has not been benchmarked as streaming blocks,
- TorchCREPE must be validated in a causal streaming implementation,
- HPSS must be removed from the real-time path.

### 10.3 Future VST Synchronization

In a VST, all parallel branches receive the same incoming waveform block.
Each branch associates its result with an acoustic timestamp. Faster results
are queued until a shared deadline:

```text
D = max(valid causal estimator latency)

emit control for acoustic frame t at wall time t + D
```

If `D = 65 ms` after final validation:

- loudness is buffered to 65 ms,
- F0 and periodicity are buffered to 65 ms,
- onset is available by the same deadline,
- causal state controls are buffered to 65 ms,
- Bass-DDSP receives a time-aligned control vector.

Serial execution inside a VST callback affects CPU budget, not signal
alignment, provided all processing finishes before the callback deadline. If
real-time factor exceeds one, the result is dropout/glitching rather than a
well-defined additional signal delay.

## 11. Visualization Contract

The notebook is:

`learn/scat_feature_extraction_colab.ipynb`

Keep the dashboard focused:

1. Log-mel spectrogram.
2. Waveform with acoustically aligned onset/offset markers and lightly shaded
   detected-note intervals.
3. Aubio Complex and HFC novelty/events in one panel.
4. F0 and periodicity.
5. Loudness and deterministic noise gate.
6. Spectral flux, high-frequency ratio, and vocal HPSS as diagnostics only.

Removed and prohibited from the main dashboard:

- Silero VAD,
- fused-gate component plots,
- ContentVec embedding heatmaps,
- linear-frequency STFT magnitude,
- arbitrary multiplication of unrelated controls.

The accuracy plot must use acoustic timestamps:

- accepted onset marker: aubio `get_last_s()` associated with the accepted
  callback event,
- offset marker: estimated acoustic boundary,
- controls: acoustic frame centers.

Availability delay must be displayed separately, for example:

```text
RMS/spectral lookahead: 16 ms
TorchCREPE lookahead: approximately 32 ms
Aubio Complex decision latency: 50-65 ms
Articulation classification latency: not implemented
Overall synchronized latency: pending; current known maximum 65 ms
```

The current dashboard still mixes callback-time accepted markers with
backdated aubio markers. The final acoustic-time latency annotation described
above remains to be implemented.

## 12. Configurable Detector Parameters

The notebook setup cell exposes:

| Parameter | Current value |
|---|---:|
| `ONSET_RETRIGGER_HOLD_MS` | 80 |
| `OFFSET_TRIGGER_HOLD_MS` | 80 |
| `OFFSET_PERIODICITY_THRESHOLD` | 0.35 |
| `NOISE_GATE_MARGIN_DB` | 10 |
| `NOISE_GATE_RELEASE_HOLD_MS` | 80 |

These are experimental defaults, not universal constants. They were selected
from a small set of personal recordings and require validation on a larger,
annotated evaluation set.

## 13. Evaluation Plan

### 13.1 Onset Detection

At least dozens of labeled recordings are needed. Five to seven informal
examples are insufficient for a paper claim.

Report:

- onset precision,
- onset recall,
- onset F1,
- timing error in milliseconds,
- false events during silence,
- performance by consonant/syllable category,
- measured decision latency distribution.

Use a stated matching tolerance, such as ±25 ms and ±50 ms.

### 13.2 Offset Detection

Report:

- offset precision/recall/F1,
- timing error,
- premature note termination rate,
- missed sustained-note termination rate,
- onset/offset count consistency,
- behavior for retriggered monophonic notes.

### 13.3 Articulation Classification

Once implemented, report:

- six-class macro F1,
- balanced accuracy,
- confusion matrix,
- per-user and user-independent results,
- classification latency,
- confidence calibration,
- ablation by feature family and context length.

### 13.4 End-to-End Control Quality

The final criterion is not classifier accuracy alone. Evaluate:

- note-boundary correctness,
- articulation stability within a note,
- onset-envelope plausibility,
- Bass-DDSP reconstruction/synthesis quality,
- end-to-end latency,
- user-rated controllability.

## 14. Immediate Next Work

1. Implement acoustic-time plotting and per-process latency reporting.
2. Remove vocal HPSS from the real-time `onset_strength(t)` control and
   integrate the bass-domain envelope lookup/model.
3. Resolve consonant-vowel double triggering.
4. Define and record a labeled user-specific scat articulation dataset.
5. Train the first short-context causal classifier.
6. Compare direct articulation supervision with hierarchical
   consonant/coda supervision.
7. Design the user adaptation workflow using easy labeled bass riffs.

Until these are complete, the correct project claim is:

> Deterministic causal note-boundary extraction is prototyped; bass-domain
> onset-envelope generation has an experimental model; vocal articulation
> classification and final latency-synchronized VST integration remain open.
