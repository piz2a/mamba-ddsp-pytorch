# DDSP-Guitar and Bass-DDSP: Control Architecture Report

## Question

Is adding more control information to DDSP a bad idea, and what does the
DDSP-Guitar architecture imply for interpreting Bass-DDSP's comparison with
the simpler Vanilla DWTS baseline?

## Short Answer

Adding controls is not inherently bad. A control is useful when it:

1. describes an audible factor that the synthesizer can represent;
2. is estimated consistently during training and inference;
3. is temporally aligned with the target;
4. changes the output in a way the loss can identify; and
5. does not unnecessarily suppress gradients or duplicate other controls.

DDSP-Guitar supports the idea of an acoustic control interface, but it does
not show that every additional explicit control improves synthesis. In fact,
the paper's best proposed model was the **unified** model, which bypassed
loudness, periodicity, and spectral centroid as explicit intermediate
bottlenecks and predicted synthesis parameters directly from MIDI. It retained
explicit F0 supervision because oscillator pitch is difficult to learn from a
spectral loss alone.

Bass-DDSP did much more than add controls. It simultaneously changed:

- additive harmonic synthesis to learned wavetable synthesis;
- two branches to three branches;
- a conventional decoder to an articulation encoder plus causal GRU;
- passive conditioning to several deterministic multiplicative envelopes;
- the data from continuous performances to generated riffs made from isolated
  notes;
- the loss by adding frame RMS, while the planned onset and branch losses
  remained inactive.

The final comparison therefore tests an entire architecture bundle, not the
isolated value of additional controls.

The corrected comparison shows that Bass-DDSP outperforms Vanilla DWTS on
MSS, LSD, loudness-envelope tracking, and onset reconstruction. An earlier
July 26 comparison incorrectly reloaded the unchanged Bass checkpoint through
modified model semantics and produced a false loudness collapse. That invalid
result is not used in this report.

## 1. What DDSP-Guitar Actually Does

### 1.1 The task

DDSP-Guitar maps **string-wise MIDI** to a six-string acoustic-guitar
waveform. Its training data is GuitarSet: 360 continuous performances totaling
slightly over three hours. Each performance provides:

- microphone audio as the final reconstruction target;
- six-channel hexaphonic pickup audio;
- string-wise pitch annotations;
- MIDI-like pitch, activity, onset, and offset information.

The hexaphonic recording is important. It lets the authors extract acoustic
features separately for each string before the six synthesized string signals
are mixed.

### 1.2 The default control-synthesis architecture

The paper's default system is:

```text
String-wise MIDI pitch, pseudo-velocity, string ID
                       |
                 Control model
                       |
       F0, loudness, periodicity, centroid
                       |
               Synthesis decoder
                       |
 harmonic partial amplitudes, global amplitude, noise bands
                       |
       6 x (harmonic + filtered noise) synth
                       |
            per-string learned reverb
                       |
                 summed waveform
```

The four synthesis controls are:

- **F0:** oscillator frequency and pitch trajectory.
- **Loudness:** framewise signal-energy information.
- **Periodicity:** tonal versus non-tonal evidence.
- **Spectral centroid:** a compact brightness/timbre description.

These controls are continuous acoustic summaries extracted from the
hexaphonic target recordings. They are not articulation labels and are not
used as hard note envelopes.

In the cloned implementation,
[`preprocessing.py`](ddsp-guitar/preprocessing.py) scales F0 logarithmically
between 35 and 1200 Hz, loudness from a dB range, leaves periodicity in its
bounded confidence range, and scales centroid by frequency range. The decoder
concatenates all four scalar features and a learned string embedding.

### 1.3 Decoder and synthesizer

The paper configuration uses a three-layer bidirectional LSTM with hidden size
512. The decoder emits, per string and frame:

- 128 normalized harmonic partial amplitudes;
- one global harmonic amplitude;
- 128 filtered-noise band amplitudes.

The synthesizer then generates six string signals and sums them. A trainable
0.25-second impulse response per string models the string/body/microphone
transfer differences. This "reverb" is therefore also an instrument-body and
recording-response model, not merely room ambience.

The relevant implementation is in
[`synthesis_model.py`](ddsp-guitar/synthesis_model.py):

- controls are concatenated at lines 205-210;
- the bidirectional RNN is applied at lines 214-221;
- harmonic and noise parameters are produced at lines 223-229;
- harmonic and noise audio are summed at lines 115-143.

### 1.4 Why bidirectional context helped their setting

DDSP-Guitar renders offline, overlapping eight-second excerpts. A
bidirectional LSTM can use future frames to infer:

- whether a short noisy event belongs to an attack or release;
- the direction and extent of a bend;
- local phrase-level loudness and timbre;
- interactions across a note boundary.

That is useful for offline reconstruction but is not directly suitable for
strict low-latency streaming. A short future buffer only makes it
fixed-lookahead, not causal: its latency is at least the lookahead duration.

Bass-DDSP deliberately uses a unidirectional GRU and causal state, so it gives
up this future information to support the scat-to-bass real-time goal.

### 1.5 The paper does not conclude that more controls are always better

The paper evaluates four systems:

| System | Explicit intermediate controls | Training |
|---|---|---|
| `ctr-syn-rg` | F0, loudness, periodicity, centroid | separately trained regression |
| `ctr-syn-cl` | same four controls | quantized classification |
| `ctr-syn-jt` | same interface | joint waveform training |
| `unified` | no explicit loudness/periodicity/centroid bottleneck | direct MIDI-to-synthesis parameters |

Two results are especially relevant:

1. Regression control prediction failed badly, especially for F0. Quantized
   classification was much better.
2. The **unified model was the best proposed model subjectively**, with MOS
   3.38 versus 3.00 for the jointly trained control-synthesis model.

During joint training, the authors removed direct supervision from predicted
loudness, periodicity, and centroid. Those channels were allowed to carry
whatever information minimized waveform loss. They kept F0 supervision.

Thus, DDSP-Guitar provides two lessons at once:

- acoustic controls can make a useful and interpretable synthesis interface;
- forcing every intermediate variable to remain a literal measured control
  can also become an information bottleneck.

## 2. What Bass-DDSP Does

### 2.1 Current input interface

Bass-DDSP receives:

```text
F0
loudness
articulation ID
onset_strength
offset
gate
note_age
periodicity
```

The trained model uses six observed articulation classes:

```text
FS_NO, MU_NO, PK_NO, SP_NO, ST_NO, FS_DN
```

F0, loudness, and an encoded articulation latent are processed through
separate MLP paths. The articulation latent `z(t)` is generated from:

```text
articulation embedding
    + onset_strength
    + offset
    + gate
    + note_age
    + periodicity
```

The three paths have hidden widths 64, 64, and 256 respectively. These widths
set representational capacity; they do not guarantee a fixed percentage of
model dependence. A 256-dimensional `z` path can still be ignored by later
layers if the loss can be minimized without it.

The implementation is in [`model.py`](bass_ddsp/model.py):

- articulation encoding: lines 115-134 and 537-566;
- independent F0/loudness/z MLPs: lines 289-302;
- causal GRU and output stack: lines 304-322 and 568-605.

### 2.2 Synthesis architecture

Bass-DDSP produces:

```text
signal = wavetable sustain + filtered noise + DCT-bank transient
```

The final trained configuration uses:

- 16 learned 512-sample wavetables;
- 65 filtered-noise bands;
- six articulation-indexed, 300 ms DCT-bank transient waveforms;
- a unidirectional GRU with hidden size 256;
- no reverb/body impulse response;
- fixed branch gains: sustain +12 dB, noise 0 dB, transient 0 dB.

The sustain branch is not the original DDSP harmonic oscillator bank. It
selects and mixes learned wavetables, predicts an amplitude, and then
multiplies the result by:

```text
gate * 6 ms fade-in * deterministic loudness gain * periodicity gate
```

The noise branch is multiplied by the note gate in the final run. The
transient branch is indexed by articulation and note age, then multiplied by
a 300 ms transient window and onset-dependent velocity.

These operations are in [`model.py`](bass_ddsp/model.py):

- sustain: lines 710-752;
- filtered noise: lines 754-765;
- DCT-bank transient: lines 850-882;
- final branch routing and summation: lines 901-952.

### 2.3 Data and controls

The isolated IDMT set has approximately 3.3 hours of unique note audio, but it
is not three hours of natural performance. The riff dataset randomly
concatenates trimmed isolated notes into 2.048-second training examples.

For the final run:

- examples per generated epoch: 1024;
- note duration range: 0.28 to 1.10 seconds;
- crossfade range: 30 to 75 ms;
- pitch: dataset labels;
- loudness: extracted from the generated target audio;
- onset strength: HPSS percussive energy with an annotated pulse floor;
- periodicity: TorchCREPE confidence aligned from source notes;
- gate, offset, and note age: generated from known note intervals.

Several controls are therefore **oracle target controls** during the current
bass-synth evaluation. The eventual scat front end will not produce them with
the same extraction process or error distribution.

### 2.4 Training objective

The final training objective was:

```text
MSS(target, reconstruction)
    + 1.0 * frame_log_RMS_L1(target, reconstruction)
```

The logged `onset_spectral_loss`, `transient_loss`, and
`transient_branch_loss` are explicitly initialized to zero and never added to
the objective in [`train.py`](bass_ddsp/train.py), lines 288-305.

Therefore:

- no target directly says what the transient branch should synthesize;
- no target directly says what the noise branch should synthesize;
- no loss prevents the sustain branch from explaining transient/noise energy;
- no class loss forces two articulation IDs to produce perceptually distinct
  results.

The branch decomposition is underdetermined. Only the summed waveform is
reconstructed.

## 3. Direct Architectural Comparison

| Property | DDSP-Guitar synthesis model | Bass-DDSP |
|---|---|---|
| Primary task | six-string MIDI to guitar performance | monophonic bass controls to audio |
| Training context | natural continuous performances | generated riffs from isolated notes |
| Input frames | 128 Hz | 62.5 Hz |
| Audio rate | 48 kHz | 16 kHz |
| Temporal model | 3-layer bidirectional LSTM, H=512 | causal 1-layer GRU, H=256 |
| Synthesis parameters | harmonic distribution, global amp, noise bands | wavetable mixture/amp, noise bands, transient gain |
| Sustain synthesis | 128 additive harmonics | 16 learned wavetables |
| Noise | 128 bands | 65 bands |
| Explicit transient | none | 300 ms articulation DCT bank |
| Instrument/body model | learned IR per string | disabled |
| String handling | six parallel voices plus string embedding | one monophonic voice |
| Explicit controls | F0, loudness, periodicity, centroid, string | F0, loudness, articulation, onset, offset, gate, age, periodicity |
| Parameter count | 18.2M synthesis model in paper | 1.025M |
| Main synthesis loss | MSS | MSS + frame log-RMS |
| Real-time causal | no | intended yes |

The Bass-DDSP model is not simply "DDSP-Guitar plus more controls." It is a
smaller, causal, monophonic, transient-assisted wavetable model trained on a
different form of data.

## 4. Why Additional Controls Did Not Guarantee Improvement

### 4.1 Controls can be redundant

`gate`, `offset`, and `note_age` are all derived from the same note interval.
`onset_strength` also peaks near that interval's beginning. This does not make
them invalid, but it means adding five scalar channels does not provide five
independent facts.

DDSP-Guitar's centroid and periodicity describe complementary framewise
acoustic properties. Bass-DDSP's event controls mostly describe different
views of note timing.

### 4.2 Conditioning and hard signal manipulation are different

DDSP-Guitar passes periodicity and centroid to a decoder. The network decides
how they affect harmonic and noise parameters.

Bass-DDSP both conditions its latent representation on controls and applies
some controls deterministically afterward:

- gate directly zeros branches;
- periodicity directly attenuates sustain through a sigmoid gate;
- loudness directly multiplies sustain by a bounded gain;
- note age indexes the transient waveform and transient window.

This is a stronger inductive bias. It can improve controllability when the
mapping is correct, but it can also enforce the wrong amplitude or timing even
when the decoder predicts better synthesis parameters.

### 4.3 The branch assignment is not identified by the loss

MSS and RMS losses see only:

```text
sustain + noise + transient
```

Many branch combinations produce similar spectra. The optimizer usually puts
energy into the easiest stable branch. In the final evaluation, sustain RMS
was 96.95% of final-signal RMS, while noise was 0.87% and transient 14.73%.
RMS percentages do not add linearly because branches can correlate or cancel.

The explicit transient branch therefore does not automatically learn "the
true bass transient." It learns any waveform contribution that helps the
summed loss.

### 4.4 Articulation labels are coarse relative to the acoustic variation

One 300 ms transient bank waveform is shared by every example of an
articulation class. Real attacks also vary with:

- pitch;
- plucking position and force;
- instrument and pickup response;
- residual trimming alignment;
- source loudness;
- string/fret mechanics that are intentionally omitted from the interface.

The predicted transient gain can adapt, but the class-indexed waveform itself
encourages an average attack. That average can become quieter or blurrier than
individual targets.

### 4.5 Oracle controls create a future inference mismatch

Current evaluation supplies onset strength and periodicity extracted from bass
target audio. The final system must derive events from voice and generate a
bass-domain onset envelope after articulation classification.

DDSP-Guitar explicitly studied this gap:

1. train the synthesizer with ground-truth acoustic controls;
2. train a MIDI-to-control predictor;
3. jointly fine-tune or remove the intermediate bottleneck;
4. evaluate predicted-control audio separately from oracle-control audio.

Bass-DDSP has only completed the bass-side oracle-control stage. Good
reconstruction under those controls would not yet prove robustness to scat
controls; poor reconstruction means the bass synthesizer itself still needs
work before that mismatch is introduced.

### 4.6 Causality and capacity differ substantially

DDSP-Guitar's synthesis decoder alone has 18.2M parameters and sees both past
and future context over eight-second excerpts. Bass-DDSP has about 1.025M
parameters and is causal. The Vanilla DWTS and Vanilla DDSP baselines have
about 0.543M and 0.556M parameters, respectively.

Bass-DDSP is larger than its local baselines, but far smaller and more
constrained than DDSP-Guitar. More control dimensions do not compensate for
less temporal context or a synthesis family that does not match the target.

### 4.7 The experiment changed too many factors at once

The final 32-riff comparison was:

| Metric | Bass-DDSP | Vanilla DWTS | Vanilla DDSP |
|---|---:|---:|---:|
| MSS, lower | **5.6478** | 5.8232 | 6.0489 |
| LSD dB, lower | **12.453** | 12.654 | 13.728 |
| RMS ratio, target 1 | **0.925** | 0.851 | 0.743 |
| RMS correlation, higher | **0.919** | 0.852 | 0.786 |
| Gross pitch error %, lower | 8.89 | **6.32** | 8.13 |
| Onset HF log error, lower | **0.456** | 0.459 | 0.548 |
| Onset energy ratio, target 1 | **0.917** | 0.866 | 0.582 |

Bass-DDSP has the best reconstruction, loudness, and onset metrics in this
evaluation. Vanilla DWTS, using only F0 and loudness, has the lowest gross
pitch error. This is positive evidence for the complete Bass-DDSP bundle, but
it still does **not** prove that every articulation or event control is useful
individually.

The Vanilla DWTS comparison is especially useful because it retains a
wavetable sustain family. The models are close on MSS, LSD, and onset HF
error, while Bass-DDSP is clearly better on RMS tracking and onset energy.
Controlled ablation is still required to determine whether this advantage
comes from the added controls, transient branch, loss, fixed sustain gain, or
their interaction.

## 5. What Should Be Kept

The following ideas remain justified for scat-to-bass:

- **F0:** necessary and should remain explicitly supervised.
- **Loudness:** useful as a continuous expressive control, although its exact
  deterministic gain mapping needs calibration.
- **Gate:** useful as the final monophonic note-active contract.
- **Note age:** causal and physically meaningful for attack/decay evolution.
- **Articulation ID:** central to the user-facing goal, provided its effect is
  demonstrated with controlled evaluation.
- **Causal recurrent state:** required for real-time inference.

`offset` can remain an event for state reset and release behavior without also
needing to consume a large latent pathway. `periodicity` is useful as tonalness
evidence, but its current hard sustain attenuation should be treated as a
separate hypothesis. `onset_strength` is useful only after the vocal-to-bass
semantic mismatch is resolved.

## 6. Recommended Experiment

Do not answer the control question with another full architecture rewrite.
Perform a cumulative ablation using the same DWTS sustain, parameter budget,
dataset split, seed, steps, and losses:

| Model | Added information |
|---|---|
| A | F0 + loudness |
| B | A + articulation embedding |
| C | B + gate + note age |
| D | C + periodicity as decoder conditioning only |
| E | D + onset strength and offset |
| F | E + deterministic periodicity/loudness/transient routing |

For every stage, measure:

- MSS and LSD;
- RMS ratio and correlation;
- onset HF error and onset energy ratio;
- articulation-conditioned listening preference;
- control sensitivity: change one control while holding all others fixed.

The control-sensitivity test is essential. A reconstruction metric cannot show
whether `PK_NO` and `FS_NO` are actually controllable. For each held-out note,
render the same F0/loudness trajectory under all six articulation IDs and
measure whether differences are:

- repeatable within a class;
- larger between classes than within classes;
- concentrated in plausible attack/timbre regions;
- preferred by listeners when matched to the target label.

## 7. Architectural Recommendation

Use a simpler causal control interface:

```text
F0 MLP ------------------\
loudness MLP -------------+--> causal GRU --> shared hidden state
articulation embedding ---/
gate + note_age ----------/

shared hidden state:
    -> sustain parameters
    -> filtered-noise parameters
    -> transient parameters

DSP routing:
    gate: hard final note-active mask
    note_age: causal phase within the note
    all other controls: learned conditioning first
```

Initially avoid hard periodicity attenuation and hand-tuned branch decay.
Reintroduce each only when an ablation demonstrates improvement. Inject
realistic noise, timing jitter, and classifier uncertainty into articulation
and event controls during training so that the synthesizer does not require
perfect oracle inputs.

For branch specialization, choose one of two explicit positions:

1. **Interpretable decomposition:** provide justified branch targets or
   regularizers, such as carefully validated harmonic/noise separation and
   onset-region supervision.
2. **Best reconstruction:** do not claim that branches correspond to physical
   components; allow them to cooperate under waveform loss.

The current model informally expects physical specialization but trains mostly
for summed reconstruction. That mismatch is more problematic than the raw
number of controls.

## Conclusion

DDSP-Guitar does not invalidate the Bass-DDSP control idea. It shows that
well-defined acoustic controls can support synthesis, especially when they are
extracted from source-separated recordings and predicted with a carefully
trained upstream model.

It also gives a warning directly relevant to this project: explicit
intermediate controls can become a bottleneck, regression targets can be
noisy, and a simpler unified path may sound better.

The present Bass-DDSP result should be interpreted as:

> The complete transient-assisted, hard-routed, articulation-conditioned
> architecture improved generated-riff reconstruction, loudness tracking, and
> onset energy over the two fully trained baselines, while Vanilla DWTS
> retained better gross pitch accuracy.

It should not be interpreted as:

> Additional control information is bad for DDSP.

The next research contribution is to identify which controls produce genuine,
robust, user-audible controllability and which components are responsible for
the measured improvement.

## Sources Inspected

- [`2309.07658v1-DDSP-Guitar.pdf`](papers/2309.07658v1-DDSP-Guitar.pdf)
- [`ddsp-guitar/synthesis_model.py`](ddsp-guitar/synthesis_model.py)
- [`ddsp-guitar/preprocessing.py`](ddsp-guitar/preprocessing.py)
- [`ddsp-guitar/train_synthesis.py`](ddsp-guitar/train_synthesis.py)
- [`bass_ddsp/model.py`](bass_ddsp/model.py)
- [`bass_ddsp/dataset.py`](bass_ddsp/dataset.py)
- [`bass_ddsp/train.py`](bass_ddsp/train.py)
- [`corrected final model comparison`](runs/model_comparison_branchbalance_corrected_preserved_bass_full_dwts_20260729/REPORT.md)
- [`final Bass-DDSP configuration`](runs/bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513/config.yaml)
