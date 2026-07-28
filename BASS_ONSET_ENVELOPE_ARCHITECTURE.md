# Bass Onset-Strength Envelope Model

## Purpose

The scat front end should not estimate bass `onset_strength(t)` with HPSS.
Instead, articulation classification selects a bass attack template learned
from clean IDMT-SMT-BASS notes. A small causal network modifies that template
using the controls available at the onset.

## Runtime Contract

At an accepted onset frame `t0`, provide:

```text
articulation_id(t0)  categorical observed IDMT articulation
f0(t0)               Hz
loudness(t0)         DDSP loudness control
periodicity(t0)      TorchCREPE confidence
```

The model emits `K=20` frames at the 16 ms control rate, covering 320 ms:

```text
e[0:K] = predicted bass-like onset_strength envelope
```

This is causal because it uses no note duration or future frame. The output
is rendered by a small event state machine:

1. At onset, start the envelope at its first element.
2. While active, advance one envelope element per control frame.
3. At a new onset, terminate the old envelope and restart the new one.
4. At offset, clear the active envelope and output zero.

Thus a consonant inside a sustain can create an offset and a new attack even
when a separate acoustic offset detector is uncertain.

## Network

```text
articulation_id -> Embedding --------------------┐
                                                  ├─> modulation MLP -> K values
f0, loudness, periodicity at onset -> normalized -┘

articulation Embedding -> base template logits -> K values

prediction = sigmoid(base_template + modulation)
```

The embedding is an explicit lookup table. The MLP does not replace it; it
modulates the articulation-specific shape for pitch, energy, and periodicity.
The network is intentionally small: two hidden layers of 128 units and one
320 ms output head. No Mamba, ContentVec, or recurrent future context is used.

## Training Targets

`IDMTBassRiffDataset` remains the source of truth. For every generated riff:

- use the observed articulation label at each interval;
- use label F0, dataset loudness, and dataset TorchCREPE periodicity at onset;
- take the dataset's normalized HPSS onset-strength track after the event;
- stop the target at the interval end and pad the remainder with zero;
- train with bounded MSE plus a small first-frame weighted MSE.

The HPSS curve is used only to learn clean bass templates. It is not used at
scat inference.

## Why 320 ms

It is long enough to contain the dominant finger/pick attack and early decay,
while remaining a fixed causal buffer. It can later be changed to 256 or 384
ms without changing the model interface.

## Limitations

This stage assumes articulation and onset/offset decisions already exist. It
does not train the vocal classifier or offset detector. The first training
run is a proof that articulation-conditioned envelopes can be learned before
connecting scat features to the bass synthesizer.
