# Scat Onset-Strength Preprocessing

## Goal

Produce a real-time `onset_strength(t)` for scat audio whose behavior matches the Bass-DDSP training target:

```text
continuous percussive evidence
with a short annotated/onset pulse as a minimum floor
```

The control should not be a Dirac delta and should not be formed by multiplying unrelated controls such as loudness and periodicity.

## What Bass-DDSP Learned

In `bass_ddsp/dataset.py`, the training target is:

```python
hpss_strength = normalized continuous percussive energy
onset_strength = maximum(hpss_strength, triangular_event_pulse)
```

The event pulse uses a `32 ms` width while the frame hop is `16 ms`. Therefore, the model sees a short multi-frame attack floor, not a single-frame impulse.

## Corrected Scat Path

HPSS must remain the primary `onset_strength(t)` source because that is what
Bass-DDSP saw during training. Complex novelty from vocal audio is not
distributionally equivalent to bass percussive energy and should not replace
HPSS merely because it produces visible peaks.

Use the same HPSS-style extraction for scat:

```text
audio frame
  -> STFT with the Bass-DDSP-compatible hop
  -> HPSS percussive component
  -> percussive magnitude/energy
  -> normalization matched to Bass-DDSP training
  -> continuous hpss_strength(t)
```

In parallel, run:

```text
audio frame
  -> aubio.onset("complex")
  -> peak picker + threshold + silence check + minimum IOI
  -> delay-compensated event timestamp
```

The Bass-DDSP control should therefore be:

```text
onset_strength(t) = normalized HPSS percussive energy
```

Complex/HFC aubio events remain separate diagnostic candidates for future
note-boundary detection. They must not be injected into `onset_strength(t)`.

Do not add a Complex-derived pulse to this control at this stage. The scat
front end has no bass-aligned onset annotation from which to construct the
training fallback pulse, and aubio events are not reliable enough to be used
as transient amplitude targets. Keep them available for later note-boundary
experiments only.

## Why Aubio Events and Novelty Differ

`aubio.specdesc("complex")` returns a continuous novelty descriptor. It is expected to contain extra peaks that are not note onsets.

`aubio.onset("complex")` internally performs:

```text
phase-vocoder analysis
optional whitening/compression
Complex descriptor
peak picking
adaptive thresholding
silence rejection
minimum inter-onset interval
delay compensation
```

The aubio source documents its output as `0` for no accepted event and `1 + a` for an accepted event, where `a` is a sub-hop timing offset. This output is an event marker, not a strength or probability. See `aubio/src/onset/onset.h` and `aubio/src/onset/onset.c`.

Both `specdesc` and `onset` can run incrementally frame by frame. The detector has algorithmic latency determined by its analysis window, peak-picking history, and configured delay; use `get_last_s()` for the reported event time.

## Control Contract

Keep these signals separate:

| Signal | Meaning | Bass-DDSP use |
|---|---|---|
| `complex_strength(t)` | continuous acoustic change evidence | diagnostic only |
| `complex_event(t)` | sparse accepted onset decision | note-boundary candidate only |
| `onset_strength(t)` | continuous strength with causal pulse floor | decoder `onset_strength` input |
| `loudness(t)` | performance energy | sustain/performance amplitude |
| `periodicity(t)` | TorchCREPE confidence | harmonic indicator |

Do not multiply Complex novelty by loudness or periodicity. If noise suppression is needed, use aubio's silence threshold and causal normalization before constructing the pulse floor.

## Limitations

Complex onset detection can still miss vowel-only emphasis and fast consonant changes. Those cases should later be handled by a causal articulation/restart classifier. Raising the onset threshold alone will not solve ambiguous vowel-only boundaries; it will mostly remove more weak events.

## Recommendation

Adopt this as the first scat preprocessing baseline:

```text
HPSS percussive energy -> Bass-DDSP-compatible normalization
                       -> onset_strength(t)

Complex/HFC aubio      -> diagnostic event candidates only
```

This preserves the training distribution instead of treating vocal Complex
novelty spikes as bass attack energy. A learned scat-specific mapping can be
added later after comparing HPSS behavior on vocal recordings.
