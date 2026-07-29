# Native Latency Report

## What Was Measured

The benchmark feeds complete causal hops into `aubio.onset("complex")`.
Callback time is the end of the hop in which aubio returns a nonzero onset.
This corrects the old Python diagnostic, which recorded the start of that hop
and understated callback availability by one hop.

Two quantities must not be confused:

1. **Callback minus aubio timestamp** measures how far aubio backdates an
   accepted event.
2. **Callback minus physical onset** measures useful response latency, but it
   requires trustworthy ground-truth onset labels. It is available here only
   for the synthetic attacks.

## Aubio Results

All results use 16 kHz audio, a 512-sample (32 ms) window, Complex onset
novelty, threshold `0.30`, silence `-45 dB`, and minimum IOI `80 ms`.

| Hop | Aubio timestamp delay | Synthetic physical-onset latency | Compute p50/hop |
|---|---:|---:|---:|
| 256 samples / 16 ms | 73.56 ms | 46.0 ms median, 44-48 ms | 16-30 us |
| 128 samples / 8 ms | 36.75 ms | 30.0 ms median, 20-32 ms | 16-23 us |

For the real recordings, where no hand labels exist, the 8 ms-hop callback
was `36.3-37.4 ms` later than aubio's backdated timestamp at the median:

| Recording | Events | 16 ms-hop median | 8 ms-hop median |
|---|---:|---:|---:|
| Scat 1 | 20 / 17 | 73.25 ms | 36.81 ms |
| Scat 5 | 8 / 9 | 68.97 ms | 37.44 ms |
| Scat 6 | 3 / 3 | 77.63 ms | 36.38 ms |

The event counts change with hop size, so lower latency is not automatically
better detection. The raw data are in `aubio_latency.json`, reproducible with:

```bash
python tools/benchmark_aubio_latency.py
```

## Why C++ Does Not Erase 50 ms

The detector's compute cost is roughly `0.02 ms` per hop. Its tens of
milliseconds of latency come primarily from spectral analysis, peak-picking
lookahead, and timestamp correction. Running aubio's C implementation inside
JUCE eliminates Python scheduling and crossing overhead; it does not remove
the causal evidence the algorithm waits for.

aubio is still useful because it provides a mature deterministic Complex
novelty and peak-picker implementation, stable real-time cost, explicit
threshold/minimum-IOI controls, and no onset-classifier training requirement.
The plugin uses the 8 ms hop, reducing aubio's configured delay from 73.56 to
36.75 ms.

## ONNX and Plugin Timing

Native smoke timing on the server's Intel Xeon Gold 6226R CPU:

| Component | Mean per 16 ms update |
|---|---:|
| tiny CREPE ONNX | 1.50 ms |
| recurrent Bass-DDSP controller ONNX | 0.38 ms |
| onset-envelope ONNX | 0.036 ms |

The cached wavetable and transient tables avoid recomputing the DCT bank.
Inference runs off the audio callback. The plugin currently reports a fixed
`64 ms` host latency, equal to the 1024-sample CREPE analysis frame at 16 kHz,
as the conservative maximum analysis horizon. The 80 ms retrigger guard and
80 ms minimum note age before periodicity offset are causal state rules, not
added lookahead latency.

Final production latency still needs measurement on the target MacBook with
its actual sample rate, host buffer size, DAW, ONNX Runtime architecture, and
audio driver.
