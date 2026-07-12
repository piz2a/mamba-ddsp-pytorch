# Voice Style Classifier Demo

Browser demo for immediate mic testing of:

- Vowels: `ah`, `uh`, `oh`, `woo`, `eu`, `ee`
- Syllables: `deu`, `dng`

Run it from `/workspace`:

```bash
python -m http.server 8000 -d voice-classifier-demo
```

Open `http://localhost:8000`.

The first prediction path is heuristic: RMS, ZCR, F0 autocorrelation, spectral ratios, and LPC-style F1/F2 formants. The capture buttons add short personal calibration samples and switch the live classifier toward a k-NN model stored in browser `localStorage`.

Suggested quick test:

1. Start mic.
2. Press each vowel button once or twice while sustaining that vowel.
3. Press `deu` and `dng` once or twice with the intended bass-mimic articulation.
4. Press `Validate` to run leave-one-out testing on the collected samples.
