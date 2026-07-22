# Bass-DDSP v2 vs Vanilla DDSP Comparison

Generated on 2026-07-21 from the completed riff-stage checkpoints.

Detailed artifacts:

- Detailed report: `runs/model_comparison_bass_vs_vanilla_20260721_001/REPORT.md`
- Metrics CSV: `runs/model_comparison_bass_vs_vanilla_20260721_001/per_sample_metrics.csv`
- Metric bars: `runs/model_comparison_bass_vs_vanilla_20260721_001/metric_bars.png`
- Loss curves: `runs/model_comparison_bass_vs_vanilla_20260721_001/loss_curves.png`
- Example comparison plot: `runs/model_comparison_bass_vs_vanilla_20260721_001/sample_00_idx_0379/comparison.png`
- Example listening WAV: `runs/model_comparison_bass_vs_vanilla_20260721_001/sample_00_idx_0379/target_bass_vanilla.wav`

## Compared Runs

| Model | Run |
|---|---|
| Bass-DDSP v2 | `runs/bass_ddsp_v2_riff_20260720_023814` |
| Vanilla DDSP | `runs/vanilla_ddsp_riff_20260720_090532` |

Evaluation used 32 deterministic generated-riff samples with seed `20260721`.
Both models were evaluated against the same target audio and controls.

## Objective Result

| Metric | Direction | Bass-DDSP v2 | Vanilla DDSP | Winner |
|---|---:|---:|---:|---|
| Multi-scale STFT loss | lower | 5.6918 | 6.1611 | Bass-DDSP v2 |
| Log Spectral Distance, dB | lower | 12.294 | 13.804 | Bass-DDSP v2 |
| Frame RMS ratio | close to 1 | 0.983 | 0.837 | Bass-DDSP v2 |
| Frame RMS correlation | higher | 0.916 | 0.803 | Bass-DDSP v2 |
| F0 median error, cents | lower | 0.8 | 2.1 | Bass-DDSP v2 |
| Gross pitch error, percent | lower | 8.20 | 9.35 | Bass-DDSP v2 |
| Onset high-frequency log error | lower | 0.511 | 0.542 | Bass-DDSP v2 |
| Onset energy ratio | close to 1 | 0.963 | 0.660 | Bass-DDSP v2 |

## Training Tail

Both riff-stage runs trained for 200,000 steps.

| Metric | Bass-DDSP v2 | Vanilla DDSP |
|---|---:|---:|
| Tail total loss | 5.7907 | 6.4923 |
| Tail spectral loss | 5.5125 | 6.0381 |
| Tail RMS loss | 0.2782 | 0.4541 |

## Branch Diagnosis

| Branch RMS / final signal | Bass-DDSP v2 | Vanilla DDSP |
|---|---:|---:|
| sustain | 99.99% | 100.00% |
| noise | 0.06% | 0.30% |
| transient | 0.49% | 0.00% |

The comparison is favorable to Bass-DDSP v2 overall, especially for loudness envelope and onset energy.
However, the current Bass-DDSP v2 output is still sustain-dominated. The explicit transient branch exists, but its RMS contribution is only about 0.49% of the final signal on this evaluation set. That means the better onset metric is mostly coming from the sustain branch and branch-gain/loudness behavior, not from a strong dedicated transient synthesizer yet.

## Interpretation

Bass-DDSP v2 is currently the better checkpoint by objective metrics:

- It reconstructs the spectrum better.
- It tracks the target loudness envelope better.
- It is much closer to the target onset energy.
- It is slightly better on pitch tracking.

The main unresolved issue is architectural, not just training length:

- The transient branch remains too quiet.
- Noise is also nearly absent.
- The model still behaves mostly like a stronger sustain synthesizer with better conditioning.

Next evaluation should include a small listening test using the exported `target_bass_vanilla.wav` files. The objective metrics say Bass-DDSP v2 is better, but the final decision must check whether humans hear stronger bass attacks and more realistic articulation.

FAD was not computed in this pass. It should only be added after choosing a fixed embedding model and a larger held-out audio set; otherwise the number will be hard to interpret.
