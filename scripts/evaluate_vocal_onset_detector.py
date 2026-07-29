#!/usr/bin/env python3
"""Evaluate the DSP-only vocal onset path without claiming unlabeled accuracy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from vocal_controls import (
    VocalControlConfig,
    align_length,
    apply_gate_release_hold,
    apply_retrigger_hold,
    calibrated_noise_gate,
    compute_frame_features,
    extract_aubio_onsets,
    list_audio_inputs,
    load_audio_file,
    load_selected_audio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/workspace/learn/voice_inputs"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--noise-margins", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.0])
    parser.add_argument("--aubio-thresholds", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--retrigger-ms", type=float, nargs="+", default=[64.0, 80.0, 100.0, 120.0])
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("/workspace/runs") / f"vocal_onset_dsp_evaluation_{stamp}"


def greedy_match_count(reference: np.ndarray, candidate: np.ndarray, tolerance: float) -> int:
    """Count one-to-one temporal matches without treating either detector as truth."""
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    candidate = np.sort(np.asarray(candidate, dtype=np.float64))
    used = np.zeros(len(candidate), dtype=bool)
    matches = 0
    for event in reference:
        choices = np.flatnonzero((~used) & (np.abs(candidate - event) <= tolerance))
        if choices.size:
            best = choices[np.argmin(np.abs(candidate[choices] - event))]
            used[best] = True
            matches += 1
    return matches


def event_rows(
    recording: str,
    accepted: np.ndarray,
    times: np.ndarray,
    rms_db: np.ndarray,
    threshold_db: float,
    aubio_result: dict,
) -> list[dict]:
    decision_times = np.asarray(aubio_result["complex_decision_times"])
    reported_times = np.asarray(aubio_result["complex_event_times"])
    rows = []
    for event_index, frame_index in enumerate(np.flatnonzero(accepted > 0.5), start=1):
        callback_time = float(times[frame_index])
        pair_index = int(np.argmin(np.abs(decision_times - callback_time))) if decision_times.size else -1
        reported_time = float(reported_times[pair_index]) if pair_index >= 0 else callback_time
        rows.append(
            {
                "recording": recording,
                "event_index": event_index,
                "reported_acoustic_time_s": reported_time,
                "causal_callback_time_s": callback_time,
                "estimated_latency_ms": (callback_time - reported_time) * 1000.0,
                "callback_rms_db": float(rms_db[frame_index]),
                "gate_threshold_db": threshold_db,
                "gate_margin_db": float(rms_db[frame_index] - threshold_db),
                "is_true_onset": "",
                "corrected_onset_time_s": "",
                "comment": "",
            }
        )
    return rows


def evaluate_recording(path: Path, config: VocalControlConfig) -> tuple[dict, list[dict], dict]:
    raw_y, raw_sr, _ = load_audio_file(path, config.sample_rate)
    loaded = load_selected_audio(config, path)
    y = loaded["y"]
    features = compute_frame_features(y, config)
    times = features["times"]
    frame_count = len(times)
    aubio_result = extract_aubio_onsets(y, config)
    complex_events = align_length(aubio_result["complex_onset"], frame_count)
    hfc_events = align_length(aubio_result["hfc_onset"], frame_count)
    raw_gate, noise_peak_db, threshold_db = calibrated_noise_gate(
        features["rms_db"], times, config.noise_profile_seconds, config.noise_gate_margin_db
    )
    effective_gate = apply_gate_release_hold(
        raw_gate, config.noise_gate_release_seconds, config.hop_seconds
    )
    masked = (complex_events * effective_gate).astype(np.float32)
    accepted = apply_retrigger_hold(
        masked, config.onset_retrigger_hold_seconds, config.hop_seconds
    )

    complex_decisions = np.asarray(aubio_result["complex_decision_times"])
    complex_reported = np.asarray(aubio_result["complex_event_times"])
    latencies_ms = (complex_decisions - complex_reported) * 1000.0
    accepted_times = times[accepted > 0.5]
    iois = np.diff(accepted_times)
    complex_reported_all = np.asarray(aubio_result["complex_event_times"])
    hfc_reported_all = np.asarray(aubio_result["hfc_event_times"])
    agreement = greedy_match_count(complex_reported_all, hfc_reported_all, tolerance=0.050)
    raw_gate_rises = int(np.sum(np.diff(np.pad(raw_gate > 0.5, (1, 0))) == 1))
    held_gate_rises = int(np.sum(np.diff(np.pad(effective_gate > 0.5, (1, 0))) == 1))
    initial_events = int(np.sum(accepted_times < config.noise_profile_seconds))

    metrics = {
        "recording": path.name,
        "raw_duration_s": len(raw_y) / raw_sr,
        "processed_duration_s": len(y) / config.sample_rate,
        "trimmed_duration_s": max(0.0, len(raw_y) / raw_sr - len(y) / config.sample_rate),
        "frames": frame_count,
        "noise_peak_db": noise_peak_db,
        "gate_threshold_db": threshold_db,
        "raw_gate_active_fraction": float(np.mean(raw_gate)),
        "effective_gate_active_fraction": float(np.mean(effective_gate)),
        "raw_gate_rises": raw_gate_rises,
        "effective_gate_rises": held_gate_rises,
        "complex_raw_events": int(np.sum(complex_events)),
        "hfc_raw_events": int(np.sum(hfc_events)),
        "complex_rejected_by_gate": int(np.sum((complex_events > 0.5) & (effective_gate <= 0.5))),
        "complex_masked_events": int(np.sum(masked)),
        "accepted_after_retrigger": int(np.sum(accepted)),
        "accepted_in_initial_profile": initial_events,
        "event_rate_per_s": float(np.sum(accepted) / max(len(y) / config.sample_rate, 1e-8)),
        "median_ioi_ms": float(np.median(iois) * 1000.0) if iois.size else np.nan,
        "minimum_ioi_ms": float(np.min(iois) * 1000.0) if iois.size else np.nan,
        "latency_median_ms": float(np.median(latencies_ms)) if latencies_ms.size else np.nan,
        "latency_mean_ms": float(np.mean(latencies_ms)) if latencies_ms.size else np.nan,
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms.size else np.nan,
        "latency_max_ms": float(np.max(latencies_ms)) if latencies_ms.size else np.nan,
        "complex_hfc_matches_50ms": agreement,
        "complex_hfc_agreement_fraction": agreement / max(len(complex_reported_all), 1),
    }
    diagnostics = {
        "features": features,
        "aubio": aubio_result,
        "complex_events": complex_events,
        "effective_gate": effective_gate,
    }
    return metrics, event_rows(
        path.name, accepted, times, features["rms_db"], threshold_db, aubio_result
    ), diagnostics


def run_sensitivity(
    path: Path,
    base_config: VocalControlConfig,
    diagnostics: dict,
    noise_margins: list[float],
    aubio_thresholds: list[float],
    retrigger_ms: list[float],
) -> list[dict]:
    features = diagnostics["features"]
    times = features["times"]
    frame_count = len(times)
    rows = []
    for aubio_threshold in aubio_thresholds:
        config = replace(base_config, aubio_threshold=aubio_threshold)
        loaded = load_selected_audio(config, path)
        aubio_result = extract_aubio_onsets(loaded["y"], config)
        events = align_length(aubio_result["complex_onset"], frame_count)
        for noise_margin in noise_margins:
            raw_gate, _, _ = calibrated_noise_gate(
                features["rms_db"], times, config.noise_profile_seconds, noise_margin
            )
            gate = apply_gate_release_hold(
                raw_gate, config.noise_gate_release_seconds, config.hop_seconds
            )
            masked = events * gate
            for hold_ms in retrigger_ms:
                accepted = apply_retrigger_hold(masked, hold_ms / 1000.0, config.hop_seconds)
                rows.append(
                    {
                        "recording": path.name,
                        "aubio_threshold": aubio_threshold,
                        "noise_margin_db": noise_margin,
                        "retrigger_hold_ms": hold_ms,
                        "raw_complex_events": int(np.sum(events)),
                        "accepted_events": int(np.sum(accepted)),
                    }
                )
    return rows


def make_plots(metrics: pd.DataFrame, sensitivity: pd.DataFrame, output_dir: Path) -> None:
    labels = metrics["recording"]
    x = np.arange(len(metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 1.5 * width, metrics["complex_raw_events"], width, label="Complex raw")
    ax.bar(x - 0.5 * width, metrics["complex_masked_events"], width, label="After gate")
    ax.bar(x + 0.5 * width, metrics["accepted_after_retrigger"], width, label="After 80 ms hold")
    ax.bar(x + 1.5 * width, metrics["hfc_raw_events"], width, label="HFC diagnostic", alpha=0.7)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Event count")
    ax.set_title("DSP onset event counts (not ground-truth accuracy)")
    ax.legend(ncols=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "event_counts.png", dpi=180)
    plt.close(fig)

    finite_latency = metrics["latency_median_ms"].dropna()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, metrics["latency_median_ms"], label="Median")
    ax.scatter(x, metrics["latency_p95_ms"], color="black", marker="_", s=180, label="P95")
    ax.set_ylabel("Estimated aubio callback latency (ms)")
    ax.set_title("Callback time minus aubio-reported acoustic time")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    if finite_latency.empty:
        ax.text(0.5, 0.5, "No Complex events", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "latency.png", dpi=180)
    plt.close(fig)

    aggregate = (
        sensitivity.groupby(["aubio_threshold", "noise_margin_db"], as_index=False)["accepted_events"]
        .sum()
        .pivot(index="noise_margin_db", columns="aubio_threshold", values="accepted_events")
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(aggregate.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(aggregate.columns)), [f"{v:.2f}" for v in aggregate.columns])
    ax.set_yticks(np.arange(len(aggregate.index)), [f"{v:.0f}" for v in aggregate.index])
    ax.set_xlabel("Aubio Complex threshold")
    ax.set_ylabel("Noise gate margin (dB)")
    ax.set_title("Accepted events across all files (80 ms retrigger)")
    for row in range(aggregate.shape[0]):
        for col in range(aggregate.shape[1]):
            ax.text(col, row, int(aggregate.iloc[row, col]), ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Accepted event count")
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_sensitivity.png", dpi=180)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, sensitivity: pd.DataFrame, output_dir: Path) -> None:
    total_raw = int(metrics["complex_raw_events"].sum())
    total_masked = int(metrics["complex_masked_events"].sum())
    total_accepted = int(metrics["accepted_after_retrigger"].sum())
    rejected = int(metrics["complex_rejected_by_gate"].sum())
    median_latency = float(metrics["latency_median_ms"].median())
    max_trim = float(metrics["trimmed_duration_s"].max())
    recording_stems = metrics["recording"].map(lambda name: Path(name).stem)
    duplicated_stems = sorted(recording_stems[recording_stems.duplicated(keep=False)].unique())
    default_sensitivity = sensitivity[
        (np.isclose(sensitivity["retrigger_hold_ms"], 80.0))
    ]
    totals = default_sensitivity.groupby(["aubio_threshold", "noise_margin_db"])["accepted_events"].sum()
    sensitivity_range = (int(totals.min()), int(totals.max())) if len(totals) else (0, 0)
    table_columns = [
        "recording",
        "processed_duration_s",
        "complex_raw_events",
        "complex_rejected_by_gate",
        "accepted_after_retrigger",
        "raw_gate_rises",
        "latency_median_ms",
    ]
    table = metrics[table_columns].round(2).to_markdown(index=False)
    report = f"""# DSP Vocal Onset Detector Evaluation

## Scope

This evaluates the current **DSP-only onset path**:

`audio -> aubio Complex onset -> calibrated RMS gate veto -> 80 ms retrigger hold`

It does not evaluate articulation classification, which is not implemented. It also does not
use TorchCREPE. The complete vocal-control pipeline's offset state machine is therefore outside
this audit because its periodicity input comes from a neural pitch estimator.

## Main Result

- Recordings: **{len(metrics)}**
- Raw aubio Complex callbacks: **{total_raw}**
- Rejected by the effective noise gate: **{rejected}**
- Events after gate masking: **{total_masked}**
- Events after the retrigger hold: **{total_accepted}**
- Median estimated aubio callback latency across recordings: **{median_latency:.1f} ms**
- Accepted-event range over the tested aubio/gate settings: **{sensitivity_range[0]} to {sensitivity_range[1]}**

These are detector diagnostics, not accuracy. There are no human onset labels for these
recordings, so precision, recall, F1, false-positive rate, and timing error cannot honestly be
computed yet.
{"- **Duplicate-source caution:** " + ", ".join(duplicated_stems) + " occurs in multiple file formats, so aggregate event totals do not represent fully independent recordings." if duplicated_stems else ""}

## Per-Recording Diagnostics

{table}

`raw_gate_rises` is deliberately not treated as an onset count. A gate transition is only
permission for a Complex event to survive.

## Findings

1. **The onset front end is genuinely non-neural.** Aubio Complex, frame RMS, Boolean gate
   masking, gate release state, and retrigger suppression are deterministic DSP/state logic.
2. **The detector has algorithmic decision latency.** `latency.png` measures callback time
   minus aubio's backdated acoustic estimate. Backdating is suitable for offline plots, while
   callback time is the realizable causal output time.
3. **Parameter sensitivity is measurable but not correctness.** `parameter_sensitivity.png`
   shows whether event counts are fragile under plausible threshold changes. A stable count can
   still be consistently wrong.
4. **The loader can compromise noise calibration.** `load_selected_audio()` runs
   `librosa.effects.trim(top_db=45)` before taking the first 500 ms noise profile. Up to
   **{max_trim:.3f} s** was removed from a file in this set. The retained first 500 ms is not
   guaranteed to be the originally recorded ambient segment.
5. **Peak normalization does not invalidate a relative +10 dB gate by itself**, because signal
   and profile shift together. Trimming the calibration segment is the larger concern.
6. **Known qualitative failure modes remain:** rapid consonant sequences, weak vowel-only
   emphasis, `/l/` attacks, and consonant/vowel double triggers. These require labels to quantify.

## Files

- `metrics.csv`: one row per recording.
- `parameter_sensitivity.csv`: threshold/margin/retrigger sweep.
- `onset_event_review.csv`: accepted events with acoustic and causal timestamps plus blank
  annotation columns.
- `event_counts.png`: each stage's event count.
- `latency.png`: estimated causal aubio delay.
- `parameter_sensitivity.png`: event-count robustness, not accuracy.

## Minimum Honest Accuracy Evaluation

Annotate `is_true_onset` in `onset_event_review.csv`, and also add any missed events to a
separate ground-truth onset list. Reviewing detector outputs alone can measure precision but
cannot reveal false negatives; a full independent onset annotation is required for recall/F1.
Use a timing tolerance such as +/-50 ms and report both event F1 and median absolute timing
error. Keep acoustic-time scoring separate from real-time callback latency.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = VocalControlConfig(input_dir=args.input_dir)
    files = sorted(list_audio_inputs(args.input_dir), key=lambda path: path.name.lower())
    if not files:
        raise FileNotFoundError(f"No supported recordings found in {args.input_dir}")

    metric_rows: list[dict] = []
    review_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    for path in files:
        print(f"Evaluating {path.name} ...", flush=True)
        metrics, events, diagnostics = evaluate_recording(path, config)
        metric_rows.append(metrics)
        review_rows.extend(events)
        sensitivity_rows.extend(
            run_sensitivity(
                path,
                config,
                diagnostics,
                args.noise_margins,
                args.aubio_thresholds,
                args.retrigger_ms,
            )
        )

    metrics_df = pd.DataFrame(metric_rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    sensitivity_df.to_csv(output_dir / "parameter_sensitivity.csv", index=False)
    review_columns = [
        "recording",
        "event_index",
        "reported_acoustic_time_s",
        "causal_callback_time_s",
        "estimated_latency_ms",
        "callback_rms_db",
        "gate_threshold_db",
        "gate_margin_db",
        "is_true_onset",
        "corrected_onset_time_s",
        "comment",
    ]
    with (output_dir / "onset_event_review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_columns)
        writer.writeheader()
        writer.writerows(review_rows)
    make_plots(metrics_df, sensitivity_df, output_dir)
    write_report(metrics_df, sensitivity_df, output_dir)
    print(f"Evaluation written to {output_dir}")


if __name__ == "__main__":
    main()
