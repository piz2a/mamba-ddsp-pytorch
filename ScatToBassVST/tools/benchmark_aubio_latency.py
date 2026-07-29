#!/usr/bin/env python3
"""Measure aubio callback delay separately from detector execution time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import aubio
import numpy as np

import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from vocal_controls import load_audio_file


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values):
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50) if values else None,
        "p95": percentile(values, 95) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def analyze(audio, sample_rate, window, hop, reference_onsets=None):
    detector = aubio.onset("complex", window, hop, sample_rate)
    detector.set_threshold(0.30)
    detector.set_silence(-45.0)
    detector.set_minioi_s(0.080)
    frame_count = int(np.ceil(len(audio) / hop))
    padded = np.pad(audio.astype(np.float32), (0, frame_count * hop - len(audio)))
    delays_ms = []
    compute_us = []
    reported = []
    callbacks = []
    for index in range(frame_count):
        chunk = aubio.fvec(padded[index * hop : (index + 1) * hop])
        started = time.perf_counter_ns()
        decision = detector(chunk)
        compute_us.append((time.perf_counter_ns() - started) / 1000.0)
        if bool(decision[0]):
            callback_time = (index + 1) * hop / sample_rate
            reported_time = float(detector.get_last_s())
            callbacks.append(callback_time)
            reported.append(reported_time)
            delays_ms.append((callback_time - reported_time) * 1000.0)
    acoustic_delays_ms = []
    if reference_onsets is not None:
        remaining_callbacks = list(callbacks)
        for onset in reference_onsets:
            candidates = [
                callback
                for callback in remaining_callbacks
                if onset <= callback <= onset + 0.2
            ]
            if candidates:
                callback = min(candidates)
                acoustic_delays_ms.append((callback - onset) * 1000.0)
                remaining_callbacks.remove(callback)

    return {
        "window_samples": window,
        "hop_samples": hop,
        "window_ms": 1000.0 * window / sample_rate,
        "hop_ms": 1000.0 * hop / sample_rate,
        "aubio_configured_timestamp_delay_ms": float(detector.get_delay_s() * 1000.0),
        "event_count": len(delays_ms),
        "callback_minus_reported_ms": summarize(delays_ms),
        "callback_minus_reference_onset_ms": (
            summarize(acoustic_delays_ms) if reference_onsets is not None else None
        ),
        "detector_compute_us_per_hop": {
            "mean": statistics.fmean(compute_us),
            "p50": percentile(compute_us, 50),
            "p95": percentile(compute_us, 95),
            "p99": percentile(compute_us, 99),
            "max": max(compute_us),
        },
        "reported_event_seconds": reported,
        "causal_callback_seconds": callbacks,
    }


def synthetic_signal(sample_rate):
    duration = 3.0
    audio = np.zeros(int(duration * sample_rate), dtype=np.float32)
    for onset, frequency in ((0.5, 130.0), (1.25, 180.0), (2.0, 95.0)):
        start = int(onset * sample_rate)
        count = int(0.45 * sample_rate)
        t = np.arange(count, dtype=np.float32) / sample_rate
        note = np.sin(2 * np.pi * frequency * t) * np.exp(-5.0 * t)
        note[: min(32, count)] += np.hanning(min(64, count))[: min(32, count)] * 0.8
        audio[start : start + count] += note
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "ScatToBassVST" / "docs" / "aubio_latency.json",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=[
            WORKSPACE / "learn" / "voice_inputs" / "Scat 1.wav",
            WORKSPACE / "learn" / "voice_inputs" / "Scat 5.m4a",
            WORKSPACE / "learn" / "voice_inputs" / "Scat 6.m4a",
        ],
    )
    args = parser.parse_args()
    sample_rate = 16000
    synthetic_onsets = [0.5, 1.25, 2.0]
    sources = [("synthetic", synthetic_signal(sample_rate), synthetic_onsets)]
    for path in args.inputs:
        if path.exists():
            audio, _, _ = load_audio_file(path, sample_rate)
            sources.append((path.name, audio, None))
    configurations = [(512, 256), (512, 128)]
    results = {
        name: {
            f"window_{window}_hop_{hop}": analyze(
                audio, sample_rate, window, hop, reference_onsets
            )
            for window, hop in configurations
        }
        for name, audio, reference_onsets in sources
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
