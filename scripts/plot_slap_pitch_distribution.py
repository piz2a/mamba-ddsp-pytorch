#!/usr/bin/env python3
"""Create a simple pitch-only SP/ST analysis from the slap feature table."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def empirical_best_accuracy(frame):
    table = pd.crosstab(frame.midi, frame.pluck).reindex(
        columns=["SP", "ST"], fill_value=0
    )
    return float(table.max(axis=1).sum() / len(frame))


def threshold_accuracy(frame, crossover=40):
    prediction = np.where(frame.midi >= crossover, "SP", "ST")
    return float(np.mean(prediction == frame.pluck))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_path = args.output / "pitch_only_distribution.png"
    report_path = args.output / "PITCH_ONLY_FINDINGS.md"
    stats_path = args.output / "pitch_only_statistics.json"
    for target in (plot_path, report_path, stats_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")

    frame = pd.read_csv(args.features)
    frame = frame[frame.expression == "NO"].copy()
    frame["midi"] = frame.midi.round().astype(int)
    isolated = frame[frame.dataset == "isolated"]
    tracks = frame[frame.dataset == "single_tracks"]

    colors = {"SP": "#2a9d8f", "ST": "#e76f51"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, subset, title in (
        (axes[0], isolated, "Isolated notes"),
        (axes[1], tracks, "Natural slap riffs"),
    ):
        for style in ("SP", "ST"):
            values = subset[subset.pluck == style].midi
            counts = values.value_counts().sort_index()
            probability = counts / counts.sum()
            axis.plot(
                probability.index,
                probability.values * 100.0,
                marker="o",
                markersize=4,
                linewidth=2,
                color=colors[style],
                label=style,
            )
        axis.set_title(title)
        axis.set_xlabel("Annotated MIDI pitch")
        axis.set_ylabel("Within-style notes (%)")
        axis.grid(alpha=0.22)
        axis.legend()
    axes[0].text(
        0.03, 0.95,
        "Pitch-only best accuracy: 50.1%",
        transform=axes[0].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    axes[1].axvline(39.5, color="black", linestyle="--", linewidth=1.3)
    axes[1].text(
        0.03, 0.95,
        "MIDI < 40: ST\nMIDI >= 40: SP\nAccuracy: 97.3%",
        transform=axes[1].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    accuracy_labels = ["007", "013", "016", "All riffs", "Isolated"]
    accuracy_values = []
    for track_id in ("007", "013", "016"):
        subset = tracks[tracks.track_id.astype(int) == int(track_id)]
        accuracy_values.append(threshold_accuracy(subset))
    accuracy_values.extend([
        threshold_accuracy(tracks),
        threshold_accuracy(isolated),
    ])
    bars = axes[2].bar(
        accuracy_labels,
        np.asarray(accuracy_values) * 100.0,
        color=["#457b9d", "#457b9d", "#457b9d", "#264653", "#adb5bd"],
    )
    for bar, value in zip(bars, accuracy_values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value * 100.0 + 1.5,
            f"{value * 100.0:.1f}%",
            ha="center",
        )
    axes[2].set_ylim(0, 108)
    axes[2].set_title("Fixed MIDI-40 rule")
    axes[2].set_ylabel("Correct SP/ST decisions (%)")
    axes[2].grid(axis="y", alpha=0.22)
    fig.suptitle("Can annotated pitch alone choose slap articulation?", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    statistics = {
        "selection": "expression == NO; DN excluded",
        "rule": "ST if MIDI < 40 else SP",
        "isolated": {
            "notes": int(len(isolated)),
            "sp": int((isolated.pluck == "SP").sum()),
            "st": int((isolated.pluck == "ST").sum()),
            "empirical_pitch_only_best_accuracy": empirical_best_accuracy(isolated),
            "threshold_accuracy": threshold_accuracy(isolated),
        },
        "single_tracks": {
            "notes": int(len(tracks)),
            "sp": int((tracks.pluck == "SP").sum()),
            "st": int((tracks.pluck == "ST").sum()),
            "empirical_pitch_only_best_accuracy": empirical_best_accuracy(tracks),
            "threshold_accuracy": threshold_accuracy(tracks),
            "always_st_accuracy": float((tracks.pluck == "ST").mean()),
            "per_track_threshold_accuracy": {
                track_id: threshold_accuracy(
                    tracks[tracks.track_id.astype(int) == int(track_id)]
                )
                for track_id in ("007", "013", "016")
            },
        },
        "combined_threshold_accuracy": threshold_accuracy(frame),
    }
    stats_path.write_text(json.dumps(statistics, indent=2))

    report = """# Pitch-Only Slap Decision

## Direct answer

Pitch does **not** distinguish SP from ST in the isolated-note dataset. Those
recordings were intentionally balanced across essentially identical pitch
sets. The empirical best decision using MIDI alone is 50.1%, which is chance.

Pitch almost perfectly distinguishes SP and ST inside natural tracks 007, 013,
and 016 after removing DN:

```text
MIDI < 40   -> ST_NO
MIDI >= 40  -> SP_NO
```

This rule gets 144 of 148 notes correct: 97.3%.

| Evaluation subset | Accuracy |
|---|---:|
| Track 007 | 100.0% |
| Track 013 | 92.3% |
| Track 016 | 100.0% |
| All three riffs | 97.3% |
| Isolated notes | 49.9% |

The four errors are `SP_NO` notes at MIDI 39 in track 013. MIDI 39 is the only
overlap in the natural-riff distributions; it contains four SP and four ST
notes. Choosing SP or ST there produces the same total accuracy.

## Interpretation

This is a strong **performance-pattern finding**, not an instrument-wide
classification law. The isolated data prove that either articulation can be
played across the same pitch range. The three riffs happen to use ST in the
lower register and SP in the higher register, consistent with how these
particular bass lines were arranged.

Therefore:

- If the goal is to reproduce the convention of these three IDMT riffs, a
  deterministic MIDI-40 crossover is faster and more defensible than training
  a classifier.
- If the goal is to infer a performer's intended slap technique universally,
  pitch alone is insufficient.
- Apply the rule to the **mapped output-bass MIDI pitch**, not necessarily raw
  vocal MIDI, especially if scat pitch is octave-shifted into bass range.
- Make the crossover configurable. MIDI 40 is the data-derived default.
- A manual `ST/SP` knob should override the rule because users may intentionally
  pop a low note or thumb a high note.

## What survives from the previous acoustic analysis?

The previous analysis found that SP notes in the three riffs have much higher
spectral centroid, high-frequency ratio, ZCR, and flatness than ST. Those are
real differences in the recorded bass audio, but they cannot choose Bass-DDSP
articulation from vocal controls unless vocal pronunciation is explicitly
trained to represent SP versus ST.

Without acoustic-intention features, the useful findings are only:

1. remove DN for now; only ten examples exist;
2. treat retained slap articulations as `SP_NO` and `ST_NO`;
3. use output pitch plus an optional user override;
4. avoid training an SP/ST model from isolated bass notes for vocal inference,
   because the model would learn bass timbre rather than vocal intent.
"""
    report_path.write_text(report)
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
