#!/usr/bin/env python3
"""Analyze SP/ST slap notes in IDMT isolated notes and natural slap riffs."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import chi2_contingency, fisher_exact, pointbiserialr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bass_ddsp.dataset import (
    IDMTBassRiffDataset,
    IDMTBassSingleTrackDataset,
    parse_idmt_bass_note,
)


SAMPLE_RATE = 16000
SLAP_STYLES = ("SP", "ST")
RETAINED_EXPRESSIONS = ("NO", "DN")
TRACK_IDS = ("007", "013", "016")
FEATURES = (
    "early_rms_db",
    "early_crest_db",
    "early_zcr",
    "early_centroid_hz",
    "early_hf_ratio_3k",
    "early_flatness",
    "attack_time_ms",
    "decay_150ms_db",
)


def load_mono(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(audio, dtype=np.float32)


def spectral_features(audio):
    if audio.size < 512:
        audio = np.pad(audio, (0, 512 - audio.size))
    magnitude = np.abs(
        librosa.stft(audio, n_fft=512, hop_length=128, win_length=512, center=False)
    )
    power = magnitude * magnitude
    frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=512)
    total_power = float(power.sum()) + 1e-12
    return {
        "centroid_hz": float(librosa.feature.spectral_centroid(
            S=magnitude, sr=SAMPLE_RATE
        ).mean()),
        "bandwidth_hz": float(librosa.feature.spectral_bandwidth(
            S=magnitude, sr=SAMPLE_RATE
        ).mean()),
        "rolloff85_hz": float(librosa.feature.spectral_rolloff(
            S=magnitude, sr=SAMPLE_RATE, roll_percent=0.85
        ).mean()),
        "hf_ratio_3k": float(power[frequencies >= 3000.0].sum() / total_power),
        "flatness": float(librosa.feature.spectral_flatness(S=magnitude).mean()),
    }


def db(value):
    return 20.0 * math.log10(max(float(value), 1e-8))


def rms_region(audio, start_seconds, end_seconds):
    start = max(0, int(round(start_seconds * SAMPLE_RATE)))
    end = min(audio.size, int(round(end_seconds * SAMPLE_RATE)))
    if end <= start:
        return 1e-8
    return float(np.sqrt(np.mean(audio[start:end] ** 2) + 1e-12))


def extract_features(audio):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    early_samples = int(round(0.128 * SAMPLE_RATE))
    early = audio[:early_samples]
    if early.size < early_samples:
        early = np.pad(early, (0, early_samples - early.size))

    early_rms = float(np.sqrt(np.mean(early * early) + 1e-12))
    early_peak = float(np.max(np.abs(early))) if early.size else 0.0
    whole_rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    whole_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    early_spec = spectral_features(early)
    whole_spec = spectral_features(audio)

    frame_length = min(256, max(32, audio.size))
    rms_curve = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=64,
        center=False,
    )[0]
    search_frames = max(1, int(round(0.2 * SAMPLE_RATE / 64)))
    attack_index = int(np.argmax(rms_curve[:search_frames])) if rms_curve.size else 0
    attack_time_ms = attack_index * 64.0 * 1000.0 / SAMPLE_RATE
    onset_rms = rms_region(audio, 0.0, 0.032)
    later_rms = rms_region(audio, 0.096, 0.160)

    return {
        "duration_ms": audio.size * 1000.0 / SAMPLE_RATE,
        "rms_db": db(whole_rms),
        "peak_db": db(whole_peak),
        "crest_db": db(whole_peak / max(whole_rms, 1e-8)),
        "zcr": float(librosa.feature.zero_crossing_rate(
            early, frame_length=256, hop_length=128, center=False
        ).mean()),
        "attack_time_ms": attack_time_ms,
        "decay_150ms_db": db(later_rms / max(onset_rms, 1e-8)),
        "early_rms_db": db(early_rms),
        "early_crest_db": db(early_peak / max(early_rms, 1e-8)),
        "early_zcr": float(librosa.feature.zero_crossing_rate(
            early, frame_length=256, hop_length=128, center=False
        ).mean()),
        **{f"early_{key}": value for key, value in early_spec.items()},
        **whole_spec,
    }


def isolated_rows(root):
    helper = IDMTBassRiffDataset(
        data_location=root,
        sampling_rate=SAMPLE_RATE,
        block_size=256,
        signal_length=32768,
        examples_per_epoch=1,
        include_expression_styles=RETAINED_EXPRESSIONS,
        include_string_numbers=(1, 2, 3, 4, 5),
        periodicity_use_crepe_confidence=False,
        pitch_source="labels",
        cache_size=1,
    )
    rows = []
    for note in helper.notes:
        if note.pluck not in SLAP_STYLES:
            continue
        audio, trim = helper._load_note_audio(note)
        tokens = note.path.stem.split("_")
        row = {
            "dataset": "isolated",
            "track_id": "",
            "note_index": "",
            "path": str(note.path),
            "pluck": note.pluck,
            "expression": note.expression,
            "articulation": note.articulation,
            "midi": 69.0 + 12.0 * math.log2(note.frequency / 440.0),
            "frequency_hz": note.frequency,
            "string": note.string,
            "fret": note.fret,
            "bass_id": tokens[1] if len(tokens) > 3 else "",
            "eq_id": tokens[3] if len(tokens) > 3 else "",
            "modulation_frequency_hz": 0.0,
            "modulation_range_cents": 0.0,
            "trim_start_ms": trim["trim_start_sample"] * 1000.0 / SAMPLE_RATE,
            "trim_end_ms": trim["trim_end_sample"] * 1000.0 / SAMPLE_RATE,
        }
        row.update(extract_features(audio))
        rows.append(row)
    return rows


def read_track_events(notes_path):
    return IDMTBassSingleTrackDataset._read_single_track_events(notes_path)


def track_rows(root):
    rows = []
    all_annotations = []
    for track_id in TRACK_IDS:
        audio_path = root / "audio" / f"{track_id}.wav"
        notes_path = root / "misc" / "notes_csv" / f"{track_id}_note_parameters.csv"
        audio = load_mono(audio_path)
        events = read_track_events(notes_path)
        for note_index, event in enumerate(events):
            annotation = {
                "dataset": "single_tracks",
                "track_id": track_id,
                "note_index": note_index,
                **event,
            }
            all_annotations.append(annotation)
            if (
                event["pluck"] not in SLAP_STYLES
                or event["expression"] not in RETAINED_EXPRESSIONS
            ):
                continue
            start = max(0, int(round(event["start_seconds"] * SAMPLE_RATE)))
            end = min(audio.size, int(round(event["end_seconds"] * SAMPLE_RATE)))
            segment = audio[start:end]
            row = {
                **annotation,
                "path": f"{audio_path}#note-{note_index:04d}",
                "frequency_hz": 440.0 * 2.0 ** ((event["midi"] - 69.0) / 12.0),
                "bass_id": "",
                "eq_id": "",
                "trim_start_ms": 0.0,
                "trim_end_ms": (end - start) * 1000.0 / SAMPLE_RATE,
            }
            row.update(extract_features(segment))
            rows.append(row)
    return rows, all_annotations


def cramers_v(table):
    chi2, p_value, _, _ = chi2_contingency(table, correction=False)
    n = table.sum()
    denominator = max(1, min(table.shape) - 1)
    return math.sqrt(chi2 / (n * denominator)), p_value


def label_statistics(frame):
    stats = {}
    for source in ("isolated", "single_tracks", "combined"):
        subset = frame if source == "combined" else frame[frame.dataset == source]
        table = pd.crosstab(subset["pluck"], subset["expression"]).reindex(
            index=SLAP_STYLES, columns=RETAINED_EXPRESSIONS, fill_value=0
        )
        values = table.to_numpy()
        try:
            association, p_value = cramers_v(values)
        except ValueError:
            association, p_value = float("nan"), float("nan")
        odds_ratio, fisher_p = fisher_exact(values)
        stats[source] = {
            "counts": {
                pluck: {
                    expression: int(table.loc[pluck, expression])
                    for expression in RETAINED_EXPRESSIONS
                }
                for pluck in SLAP_STYLES
            },
            "cramers_v": association,
            "chi_square_p": p_value,
            "fisher_odds_ratio": float(odds_ratio),
            "fisher_p": float(fisher_p),
        }
    return stats


def feature_statistics(frame):
    output = {}
    for source in ("isolated", "single_tracks", "combined"):
        subset = frame if source == "combined" else frame[frame.dataset == source]
        target = (subset.pluck == "ST").astype(float).to_numpy()
        output[source] = {}
        for feature in FEATURES:
            values = subset[feature].to_numpy(dtype=float)
            correlation, p_value = pointbiserialr(target, values)
            sp = subset.loc[subset.pluck == "SP", feature].to_numpy(dtype=float)
            st = subset.loc[subset.pluck == "ST", feature].to_numpy(dtype=float)
            pooled = math.sqrt(
                ((sp.size - 1) * np.var(sp, ddof=1) + (st.size - 1) * np.var(st, ddof=1))
                / max(1, sp.size + st.size - 2)
            )
            effect = (np.mean(st) - np.mean(sp)) / max(pooled, 1e-12)
            output[source][feature] = {
                "point_biserial_r": float(correlation),
                "p_value": float(p_value),
                "cohens_d_st_minus_sp": float(effect),
                "sp_median": float(np.median(sp)),
                "st_median": float(np.median(st)),
            }
    return output


def plot_label_distribution(frame, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, source in zip(axes[:2], ("isolated", "single_tracks")):
        subset = frame[frame.dataset == source]
        table = pd.crosstab(subset.pluck, subset.expression).reindex(
            index=SLAP_STYLES, columns=RETAINED_EXPRESSIONS, fill_value=0
        )
        bottom = np.zeros(2)
        for expression, color in zip(RETAINED_EXPRESSIONS, ("#247ba0", "#e4572e")):
            values = table[expression].to_numpy()
            axis.bar(SLAP_STYLES, values, bottom=bottom, label=expression, color=color)
            for index, value in enumerate(values):
                if value:
                    axis.text(index, bottom[index] + value / 2, str(value),
                              ha="center", va="center", color="white", fontweight="bold")
            bottom += values
        axis.set_title("Isolated notes" if source == "isolated" else "Tracks 007, 013, 016")
        axis.set_ylabel("Retained note count")
        axis.legend(title="Expression")
        axis.grid(axis="y", alpha=0.2)

    track_counts = pd.crosstab(
        [frame[frame.dataset == "single_tracks"].track_id,
         frame[frame.dataset == "single_tracks"].pluck],
        frame[frame.dataset == "single_tracks"].expression,
    ).reindex(columns=RETAINED_EXPRESSIONS, fill_value=0)
    labels = [f"{track}-{pluck}" for track, pluck in track_counts.index]
    bottom = np.zeros(len(labels))
    for expression, color in zip(RETAINED_EXPRESSIONS, ("#247ba0", "#e4572e")):
        values = track_counts[expression].to_numpy()
        axes[2].bar(labels, values, bottom=bottom, label=expression, color=color)
        bottom += values
    axes[2].set_title("Natural-riff composition")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_ylabel("Note count")
    axes[2].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "label_distribution.png", dpi=180)
    plt.close(fig)


def plot_feature_distributions(frame, output):
    labels = [("isolated", "SP"), ("isolated", "ST"),
              ("single_tracks", "SP"), ("single_tracks", "ST")]
    colors = ["#2a9d8f", "#e76f51", "#74c69d", "#f4a261"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    display_names = {
        "early_rms_db": "Early RMS (dBFS)",
        "early_crest_db": "Early crest factor (dB)",
        "early_zcr": "Early ZCR",
        "early_centroid_hz": "Early centroid (Hz)",
        "early_hf_ratio_3k": "Early >3 kHz energy ratio",
        "early_flatness": "Early spectral flatness",
        "attack_time_ms": "RMS peak time (ms)",
        "decay_150ms_db": "96-160 ms / onset RMS (dB)",
    }
    rng = np.random.default_rng(20260729)
    for axis, feature in zip(axes.flat, FEATURES):
        groups = [
            frame.loc[(frame.dataset == source) & (frame.pluck == pluck), feature]
            .to_numpy(dtype=float)
            for source, pluck in labels
        ]
        boxes = axis.boxplot(groups, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for index, values in enumerate(groups, start=1):
            if values.size > 250:
                values = rng.choice(values, 250, replace=False)
            jitter = rng.normal(index, 0.045, values.size)
            axis.scatter(jitter, values, s=7, alpha=0.22, color=colors[index - 1])
        axis.set_xticks(range(1, 5), ["Iso SP", "Iso ST", "Riff SP", "Riff ST"])
        axis.set_title(display_names[feature])
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Every retained slap note contributes to distributions; points are display-subsampled only", y=1.01)
    fig.tight_layout()
    fig.savefig(output / "acoustic_feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(feature_stats, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    y = np.arange(len(FEATURES))
    for offset, source, color in (
        (-0.18, "isolated", "#277da1"),
        (0.18, "single_tracks", "#f3722c"),
    ):
        values = [feature_stats[source][feature]["point_biserial_r"] for feature in FEATURES]
        axes[0].barh(y + offset, values, height=0.32, label=source, color=color)
        effects = [feature_stats[source][feature]["cohens_d_st_minus_sp"] for feature in FEATURES]
        axes[1].barh(y + offset, effects, height=0.32, label=source, color=color)
    short_names = [
        "early RMS", "crest", "ZCR", "centroid", ">3 kHz ratio",
        "flatness", "attack time", "150 ms decay",
    ]
    for axis, title, xlabel in (
        (axes[0], "Point-biserial correlation with ST=1", "r"),
        (axes[1], "Standardized effect: ST minus SP", "Cohen's d"),
    ):
        axis.set_yticks(y, short_names)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "feature_style_correlations.png", dpi=180)
    plt.close(fig)


def plot_pca(frame, output):
    matrix = frame.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median())
    projected = PCA(n_components=2, random_state=20260729).fit_transform(
        StandardScaler().fit_transform(matrix)
    )
    fig, axis = plt.subplots(figsize=(9, 6.5))
    styles = {
        ("isolated", "SP"): ("#2a9d8f", "o"),
        ("isolated", "ST"): ("#e76f51", "o"),
        ("single_tracks", "SP"): ("#006d77", "^"),
        ("single_tracks", "ST"): ("#9b2226", "^"),
    }
    for (source, pluck), (color, marker) in styles.items():
        mask = (frame.dataset == source) & (frame.pluck == pluck)
        axis.scatter(projected[mask, 0], projected[mask, 1], s=22, alpha=0.45,
                     color=color, marker=marker, label=f"{source} {pluck}")
    dn = (frame.expression == "DN").to_numpy()
    axis.scatter(projected[dn, 0], projected[dn, 1], s=90, facecolors="none",
                 edgecolors="black", linewidths=1.3, label="DN annotation")
    axis.set_title("PCA of causal 128 ms acoustic descriptors")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "feature_pca.png", dpi=180)
    plt.close(fig)


def plot_track_timelines(annotations, output):
    frame = pd.DataFrame(annotations)
    fig, axes = plt.subplots(3, 1, figsize=(15, 6.8), sharex=False)
    colors = {"SP": "#2a9d8f", "ST": "#e76f51"}
    hatches = {"NO": "", "DN": "//", "VI": "xx", "SL": ".."}
    for axis, track_id in zip(axes, TRACK_IDS):
        subset = frame[frame.track_id == track_id]
        for _, row in subset.iterrows():
            axis.broken_barh(
                [(row.start_seconds, row.end_seconds - row.start_seconds)],
                (0.1, 0.8),
                facecolors=colors[row.pluck],
                hatch=hatches.get(row.expression, ""),
                edgecolors="black" if row.expression != "NO" else colors[row.pluck],
                linewidth=0.7,
            )
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_title(f"Track {track_id}: SP teal, ST orange; hatch marks non-NO expression")
        axis.set_xlabel("Time (s)")
        axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "slap_track_timelines.png", dpi=180)
    plt.close(fig)


def plot_modulation(annotations, output):
    frame = pd.DataFrame(annotations)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    expressions = sorted(frame.expression.unique())
    positions = {expression: index for index, expression in enumerate(expressions)}
    colors = {"SP": "#2a9d8f", "ST": "#e76f51"}
    for pluck in SLAP_STYLES:
        subset = frame[frame.pluck == pluck]
        x = np.asarray([positions[value] for value in subset.expression], dtype=float)
        offset = -0.08 if pluck == "SP" else 0.08
        axes[0].scatter(x + offset, subset.modulation_frequency_hz,
                        color=colors[pluck], alpha=0.65, label=pluck)
        axes[1].scatter(x + offset, subset.modulation_range_cents,
                        color=colors[pluck], alpha=0.65, label=pluck)
    for axis, title, ylabel in (
        (axes[0], "Annotated modulation frequency", "Hz"),
        (axes[1], "Annotated modulation range", "cents"),
    ):
        axis.set_xticks(range(len(expressions)), expressions)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend()
        for expression in RETAINED_EXPRESSIONS:
            if expression in positions:
                axis.axvspan(positions[expression] - 0.35, positions[expression] + 0.35,
                             color="#90be6d", alpha=0.08)
    fig.suptitle("Green columns are retained NO/DN; all their modulation annotations are zero")
    fig.tight_layout()
    fig.savefig(output / "modulation_annotations.png", dpi=180)
    plt.close(fig)


def write_report(frame, annotations, label_stats, feature_stats, output):
    counts = Counter(zip(frame.dataset, frame.pluck, frame.expression))
    annotation_frame = pd.DataFrame(annotations)
    nonzero = annotation_frame[
        (annotation_frame.modulation_frequency_hz != 0)
        | (annotation_frame.modulation_range_cents != 0)
    ]
    ranked = sorted(
        FEATURES,
        key=lambda feature: abs(feature_stats["single_tracks"][feature]["cohens_d_st_minus_sp"]),
        reverse=True,
    )
    report = f"""# IDMT Slap-Note Analysis

## Scope

- Isolated dataset: `/disk1/ahnjiho/IDMT-SMT-BASS`
- Natural riffs: tracks 007, 013, and 016 from
  `/disk1/ahnjiho/IDMT-SMT-BASS-SINGLE-TRACKS`
- Plucking styles: `SP`, `ST`
- Retained expression labels: `NO`, `DN`
- Excluded from classifier analysis: `BE`, `VI`, `SL`, `HA`
- Acoustic analysis sample rate: {SAMPLE_RATE} Hz
- Causal feature window: first 128 ms of each annotated/trimmed note

## Label inventory

| Source | SP_NO | SP_DN | ST_NO | ST_DN | Total |
|---|---:|---:|---:|---:|---:|
| Isolated | {counts[("isolated", "SP", "NO")]} | {counts[("isolated", "SP", "DN")]} | {counts[("isolated", "ST", "NO")]} | {counts[("isolated", "ST", "DN")]} | {(frame.dataset == "isolated").sum()} |
| Tracks | {counts[("single_tracks", "SP", "NO")]} | {counts[("single_tracks", "SP", "DN")]} | {counts[("single_tracks", "ST", "NO")]} | {counts[("single_tracks", "ST", "DN")]} | {(frame.dataset == "single_tracks").sum()} |

The isolated corpus does not jointly annotate plucking and expression:
SP/ST are PS-category recordings and are therefore all `NO`. It provides no
evidence about SP_DN or ST_DN. In the three natural riffs, every retained DN
note is ST, but there are only {counts[("single_tracks", "ST", "DN")]} such
notes. A classifier must not learn “DN means ST” from this sampling artifact.

One isolated `SP_NO` file is annotated as string 5
(`BS_2_EQ_1_SP_NO_5_0.wav`). It is included here because this audit covers every
physical slap WAV, but it lies outside the four-string Bass-DDSP training
constraint and should be excluded from a strictly matched classifier split.

Track-only SP/ST versus NO/DN association:

- Cramer's V: {label_stats["single_tracks"]["cramers_v"]:.4f}
- Fisher exact p-value: {label_stats["single_tracks"]["fisher_p"]:.6g}
- Fisher odds ratio: {label_stats["single_tracks"]["fisher_odds_ratio"]}

These statistics describe this dataset, not a universal performance rule.

## Modulation annotations

There are {len(nonzero)} non-zero modulation annotations across all slap notes
in tracks 007/013/016. None belongs to retained `NO` or `DN`:

- track 007: two `ST_VI` notes, 4 Hz and 25 cents;
- track 013: four `SP_SL` notes, approximately 0.62-0.86 Hz and 100 cents;
- track 016: no non-zero modulation annotations.

Therefore no modulation-frequency or cents distribution exists for retained
NO/DN beyond a point mass at zero. `modulation_annotations.png` displays the
excluded non-zero labels and the zero-valued retained columns.

## Acoustic descriptors

Every retained note is recorded in `all_slap_note_features.csv`. Features use
the first 128 ms where possible, making them more relevant to a causal
classifier than whole-note averages. The strongest track-domain standardized
SP/ST differences, ranked by absolute Cohen's d, are:

"""
    for feature in ranked:
        stat = feature_stats["single_tracks"][feature]
        report += (
            f"- `{feature}`: d={stat['cohens_d_st_minus_sp']:.3f}, "
            f"r={stat['point_biserial_r']:.3f}, "
            f"SP median={stat['sp_median']:.4g}, "
            f"ST median={stat['st_median']:.4g}\n"
        )
    report += """

Compare isolated and natural-riff panels before treating any descriptor as
robust. Separation that reverses between sources is likely recording-domain,
instrument, player, EQ, duration, or segmentation bias.

## Interpretation Before Choosing a Classifier

1. The task supported by these data is primarily `SP` versus `ST`, not a
   general articulation classifier.
2. The natural-riff test domain is small: 158 retained notes from only three
   recordings. Random note-level train/test splits would leak track identity.
3. Evaluation must split by track, player/instrument/EQ where possible.
4. If a few causal descriptors separate SP/ST consistently in both domains, a
   shallow decision tree, logistic regression, or RBF/linear SVM is defensible.
5. If separation is strongly nonlinear but stable across tracks, use a small
   temporal model on frame sequences. Do not choose it merely because PCA
   overlaps; PCA is an unsupervised linear projection.
6. Because articulation rarely changes inside the 17 riff performances, a
   user knob remains the safest default. A slap-only SP/ST classifier can be an
   optional mode with temporal hysteresis rather than replacing that control.

No classifier was trained here. These outputs are intended to support that
architecture decision without contaminating the analysis with note-level
cross-validation leakage.

## Files

- `all_slap_note_features.csv`: one row per retained isolated/riff note.
- `all_track_slap_annotations.csv`: all slap annotations, including excluded
  VI and SL notes and their modulation values.
- `label_statistics.json`: SP/ST x NO/DN contingency statistics.
- `feature_statistics.json`: per-source correlations, effect sizes, and medians.
- `label_distribution.png`: articulation-count distributions.
- `acoustic_feature_distributions.png`: every-note feature distributions.
- `feature_style_correlations.png`: SP/ST correlation and effect size.
- `feature_pca.png`: unsupervised 128 ms feature projection.
- `slap_track_timelines.png`: all notes in tracks 007/013/016.
- `modulation_annotations.png`: modulation frequency/range audit.
"""
    (output / "REPORT.md").write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--isolated-root",
        type=Path,
        default=Path("/disk1/ahnjiho/IDMT-SMT-BASS"),
    )
    parser.add_argument(
        "--tracks-root",
        type=Path,
        default=Path("/disk1/ahnjiho/IDMT-SMT-BASS-SINGLE-TRACKS"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    rows = isolated_rows(args.isolated_root)
    track_feature_rows, annotations = track_rows(args.tracks_root)
    rows.extend(track_feature_rows)
    frame = pd.DataFrame(rows)
    annotation_frame = pd.DataFrame(annotations)
    frame.to_csv(args.output / "all_slap_note_features.csv", index=False)
    annotation_frame.to_csv(args.output / "all_track_slap_annotations.csv", index=False)

    label_stats = label_statistics(frame)
    feature_stats = feature_statistics(frame)
    with (args.output / "label_statistics.json").open("w") as handle:
        json.dump(label_stats, handle, indent=2)
    with (args.output / "feature_statistics.json").open("w") as handle:
        json.dump(feature_stats, handle, indent=2)

    plot_label_distribution(frame, args.output)
    plot_feature_distributions(frame, args.output)
    plot_correlations(feature_stats, args.output)
    plot_pca(frame, args.output)
    plot_track_timelines(annotations, args.output)
    plot_modulation(annotations, args.output)
    write_report(frame, annotations, label_stats, feature_stats, args.output)
    print(frame.groupby(["dataset", "pluck", "expression"]).size())
    print(f"Saved {len(frame)} retained notes to {args.output}")


if __name__ == "__main__":
    main()
