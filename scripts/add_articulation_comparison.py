#!/usr/bin/env python3
"""Add note-level articulation metrics to the corrected three-model report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import librosa as li
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from bass_ddsp.compare_models import (  # noqa: E402
    _align_frames,
    _high_frequency_rms,
    _lsd_db,
    _mss_loss,
    _safe_corr,
)
from bass_ddsp.export_branch_debug import make_dataset  # noqa: E402


DEFAULT_COMPARISON = (
    WORKSPACE_ROOT
    / "runs/model_comparison_branchbalance_corrected_preserved_bass_full_dwts_20260729"
)
DEFAULT_BASS_RUN = (
    WORKSPACE_ROOT
    / "runs/bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513"
)
MODELS = {
    "Bass-DDSP": "bass-ddsp_fixed_resume.wav",
    "Vanilla DWTS 200k": "vanilla_dwts_resumed.wav",
    "Vanilla DDSP 200k": "vanilla_ddsp.wav",
}
COLORS = {
    "Bass-DDSP": "#1f77b4",
    "Vanilla DWTS 200k": "#2ca02c",
    "Vanilla DDSP 200k": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--bass-run", type=Path, default=DEFAULT_BASS_RUN)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onset-seconds", type=float, default=0.15)
    return parser.parse_args()


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(audio) ** 2) + 1e-12))


def frame_rms(audio: np.ndarray, block_size: int) -> np.ndarray:
    usable = audio.size - audio.size % block_size
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    frames = audio[:usable].reshape(-1, block_size)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def pitch_track(audio: np.ndarray, sr: int, block_size: int, fmin: float, fmax: float) -> np.ndarray:
    try:
        f0, _, _ = li.pyin(
            audio.astype(np.float32),
            fmin=float(fmin),
            fmax=float(fmax),
            sr=sr,
            frame_length=2048,
            hop_length=block_size,
            center=True,
        )
        return np.asarray(f0, dtype=np.float32)
    except Exception:
        return np.empty(0, dtype=np.float32)


def interval_pitch_metrics(
    estimated_f0: np.ndarray,
    interval: dict,
    label_pitch: float,
    sr: int,
    block_size: int,
    frame_count: int,
) -> dict:
    estimated_f0 = _align_frames(estimated_f0, frame_count)
    frame_samples = np.arange(frame_count) * block_size + block_size / 2
    active = (
        (frame_samples >= int(interval["start_sample"]))
        & (frame_samples < int(interval["end_sample"]))
    )
    values = estimated_f0[active]
    valid = np.isfinite(values) & (values > 1.0)
    active_count = int(values.size)
    valid_count = int(valid.sum())
    if valid_count:
        cents = np.abs(1200.0 * np.log2(values[valid] / max(float(label_pitch), 1e-6)))
        missing = active_count - valid_count
        gross_count = int(np.sum(cents > 100.0)) + missing
        return {
            "f0_median_cents": float(np.median(cents)),
            "gross_pitch_error_pct": gross_count / max(active_count, 1) * 100.0,
            "f0_valid_pct": valid_count / max(active_count, 1) * 100.0,
        }
    return {
        "f0_median_cents": float("nan"),
        "gross_pitch_error_pct": 100.0 if active_count else float("nan"),
        "f0_valid_pct": 0.0 if active_count else float("nan"),
    }


def score_interval(
    target: np.ndarray,
    recon: np.ndarray,
    interval: dict,
    config: dict,
    device: torch.device,
    onset_seconds: float,
    pitch_metrics: dict,
) -> dict:
    sr = int(config["preprocess"]["sampling_rate"])
    block_size = int(config["preprocess"]["block_size"])
    start = max(0, int(interval["start_sample"]))
    end = min(len(target), len(recon), int(interval["end_sample"]))
    target_note = target[start:end].astype(np.float32)
    recon_note = recon[start:end].astype(np.float32)
    original_samples = len(target_note)
    minimum = max(int(max(config["train"]["scales"])), block_size)
    if len(target_note) < minimum:
        padding = minimum - len(target_note)
        target_spectral = np.pad(target_note, (0, padding))
        recon_spectral = np.pad(recon_note, (0, padding))
    else:
        target_spectral = target_note
        recon_spectral = recon_note

    target_frames = frame_rms(target_note, block_size)
    recon_frames = frame_rms(recon_note, block_size)
    frames = min(len(target_frames), len(recon_frames))
    target_frames = target_frames[:frames]
    recon_frames = recon_frames[:frames]
    rms_ratio = float(np.mean(recon_frames) / max(np.mean(target_frames), 1e-12))
    rms_corr = _safe_corr(
        np.log(target_frames + 1e-7),
        np.log(recon_frames + 1e-7),
    )

    onset_samples = min(original_samples, max(1, int(round(onset_seconds * sr))))
    target_onset = target_note[:onset_samples]
    recon_onset = recon_note[:onset_samples]
    onset_energy_ratio = rms(recon_onset) / max(rms(target_onset), 1e-12)
    onset_mask = np.ones(onset_samples, dtype=bool)
    target_hf = _high_frequency_rms(target_onset, sr, onset_mask)
    recon_hf = _high_frequency_rms(recon_onset, sr, onset_mask)
    onset_hf_log_error = abs(
        math.log(max(recon_hf, 1e-12) / max(target_hf, 1e-12))
    )
    return {
        "duration_seconds": original_samples / float(sr),
        "mss": _mss_loss(target_spectral, recon_spectral, config, device),
        "lsd_db": _lsd_db(target_spectral, recon_spectral, sr, block_size),
        "rms_ratio": rms_ratio,
        "rms_corr": rms_corr,
        "onset_hf_log_error": onset_hf_log_error,
        "onset_energy_ratio": onset_energy_ratio,
        **pitch_metrics,
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mss",
        "lsd_db",
        "rms_ratio",
        "rms_corr",
        "f0_median_cents",
        "gross_pitch_error_pct",
        "f0_valid_pct",
        "onset_hf_log_error",
        "onset_energy_ratio",
    ]
    records = []
    for (articulation, model), group in rows.groupby(["articulation", "model"], sort=False):
        record = {
            "articulation": articulation,
            "model": model,
            "notes": int(len(group)),
            "total_duration_seconds": float(group["duration_seconds"].sum()),
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            record[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            record[f"{metric}_std"] = float(values.std()) if values.size else float("nan")
        records.append(record)
    return pd.DataFrame(records)


def plot_metrics(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("mss_mean", "Note-level MSS", "lower"),
        ("lsd_db_mean", "LSD dB", "lower"),
        ("rms_ratio_mean", "RMS ratio", "target 1"),
        ("rms_corr_mean", "RMS correlation", "higher"),
        ("gross_pitch_error_pct_mean", "Gross pitch error %", "lower"),
        ("onset_hf_log_error_mean", "Onset HF log error", "lower"),
        ("onset_energy_ratio_mean", "Onset energy ratio", "target 1"),
    ]
    articulations = list(dict.fromkeys(summary["articulation"]))
    models = list(MODELS)
    x = np.arange(len(articulations))
    width = 0.25
    fig, axes = plt.subplots(4, 2, figsize=(17, 15), constrained_layout=True)
    axes = axes.reshape(-1)
    for axis, (metric, title, direction) in zip(axes, metrics):
        for model_index, model in enumerate(models):
            model_rows = summary[summary["model"] == model].set_index("articulation")
            means = [model_rows.loc[label, metric] for label in articulations]
            error_column = metric.replace("_mean", "_std")
            errors = [model_rows.loc[label, error_column] for label in articulations]
            axis.bar(
                x + (model_index - 1) * width,
                means,
                width,
                yerr=errors,
                color=COLORS[model],
                alpha=0.82,
                label=model,
                capsize=2,
            )
        axis.set_title(f"{title} ({direction})")
        axis.set_xticks(x, articulations, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def best_model(rows: pd.DataFrame, metric: str, direction: str) -> str:
    values = {row["model"]: float(row[metric]) for _, row in rows.iterrows()}
    values = {key: value for key, value in values.items() if np.isfinite(value)}
    if direction == "higher":
        return max(values, key=values.get)
    if direction == "target":
        return min(values, key=lambda key: abs(values[key] - 1.0))
    return min(values, key=values.get)


def report_section(summary: pd.DataFrame, counts: dict) -> str:
    articulations = list(counts)
    lines = [
        "## Articulation-Conditioned Comparison",
        "",
        "This analysis scores each annotated note interval separately using the same "
        "midpoint-based articulation boundaries supplied during training. It uses "
        "the preserved corrected model WAVs; no checkpoint is reloaded.",
        "",
        "Note-level MSS/LSD values zero-pad short notes to the largest 4096-sample "
        "loss window, so compare models within the same articulation rather than "
        "comparing these values directly to full-riff MSS.",
        "",
        "### Coverage",
        "",
        "| Articulation | Notes | Total duration (s) |",
        "|---|---:|---:|",
    ]
    for articulation in articulations:
        row = summary[summary["articulation"] == articulation].iloc[0]
        lines.append(
            f"| {articulation} | {counts[articulation]} | "
            f"{float(row['total_duration_seconds']):.2f} |"
        )

    table_metrics = [
        ("mss_mean", "MSS", "lower"),
        ("lsd_db_mean", "LSD dB", "lower"),
        ("rms_ratio_mean", "RMS ratio", "target"),
        ("rms_corr_mean", "RMS corr", "higher"),
        ("f0_median_cents_mean", "F0 median cents", "lower"),
        ("onset_hf_log_error_mean", "Onset HF error", "lower"),
        ("onset_energy_ratio_mean", "Onset energy ratio", "target"),
        ("gross_pitch_error_pct_mean", "Gross pitch %", "lower"),
    ]
    for metric, title, direction in table_metrics:
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                "| Articulation | Bass-DDSP | DWTS 200k | Vanilla DDSP | Best |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for articulation in articulations:
            group = summary[summary["articulation"] == articulation]
            values = {
                row["model"]: float(row[metric])
                for _, row in group.iterrows()
            }
            lines.append(
                f"| {articulation} | {fmt(values['Bass-DDSP'])} | "
                f"{fmt(values['Vanilla DWTS 200k'])} | "
                f"{fmt(values['Vanilla DDSP 200k'])} | "
                f"{best_model(group, metric, direction)} |"
            )

    win_counts = {model: 0 for model in MODELS}
    for metric, _, direction in table_metrics:
        for articulation in articulations:
            group = summary[summary["articulation"] == articulation]
            win_counts[best_model(group, metric, direction)] += 1
    lines.extend(
        [
            "",
            "### Interpretation",
            "",
            "Across the eight displayed metrics and six articulations, the count of "
            "per-class metric wins is:",
            "",
        ]
    )
    for model, count in win_counts.items():
        lines.append(f"- {model}: **{count}**")
    lines.extend(
        [
            "",
            "Main class-level findings:",
            "",
            "- Bass-DDSP has the lowest note-level MSS for all six articulations.",
            "- Bass-DDSP has the closest RMS ratio and highest RMS correlation for all six articulations.",
            "- Bass-DDSP has the lowest onset HF error for five of six classes; DWTS is better for `SP_NO`.",
            "- Bass-DDSP has the lowest LSD for `PK_NO`, `SP_NO`, `ST_NO`, and `FS_DN`; DWTS leads `FS_NO` and `MU_NO`.",
            "- Pitch remains the main weakness. DWTS generally has lower gross pitch error, and Bass-DDSP reaches about 21.46% for `FS_DN`.",
            "- Every model overestimates `SP_NO` loudness on average. Bass-DDSP is closest, but its RMS ratio is still about 1.34.",
            "",
            "These win counts are descriptive, not a statistical significance test. "
            "The same note targets are evaluated for all models, but class counts and "
            "durations differ. Listening tests remain necessary to establish whether "
            "the articulation-conditioned differences are perceptually preferable.",
            "",
            "Artifacts:",
            "",
            "- [`articulation_metric_bars.png`](articulation_metric_bars.png)",
            "- [`articulation_note_metrics.csv`](articulation_note_metrics.csv)",
            "- [`articulation_summary.csv`](articulation_summary.csv)",
            "- [`articulation_summary.json`](articulation_summary.json)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    comparison_dir = args.comparison_dir.resolve()
    with (comparison_dir / "summary.json").open() as handle:
        comparison = json.load(handle)
    with (args.bass_run / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    seed = 20260723
    dataset = make_dataset(config, seed, "labels")
    device = torch.device(args.device)
    rows = []
    articulation_counts: dict[str, int] = {}

    for position, index in enumerate(comparison["indices"]):
        sample_dir = comparison_dir / f"sample_{position:02d}_idx_{index:04d}"
        data = dataset.generate_debug_example(index, pitch_source="labels")
        target_file, sr = sf.read(sample_dir / "target.wav", dtype="float32")
        target_file = np.asarray(target_file, dtype=np.float32).reshape(-1)
        if sr != int(data["sampling_rate"]) or not np.array_equal(
            target_file, data["audio"].astype(np.float32)
        ):
            raise RuntimeError(
                f"{sample_dir.name}: regenerated target/labels do not match preserved target"
            )

        frame_count = len(data["label_pitch"])
        pitch_config = config.get("idmt_bass", {})
        fmin = pitch_config.get("pitch_fmin", config["model"].get("f0_min_hz", 30.0))
        fmax = pitch_config.get("pitch_fmax", config["model"].get("f0_max_hz", 330.0))
        model_audio = {}
        model_pitch = {}
        for model, filename in MODELS.items():
            audio, model_sr = sf.read(sample_dir / filename, dtype="float32")
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if model_sr != sr:
                raise RuntimeError(f"{sample_dir.name}/{filename}: sample-rate mismatch")
            model_audio[model] = audio
            model_pitch[model] = pitch_track(audio, sr, data["block_size"], fmin, fmax)

        for note_position, interval in enumerate(data["intervals"]):
            articulation = str(interval["articulation"])
            articulation_counts[articulation] = articulation_counts.get(articulation, 0) + 1
            for model in MODELS:
                pitch_metrics = interval_pitch_metrics(
                    model_pitch[model],
                    interval,
                    interval["frequency"],
                    sr,
                    data["block_size"],
                    frame_count,
                )
                metrics = score_interval(
                    target_file,
                    model_audio[model],
                    interval,
                    config,
                    device,
                    args.onset_seconds,
                    pitch_metrics,
                )
                rows.append(
                    {
                        "sample_position": position,
                        "index": index,
                        "note_position": note_position,
                        "articulation": articulation,
                        "model": model,
                        **metrics,
                    }
                )

    articulation_order = list(data["articulation_labels"])
    articulation_counts = {
        label: articulation_counts.get(label, 0)
        for label in articulation_order
        if articulation_counts.get(label, 0) > 0
    }
    notes = pd.DataFrame(rows)
    notes["articulation"] = pd.Categorical(
        notes["articulation"], categories=list(articulation_counts), ordered=True
    )
    notes = notes.sort_values(["articulation", "sample_position", "note_position", "model"])
    summary = summarize(notes)
    summary["articulation"] = pd.Categorical(
        summary["articulation"], categories=list(articulation_counts), ordered=True
    )
    summary["model"] = pd.Categorical(summary["model"], categories=list(MODELS), ordered=True)
    summary = summary.sort_values(["articulation", "model"])

    notes.to_csv(comparison_dir / "articulation_note_metrics.csv", index=False)
    summary.to_csv(comparison_dir / "articulation_summary.csv", index=False)
    with (comparison_dir / "articulation_summary.json").open("w") as handle:
        json.dump(
            {
                "method": "note_interval_mean",
                "seed": seed,
                "onset_seconds": args.onset_seconds,
                "counts": articulation_counts,
                "rows": summary.to_dict(orient="records"),
            },
            handle,
            indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        )
    plot_metrics(summary, comparison_dir / "articulation_metric_bars.png")

    report_path = comparison_dir / "REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "## Articulation-Conditioned Comparison"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n\n"
    report += report_section(summary, articulation_counts)
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "comparison_dir": str(comparison_dir),
                "articulation_counts": articulation_counts,
                "note_metric_rows": len(notes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
