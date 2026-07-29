#!/usr/bin/env python3
"""Rebuild the final comparison from preserved, validated model renders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from bass_ddsp.compare_models import (  # noqa: E402
    _plot_loss_curves,
    _plot_metric_bars,
    _read_loss_tail,
    _spectrogram_db,
    _summarize,
)


DEFAULT_JULY24 = WORKSPACE_ROOT / "runs/model_comparison_branchbalance_fixedresume_dwts_vanilla_20260724_043513"
DEFAULT_JULY26 = WORKSPACE_ROOT / "runs/model_comparison_branchbalance_final_dwts_resumed_20260726_071231"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "runs/model_comparison_branchbalance_corrected_preserved_bass_full_dwts_20260729"

BASS_LABEL = "Bass-DDSP fixed resume"
DWTS_LABEL = "Vanilla DWTS resumed"
DDSP_LABEL = "Vanilla DDSP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--july24-dir", type=Path, default=DEFAULT_JULY24)
    parser.add_argument("--july26-dir", type=Path, default=DEFAULT_JULY26)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-plots", type=int, default=4)
    return parser.parse_args()


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)


def audio_sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest()


def assert_audio_equal(first: Path, second: Path, label: str) -> dict:
    a, sr_a = load_audio(first)
    b, sr_b = load_audio(second)
    if sr_a != sr_b:
        raise RuntimeError(f"{label}: sample-rate mismatch: {sr_a} != {sr_b}")
    if a.shape != b.shape or not np.array_equal(a, b):
        maximum = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
        raise RuntimeError(
            f"{label}: preserved audio differs between comparisons; "
            f"shapes={a.shape}/{b.shape}, max_abs_diff={maximum}"
        )
    return {
        "samples": int(a.size),
        "sample_rate": sr_a,
        "float32_audio_sha256": audio_sha256(a),
    }


def frame_rms(audio: np.ndarray, frame_size: int = 256) -> np.ndarray:
    usable = audio.size - audio.size % frame_size
    frames = audio[:usable].reshape(-1, frame_size)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def plot_sample(path: Path, title: str, files: list[tuple[str, Path, str]]) -> None:
    loaded = []
    for label, audio_path, color in files:
        audio, sr = load_audio(audio_path)
        loaded.append((label, audio, sr, color))
    sample_rates = {entry[2] for entry in loaded}
    if len(sample_rates) != 1:
        raise RuntimeError(f"{title}: inconsistent sample rates: {sample_rates}")
    sr = loaded[0][2]
    length = min(len(entry[1]) for entry in loaded)
    times = np.arange(length) / float(sr)

    fig, axes = plt.subplots(
        len(loaded) + 2,
        1,
        figsize=(15, 4.0 + 2.0 * len(loaded)),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.2] + [1.1] * len(loaded) + [1.0]},
    )
    for label, audio, _, color in loaded:
        axes[0].plot(times, audio[:length], color=color, linewidth=0.55, alpha=0.82, label=label)
    axes[0].set_title(f"{title}: waveform")
    axes[0].set_xlim(0, length / float(sr))
    axes[0].set_ylabel("amplitude")
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncols=len(loaded), fontsize=8)

    for axis, (label, audio, _, _) in zip(axes[1:-1], loaded):
        db, freqs, stft_times = _spectrogram_db(audio[:length], sr, 256)
        image = axis.pcolormesh(
            stft_times,
            freqs,
            db,
            shading="auto",
            cmap="magma",
            vmin=-80,
            vmax=0,
        )
        axis.set_title(f"{label} STFT")
        axis.set_xlim(0, length / float(sr))
        axis.set_ylim(0, min(2200, sr / 2))
        axis.set_ylabel("Hz")
        fig.colorbar(image, ax=axis, label="dB")

    rms_axis = axes[-1]
    for label, audio, _, color in loaded:
        rms = frame_rms(audio[:length])
        rms_times = (np.arange(len(rms)) * 256 + 128) / float(sr)
        rms_axis.plot(rms_times, rms, color=color, linewidth=1.0, label=label)
    rms_axis.set_title("Frame RMS envelope")
    rms_axis.set_xlim(0, length / float(sr))
    rms_axis.set_xlabel("seconds")
    rms_axis.set_ylabel("RMS")
    rms_axis.grid(alpha=0.2)
    rms_axis.legend(ncols=len(loaded), fontsize=8)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def validate_metric_identity(july24: pd.DataFrame, july26: pd.DataFrame) -> None:
    old = july24[july24["model"] == DDSP_LABEL].sort_values(["sample_position", "index"])
    new = july26[july26["model"] == DDSP_LABEL].sort_values(["sample_position", "index"])
    if old[["sample_position", "index"]].reset_index(drop=True).equals(
        new[["sample_position", "index"]].reset_index(drop=True)
    ) is False:
        raise RuntimeError("Vanilla DDSP sample identities changed between comparisons")
    numeric = [
        column
        for column in old.columns
        if column not in {"model"} and np.issubdtype(old[column].dtype, np.number)
    ]
    if not np.allclose(
        old[numeric].to_numpy(),
        new[numeric].to_numpy(),
        rtol=0.0,
        atol=1e-7,
        equal_nan=True,
    ):
        raise RuntimeError("Vanilla DDSP metrics changed between comparisons")


def winner(summary: dict, key: str, direction: str) -> str:
    values = {label: summary[label][key] for label in summary}
    values = {label: value for label, value in values.items() if np.isfinite(value)}
    if direction == "lower":
        return min(values, key=values.get)
    if direction == "higher":
        return max(values, key=values.get)
    return min(values, key=lambda label: abs(values[label] - 1.0))


def fmt(value: float, digits: int) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    path: Path,
    summary: dict,
    loss_summary: dict,
    output_dir: Path,
    validation: dict,
    july24_dir: Path,
    july26_dir: Path,
) -> None:
    metrics = [
        ("MSS loss", "lower", "mss_mean", 4),
        ("LSD dB", "lower", "lsd_db_mean", 3),
        ("Frame RMS ratio", "close to 1", "rms_ratio_mean", 3),
        ("Frame RMS correlation", "higher", "rms_corr_mean", 3),
        ("F0 median cents", "lower", "f0_median_cents_mean", 1),
        ("Gross pitch error %", "lower", "gross_pitch_error_pct_mean", 2),
        ("Onset HF log error", "lower", "onset_hf_log_error_mean", 3),
        ("Onset energy ratio", "close to 1", "onset_energy_ratio_mean", 3),
    ]
    labels = [BASS_LABEL, DWTS_LABEL, DDSP_LABEL]
    lines = [
        "# Corrected Three-Way DDSP Model Comparison",
        "",
        "This is the corrected final comparison. It preserves the valid July 24 "
        "Bass-DDSP renders and combines them with the fully trained July 26 "
        "Vanilla DWTS renders.",
        "",
        "## Correction",
        "",
        f"- Valid Bass-DDSP source: `{july24_dir}`",
        f"- Fully trained DWTS source: `{july26_dir}`",
        "- Vanilla DDSP: numerically identical in both source comparisons.",
        "- Samples: 32, using the same fixed indices and target audio.",
        "- No checkpoint was reloaded to create this correction.",
        "- No previous run or comparison output was overwritten.",
        "",
        "The July 26 report was invalid for Bass-DDSP: it named the same checkpoint "
        "but produced different Bass audio after the model implementation changed. "
        "Resuming DWTS should never change Bass-DDSP.",
        "",
        "## Validation",
        "",
        f"- Target pairs checked: **{validation['target_pairs_checked']}**, all sample-identical.",
        f"- Vanilla DDSP pairs checked: **{validation['vanilla_pairs_checked']}**, all sample-identical.",
        f"- Shared sample rate: **{validation['sample_rate']} Hz**.",
        "- July 24 and July 26 use the same ordered sample indices.",
        "- July 24 Bass audio is used unchanged; July 26 resumed-DWTS audio is used unchanged.",
        "",
        "## Objective Metrics",
        "",
        "| Metric | Direction | Bass-DDSP | Vanilla DWTS 200k | Vanilla DDSP 200k | Winner |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, direction, key, digits in metrics:
        values = [fmt(summary[label][key], digits) for label in labels]
        lines.append(
            "| " + " | ".join([name, direction, *values, winner(summary, key, direction)]) + " |"
        )

    lines.extend(
        [
            "",
            "## Training Tail",
            "",
            "| Model | Steps | Tail loss | Tail spectral | Tail RMS |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in labels:
        tail = loss_summary[label]
        lines.append(
            f"| {label} | {fmt(tail.get('steps', np.nan), 0)} | "
            f"{fmt(tail.get('tail_loss_mean', np.nan), 4)} | "
            f"{fmt(tail.get('tail_spectral_loss_mean', np.nan), 4)} | "
            f"{fmt(tail.get('tail_rms_loss_mean', np.nan), 4)} |"
        )

    lines.extend(
        [
            "",
            "## Branch Diagnostics",
            "",
            "| Model | Sustain RMS / signal % | Noise RMS / signal % | Transient RMS / signal % |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in labels:
        item = summary[label]
        lines.append(
            f"| {label} | {fmt(item['sustain_rms_vs_signal_pct_mean'], 2)} | "
            f"{fmt(item['noise_rms_vs_signal_pct_mean'], 2)} | "
            f"{fmt(item['transient_rms_vs_signal_pct_mean'], 2)} |"
        )

    lines.extend(
        [
            "",
            "## Correct Interpretation",
            "",
            "With the correct preserved Bass renders, Bass-DDSP has the best MSS, LSD, "
            "RMS ratio, RMS correlation, onset HF error, and onset energy ratio. "
            "The fully trained Vanilla DWTS has the lowest gross pitch error. The "
            "differences between Bass-DDSP and DWTS are small for onset HF error, "
            "but the July 26 conclusion that Bass-DDSP had collapsed loudness was "
            "an evaluation artifact.",
            "",
            "This corrected result replaces the objective table in "
            "`model_comparison_branchbalance_final_dwts_resumed_20260726_071231`. "
            "That directory is retained as evidence and must not be used as the "
            "final Bass-DDSP comparison.",
            "",
            "## Artifacts",
            "",
            "- [`metric_bars.png`](metric_bars.png)",
            "- [`loss_curves.png`](loss_curves.png)",
            "- [`per_sample_metrics.csv`](per_sample_metrics.csv)",
            "- [`summary.json`](summary.json)",
            "",
            "Each sample directory contains target, corrected Bass-DDSP, resumed "
            "Vanilla DWTS, Vanilla DDSP, and a concatenated listening WAV. The first "
            "four also contain waveform, STFT, and frame-RMS plots.",
            "",
        ]
    )
    for sample_dir in sorted(output_dir.glob("sample_*"))[:4]:
        lines.append(f"- [`{sample_dir.name}`]({sample_dir.name}/comparison.png)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    july24_dir = args.july24_dir.resolve()
    july26_dir = args.july26_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    with (july24_dir / "summary.json").open() as handle:
        july24_summary = json.load(handle)
    with (july26_dir / "summary.json").open() as handle:
        july26_summary = json.load(handle)
    if july24_summary["indices"] != july26_summary["indices"]:
        raise RuntimeError("The two comparisons used different sample indices")

    old_metrics = pd.read_csv(july24_dir / "per_sample_metrics.csv")
    new_metrics = pd.read_csv(july26_dir / "per_sample_metrics.csv")
    validate_metric_identity(old_metrics, new_metrics)
    rows = pd.concat(
        [
            old_metrics[old_metrics["model"] == BASS_LABEL],
            new_metrics[new_metrics["model"] == DWTS_LABEL],
            old_metrics[old_metrics["model"] == DDSP_LABEL],
        ],
        ignore_index=True,
    )
    labels = [BASS_LABEL, DWTS_LABEL, DDSP_LABEL]
    summary = _summarize(rows.to_dict(orient="records"), labels)

    old_sample_dirs = sorted(july24_dir.glob("sample_*"))
    new_sample_dirs = sorted(july26_dir.glob("sample_*"))
    if [path.name for path in old_sample_dirs] != [path.name for path in new_sample_dirs]:
        raise RuntimeError("Sample directory names differ between source comparisons")

    validation_pairs = []
    colors = [
        ("target", "target.wav", "black"),
        (BASS_LABEL, "bass-ddsp_fixed_resume.wav", "#1f77b4"),
        (DWTS_LABEL, "vanilla_dwts_resumed.wav", "#2ca02c"),
        (DDSP_LABEL, "vanilla_ddsp.wav", "#ff7f0e"),
    ]
    for position, (old_dir, new_dir) in enumerate(zip(old_sample_dirs, new_sample_dirs)):
        target_validation = assert_audio_equal(
            old_dir / "target.wav", new_dir / "target.wav", f"{old_dir.name} target"
        )
        ddsp_validation = assert_audio_equal(
            old_dir / "vanilla_ddsp.wav",
            new_dir / "vanilla_ddsp.wav",
            f"{old_dir.name} Vanilla DDSP",
        )
        validation_pairs.append(
            {
                "sample": old_dir.name,
                "target": target_validation,
                "vanilla_ddsp": ddsp_validation,
            }
        )

        sample_dir = output_dir / old_dir.name
        sample_dir.mkdir()
        source_files = {
            "target.wav": old_dir / "target.wav",
            "bass-ddsp_fixed_resume.wav": old_dir / "bass-ddsp_fixed_resume.wav",
            "vanilla_dwts_resumed.wav": new_dir / "vanilla_dwts_resumed.wav",
            "vanilla_ddsp.wav": old_dir / "vanilla_ddsp.wav",
        }
        for name, source in source_files.items():
            shutil.copy2(source, sample_dir / name)

        target, sr = load_audio(sample_dir / "target.wav")
        silence = np.zeros(int(0.25 * sr), dtype=np.float32)
        sequence = [target]
        for name in [
            "bass-ddsp_fixed_resume.wav",
            "vanilla_dwts_resumed.wav",
            "vanilla_ddsp.wav",
        ]:
            audio, audio_sr = load_audio(sample_dir / name)
            if audio_sr != sr:
                raise RuntimeError(f"{sample_dir.name}/{name}: sample-rate mismatch")
            sequence.extend([silence, audio])
        sf.write(
            sample_dir / "target_then_models.wav",
            np.concatenate(sequence),
            sr,
            subtype="FLOAT",
        )
        if position < args.num_plots:
            plot_sample(
                sample_dir / "comparison.png",
                old_dir.name,
                [(label, sample_dir / name, color) for label, name, color in colors],
            )

    rows.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    bass_run = WORKSPACE_ROOT / "runs/bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513"
    dwts_run = WORKSPACE_ROOT / "runs/vanilla_dwts_riff_resume_to_200k_20260726_071231"
    ddsp_run = WORKSPACE_ROOT / "runs/vanilla_ddsp_riff_20260720_090532"
    run_dirs = {
        BASS_LABEL: bass_run,
        DWTS_LABEL: dwts_run,
        DDSP_LABEL: ddsp_run,
    }
    loss_summary = {label: _read_loss_tail(run_dir) for label, run_dir in run_dirs.items()}
    validation = {
        "target_pairs_checked": len(validation_pairs),
        "vanilla_pairs_checked": len(validation_pairs),
        "sample_rate": validation_pairs[0]["target"]["sample_rate"],
        "pairs": validation_pairs,
    }
    result = {
        "status": "corrected_from_preserved_renders",
        "source_comparisons": {
            "july24_valid_bass": str(july24_dir),
            "july26_valid_resumed_dwts": str(july26_dir),
        },
        "models": {label: str(run_dir) for label, run_dir in run_dirs.items()},
        "indices": july24_summary["indices"],
        "summary": summary,
        "loss_summary": loss_summary,
        "validation": validation,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    _plot_metric_bars(output_dir / "metric_bars.png", summary)
    _plot_loss_curves(output_dir / "loss_curves.png", run_dirs)
    write_report(
        output_dir / "REPORT.md",
        summary,
        loss_summary,
        output_dir,
        validation,
        july24_dir,
        july26_dir,
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
