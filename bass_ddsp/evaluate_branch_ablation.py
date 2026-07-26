import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bass_ddsp.compare_models import (
    _audio_metrics,
    _draw_intervals,
    _format_float,
    _frame_rms,
    _frame_times,
    _mean_std,
    _spectrogram_db,
    _write_csv,
)
from bass_ddsp.export_branch_debug import load_model, make_dataset, reconstruct


def _load_yaml(path):
    import yaml

    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _rms(audio):
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio * audio) + 1e-12))


def _peak(audio):
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def _safe_audio(branches, name, fallback):
    audio = branches.get(name)
    if audio is None:
        return np.zeros_like(fallback, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)


def _write_audio_set(sample_dir, data, audios):
    sr = int(data["sampling_rate"])
    gap = np.zeros(int(round(0.25 * sr)), dtype=np.float32)
    order = ["target", "full", "sustain_only", "transient", "noise", "transient_plus_noise"]
    concat = []
    for name in order:
        audio = np.asarray(audios[name], dtype=np.float32)
        sf.write(sample_dir / f"{name}.wav", audio, sr, subtype="FLOAT")
        if concat:
            concat.append(gap)
        concat.append(audio)
    sf.write(sample_dir / "target_full_sustain_transient_noise_residual.wav", np.concatenate(concat), sr, subtype="FLOAT")


def _plot_sample(sample_dir, sample_name, data, audios):
    sr = int(data["sampling_rate"])
    block_size = int(data["block_size"])
    target = audios["target"]
    n = min(len(v) for v in audios.values())
    audio_t = np.arange(n) / float(sr)
    frame_count = min(len(data["pitch"]), len(data["gate"]))
    frame_t = _frame_times(frame_count, sr, block_size)
    labels = data.get("articulation_labels", [])
    intervals = data.get("intervals", [])

    plot_order = [
        ("target", "black"),
        ("full", "#1f77b4"),
        ("sustain_only", "#2ca02c"),
        ("transient", "#d62728"),
        ("noise", "#9467bd"),
        ("transient_plus_noise", "#ff7f0e"),
    ]

    fig, axes = plt.subplots(
        9,
        1,
        figsize=(15, 20),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0, 1.15, 1.15, 1.15, 0.95, 0.85]},
    )

    ax = axes[0]
    _draw_intervals(ax, intervals, labels)
    for name, color in plot_order[:3]:
        ax.plot(audio_t, audios[name][:n], color=color, linewidth=0.55, alpha=0.86, label=name)
    ax.set_title(f"{sample_name}: target vs full vs sustain-only")
    ax.set_xlim(0, n / float(sr))
    ax.set_ylabel("amp")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", ncol=3, fontsize=8)

    ax = axes[1]
    _draw_intervals(ax, intervals, labels)
    for name, color in plot_order[3:]:
        ax.plot(audio_t, audios[name][:n], color=color, linewidth=0.55, alpha=0.86, label=name)
    ax.set_title("non-sustain branch audio")
    ax.set_xlim(0, n / float(sr))
    ax.set_ylabel("amp")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", ncol=3, fontsize=8)

    ax = axes[2]
    _draw_intervals(ax, intervals, labels)
    for name, color in plot_order[:4]:
        rms = _frame_rms(audios[name][:n], block_size)
        frames = min(len(rms), frame_count)
        ax.plot(frame_t[:frames], rms[:frames], color=color, linewidth=1.0, label=name)
    ax.set_title("frame RMS envelopes")
    ax.set_xlim(0, n / float(sr))
    ax.set_ylabel("RMS")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", ncol=4, fontsize=8)

    ax = axes[3]
    _draw_intervals(ax, intervals, labels)
    for name, color in plot_order[4:]:
        rms = _frame_rms(audios[name][:n], block_size)
        frames = min(len(rms), frame_count)
        ax.plot(frame_t[:frames], rms[:frames], color=color, linewidth=1.0, label=name)
    ax.set_title("noise/residual RMS envelopes")
    ax.set_xlim(0, n / float(sr))
    ax.set_ylabel("RMS")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", ncol=3, fontsize=8)

    for ax, name in zip(axes[4:7], ["target", "full", "sustain_only"]):
        db, freqs, times = _spectrogram_db(audios[name][:n], sr, block_size)
        image = ax.pcolormesh(times, freqs, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        _draw_intervals(ax, intervals, labels)
        ax.set_title(f"{name} STFT")
        ax.set_ylabel("Hz")
        ax.set_ylim(0, min(2200, sr / 2))
        ax.set_xlim(0, n / float(sr))
        fig.colorbar(image, ax=ax, label="dB")

    ax = axes[7]
    db, freqs, times = _spectrogram_db(audios["transient_plus_noise"][:n], sr, block_size)
    image = ax.pcolormesh(times, freqs, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
    _draw_intervals(ax, intervals, labels)
    ax.set_title("transient + noise STFT")
    ax.set_ylabel("Hz")
    ax.set_ylim(0, min(2200, sr / 2))
    ax.set_xlim(0, n / float(sr))
    fig.colorbar(image, ax=ax, label="dB")

    ax = axes[8]
    _draw_intervals(ax, intervals, labels)
    ax.plot(frame_t[:frame_count], data["pitch"][:frame_count], color="black", linewidth=1.0, label="label f0")
    ax2 = ax.twinx()
    ax2.plot(frame_t[:frame_count], data["onset_strength"][:frame_count], color="#d62728", linewidth=0.9, label="onset")
    ax2.plot(frame_t[:frame_count], data["gate"][:frame_count], color="#2ca02c", linewidth=0.9, label="gate")
    ax.set_title("controls")
    ax.set_ylabel("Hz")
    ax2.set_ylabel("control")
    ax.set_xlim(0, n / float(sr))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    fig.savefig(sample_dir / "branch_ablation.png", dpi=150)
    plt.close(fig)


def _write_branch_metrics(sample_dir, audios):
    full_rms = max(_rms(audios["full"]), 1e-12)
    rows = []
    for name, audio in audios.items():
        value = _rms(audio)
        rows.append({
            "name": name,
            "rms": value,
            "peak": _peak(audio),
            "rms_vs_full_percent": value / full_rms * 100.0,
        })
    _write_csv(sample_dir / "branch_metrics.csv", rows)
    return rows


def _write_report(path, args, run_dir, summary, rows, sample_dirs, dwts_note):
    labels = ["full", "sustain_only"]
    metric_rows = [
        ("MSS loss", "lower", "mss_mean", 4),
        ("LSD dB", "lower", "lsd_db_mean", 3),
        ("Frame RMS ratio", "close to 1", "rms_ratio_mean", 3),
        ("Frame RMS correlation", "higher", "rms_corr_mean", 3),
        ("F0 median cents", "lower", "f0_median_cents_mean", 1),
        ("Gross pitch error %", "lower", "gross_pitch_error_pct_mean", 2),
        ("Onset HF log error", "lower", "onset_hf_log_error_mean", 3),
        ("Onset energy ratio", "close to 1", "onset_energy_ratio_mean", 3),
    ]

    lines = [
        "# Bass-DDSP Frozen Branch Ablation",
        "",
        f"- Run: `{run_dir}`",
        f"- Evaluation samples: `{summary['full']['num_samples']}`",
        f"- Seed: `{args.seed}`",
        f"- Pitch source: `{args.pitch_source}`",
        f"- Onset metric window: `{args.onset_seconds}` seconds",
        "",
        "## Validity Note",
        "",
        "This evaluation freezes the trained Bass-DDSP model. No weights are updated. It compares the model's normal summed output against the same model's sustain branch alone.",
        "",
        dwts_note,
        "",
        "## Full Output vs Sustain-Only",
        "",
        "| Metric | Direction | Full model | Sustain only | Full minus sustain-only |",
        "|---|---|---:|---:|---:|",
    ]
    for name, direction, key, digits in metric_rows:
        full = summary["full"][key]
        sustain = summary["sustain_only"][key]
        delta = full - sustain if np.isfinite(full) and np.isfinite(sustain) else float("nan")
        lines.append(
            "| "
            + " | ".join([
                name,
                direction,
                _format_float(full, digits),
                _format_float(sustain, digits),
                _format_float(delta, digits),
            ])
            + " |"
        )

    branch_keys = [
        "full_rms",
        "sustain_only_rms",
        "transient_rms",
        "noise_rms",
        "transient_plus_noise_rms",
        "transient_rms_vs_full_percent",
        "noise_rms_vs_full_percent",
        "transient_plus_noise_rms_vs_full_percent",
    ]
    branch_summary = {}
    for key in branch_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        branch_summary[key] = (float(values.mean()), float(values.std(ddof=0)))

    lines.extend([
        "",
        "## Branch RMS Summary",
        "",
        "| Quantity | Mean | Std |",
        "|---|---:|---:|",
    ])
    for key, (mean, std) in branch_summary.items():
        lines.append(f"| {key} | {_format_float(mean, 4)} | {_format_float(std, 4)} |")

    lines.extend([
        "",
        "## Exported Listening Samples",
        "",
    ])
    for sample_dir in sample_dirs:
        rel = sample_dir.name
        lines.append(f"- [`{rel}`]({rel}/branch_ablation.png): `target.wav`, `full.wav`, `sustain_only.wav`, `transient.wav`, `noise.wav`, `transient_plus_noise.wav`")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- If full output improves MSS/LSD/RMS/onset metrics over sustain-only, the transient/noise branches are contributing measurable reconstruction value.",
        "- If full and sustain-only are nearly identical, the added branches are mostly decorative or too quiet to matter.",
        "- Pitch metrics here are estimated from reconstructed audio using `librosa.pyin`; they are useful diagnostics but are not ground-truth pitch labels.",
        "",
    ])
    path.write_text("\n".join(lines))


def _summarize_ablation(rows, labels):
    keys = [
        "mss",
        "lsd_db",
        "rms_ratio",
        "rms_corr",
        "f0_median_cents",
        "f0_mean_cents",
        "gross_pitch_error_pct",
        "f0_valid_pct",
        "onset_hf_log_error",
        "onset_energy_ratio",
        "rms",
        "peak",
    ]
    summary = {}
    for label in labels:
        model_rows = [row for row in rows if row["model"] == label]
        summary[label] = {"num_samples": len(model_rows)}
        for key in keys:
            mean, std = _mean_std([float(row[key]) for row in model_rows])
            summary[label][f"{key}_mean"] = mean
            summary[label][f"{key}_std"] = std
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-listen", type=int, default=5)
    parser.add_argument("--pitch-source", choices=["labels", "torchcrepe"], default="labels")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onset-seconds", type=float, default=0.15)
    args = parser.parse_args()

    run_dir = Path(args.run)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"bass_ddsp_branch_ablation_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    config = _load_yaml(run_dir / "config.yaml")
    dataset = make_dataset(config, args.seed, args.pitch_source)
    device = torch.device(args.device)
    metric_device = torch.device("cpu")
    model = load_model(config, run_dir, dataset, device)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    rows = []
    branch_rows = []
    sample_dirs = []
    for position, idx in enumerate(indices):
        torch.manual_seed(args.seed + idx)
        data = dataset.generate_debug_example(idx, pitch_source=args.pitch_source)
        branches = reconstruct(model, config, data, device)
        target = np.asarray(data["audio"], dtype=np.float32)
        full = _safe_audio(branches, "signal", target)
        sustain = _safe_audio(branches, "sustain", full)
        transient = _safe_audio(branches, "transient", full)
        noise = _safe_audio(branches, "noise", full)
        residual = (transient + noise).astype(np.float32)

        variants = {
            "full": full,
            "sustain_only": sustain,
        }
        metrics = {}
        for variant_name, audio in variants.items():
            variant_metrics = _audio_metrics(
                target,
                audio.astype(np.float32),
                data,
                config,
                metric_device,
                args.onset_seconds,
            )
            rows.append({
                "sample_position": position,
                "index": idx,
                "model": variant_name,
                **variant_metrics,
            })
            metrics.update({f"{variant_name}_{key}": value for key, value in variant_metrics.items()})

        full_rms = max(_rms(full), 1e-12)
        branch_row = {
            "sample_position": position,
            "index": idx,
            "full_rms": _rms(full),
            "sustain_only_rms": _rms(sustain),
            "transient_rms": _rms(transient),
            "noise_rms": _rms(noise),
            "transient_plus_noise_rms": _rms(residual),
            "sustain_only_rms_vs_full_percent": _rms(sustain) / full_rms * 100.0,
            "transient_rms_vs_full_percent": _rms(transient) / full_rms * 100.0,
            "noise_rms_vs_full_percent": _rms(noise) / full_rms * 100.0,
            "transient_plus_noise_rms_vs_full_percent": _rms(residual) / full_rms * 100.0,
            **metrics,
        }

        if position < args.num_listen:
            sample_dir = out_dir / f"sample_{position:02d}_idx_{idx:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            audios = {
                "target": target,
                "full": full,
                "sustain_only": sustain,
                "transient": transient,
                "noise": noise,
                "transient_plus_noise": residual,
            }
            _write_audio_set(sample_dir, data, audios)
            branch_metrics = _write_branch_metrics(sample_dir, audios)
            _plot_sample(sample_dir, sample_dir.name, data, audios)
            with open(sample_dir / "intervals.json", "w") as handle:
                json.dump(data.get("intervals", []), handle, indent=2)
            with open(sample_dir / "branch_metrics.json", "w") as handle:
                json.dump(branch_metrics, handle, indent=2)
            sample_dirs.append(sample_dir)

        branch_rows.append(branch_row)

    summary = _summarize_ablation(rows, ["full", "sustain_only"])
    _write_csv(out_dir / "per_sample_ablation_metrics.csv", rows)
    _write_csv(out_dir / "branch_rms_rows.csv", branch_rows)

    dwts_note = (
        "The previous Vanilla DWTS comparison is not a strict equal-budget comparison: "
        "`runs/vanilla_dwts_riff_20260722_040632/loss.csv` stops at step `32269`, "
        "while Bass-DDSP and Vanilla DDSP reached step `199999`."
    )
    with open(out_dir / "summary.json", "w") as handle:
        json.dump({
            "run": str(run_dir),
            "args": vars(args),
            "indices": indices,
            "summary": summary,
            "dwts_validity_note": dwts_note,
            "samples": [str(path) for path in sample_dirs],
        }, handle, indent=2)
    _write_report(out_dir / "REPORT.md", args, run_dir, summary, branch_rows, sample_dirs, dwts_note)

    print(json.dumps({
        "out_dir": str(out_dir),
        "report": str(out_dir / "REPORT.md"),
        "indices": indices,
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
