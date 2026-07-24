import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bass_ddsp.compare_models import (
    _audio_metrics,
    _branch_metrics,
    _draw_intervals,
    _format_float,
    _frame_rms,
    _frame_times,
    _load_yaml,
    _plot_loss_curves,
    _plot_metric_bars,
    _read_loss_tail,
    _spectrogram_db,
    _summarize,
    _write_csv,
)
from bass_ddsp.export_branch_debug import load_model, make_dataset, reconstruct


def _parse_model_spec(spec):
    if "=" not in spec:
        raise ValueError(
            "--model must be formatted as label=run_dir, "
            f"got {spec!r}"
        )
    label, run_dir = spec.split("=", 1)
    label = label.strip()
    run_dir = Path(run_dir.strip())
    if not label:
        raise ValueError(f"empty model label in {spec!r}")
    if not (run_dir / "config.yaml").exists():
        raise FileNotFoundError(f"missing config.yaml for {label}: {run_dir}")
    if not (run_dir / "state.pth").exists():
        raise FileNotFoundError(f"missing state.pth for {label}: {run_dir}")
    return label, run_dir


def _plot_sample(path, sample_name, data, model_outputs, colors):
    sr = int(data["sampling_rate"])
    block_size = int(data["block_size"])
    target = data["audio"].astype(np.float32)
    n = min([len(target)] + [len(branches["signal"]) for branches in model_outputs.values()])
    target = target[:n]
    audio_t = np.arange(n) / float(sr)
    frame_count = min(len(data["pitch"]), len(data["gate"]))
    frame_t = _frame_times(frame_count, sr, block_size)
    labels = data.get("articulation_labels", [])
    intervals = data.get("intervals", [])

    rows = 4 + len(model_outputs)
    heights = [1.1, 1.25] + [1.25] * len(model_outputs) + [0.9, 0.85]
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(15, 3.0 + 2.0 * rows),
        constrained_layout=True,
        gridspec_kw={"height_ratios": heights},
    )

    ax = axes[0]
    _draw_intervals(ax, intervals, labels)
    ax.plot(audio_t, target, color="black", linewidth=0.55, label="target")
    for model_name, branches in model_outputs.items():
        ax.plot(
            audio_t,
            branches["signal"][:n],
            color=colors[model_name],
            linewidth=0.55,
            alpha=0.82,
            label=model_name,
        )
    ax.set_title(f"{sample_name}: waveform")
    ax.set_ylabel("amp")
    ax.set_xlim(0, n / float(sr))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", ncol=min(4, len(model_outputs) + 1), fontsize=8)

    stft_axes = axes[1 : 2 + len(model_outputs)]
    for ax, audio, title in [(stft_axes[0], target, "target STFT")] + [
        (stft_axes[idx + 1], branches["signal"][:n], f"{model_name} STFT")
        for idx, (model_name, branches) in enumerate(model_outputs.items())
    ]:
        db, freqs, times = _spectrogram_db(audio, sr, block_size)
        image = ax.pcolormesh(times, freqs, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        _draw_intervals(ax, intervals, labels)
        ax.set_title(title)
        ax.set_ylabel("Hz")
        ax.set_ylim(0, min(2200, sr / 2))
        ax.set_xlim(0, n / float(sr))
        fig.colorbar(image, ax=ax, label="dB")

    ax = axes[-2]
    _draw_intervals(ax, intervals, labels)
    target_rms = _frame_rms(target, block_size)
    frames = min(len(target_rms), frame_count)
    ax.plot(frame_t[:frames], target_rms[:frames], color="black", linewidth=1.1, label="target")
    for model_name, branches in model_outputs.items():
        model_rms = _frame_rms(branches["signal"][:n], block_size)
        frames = min(len(target_rms), len(model_rms), frame_count)
        ax.plot(
            frame_t[:frames],
            model_rms[:frames],
            color=colors[model_name],
            linewidth=1.0,
            label=model_name,
        )
    ax.set_title("frame RMS envelope")
    ax.set_ylabel("RMS")
    ax.set_xlim(0, n / float(sr))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", ncol=min(4, len(model_outputs) + 1), fontsize=8)

    ax = axes[-1]
    _draw_intervals(ax, intervals, labels)
    ax.plot(frame_t[:frame_count], data["pitch"][:frame_count], label="f0 label", linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(frame_t[:frame_count], data["onset_strength"][:frame_count], color="#d62728", label="onset", linewidth=0.9)
    ax2.plot(frame_t[:frame_count], data["gate"][:frame_count], color="#2ca02c", label="gate", linewidth=0.9)
    ax.set_title("controls")
    ax.set_ylabel("Hz")
    ax2.set_ylabel("control")
    ax.set_xlim(0, n / float(sr))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(path, args, model_specs, summary, loss_summary, out_dir):
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

    labels = [label for label, _ in model_specs]

    def winner(key, direction):
        values = {label: summary[label][key] for label in labels}
        finite = {k: v for k, v in values.items() if np.isfinite(v)}
        if not finite:
            return "n/a"
        if direction == "lower":
            return min(finite, key=finite.get)
        if direction == "higher":
            return max(finite, key=finite.get)
        if direction == "close to 1":
            return min(finite, key=lambda k: abs(finite[k] - 1.0))
        return "n/a"

    lines = [
        "# Three-Way DDSP Model Comparison",
        "",
        "This report compares the new Bass-DDSP run against the completed Vanilla DWTS and Vanilla DDSP baselines on the same deterministic generated-riff evaluation set.",
        "",
        "## Runs",
        "",
    ]
    for label, run_dir in model_specs:
        lines.append(f"- {label}: `{run_dir}`")
    lines.extend([
        f"- Samples evaluated: `{summary[labels[0]]['num_samples']}`",
        f"- Evaluation seed: `{args.seed}`",
        f"- Pitch source: `{args.pitch_source}`",
        f"- Output directory: `{out_dir}`",
        "",
        "## Objective Metrics",
        "",
        "| Metric | Direction | " + " | ".join(labels) + " | Winner |",
        "|---|---|" + "|".join(["---:" for _ in labels]) + "|---|",
    ])
    for name, direction, key, digits in metric_rows:
        values = [_format_float(summary[label][key], digits) for label in labels]
        lines.append("| " + " | ".join([name, direction, *values, winner(key, direction)]) + " |")

    lines.extend([
        "",
        "## Training Tail",
        "",
        "| Model | Steps | Tail loss | Tail spectral | Tail RMS |",
        "|---|---:|---:|---:|---:|",
    ])
    for label in labels:
        tail = loss_summary[label]
        lines.append(
            "| "
            + " | ".join([
                label,
                _format_float(tail.get("steps"), 0),
                _format_float(tail.get("tail_loss_mean"), 4),
                _format_float(tail.get("tail_spectral_loss_mean"), 4),
                _format_float(tail.get("tail_rms_loss_mean"), 4),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Branch Diagnostics",
        "",
        "| Model | Sustain RMS / signal % | Noise RMS / signal % | Transient RMS / signal % |",
        "|---|---:|---:|---:|",
    ])
    for label in labels:
        model_summary = summary[label]
        lines.append(
            "| "
            + " | ".join([
                label,
                _format_float(model_summary["sustain_rms_vs_signal_pct_mean"], 2),
                _format_float(model_summary["noise_rms_vs_signal_pct_mean"], 2),
                _format_float(model_summary["transient_rms_vs_signal_pct_mean"], 2),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Visualizations",
        "",
        "- [`metric_bars.png`](metric_bars.png)",
        "- [`loss_curves.png`](loss_curves.png)",
        "",
        "Sample comparison plots:",
        "",
    ])
    for sample_dir in sorted(Path(out_dir).glob("sample_*")):
        plot = sample_dir / "comparison.png"
        if plot.exists():
            lines.append(f"- [`{sample_dir.name}`]({sample_dir.name}/comparison.png)")
    lines.extend([
        "",
        "## Interpretation Rule",
        "",
        "The new Bass-DDSP run is useful only if it improves attack, loudness tracking, pitch correctness, or listening preference. Architecture complexity alone is not a result.",
        "",
    ])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="label=run_dir; repeat for each model")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-plots", type=int, default=4)
    parser.add_argument("--pitch-source", choices=["labels", "torchcrepe"], default="labels")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onset-seconds", type=float, default=0.15)
    args = parser.parse_args()

    model_specs = [_parse_model_spec(spec) for spec in args.model]
    labels = [label for label, _ in model_specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"model labels must be unique: {labels}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    configs = {label: _load_yaml(run_dir / "config.yaml") for label, run_dir in model_specs}
    dataset = make_dataset(configs[labels[0]], args.seed, args.pitch_source)
    device = torch.device(args.device)
    metric_device = torch.device("cpu")
    models = {
        label: load_model(configs[label], run_dir, dataset, device)
        for label, run_dir in model_specs
    }
    colors = {
        label: color
        for label, color in zip(labels, ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"])
    }

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    rows = []
    samples = []

    for position, idx in enumerate(indices):
        data = dataset.generate_debug_example(idx, pitch_source=args.pitch_source)
        target = data["audio"].astype(np.float32)
        outputs = {}
        for label in labels:
            torch.manual_seed(args.seed + idx)
            outputs[label] = reconstruct(models[label], configs[label], data, device)
            metrics = _audio_metrics(
                target,
                outputs[label]["signal"].astype(np.float32),
                data,
                configs[label],
                metric_device,
                args.onset_seconds,
            )
            rows.append({
                "sample_position": position,
                "index": idx,
                "model": label,
                **metrics,
                **_branch_metrics(outputs[label]),
            })

        sample_name = f"sample_{position:02d}_idx_{idx:04d}"
        sample_dir = out_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sr = int(data["sampling_rate"])
        sf.write(sample_dir / "target.wav", target, sr, subtype="FLOAT")
        concat = [target]
        for label in labels:
            audio = outputs[label]["signal"].astype(np.float32)
            safe = label.lower().replace(" ", "_").replace("/", "_")
            sf.write(sample_dir / f"{safe}.wav", audio, sr, subtype="FLOAT")
            concat.extend([np.zeros(int(0.25 * sr), dtype=np.float32), audio])
        sf.write(sample_dir / "target_then_models.wav", np.concatenate(concat), sr, subtype="FLOAT")
        if position < args.num_plots:
            _plot_sample(sample_dir / "comparison.png", sample_name, data, outputs, colors)
        samples.append({"index": idx, "directory": str(sample_dir)})

    summary = _summarize(rows, labels)
    loss_summary = {
        label: _read_loss_tail(run_dir)
        for label, run_dir in model_specs
    }
    _write_csv(out_dir / "per_sample_metrics.csv", rows)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump({
            "args": vars(args),
            "models": {label: str(run_dir) for label, run_dir in model_specs},
            "indices": indices,
            "summary": summary,
            "loss_summary": loss_summary,
            "samples": samples,
        }, handle, indent=2)
    _plot_metric_bars(out_dir / "metric_bars.png", summary)
    _plot_loss_curves(out_dir / "loss_curves.png", {label: run_dir for label, run_dir in model_specs})
    _write_report(out_dir / "REPORT.md", args, model_specs, summary, loss_summary, out_dir)

    print(json.dumps({
        "out_dir": str(out_dir),
        "report": str(out_dir / "REPORT.md"),
        "summary": summary,
        "indices": indices,
    }, indent=2))


if __name__ == "__main__":
    main()
