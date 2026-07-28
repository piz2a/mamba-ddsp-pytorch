import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bass_ddsp.compare_models import (
    _audio_metrics,
    _format_float,
    _frame_rms,
    _frame_times,
    _lsd_db,
    _mean_std,
    _mss_loss,
    _safe_corr,
    _spectrogram_db,
    _write_csv,
)
from bass_ddsp.dataset import IDMTBassSingleTrackDataset
from bass_ddsp.train import clean_state_dict, make_model


def load_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def make_dataset(config, root, examples, seed, periodicity_fmin):
    idmt_config = dict(config.get("idmt_bass", {}))
    idmt_config["examples_per_epoch"] = int(examples)
    idmt_config["seed"] = int(seed)
    idmt_config["allowed_articulation_labels"] = list(config["data"]["articulation_labels"])
    idmt_config["pitch_fmin"] = float(periodicity_fmin)
    return IDMTBassSingleTrackDataset(
        data_location=root,
        sampling_rate=int(config["preprocess"]["sampling_rate"]),
        block_size=int(config["preprocess"]["block_size"]),
        signal_length=int(config["preprocess"]["signal_length"]),
        **idmt_config,
    )


def load_bass_model(run_dir, config, device):
    model_config = dict(config["model"])
    model_config["n_articulation"] = len(config["data"]["articulation_labels"])
    model = make_model({"model": model_config}).to(device)
    state = clean_state_dict(torch.load(run_dir / "state.pth", map_location=device))
    model.load_state_dict(state)
    model.eval()
    return model


def debug_example_to_data(example):
    frames = len(example["loudness"])
    sr = int(example["sampling_rate"])
    block_size = int(example["block_size"])
    expression_labels = list(example["expression_labels"])
    pluck_labels = list(example["pluck_labels"])
    expression_track = np.asarray(
        [expression_labels[int(idx)] for idx in example["expression"][:frames]],
        dtype=object,
    )
    pluck_track = np.asarray(
        [pluck_labels[int(idx)] for idx in example["pluck"][:frames]],
        dtype=object,
    )
    inactive = example["gate"][:frames] <= 0.5
    expression_track[inactive] = "NONE"
    pluck_track[inactive] = "NONE"
    return {
        "track_id": example["track_id"],
        "example_id": f"{example['track_id']}_{int(example['segment_start_sample']):08d}",
        "audio": example["audio"].astype(np.float32),
        "sampling_rate": sr,
        "block_size": block_size,
        "times": _frame_times(frames, sr, block_size).astype(np.float32),
        "pitch": example["pitch"].astype(np.float32),
        "label_pitch": example["label_pitch"].astype(np.float32),
        "loudness": example["loudness"].astype(np.float32),
        "articulation": example["articulation"].astype(np.int64),
        "onset_strength": example["onset_strength"].astype(np.float32),
        "offset": example["offset"].astype(np.float32),
        "gate": example["gate"].astype(np.float32),
        "note_age": example["note_age"].astype(np.float32),
        "periodicity": example["periodicity"].astype(np.float32),
        "intervals": list(example["intervals"]),
        "events": list(example["intervals"]),
        "expression_track": expression_track,
        "pluck_track": pluck_track,
        "articulation_labels": list(example["articulation_labels"]),
        "source_audio": example["source_audio"],
        "source_notes": example["source_notes"],
        "segment_start_sample": int(example["segment_start_sample"]),
        "segment_start_seconds": float(example["segment_start_seconds"]),
    }


def reconstruct_example(model, config, data, device):
    mean_loudness = float(config["data"]["mean_loudness"])
    std_loudness = max(float(config["data"]["std_loudness"]), 1e-8)
    tensors = {
        "pitch": torch.from_numpy(data["pitch"]).float().unsqueeze(0).unsqueeze(-1).to(device),
        "loudness": torch.from_numpy((data["loudness"] - mean_loudness) / std_loudness).float().unsqueeze(0).unsqueeze(-1).to(device),
        "articulation": torch.from_numpy(data["articulation"]).long().unsqueeze(0).to(device),
        "onset_strength": torch.from_numpy(data["onset_strength"]).float().unsqueeze(0).unsqueeze(-1).to(device),
        "offset": torch.from_numpy(data["offset"]).float().unsqueeze(0).unsqueeze(-1).to(device),
        "gate": torch.from_numpy(data["gate"]).float().unsqueeze(0).unsqueeze(-1).to(device),
        "note_age": torch.from_numpy(data["note_age"]).float().unsqueeze(0).unsqueeze(-1).to(device),
        "periodicity": torch.from_numpy(data["periodicity"]).float().unsqueeze(0).unsqueeze(-1).to(device),
    }
    with torch.no_grad():
        signal = model(
            tensors["pitch"],
            tensors["loudness"],
            articulation=tensors["articulation"],
            onset_strength=tensors["onset_strength"],
            offset=tensors["offset"],
            gate=tensors["gate"],
            note_age=tensors["note_age"],
            periodicity=tensors["periodicity"],
        )
    outputs = {
        "signal": signal.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32),
    }
    for name, tensor in model.last_branch_outputs.items():
        outputs[name] = tensor.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
    return outputs


def control_summary(data):
    """Return observable ranges used to audit evaluator controls."""
    summary = {}
    for name in [
        "label_pitch",
        "loudness",
        "onset_strength",
        "offset",
        "gate",
        "note_age",
        "periodicity",
    ]:
        values = np.asarray(data[name], dtype=np.float32)
        summary[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    summary["pitch_source"] = "label_pitch"
    return summary


def align_audio(target, recon):
    n = min(len(target), len(recon))
    return target[:n].astype(np.float32), recon[:n].astype(np.float32)


def gain_match(target, recon):
    target_rms = float(np.sqrt(np.mean(target.astype(np.float64) ** 2) + 1e-12))
    recon_rms = float(np.sqrt(np.mean(recon.astype(np.float64) ** 2) + 1e-12))
    gain = target_rms / max(recon_rms, 1e-12)
    return (recon * gain).astype(np.float32), gain


def branch_metrics(outputs):
    signal = outputs["signal"]
    signal_rms = max(float(np.sqrt(np.mean(signal * signal) + 1e-12)), 1e-12)
    out = {}
    for name in ["sustain", "noise", "transient"]:
        audio = outputs.get(name, np.zeros_like(signal))
        value = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        out[f"{name}_rms"] = value
        out[f"{name}_rms_vs_signal_pct"] = value / signal_rms * 100.0
    return out


def expression_metrics(data, target, recon):
    block_size = int(data["block_size"])
    target_rms = _frame_rms(target, block_size)
    recon_rms = _frame_rms(recon, block_size)
    frames = min(len(target_rms), len(recon_rms), len(data["expression_track"]))
    rows = []
    for expression in sorted(set(data["expression_track"])):
        if expression == "NONE":
            continue
        mask = data["expression_track"][:frames] == expression
        if not np.any(mask):
            continue
        t_rms = target_rms[:frames][mask]
        r_rms = recon_rms[:frames][mask]
        rows.append({
            "example_id": data["example_id"],
            "track_id": data["track_id"],
            "expression": expression,
            "frames": int(mask.sum()),
            "rms_ratio": float(np.mean(r_rms) / max(float(np.mean(t_rms)), 1e-12)),
            "rms_corr": _safe_corr(np.log(t_rms + 1e-7), np.log(r_rms + 1e-7)),
        })
    return rows


def summarize(rows, group_key, metric_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    out = {}
    for group, group_rows in grouped.items():
        out[group] = {"count": len(group_rows)}
        for key in metric_keys:
            mean, std = _mean_std([float(row[key]) for row in group_rows])
            out[group][f"{key}_mean"] = mean
            out[group][f"{key}_std"] = std
    return dict(sorted(out.items()))


def plot_example(sample_dir, data, target, outputs):
    sr = int(data["sampling_rate"])
    block_size = int(data["block_size"])
    recon = outputs["signal"]
    n = min(len(target), len(recon))
    target = target[:n]
    recon = recon[:n]
    t = np.arange(n) / float(sr)
    frames = min(len(data["times"]), len(data["label_pitch"]))
    frame_t = data["times"][:frames]
    labels = data["articulation_labels"]
    intervals = data["intervals"]

    fig, axes = plt.subplots(
        8,
        1,
        figsize=(16, 20),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.25, 1.25, 1.0, 1.0, 0.8, 0.8]},
    )

    def draw_intervals(ax):
        cmap = plt.get_cmap("tab20")
        colors = {label: cmap(idx % cmap.N) for idx, label in enumerate(labels)}
        for interval in intervals:
            color = colors.get(interval.get("articulation", ""), "0.85")
            start = max(float(interval["start_seconds"]), 0.0)
            end = min(float(interval["end_seconds"]), n / float(sr))
            ax.axvspan(start, end, color=color, alpha=0.10, linewidth=0)
            if 0.0 <= float(interval["start_seconds"]) <= n / float(sr):
                ax.axvline(float(interval["start_seconds"]), color="black", linewidth=0.25, alpha=0.25)

    ax = axes[0]
    draw_intervals(ax)
    ax.plot(t, target, color="black", linewidth=0.45, label="target")
    ax.plot(t, recon, color="#1f77b4", linewidth=0.45, alpha=0.85, label="Bass-DDSP")
    ax.set_title(f"{data['example_id']}: natural SINGLE-TRACKS window")
    ax.set_xlim(0, n / float(sr))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    draw_intervals(ax)
    ax.plot(t, outputs.get("sustain", np.zeros_like(recon))[:n], linewidth=0.45, label="sustain")
    ax.plot(t, outputs.get("transient", np.zeros_like(recon))[:n], linewidth=0.45, label="transient")
    ax.plot(t, outputs.get("noise", np.zeros_like(recon))[:n], linewidth=0.45, label="noise")
    ax.set_title("branch waveforms")
    ax.set_xlim(0, n / float(sr))
    ax.legend(loc="upper right", ncol=3)
    ax.grid(True, alpha=0.2)

    for ax, audio, title in [
        (axes[2], target, "target STFT"),
        (axes[3], recon, "Bass-DDSP STFT"),
    ]:
        db, freqs, times = _spectrogram_db(audio, sr, block_size)
        image = ax.pcolormesh(times, freqs, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        draw_intervals(ax)
        ax.set_title(title)
        ax.set_ylabel("Hz")
        ax.set_ylim(0, min(2500, sr / 2))
        ax.set_xlim(0, n / float(sr))
        fig.colorbar(image, ax=ax, label="dB")

    ax = axes[4]
    draw_intervals(ax)
    ax.plot(frame_t, data["label_pitch"][:frames], color="black", linewidth=1.0, label="label f0")
    ax.set_title("pitch input")
    ax.set_ylabel("Hz")
    ax.set_xlim(0, n / float(sr))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[5]
    draw_intervals(ax)
    target_rms = _frame_rms(target, block_size)
    recon_rms = _frame_rms(recon, block_size)
    rms_frames = min(len(target_rms), len(recon_rms), frames)
    ax.plot(frame_t[:rms_frames], target_rms[:rms_frames], color="black", linewidth=1.0, label="target RMS")
    ax.plot(frame_t[:rms_frames], recon_rms[:rms_frames], color="#1f77b4", linewidth=1.0, label="recon RMS")
    ax.set_title("frame RMS")
    ax.set_xlim(0, n / float(sr))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[6]
    draw_intervals(ax)
    ax.plot(frame_t, data["loudness"][:frames], label="loudness", linewidth=0.9)
    ax2 = ax.twinx()
    ax2.plot(frame_t, data["onset_strength"][:frames], color="#d62728", label="onset_strength", linewidth=0.8)
    ax2.plot(frame_t, data["offset"][:frames], color="#ff7f0e", label="offset", linewidth=0.8)
    ax2.plot(frame_t, data["periodicity"][:frames], color="#9467bd", label="periodicity", linewidth=0.8)
    ax.set_title("controls from IDMTBassSingleTrackDataset")
    ax.set_xlim(0, n / float(sr))
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[7]
    draw_intervals(ax)
    art = data["articulation"][:frames].astype(np.float32)
    art[data["gate"][:frames] <= 0.5] = np.nan
    ax.step(frame_t, art, where="mid", color="black", linewidth=0.9, label="articulation_id")
    ax.step(frame_t, data["gate"][:frames], where="mid", color="#2ca02c", linewidth=0.9, label="gate")
    ax.plot(frame_t, data["note_age"][:frames], color="#8c564b", linewidth=0.9, label="note_age")
    ax.set_title("articulation, gate, note_age")
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, n / float(sr))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    fig.savefig(sample_dir / "overview.png", dpi=150)
    plt.close(fig)


def write_report(path, args, run_dir, dataset, summary, expression_summary, dataset_stats, output_dir):
    lines = [
        "# Bass-DDSP SINGLE-TRACKS Dataset-Backed Evaluation",
        "",
        f"- Bass-DDSP run: `{run_dir}`",
        f"- Dataset: `{args.dataset_root}`",
        f"- Output directory: `{output_dir}`",
        f"- Natural tracks kept by dataset: `{dataset_stats['kept_tracks']}`",
        f"- Natural tracks excluded by dataset: `{dataset_stats['excluded_tracks']}`",
        f"- Examples evaluated: `{dataset_stats['examples']}`",
        "",
        "## Scope",
        "",
        "This evaluator does not build controls by hand. It instantiates `IDMTBassSingleTrackDataset` from `bass_ddsp/dataset.py` and consumes `generate_debug_example()`.",
        "",
        "Therefore `onset_strength`, `offset`, `gate`, `note_age`, `periodicity`, `pitch`, label tracks, loudness extraction, and tensor formatting follow the same dataset method family used by `IDMTBassRiffDataset`.",
        "",
        "Tracks containing labels outside the trained articulation set are excluded by the dataset in strict mode.",
        "",
        "## Dataset Filtering",
        "",
        f"- Allowed articulations: `{list(dataset.articulation_labels)}`",
        "",
        "| Excluded track | Invalid label counts |",
        "|---|---|",
    ]
    for item in dataset.excluded_tracks:
        lines.append(f"| {item['track_id']} | `{item['invalid_articulation_counts']}` |")
    if not dataset.excluded_tracks:
        lines.append("| none | `{}` |")

    lines.extend([
        "",
        "## Control Audit",
        "",
        f"`periodicity` is the raw TorchCREPE confidence curve, clipped to `[0, 1]`, using `fmin={args.periodicity_fmin:g} Hz`. No articulation prior or constant fallback is used. F0 is the annotated label pitch, matching the saved training configuration.",
        "",
        "| Example | Track | Periodicity min | Periodicity max | Periodicity std | Gate mean | Onset max | Offset max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in dataset_stats["control_audit"]:
        controls = item["controls"]
        lines.append(
            f"| {item['example_id']} | {item['track_id']} | "
            f"{controls['periodicity']['min']:.4f} | {controls['periodicity']['max']:.4f} | "
            f"{controls['periodicity']['std']:.4f} | {controls['gate']['mean']:.4f} | "
            f"{controls['onset_strength']['max']:.4f} | {controls['offset']['max']:.4f} |"
        )

    lines.extend([
        "",
        "## Metrics",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ])
    for key in [
        "mss",
        "mss_gain_matched",
        "lsd_db",
        "lsd_db_gain_matched",
        "rms_ratio",
        "rms_corr",
        "onset_hf_log_error",
        "onset_energy_ratio",
    ]:
        lines.append(
            f"| {key} | {_format_float(summary[key + '_mean'], 4)} | "
            f"{_format_float(summary[key + '_std'], 4)} |"
        )

    lines.extend([
        "",
        "## Branch RMS",
        "",
        "| Branch | RMS / signal mean % | Std |",
        "|---|---:|---:|",
    ])
    for key in ["sustain", "noise", "transient"]:
        lines.append(
            f"| {key} | {_format_float(summary[f'{key}_rms_vs_signal_pct_mean'], 3)} | "
            f"{_format_float(summary[f'{key}_rms_vs_signal_pct_std'], 3)} |"
        )

    lines.extend([
        "",
        "## Expression RMS Metrics",
        "",
        "| Expression | Count | RMS ratio | RMS corr |",
        "|---|---:|---:|---:|",
    ])
    for expression, values in expression_summary.items():
        lines.append(
            f"| {expression} | {values['count']} | "
            f"{_format_float(values['rms_ratio_mean'], 3)} | "
            f"{_format_float(values['rms_corr_mean'], 3)} |"
        )

    lines.extend([
        "",
        "## Visualizations",
        "",
    ])
    for sample in dataset_stats["visualized_examples"]:
        lines.append(f"- [`{sample}`]({sample}/overview.png)")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513")
    parser.add_argument("--dataset-root", default="/disk1/ahnjiho/IDMT-SMT-BASS-SINGLE-TRACKS")
    parser.add_argument("--out-dir")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--num-visualizations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--pitch-source", choices=["labels", "torchcrepe"])
    parser.add_argument("--periodicity-fmin", type=float, default=40.0)
    args = parser.parse_args()

    run_dir = Path(args.run)
    root = Path(args.dataset_root)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"bass_ddsp_single_tracks_dataset_eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    config = load_yaml(run_dir / "config.yaml")
    dataset = make_dataset(config, root, args.examples, args.seed, args.periodicity_fmin)
    device = torch.device(args.device)
    model = load_bass_model(run_dir, config, device)

    rows = []
    expression_rows = []
    visualized = []
    control_audit = []
    for idx in range(args.examples):
        example = dataset.generate_debug_example(
            idx,
            pitch_source=args.pitch_source or config.get("idmt_bass", {}).get("pitch_source", "labels"),
        )
        data = debug_example_to_data(example)
        controls = control_summary(data)
        control_audit.append({
            "example_id": data["example_id"],
            "track_id": data["track_id"],
            "controls": controls,
        })
        outputs = reconstruct_example(model, config, data, device)
        target, recon = align_audio(data["audio"], outputs["signal"])
        recon_gain, gain = gain_match(target, recon)
        whole = _audio_metrics(target, recon, data, config, torch.device("cpu"), 0.15)
        row = {
            "example_id": data["example_id"],
            "track_id": data["track_id"],
            **whole,
            "mss_gain_matched": _mss_loss(target, recon_gain, config, torch.device("cpu")),
            "lsd_db_gain_matched": _lsd_db(target, recon_gain, data["sampling_rate"], data["block_size"]),
            "gain_match": gain,
            "gain_match_db": 20.0 * math.log10(max(abs(gain), 1e-12)),
            **branch_metrics(outputs),
        }
        rows.append(row)
        expression_rows.extend(expression_metrics(data, target, recon))

        if len(visualized) < args.num_visualizations:
            sample_dir = out_dir / data["example_id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            sf.write(sample_dir / "target.wav", target, data["sampling_rate"], subtype="FLOAT")
            sf.write(sample_dir / "reconstruction.wav", recon, data["sampling_rate"], subtype="FLOAT")
            sf.write(sample_dir / "reconstruction_gain_matched.wav", recon_gain, data["sampling_rate"], subtype="FLOAT")
            for branch in ["sustain", "transient", "noise"]:
                sf.write(
                    sample_dir / f"{branch}.wav",
                    outputs.get(branch, np.zeros_like(recon))[: len(recon)],
                    data["sampling_rate"],
                    subtype="FLOAT",
                )
            plot_example(sample_dir, data, target, outputs)
            with open(sample_dir / "controls_summary.json", "w") as handle:
                json.dump({
                    "example_id": data["example_id"],
                    "track_id": data["track_id"],
                    "source_audio": data["source_audio"],
                    "source_notes": data["source_notes"],
                    "segment_start_seconds": data["segment_start_seconds"],
                    "controls": controls,
                    "metrics": row,
                }, handle, indent=2)
            visualized.append(data["example_id"])

    metric_keys = [
        "mss",
        "mss_gain_matched",
        "lsd_db",
        "lsd_db_gain_matched",
        "rms_ratio",
        "rms_corr",
        "onset_hf_log_error",
        "onset_energy_ratio",
        "sustain_rms_vs_signal_pct",
        "noise_rms_vs_signal_pct",
        "transient_rms_vs_signal_pct",
        "gain_match_db",
    ]
    summary = {}
    for key in metric_keys:
        mean, std = _mean_std([float(row[key]) for row in rows])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    expression_summary = summarize(
        expression_rows,
        "expression",
        ["rms_ratio", "rms_corr"],
    )
    dataset_stats = {
        "kept_tracks": len(dataset.track_records),
        "excluded_tracks": len(dataset.excluded_tracks),
        "examples": len(rows),
        "visualized_examples": visualized,
        "control_audit": control_audit,
    }
    _write_csv(out_dir / "example_metrics.csv", rows)
    _write_csv(out_dir / "expression_metrics.csv", expression_rows)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump({
            "run": str(run_dir),
            "dataset_root": str(root),
            "summary": summary,
            "expression_summary": expression_summary,
            "dataset_stats": dataset_stats,
            "excluded_tracks": dataset.excluded_tracks,
        }, handle, indent=2)
    write_report(out_dir / "REPORT.md", args, run_dir, dataset, summary, expression_summary, dataset_stats, out_dir)
    print(json.dumps({
        "out_dir": str(out_dir),
        "report": str(out_dir / "REPORT.md"),
        "kept_tracks": len(dataset.track_records),
        "excluded_tracks": len(dataset.excluded_tracks),
        "examples": len(rows),
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
