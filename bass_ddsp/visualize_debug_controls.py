import argparse
import csv
import json
import random
from pathlib import Path

import librosa as li
import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bass_ddsp.export_branch_debug import make_dataset, load_model, reconstruct


def _time_axis(length, sampling_rate):
    return np.arange(length, dtype=np.float32) / float(sampling_rate)


def _frame_times(frame_count, sampling_rate, block_size):
    return (
        np.arange(frame_count, dtype=np.float32) * float(block_size)
        + float(block_size) / 2.0
    ) / float(sampling_rate)


def _spectrogram_db(audio, sampling_rate, hop_length):
    n_fft = min(2048, max(256, int(2 ** np.ceil(np.log2(hop_length * 4)))))
    n_fft = min(n_fft, max(256, audio.shape[-1]))
    D = li.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=True,
    )
    mag = np.abs(D)
    if not np.any(mag > 0):
        mag = mag + 1e-8
    db = li.amplitude_to_db(mag, ref=np.max)
    freqs = li.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    times = li.frames_to_time(
        np.arange(db.shape[1]),
        sr=sampling_rate,
        hop_length=hop_length,
    )
    return db, times, freqs


def _plot_spectrogram(ax, audio, sampling_rate, hop_length, title):
    db, times, freqs = _spectrogram_db(audio, sampling_rate, hop_length)
    image = ax.pcolormesh(times, freqs, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
    ax.set_title(title)
    ax.set_ylabel("Hz")
    ax.set_ylim(0, min(2000, sampling_rate / 2))
    return image


def _interval_color_map(labels):
    cmap = plt.get_cmap("tab20")
    return {label: cmap(idx % cmap.N) for idx, label in enumerate(labels)}


def _draw_intervals(ax, intervals, colors, ymin=0.0, ymax=1.0, alpha=0.14):
    for interval in intervals:
        label = interval.get("articulation", "")
        color = colors.get(label, "0.85")
        ax.axvspan(
            interval["start_seconds"],
            interval["end_seconds"],
            ymin=ymin,
            ymax=ymax,
            color=color,
            alpha=alpha,
            linewidth=0,
        )


def _write_control_csv(path, data):
    sr = int(data["sampling_rate"])
    block_size = int(data["block_size"])
    frames = len(data["pitch"])
    times = _frame_times(frames, sr, block_size)
    keys = [
        "pitch",
        "label_pitch",
        "loudness",
        "onset_strength",
        "offset",
        "gate",
        "note_age",
        "periodicity",
        "articulation",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_seconds", *keys])
        for idx, time_value in enumerate(times):
            row = [float(time_value)]
            for key in keys:
                value = data.get(key)
                if value is None:
                    row.append("")
                else:
                    row.append(float(value[idx]) if key != "articulation" else int(value[idx]))
            writer.writerow(row)


def _safe_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    lo = float(np.nanmin(values)) if values.size else 0.0
    hi = float(np.nanmax(values)) if values.size else 0.0
    if hi - lo < 1e-8:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def write_debug_plot(sample_dir, data, branches, config, title):
    sample_dir.mkdir(parents=True, exist_ok=True)
    audio = data["audio"].astype(np.float32)
    sr = int(data["sampling_rate"])
    block_size = int(data["block_size"])
    frames = len(data["pitch"])
    frame_t = _frame_times(frames, sr, block_size)
    audio_t = _time_axis(audio.shape[-1], sr)
    intervals = data.get("intervals", [])
    labels = data.get("articulation_labels", [])
    colors = _interval_color_map(labels)

    recon = None if branches is None else branches.get("signal")
    row_count = 8 if recon is not None else 7
    height_ratios = [1.2, 1.35, 1.35, 0.9, 0.9, 0.9, 0.75, 0.65]
    if recon is None:
        height_ratios = [1.2, 1.45, 0.9, 0.9, 0.9, 0.75, 0.65]

    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(15, 2.05 * row_count),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios[:row_count]},
    )
    axes = np.asarray(axes).reshape(-1)

    axis_idx = 0
    waveform_ax = axes[axis_idx]
    axis_idx += 1
    _draw_intervals(waveform_ax, intervals, colors)
    waveform_ax.plot(audio_t, audio, linewidth=0.65, label="target", color="black")
    if recon is not None:
        recon_t = _time_axis(recon.shape[-1], sr)
        waveform_ax.plot(recon_t, recon, linewidth=0.65, label="reconstruction", color="#1f77b4", alpha=0.8)
    waveform_ax.set_title(f"{title}: waveform")
    waveform_ax.set_ylabel("amp")
    waveform_ax.legend(loc="upper right", ncol=2)
    waveform_ax.grid(True, alpha=0.25)

    spec_ax = axes[axis_idx]
    axis_idx += 1
    image = _plot_spectrogram(spec_ax, audio, sr, block_size, "target STFT")
    _draw_intervals(spec_ax, intervals, colors, alpha=0.08)
    fig.colorbar(image, ax=spec_ax, label="dB")

    if recon is not None:
        recon_spec_ax = axes[axis_idx]
        axis_idx += 1
        image = _plot_spectrogram(recon_spec_ax, recon.astype(np.float32), sr, block_size, "reconstruction STFT")
        _draw_intervals(recon_spec_ax, intervals, colors, alpha=0.08)
        fig.colorbar(image, ax=recon_spec_ax, label="dB")

    f0_ax = axes[axis_idx]
    axis_idx += 1
    _draw_intervals(f0_ax, intervals, colors)
    f0_ax.plot(frame_t, data["pitch"], marker=".", markersize=2.5, linewidth=1.0, label="pitch input")
    if "label_pitch" in data:
        f0_ax.plot(frame_t, data["label_pitch"], linestyle="--", linewidth=0.8, label="label pitch", alpha=0.75)
    f0_ax.set_title("F0 controls")
    f0_ax.set_ylabel("Hz")
    f0_ax.grid(True, alpha=0.25)
    f0_ax.legend(loc="upper right", ncol=2)

    loud_ax = axes[axis_idx]
    axis_idx += 1
    _draw_intervals(loud_ax, intervals, colors)
    loudness = data["loudness"].astype(np.float32)
    loud_ax.plot(frame_t, loudness, linewidth=1.0, label="raw loudness")
    if "data" in config and "mean_loudness" in config["data"] and "std_loudness" in config["data"]:
        mean = float(config["data"]["mean_loudness"])
        std = max(float(config["data"]["std_loudness"]), 1e-8)
        loud_ax.plot(frame_t, (loudness - mean) / std, linewidth=1.0, label="normalized loudness")
    loud_ax.set_title("loudness(t)")
    loud_ax.set_ylabel("value")
    loud_ax.grid(True, alpha=0.25)
    loud_ax.legend(loc="upper right", ncol=2)

    event_ax = axes[axis_idx]
    axis_idx += 1
    _draw_intervals(event_ax, intervals, colors)
    event_ax.plot(frame_t, data["onset_strength"], linewidth=1.2, label="onset_strength")
    event_ax.plot(frame_t, data["gate"], linewidth=1.0, label="gate")
    event_ax.plot(frame_t, data["offset"], linewidth=1.0, label="offset")
    event_ax.plot(frame_t, data["periodicity"], linewidth=1.0, label="periodicity")
    if branches is not None and "_sustain_harmonic_gate" in branches:
        harmonic_gate = branches["_sustain_harmonic_gate"]
        hg_t = _time_axis(harmonic_gate.shape[-1], sr)
        event_ax.plot(hg_t, _safe_normalize(harmonic_gate), linewidth=0.85, label="harmonic_gate normalized")
    event_ax.set_title("event and harmonic controls")
    event_ax.set_ylabel("[0, 1]")
    event_ax.set_ylim(-0.05, 1.05)
    event_ax.grid(True, alpha=0.25)
    event_ax.legend(loc="upper right", ncol=3)

    age_ax = axes[axis_idx]
    axis_idx += 1
    _draw_intervals(age_ax, intervals, colors)
    age_ax.plot(frame_t, data["note_age"], linewidth=1.1, label="note_age")
    if branches is not None and "_sustain_loudness_gain" in branches:
        gain = branches["_sustain_loudness_gain"]
        gain_t = _time_axis(gain.shape[-1], sr)
        gain_ax = age_ax.twinx()
        gain_ax.plot(gain_t, gain, linewidth=0.8, color="#d62728", alpha=0.7, label="sustain loudness gain")
        gain_ax.set_ylabel("gain")
    age_ax.set_title("note_age(t) and sustain gain")
    age_ax.set_ylabel("seconds")
    age_ax.grid(True, alpha=0.25)

    label_ax = axes[axis_idx]
    _draw_intervals(label_ax, intervals, colors, alpha=0.35)
    label_ax.set_title("articulation intervals")
    label_ax.set_ylim(0, 1)
    label_ax.set_yticks([])
    for interval in intervals:
        start = float(interval["start_seconds"])
        end = float(interval["end_seconds"])
        mid = (start + end) * 0.5
        label = interval.get("articulation", "")
        if end - start > 0.08:
            label_ax.text(
                mid,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                clip_on=True,
            )
    label_ax.set_xlabel("time (s)")

    duration = audio.shape[-1] / float(sr)
    for ax in axes:
        ax.set_xlim(0, duration)

    fig.savefig(sample_dir / "debug_controls.png", dpi=150)
    plt.close(fig)

    sf.write(sample_dir / "target.wav", audio, sr, subtype="FLOAT")
    if recon is not None:
        sf.write(sample_dir / "reconstruction.wav", recon.astype(np.float32), sr, subtype="FLOAT")
    _write_control_csv(sample_dir / "controls.csv", data)
    with open(sample_dir / "intervals.json", "w") as handle:
        json.dump(intervals, handle, indent=2)


def load_config(args):
    if args.run:
        run_dir = Path(args.run)
        config_path = run_dir / "config.yaml"
    else:
        run_dir = None
        config_path = Path(args.config)
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)
    return run_dir, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Optional trained run directory. If set, reconstruction is plotted.")
    parser.add_argument("--config", default="configs/bass_ddsp_v2_riff.yaml")
    parser.add_argument("--out-dir")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--index", type=int)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--pitch-source", choices=["labels", "torchcrepe"], default="labels")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir, config = load_config(args)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (run_dir / "control_debug" if run_dir else Path("debug/control_debug"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(config, args.seed, args.pitch_source)
    model = None
    if run_dir is not None and (run_dir / "state.pth").exists():
        model = load_model(config, run_dir, dataset, torch.device(args.device))

    if args.index is not None:
        indices = [int(args.index)]
    else:
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    summary = {
        "run": str(run_dir) if run_dir is not None else None,
        "config": args.config if run_dir is None else str(run_dir / "config.yaml"),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "indices": indices,
        "pitch_source": args.pitch_source,
        "samples": [],
    }
    for position, idx in enumerate(indices):
        data = dataset.generate_debug_example(idx, pitch_source=args.pitch_source)
        branches = reconstruct(model, config, data, torch.device(args.device)) if model else None
        sample_dir = out_dir / f"sample_{position:02d}_idx_{idx:04d}"
        write_debug_plot(sample_dir, data, branches, config, f"sample {position:02d} idx {idx:04d}")
        summary["samples"].append({
            "index": idx,
            "directory": str(sample_dir),
            "plot": str(sample_dir / "debug_controls.png"),
            "intervals": data.get("intervals", []),
        })

    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({
        "out_dir": str(out_dir),
        "indices": indices,
        "plots": [sample["plot"] for sample in summary["samples"]],
    }, indent=2))


if __name__ == "__main__":
    main()
