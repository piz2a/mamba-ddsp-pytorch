import argparse
import csv
import json
from pathlib import Path

import librosa as li
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml

from idmt_bass import IDMTBassRiffDataset


PLUCK_COLORS = {
    "FS": "#4c78a8",
    "MU": "#f58518",
    "PK": "#54a24b",
    "SP": "#e45756",
    "ST": "#72b7b2",
}

EXPRESSION_COLORS = {
    "NO": "#bab0ac",
    "BE": "#b279a2",
    "DN": "#9d755d",
    "HA": "#59a14f",
    "VI": "#edc948",
}


def make_dataset(config, args):
    idmt_config = dict(config.get("idmt_bass", {}))
    if args.seed is not None:
        idmt_config["seed"] = args.seed
    if args.pitch_source:
        idmt_config["pitch_source"] = args.pitch_source

    return IDMTBassRiffDataset(
        data_location=config["data"]["data_location"],
        sampling_rate=config["preprocess"]["sampling_rate"],
        block_size=config["preprocess"]["block_size"],
        signal_length=config["preprocess"]["signal_length"],
        **idmt_config,
    )


def write_intervals(path, intervals):
    fields = [
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "pluck",
        "expression",
        "frequency",
        "string",
        "fret",
        "crossfade_samples",
        "original_samples",
        "trim_start_sample",
        "trim_end_sample",
        "trimmed_samples",
        "trim_peak_rms",
        "trim_noise_rms",
        "trim_threshold",
        "trim_onset_threshold",
        "segment_samples",
        "target_samples",
        "cropped",
        "source_path",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for interval in intervals:
            writer.writerow({field: interval.get(field) for field in fields})


def draw_intervals(ax, intervals, y0, y1, colors, key, alpha=0.18, text=True):
    for interval in intervals:
        label = interval[key]
        start = interval["start_seconds"]
        end = interval["end_seconds"]
        ax.axvspan(start, end, color=colors.get(label, "#999999"), alpha=alpha, lw=0)
        ax.axvline(start, color="#222222", alpha=0.18, lw=0.7)
        if text and end - start > 0.18:
            ax.text(
                (start + end) * 0.5,
                y0 + (y1 - y0) * 0.88,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
                clip_on=True,
            )


def plot_debug(data, out_png, out_zoom_png=None):
    audio = data["audio"]
    sr = data["sampling_rate"]
    block_size = data["block_size"]
    intervals = data["intervals"]

    time = np.arange(audio.shape[0]) / sr
    frame_time = (np.arange(data["pitch"].shape[0]) * block_size + block_size / 2) / sr
    duration = audio.shape[0] / sr
    peak = max(float(np.max(np.abs(audio))), 1e-6)

    stft = li.stft(audio, n_fft=1024, hop_length=128, win_length=1024)
    spec_db = li.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0, 1.0, 2.2, 0.9]},
        constrained_layout=True,
    )

    ax = axes[0]
    ax.plot(time, audio, color="#1f2933", lw=0.65)
    draw_intervals(ax, intervals, -peak, peak, PLUCK_COLORS, "pluck")
    ax.set_ylim(-peak * 1.08, peak * 1.08)
    ax.set_ylabel("audio")
    ax.set_title("Generated IDMT-SMT-BASS riff debug view")

    ax = axes[1]
    ax.plot(frame_time, data["loudness"], color="#7c4dff", lw=1.1)
    draw_intervals(
        ax,
        intervals,
        float(np.min(data["loudness"])),
        float(np.max(data["loudness"])),
        EXPRESSION_COLORS,
        "expression",
        alpha=0.14,
        text=False,
    )
    ax.set_ylabel("loudness")

    ax = axes[2]
    ax.plot(frame_time, data["pitch"], label="pitch", color="#0b7285", lw=1.0)
    ax.plot(
        frame_time,
        data["label_pitch"],
        label="label pitch",
        color="#d9480f",
        lw=0.9,
        alpha=0.75,
    )
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylabel("Hz")

    ax = axes[3]
    img = librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=128,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=ax,
    )
    ax.set_ylim(20, 1600)
    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    ax.set_ylabel("STFT")

    ax = axes[4]
    ax.set_ylim(0, 2)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["ES", "PS"])
    for interval in intervals:
        start = interval["start_seconds"]
        end = interval["end_seconds"]
        ax.barh(
            1.5,
            end - start,
            left=start,
            height=0.8,
            color=PLUCK_COLORS.get(interval["pluck"], "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
        ax.barh(
            0.5,
            end - start,
            left=start,
            height=0.8,
            color=EXPRESSION_COLORS.get(interval["expression"], "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
        if end - start > 0.28:
            ax.text((start + end) * 0.5, 1.5, interval["pluck"],
                    ha="center", va="center", fontsize=8)
            ax.text((start + end) * 0.5, 0.5, interval["expression"],
                    ha="center", va="center", fontsize=8)
    ax.set_xlim(0, duration)
    ax.set_xlabel("time (s)")

    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    if out_zoom_png:
        first_seconds = min(1.5, duration)
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 5),
            sharex=True,
            constrained_layout=True,
        )
        zoom = time <= first_seconds
        axes[0].plot(time[zoom], audio[zoom], color="#1f2933", lw=0.8)
        draw_intervals(axes[0], intervals, -peak, peak, PLUCK_COLORS, "pluck")
        axes[0].set_xlim(0, first_seconds)
        axes[0].set_ylim(-peak * 1.08, peak * 1.08)
        axes[0].set_ylabel("audio")

        rms = li.feature.rms(
            y=audio,
            frame_length=512,
            hop_length=64,
            center=False,
        )[0]
        rms_time = np.arange(rms.shape[0]) * 64 / sr
        axes[1].plot(rms_time, rms, color="#7c4dff", lw=1.0)
        axes[1].set_xlim(0, first_seconds)
        axes[1].set_ylabel("RMS")
        axes[1].set_xlabel("time (s)")
        fig.savefig(out_zoom_png, dpi=160)
        plt.close(fig)


def summarize(data):
    intervals = data["intervals"]
    durations = np.asarray([item["duration_seconds"] for item in intervals])
    crossfades = np.asarray([item["crossfade_samples"] for item in intervals])
    return {
        "n_intervals": len(intervals),
        "duration_seconds": len(data["audio"]) / data["sampling_rate"],
        "peak": float(np.max(np.abs(data["audio"]))),
        "rms": float(np.sqrt(np.mean(data["audio"] ** 2))),
        "min_interval_seconds": float(durations.min()),
        "median_interval_seconds": float(np.median(durations)),
        "max_interval_seconds": float(durations.max()),
        "median_crossfade_ms": float(np.median(crossfades) / data["sampling_rate"] * 1000),
        "pluck_labels": data["pluck_labels"],
        "expression_labels": data["expression_labels"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_idmt_bass.yaml")
    parser.add_argument("--out-dir", default="debug/idmt_riff")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pitch-source", choices=["labels", "torchcrepe"])
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(config, args)
    data = dataset.generate_debug_riff(args.index, pitch_source=args.pitch_source)

    wav_path = out_dir / "riff.wav"
    png_path = out_dir / "riff_debug.png"
    zoom_path = out_dir / "riff_debug_zoom.png"
    csv_path = out_dir / "intervals.csv"
    json_path = out_dir / "riff_debug.json"

    sf.write(wav_path, data["audio"], data["sampling_rate"])
    write_intervals(csv_path, data["intervals"])
    plot_debug(data, png_path, zoom_path)

    summary = summarize(data)
    with open(json_path, "w") as handle:
        json.dump({"summary": summary, "intervals": data["intervals"]}, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {wav_path}")
    print(f"wrote {png_path}")
    print(f"wrote {zoom_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
