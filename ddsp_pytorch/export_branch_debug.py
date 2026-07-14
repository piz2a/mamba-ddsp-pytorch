import argparse
import csv
import json
import random
from pathlib import Path

import librosa as li
import librosa.display as lidisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml

from ddsp.model import DDSP
from idmt_bass import IDMTBassNoteDataset, IDMTBassRiffDataset
from visualize_idmt_riff import (
    EXPRESSION_COLORS,
    PLUCK_COLORS,
    draw_intervals,
    write_intervals,
)


def make_dataset(config, seed, pitch_source):
    idmt_config = dict(config.get("idmt_bass", {}))
    idmt_config["seed"] = seed
    if pitch_source:
        idmt_config["pitch_source"] = pitch_source

    dataset_cls = (
        IDMTBassNoteDataset
        if config.get("data", {}).get("dataset") == "idmt_bass_note"
        else IDMTBassRiffDataset
    )
    return dataset_cls(
        data_location=config["data"]["data_location"],
        sampling_rate=config["preprocess"]["sampling_rate"],
        block_size=config["preprocess"]["block_size"],
        signal_length=config["preprocess"]["signal_length"],
        **idmt_config,
    )


def load_model(config, run_dir, dataset, device):
    model_config = dict(config["model"])
    model_config.setdefault("n_pluck", dataset.n_pluck)
    model_config.setdefault("n_expression", dataset.n_expression)
    model_config.setdefault("n_articulation", dataset.n_articulation)
    model = DDSP(**model_config).to(device)
    state = torch.load(run_dir / "state.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def reconstruct_with_branches(model, config, data, device):
    pitch = torch.from_numpy(data["pitch"]).float().unsqueeze(0).unsqueeze(-1).to(device)
    loudness = torch.from_numpy(data["loudness"]).float().unsqueeze(0).unsqueeze(-1)
    loudness = (
        loudness - float(config["data"]["mean_loudness"])
    ) / float(config["data"]["std_loudness"])
    loudness = loudness.to(device)
    onset = torch.from_numpy(data["onset"]).float().unsqueeze(0).unsqueeze(-1).to(device)
    offset = torch.from_numpy(data["offset"]).float().unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        if config["model"].get("architecture") == "bass_ddsp_v2":
            articulation = torch.from_numpy(data["articulation"]).long().unsqueeze(0).to(device)
            gate = torch.from_numpy(data["gate"]).float().unsqueeze(0).unsqueeze(-1).to(device)
            note_age = torch.from_numpy(data["note_age"]).float().unsqueeze(0).unsqueeze(-1).to(device)
            note_progress = torch.from_numpy(data["note_progress"]).float().unsqueeze(0).unsqueeze(-1).to(device)
            signal = model(
                pitch,
                loudness,
                articulation=articulation,
                onset=onset,
                offset=offset,
                gate=gate,
                note_age=note_age,
                note_progress=note_progress,
            )
        else:
            pluck = torch.from_numpy(data["pluck"]).long().unsqueeze(0).to(device)
            expression = torch.from_numpy(data["expression"]).long().unsqueeze(0).to(device)
            signal = model(pitch, loudness, pluck, expression, onset, offset)

    branches = {}
    for name, tensor in model.last_branch_outputs.items():
        branches[name] = tensor.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
    branches.setdefault(
        "signal",
        signal.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32),
    )
    return branches


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def peak(audio):
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def plot_branch_debug(data, branches, out_png):
    sr = data["sampling_rate"]
    block_size = data["block_size"]
    target = data["audio"]
    signal = branches["signal"]
    names = ["target", "signal", "sustain", "transient", "noise"]
    audios = {
        "target": target,
        "signal": signal,
        "sustain": branches.get("sustain", np.zeros_like(signal)),
        "transient": branches.get("transient", np.zeros_like(signal)),
        "noise": branches.get("noise", np.zeros_like(signal)),
    }

    length = min(len(audio) for audio in audios.values())
    audios = {name: audio[:length] for name, audio in audios.items()}
    time = np.arange(length) / sr
    frame_time = (np.arange(data["pitch"].shape[0]) * block_size + block_size / 2) / sr
    plot_peak = max(peak(audio) for audio in audios.values())
    plot_peak = max(plot_peak, 1e-4)

    fig, axes = plt.subplots(
        8,
        1,
        figsize=(16, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 0.75, 1.6, 0.8]},
        constrained_layout=True,
    )
    colors = {
        "target": "#1f2933",
        "signal": "#0b7285",
        "sustain": "#364fc7",
        "transient": "#e67700",
        "noise": "#5c940d",
    }

    for ax, name in zip(axes[:5], names):
        ax.plot(time, audios[name], color=colors[name], lw=0.65)
        draw_intervals(ax, data["intervals"], -plot_peak, plot_peak, PLUCK_COLORS, "pluck")
        ax.set_ylim(-plot_peak * 1.08, plot_peak * 1.08)
        ax.set_ylabel(name)
        ax.text(
            0.99,
            0.82,
            f"rms={rms(audios[name]):.5f}, peak={peak(audios[name]):.5f}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )

    axes[0].set_title("Bass-DDSP branch contribution debug")
    axes[5].plot(frame_time, data["pitch"], color="#0b7285", lw=1.0, label="pitch input")
    axes[5].plot(frame_time, data["label_pitch"], color="#d9480f", lw=0.9, label="label")
    axes[5].plot(frame_time, data["gate"] * max(float(np.max(data["label_pitch"])), 1.0),
                 color="#343a40", lw=0.8, alpha=0.6, label="gate scaled")
    axes[5].legend(loc="upper right", frameon=False)
    axes[5].set_ylabel("Hz")

    signal_db = li.amplitude_to_db(
        np.abs(li.stft(signal[:length], n_fft=1024, hop_length=128, win_length=1024)),
        ref=np.max,
    )
    img = lidisplay.specshow(
        signal_db,
        sr=sr,
        hop_length=128,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=axes[6],
    )
    axes[6].set_ylim(20, 1600)
    axes[6].set_ylabel("signal STFT")
    fig.colorbar(img, ax=axes[6], format="%+2.0f dB", pad=0.01)

    axes[7].set_ylim(0, 2)
    axes[7].set_yticks([0.5, 1.5])
    axes[7].set_yticklabels(["ES", "PS"])
    for interval in data["intervals"]:
        start = interval["start_seconds"]
        end = interval["end_seconds"]
        axes[7].barh(
            1.5,
            end - start,
            left=start,
            height=0.8,
            color=PLUCK_COLORS.get(interval["pluck"], "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
        axes[7].barh(
            0.5,
            end - start,
            left=start,
            height=0.8,
            color=EXPRESSION_COLORS.get(interval["expression"], "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
        if end - start > 0.24:
            axes[7].text((start + end) * 0.5, 1.5, interval["pluck"],
                         ha="center", va="center", fontsize=8)
            axes[7].text((start + end) * 0.5, 0.5, interval["expression"],
                         ha="center", va="center", fontsize=8)
    axes[7].set_xlim(0, length / sr)
    axes[7].set_xlabel("time (s)")

    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_sample(out_dir, sample_name, data, branches):
    sample_dir = out_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    sr = data["sampling_rate"]

    audio_names = ["target", "signal", "sustain", "transient", "noise"]
    audios = {
        "target": data["audio"].astype(np.float32),
        "signal": branches["signal"],
        "sustain": branches.get("sustain", np.zeros_like(branches["signal"])),
        "transient": branches.get("transient", np.zeros_like(branches["signal"])),
        "noise": branches.get("noise", np.zeros_like(branches["signal"])),
    }
    for name in audio_names:
        sf.write(sample_dir / f"{name}.wav", audios[name], sr, subtype="FLOAT")

    sf.write(
        sample_dir / "target_signal_sustain_transient_noise.wav",
        np.concatenate([audios[name] for name in audio_names]),
        sr,
        subtype="FLOAT",
    )
    write_intervals(sample_dir / "intervals.csv", data["intervals"])
    plot_branch_debug(data, branches, sample_dir / "branch_debug.png")

    rows = []
    for name in audio_names:
        rows.append({
            "name": name,
            "rms": rms(audios[name]),
            "peak": peak(audios[name]),
        })
    with open(sample_dir / "branch_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "rms", "peak"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "sample": sample_name,
        "metrics": rows,
        "intervals": data["intervals"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--pitch-source", choices=["labels", "torchcrepe"], default="labels")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "branch_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "r") as handle:
        config = yaml.safe_load(handle)

    device = torch.device(args.device)
    dataset = make_dataset(config, args.seed, args.pitch_source)
    model = load_model(config, run_dir, dataset, device)

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    summary = {
        "run": str(run_dir),
        "seed": args.seed,
        "pitch_source": args.pitch_source,
        "indices": indices,
        "samples": [],
        "branch_notes": {
            "signal": "final model output after optional reverb",
            "sustain": "harmonic additive DDSP branch",
            "noise": "filtered noise branch; stochastic per forward pass",
            "transient": "learned articulation-conditioned transient waveform bank",
        },
    }

    for position, idx in enumerate(indices):
        data = dataset.generate_debug_example(idx, pitch_source=args.pitch_source)
        branches = reconstruct_with_branches(model, config, data, device)
        sample_name = f"sample_{position:02d}_idx_{idx:04d}"
        summary["samples"].append(write_sample(out_dir, sample_name, data, branches))

    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps({
        "out_dir": str(out_dir),
        "indices": indices,
        "samples": [
            {
                "sample": sample["sample"],
                "metrics": sample["metrics"],
            }
            for sample in summary["samples"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
