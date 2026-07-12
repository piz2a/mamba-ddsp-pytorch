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
import torch
import yaml

from ddsp.core import multiscale_fft, safe_log
from ddsp.model import DDSP
from idmt_bass import IDMTBassRiffDataset
from visualize_idmt_riff import (
    EXPRESSION_COLORS,
    PLUCK_COLORS,
    draw_intervals,
    summarize,
    write_intervals,
)


def make_dataset(config, seed, pitch_source):
    idmt_config = dict(config.get("idmt_bass", {}))
    idmt_config["seed"] = seed
    if pitch_source:
        idmt_config["pitch_source"] = pitch_source

    return IDMTBassRiffDataset(
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
    model = DDSP(**model_config).to(device)
    state = torch.load(run_dir / "state.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def reconstruct(model, config, data, device):
    pitch = torch.from_numpy(data["pitch"]).float().unsqueeze(0).unsqueeze(-1).to(device)
    loudness = torch.from_numpy(data["loudness"]).float().unsqueeze(0).unsqueeze(-1)
    loudness = (
        loudness - float(config["data"]["mean_loudness"])
    ) / float(config["data"]["std_loudness"])
    loudness = loudness.to(device)
    pluck = torch.from_numpy(data["pluck"]).long().unsqueeze(0).to(device)
    expression = torch.from_numpy(data["expression"]).long().unsqueeze(0).to(device)
    onset = torch.from_numpy(data["onset"]).float().unsqueeze(0).unsqueeze(-1).to(device)
    offset = torch.from_numpy(data["offset"]).float().unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        audio = model(
            pitch,
            loudness,
            pluck,
            expression,
            onset,
            offset,
        ).squeeze(0).squeeze(-1)
    return audio.detach().cpu().numpy().astype(np.float32)


def spectral_loss(target, reconstruction, scales, overlap, device):
    target_t = torch.from_numpy(target).float().unsqueeze(0).to(device)
    recon_t = torch.from_numpy(reconstruction).float().unsqueeze(0).to(device)
    with torch.no_grad():
        target_stft = multiscale_fft(target_t, scales, overlap)
        recon_stft = multiscale_fft(recon_t, scales, overlap)
        loss = 0.0
        for sx, sy in zip(target_stft, recon_stft):
            loss = loss + (sx - sy).abs().mean()
            loss = loss + (safe_log(sx) - safe_log(sy)).abs().mean()
    return float(loss.detach().cpu())


def plot_loss_curve(loss_csv, out_png):
    if not loss_csv.exists():
        return
    steps = []
    losses = []
    with open(loss_csv, "r") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    if not steps:
        return

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(steps, losses, color="#0b7285", lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("multiscale spectral loss")
    ax.set_title("Training loss")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_reconstruction(data, reconstruction, out_png):
    target = data["audio"]
    sr = data["sampling_rate"]
    block_size = data["block_size"]
    intervals = data["intervals"]
    duration = target.shape[0] / sr
    peak = max(float(np.max(np.abs(target))), float(np.max(np.abs(reconstruction))), 1e-6)
    time = np.arange(target.shape[0]) / sr
    frame_time = (np.arange(data["pitch"].shape[0]) * block_size + block_size / 2) / sr

    target_db = li.amplitude_to_db(
        np.abs(li.stft(target, n_fft=1024, hop_length=128, win_length=1024)),
        ref=np.max,
    )
    recon_db = li.amplitude_to_db(
        np.abs(li.stft(reconstruction, n_fft=1024, hop_length=128, win_length=1024)),
        ref=np.max,
    )

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(16, 14.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.1, 0.75, 0.75, 1.8, 1.8, 0.8]},
        constrained_layout=True,
    )

    axes[0].plot(time, target, color="#1f2933", lw=0.65)
    draw_intervals(axes[0], intervals, -peak, peak, PLUCK_COLORS, "pluck")
    axes[0].set_ylim(-peak * 1.08, peak * 1.08)
    axes[0].set_ylabel("target")
    axes[0].set_title("DDSP reconstruction debug view")

    axes[1].plot(time, reconstruction, color="#0b7285", lw=0.65)
    draw_intervals(axes[1], intervals, -peak, peak, PLUCK_COLORS, "pluck")
    axes[1].set_ylim(-peak * 1.08, peak * 1.08)
    axes[1].set_ylabel("recon")

    pitch_label = f"pitch input ({data.get('pitch_source', 'unknown')})"
    axes[2].plot(frame_time, data["pitch"], color="#0b7285", lw=1.0, label=pitch_label)
    axes[2].plot(frame_time, data["label_pitch"], color="#d9480f", lw=0.9, label="label")
    axes[2].legend(loc="upper right", frameon=False)
    axes[2].set_ylabel("Hz")

    axes[3].plot(frame_time, data["onset"], color="#2f9e44", lw=1.0, label="onset")
    axes[3].plot(frame_time, data["offset"], color="#c92a2a", lw=1.0, label="offset")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(loc="upper right", frameon=False)
    axes[3].set_ylabel("events")

    img = librosa.display.specshow(
        target_db,
        sr=sr,
        hop_length=128,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=axes[4],
    )
    axes[4].set_ylim(20, 1600)
    axes[4].set_ylabel("target STFT")
    fig.colorbar(img, ax=axes[4], format="%+2.0f dB", pad=0.01)

    img = librosa.display.specshow(
        recon_db,
        sr=sr,
        hop_length=128,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=axes[5],
    )
    axes[5].set_ylim(20, 1600)
    axes[5].set_ylabel("recon STFT")
    fig.colorbar(img, ax=axes[5], format="%+2.0f dB", pad=0.01)

    axes[6].set_ylim(0, 2)
    axes[6].set_yticks([0.5, 1.5])
    axes[6].set_yticklabels(["ES", "PS"])
    for interval in intervals:
        start = interval["start_seconds"]
        end = interval["end_seconds"]
        axes[6].barh(
            1.5,
            end - start,
            left=start,
            height=0.8,
            color=PLUCK_COLORS.get(interval["pluck"], "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
        axes[6].barh(
            0.5,
            end - start,
            left=start,
            height=0.8,
            color=EXPRESSION_COLORS.get(interval["expression"], "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
        if end - start > 0.28:
            axes[6].text((start + end) * 0.5, 1.5, interval["pluck"],
                         ha="center", va="center", fontsize=8)
            axes[6].text((start + end) * 0.5, 0.5, interval["expression"],
                         ha="center", va="center", fontsize=8)
    axes[6].set_xlim(0, duration)
    axes[6].set_xlabel("time (s)")

    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--pitch-source", choices=["labels", "torchcrepe"], default="torchcrepe")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "r") as handle:
        config = yaml.safe_load(handle)

    device = torch.device(args.device)
    dataset = make_dataset(config, args.seed, args.pitch_source)
    data = dataset.generate_debug_riff(args.index, pitch_source=args.pitch_source)
    model = load_model(config, run_dir, dataset, device)
    reconstruction = reconstruct(model, config, data, device)

    loss = spectral_loss(
        data["audio"],
        reconstruction,
        config["train"]["scales"],
        config["train"]["overlap"],
        device,
    )

    sf.write(out_dir / "target.wav", data["audio"], data["sampling_rate"], subtype="FLOAT")
    sf.write(out_dir / "reconstruction.wav", reconstruction, data["sampling_rate"], subtype="FLOAT")
    sf.write(
        out_dir / "target_then_reconstruction.wav",
        np.concatenate([data["audio"], reconstruction]),
        data["sampling_rate"],
        subtype="FLOAT",
    )
    write_intervals(out_dir / "intervals.csv", data["intervals"])
    plot_reconstruction(data, reconstruction, out_dir / "reconstruction_debug.png")
    plot_loss_curve(run_dir / "loss.csv", out_dir / "loss.png")

    summary = summarize(data)
    summary["reconstruction_peak"] = float(np.max(np.abs(reconstruction)))
    summary["reconstruction_rms"] = float(np.sqrt(np.mean(reconstruction ** 2)))
    summary["spectral_loss"] = loss
    with open(out_dir / "summary.json", "w") as handle:
        json.dump({"summary": summary, "intervals": data["intervals"]}, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'target.wav'}")
    print(f"wrote {out_dir / 'reconstruction.wav'}")
    print(f"wrote {out_dir / 'reconstruction_debug.png'}")
    print(f"wrote {out_dir / 'loss.png'}")


if __name__ == "__main__":
    main()
