import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml

from bass_ddsp.dataset import IDMTBassNoteDataset, IDMTBassRiffDataset
from bass_ddsp.model import BassDDSPV2


def make_dataset(config):
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
        **config.get("idmt_bass", {}),
    )


def load_model(run_dir, config, n_articulation, device):
    model_config = dict(config["model"])
    model_config["n_articulation"] = n_articulation
    model = BassDDSPV2(**model_config).to(device)
    state = torch.load(run_dir / "state.pth", map_location=device)
    load_result = model.load_state_dict(state, strict=False)
    model.eval()
    return model, load_result


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def peak(x):
    return float(np.max(np.abs(x))) if x.size else 0.0


def synthesize_style_prototypes(model, labels, config, device, pitch_hz, seconds):
    sr = int(config["preprocess"]["sampling_rate"])
    block_size = int(config["preprocess"]["block_size"])
    frames = max(1, int(np.ceil(seconds * sr / block_size)))
    length = frames * block_size
    batch = len(labels)

    pitch = torch.full((batch, frames, 1), float(pitch_hz), device=device)
    loudness = torch.zeros(batch, frames, 1, device=device)
    articulation = torch.arange(batch, device=device).reshape(batch, 1).repeat(1, frames)
    onset = torch.zeros(batch, frames, 1, device=device)
    onset[:, 0, 0] = 1.0
    offset = torch.zeros(batch, frames, 1, device=device)
    gate = torch.ones(batch, frames, 1, device=device)
    note_age = (
        torch.arange(frames, device=device, dtype=torch.float32).reshape(1, frames, 1)
        * block_size
        / sr
    ).repeat(batch, 1, 1)

    with torch.no_grad():
        model(
            pitch,
            loudness,
            articulation=articulation,
            onset=onset,
            offset=offset,
            gate=gate,
            note_age=note_age,
        )
    transient = (
        model.last_branch_outputs["transient"]
        .squeeze(-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return transient[:, :length]


def write_style_grid(out_dir, labels, transients, sr, title, filename, max_ms=None):
    cols = 2
    rows = int(np.ceil(len(labels) / cols))
    if max_ms is None:
        plot_transients = transients
    else:
        sample_count = max(1, min(transients.shape[1], int(round(max_ms * sr / 1000.0))))
        plot_transients = transients[:, :sample_count]
    t_ms = np.arange(plot_transients.shape[1]) / sr * 1000.0
    fig, axes = plt.subplots(rows, cols, figsize=(13, max(3, rows * 2.4)), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    max_abs = max(float(np.max(np.abs(plot_transients))), 1e-6)

    for idx, label in enumerate(labels):
        ax = axes[idx]
        ax.plot(t_ms, plot_transients[idx], linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("ms")
        ax.set_ylabel("amplitude")
        ax.set_ylim(-max_abs * 1.05, max_abs * 1.05)
        ax.grid(True)
    for idx in range(len(labels), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title)
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def write_style_overlay(out_dir, labels, transients, sr, filename, max_ms=None):
    if max_ms is None:
        plot_transients = transients
    else:
        sample_count = max(1, min(transients.shape[1], int(round(max_ms * sr / 1000.0))))
        plot_transients = transients[:, :sample_count]
    t_ms = np.arange(plot_transients.shape[1]) / sr * 1000.0
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for label, audio in zip(labels, plot_transients):
        ax.plot(t_ms, audio, linewidth=1.0, label=label)
    ax.set_title("Transient style prototype overlay")
    ax.set_xlabel("ms")
    ax.set_ylabel("amplitude")
    ax.grid(True)
    ax.legend(ncol=2)
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def write_raw_waveform_bank(out_dir, model, labels, sr):
    if hasattr(model, "transient_bank"):
        bank = model.transient_bank.detach().cpu().numpy().astype(np.float32)
        title = "Raw learned transient_bank rows"
    elif hasattr(model, "transient_dct_bank_coeff"):
        with torch.no_grad():
            bank = (
                model.transient_dct_bank_waveforms()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        title = "Raw learned DCT-bank transient waveforms"
    else:
        return None
    write_style_grid(
        out_dir,
        labels[: bank.shape[0]],
        bank,
        sr,
        title,
        "raw_waveform_bank.png",
    )
    write_style_overlay(
        out_dir,
        labels[: bank.shape[0]],
        bank,
        sr,
        "raw_waveform_bank_overlay.png",
        max_ms=30.0,
    )
    np.save(out_dir / "raw_waveform_bank.npy", bank)
    return "raw_waveform_bank.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pitch-hz", type=float, default=82.41)
    parser.add_argument("--seconds", type=float, default=0.25)
    parser.add_argument("--zoom-ms", type=float, default=30.0)
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "transient_style_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "r") as handle:
        config = yaml.safe_load(handle)
    dataset = make_dataset(config)
    labels = list(dataset.articulation_labels)
    device = torch.device(args.device)
    model, load_result = load_model(run_dir, config, len(labels), device)

    transients = synthesize_style_prototypes(
        model,
        labels,
        config,
        device,
        pitch_hz=args.pitch_hz,
        seconds=args.seconds,
    )
    sr = int(config["preprocess"]["sampling_rate"])

    for label, audio in zip(labels, transients):
        safe_label = label.replace("/", "_")
        sf.write(out_dir / f"{safe_label}_transient.wav", audio, sr, subtype="FLOAT")
    np.save(out_dir / "transient_style_prototypes.npy", transients)
    write_style_grid(
        out_dir,
        labels,
        transients,
        sr,
        f"{model.transient_type} transient style prototypes",
        "transient_style_prototypes.png",
    )
    write_style_grid(
        out_dir,
        labels,
        transients,
        sr,
        f"{model.transient_type} transient style prototypes, first {args.zoom_ms:g} ms",
        "transient_style_prototypes_zoom.png",
        max_ms=args.zoom_ms,
    )
    write_style_overlay(
        out_dir,
        labels,
        transients,
        sr,
        "transient_style_prototypes_overlay.png",
        max_ms=args.zoom_ms,
    )
    raw_bank_plot = write_raw_waveform_bank(out_dir, model, labels, sr)

    rows = []
    for label, audio in zip(labels, transients):
        rows.append({
            "label": label,
            "rms": rms(audio),
            "peak": peak(audio),
        })
    summary = {
        "run": str(run_dir),
        "out_dir": str(out_dir),
        "transient_type": model.transient_type,
        "pitch_hz": args.pitch_hz,
        "seconds": args.seconds,
        "zoom_ms": args.zoom_ms,
        "labels": labels,
        "metrics": rows,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "raw_waveform_bank_plot": raw_bank_plot,
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
