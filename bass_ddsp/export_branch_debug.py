import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from bass_ddsp.dataset import IDMTBassNoteDataset, IDMTBassRiffDataset
from bass_ddsp.model import BassDDSPV2


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
    model_config["n_articulation"] = dataset.n_articulation
    model = BassDDSPV2(**model_config).to(device)
    state = torch.load(run_dir / "state.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def peak(audio):
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def reconstruct(model, config, data, device):
    pitch = torch.from_numpy(data["pitch"]).float().unsqueeze(0).unsqueeze(-1).to(device)
    loudness = torch.from_numpy(data["loudness"]).float().unsqueeze(0).unsqueeze(-1)
    loudness = (
        loudness - float(config["data"]["mean_loudness"])
    ) / float(config["data"]["std_loudness"])
    loudness = loudness.to(device)
    kwargs = {
        "articulation": torch.from_numpy(data["articulation"]).long().unsqueeze(0).to(device),
        "onset": torch.from_numpy(data["onset"]).float().unsqueeze(0).unsqueeze(-1).to(device),
        "offset": torch.from_numpy(data["offset"]).float().unsqueeze(0).unsqueeze(-1).to(device),
        "gate": torch.from_numpy(data["gate"]).float().unsqueeze(0).unsqueeze(-1).to(device),
        "note_age": torch.from_numpy(data["note_age"]).float().unsqueeze(0).unsqueeze(-1).to(device),
        "note_progress": torch.from_numpy(data["note_progress"]).float().unsqueeze(0).unsqueeze(-1).to(device),
    }
    with torch.no_grad():
        signal = model(pitch, loudness, **kwargs)

    branches = {}
    for name, tensor in model.last_branch_outputs.items():
        branches[name] = tensor.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
    branches.setdefault(
        "signal",
        signal.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32),
    )
    return branches


def write_sample(out_dir, sample_name, data, branches):
    sample_dir = out_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    sr = data["sampling_rate"]
    names = [
        "target",
        "signal",
        "sustain",
        "noise",
        "transient",
        "sustain_raw",
        "noise_raw",
        "transient_raw",
    ]
    audios = {"target": data["audio"].astype(np.float32)}
    for name in names:
        if name == "target":
            continue
        audios[name] = branches.get(name, np.zeros_like(branches["signal"]))
    for name in names:
        sf.write(sample_dir / f"{name}.wav", audios[name], sr, subtype="FLOAT")
    sf.write(
        sample_dir / "target_signal_sustain_noise_transient.wav",
        np.concatenate([audios[name] for name in ["target", "signal", "sustain", "noise", "transient"]]),
        sr,
        subtype="FLOAT",
    )

    rows = []
    signal_rms = max(rms(audios["signal"]), 1e-12)
    for name in names:
        value_rms = rms(audios[name])
        rows.append({
            "name": name,
            "rms": value_rms,
            "peak": peak(audios[name]),
            "rms_vs_signal_percent": value_rms / signal_rms * 100.0,
        })
    with open(sample_dir / "branch_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "rms", "peak", "rms_vs_signal_percent"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(sample_dir / "intervals.json", "w") as handle:
        json.dump(data["intervals"], handle, indent=2)
    return {"sample": sample_name, "metrics": rows, "intervals": data["intervals"]}


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
        "branch_gain_db": model.branch_gain_db(),
        "samples": [],
    }
    for position, idx in enumerate(indices):
        data = dataset.generate_debug_example(idx, pitch_source=args.pitch_source)
        branches = reconstruct(model, config, data, device)
        summary["samples"].append(
            write_sample(out_dir, f"sample_{position:02d}_idx_{idx:04d}", data, branches)
        )

    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({
        "out_dir": str(out_dir),
        "indices": indices,
        "branch_gain_db": summary["branch_gain_db"],
    }, indent=2))


if __name__ == "__main__":
    main()
