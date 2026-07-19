import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from bass_ddsp.model import BassDDSPV2


def load_run(run_dir, device):
    with open(run_dir / "config.yaml", "r") as handle:
        config = yaml.safe_load(handle)
    labels = config.get("data", {}).get("articulation_labels") or ["FS_NO"]
    model_config = dict(config["model"])
    model_config["n_articulation"] = len(labels)
    model = BassDDSPV2(**model_config).to(device)
    state = torch.load(run_dir / "state.pth", map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return config, labels, model


def pitch_curve(kind, frames, start_hz, end_hz):
    u = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    if kind == "steady":
        return np.full(frames, start_hz, dtype=np.float32)
    if kind == "bend_up":
        curve = 0.5 - 0.5 * np.cos(np.pi * u)
        return start_hz + (end_hz - start_hz) * curve
    if kind == "bend_down":
        curve = 0.5 - 0.5 * np.cos(np.pi * u)
        return end_hz + (start_hz - end_hz) * curve
    if kind == "slide_up_down":
        curve = np.sin(np.pi * u)
        return start_hz + (end_hz - start_hz) * curve
    raise ValueError(f"unknown curve kind: {kind}")


def make_controls(config, labels, articulation, kind, seconds, start_hz, end_hz, loudness):
    sr = int(config["preprocess"]["sampling_rate"])
    block_size = int(config["preprocess"]["block_size"])
    frames = max(1, int(np.ceil(seconds * sr / block_size)))
    note_age = np.arange(frames, dtype=np.float32) * block_size / sr
    onset_strength = np.exp(-note_age / 0.08).astype(np.float32)
    onset_strength[note_age > 0.30] = 0.0

    if articulation in labels:
        articulation_id = labels.index(articulation)
    else:
        articulation_id = 0

    controls = {
        "pitch": pitch_curve(kind, frames, start_hz, end_hz),
        "loudness": np.full(frames, loudness, dtype=np.float32),
        "articulation": np.full(frames, articulation_id, dtype=np.int64),
        "onset_strength": onset_strength,
        "offset": np.zeros(frames, dtype=np.float32),
        "gate": np.ones(frames, dtype=np.float32),
        "note_age": note_age,
        "periodicity": np.ones(frames, dtype=np.float32),
    }
    return controls


def to_tensor(controls, device):
    out = {}
    for key, value in controls.items():
        tensor = torch.from_numpy(value).unsqueeze(0).to(device)
        if key == "articulation":
            out[key] = tensor.long()
        else:
            out[key] = tensor.float().unsqueeze(-1)
    return out


def write_case(model, config, labels, args, kind, out_dir, device):
    controls = make_controls(
        config,
        labels,
        args.articulation,
        kind,
        args.seconds,
        args.start_hz,
        args.end_hz,
        args.normalized_loudness,
    )
    tensors = to_tensor(controls, device)
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
    audio = signal.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)
    sr = int(config["preprocess"]["sampling_rate"])
    sf.write(out_dir / f"{kind}.wav", audio, sr, subtype="FLOAT")
    np.savez(out_dir / f"{kind}_controls.npz", **controls)
    return {
        "kind": kind,
        "wav": str(out_dir / f"{kind}.wav"),
        "controls": str(out_dir / f"{kind}_controls.npz"),
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "rms": float(np.sqrt(np.mean(audio * audio) + 1e-12)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--start-hz", type=float, default=82.41)
    parser.add_argument("--end-hz", type=float, default=110.0)
    parser.add_argument("--normalized-loudness", type=float, default=0.0)
    parser.add_argument("--articulation", default="FS_NO")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "bend_slide_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config, labels, model = load_run(run_dir, device)
    cases = [
        write_case(model, config, labels, args, kind, out_dir, device)
        for kind in ["steady", "bend_up", "bend_down", "slide_up_down"]
    ]
    summary = {
        "run": str(run_dir),
        "out_dir": str(out_dir),
        "articulation": args.articulation,
        "articulation_labels": labels,
        "cases": cases,
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
