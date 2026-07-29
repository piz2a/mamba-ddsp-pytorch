#!/usr/bin/env python3
"""Reproducible latency benchmark for the three trained DDSP models."""

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bass_ddsp.train import clean_state_dict, make_model


MODELS = {
    "Bass-DDSP": ROOT / "runs/bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513",
    "Vanilla DWTS": ROOT / "runs/vanilla_dwts_riff_resume_to_200k_20260726_071231",
    "Vanilla DDSP": ROOT / "runs/vanilla_ddsp_riff_20260720_090532",
}
CONTROL_NAMES = (
    "pitch",
    "loudness",
    "articulation",
    "onset_strength",
    "offset",
    "gate",
    "note_age",
    "periodicity",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--chunk-warmup", type=int, default=30)
    parser.add_argument("--chunk-iterations", type=int, default=200)
    parser.add_argument("--frame-warmup", type=int, default=100)
    parser.add_argument("--frame-iterations", type=int, default=1000)
    parser.add_argument("--cpu-chunk-iterations", type=int, default=80)
    parser.add_argument("--cpu-frame-iterations", type=int, default=400)
    return parser.parse_args()


def load_model(run_dir, device):
    with (run_dir / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    model = make_model(config)
    state = clean_state_dict(torch.load(run_dir / "state.pth", map_location="cpu"))
    model.load_state_dict(state)
    model.eval().to(device)
    return model, config


def make_controls(config, device):
    frames = config["preprocess"]["signal_length"] // config["preprocess"]["block_size"]
    t = torch.arange(frames, dtype=torch.float32, device=device)
    gate = ((t % 48) < 40).float()
    note_phase = t % 48
    controls = {
        "pitch": (55.0 * torch.pow(2.0, ((t // 24) % 5) / 12.0)).reshape(1, frames, 1),
        "loudness": (0.35 * torch.sin(t * 0.11)).reshape(1, frames, 1),
        "articulation": ((t // 24) % 6).long().reshape(1, frames),
        "onset_strength": (note_phase == 0).float().reshape(1, frames, 1),
        "offset": (note_phase == 40).float().reshape(1, frames, 1),
        "gate": gate.reshape(1, frames, 1),
        "note_age": ((note_phase * config["preprocess"]["block_size"]) / config["preprocess"]["sampling_rate"]).reshape(1, frames, 1),
        "periodicity": (0.82 * gate + 0.05).reshape(1, frames, 1),
    }
    return controls


def frame_controls(controls):
    frames = controls["pitch"].shape[1]
    return [
        {
            key: value[:, index:index + 1]
            for key, value in controls.items()
        }
        for index in range(frames)
    ]


def reset_stream(model):
    if hasattr(model, "reset_recurrent_state"):
        model.reset_recurrent_state()
    phase = getattr(model, "phase", None)
    if phase is not None:
        phase.zero_()


def call_full(model, controls):
    return model(**controls)


def call_frame(model, controls, stateful):
    if stateful:
        return model.realtime_forward(**controls)
    return model(**controls)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(values, audio_ms):
    array = np.asarray(values, dtype=np.float64)
    return {
        "iterations": int(array.size),
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std()),
        "min_ms": float(array.min()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
        "audio_ms": float(audio_ms),
        "rtf_p50": float(np.percentile(array, 50) / audio_ms),
        "rtf_p99": float(np.percentile(array, 99) / audio_ms),
        "deadline_miss_percent": float(np.mean(array > audio_ms) * 100.0),
    }


def benchmark(model, controls, device, mode, warmup, iterations, stateful):
    frames = frame_controls(controls)

    def invoke(index):
        if mode == "full_chunk":
            return call_full(model, controls)
        return call_frame(model, frames[index % len(frames)], stateful)

    reset_stream(model)
    with torch.inference_mode():
        for index in range(warmup):
            invoke(index)
        synchronize(device)

        wall_ms = []
        cuda_ms = []
        for index in range(iterations):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            start_ns = time.perf_counter_ns()
            invoke(index)
            if device.type == "cuda":
                end_event.record()
            synchronize(device)
            wall_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
            if device.type == "cuda":
                cuda_ms.append(start_event.elapsed_time(end_event))
    return wall_ms, cuda_ms


def nvidia_smi(gpu):
    fields = "index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,pstate"
    command = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu),
    ]
    return subprocess.check_output(command, text=True).strip()


def write_plot(results, output):
    import matplotlib.pyplot as plt

    labels = list(MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, mode, title in (
        (axes[0], "one_frame", "One 16 ms control frame"),
        (axes[1], "full_chunk", "Full 2.048 s chunk"),
    ):
        x = np.arange(len(labels))
        width = 0.25
        for offset, quantile in enumerate(("p50_ms", "p95_ms", "p99_ms")):
            values = [
                results[name]["cuda"][mode]["wall"][quantile]
                for name in labels
            ]
            axis.bar(x + (offset - 1) * width, values, width, label=quantile[:-3])
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.set_ylabel("Synchronized wall latency (ms)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if mode == "one_frame":
            axis.axhline(16.0, color="crimson", linestyle="--", label="16 ms deadline")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "latency_quantiles.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.gpu)
    devices = [("cuda", torch.device(f"cuda:{args.gpu}")), ("cpu", torch.device("cpu"))]
    results = {}
    raw_rows = []

    for name, run_dir in MODELS.items():
        results[name] = {}
        for device_name, device in devices:
            model, config = load_model(run_dir, device)
            controls = make_controls(config, device)
            sample_rate = config["preprocess"]["sampling_rate"]
            block_size = config["preprocess"]["block_size"]
            frame_ms = 1000.0 * block_size / sample_rate
            chunk_ms = frame_ms * controls["pitch"].shape[1]
            stateful = name == "Bass-DDSP"
            results[name][device_name] = {
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "stateful_streaming": stateful,
            }

            modes = (
                ("one_frame", frame_ms, args.frame_warmup,
                 args.frame_iterations if device_name == "cuda" else args.cpu_frame_iterations),
                ("full_chunk", chunk_ms, args.chunk_warmup,
                 args.chunk_iterations if device_name == "cuda" else args.cpu_chunk_iterations),
            )
            for mode, audio_ms, warmup, iterations in modes:
                wall, cuda = benchmark(
                    model, controls, device, mode, warmup, iterations, stateful
                )
                entry = {"wall": summarize(wall, audio_ms)}
                if cuda:
                    entry["device_execution"] = summarize(cuda, audio_ms)
                results[name][device_name][mode] = entry
                for index, value in enumerate(wall):
                    raw_rows.append({
                        "model": name,
                        "device": device_name,
                        "mode": mode,
                        "measurement": "synchronized_wall",
                        "iteration": index,
                        "latency_ms": value,
                    })
                for index, value in enumerate(cuda):
                    raw_rows.append({
                        "model": name,
                        "device": device_name,
                        "mode": mode,
                        "measurement": "cuda_event",
                        "iteration": index,
                        "latency_ms": value,
                    })
            del model, controls
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": nvidia_smi(args.gpu),
        "cpu_threads": 1,
        "batch_size": 1,
        "dtype": "float32",
        "control_hop_ms": 16.0,
        "chunk_seconds": 2.048,
        "warmup": {
            "chunk": args.chunk_warmup,
            "frame": args.frame_warmup,
        },
    }
    with (output / "summary.json").open("w") as handle:
        json.dump({"metadata": metadata, "results": results}, handle, indent=2)
    with (output / "latency_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)
    write_plot(results, output)
    print(json.dumps({"metadata": metadata, "results": results}, indent=2))


if __name__ == "__main__":
    main()
