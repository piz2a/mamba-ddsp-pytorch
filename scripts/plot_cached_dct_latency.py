#!/usr/bin/env python3
"""Replot latency with an inference-cached Bass-DDSP transient DCT bank."""

import copy
import csv
import json
import sys
import types
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bass_ddsp.train import clean_state_dict, make_model
import scripts.benchmark_ddsp_latency as latency_benchmark
from scripts.benchmark_ddsp_latency import (
    benchmark,
    make_controls,
    summarize,
    write_plot,
)


RUN_DIR = (
    ROOT
    / "runs"
    / "bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513"
)
OUTPUT_DIR = ROOT / "runs" / "ddsp_realtime_latency_pytorch_20260729"


def main():
    target_plot = OUTPUT_DIR / "latency_quantiles_cached.png"
    target_summary = OUTPUT_DIR / "cached_dct_summary.json"
    target_samples = OUTPUT_DIR / "cached_dct_latency_samples.csv"
    for target in (target_plot, target_summary, target_samples):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")

    gpu = 5
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    with (RUN_DIR / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    model = make_model(config)
    model.load_state_dict(
        clean_state_dict(torch.load(RUN_DIR / "state.pth", map_location="cpu"))
    )
    model.eval().to(device)

    with torch.inference_mode():
        cached_bank = model.transient_dct_bank_waveforms().detach()
    model.transient_dct_bank_waveforms = types.MethodType(
        lambda self: cached_bank,
        model,
    )

    controls = make_controls(config, device)
    sampling_rate = config["preprocess"]["sampling_rate"]
    block_size = config["preprocess"]["block_size"]
    frame_ms = 1000.0 * block_size / sampling_rate
    chunk_ms = frame_ms * controls["pitch"].shape[1]

    cached_results = {}
    rows = []
    for mode, audio_ms, warmup, iterations in (
        ("one_frame", frame_ms, 100, 1000),
        ("full_chunk", chunk_ms, 30, 200),
    ):
        wall, cuda = benchmark(
            model=model,
            controls=controls,
            device=device,
            mode=mode,
            warmup=warmup,
            iterations=iterations,
            stateful=True,
        )
        cached_results[mode] = {
            "wall": summarize(wall, audio_ms),
            "device_execution": summarize(cuda, audio_ms),
        }
        for measurement, values in (
            ("synchronized_wall", wall),
            ("cuda_event", cuda),
        ):
            for iteration, latency_ms in enumerate(values):
                rows.append({
                    "model": "Bass-DDSP cached DCT",
                    "device": "cuda",
                    "mode": mode,
                    "measurement": measurement,
                    "iteration": iteration,
                    "latency_ms": latency_ms,
                })

    with (OUTPUT_DIR / "summary.json").open() as handle:
        original = json.load(handle)
    plotted_results = copy.deepcopy(original["results"])
    cached_name = "Bass-DDSP cached DCT"
    plotted_results[cached_name] = plotted_results.pop("Bass-DDSP")
    plotted_results[cached_name]["cuda"]["one_frame"] = cached_results["one_frame"]
    plotted_results[cached_name]["cuda"]["full_chunk"] = cached_results["full_chunk"]
    original_models = latency_benchmark.MODELS
    latency_benchmark.MODELS = {
        cached_name: original_models["Bass-DDSP"],
        "Vanilla DWTS": original_models["Vanilla DWTS"],
        "Vanilla DDSP": original_models["Vanilla DDSP"],
    }
    write_plot(plotted_results, OUTPUT_DIR)
    latency_benchmark.MODELS = original_models
    generated_plot = OUTPUT_DIR / "latency_quantiles.png"
    generated_plot.replace(target_plot)

    with target_summary.open("w") as handle:
        json.dump(
            {
                "method": (
                    "transient_dct_bank_waveforms() computed once after checkpoint "
                    "load and reused without changing any dynamic controls"
                ),
                "source_summary": str(OUTPUT_DIR / "summary.json"),
                "bass_ddsp_cached": cached_results,
            },
            handle,
            indent=2,
        )
    with target_samples.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Restore the original plot from the unchanged original summary.
    write_plot(original["results"], OUTPUT_DIR)
    print(json.dumps(cached_results, indent=2))


if __name__ == "__main__":
    main()
