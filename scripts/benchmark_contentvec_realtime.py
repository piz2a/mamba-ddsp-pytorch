#!/usr/bin/env python3
"""Benchmark ContentVec throughput and independent-chunk representation drift."""

import argparse
import json
import statistics
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils


SAMPLE_RATE = 16000
FEATURE_STRIDE_SAMPLES = 320


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_features(model, waveform):
    output = model.extract_features(
        source=waveform,
        padding_mask=None,
        mask=False,
    )
    return output[0] if isinstance(output, tuple) else output["x"]


def audio_segment(audio, samples):
    if len(audio) < samples:
        audio = np.tile(audio, int(np.ceil(samples / max(len(audio), 1))))
    return np.asarray(audio[:samples], dtype=np.float32)


def benchmark_duration(model, audio, duration, device, repeats, warmups):
    samples = int(round(duration * SAMPLE_RATE))
    waveform = torch.from_numpy(audio_segment(audio, samples))[None].to(device)
    with torch.inference_mode():
        for _ in range(warmups):
            features = model_features(model, waveform)
        synchronize(device)
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            features = model_features(model, waveform)
            synchronize(device)
            elapsed.append(time.perf_counter() - start)
    return {
        "audio_seconds": duration,
        "output_frames": int(features.shape[1]),
        "median_ms": statistics.median(elapsed) * 1000.0,
        "p95_ms": float(np.percentile(elapsed, 95)) * 1000.0,
        "median_rtf": statistics.median(elapsed) / duration,
    }


def chunk_consistency(model, audio, total_seconds, chunk_seconds, device):
    total_samples = int(total_seconds * SAMPLE_RATE)
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    chunk_samples -= chunk_samples % FEATURE_STRIDE_SAMPLES
    waveform = torch.from_numpy(audio_segment(audio, total_samples))[None].to(device)
    similarities = []
    with torch.inference_mode():
        reference = model_features(model, waveform)[0]
        for start in range(0, total_samples - chunk_samples + 1, chunk_samples):
            chunk = waveform[:, start:start + chunk_samples]
            chunk_features = model_features(model, chunk)[0]
            first_frame = start // FEATURE_STRIDE_SAMPLES
            stop_frame = min(first_frame + len(chunk_features), len(reference))
            usable = stop_frame - first_frame
            if usable <= 0:
                continue
            cosine = F.cosine_similarity(
                chunk_features[:usable],
                reference[first_frame:stop_frame],
                dim=-1,
            )
            similarities.append(cosine.cpu())
    if not similarities:
        return {"chunk_seconds": chunk_seconds, "compared_frames": 0}
    similarities = torch.cat(similarities).numpy()
    return {
        "chunk_seconds": chunk_seconds,
        "compared_frames": int(len(similarities)),
        "mean_cosine_to_full_context": float(np.mean(similarities)),
        "p10_cosine_to_full_context": float(np.percentile(similarities, 10)),
        "minimum_cosine_to_full_context": float(np.min(similarities)),
    }


def architecture_summary(model):
    receptive_field = 1
    stride = 1
    convolution_layers = []
    for layer in model.feature_extractor.conv_layers:
        convolution = layer[0]
        kernel = int(convolution.kernel_size[0])
        layer_stride = int(convolution.stride[0])
        receptive_field += (kernel - 1) * stride
        stride *= layer_stride
        convolution_layers.append({"kernel": kernel, "stride": layer_stride})
    positional_convolution = model.encoder.pos_conv[0]
    return {
        "encoder_layers": len(model.encoder.layers),
        "embedding_dimension": int(model.encoder.embedding_dim),
        "attention_is_bidirectional": True,
        "feature_convolutions": convolution_layers,
        "feature_stride_samples": stride,
        "feature_stride_ms": stride / SAMPLE_RATE * 1000.0,
        "feature_receptive_field_samples": receptive_field,
        "feature_receptive_field_ms": receptive_field / SAMPLE_RATE * 1000.0,
        "positional_convolution_frames": int(positional_convolution.kernel_size[0]),
        "positional_convolution_span_ms": (
            int(positional_convolution.kernel_size[0]) * stride / SAMPLE_RATE * 1000.0
        ),
        "causal_without_model_changes": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/workspace/contentvec/checkpoints/checkpoint_best_legacy_100.pt",
    )
    parser.add_argument(
        "--audio",
        default="/workspace/learn/voice_inputs/Scat 1.wav",
    )
    parser.add_argument(
        "--output",
        default="/workspace/contentvec_latency_benchmark.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    load_start = time.perf_counter()
    models, config, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(Path(args.checkpoint))]
    )
    model = models[0].eval().to(requested_device)
    load_seconds = time.perf_counter() - load_start
    audio, _ = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True)

    if requested_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(requested_device)

    durations = [0.032, 0.064, 0.080, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00]
    throughput = [
        benchmark_duration(
            model, audio, duration, requested_device, args.repeats, args.warmups
        )
        for duration in durations
    ]
    consistency = [
        chunk_consistency(model, audio, 4.0, duration, requested_device)
        for duration in [0.25, 0.50, 1.00]
    ]
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "audio": str(Path(args.audio)),
        "device": str(requested_device),
        "gpu_name": (
            torch.cuda.get_device_name(requested_device)
            if requested_device.type == "cuda"
            else None
        ),
        "load_seconds": load_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
        "architecture": architecture_summary(model),
        "throughput": throughput,
        "chunk_consistency": consistency,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(requested_device)
            if requested_device.type == "cuda"
            else None
        ),
        "config_encoder_layers": int(config.model.encoder_layers),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
