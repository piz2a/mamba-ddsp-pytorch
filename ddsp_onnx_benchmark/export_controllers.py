#!/usr/bin/env python3
"""Export stateful one-frame neural controllers for native DSP integration."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bass_ddsp.train import clean_state_dict, make_model
from ddsp.core import scale_function


RUNS = {
    "bass_ddsp": ROOT / "runs/bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513",
    "vanilla_dwts": ROOT / "runs/vanilla_dwts_riff_resume_to_200k_20260726_071231",
    "vanilla_ddsp": ROOT / "runs/vanilla_ddsp_riff_20260720_090532",
}


class BaselineController(nn.Module):
    def __init__(self, model, model_type):
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, pitch, loudness, hidden_state):
        pitch_control = self.model._pitch_for_network(pitch)
        condition = torch.cat([
            self.model.pitch_mlp(pitch_control),
            self.model.loudness_mlp(loudness),
        ], dim=-1)
        recurrent, next_hidden = self.model.gru(condition, hidden_state)
        hidden = self.model.out_mlp(
            torch.cat([recurrent, pitch_control, loudness], dim=-1)
        )
        noise_bands = scale_function(self.model.noise_proj(hidden) - 5.0)
        if self.model_type == "vanilla_ddsp":
            harmonic = scale_function(self.model.harmonic_proj(hidden))
            return harmonic, noise_bands, next_hidden
        wavetable_weights = torch.softmax(
            self.model.sustain_attention_proj(hidden), dim=-1
        )
        sustain_amplitude = scale_function(self.model.sustain_amp_proj(hidden))
        return wavetable_weights, sustain_amplitude, noise_bands, next_hidden


class BassController(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        pitch,
        loudness,
        articulation,
        onset_strength,
        offset,
        gate,
        note_age,
        periodicity,
        hidden_state,
    ):
        z = self.model._encode_articulation(
            pitch,
            articulation,
            onset_strength,
            offset,
            gate,
            note_age,
            periodicity,
        )
        pitch_control = self.model._pitch_for_network(pitch)
        condition = torch.cat([
            self.model.in_mlps[0](pitch_control),
            self.model.in_mlps[1](loudness),
            self.model.in_mlps[2](z),
        ], dim=-1)
        recurrent, next_hidden = self.model.gru(condition, hidden_state)
        hidden = self.model.out_mlp(torch.cat([
            recurrent,
            pitch_control,
            loudness,
            z,
        ], dim=-1))
        wavetable_weights = torch.softmax(
            self.model.sustain_attention_proj(hidden), dim=-1
        )
        sustain_amplitude = scale_function(self.model.sustain_amp_proj(hidden))
        noise_bands = scale_function(self.model.noise_proj(hidden) - 5.0)
        transient_gain = (
            scale_function(self.model.transient_gain(hidden))
            + self.model.transient_dct_gain_floor
        )
        return (
            wavetable_weights,
            sustain_amplitude,
            noise_bands,
            transient_gain,
            next_hidden,
        )


def inputs_for(name, hidden_size):
    pitch = torch.tensor([[[55.0]]], dtype=torch.float32)
    loudness = torch.tensor([[[0.0]]], dtype=torch.float32)
    hidden = torch.zeros(1, 1, hidden_size, dtype=torch.float32)
    if name != "bass_ddsp":
        return (pitch, loudness, hidden), ["pitch", "loudness", "hidden_state"]
    scalar = torch.zeros(1, 1, 1, dtype=torch.float32)
    return (
        pitch,
        loudness,
        torch.zeros(1, 1, dtype=torch.int64),
        torch.ones_like(scalar),
        scalar,
        torch.ones_like(scalar),
        scalar,
        torch.full_like(scalar, 0.85),
        hidden,
    ), [
        "pitch",
        "loudness",
        "articulation",
        "onset_strength",
        "offset",
        "gate",
        "note_age",
        "periodicity",
        "hidden_state",
    ]


def export_one(name, run_dir, output):
    with (run_dir / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    model = make_model(config).eval()
    model.load_state_dict(
        clean_state_dict(torch.load(run_dir / "state.pth", map_location="cpu"))
    )
    hidden_size = int(config["model"]["hidden_size"])
    if name == "bass_ddsp":
        wrapper = BassController(model).eval()
        output_names = [
            "wavetable_weights",
            "sustain_amplitude",
            "noise_bands",
            "transient_gain",
            "next_hidden_state",
        ]
    else:
        wrapper = BaselineController(model, name).eval()
        output_names = (
            ["harmonic_parameters", "noise_bands", "next_hidden_state"]
            if name == "vanilla_ddsp"
            else [
                "wavetable_weights",
                "sustain_amplitude",
                "noise_bands",
                "next_hidden_state",
            ]
        )
    args, input_names = inputs_for(name, hidden_size)
    target = output / f"{name}_controller.onnx"
    with torch.inference_mode():
        expected = wrapper(*args)
        torch.onnx.export(
            wrapper,
            args,
            target,
            opset_version=17,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(target))
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    feed = {
        input_name: tensor.detach().cpu().numpy()
        for input_name, tensor in zip(input_names, args)
    }
    actual = session.run(None, feed)
    errors = [
        float(np.max(np.abs(reference.detach().numpy() - value)))
        for reference, value in zip(expected, actual)
    ]
    return {
        "path": str(target),
        "source_run": str(run_dir),
        "inputs": input_names,
        "outputs": output_names,
        "parameter_count": sum(parameter.numel() for parameter in wrapper.parameters()),
        "max_abs_error_by_output": errors,
        "max_abs_error": max(errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "ddsp_onnx_benchmark/models"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        name: export_one(name, run_dir, output)
        for name, run_dir in RUNS.items()
    }
    with (output / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
