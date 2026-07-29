#!/usr/bin/env python3
"""Export the native plugin's ONNX controllers and cached synthesis tables."""

from __future__ import annotations

import functools
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from bass_ddsp.train import clean_state_dict, make_model
from onset_envelope_net.model import StructuredBassOnsetEnvelopeNet


BASS_RUN = (
    WORKSPACE
    / "runs"
    / "bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513"
)
ONSET_RUN = WORKSPACE / "runs" / "bass_onset_single_note_ablation_v2"
OUTPUT = WORKSPACE / "ScatToBassVST" / "Models"
CONTROLLER_SOURCE = (
    WORKSPACE / "ddsp_onnx_benchmark" / "models" / "bass_ddsp_controller.onnx"
)
CONTROLLER_MANIFEST = (
    WORKSPACE / "ddsp_onnx_benchmark" / "models" / "manifest.json"
)
TORCHCREPE_ROOT = Path("/usr/local/lib/python3.10/dist-packages/torchcrepe")


class TinyCrepe(torch.nn.Module):
    """TorchCREPE tiny architecture, isolated from its torchaudio import."""

    def __init__(self):
        super().__init__()
        incoming = [1, 128, 16, 16, 16, 32]
        outgoing = [128, 16, 16, 16, 32, 64]
        kernels = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]
        batch_norm = functools.partial(
            torch.nn.BatchNorm2d,
            eps=0.0010000000474974513,
            momentum=0.0,
        )
        for index in range(6):
            setattr(
                self,
                f"conv{index + 1}",
                torch.nn.Conv2d(
                    incoming[index],
                    outgoing[index],
                    kernels[index],
                    strides[index],
                ),
            )
            setattr(
                self,
                f"conv{index + 1}_BN",
                batch_norm(outgoing[index]),
            )
        self.classifier = torch.nn.Linear(256, 360)

    @staticmethod
    def layer(x, convolution, normalization, padding=(0, 0, 31, 32)):
        x = F.pad(x, padding)
        x = convolution(x)
        x = F.relu(x)
        x = normalization(x)
        return F.max_pool2d(x, (2, 1), (2, 1))

    def forward(self, frame):
        x = frame[:, None, :, None]
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        x = self.layer(x, self.conv5, self.conv5_BN)
        x = self.layer(x, self.conv6, self.conv6_BN)
        x = x.permute(0, 2, 1, 3).reshape(-1, 256)
        return torch.sigmoid(self.classifier(x))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_onnx(path, inputs, expected):
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, inputs)
    return max(
        float(np.max(np.abs(reference.detach().numpy() - value)))
        for reference, value in zip(expected, actual)
    )


def export_crepe():
    model = TinyCrepe().eval()
    weights = torch.load(
        TORCHCREPE_ROOT / "assets" / "tiny.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(weights)
    frame = torch.linspace(-1.0, 1.0, 1024).reshape(1, 1024)
    target = OUTPUT / "torchcrepe_tiny.onnx"
    with torch.inference_mode():
        expected = model(frame)
        torch.onnx.export(
            model,
            frame,
            target,
            opset_version=17,
            input_names=["audio_frame"],
            output_names=["pitch_probabilities"],
            do_constant_folding=True,
        )
    error = check_onnx(
        target,
        {"audio_frame": frame.numpy()},
        [expected],
    )
    return {"path": target.name, "max_abs_error": error, "sha256": sha256(target)}


def export_onset_model():
    checkpoint_path = ONSET_RUN / "articulation_f0_loudness.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = StructuredBassOnsetEnvelopeNet(
        num_articulations=int(checkpoint["num_articulations"]),
        control_dim=len(checkpoint["control_indices"]),
        envelope_frames=int(checkpoint["envelope_frames"]),
    ).eval()
    model.load_state_dict(checkpoint["model"])
    articulation = torch.tensor([0], dtype=torch.int64)
    controls = torch.tensor([[0.2, 0.0]], dtype=torch.float32)
    target = OUTPUT / "onset_envelope.onnx"
    with torch.inference_mode():
        expected = model(articulation, controls)
        torch.onnx.export(
            model,
            (articulation, controls),
            target,
            opset_version=17,
            input_names=["articulation", "controls"],
            output_names=["onset_envelope"],
            do_constant_folding=True,
        )
    error = check_onnx(
        target,
        {"articulation": articulation.numpy(), "controls": controls.numpy()},
        [expected],
    )
    labels = torch.load(
        ONSET_RUN / "single_note_tensors.pt",
        map_location="cpu",
        weights_only=False,
    )["articulation_labels"]
    return {
        "path": target.name,
        "max_abs_error": error,
        "sha256": sha256(target),
        "labels": labels,
        "control_names": checkpoint["control_names"],
        "frames": int(checkpoint["envelope_frames"]),
    }


def export_synthesis_tables():
    import yaml

    with (BASS_RUN / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    model = make_model(config).eval()
    checkpoint_path = BASS_RUN / "state.pth"
    model.load_state_dict(
        clean_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        )
    )
    with torch.inference_mode():
        wavetables = torch.tanh(model.sustain_wavetables).numpy().astype("<f4")
        transient = (
            model.transient_dct_bank_waveforms().numpy().astype("<f4")
        )
    wavetable_path = OUTPUT / "sustain_wavetables.f32"
    transient_path = OUTPUT / "transient_bank.f32"
    wavetables.tofile(wavetable_path)
    transient.tofile(transient_path)
    return {
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "articulation_labels": config["data"]["articulation_labels"],
        "wavetables": {
            "path": wavetable_path.name,
            "shape": list(wavetables.shape),
            "sha256": sha256(wavetable_path),
        },
        "transient_bank": {
            "path": transient_path.name,
            "shape": list(transient.shape),
            "sha256": sha256(transient_path),
        },
        "sample_rate": int(config["model"]["sampling_rate"]),
        "hop_samples": int(config["model"]["block_size"]),
        "noise_bands": int(config["model"]["n_bands"]),
        "sustain_gain_db": float(config["model"]["sustain_gain_db"]),
        "loudness_gain_db_per_std": float(
            config["model"]["loudness_gain_db_per_std"]
        ),
        "harmonic_indicator_a": float(config["model"]["harmonic_indicator_a"]),
        "harmonic_indicator_b": float(config["model"]["harmonic_indicator_b"]),
        "harmonic_gate_floor": float(config["model"]["harmonic_gate_floor"]),
        "sustain_fade_seconds": float(config["model"]["sustain_fade_seconds"]),
        "transient_seconds": float(config["model"]["transient_seconds"]),
        "transient_window_fade_seconds": float(
            config["model"]["transient_window_fade_seconds"]
        ),
        "transient_velocity_floor": float(
            config["model"]["transient_velocity_floor"]
        ),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    controller_metadata = json.loads(CONTROLLER_MANIFEST.read_text())["bass_ddsp"]
    if Path(controller_metadata["source_run"]).resolve() != BASS_RUN.resolve():
        raise RuntimeError(
            "Bass controller does not come from the pinned 2026-07-24 run: "
            f"{controller_metadata['source_run']}"
        )
    controller_target = OUTPUT / "bass_ddsp_controller.onnx"
    shutil.copy2(CONTROLLER_SOURCE, controller_target)
    manifest = {
        "bass_controller": {
            "path": controller_target.name,
            "sha256": sha256(controller_target),
            "source_run": str(BASS_RUN),
            "max_abs_error": controller_metadata["max_abs_error"],
        },
        "torchcrepe": export_crepe(),
        "onset_envelope": export_onset_model(),
        "synthesis": export_synthesis_tables(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
