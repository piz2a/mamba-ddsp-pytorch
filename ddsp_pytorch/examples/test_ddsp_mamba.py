import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKSPACE / "mamba"))

from ddsp.core import extract_loudness, extract_pitch
from ddsp.model import DDSP
from mamba_ssm import Mamba


def make_sine(freq=220.0, sampling_rate=16000, seconds=0.5):
    t = np.arange(int(sampling_rate * seconds), dtype=np.float32) / sampling_rate
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_preprocessing():
    sampling_rate = 16000
    block_size = 160
    audio = make_sine(sampling_rate=sampling_rate)

    pitch = extract_pitch(audio, sampling_rate, block_size)
    loudness = extract_loudness(audio, sampling_rate, block_size)

    print("preprocess")
    print("  pitch:", pitch.shape, "mean_f0:", float(np.nanmean(pitch)))
    print("  loudness:", loudness.shape, "finite:", bool(np.isfinite(loudness).all()))

    return pitch, loudness


def test_standalone_mamba(device):
    model = Mamba(
        d_model=16,
        d_state=4,
        d_conv=2,
        expand=1,
        use_fast_path=False,
    ).to(device)
    x = torch.randn(2, 12, 16, device=device)
    y = model(x)
    y.mean().backward()

    print("standalone mamba")
    print("  output:", tuple(y.shape), "finite:", bool(torch.isfinite(y).all()))


def test_ddsp(recurrent_type, device):
    model = DDSP(
        hidden_size=32,
        n_harmonic=16,
        n_bands=16,
        sampling_rate=16000,
        block_size=80,
        recurrent_type=recurrent_type,
        mamba_d_state=4,
        mamba_d_conv=2,
        mamba_expand=1,
    ).to(device)

    frames = 20
    pitch = torch.full((2, frames, 1), 220.0, device=device)
    loudness = torch.zeros(2, frames, 1, device=device)

    audio = model(pitch, loudness)
    loss = audio.abs().mean()
    loss.backward()

    realtime_audio = model.realtime_forward(pitch[:, :3], loudness[:, :3])

    print(f"ddsp {recurrent_type}")
    print("  train output:", tuple(audio.shape), "finite:", bool(torch.isfinite(audio).all()))
    print("  realtime output:", tuple(realtime_audio.shape), "finite:", bool(torch.isfinite(realtime_audio).all()))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("torch:", torch.__version__)

    test_preprocessing()
    test_standalone_mamba(device)
    test_ddsp("gru", device)
    test_ddsp("mamba", device)

    print("ok")


if __name__ == "__main__":
    main()
