import math

import torch
import torch.nn as nn

from ddsp.core import (
    amp_to_impulse_response,
    fft_convolve,
    harmonic_synth,
    mlp,
    remove_above_nyquist,
    scale_function,
    upsample,
)


class VanillaDDSP(nn.Module):
    """Original-style DDSP baseline: f0/loudness -> harmonic + noise."""

    def __init__(
        self,
        hidden_size,
        n_harmonic,
        n_bands,
        sampling_rate,
        block_size,
        recurrent_type="gru",
        scale_f0_input=True,
        f0_min_hz=30.0,
        f0_max_hz=330.0,
        pitch_hidden_size=64,
        loudness_hidden_size=64,
        sustain_gain_db=0.0,
        noise_gain_db=0.0,
        transient_gain_db=0.0,
        learnable_branch_gains=True,
        **kwargs,
    ):
        super().__init__()
        if recurrent_type != "gru":
            raise ValueError("VanillaDDSP currently supports recurrent_type='gru' only")
        self.n_harmonic = int(n_harmonic)
        self.n_bands = int(n_bands)
        self.register_buffer("sampling_rate", torch.tensor(int(sampling_rate)))
        self.register_buffer("block_size", torch.tensor(int(block_size)))
        self.scale_f0_input = bool(scale_f0_input)

        f0_min_hz = max(float(f0_min_hz), 1e-6)
        f0_max_hz = max(float(f0_max_hz), f0_min_hz + 1e-6)
        f0_min_midi = 69.0 + 12.0 * math.log2(f0_min_hz / 440.0)
        f0_max_midi = 69.0 + 12.0 * math.log2(f0_max_hz / 440.0)
        self.register_buffer("f0_min_midi", torch.tensor(f0_min_midi))
        self.register_buffer("f0_max_midi", torch.tensor(f0_max_midi))

        self.pitch_mlp = mlp(1, int(pitch_hidden_size), 3)
        self.loudness_mlp = mlp(1, int(loudness_hidden_size), 3)
        condition_size = int(pitch_hidden_size) + int(loudness_hidden_size)
        self.gru = nn.GRU(condition_size, hidden_size, batch_first=True)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)
        self.harmonic_proj = nn.Linear(hidden_size, self.n_harmonic + 1)
        self.noise_proj = nn.Linear(hidden_size, self.n_bands)

        gain_db = torch.tensor([
            float(sustain_gain_db),
            float(noise_gain_db),
            float(transient_gain_db),
        ])
        log_gain = gain_db * (math.log(10.0) / 20.0)
        if learnable_branch_gains:
            self.branch_log_gains = nn.Parameter(log_gain)
        else:
            self.register_buffer("branch_log_gains", log_gain)

        self.register_buffer("phase", torch.zeros(1))
        self.last_branch_outputs = {}

    def _pitch_for_network(self, pitch):
        if not self.scale_f0_input:
            return pitch
        pitch = pitch.clamp_min(1e-6)
        midi = 69.0 + 12.0 * torch.log2(pitch / 440.0)
        denom = (self.f0_max_midi - self.f0_min_midi).clamp_min(1e-6)
        return ((midi - self.f0_min_midi) / denom).clamp(0.0, 1.0)

    def _branch_gain(self, index, reference):
        return torch.exp(self.branch_log_gains[index]).to(
            device=reference.device,
            dtype=reference.dtype,
        )

    def branch_gain_db(self):
        gains = self.branch_log_gains.detach().cpu() * (20.0 / math.log(10.0))
        return {
            "sustain_gain_db": float(gains[0]),
            "noise_gain_db": float(gains[1]),
            "transient_gain_db": float(gains[2]),
        }

    def _hidden(self, pitch, loudness):
        pitch_control = self._pitch_for_network(pitch)
        condition = torch.cat([
            self.pitch_mlp(pitch_control),
            self.loudness_mlp(loudness),
        ], dim=-1)
        recurrent = self.gru(condition)[0]
        return self.out_mlp(torch.cat([recurrent, pitch_control, loudness], dim=-1))

    def _harmonic_branch(self, hidden, pitch):
        param = scale_function(self.harmonic_proj(hidden))
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        amplitudes = remove_above_nyquist(amplitudes, pitch, self.sampling_rate)
        amplitudes = amplitudes / amplitudes.sum(-1, keepdim=True).clamp_min(1e-7)
        amplitudes = amplitudes * total_amp
        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)
        return harmonic_synth(pitch, amplitudes, self.sampling_rate)

    def _noise_branch(self, hidden):
        param = scale_function(self.noise_proj(hidden) - 5)
        impulse = amp_to_impulse_response(param, self.block_size)
        block_size = int(self.block_size.detach().cpu().item())
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            block_size,
            device=impulse.device,
            dtype=impulse.dtype,
        ) * 2 - 1
        noise = fft_convolve(noise, impulse).contiguous()
        return noise.reshape(noise.shape[0], -1, 1)

    def forward(
        self,
        pitch,
        loudness,
        articulation=None,
        onset_strength=None,
        offset=None,
        gate=None,
        note_age=None,
        periodicity=None,
    ):
        hidden = self._hidden(pitch, loudness)
        sustain_raw = self._harmonic_branch(hidden, pitch)
        noise_raw = self._noise_branch(hidden)
        length = min(sustain_raw.shape[1], noise_raw.shape[1])
        sustain_raw = sustain_raw[:, :length]
        noise_raw = noise_raw[:, :length]
        if gate is not None:
            gate_audio = upsample(gate, self.block_size)[:, :length].clamp(0.0, 1.0)
        else:
            gate_audio = 1.0
        sustain = sustain_raw * gate_audio * self._branch_gain(0, sustain_raw)
        noise = noise_raw * gate_audio * self._branch_gain(1, noise_raw)
        transient = torch.zeros_like(sustain)
        signal = sustain + noise
        self.last_branch_outputs = {
            "sustain": sustain,
            "noise": noise,
            "transient": transient,
            "signal": signal,
            "sustain_raw": sustain_raw,
            "noise_raw": noise_raw,
            "transient_raw": transient,
        }
        return signal
