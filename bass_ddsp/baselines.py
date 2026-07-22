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


class VanillaDWTS(nn.Module):
    """Vanilla-control DWTS baseline: f0/loudness -> wavetable sustain + noise."""

    def __init__(
        self,
        hidden_size,
        n_bands,
        sampling_rate,
        block_size,
        recurrent_type="gru",
        scale_f0_input=True,
        f0_min_hz=30.0,
        f0_max_hz=330.0,
        pitch_hidden_size=64,
        loudness_hidden_size=64,
        n_wavetables=16,
        wavetable_length=512,
        wavetable_init_std=0.01,
        sustain_gain_db=0.0,
        noise_gain_db=0.0,
        transient_gain_db=0.0,
        learnable_branch_gains=True,
        **kwargs,
    ):
        super().__init__()
        if recurrent_type != "gru":
            raise ValueError("VanillaDWTS currently supports recurrent_type='gru' only")
        self.n_bands = int(n_bands)
        self.n_wavetables = int(n_wavetables)
        self.wavetable_length = int(wavetable_length)
        if self.n_wavetables <= 0:
            raise ValueError("n_wavetables must be positive")
        if self.wavetable_length <= 1:
            raise ValueError("wavetable_length must be greater than 1")

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
        self.sustain_attention_proj = nn.Linear(hidden_size, self.n_wavetables)
        self.sustain_amp_proj = nn.Linear(hidden_size, 1)
        self.noise_proj = nn.Linear(hidden_size, self.n_bands)

        wavetable = torch.randn(
            self.n_wavetables,
            self.wavetable_length,
        ) * float(wavetable_init_std)
        phase = torch.linspace(
            0.0,
            2.0 * math.pi,
            self.wavetable_length + 1,
        )[:-1]
        wavetable[0] = torch.sin(phase)
        if self.n_wavetables > 1:
            wavetable[1] = torch.sin(2.0 * phase) * 0.75
        if self.n_wavetables > 2:
            wavetable[2] = torch.sin(3.0 * phase) * 0.5
        if self.n_wavetables > 3:
            wavetable[3] = torch.sin(4.0 * phase) * 0.35
        self.sustain_wavetables = nn.Parameter(wavetable)

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
        self.last_sustain_attention = None

    def _pitch_for_network(self, pitch):
        if not self.scale_f0_input:
            return pitch
        pitch = pitch.clamp_min(1e-6)
        midi = 69.0 + 12.0 * torch.log2(pitch / 440.0)
        denom = (self.f0_max_midi - self.f0_min_midi).clamp_min(1e-6)
        return ((midi - self.f0_min_midi) / denom).clamp(0.0, 1.0)

    def _upsample_linear(self, control, length=None):
        control = control.permute(0, 2, 1)
        size = control.shape[-1] * int(self.block_size.detach().cpu().item())
        control = nn.functional.interpolate(
            control,
            size=size,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        return control if length is None else control[:, :length]

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

    def _wavetable_lookup(self, pitch):
        wavetable = torch.tanh(self.sustain_wavetables)
        length = pitch.shape[1]
        phase_increment = pitch.squeeze(-1) * self.wavetable_length / self.sampling_rate
        phase = torch.cumsum(phase_increment, dim=1)
        phase = torch.remainder(phase, self.wavetable_length)
        index_0 = torch.floor(phase).long()
        index_1 = torch.remainder(index_0 + 1, self.wavetable_length)
        alpha = (phase - index_0.to(phase)).unsqueeze(-1)

        flat_0 = index_0.reshape(-1)
        flat_1 = index_1.reshape(-1)
        value_0 = wavetable[:, flat_0].transpose(0, 1).reshape(
            pitch.shape[0],
            length,
            self.n_wavetables,
        )
        value_1 = wavetable[:, flat_1].transpose(0, 1).reshape(
            pitch.shape[0],
            length,
            self.n_wavetables,
        )
        return value_0 + alpha * (value_1 - value_0)

    def _wavetable_branch(self, hidden, pitch):
        length = hidden.shape[1] * int(self.block_size.detach().cpu().item())
        pitch_audio = self._upsample_linear(pitch, length)
        waves = self._wavetable_lookup(pitch_audio)
        weights = torch.softmax(self.sustain_attention_proj(hidden), dim=-1)
        weights_audio = self._upsample_linear(weights, length)
        amplitude = scale_function(self.sustain_amp_proj(hidden))
        amplitude = self._upsample_linear(amplitude, length)
        self.last_sustain_attention = weights.detach()
        return torch.sum(waves * weights_audio, dim=-1, keepdim=True) * amplitude

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
        sustain_raw = self._wavetable_branch(hidden, pitch)
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
