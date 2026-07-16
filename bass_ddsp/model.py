import math
import sys
from pathlib import Path

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


def _import_mamba():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        return Mamba
    except ImportError:
        workspace_mamba = Path(__file__).resolve().parents[1] / "mamba"
        if workspace_mamba.exists() and str(workspace_mamba) not in sys.path:
            sys.path.insert(0, str(workspace_mamba))
        from mamba_ssm.modules.mamba_simple import Mamba
        return Mamba


class MambaRecurrent(nn.Module):
    def __init__(self, input_size, hidden_size, d_state=16, d_conv=4, expand=2):
        super().__init__()
        Mamba = _import_mamba()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.register_buffer("conv_state", torch.empty(0), persistent=False)
        self.register_buffer("ssm_state", torch.empty(0), persistent=False)

    def forward(self, hidden):
        return self.norm(self.mamba(self.input_proj(hidden)))

    def reset_state(self):
        self.conv_state = torch.empty(0, device=self.input_proj.weight.device)
        self.ssm_state = torch.empty(0, device=self.input_proj.weight.device)

    def _ensure_state(self, hidden):
        batch = hidden.shape[0]
        expected_conv = (
            batch,
            self.mamba.d_model * self.mamba.expand,
            self.mamba.d_conv,
        )
        expected_ssm = (
            batch,
            self.mamba.d_model * self.mamba.expand,
            self.mamba.d_state,
        )
        if (
            tuple(self.conv_state.shape) != expected_conv
            or tuple(self.ssm_state.shape) != expected_ssm
            or self.conv_state.device != hidden.device
            or self.ssm_state.device != hidden.device
            or self.conv_state.dtype != hidden.dtype
            or self.ssm_state.dtype != hidden.dtype
        ):
            self.conv_state, self.ssm_state = self.mamba.allocate_inference_cache(
                batch,
                max_seqlen=1,
                dtype=hidden.dtype,
            )

    def step(self, hidden):
        hidden = self.input_proj(hidden)
        self._ensure_state(hidden)
        hidden, conv_state, ssm_state = self.mamba.step(
            hidden,
            self.conv_state,
            self.ssm_state,
        )
        self.conv_state = conv_state
        self.ssm_state = ssm_state
        return self.norm(hidden)


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0.0, initial_decay=5.0):
        super().__init__()
        self.length = int(length)
        self.sampling_rate = int(sampling_rate)
        self.noise = nn.Parameter((torch.rand(self.length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))
        t = torch.arange(self.length) / self.sampling_rate
        self.register_buffer("t", t.reshape(1, -1, 1))

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        impulse = self.noise * t * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        len_x = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, len_x - self.length))
        return fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)


class ArticulationEncoder(nn.Module):
    def __init__(self, n_articulation, embedding_size, z_size, hidden_size,
                 n_controls):
        super().__init__()
        self.articulation = nn.Embedding(n_articulation, embedding_size)
        self.n_controls = int(n_controls)
        self.net = nn.Sequential(
            mlp(embedding_size + self.n_controls, hidden_size, 2),
            nn.Linear(hidden_size, z_size),
            nn.LayerNorm(z_size),
            nn.LeakyReLU(),
        )

    def forward(self, articulation, controls):
        if articulation.ndim == 3:
            articulation = articulation.squeeze(-1)
        features = [self.articulation(articulation.long())]
        if self.n_controls:
            features.append(controls.to(self.articulation.weight.dtype))
        return self.net(torch.cat(features, dim=-1))


class BassDDSPV2(nn.Module):
    """Bass-specific DDSP model without a legacy architecture switch."""

    def __init__(
        self,
        hidden_size,
        n_harmonic,
        n_bands,
        sampling_rate,
        block_size,
        n_articulation,
        z_size=64,
        articulation_embedding_size=24,
        articulation_hidden_size=None,
        use_note_shape_controls=True,
        recurrent_type="mamba",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        use_transient_branch=True,
        use_sustain_branch=True,
        use_noise_branch=True,
        use_reverb=False,
        transient_seconds=0.20,
        scale_f0_input=True,
        f0_min_hz=30.0,
        f0_max_hz=330.0,
        pitch_hidden_size=64,
        loudness_hidden_size=64,
        z_hidden_size=256,
        sustain_gain_db=0.0,
        noise_gain_db=0.0,
        transient_gain_db=0.0,
        learnable_branch_gains=False,
    ):
        super().__init__()
        if n_articulation <= 0:
            raise ValueError("BassDDSPV2 requires n_articulation > 0")

        self.register_buffer("sampling_rate", torch.tensor(int(sampling_rate)))
        self.register_buffer("block_size", torch.tensor(int(block_size)))
        self.recurrent_type = str(recurrent_type)
        self.use_note_shape_controls = bool(use_note_shape_controls)
        self.use_transient_branch = bool(use_transient_branch)
        self.use_sustain_branch = bool(use_sustain_branch)
        self.use_noise_branch = bool(use_noise_branch)
        self.use_reverb = bool(use_reverb)
        self.transient_seconds = float(transient_seconds)
        self.scale_f0_input = bool(scale_f0_input)
        self.z_size = int(z_size)

        f0_min_hz = max(float(f0_min_hz), 1e-6)
        f0_max_hz = max(float(f0_max_hz), f0_min_hz + 1e-6)
        f0_min_midi = 69.0 + 12.0 * math.log2(f0_min_hz / 440.0)
        f0_max_midi = 69.0 + 12.0 * math.log2(f0_max_hz / 440.0)
        self.register_buffer("f0_min_midi", torch.tensor(f0_min_midi))
        self.register_buffer("f0_max_midi", torch.tensor(f0_max_midi))

        self.n_articulation_controls = 4 if self.use_note_shape_controls else 2
        self.articulation_encoder = ArticulationEncoder(
            n_articulation=n_articulation,
            embedding_size=articulation_embedding_size,
            z_size=z_size,
            hidden_size=articulation_hidden_size or hidden_size,
            n_controls=self.n_articulation_controls,
        )

        branch_input_sizes = [1, 1, z_size]
        branch_hidden_sizes = [
            int(pitch_hidden_size),
            int(loudness_hidden_size),
            int(z_hidden_size),
        ]
        self.in_mlps = nn.ModuleList([
            mlp(input_size, branch_hidden_size, 3)
            for input_size, branch_hidden_size in zip(
                branch_input_sizes,
                branch_hidden_sizes,
            )
        ])
        condition_size = sum(branch_hidden_sizes)

        if self.recurrent_type == "gru":
            self.gru = nn.GRU(condition_size, hidden_size, batch_first=True)
            self.recurrent = self.gru
            self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        elif self.recurrent_type == "mamba":
            self.recurrent = MambaRecurrent(
                condition_size,
                hidden_size,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
        else:
            raise ValueError(
                "recurrent_type must be 'gru' or 'mamba', "
                f"got {recurrent_type!r}"
            )

        self.out_mlp = mlp(hidden_size + 2 + z_size, hidden_size, 3)
        self.harmonic_proj = nn.Linear(hidden_size, n_harmonic + 1)
        self.noise_proj = nn.Linear(hidden_size, n_bands)

        transient_samples = max(
            1,
            int(round(self.transient_seconds * int(sampling_rate))),
        )
        self.transient_gain = nn.Linear(hidden_size, 1)
        self.transient_bank = nn.Parameter(
            torch.randn(n_articulation, transient_samples) * 0.02
        )

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

        self.reverb = Reverb(sampling_rate, sampling_rate)
        self.register_buffer("phase", torch.zeros(1))
        self.last_branch_outputs = {}

    def _pitch_for_network(self, pitch):
        if not self.scale_f0_input:
            return pitch
        pitch = pitch.clamp_min(1e-6)
        midi = 69.0 + 12.0 * torch.log2(pitch / 440.0)
        denom = (self.f0_max_midi - self.f0_min_midi).clamp_min(1e-6)
        return ((midi - self.f0_min_midi) / denom).clamp(0.0, 1.0)

    def _frame_control(self, control, pitch, fill_value=0.0):
        shape = (*pitch.shape[:2], 1)
        if control is None:
            return torch.full(
                shape,
                float(fill_value),
                dtype=pitch.dtype,
                device=pitch.device,
            )
        if control.ndim == 2:
            control = control.unsqueeze(-1)
        return control.to(pitch.device, dtype=pitch.dtype)

    def _run_recurrent(self, hidden):
        if self.recurrent_type == "gru":
            return self.gru(hidden)[0]
        return self.recurrent(hidden)

    def _run_recurrent_realtime(self, hidden):
        if self.recurrent_type == "gru":
            expected_cache = (1, hidden.shape[0], self.gru.hidden_size)
            if (
                tuple(self.cache_gru.shape) != expected_cache
                or self.cache_gru.device != hidden.device
                or self.cache_gru.dtype != hidden.dtype
            ):
                self.cache_gru = torch.zeros(
                    expected_cache,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            recurrent_out, cache = self.gru(hidden, self.cache_gru)
            self.cache_gru.copy_(cache)
            return recurrent_out

        return torch.cat([
            self.recurrent.step(hidden[:, i:i + 1])
            for i in range(hidden.shape[1])
        ], dim=1)

    def reset_recurrent_state(self):
        if self.recurrent_type == "gru":
            self.cache_gru.zero_()
        else:
            self.recurrent.reset_state()

    def _encode_articulation(
        self,
        pitch,
        articulation,
        onset,
        offset,
        gate,
        note_age,
    ):
        shape = pitch.shape[:2]
        if articulation is None:
            articulation = torch.zeros(shape, dtype=torch.long, device=pitch.device)
        if articulation.ndim == 3:
            articulation = articulation.squeeze(-1)

        controls = [
            self._frame_control(onset, pitch, 0.0),
            self._frame_control(offset, pitch, 0.0),
        ]
        if self.use_note_shape_controls:
            controls.extend([
                self._frame_control(gate, pitch, 1.0),
                self._frame_control(note_age, pitch, 0.0),
            ])
        return self.articulation_encoder(
            articulation.to(pitch.device),
            torch.cat(controls, dim=-1),
        )

    def _decoder_hidden(
        self,
        pitch,
        loudness,
        articulation,
        onset,
        offset,
        gate,
        note_age,
        realtime,
    ):
        z = self._encode_articulation(
            pitch,
            articulation,
            onset,
            offset,
            gate,
            note_age,
        )
        pitch_control = self._pitch_for_network(pitch)
        hidden = torch.cat([
            self.in_mlps[0](pitch_control),
            self.in_mlps[1](loudness),
            self.in_mlps[2](z),
        ], dim=-1)
        recurrent = (
            self._run_recurrent_realtime(hidden)
            if realtime
            else self._run_recurrent(hidden)
        )
        return self.out_mlp(torch.cat([
            recurrent,
            pitch_control,
            loudness,
            z,
        ], dim=-1))

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

    def _harmonic_branch(self, hidden, pitch, gate, realtime=False):
        param = scale_function(self.harmonic_proj(hidden))
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        amplitudes = remove_above_nyquist(amplitudes, pitch, self.sampling_rate)
        amplitudes = amplitudes / amplitudes.sum(-1, keepdim=True).clamp_min(1e-7)
        amplitudes = amplitudes * total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)
        gate = upsample(gate, self.block_size)

        if realtime:
            n_harmonic = amplitudes.shape[-1]
            omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)
            omega = omega + self.phase
            self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))
            omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
            harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
        else:
            harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)
        return harmonic * gate

    def _noise_branch(self, hidden, gate):
        param = scale_function(self.noise_proj(hidden) - 5)
        impulse = amp_to_impulse_response(param, self.block_size)
        block_size = int(self.block_size.detach().cpu().item())
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            block_size,
        ).to(impulse) * 2 - 1
        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)
        return noise * upsample(gate, self.block_size)

    def _transient_branch(self, hidden, articulation, gate, note_age):
        block_size = int(self.block_size.detach().cpu().item())
        length = hidden.shape[1] * block_size
        gate_audio = upsample(gate, self.block_size)[:, :length]
        note_age_audio = upsample(note_age, self.block_size)[:, :length]
        gain = scale_function(self.transient_gain(hidden))
        gain = upsample(gain, self.block_size)[:, :length]

        if articulation is None:
            articulation = torch.zeros(
                hidden.shape[:2],
                dtype=torch.long,
                device=hidden.device,
            )
        if articulation.ndim == 3:
            articulation = articulation.squeeze(-1)
        articulation_audio = articulation.to(hidden.device).long()
        articulation_audio = articulation_audio.repeat_interleave(
            block_size,
            dim=1,
        )[:, :length]

        age_seconds = note_age_audio.squeeze(-1).clamp_min(0.0)
        sample_index = (age_seconds * self.sampling_rate).long()
        sample_index = sample_index.clamp(0, self.transient_bank.shape[1] - 1)
        articulation_audio = articulation_audio.clamp(
            0,
            self.transient_bank.shape[0] - 1,
        )

        waveform = self.transient_bank[
            articulation_audio.reshape(-1),
            sample_index.reshape(-1),
        ].reshape(hidden.shape[0], length, 1)
        envelope = (
            1.0 - age_seconds.unsqueeze(-1) / max(self.transient_seconds, 1e-4)
        ).clamp(0.0, 1.0) ** 2
        return waveform * envelope * gate_audio * gain

    def _forward_impl(
        self,
        pitch,
        loudness,
        articulation=None,
        onset=None,
        offset=None,
        gate=None,
        note_age=None,
        realtime=False,
    ):
        gate = self._frame_control(gate, pitch, 1.0)
        note_age = self._frame_control(note_age, pitch, 0.0)
        hidden = self._decoder_hidden(
            pitch,
            loudness,
            articulation,
            onset,
            offset,
            gate,
            note_age,
            realtime,
        )

        block_size = int(self.block_size.detach().cpu().item())
        length = hidden.shape[1] * block_size
        zero = torch.zeros(
            hidden.shape[0],
            length,
            1,
            dtype=hidden.dtype,
            device=hidden.device,
        )

        sustain_raw = (
            self._harmonic_branch(hidden, pitch, gate, realtime)
            if self.use_sustain_branch
            else zero
        )
        noise_raw = (
            self._noise_branch(hidden, gate)
            if self.use_noise_branch
            else zero
        )
        transient_raw = (
            self._transient_branch(hidden, articulation, gate, note_age)
            if self.use_transient_branch
            else zero
        )

        sustain = sustain_raw * self._branch_gain(0, sustain_raw)
        noise = noise_raw * self._branch_gain(1, noise_raw)
        transient = transient_raw * self._branch_gain(2, transient_raw)
        signal = sustain + noise + transient
        if self.use_reverb:
            signal = self.reverb(signal)

        self.last_branch_outputs = {
            "sustain": sustain,
            "noise": noise,
            "transient": transient,
            "signal": signal,
            "sustain_raw": sustain_raw,
            "noise_raw": noise_raw,
            "transient_raw": transient_raw,
        }
        return signal

    def forward(
        self,
        pitch,
        loudness,
        articulation=None,
        onset=None,
        offset=None,
        gate=None,
        note_age=None,
    ):
        return self._forward_impl(
            pitch,
            loudness,
            articulation=articulation,
            onset=onset,
            offset=offset,
            gate=gate,
            note_age=note_age,
            realtime=False,
        )

    def realtime_forward(
        self,
        pitch,
        loudness,
        articulation=None,
        onset=None,
        offset=None,
        gate=None,
        note_age=None,
    ):
        return self._forward_impl(
            pitch,
            loudness,
            articulation=articulation,
            onset=onset,
            offset=offset,
            gate=gate,
            note_age=note_age,
            realtime=True,
        )
