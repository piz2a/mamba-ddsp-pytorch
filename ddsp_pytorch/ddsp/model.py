import torch
import torch.nn as nn
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve
from .core import resample
import math
import sys
from pathlib import Path


def _import_mamba():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        return Mamba
    except ImportError:
        workspace_mamba = Path(__file__).resolve().parents[2] / "mamba"
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
        hidden = self.input_proj(hidden)
        hidden = self.mamba(hidden)
        return self.norm(hidden)

    def reset_state(self):
        self.conv_state = torch.empty(0, device=self.input_proj.weight.device)
        self.ssm_state = torch.empty(0, device=self.input_proj.weight.device)

    def _ensure_state(self, hidden):
        batch = hidden.shape[0]
        expected_conv = (batch, self.mamba.d_model * self.mamba.expand,
                         self.mamba.d_conv)
        expected_ssm = (batch, self.mamba.d_model * self.mamba.expand,
                        self.mamba.d_state)
        if (tuple(self.conv_state.shape) != expected_conv
                or tuple(self.ssm_state.shape) != expected_ssm
                or self.conv_state.device != hidden.device
                or self.ssm_state.device != hidden.device
                or self.conv_state.dtype != hidden.dtype
                or self.ssm_state.dtype != hidden.dtype):
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
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class DDSP(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size, recurrent_type="gru", mamba_d_state=16,
                 mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.recurrent_type = recurrent_type

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        if recurrent_type == "gru":
            self.gru = gru(2, hidden_size)
            self.recurrent = self.gru
        elif recurrent_type == "mamba":
            self.recurrent = MambaRecurrent(
                2 * hidden_size,
                hidden_size,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
        else:
            raise ValueError(
                "recurrent_type must be either 'gru' or 'mamba', "
                f"got {recurrent_type!r}"
            )
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        if recurrent_type == "gru":
            self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def _run_recurrent(self, hidden):
        if self.recurrent_type == "gru":
            return self.gru(hidden)[0]
        return self.recurrent(hidden)

    def _run_recurrent_realtime(self, hidden):
        if self.recurrent_type == "gru":
            expected_cache = (1, hidden.shape[0], self.gru.hidden_size)
            if (tuple(self.cache_gru.shape) != expected_cache
                    or self.cache_gru.device != hidden.device
                    or self.cache_gru.dtype != hidden.dtype):
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

    def forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self._run_recurrent(hidden), pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)

        return signal

    def realtime_forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        recurrent_out = self._run_recurrent_realtime(hidden)
        hidden = torch.cat([recurrent_out, pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)

        omega = omega + self.phase
        self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)

        harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        return signal
