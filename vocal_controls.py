from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import importlib
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import types
import warnings

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import torch


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
ARTICULATION_LABELS = ("FS_NO", "MU_NO", "PK_NO", "SP_NO", "ST_NO", "FS_DN")
UNKNOWN_ARTICULATION_ID = -1


@dataclass
class VocalControlConfig:
    workspace_root: Path = Path("/workspace")
    input_dir: Path | None = None
    contentvec_repo: Path | None = None
    silero_vad_repo: Path | None = None
    contentvec_checkpoint: Path | None = None

    sample_rate: int = 16000
    hop_seconds: float = 0.016
    frame_seconds: float = 0.032
    n_mfcc: int = 20
    high_freq_cutoff_hz: float = 3000.0
    low_mid_min_hz: float = 250.0
    low_mid_max_hz: float = 3000.0

    silero_chunk_samples: int = 512
    silero_open_threshold: float = 0.50
    silero_close_threshold: float = 0.35

    # Final gate is not Silero. Silero is diagnostic/helpful evidence only.
    fused_gate_open_threshold: float = 0.34
    fused_gate_close_threshold: float = 0.22
    noise_floor_percentile: float = 10.0
    energy_margin_db: float = 6.0
    energy_softness_db: float = 4.0
    silero_activity_weight: float = 0.35

    torchcrepe_model: str = "tiny"
    torchcrepe_device: str = "cpu"
    torchcrepe_fmin: float = 50.0
    torchcrepe_fmax: float = 1000.0

    contentvec_frame_seconds: float = 0.020
    contentvec_layer: int = 12
    contentvec_spk_emb_dim: int = 256

    onset_min_distance_seconds: float = 0.200
    onset_internal_height: float = 0.85
    onset_internal_prominence: float = 0.15
    onset_classify_seconds: float = 0.128

    articulation_labels: tuple[str, ...] = field(default_factory=lambda: ARTICULATION_LABELS)

    def __post_init__(self):
        self.workspace_root = Path(self.workspace_root).expanduser()
        if self.input_dir is None:
            self.input_dir = self.workspace_root / "learn" / "voice_inputs"
        else:
            self.input_dir = Path(self.input_dir).expanduser()
        if self.contentvec_repo is None:
            self.contentvec_repo = self.workspace_root / "contentvec"
        else:
            self.contentvec_repo = Path(self.contentvec_repo).expanduser()
        if self.silero_vad_repo is None:
            self.silero_vad_repo = self.workspace_root / "silero-vad"
        else:
            self.silero_vad_repo = Path(self.silero_vad_repo).expanduser()

        env_checkpoint = os.environ.get("CONTENTVEC_CHECKPOINT", "").strip()
        default_checkpoint = self.contentvec_repo / "checkpoints" / "checkpoint_best_legacy_100.pt"
        if self.contentvec_checkpoint is not None:
            self.contentvec_checkpoint = Path(self.contentvec_checkpoint).expanduser()
        elif env_checkpoint:
            self.contentvec_checkpoint = Path(env_checkpoint).expanduser()
        elif default_checkpoint.exists():
            self.contentvec_checkpoint = default_checkpoint

        self.input_dir.mkdir(parents=True, exist_ok=True)

    @property
    def hop_length(self) -> int:
        return int(round(self.hop_seconds * self.sample_rate))

    @property
    def win_length(self) -> int:
        return int(round(self.frame_seconds * self.sample_rate))

    @property
    def n_fft(self) -> int:
        n_fft = 1
        while n_fft < self.win_length:
            n_fft *= 2
        return n_fft


def resolve_workspace_root() -> Path:
    for candidate in (Path("/workspace"), Path.cwd(), Path.cwd().parent):
        if (candidate / "learn").exists() or (candidate / "contentvec").exists():
            return candidate
    return Path.cwd()


def ensure_notebook_dependencies(config: VocalControlConfig | None = None):
    config = config or VocalControlConfig(resolve_workspace_root())
    packages = [
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("ipywidgets", "ipywidgets"),
        ("imageio-ffmpeg", "imageio_ffmpeg"),
    ]
    if config.contentvec_checkpoint is not None and config.contentvec_checkpoint.exists():
        packages.append(("fairseq", "fairseq"))
    for package, import_name in packages:
        if importlib.util.find_spec(import_name) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def list_audio_inputs(input_dir: Path, extensions: tuple[str, ...] = SUPPORTED_AUDIO_EXTENSIONS):
    input_dir = Path(input_dir)
    files = []
    if not input_dir.exists():
        return []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
        files.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)


def synthesize_fallback_scat(sr: int = 16000):
    rng = np.random.default_rng(7)
    syllables = []
    f0s = [105.0, 138.0, 92.0, 164.0]
    durations = [0.42, 0.36, 0.50, 0.44]
    for i, (f0, dur) in enumerate(zip(f0s, durations)):
        n = int(round(dur * sr))
        t = np.arange(n) / sr
        onset_noise = rng.normal(0, 1, n) * np.exp(-t / 0.018)
        vowel_env = (1.0 - np.exp(-t / 0.035)) * np.exp(-t / 0.65)
        vibrato = 0.012 * np.sin(2 * np.pi * 5.5 * t + i)
        phase = 2 * np.pi * np.cumsum(f0 * (1 + vibrato)) / sr
        harmonic = (
            np.sin(phase)
            + 0.35 * np.sin(2 * phase + 0.4)
            + 0.16 * np.sin(3 * phase + 1.1)
        )
        syllables.append((0.65 * harmonic * vowel_env + 0.20 * onset_noise).astype(np.float32))
        syllables.append(np.zeros(int(0.06 * sr), dtype=np.float32))
    y = np.concatenate(syllables)
    y = y / max(np.max(np.abs(y)), 1e-7) * 0.9
    return y.astype(np.float32), sr, "synthetic_fallback_scat.wav"


def _decode_audio_with_ffmpeg(audio_path: Path, target_sr: int):
    try:
        import imageio_ffmpeg
    except Exception as exc:
        raise RuntimeError(
            "imageio-ffmpeg is unavailable. Install it or convert the file to WAV/FLAC."
        ) from exc
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed for {audio_path}: {stderr}")
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if y.size == 0:
        raise RuntimeError(f"ffmpeg decoded zero samples from {audio_path}")
    return y.astype(np.float32), target_sr


def load_audio_file(audio_path: Path, target_sr: int = 16000):
    audio_path = Path(audio_path).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file does not exist: {audio_path}")
    if audio_path.stat().st_size == 0:
        raise ValueError(f"audio file is empty: {audio_path}")

    compressed = {".m4a", ".aac", ".mp3", ".ogg"}
    prefer_ffmpeg = audio_path.suffix.lower() in compressed
    first_exc = None
    if not prefer_ffmpeg:
        try:
            y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return y.astype(np.float32), sr, "librosa/soundfile"
        except Exception as exc:
            first_exc = exc

    try:
        y, sr = _decode_audio_with_ffmpeg(audio_path, target_sr)
        return y.astype(np.float32), sr, "imageio-ffmpeg"
    except Exception as second_exc:
        if prefer_ffmpeg:
            try:
                y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
                return y.astype(np.float32), sr, "librosa/soundfile"
            except Exception as exc:
                first_exc = exc
        raise RuntimeError(
            "Could not decode audio file. "
            f"Path: {audio_path}\n"
            f"File size: {audio_path.stat().st_size} bytes\n"
            f"librosa/soundfile error: {type(first_exc).__name__}: {first_exc}\n"
            f"ffmpeg fallback error: {type(second_exc).__name__}: {second_exc}\n"
            "Fast workaround: convert the recording to WAV/FLAC."
        ) from second_exc


def load_selected_audio(config: VocalControlConfig, audio_path: str | Path | None = None):
    if audio_path:
        chosen_path = Path(audio_path).expanduser()
        reason = "explicit AUDIO_PATH"
    else:
        candidates = list_audio_inputs(config.input_dir)
        chosen_path = candidates[0] if candidates else None
        reason = "newest file in voice_inputs" if chosen_path is not None else "fallback synthetic signal"

    if chosen_path is None:
        y, sr, source = synthesize_fallback_scat(config.sample_rate)
        decoder = "synthetic"
    else:
        y, sr, decoder = load_audio_file(chosen_path, config.sample_rate)
        source = str(chosen_path)

    y, _ = librosa.effects.trim(y, top_db=45)
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-7:
        y = y / peak * 0.95
    return {
        "y": y,
        "sr": sr,
        "source_name": source,
        "source_reason": f"{reason}; decoder={decoder}",
        "available_inputs": list_audio_inputs(config.input_dir),
    }


def display_upload_widget(input_dir: Path, extensions: tuple[str, ...] = SUPPORTED_AUDIO_EXTENSIONS):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:
        print("ipywidgets upload UI is unavailable:", type(exc).__name__, exc)
        print("Manual path: copy audio files into", input_dir)
        return None

    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    uploader = widgets.FileUpload(
        accept=",".join(extensions),
        multiple=False,
        description="Choose audio",
    )
    button = widgets.Button(description="Save uploaded audio", button_style="primary")
    output = widgets.Output()

    def iter_uploads(value):
        if isinstance(value, dict):
            return list(value.values())
        if isinstance(value, (tuple, list)):
            return list(value)
        return []

    def save_uploaded_audio(_=None):
        with output:
            output.clear_output()
            items = iter_uploads(uploader.value)
            if not items:
                print("No file selected yet.")
                return
            item = items[0]
            name = item.get("name", "uploaded_audio.wav")
            content = item.get("content", b"")
            if isinstance(content, memoryview):
                content = content.tobytes()
            out_path = input_dir / Path(name).name
            out_path.write_bytes(bytes(content))
            print("Saved:", out_path)
            print("Set AUDIO_PATH to this path or rerun the loader to auto-pick newest.")

    button.on_click(save_uploaded_audio)
    box = widgets.VBox([uploader, button, output])
    display(box)
    return box


def align_length(x, target_len: int, fill: float = 0.0):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == target_len:
        return x
    if len(x) == 0:
        return np.full(target_len, fill, dtype=np.float32)
    if len(x) > target_len:
        return x[:target_len].astype(np.float32)
    return np.pad(x, (0, target_len - len(x)), constant_values=fill).astype(np.float32)


def robust_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    lo, hi = np.percentile(x, [5, 95])
    return np.clip((x - lo) / max(float(hi - lo), 1e-7), 0.0, 1.0).astype(np.float32)


def masked_zscore(x, mask=None):
    x = np.asarray(x, dtype=np.float32)
    if mask is not None and np.any(mask):
        ref = x[np.asarray(mask).astype(bool)]
    else:
        ref = x
    mu = float(np.mean(ref)) if ref.size else 0.0
    sigma = float(np.std(ref)) if ref.size else 1.0
    return ((x - mu) / max(sigma, 1e-6)).astype(np.float32)


def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))).astype(np.float32)


def compute_frame_features(y, config: VocalControlConfig):
    sr = config.sample_rate
    hop_length = config.hop_length
    win_length = config.win_length
    n_fft = config.n_fft

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
    )
    mag = np.abs(stft).astype(np.float32)
    power = mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length)
    frame_end_times = times + config.hop_seconds / 2.0

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=config.n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    zcr = align_length(
        librosa.feature.zero_crossing_rate(
            y,
            frame_length=win_length,
            hop_length=hop_length,
            center=True,
        )[0],
        len(times),
    )

    mag_norm = mag / np.maximum(np.sum(mag, axis=0, keepdims=True), 1e-8)
    diff = np.diff(mag_norm, axis=1)
    positive_diff = np.maximum(diff, 0.0)
    negative_diff = np.maximum(-diff, 0.0)
    spectral_flux = np.concatenate([[0.0], np.sqrt(np.sum(positive_diff ** 2, axis=0))]).astype(np.float32)
    spectral_drop = np.concatenate([[0.0], np.sqrt(np.sum(negative_diff ** 2, axis=0))]).astype(np.float32)

    frame_rms = align_length(
        librosa.feature.rms(S=mag, frame_length=n_fft, hop_length=hop_length, center=True)[0],
        len(times),
    )
    rms_db = (20.0 * np.log10(np.maximum(frame_rms, 1e-7))).astype(np.float32)

    try:
        _, percussive_stft = librosa.decompose.hpss(stft, margin=8.0)
        hpss_percussive = np.mean(np.abs(percussive_stft), axis=0).astype(np.float32)
    except Exception as exc:
        print("HPSS failed:", type(exc).__name__, exc)
        hpss_percussive = np.zeros(len(times), dtype=np.float32)
    hpss_percussive = align_length(hpss_percussive, len(times))

    high_mask = freqs >= config.high_freq_cutoff_hz
    low_mid_mask = (freqs >= config.low_mid_min_hz) & (freqs < config.low_mid_max_hz)
    total_energy = np.maximum(np.sum(power, axis=0), 1e-10)
    high_energy = np.sum(power[high_mask], axis=0) if np.any(high_mask) else np.zeros_like(total_energy)
    low_mid_energy = np.sum(power[low_mid_mask], axis=0) if np.any(low_mid_mask) else np.zeros_like(total_energy)
    high_freq_ratio = np.clip(high_energy / total_energy, 0.0, 1.0).astype(np.float32)
    spectral_tilt = (10.0 * np.log10((high_energy + 1e-10) / (low_mid_energy + 1e-10))).astype(np.float32)

    feature_tensor = np.concatenate(
        [
            mfcc.T,
            mfcc_delta.T,
            zcr[:, None],
            spectral_flux[:, None],
            spectral_drop[:, None],
            rms_db[:, None],
            high_freq_ratio[:, None],
            spectral_tilt[:, None],
            hpss_percussive[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    feature_names = (
        [f"mfcc_{i:02d}" for i in range(config.n_mfcc)]
        + [f"delta_mfcc_{i:02d}" for i in range(config.n_mfcc)]
        + [
            "zcr",
            "spectral_flux",
            "spectral_drop",
            "rms_db",
            "high_freq_ratio",
            "spectral_tilt_db",
            "hpss_percussive",
        ]
    )

    return {
        "stft": stft,
        "mag": mag,
        "power": power,
        "freqs": freqs,
        "times": times.astype(np.float32),
        "frame_end_times": frame_end_times.astype(np.float32),
        "mfcc": mfcc.astype(np.float32),
        "mfcc_delta": mfcc_delta.astype(np.float32),
        "zcr": zcr.astype(np.float32),
        "zcr_norm": robust_normalize(zcr),
        "spectral_flux": spectral_flux,
        "spectral_drop": spectral_drop,
        "flux_norm": robust_normalize(spectral_flux),
        "drop_norm": robust_normalize(spectral_drop),
        "frame_rms": frame_rms.astype(np.float32),
        "rms_db": rms_db,
        "energy_norm": robust_normalize(rms_db),
        "hpss_percussive": hpss_percussive,
        "hpss_onset_norm": robust_normalize(hpss_percussive),
        "high_freq_ratio": high_freq_ratio,
        "high_freq_ratio_norm": robust_normalize(high_freq_ratio),
        "spectral_tilt": spectral_tilt,
        "spectral_tilt_norm": robust_normalize(spectral_tilt),
        "feature_tensor": feature_tensor,
        "feature_names": feature_names,
    }


def causal_hysteresis(prob, open_threshold, close_threshold):
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    gate = np.zeros_like(prob, dtype=np.float32)
    active = False
    for i, value in enumerate(prob):
        if active:
            if value <= close_threshold:
                active = False
        else:
            if value >= open_threshold:
                active = True
        gate[i] = 1.0 if active else 0.0
    return gate


def pulses_from_gate(gate):
    gate = np.asarray(gate, dtype=np.float32).reshape(-1)
    prev = np.concatenate([[0.0], gate[:-1]])
    onset = ((gate > 0.5) & (prev <= 0.5)).astype(np.float32)
    offset = ((gate <= 0.5) & (prev > 0.5)).astype(np.float32)
    return onset, offset


def causal_note_age(gate, onset, frame_seconds):
    gate = np.asarray(gate, dtype=np.float32).reshape(-1)
    onset = np.asarray(onset, dtype=np.float32).reshape(-1)
    age = np.zeros_like(gate, dtype=np.float32)
    current_age = 0.0
    for i in range(len(gate)):
        if onset[i] > 0.5:
            current_age = 0.0
        if gate[i] > 0.5:
            age[i] = current_age
            current_age += frame_seconds
        else:
            current_age = 0.0
            age[i] = 0.0
    return age


def _sample_hold_to_frame_times(values, available_times, target_times, fill=0.0):
    values = np.asarray(values, dtype=np.float32)
    available_times = np.asarray(available_times, dtype=np.float32)
    target_times = np.asarray(target_times, dtype=np.float32)
    if values.size == 0 or available_times.size == 0 or target_times.size == 0:
        return np.full(target_times.shape, fill, dtype=np.float32)
    idx = np.searchsorted(available_times, target_times, side="right") - 1
    out = np.full(target_times.shape, fill, dtype=np.float32)
    valid = idx >= 0
    out[valid] = values[np.clip(idx[valid], 0, len(values) - 1)]
    if np.any(~valid):
        out[~valid] = values[0]
    return out.astype(np.float32)


def _interp_to_frame_times(values, value_times, target_times, fill=0.0):
    values = np.asarray(values, dtype=np.float32)
    value_times = np.asarray(value_times, dtype=np.float32)
    target_times = np.asarray(target_times, dtype=np.float32)
    if values.size == 0 or value_times.size == 0 or target_times.size == 0:
        return np.full(target_times.shape, fill, dtype=np.float32)
    return np.interp(
        target_times,
        value_times,
        values,
        left=float(values[0]),
        right=float(values[-1]),
    ).astype(np.float32)


def run_causal_silero_vad(y, config: VocalControlConfig):
    jit_path = config.silero_vad_repo / "src" / "silero_vad" / "data" / "silero_vad.jit"
    if not jit_path.exists():
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            f"missing Silero JIT file: {jit_path}",
        )
    model = torch.jit.load(str(jit_path), map_location="cpu")
    model.eval()
    if hasattr(model, "reset_states"):
        model.reset_states()

    wav = torch.from_numpy(np.asarray(y, dtype=np.float32))
    pad = (-len(wav)) % config.silero_chunk_samples
    if pad:
        wav = torch.nn.functional.pad(wav, (0, pad))

    probs = []
    available_times = []
    with torch.no_grad():
        for start in range(0, wav.numel(), config.silero_chunk_samples):
            chunk = wav[start:start + config.silero_chunk_samples]
            prob = float(model(chunk, config.sample_rate).reshape(-1)[0].item())
            probs.append(prob)
            available_times.append((start + config.silero_chunk_samples) / config.sample_rate)

    return (
        np.asarray(probs, dtype=np.float32),
        np.asarray(available_times, dtype=np.float32),
        f"loaded {jit_path}",
    )


def import_torchcrepe_safe():
    try:
        def install_torchaudio_stub():
            torchaudio = types.ModuleType("torchaudio")
            torchaudio.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)

            def missing_load(*args, **kwargs):
                raise RuntimeError(
                    "torchaudio file loading is unavailable in this container; "
                    "audio is decoded by librosa/ffmpeg."
                )

            torchaudio.load = missing_load
            sys.modules["torchaudio"] = torchaudio

        for module_name in list(sys.modules):
            if module_name == "torchaudio" or module_name.startswith("torchaudio."):
                del sys.modules[module_name]
        install_torchaudio_stub()

        try:
            import torchcrepe
            return torchcrepe, "imported torchcrepe with torchaudio stub"
        except Exception:
            for module_name in list(sys.modules):
                if module_name == "torchcrepe" or module_name.startswith("torchcrepe."):
                    del sys.modules[module_name]
            install_torchaudio_stub()
            import torchcrepe
            return torchcrepe, "imported torchcrepe with torchaudio stub after retry"
    except Exception as exc:
        return None, f"torchcrepe unavailable: {type(exc).__name__}: {exc}"


def extract_torchcrepe_controls(y, frame_count: int, config: VocalControlConfig):
    torchcrepe, import_status = import_torchcrepe_safe()
    if torchcrepe is None:
        return (
            np.zeros(frame_count, dtype=np.float32),
            np.zeros(frame_count, dtype=np.float32),
            import_status,
        )
    try:
        device = config.torchcrepe_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        wav = torch.from_numpy(np.asarray(y, dtype=np.float32)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            f0, periodicity = torchcrepe.predict(
                wav,
                config.sample_rate,
                hop_length=config.hop_length,
                fmin=config.torchcrepe_fmin,
                fmax=config.torchcrepe_fmax,
                model=config.torchcrepe_model,
                decoder=torchcrepe.decode.weighted_argmax,
                return_periodicity=True,
                batch_size=512,
                device=device,
                pad=True,
            )
        f0 = f0.squeeze(0).detach().cpu().numpy().astype(np.float32)
        periodicity = periodicity.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return (
            align_length(f0, frame_count, fill=0.0),
            np.clip(align_length(periodicity, frame_count, fill=0.0), 0.0, 1.0),
            f"{import_status}; model={config.torchcrepe_model}; device={device}",
        )
    except Exception as exc:
        return (
            np.zeros(frame_count, dtype=np.float32),
            np.zeros(frame_count, dtype=np.float32),
            f"torchcrepe extraction failed: {type(exc).__name__}: {exc}",
        )


_CONTENTVEC_MODEL_CACHE = {}


def _find_contentvec_checkpoint(config: VocalControlConfig):
    candidates = []
    if config.contentvec_checkpoint is not None:
        candidates.append(config.contentvec_checkpoint)
    candidates.extend([
        config.contentvec_repo / "checkpoints" / "checkpoint_best_legacy_100.pt",
        config.workspace_root / "checkpoints" / "contentvec" / "checkpoint_best_legacy_100.pt",
        config.workspace_root / "checkpoints" / "contentvec" / "checkpoint_best_legacy.pt",
        config.workspace_root / "checkpoints" / "contentvec" / "checkpoint_best.pt",
        config.contentvec_repo / "checkpoints" / "checkpoint_best_legacy.pt",
        config.contentvec_repo / "checkpoints" / "checkpoint_best.pt",
    ])
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return Path(candidate)
    return None


def _pca_numpy(x, n_components=2):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return np.zeros((x.shape[0] if x.ndim == 2 else 0, n_components), dtype=np.float32)
    centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n = min(n_components, vt.shape[0])
    projected = centered @ vt[:n].T
    if n < n_components:
        projected = np.pad(projected, ((0, 0), (0, n_components - n)))
    return projected.astype(np.float32)


def extract_contentvec_features(y, config: VocalControlConfig):
    ckpt = _find_contentvec_checkpoint(config)
    if ckpt is None:
        return None, None, "skipped: no ContentVec checkpoint found."
    if not config.contentvec_repo.exists():
        return None, None, f"skipped: missing ContentVec repo at {config.contentvec_repo}"
    if importlib.util.find_spec("fairseq") is None:
        return None, None, "skipped: fairseq is not importable."

    if str(config.contentvec_repo) not in sys.path:
        sys.path.insert(0, str(config.contentvec_repo))

    try:
        import fairseq
    except Exception as exc:
        return None, None, f"skipped: fairseq import failed ({type(exc).__name__}: {exc})"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = (str(ckpt), device, config.contentvec_layer)
    try:
        if cache_key not in _CONTENTVEC_MODEL_CACHE:
            if "legacy" not in ckpt.name:
                try:
                    import contentvec.models.hubert.contentvec  # noqa: F401
                    import contentvec.tasks.contentvec_pretraining  # noqa: F401
                except Exception as exc:
                    warnings.warn(
                        f"ContentVec class registration warning: {type(exc).__name__}: {exc}"
                    )
            models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(ckpt)])
            _CONTENTVEC_MODEL_CACHE[cache_key] = models[0].to(device).eval()

        model = _CONTENTVEC_MODEL_CACHE[cache_key]
        wav = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0).to(device)
        padding_mask = torch.zeros(wav.shape, dtype=torch.bool, device=device)

        with torch.no_grad():
            try:
                out = model.extract_features(
                    source=wav,
                    padding_mask=padding_mask,
                    mask=False,
                    output_layer=config.contentvec_layer,
                )
            except TypeError:
                spk_emb = torch.zeros(1, config.contentvec_spk_emb_dim, device=device)
                try:
                    out = model.extract_features(
                        wav,
                        spk_emb,
                        padding_mask=padding_mask,
                        mask=False,
                        output_layer=config.contentvec_layer,
                    )
                except TypeError:
                    out = model.extract_features(
                        source=wav,
                        spk_emb=spk_emb,
                        padding_mask=padding_mask,
                        mask=False,
                        output_layer=config.contentvec_layer,
                    )

        if isinstance(out, (tuple, list)):
            feats = out[0]
        elif isinstance(out, dict):
            feats = out.get("x", out.get("features"))
        else:
            feats = out
        if feats is None:
            return None, None, "skipped: ContentVec returned no feature tensor."
        feats = feats.squeeze(0).detach().cpu().numpy().astype(np.float32)
        cv_times = np.arange(feats.shape[0], dtype=np.float32) * config.contentvec_frame_seconds
        return feats, cv_times, f"loaded {ckpt} on {device}"
    except Exception as exc:
        return None, None, f"skipped: ContentVec extraction failed ({type(exc).__name__}: {exc})"


def compute_activity_gate(features, periodicity, causal_vad_probs, causal_vad_times, config):
    times = features["times"]
    frame_end_times = features["frame_end_times"]
    vad_prob_interp = _interp_to_frame_times(causal_vad_probs, causal_vad_times, times, fill=0.0)
    vad_prob_hold = _sample_hold_to_frame_times(causal_vad_probs, causal_vad_times, frame_end_times, fill=0.0)
    silero_gate = causal_hysteresis(
        vad_prob_hold,
        config.silero_open_threshold,
        config.silero_close_threshold,
    )

    rms_db = features["rms_db"]
    noise_floor_db = float(np.percentile(rms_db, config.noise_floor_percentile)) if rms_db.size else -120.0
    energy_threshold_db = noise_floor_db + config.energy_margin_db
    energy_activity = _sigmoid((rms_db - energy_threshold_db) / max(config.energy_softness_db, 1e-6))

    periodicity = np.clip(align_length(periodicity, len(times), fill=0.0), 0.0, 1.0)
    voiced_activity = np.sqrt(np.clip(periodicity * energy_activity, 0.0, 1.0)).astype(np.float32)
    noise_shape = np.clip(
        0.45 * features["high_freq_ratio_norm"]
        + 0.35 * features["zcr_norm"]
        + 0.20 * features["hpss_onset_norm"],
        0.0,
        1.0,
    ).astype(np.float32)
    unvoiced_activity = np.clip(energy_activity * noise_shape, 0.0, 1.0).astype(np.float32)
    percussive_activity = np.clip(
        energy_activity
        * (
            0.45 * features["hpss_onset_norm"]
            + 0.30 * features["flux_norm"]
            + 0.25 * features["high_freq_ratio_norm"]
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    silero_activity = np.clip(config.silero_activity_weight * vad_prob_hold, 0.0, 1.0).astype(np.float32)

    activity_evidence = np.maximum.reduce([
        voiced_activity,
        unvoiced_activity,
        percussive_activity,
        silero_activity,
    ]).astype(np.float32)
    gate = causal_hysteresis(
        activity_evidence,
        config.fused_gate_open_threshold,
        config.fused_gate_close_threshold,
    )
    onset, offset = pulses_from_gate(gate)
    silero_onset, silero_offset = pulses_from_gate(silero_gate)

    return {
        "vad_prob_interp_16ms": vad_prob_interp,
        "vad_prob_16ms": vad_prob_hold,
        "silero_gate": silero_gate,
        "silero_onset": silero_onset,
        "silero_offset": silero_offset,
        "energy_activity": energy_activity,
        "energy_threshold_db": energy_threshold_db,
        "noise_floor_db": noise_floor_db,
        "voiced_activity": voiced_activity,
        "unvoiced_activity": unvoiced_activity,
        "percussive_activity": percussive_activity,
        "silero_activity": silero_activity,
        "activity_evidence": activity_evidence,
        "gate": gate,
        "activity_onset": onset,
        "offset": offset,
    }


def detect_internal_note_onsets(gate, boundary_evidence, activity_onset, config):
    gate = np.asarray(gate, dtype=np.float32)
    boundary_evidence = np.asarray(boundary_evidence, dtype=np.float32)
    activity_onset = np.asarray(activity_onset, dtype=np.float32)
    min_distance = max(1, int(round(config.onset_min_distance_seconds / config.hop_seconds)))
    active_evidence = boundary_evidence * gate
    peaks, props = find_peaks(
        active_evidence,
        height=config.onset_internal_height,
        prominence=config.onset_internal_prominence,
        distance=min_distance,
    )
    note_onset = activity_onset.copy()
    for peak in peaks:
        if not np.any(note_onset[max(0, peak - min_distance):peak + 1] > 0.5):
            note_onset[peak] = 1.0
    return note_onset.astype(np.float32), peaks.astype(np.int64), props


def causal_onset_strength(gate, note_onset, raw_evidence, config):
    gate = np.asarray(gate, dtype=np.float32)
    note_onset = np.asarray(note_onset, dtype=np.float32)
    raw_evidence = np.asarray(raw_evidence, dtype=np.float32)
    window_frames = max(1, int(round(config.onset_classify_seconds / config.hop_seconds)))
    out = np.zeros_like(raw_evidence, dtype=np.float32)
    age = -1
    peak = 0.0
    for i in range(len(raw_evidence)):
        if gate[i] <= 0.5:
            age = -1
            peak = 0.0
            continue
        if note_onset[i] > 0.5:
            age = 0
            peak = max(0.10, float(raw_evidence[i]))
        elif age >= 0:
            age += 1
        if 0 <= age < window_frames:
            peak = max(peak * 0.98, float(raw_evidence[i]))
            out[i] = peak
        else:
            out[i] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _articulation_scores_for_frame(i, arrays, labels):
    hf = float(arrays["high_freq_ratio_norm"][i])
    tilt = float(arrays["spectral_tilt_norm"][i])
    per = float(arrays["periodicity"][i])
    z = float(arrays["zcr_norm"][i])
    hp = float(arrays["hpss_onset_norm"][i])
    fl = float(arrays["flux_norm"][i])
    dr = float(arrays["drop_norm"][i])
    loud = float(arrays["energy_norm"][i])
    cv = float(arrays["contentvec_boundary_16ms"][i])
    all_scores = {
        "FS_NO": 0.45 * per + 0.25 * (1.0 - hf) + 0.20 * (1.0 - z) + 0.10 * loud,
        "MU_NO": 0.30 * (1.0 - per) + 0.25 * z + 0.20 * (1.0 - loud) + 0.15 * dr + 0.10 * cv,
        "PK_NO": 0.35 * hp + 0.25 * hf + 0.20 * z + 0.10 * fl + 0.10 * per,
        "SP_NO": 0.35 * hf + 0.25 * tilt + 0.25 * hp + 0.15 * per,
        "ST_NO": 0.45 * hf + 0.35 * tilt + 0.10 * z + 0.10 * (1.0 - per),
        "FS_DN": 0.35 * dr + 0.25 * (1.0 - per) + 0.20 * fl + 0.10 * cv + 0.10 * (1.0 - hf),
    }
    scores = np.asarray([all_scores[label] for label in labels], dtype=np.float32)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    return (probs / np.maximum(np.sum(probs), 1e-8)).astype(np.float32)


def latch_articulation(gate, note_onset, offset, arrays, config):
    labels = config.articulation_labels
    gate = np.asarray(gate, dtype=np.float32)
    note_onset = np.asarray(note_onset, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    classify_frames = max(1, int(round(config.onset_classify_seconds / config.hop_seconds)))
    articulation_id = np.full(len(gate), UNKNOWN_ARTICULATION_ID, dtype=np.int64)
    confidence = np.zeros(len(gate), dtype=np.float32)
    probs_track = np.zeros((len(gate), len(labels)), dtype=np.float32)
    accum = np.zeros(len(labels), dtype=np.float32)
    current_id = UNKNOWN_ARTICULATION_ID
    age = -1
    n_accum = 0
    for i in range(len(gate)):
        if gate[i] <= 0.5:
            current_id = UNKNOWN_ARTICULATION_ID
            age = -1
            n_accum = 0
            accum[:] = 0.0
            continue
        if note_onset[i] > 0.5:
            age = 0
            n_accum = 0
            accum[:] = 0.0
        elif age >= 0:
            age += 1
        if 0 <= age < classify_frames:
            probs = _articulation_scores_for_frame(i, arrays, labels)
            accum += probs
            n_accum += 1
            avg = accum / max(n_accum, 1)
            current_id = int(np.argmax(avg))
            confidence[i] = float(np.max(avg))
            probs_track[i] = avg
        else:
            if current_id >= 0:
                confidence[i] = confidence[i - 1] if i > 0 else 0.0
                probs_track[i] = probs_track[i - 1] if i > 0 else 0.0
        articulation_id[i] = current_id
        if offset[i] > 0.5:
            current_id = UNKNOWN_ARTICULATION_ID
            age = -1
            n_accum = 0
            accum[:] = 0.0
    names = np.asarray([labels[int(idx)] if idx >= 0 else "NONE" for idx in articulation_id], dtype=object)
    return articulation_id, names, confidence, probs_track


def extract_voice_controls(audio, config: VocalControlConfig | None = None, source_name: str | None = None):
    config = config or VocalControlConfig(resolve_workspace_root())
    if isinstance(audio, dict):
        y = np.asarray(audio["y"], dtype=np.float32).reshape(-1)
        if "sr" in audio and int(audio["sr"]) != config.sample_rate:
            y = librosa.resample(y, orig_sr=int(audio["sr"]), target_sr=config.sample_rate).astype(np.float32)
        source_name = source_name or audio.get("source_name", "audio")
    elif isinstance(audio, (str, Path)):
        loaded = load_selected_audio(config, audio)
        y = np.asarray(loaded["y"], dtype=np.float32).reshape(-1)
        source_name = source_name or loaded["source_name"]
    else:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        source_name = source_name or "audio"
    features = compute_frame_features(y, config)
    frame_count = len(features["times"])

    causal_vad_probs, causal_vad_times, causal_vad_status = run_causal_silero_vad(y, config)
    f0_hz, periodicity, torchcrepe_status = extract_torchcrepe_controls(y, frame_count, config)
    contentvec_features, contentvec_times, contentvec_status = extract_contentvec_features(y, config)

    if contentvec_features is not None:
        contentvec_pca = _pca_numpy(contentvec_features, 2)
        norms = np.linalg.norm(contentvec_features, axis=1, keepdims=True).clip(1e-8)
        unit = contentvec_features / norms
        cosine_similarity = np.sum(unit[1:] * unit[:-1], axis=1)
        contentvec_boundary = np.concatenate([[0.0], 1.0 - cosine_similarity]).astype(np.float32)
        contentvec_novelty = np.concatenate([
            [0.0],
            np.linalg.norm(np.diff(contentvec_features, axis=0), axis=1),
        ]).astype(np.float32)
        contentvec_boundary_norm = robust_normalize(contentvec_boundary)
        contentvec_novelty_norm = robust_normalize(contentvec_novelty)
        contentvec_boundary_16ms = _interp_to_frame_times(
            contentvec_boundary_norm,
            contentvec_times,
            features["times"],
            fill=0.0,
        )
        contentvec_novelty_16ms = _interp_to_frame_times(
            contentvec_novelty_norm,
            contentvec_times,
            features["times"],
            fill=0.0,
        )
        contentvec_pca_16ms = np.stack([
            _interp_to_frame_times(contentvec_pca[:, 0], contentvec_times, features["times"], fill=0.0),
            _interp_to_frame_times(contentvec_pca[:, 1], contentvec_times, features["times"], fill=0.0),
        ], axis=1).astype(np.float32)
    else:
        contentvec_pca = np.zeros((0, 2), dtype=np.float32)
        contentvec_boundary = np.zeros(0, dtype=np.float32)
        contentvec_novelty = np.zeros(0, dtype=np.float32)
        contentvec_boundary_16ms = np.zeros(frame_count, dtype=np.float32)
        contentvec_novelty_16ms = np.zeros(frame_count, dtype=np.float32)
        contentvec_pca_16ms = np.zeros((frame_count, 2), dtype=np.float32)

    gate_data = compute_activity_gate(features, periodicity, causal_vad_probs, causal_vad_times, config)
    gate = gate_data["gate"]
    offset = gate_data["offset"]
    loudness_z = masked_zscore(features["rms_db"], gate > 0.5)

    vad_rise = np.maximum(
        np.diff(np.concatenate([[gate_data["vad_prob_16ms"][0] if len(gate_data["vad_prob_16ms"]) else 0.0],
                                gate_data["vad_prob_16ms"]])),
        0.0,
    ).astype(np.float32)
    vad_rise_norm = robust_normalize(vad_rise)

    boundary_evidence = robust_normalize(
        0.30 * features["hpss_onset_norm"]
        + 0.20 * features["flux_norm"]
        + 0.20 * features["high_freq_ratio_norm"]
        + 0.20 * contentvec_boundary_16ms
        + 0.10 * features["zcr_norm"]
    )
    note_onset, internal_onset_frames, internal_onset_properties = detect_internal_note_onsets(
        gate,
        boundary_evidence,
        gate_data["activity_onset"],
        config,
    )
    note_age = causal_note_age(gate, note_onset, config.hop_seconds)
    onset_strength = causal_onset_strength(gate, note_onset, boundary_evidence, config)

    articulation_arrays = {
        **features,
        "periodicity": periodicity,
        "contentvec_boundary_16ms": contentvec_boundary_16ms,
    }
    articulation_id, articulation_name, articulation_confidence, articulation_probs = latch_articulation(
        gate,
        note_onset,
        offset,
        articulation_arrays,
        config,
    )

    control_tensor = np.stack([
        f0_hz,
        loudness_z,
        gate,
        onset_strength,
        offset,
        note_age,
        periodicity,
        note_onset,
        articulation_id.astype(np.float32),
        articulation_confidence,
        gate_data["activity_evidence"],
        gate_data["energy_activity"],
        gate_data["voiced_activity"],
        gate_data["unvoiced_activity"],
        gate_data["percussive_activity"],
        gate_data["vad_prob_16ms"],
        gate_data["silero_gate"],
        gate_data["vad_prob_interp_16ms"],
        boundary_evidence,
        features["hpss_onset_norm"],
        features["flux_norm"],
        features["high_freq_ratio"],
        features["spectral_tilt"],
        contentvec_boundary_16ms,
    ], axis=1).astype(np.float32)

    control_names = [
        "f0_hz_torchcrepe",
        "loudness_rms_z",
        "gate_fused_activity",
        "onset_strength",
        "offset_fused_activity",
        "note_age_seconds",
        "periodicity_torchcrepe",
        "note_onset",
        "articulation_id",
        "articulation_confidence",
        "activity_evidence",
        "energy_activity_relative",
        "voiced_activity",
        "unvoiced_activity",
        "percussive_activity",
        "silero_vad_prob_sample_hold",
        "silero_gate_diagnostic",
        "silero_vad_prob_interp_display",
        "boundary_evidence",
        "hpss_onset_candidate",
        "spectral_flux_candidate",
        "high_freq_ratio",
        "spectral_tilt_db",
        "contentvec_boundary_norm",
    ]

    optional_blocks = []
    optional_names = []
    if contentvec_features is not None:
        optional_blocks.append(contentvec_pca_16ms.astype(np.float32))
        optional_names.extend(["contentvec_pca_0", "contentvec_pca_1"])
        optional_blocks.append(contentvec_novelty_16ms[:, None].astype(np.float32))
        optional_names.append("contentvec_novelty_norm")

    feature_tensor_plus = np.concatenate(
        [features["feature_tensor"], control_tensor] + optional_blocks,
        axis=1,
    ).astype(np.float32)
    feature_names_plus = features["feature_names"] + control_names + optional_names

    return {
        **features,
        **gate_data,
        "config": config,
        "y": y,
        "sr": config.sample_rate,
        "source_name": source_name,
        "causal_vad_probs": causal_vad_probs,
        "causal_vad_times": causal_vad_times,
        "causal_vad_status": causal_vad_status,
        "f0_hz": f0_hz,
        "periodicity": periodicity,
        "torchcrepe_status": torchcrepe_status,
        "contentvec_features": contentvec_features,
        "contentvec_times": contentvec_times,
        "contentvec_status": contentvec_status,
        "contentvec_pca": contentvec_pca,
        "contentvec_pca_16ms": contentvec_pca_16ms,
        "contentvec_boundary": contentvec_boundary,
        "contentvec_boundary_16ms": contentvec_boundary_16ms,
        "contentvec_novelty": contentvec_novelty,
        "contentvec_novelty_16ms": contentvec_novelty_16ms,
        "loudness_z": loudness_z,
        "vad_rise_norm": vad_rise_norm,
        "boundary_evidence": boundary_evidence,
        "note_onset": note_onset,
        "internal_onset_frames": internal_onset_frames,
        "internal_onset_properties": internal_onset_properties,
        "note_age": note_age,
        "onset_strength": onset_strength,
        "articulation_id": articulation_id,
        "articulation_name": articulation_name,
        "articulation_confidence": articulation_confidence,
        "articulation_probs": articulation_probs,
        "articulation_labels": config.articulation_labels,
        "control_tensor": control_tensor,
        "control_names": control_names,
        "feature_tensor_plus": feature_tensor_plus,
        "feature_names_plus": feature_names_plus,
        "optional_names": optional_names,
    }


def print_summary(result):
    gate = result["gate"]
    periodicity = result["periodicity"]
    f0 = result["f0_hz"]
    active_names = result["articulation_name"][gate > 0.5]
    print("source:", result["source_name"])
    print("duration:", len(result["y"]) / result["sr"], "seconds")
    print("control frame interval:", result["config"].hop_seconds, "seconds")
    print("frames:", len(result["times"]))
    print("ContentVec:", result["contentvec_status"])
    if result["contentvec_features"] is not None:
        print("ContentVec features:", result["contentvec_features"].shape)
    print("TorchCREPE:", result["torchcrepe_status"])
    print("Silero VAD:", result["causal_vad_status"])
    print("Silero gate active frames:", int(np.sum(result["silero_gate"])), "/", len(gate))
    print("Fused gate active frames:", int(np.sum(gate)), "/", len(gate))
    print("Activity noise floor dB:", round(float(result["noise_floor_db"]), 2))
    print("Activity energy threshold dB:", round(float(result["energy_threshold_db"]), 2))
    print("note onset frames:", np.where(result["note_onset"] > 0.5)[0].tolist())
    print("offset frames:", np.where(result["offset"] > 0.5)[0].tolist())
    if np.any(periodicity > 0.2):
        print("f0 range on periodic frames:", (float(np.min(f0[periodicity > 0.2])), float(np.max(f0[periodicity > 0.2]))))
    print("onset_strength range:", float(np.min(result["onset_strength"])), float(np.max(result["onset_strength"])))
    print("articulation labels:", list(result["articulation_labels"]))
    print("active articulation counts:", pd.Series(active_names).value_counts().to_dict() if active_names.size else {})
    print("feature_tensor:", result["feature_tensor"].shape)
    print("control_tensor:", result["control_tensor"].shape)
    print("feature_tensor_plus:", result["feature_tensor_plus"].shape)


def plot_voice_control_dashboard(result):
    y = result["y"]
    sr = result["sr"]
    times = result["times"]
    config = result["config"]
    gate = result["gate"]
    offset = result["offset"]
    note_onset = result["note_onset"]
    internal_onset_frames = result["internal_onset_frames"]
    feature_labels = result["articulation_labels"]

    audio_time = np.arange(len(y)) / sr
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=96,
        ),
        ref=np.max,
    )

    flux_peak_frames, _ = find_peaks(
        result["flux_norm"],
        height=0.35,
        prominence=0.10,
        distance=max(1, int(round(0.080 / config.hop_seconds))),
    )
    flux_peak_times = times[flux_peak_frames]
    note_onset_times = times[note_onset > 0.5]
    offset_times = times[offset > 0.5]
    internal_onset_times = times[internal_onset_frames] if len(internal_onset_frames) else np.zeros(0, dtype=np.float32)

    fig = plt.figure(figsize=(16, 28), constrained_layout=True)
    gs = fig.add_gridspec(11, 2, height_ratios=[1.0, 1.25, 1.0, 1.0, 1.0, 1.05, 1.2, 1.1, 1.1, 1.0, 1.15])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(audio_time, y, color="black", linewidth=0.7, label="audio")
    ax.fill_between(times, -1.0, 1.0, where=gate > 0.5, color="#4c78a8", alpha=0.12, step="mid", label="fused gate")
    for t in note_onset_times:
        ax.axvline(t, color="#2ca02c", alpha=0.9, linewidth=1.1)
    for t in offset_times:
        ax.axvline(t, color="#9467bd", alpha=0.85, linewidth=1.1)
    for t in flux_peak_times:
        ax.axvline(t, color="#ff7f0e", alpha=0.28, linewidth=0.8, linestyle=":")
    ax.set_title("Waveform: fused gate shading, green note onsets, purple offsets, orange flux candidates")
    ax.set_ylabel("amplitude")
    ax.set_xlim(0, audio_time[-1] if len(audio_time) else 0)
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[1, :])
    librosa.display.specshow(mel_db, sr=sr, hop_length=config.hop_length, x_axis="time", y_axis="mel", cmap="magma", ax=ax)
    for t in note_onset_times:
        ax.axvline(t, color="#2ca02c", alpha=0.9, linewidth=0.9)
    for t in offset_times:
        ax.axvline(t, color="#9467bd", alpha=0.85, linewidth=0.9)
    for t in flux_peak_times:
        ax.axvline(t, color="#ff7f0e", alpha=0.32, linewidth=0.7, linestyle=":")
    ax.set_title("Log-mel spectrogram with proposed note boundaries")

    ax = fig.add_subplot(gs[2, :])
    f0_masked = np.where(result["periodicity"] > 0.15, result["f0_hz"], np.nan)
    ax.plot(times, f0_masked, color="#111111", linewidth=1.0, label="TorchCREPE f0 Hz")
    ax.set_ylabel("Hz")
    ax.set_ylim(0, max(500.0, np.nanmax(f0_masked) * 1.15 if np.any(np.isfinite(f0_masked)) else 500.0))
    ax2 = ax.twinx()
    ax2.plot(times, result["periodicity"], color="#1f77b4", alpha=0.7, linewidth=1.0, label="periodicity")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("periodicity")
    ax.set_title("f0(t) and periodicity(t) from TorchCREPE")
    ax.set_xlim(0, times[-1] if len(times) else 0)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = fig.add_subplot(gs[3, :])
    ax.plot(times, result["loudness_z"], color="#2ca02c", linewidth=1.2, label="loudness RMS z-score")
    ax.plot(times, result["energy_activity"], color="#111111", linewidth=1.0, label="relative energy activity")
    ax.plot(times, result["high_freq_ratio_norm"], color="#d62728", linewidth=1.0, label="high-frequency ratio norm")
    ax.plot(times, result["spectral_tilt_norm"], color="#8c564b", linewidth=1.0, label="spectral tilt norm")
    ax.set_title("Loudness and activity/articulation features")
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, times[-1] if len(times) else 0)
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[4, :])
    ax.step(result["causal_vad_times"], result["causal_vad_probs"], where="post", color="#4c78a8", linewidth=1.2, label="Silero VAD prob, diagnostic")
    ax.step(times, result["silero_gate"], where="post", color="#4c78a8", alpha=0.35, linewidth=1.4, label="Silero gate diagnostic")
    ax.plot(times, result["activity_evidence"], color="#111111", linewidth=1.3, label="fused activity evidence")
    ax.step(times, gate, where="post", color="#2ca02c", linewidth=1.5, label="FINAL gate(t)")
    ax.axhline(config.fused_gate_open_threshold, color="#2ca02c", linestyle="--", linewidth=0.8, label=f"fused open {config.fused_gate_open_threshold:.2f}")
    ax.axhline(config.fused_gate_close_threshold, color="#9467bd", linestyle="--", linewidth=0.8, label=f"fused close {config.fused_gate_close_threshold:.2f}")
    ax.set_title("Gate debug: final gate is fused activity, not Silero-only")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("[0, 1]")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, times[-1] if len(times) else 0)
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[5, :])
    ax.plot(times, result["voiced_activity"], color="#1f77b4", linewidth=1.2, label="voiced activity: periodicity x relative energy")
    ax.plot(times, result["unvoiced_activity"], color="#d62728", linewidth=1.2, label="unvoiced activity: energy x HF/ZCR/HPSS")
    ax.plot(times, result["percussive_activity"], color="#ff7f0e", linewidth=1.2, label="percussive activity")
    ax.plot(times, result["silero_activity"], color="#4c78a8", alpha=0.65, linewidth=1.0, label="Silero contribution")
    ax.set_title("Fused gate components: voiced, unvoiced/percussive, and Silero diagnostic")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("[0, 1]")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, times[-1] if len(times) else 0)
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[6, :])
    ax.plot(times, result["hpss_onset_norm"], color="#17becf", linewidth=1.1, label="HPSS percussive")
    ax.plot(times, result["flux_norm"], color="#ff7f0e", linewidth=1.1, label="spectral flux")
    ax.plot(times, result["high_freq_ratio_norm"], color="#d62728", linewidth=1.0, label="HF ratio")
    ax.plot(times, result["contentvec_boundary_16ms"], color="#b279a2", linewidth=1.1, label="ContentVec boundary")
    ax.plot(times, result["boundary_evidence"], color="#111111", linewidth=1.2, label="combined boundary evidence")
    ax.plot(times, result["onset_strength"], color="#e45756", linewidth=1.7, label="onset_strength(t)")
    if len(internal_onset_frames):
        ax.scatter(internal_onset_times, result["boundary_evidence"][internal_onset_frames], color="#2ca02c", s=40, zorder=4, label="accepted internal onsets")
    ax.set_title("onset_strength(t): boundary evidence gated by note-onset state machine")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("[0, 1]")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, times[-1] if len(times) else 0)
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[7, :])
    if result["contentvec_features"] is not None:
        n_show = min(48, result["contentvec_features"].shape[1])
        cv_show = result["contentvec_features"][:, :n_show].T
        im = ax.imshow(
            cv_show,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            extent=[result["contentvec_times"][0], result["contentvec_times"][-1], n_show, 0],
        )
        ax.set_title(f"ContentVec embedding heatmap, first {n_show} channels")
        ax.set_ylabel("channel")
        ax.set_xlabel("time (s)")
        fig.colorbar(im, ax=ax, label="activation")
    else:
        ax.text(0.01, 0.55, result["contentvec_status"], transform=ax.transAxes, va="center", wrap=True)
        ax.set_title("ContentVec embedding heatmap")
        ax.set_axis_off()

    ax = fig.add_subplot(gs[8, 0])
    if result["contentvec_features"] is not None and len(result["contentvec_pca"]):
        sc = ax.scatter(result["contentvec_pca"][:, 0], result["contentvec_pca"][:, 1], c=result["contentvec_times"], cmap="plasma", s=16, alpha=0.85)
        ax.set_title("ContentVec PCA trajectory")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="time (s)")
    else:
        ax.text(0.05, 0.5, "ContentVec PCA unavailable", transform=ax.transAxes, va="center")
        ax.set_axis_off()

    ax = fig.add_subplot(gs[8, 1])
    ax.plot(times, result["contentvec_boundary_16ms"], color="#b279a2", linewidth=1.2, label="ContentVec cosine boundary")
    ax.plot(times, result["flux_norm"], color="#ff7f0e", alpha=0.55, linewidth=1.0, label="spectral flux")
    ax.set_xlim(0, times[-1] if len(times) else 0)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    ax.set_title("Content boundary vs acoustic flux")
    ax.set_xlabel("time (s)")

    ax = fig.add_subplot(gs[9, :])
    art_plot = np.where(result["articulation_id"] >= 0, result["articulation_id"], np.nan)
    ax.step(times, art_plot, where="post", color="#111111", linewidth=1.5, label="latched articulation_id")
    ax.plot(times, result["articulation_confidence"], color="#4c78a8", linewidth=1.0, label="confidence")
    ax.step(times, gate, where="post", color="#2ca02c", linewidth=1.0, alpha=0.55, label="gate")
    for t in note_onset_times:
        ax.axvline(t, color="#2ca02c", alpha=0.55, linewidth=0.8)
    ax.set_yticks(np.arange(len(feature_labels)))
    ax.set_yticklabels(feature_labels)
    ax.set_ylim(-0.5, len(feature_labels) - 0.5)
    ax.set_xlim(0, times[-1] if len(times) else 0)
    ax.set_title("articulation_id(t): causal classify-near-onset, then latch")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[10, :])
    compact_rows = [
        robust_normalize(result["mfcc"][0]),
        result["flux_norm"],
        result["hpss_onset_norm"],
        result["high_freq_ratio_norm"],
        result["contentvec_boundary_16ms"],
        result["boundary_evidence"],
        result["onset_strength"],
        result["periodicity"],
        robust_normalize(result["loudness_z"]),
        result["activity_evidence"],
        result["silero_gate"],
        gate,
        result["note_onset"],
        offset,
        np.clip(result["note_age"] / max(np.max(result["note_age"]), 1e-6), 0.0, 1.0),
        np.where(result["articulation_id"] >= 0, result["articulation_id"] / max(len(feature_labels) - 1, 1), 0.0),
    ]
    compact_labels = [
        "MFCC0",
        "Flux",
        "HPSS",
        "HF ratio",
        "ContentVec boundary",
        "Boundary evidence",
        "Onset strength",
        "Periodicity",
        "Loudness z",
        "Activity evidence",
        "Silero gate",
        "Final gate",
        "Note onset",
        "Offset",
        "Note age",
        "Articulation ID",
    ]
    compact = np.vstack(compact_rows)
    im = ax.imshow(compact, aspect="auto", interpolation="nearest", cmap="viridis", extent=[times[0], times[-1], compact.shape[0], 0])
    ax.set_yticks(np.arange(0.5, len(compact_labels) + 0.5))
    ax.set_yticklabels(compact_labels)
    ax.set_xlabel("time (s)")
    ax.set_title("Compact 16 ms feature/control laboratory matrix")
    fig.colorbar(im, ax=ax, label="normalized")

    plt.show()
    return fig


def export_voice_control_features(result, out_prefix="scat_features_16ms"):
    out_prefix = str(out_prefix)
    out_npz = f"{out_prefix}.npz"
    out_csv = f"{out_prefix}.csv"
    df = pd.DataFrame(result["feature_tensor_plus"], columns=result["feature_names_plus"])
    df.insert(0, "time_seconds", result["times"][:len(df)])
    df["articulation_name"] = result["articulation_name"]
    df["flux_norm"] = result["flux_norm"][:len(df)]
    df["zcr_norm"] = result["zcr_norm"][:len(df)]
    df["energy_norm"] = result["energy_norm"][:len(df)]

    flux_peak_frames, _ = find_peaks(
        result["flux_norm"],
        height=0.35,
        prominence=0.10,
        distance=max(1, int(round(0.080 / result["config"].hop_seconds))),
    )
    df["is_flux_peak"] = False
    df.loc[flux_peak_frames[flux_peak_frames < len(df)], "is_flux_peak"] = True
    df["is_silero_onset"] = result["silero_onset"].astype(bool)
    df["is_note_onset"] = result["note_onset"].astype(bool)
    df["is_offset"] = result["offset"].astype(bool)

    contentvec_features = (
        np.asarray(result["contentvec_features"], dtype=np.float32)
        if result["contentvec_features"] is not None
        else np.zeros((0, 0), dtype=np.float32)
    )
    contentvec_times = (
        np.asarray(result["contentvec_times"], dtype=np.float32)
        if result["contentvec_times"] is not None
        else np.zeros(0, dtype=np.float32)
    )

    np.savez(
        out_npz,
        y=result["y"],
        sr=result["sr"],
        source_name=np.asarray(result["source_name"]),
        times=result["times"],
        hop_seconds=np.array(result["config"].hop_seconds, dtype=np.float32),
        articulation_labels=np.asarray(result["articulation_labels"], dtype=object),
        feature_tensor=result["feature_tensor"],
        feature_names=np.asarray(result["feature_names"], dtype=object),
        control_tensor=result["control_tensor"],
        control_names=np.asarray(result["control_names"], dtype=object),
        feature_tensor_plus=result["feature_tensor_plus"],
        feature_names_plus=np.asarray(result["feature_names_plus"], dtype=object),
        f0_hz=result["f0_hz"],
        loudness_z=result["loudness_z"],
        gate=result["gate"],
        silero_gate=result["silero_gate"],
        note_onset=result["note_onset"],
        offset=result["offset"],
        note_age=result["note_age"],
        periodicity=result["periodicity"],
        onset_strength=result["onset_strength"],
        activity_evidence=result["activity_evidence"],
        energy_activity=result["energy_activity"],
        voiced_activity=result["voiced_activity"],
        unvoiced_activity=result["unvoiced_activity"],
        percussive_activity=result["percussive_activity"],
        boundary_evidence=result["boundary_evidence"],
        articulation_id=result["articulation_id"],
        articulation_name=result["articulation_name"],
        articulation_confidence=result["articulation_confidence"],
        articulation_probs=result["articulation_probs"],
        causal_vad_times=result["causal_vad_times"],
        causal_vad_probs=result["causal_vad_probs"],
        vad_prob_16ms=result["vad_prob_16ms"],
        contentvec_features=contentvec_features,
        contentvec_times=contentvec_times,
        contentvec_pca=result["contentvec_pca"],
        contentvec_pca_16ms=result["contentvec_pca_16ms"],
        contentvec_boundary=result["contentvec_boundary"],
        contentvec_boundary_16ms=result["contentvec_boundary_16ms"],
        contentvec_novelty=result["contentvec_novelty"],
        contentvec_novelty_16ms=result["contentvec_novelty_16ms"],
        contentvec_status=np.asarray(result["contentvec_status"]),
        causal_vad_status=np.asarray(result["causal_vad_status"]),
        torchcrepe_status=np.asarray(result["torchcrepe_status"]),
    )
    df.to_csv(out_csv, index=False)
    return out_npz, out_csv, df
