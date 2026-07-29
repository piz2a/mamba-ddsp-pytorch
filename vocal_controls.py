from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import importlib.machinery
import os
import subprocess
import sys
import types

import aubio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
ARTICULATION_LABELS = ("FS_NO", "MU_NO", "PK_NO", "SP_NO", "ST_NO", "FS_DN")
UNKNOWN_ARTICULATION_ID = -1


@dataclass
class VocalControlConfig:
    workspace_root: Path = Path("/workspace")
    input_dir: Path | None = None
    sample_rate: int = 16000
    hop_seconds: float = 0.016
    frame_seconds: float = 0.032
    n_mfcc: int = 20
    high_freq_cutoff_hz: float = 3000.0
    low_mid_min_hz: float = 250.0
    low_mid_max_hz: float = 3000.0
    noise_floor_percentile: float = 10.0
    energy_margin_db: float = 6.0
    energy_softness_db: float = 4.0
    noise_profile_seconds: float = 0.5
    noise_gate_margin_db: float = 10.0
    noise_gate_release_seconds: float = 0.080
    torchcrepe_model: str = "tiny"
    torchcrepe_device: str = "cpu"
    torchcrepe_fmin: float = 50.0
    torchcrepe_fmax: float = 1000.0
    onset_min_distance_seconds: float = 0.200
    onset_internal_height: float = 0.85
    onset_internal_prominence: float = 0.15
    onset_classify_seconds: float = 0.128
    onset_retrigger_hold_seconds: float = 0.080
    offset_hold_seconds: float = 0.080
    offset_periodicity_threshold: float = 0.35
    aubio_threshold: float = 0.30
    aubio_silence_db: float = -45.0
    aubio_minioi_seconds: float = 0.080
    articulation_labels: tuple[str, ...] = field(default_factory=lambda: ARTICULATION_LABELS)

    def __post_init__(self):
        self.workspace_root = Path(self.workspace_root).expanduser()
        self.input_dir = Path(self.input_dir or self.workspace_root / "learn" / "voice_inputs").expanduser()
        self.input_dir.mkdir(parents=True, exist_ok=True)

    @property
    def hop_length(self) -> int:
        return int(round(self.hop_seconds * self.sample_rate))

    @property
    def win_length(self) -> int:
        return int(round(self.frame_seconds * self.sample_rate))

    @property
    def n_fft(self) -> int:
        value = 1
        while value < self.win_length:
            value *= 2
        return value


def ensure_notebook_dependencies(config=None):
    packages = [("librosa", "librosa"), ("soundfile", "soundfile"), ("aubio", "aubio"),
                ("numpy", "numpy"), ("matplotlib", "matplotlib"), ("scipy", "scipy"),
                ("pandas", "pandas"), ("ipywidgets", "ipywidgets"), ("imageio-ffmpeg", "imageio_ffmpeg")]
    for package, import_name in packages:
        try:
            __import__(import_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def list_audio_inputs(input_dir, extensions=SUPPORTED_AUDIO_EXTENSIONS):
    input_dir = Path(input_dir)
    files = [p for ext in extensions for p in input_dir.rglob(f"*{ext}")]
    files.extend(p for ext in extensions for p in input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)


def _decode_audio_with_ffmpeg(audio_path, target_sr):
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(audio_path),
               "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(target_sr), "-"]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
    return np.frombuffer(process.stdout, dtype=np.float32).copy(), target_sr


def load_audio_file(audio_path, target_sr=16000):
    audio_path = Path(audio_path).expanduser()
    try:
        if audio_path.suffix.lower() not in {".m4a", ".aac", ".mp3", ".ogg"}:
            y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return y.astype(np.float32), sr, "librosa/soundfile"
    except Exception:
        pass
    y, sr = _decode_audio_with_ffmpeg(audio_path, target_sr)
    return y.astype(np.float32), sr, "imageio-ffmpeg"


def load_selected_audio(config, audio_path=None):
    chosen = Path(audio_path).expanduser() if audio_path else (list_audio_inputs(config.input_dir)[0] if list_audio_inputs(config.input_dir) else None)
    if chosen is None:
        raise FileNotFoundError("No recording found. Put a WAV/FLAC/M4A file in learn/voice_inputs.")
    y, sr, decoder = load_audio_file(chosen, config.sample_rate)
    y, _ = librosa.effects.trim(y, top_db=45)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-7:
        y = y / peak * 0.95
    return {"y": y.astype(np.float32), "sr": sr, "source_name": str(chosen),
            "source_reason": f"audio file; decoder={decoder}",
            "available_inputs": list_audio_inputs(config.input_dir)}


def display_upload_widget(input_dir, extensions=SUPPORTED_AUDIO_EXTENSIONS):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:
        print("ipywidgets upload UI unavailable:", exc)
        return None
    input_dir = Path(input_dir)
    uploader = widgets.FileUpload(accept=",".join(extensions), multiple=False, description="Choose audio")
    button = widgets.Button(description="Save uploaded audio", button_style="primary")
    output = widgets.Output()
    def save(_):
        with output:
            output.clear_output()
            values = uploader.value.values() if isinstance(uploader.value, dict) else uploader.value
            items = list(values or [])
            if not items:
                print("No file selected.")
                return
            item = items[0]
            content = item["content"] if isinstance(item, dict) else item.content
            name = item.get("name", "uploaded_audio.wav") if isinstance(item, dict) else item.name
            out = input_dir / Path(name).name
            out.write_bytes(bytes(content))
            print("Saved:", out)
    button.on_click(save)
    box = widgets.VBox([uploader, button, output])
    display(box)
    return box


def align_length(x, target_len, fill=0.0):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) >= target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - len(x)), constant_values=fill).astype(np.float32)


def robust_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if not x.size:
        return x
    lo, hi = np.percentile(x, [5, 95])
    return np.clip((x - lo) / max(float(hi - lo), 1e-7), 0.0, 1.0).astype(np.float32)


def peak_normalize(x):
    """Match Bass-DDSP HPSS control scaling: divide by the framewise peak."""
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(x)) if x.size else 0.0
    if peak <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / peak, 0.0, 1.0).astype(np.float32)


def masked_zscore(x, mask=None):
    x = np.asarray(x, dtype=np.float32)
    ref = x[np.asarray(mask).astype(bool)] if mask is not None and np.any(mask) else x
    return ((x - float(np.mean(ref))) / max(float(np.std(ref)), 1e-6)).astype(np.float32)


def causal_hysteresis(values, open_threshold, close_threshold):
    active = False
    output = np.zeros(len(values), dtype=np.float32)
    for i, value in enumerate(np.asarray(values).reshape(-1)):
        active = value >= open_threshold if not active else value > close_threshold
        output[i] = float(active)
    return output


def apply_retrigger_hold(candidates, hold_seconds, hop_seconds):
    """Accept the first onset, then suppress retriggers during the causal hold."""
    candidates = np.asarray(candidates, dtype=np.float32).reshape(-1)
    accepted = np.zeros_like(candidates)
    hold_frames = max(1, int(np.ceil(float(hold_seconds) / float(hop_seconds))))
    last_onset = -hold_frames
    for index, candidate in enumerate(candidates):
        if candidate > 0.5 and index - last_onset >= hold_frames:
            accepted[index] = 1.0
            last_onset = index
    return accepted


def apply_gate_release_hold(gate, hold_seconds, hop_seconds):
    """Keep a causal gate open briefly after its most recent active frame."""
    gate = np.asarray(gate, dtype=np.float32).reshape(-1)
    held = np.zeros_like(gate)
    hold_frames = max(1, int(np.ceil(float(hold_seconds) / float(hop_seconds))))
    last_open = -hold_frames
    for index, is_open in enumerate(gate > 0.5):
        if is_open:
            last_open = index
        held[index] = float(index - last_open < hold_frames)
    return held


def resolve_monophonic_events(
    onset_candidates,
    periodicity,
    noise_gate,
    threshold=0.35,
    offset_hold_seconds=0.080,
    hop_seconds=0.016,
):
    """Resolve onset/offset candidates with one causal monophonic note state."""
    onset_candidates = np.asarray(onset_candidates, dtype=np.float32).reshape(-1)
    periodicity = np.asarray(periodicity, dtype=np.float32).reshape(-1)
    noise_gate = np.asarray(noise_gate, dtype=np.float32).reshape(-1)
    onset = np.zeros_like(onset_candidates)
    offset = np.zeros_like(onset_candidates)
    active_track = np.zeros_like(onset_candidates)
    active = False
    last_onset = -1
    offset_hold_frames = max(1, int(np.ceil(float(offset_hold_seconds) / float(hop_seconds))))
    for index in range(len(onset_candidates)):
        has_onset = onset_candidates[index] > 0.5
        if active:
            if noise_gate[index] <= 0.5:
                # Gate closure is a hard stop and bypasses the post-onset hold.
                offset[index] = 1.0
                active = False
            elif has_onset:
                # A monophonic retrigger terminates the old note and starts the
                # new note on the same control frame.
                offset[index] = 1.0
                onset[index] = 1.0
                last_onset = index
            elif (index - last_onset >= offset_hold_frames
                  and periodicity[index] < float(threshold)):
                offset[index] = 1.0
                active = False
        elif has_onset and noise_gate[index] > 0.5:
            onset[index] = 1.0
            active = True
            last_onset = index
        active_track[index] = float(active)
    return onset, offset, active_track


def calibrated_noise_gate(rms_db, times, profile_seconds=0.5, margin_db=10.0):
    """Calibrate a Boolean gate from the peak RMS in the initial noise profile."""
    rms_db = np.asarray(rms_db, dtype=np.float32).reshape(-1)
    times = np.asarray(times, dtype=np.float32).reshape(-1)
    profile = rms_db[times < float(profile_seconds)]
    if profile.size == 0:
        profile = rms_db[:1]
    noise_peak_db = float(np.max(profile)) if profile.size else -120.0
    threshold_db = noise_peak_db + float(margin_db)
    return (rms_db >= threshold_db).astype(np.float32), noise_peak_db, threshold_db


def causal_note_age(gate, onset, frame_seconds):
    age = np.zeros(len(gate), dtype=np.float32)
    current = 0.0
    for i, active in enumerate(np.asarray(gate) > 0.5):
        if onset[i] > 0.5:
            current = 0.0
        age[i] = current if active else 0.0
        current = current + frame_seconds if active else 0.0
    return age


def compute_frame_features(y, config):
    stft = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length, center=True)
    mag = np.abs(stft).astype(np.float32)
    power = mag ** 2
    freqs = librosa.fft_frequencies(sr=config.sample_rate, n_fft=config.n_fft)
    times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=config.sample_rate, hop_length=config.hop_length).astype(np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=config.sample_rate, n_mfcc=config.n_mfcc, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length)
    delta = librosa.feature.delta(mfcc)
    zcr = align_length(librosa.feature.zero_crossing_rate(y, frame_length=config.win_length, hop_length=config.hop_length, center=True)[0], len(times))
    normalized_mag = mag / np.maximum(np.sum(mag, axis=0, keepdims=True), 1e-8)
    difference = np.diff(normalized_mag, axis=1)
    flux = np.concatenate([[0.0], np.sqrt(np.sum(np.maximum(difference, 0.0) ** 2, axis=0))]).astype(np.float32)
    rms = align_length(librosa.feature.rms(S=mag, frame_length=config.n_fft, hop_length=config.hop_length, center=True)[0], len(times))
    rms_db = (20.0 * np.log10(np.maximum(rms, 1e-7))).astype(np.float32)
    try:
        _, percussive = librosa.decompose.hpss(stft, margin=8.0)
        hpss = align_length(np.mean(np.abs(percussive), axis=0), len(times))
    except Exception:
        hpss = np.zeros(len(times), dtype=np.float32)
    high = freqs >= config.high_freq_cutoff_hz
    low_mid = (freqs >= config.low_mid_min_hz) & (freqs < config.low_mid_max_hz)
    total = np.maximum(np.sum(power, axis=0), 1e-10)
    high_energy = np.sum(power[high], axis=0) if np.any(high) else np.zeros_like(total)
    low_mid_energy = np.sum(power[low_mid], axis=0) if np.any(low_mid) else np.zeros_like(total)
    hf_ratio = np.clip(high_energy / total, 0.0, 1.0).astype(np.float32)
    tilt = (10.0 * np.log10((high_energy + 1e-10) / (low_mid_energy + 1e-10))).astype(np.float32)
    return {"stft": stft, "mag": mag, "freqs": freqs, "times": times,
            "mfcc": mfcc.astype(np.float32), "mfcc_delta": delta.astype(np.float32),
            "zcr": zcr, "flux": flux, "flux_norm": robust_normalize(flux),
            "frame_rms": rms, "rms_db": rms_db, "hpss_percussive": hpss.astype(np.float32),
            "hpss_onset_norm": peak_normalize(hpss), "high_freq_ratio": hf_ratio,
            "high_freq_ratio_norm": robust_normalize(hf_ratio), "spectral_tilt": tilt,
            "spectral_tilt_norm": robust_normalize(tilt),
            "energy_norm": robust_normalize(rms_db)}


def import_torchcrepe_safe():
    try:
        for name in list(sys.modules):
            if name == "torchaudio" or name.startswith("torchaudio."):
                del sys.modules[name]
        stub = types.ModuleType("torchaudio")
        stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
        stub.load = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("file loading is not used"))
        sys.modules["torchaudio"] = stub
        import torchcrepe
        return torchcrepe, "imported torchcrepe with torchaudio stub"
    except Exception as exc:
        return None, f"torchcrepe unavailable: {type(exc).__name__}: {exc}"


def extract_torchcrepe_controls(y, frame_count, config):
    torchcrepe, status = import_torchcrepe_safe()
    if torchcrepe is None:
        raise RuntimeError(status)
    device = config.torchcrepe_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    wav = torch.from_numpy(np.asarray(y, dtype=np.float32)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        f0, periodicity = torchcrepe.predict(wav, config.sample_rate, hop_length=config.hop_length,
            fmin=config.torchcrepe_fmin, fmax=config.torchcrepe_fmax, model=config.torchcrepe_model,
            decoder=torchcrepe.decode.weighted_argmax, return_periodicity=True, device=device, pad=True)
    return (align_length(f0.squeeze(0).cpu().numpy(), frame_count),
            np.clip(align_length(periodicity.squeeze(0).cpu().numpy(), frame_count), 0.0, 1.0),
            f"{status}; model={config.torchcrepe_model}; device={device}")


def extract_aubio_onsets(y, config):
    """Return continuous novelty and binary decisions for Complex and HFC."""
    win_size, hop_size = config.n_fft, config.hop_length
    frame_count = int(np.ceil(len(y) / hop_size))
    padded = np.pad(np.asarray(y, dtype=np.float32), (0, frame_count * hop_size - len(y)))
    names = ("complex", "hfc")
    pvocs = {name: aubio.pvoc(win_size, hop_size) for name in names}
    descriptions = {name: aubio.specdesc(name, win_size) for name in names}
    detectors = {name: aubio.onset(name, win_size, hop_size, config.sample_rate) for name in names}
    for detector in detectors.values():
        detector.set_threshold(config.aubio_threshold)
        detector.set_silence(config.aubio_silence_db)
        detector.set_minioi_s(config.aubio_minioi_seconds)
    novelty = {name: np.zeros(frame_count, dtype=np.float32) for name in names}
    decision_events = {name: np.zeros(frame_count, dtype=np.float32) for name in names}
    reported_events = {name: np.zeros(frame_count, dtype=np.float32) for name in names}
    decision_times = {name: [] for name in names}
    event_times = {name: [] for name in names}
    for index in range(frame_count):
        chunk = aubio.fvec(padded[index * hop_size:(index + 1) * hop_size])
        for name in names:
            spectrum = pvocs[name](chunk)
            novelty[name][index] = float(descriptions[name](spectrum)[0])
            decision = detectors[name](chunk)
            # Aubio documents this output as 0 for no onset and 1 + a for an
            # accepted onset, where a is the sub-hop timing offset. It is not
            # a probability or onset strength.
            if bool(decision[0]):
                decision_events[name][index] = 1.0
                # The decision is only causally available after this complete
                # hop has been consumed. The old start-of-hop timestamp
                # understated runtime availability by one hop.
                decision_times[name].append((index + 1) * config.hop_seconds)
                reported_time = float(detectors[name].get_last_s())
                event_times[name].append(reported_time)
                reported_index = int(round(reported_time / config.hop_seconds))
                reported_index = min(max(reported_index, 0), frame_count - 1)
                reported_events[name][reported_index] = 1.0
    return {"times": np.arange(frame_count, dtype=np.float32) * config.hop_seconds,
            "complex_novelty": robust_normalize(novelty["complex"]),
            "hfc_novelty": robust_normalize(novelty["hfc"]),
            "complex_onset": decision_events["complex"],
            "hfc_onset": decision_events["hfc"],
            "complex_reported_onset": reported_events["complex"],
            "hfc_reported_onset": reported_events["hfc"],
            "complex_decision_times": np.asarray(decision_times["complex"], dtype=np.float32),
            "hfc_decision_times": np.asarray(decision_times["hfc"], dtype=np.float32),
            "complex_event_times": np.asarray(event_times["complex"], dtype=np.float32),
            "hfc_event_times": np.asarray(event_times["hfc"], dtype=np.float32)}


def _articulation_id(features, periodicity, onset, config):
    labels = config.articulation_labels
    ids = np.full(len(onset), UNKNOWN_ARTICULATION_ID, dtype=np.int64)
    confidence = np.zeros(len(onset), dtype=np.float32)
    current = UNKNOWN_ARTICULATION_ID
    for i in range(len(onset)):
        if onset[i] > 0.5:
            scores = np.asarray([
                periodicity[i], 1.0 - periodicity[i], features["hpss_onset_norm"][i],
                features["high_freq_ratio_norm"][i], features["spectral_tilt_norm"][i],
                features["flux_norm"][i]], dtype=np.float32)
            current = int(np.argmax(scores) % len(labels))
            confidence[i] = float(np.max(scores))
        elif current >= 0:
            confidence[i] = confidence[i - 1] if i else 0.0
        ids[i] = current
    names = np.asarray([labels[i] if i >= 0 else "NONE" for i in ids], dtype=object)
    return ids, names, confidence


def extract_voice_controls(audio, config=None, source_name=None):
    config = config or VocalControlConfig()
    if isinstance(audio, dict):
        y = np.asarray(audio["y"], dtype=np.float32).reshape(-1)
        source_name = source_name or audio.get("source_name", "audio")
    elif isinstance(audio, (str, Path)):
        loaded = load_selected_audio(config, audio)
        y, source_name = loaded["y"], source_name or loaded["source_name"]
    else:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        source_name = source_name or "audio"
    features = compute_frame_features(y, config)
    frame_count = len(features["times"])
    f0, periodicity, crepe_status = extract_torchcrepe_controls(y, frame_count, config)
    aubio_onsets = extract_aubio_onsets(y, config)
    aubio_complex = np.interp(features["times"], aubio_onsets["times"], aubio_onsets["complex_novelty"], left=0.0, right=0.0).astype(np.float32)
    aubio_hfc = np.interp(features["times"], aubio_onsets["times"], aubio_onsets["hfc_novelty"], left=0.0, right=0.0).astype(np.float32)
    aubio_complex_event = align_length(aubio_onsets["complex_onset"], frame_count)
    aubio_hfc_event = align_length(aubio_onsets["hfc_onset"], frame_count)
    raw_noise_gate, noise_peak_db, noise_gate_threshold_db = calibrated_noise_gate(
        features["rms_db"],
        features["times"],
        config.noise_profile_seconds,
        config.noise_gate_margin_db,
    )
    noise_gate = apply_gate_release_hold(
        raw_noise_gate,
        config.noise_gate_release_seconds,
        config.hop_seconds,
    )
    energy_activity = noise_gate.copy()
    # The noise gate is only a veto for aubio Complex events. A rising gate
    # edge must never create an onset by itself.
    onset_candidates = (aubio_complex_event * noise_gate).astype(np.float32)
    held_onset_candidates = apply_retrigger_hold(
        onset_candidates,
        config.onset_retrigger_hold_seconds,
        config.hop_seconds,
    )
    note_onset, offset, gate = resolve_monophonic_events(
        held_onset_candidates,
        periodicity,
        noise_gate,
        config.offset_periodicity_threshold,
        config.offset_hold_seconds,
        config.hop_seconds,
    )
    note_age = causal_note_age(gate, note_onset, config.hop_seconds)
    # Preserve the Bass-DDSP training semantics: onset_strength is continuous
    # HPSS percussive energy. Aubio events remain separate diagnostics and do
    # not replace or spike this control.
    onset_strength = features["hpss_onset_norm"].astype(np.float32)
    articulation_id, articulation_name, articulation_confidence = _articulation_id(features, periodicity, note_onset, config)
    control_tensor = np.stack([f0, masked_zscore(features["rms_db"], gate > 0.5), periodicity,
                               onset_strength, note_onset, offset, gate, note_age,
                               aubio_complex, aubio_hfc, articulation_id.astype(np.float32)], axis=1).astype(np.float32)
    control_names = ["f0_hz_torchcrepe", "loudness_rms_z", "periodicity_torchcrepe", "onset_strength_hpss_percussive",
                     "note_onset", "offset_monophonic_state", "gate_monophonic_state", "note_age_seconds", "aubio_complex_novelty",
                     "aubio_hfc_novelty", "articulation_id"]
    feature_tensor = np.concatenate([features["mfcc"].T, features["mfcc_delta"].T,
        features["zcr"][:, None], features["flux"][:, None], features["rms_db"][:, None],
        features["high_freq_ratio"][:, None], features["hpss_percussive"][:, None]], axis=1).astype(np.float32)
    feature_names = ([f"mfcc_{i:02d}" for i in range(config.n_mfcc)] + [f"delta_mfcc_{i:02d}" for i in range(config.n_mfcc)] +
                     ["zcr", "spectral_flux", "rms_db", "high_freq_ratio", "hpss_percussive"])
    return {**features, "y": y, "sr": config.sample_rate, "source_name": source_name, "config": config,
            "f0_hz": f0, "periodicity": periodicity, "torchcrepe_status": crepe_status,
            "aubio_onsets": aubio_onsets, "aubio_complex_novelty": aubio_complex, "aubio_hfc_novelty": aubio_hfc,
            "aubio_complex_onset": aubio_complex_event, "aubio_hfc_onset": aubio_hfc_event,
            "masked_aubio_complex_onset": onset_candidates,
            "energy_activity": energy_activity, "activity_evidence": energy_activity,
            "raw_noise_gate": raw_noise_gate, "noise_gate": noise_gate, "noise_peak_db": noise_peak_db,
            "noise_gate_threshold_db": noise_gate_threshold_db, "activity_gate": noise_gate,
            "gate": gate,
            "note_onset": note_onset, "offset": offset, "note_age": note_age, "onset_strength": onset_strength,
            "loudness_z": masked_zscore(features["rms_db"], gate > 0.5),
            "articulation_id": articulation_id, "articulation_name": articulation_name,
            "articulation_confidence": articulation_confidence, "articulation_labels": config.articulation_labels,
            "control_tensor": control_tensor, "control_names": control_names,
            "feature_tensor": feature_tensor, "feature_names": feature_names,
            "feature_tensor_plus": np.concatenate([feature_tensor, control_tensor], axis=1),
            "feature_names_plus": feature_names + control_names}


def print_summary(result):
    print("source:", result["source_name"])
    print("duration:", len(result["y"]) / result["sr"], "seconds")
    print("control frame interval:", result["config"].hop_seconds, "seconds")
    print("frames:", len(result["times"]))
    print("TorchCREPE:", result["torchcrepe_status"])
    print("aubio settings:", result["config"].aubio_threshold, "threshold;", result["config"].aubio_silence_db, "dB silence;", result["config"].aubio_minioi_seconds, "s min IOI")
    print("aubio Complex event times:", result["aubio_onsets"]["complex_event_times"].tolist())
    print("aubio HFC event times:", result["aubio_onsets"]["hfc_event_times"].tolist())
    print("noise peak / gate threshold dB:", result["noise_peak_db"], result["noise_gate_threshold_db"])
    print("offset events:", int(np.sum(result["offset"] > 0.5)), "threshold:", result["config"].offset_periodicity_threshold)
    print("onset strength range:", float(np.min(result["onset_strength"])), float(np.max(result["onset_strength"])))
    print("periodicity range:", float(np.min(result["periodicity"])), float(np.max(result["periodicity"])))
    print("control tensor:", result["control_tensor"].shape)


def note_intervals_from_events(times, onset, offset, audio_end_time):
    """Pair monophonic onset/offset events into display intervals."""
    times = np.asarray(times, dtype=np.float32).reshape(-1)
    onset = np.asarray(onset, dtype=np.float32).reshape(-1)
    offset = np.asarray(offset, dtype=np.float32).reshape(-1)
    intervals = []
    active_start = None
    for frame_time, has_onset, has_offset in zip(times, onset > 0.5, offset > 0.5):
        if has_offset and active_start is not None:
            intervals.append((active_start, float(frame_time)))
            active_start = None
        if has_onset:
            active_start = float(frame_time)
    if active_start is not None:
        intervals.append((active_start, float(audio_end_time)))
    return intervals


def plot_voice_control_dashboard(result):
    y, sr, times, config = result["y"], result["sr"], result["times"], result["config"]
    audio_time = np.arange(len(y)) / sr
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=config.n_fft,
        hop_length=config.hop_length, win_length=config.win_length, n_mels=96), ref=np.max)
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=False, constrained_layout=True)
    librosa.display.specshow(mel_db, sr=sr, hop_length=config.hop_length, x_axis="time", y_axis="mel", cmap="magma", ax=axes[0])
    axes[0].set_title("Log-mel spectrogram")
    axes[1].plot(audio_time, y, color="black", linewidth=0.6)
    note_onset = np.asarray(result.get("note_onset", np.zeros_like(times)))
    offset = np.asarray(result.get("offset", np.zeros_like(times)))
    note_onset_times = times[note_onset > 0.5]
    offset_times = times[offset > 0.5]
    note_intervals = note_intervals_from_events(
        times, note_onset, offset, len(y) / sr,
    )
    for index, (start_time, end_time) in enumerate(note_intervals):
        axes[1].axvspan(
            start_time, end_time, color="black", alpha=0.08, linewidth=0,
            label="detected note" if index == 0 else None,
        )
    for index, event_time in enumerate(note_onset_times):
        axes[1].axvline(event_time, color="#2ca02c", linestyle="--", alpha=0.55,
                        linewidth=1.0, label="accepted onset" if index == 0 else None)
    for index, event_time in enumerate(offset_times):
        axes[1].axvline(event_time, color="#d62728", linestyle=":", alpha=0.65,
                        linewidth=1.1, label="offset" if index == 0 else None)
    axes[1].set_title("Waveform with accepted onset and offset events")
    if len(note_onset_times) or len(offset_times):
        axes[1].legend(loc="upper right")
    axes[2].plot(times, result["aubio_complex_novelty"], label="aubio Complex novelty", color="#9467bd")
    axes[2].plot(times, result["aubio_hfc_novelty"], label="aubio HFC novelty", color="#22dd18")
    complex_times = result["aubio_onsets"]["complex_event_times"]
    hfc_times = result["aubio_onsets"]["hfc_event_times"]
    axes[2].scatter(complex_times, np.ones(len(complex_times)), color="#9467bd", marker="|", s=180, label="Complex event (aubio timestamp)")
    axes[2].scatter(hfc_times, np.ones(len(hfc_times)) * 0.92, color="#8c564b", marker="|", s=180, label="HFC event (aubio timestamp)")
    axes[2].scatter(note_onset_times, np.ones(len(note_onset_times)) * 0.84, color="#2ca02c",
                    marker="^", s=40, label="accepted onset after 80 ms hold")
    axes[2].scatter(offset_times, np.ones(len(offset_times)) * 0.76, color="#d62728",
                    marker="v", s=40, label="periodicity/retrigger offset")
    axes[2].set_ylim(-0.02, 1.08);
    axes[2].set_title("Onset candidates, accepted onsets, and deterministic offsets")
    axes[2].legend(loc="lower right")
    axes[3].plot(times, result["f0_hz"], color="#d62728", label="TorchCREPE F0 (Hz)")
    twin = axes[3].twinx()
    twin.plot(times, result["periodicity"], color="#1f77b4", label="TorchCREPE periodicity")
    twin.set_ylim(-0.02, 1.02)
    axes[3].set_title("F0 and periodicity")
    axes[3].legend(loc="upper left"); twin.legend(loc="upper right")
    axes[4].plot(times, result["loudness_z"], color="#2ca02c", label="RMS loudness z-score")
    gate_axis = axes[4].twinx()
    gate_axis.step(times, result.get("noise_gate", np.zeros_like(times)), where="post",
                   color="#d62728", alpha=0.75,
                   label=f"noise gate ({config.noise_gate_release_seconds * 1000:.0f} ms release)")
    gate_axis.set_ylim(-0.05, 1.05)
    axes[4].set_title("Loudness and calibrated Boolean noise gate")
    axes[4].legend(loc="upper right")
    gate_axis.legend(loc="lower right")
    axes[5].plot(times, result["flux_norm"], label="spectral flux", color="#ff7f0e")
    axes[5].plot(times, result["high_freq_ratio_norm"], label="HF ratio", color="#d62728")
    axes[5].plot(times, result["hpss_onset_norm"], label="HPSS percussive", color="#17becf")
    axes[5].set_ylim(-0.02, 1.02); axes[5].set_title("Rejected Onset-strength candidates")
    axes[5].legend(loc="upper right")
    
    for ax in axes:
        ax.set_xlim(0, audio_time[-1] if len(audio_time) else 0)
        ax.grid(alpha=0.18)
    plt.show()
    return fig


def export_voice_control_features(result, out_prefix="scat_features_16ms"):
    out_prefix = str(out_prefix)
    df = pd.DataFrame(result["feature_tensor_plus"], columns=result["feature_names_plus"])
    df.insert(0, "time_seconds", result["times"][:len(df)])
    df["articulation_name"] = result["articulation_name"]
    df["aubio_complex_onset"] = result["aubio_complex_onset"][:len(df)].astype(bool)
    df["aubio_hfc_onset"] = result["aubio_hfc_onset"][:len(df)].astype(bool)
    np.savez(out_prefix + ".npz", y=result["y"], sr=result["sr"], times=result["times"],
             feature_tensor=result["feature_tensor"], feature_names=np.asarray(result["feature_names"], dtype=object),
             control_tensor=result["control_tensor"], control_names=np.asarray(result["control_names"], dtype=object),
             aubio_complex_novelty=result["aubio_complex_novelty"], aubio_hfc_novelty=result["aubio_hfc_novelty"],
             aubio_complex_onset=result["aubio_complex_onset"], aubio_hfc_onset=result["aubio_hfc_onset"],
             periodicity=result["periodicity"], f0_hz=result["f0_hz"], onset_strength=result["onset_strength"])
    df.to_csv(out_prefix + ".csv", index=False)
    return out_prefix + ".npz", out_prefix + ".csv", df
