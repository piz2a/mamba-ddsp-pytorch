from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import subprocess
import time
import types

import numpy as np
import soundfile as sf
import torch
import yaml

from bass_ddsp.model import BassDDSPV2
from bass_ddsp.train import clean_state_dict
from onset_envelope_net.model import StructuredBassOnsetEnvelopeNet
from vocal_controls import VocalControlConfig, extract_voice_controls, load_audio_file


WORKSPACE = Path("/workspace")
DEFAULT_BASS_RUN = (
    WORKSPACE
    / "runs"
    / "bass_ddsp_v2_branchbalance_riff_20260723_052406_resume_to_50k_20260724_043513"
)
DEFAULT_ONSET_CHECKPOINT = (
    WORKSPACE
    / "runs"
    / "bass_onset_single_note_ablation_v2"
    / "articulation_f0_loudness.pt"
)
DEFAULT_ONSET_TENSORS = (
    WORKSPACE
    / "runs"
    / "bass_onset_single_note_ablation_v2"
    / "single_note_tensors.pt"
)


@dataclass
class DemoConfig:
    bass_run_dir: Path = DEFAULT_BASS_RUN
    onset_checkpoint: Path = DEFAULT_ONSET_CHECKPOINT
    onset_tensors: Path = DEFAULT_ONSET_TENSORS
    output_root: Path = WORKSPACE / "demo" / "outputs"
    device: str = "auto"
    articulation_mode: str = "Slap auto"
    pitch_shift_semitones: float = -12.0
    slap_crossover_midi: int = 40
    pitch_hold_periodicity: float = 0.10
    f0_min_hz: float = 30.0
    f0_max_hz: float = 330.0
    loudness_clip_std: float = 3.0
    random_seed: int = 20260729

    def __post_init__(self):
        self.bass_run_dir = Path(self.bass_run_dir)
        self.onset_checkpoint = Path(self.onset_checkpoint)
        self.onset_tensors = Path(self.onset_tensors)
        self.output_root = Path(self.output_root)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [tuple(int(value.strip()) for value in row.split(","))
                for row in query.stdout.strip().splitlines()]
        return torch.device(f"cuda:{max(rows, key=lambda row: row[1])[0]}")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return torch.device("cuda:0")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _f0_to_onset_control(f0_hz: float) -> float:
    f0_hz = max(float(f0_hz), 40.0)
    return float(np.log2(f0_hz / 40.0) / np.log2(600.0 / 40.0))


def _unique_output_dir(root: Path, source: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = root / f"{source.stem}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


class ScatToBassDemo:
    """Offline demo of the causal controls and cached-DCT Bass-DDSP decoder."""

    def __init__(self, config: DemoConfig | None = None):
        self.config = config or DemoConfig()
        self.device = select_device(self.config.device)
        self.setup_times: dict[str, float] = {}
        self._load_models()

    @property
    def articulation_labels(self) -> list[str]:
        return list(self.bass_config["data"]["articulation_labels"])

    @property
    def articulation_modes(self) -> tuple[str, ...]:
        fixed = tuple(label for label in self.articulation_labels
                      if label not in {"SP_NO", "ST_NO"})
        return ("Slap auto", "SP_NO", "ST_NO", *fixed)

    def _load_models(self) -> None:
        started = time.perf_counter()
        with (self.config.bass_run_dir / "config.yaml").open() as handle:
            self.bass_config = yaml.safe_load(handle)
        model_config = dict(self.bass_config["model"])
        model_config["n_articulation"] = len(self.articulation_labels)
        self.model = BassDDSPV2(**model_config).to(self.device)
        state = clean_state_dict(
            torch.load(
                self.config.bass_run_dir / "state.pth",
                map_location=self.device,
                weights_only=False,
            )
        )
        self.model.load_state_dict(state)
        self.model.eval()
        _synchronize(self.device)
        self.setup_times["load_bass_model_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        with torch.inference_mode():
            cached_bank = self.model.transient_dct_bank_waveforms().detach()
        self.cached_dct_bank = cached_bank

        def return_cached_bank(model):
            return self.cached_dct_bank

        self.model.transient_dct_bank_waveforms = types.MethodType(
            return_cached_bank, self.model
        )
        _synchronize(self.device)
        self.setup_times["cache_dct_bank_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        checkpoint = torch.load(
            self.config.onset_checkpoint,
            map_location=self.device,
            weights_only=False,
        )
        tensor_data = torch.load(
            self.config.onset_tensors,
            map_location="cpu",
            weights_only=False,
        )
        self.onset_labels = list(tensor_data["articulation_labels"])
        self.onset_model = StructuredBassOnsetEnvelopeNet(
            num_articulations=int(checkpoint["num_articulations"]),
            control_dim=len(checkpoint["control_indices"]),
            envelope_frames=int(checkpoint["envelope_frames"]),
        ).to(self.device)
        self.onset_model.load_state_dict(checkpoint["model"])
        self.onset_model.eval()
        self.onset_control_names = list(checkpoint["control_names"])
        _synchronize(self.device)
        self.setup_times["load_onset_model_seconds"] = time.perf_counter() - started

    def _map_pitch(self, controls: dict) -> np.ndarray:
        source_f0 = np.asarray(controls["f0_hz"], dtype=np.float32)
        periodicity = np.asarray(controls["periodicity"], dtype=np.float32)
        gate = np.asarray(controls["gate"], dtype=np.float32)
        factor = 2.0 ** (self.config.pitch_shift_semitones / 12.0)
        shifted = source_f0 * factor
        mapped = np.zeros_like(shifted)
        held = 0.0
        for frame in range(len(mapped)):
            candidate = float(shifted[frame])
            if gate[frame] > 0.5:
                if (
                    np.isfinite(candidate)
                    and candidate > 0.0
                    and periodicity[frame] >= self.config.pitch_hold_periodicity
                ):
                    held = float(
                        np.clip(
                            candidate,
                            self.config.f0_min_hz,
                            self.config.f0_max_hz,
                        )
                    )
                mapped[frame] = held
            else:
                held = 0.0
        return mapped

    def _choose_articulation(self, mode: str, pitch_hz: float) -> str:
        if mode != "Slap auto":
            if mode not in self.articulation_labels:
                raise ValueError(
                    f"Unknown articulation mode {mode!r}; choose from "
                    f"{self.articulation_modes}"
                )
            return mode
        crossover_hz = 440.0 * 2.0 ** (
            (self.config.slap_crossover_midi - 69.0) / 12.0
        )
        return "SP_NO" if pitch_hz >= crossover_hz else "ST_NO"

    def _articulation_track(
        self, controls: dict, mapped_f0: np.ndarray, mode: str
    ) -> tuple[np.ndarray, np.ndarray]:
        onset = np.asarray(controls["note_onset"]) > 0.5
        offset = np.asarray(controls["offset"]) > 0.5
        gate = np.asarray(controls["gate"]) > 0.5
        labels = self.articulation_labels
        ids = np.zeros(len(mapped_f0), dtype=np.int64)
        names = np.full(len(mapped_f0), "NONE", dtype=object)
        active_label: str | None = None
        last_pitch = 0.0
        for frame in range(len(ids)):
            if mapped_f0[frame] > 0.0:
                last_pitch = float(mapped_f0[frame])
            if offset[frame]:
                active_label = None
            if onset[frame]:
                active_label = self._choose_articulation(mode, last_pitch)
            if gate[frame] and active_label is not None:
                ids[frame] = labels.index(active_label)
                names[frame] = active_label
        return ids, names

    def _onset_strength(
        self,
        controls: dict,
        mapped_f0: np.ndarray,
        articulation_names: np.ndarray,
    ) -> np.ndarray:
        onset = np.asarray(controls["note_onset"]) > 0.5
        offset = np.asarray(controls["offset"]) > 0.5
        gate = np.asarray(controls["gate"]) > 0.5
        loudness = np.asarray(controls["loudness_z"], dtype=np.float32)
        output = np.zeros(len(mapped_f0), dtype=np.float32)
        envelope = np.empty(0, dtype=np.float32)
        envelope_index = 0
        for frame in range(len(output)):
            if offset[frame]:
                envelope = np.empty(0, dtype=np.float32)
                envelope_index = 0
            if onset[frame]:
                label = str(articulation_names[frame])
                if label not in self.onset_labels:
                    raise ValueError(
                        f"The onset-envelope model has no {label!r} class. "
                        f"Available labels: {self.onset_labels}"
                    )
                onset_id = self.onset_labels.index(label)
                values = {
                    "f0": _f0_to_onset_control(mapped_f0[frame]),
                    "loudness": float(loudness[frame]),
                    "periodicity": float(controls["periodicity"][frame]),
                }
                selected = [values[name] for name in self.onset_control_names]
                with torch.inference_mode():
                    prediction = self.onset_model(
                        torch.tensor([onset_id], device=self.device),
                        torch.tensor(
                            [selected], dtype=torch.float32, device=self.device
                        ),
                    )
                envelope = prediction[0].detach().cpu().numpy()
                envelope_index = 0
            if gate[frame] and envelope_index < len(envelope):
                output[frame] = envelope[envelope_index]
                envelope_index += 1
        return output

    def _model_inputs(self, controls: dict) -> dict[str, torch.Tensor]:
        def control(name: str) -> torch.Tensor:
            values = np.asarray(controls[name], dtype=np.float32)
            return torch.from_numpy(values)[None, :, None].to(self.device)

        return {
            "pitch": control("bass_f0_hz"),
            "loudness": control("bass_loudness_z"),
            "articulation": torch.from_numpy(
                np.asarray(controls["bass_articulation_id"], dtype=np.int64)
            )[None].to(self.device),
            "onset_strength": control("bass_onset_strength"),
            "offset": control("offset"),
            "gate": control("gate"),
            "note_age": control("note_age"),
            "periodicity": control("periodicity"),
        }

    def run(
        self,
        audio_path: str | Path,
        articulation_mode: str | None = None,
    ) -> dict:
        source = Path(audio_path).expanduser().resolve()
        mode = articulation_mode or self.config.articulation_mode
        timings: dict[str, float] = {}
        total_started = time.perf_counter()

        started = time.perf_counter()
        audio, sample_rate, decoder = load_audio_file(source, 16000)
        timings["audio_decode_seconds"] = time.perf_counter() - started
        if audio.size == 0:
            raise ValueError(f"No audio samples decoded from {source}")

        started = time.perf_counter()
        vocal_config = VocalControlConfig(
            torchcrepe_device=str(self.device),
            torchcrepe_fmin=50.0,
            torchcrepe_fmax=1000.0,
        )
        vocal = extract_voice_controls(
            audio, config=vocal_config, source_name=str(source)
        )
        _synchronize(self.device)
        timings["control_extraction_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        mapped_f0 = self._map_pitch(vocal)
        articulation_id, articulation_name = self._articulation_track(
            vocal, mapped_f0, mode
        )
        bass_onset_strength = self._onset_strength(
            vocal, mapped_f0, articulation_name
        )
        bass_loudness = np.clip(
            np.asarray(vocal["loudness_z"], dtype=np.float32),
            -self.config.loudness_clip_std,
            self.config.loudness_clip_std,
        )
        controls = {
            **vocal,
            "bass_f0_hz": mapped_f0,
            "bass_loudness_z": bass_loudness,
            "bass_articulation_id": articulation_id,
            "bass_articulation_name": articulation_name,
            "bass_onset_strength": bass_onset_strength,
            "articulation_mode": mode,
        }
        _synchronize(self.device)
        timings["control_mapping_seconds"] = time.perf_counter() - started

        inputs = self._model_inputs(controls)
        torch.manual_seed(self.config.random_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.random_seed)
        _synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            reconstruction = self.model(**inputs)
        _synchronize(self.device)
        timings["synthesis_seconds"] = time.perf_counter() - started
        reconstruction = (
            reconstruction[0, : len(audio), 0].detach().cpu().numpy().astype(np.float32)
        )
        branches = {
            name: value[0, : len(audio), 0].detach().cpu().numpy().astype(np.float32)
            for name, value in self.model.last_branch_outputs.items()
            if name in {"sustain", "noise", "transient"}
        }

        started = time.perf_counter()
        output_dir = _unique_output_dir(self.config.output_root, source)
        raw_path = output_dir / "bass_reconstruction_raw.wav"
        listen_path = output_dir / "bass_reconstruction_listen.wav"
        input_path = output_dir / "vocal_input.wav"
        sf.write(input_path, audio, sample_rate, subtype="FLOAT")
        sf.write(raw_path, reconstruction, sample_rate, subtype="FLOAT")
        listen_audio = reconstruction.copy()
        peak = float(np.max(np.abs(listen_audio))) if listen_audio.size else 0.0
        listen_scale = 1.0
        if peak > 0.95:
            listen_scale = 0.95 / peak
            listen_audio *= listen_scale
        sf.write(listen_path, listen_audio, sample_rate, subtype="PCM_16")
        for name, branch in branches.items():
            sf.write(output_dir / f"branch_{name}.wav", branch, sample_rate, subtype="FLOAT")
        timings["output_write_seconds"] = time.perf_counter() - started
        timings["total_seconds"] = time.perf_counter() - total_started

        duration = len(audio) / float(sample_rate)
        timings["synthesis_rtf"] = timings["synthesis_seconds"] / duration
        timings["total_rtf"] = timings["total_seconds"] / duration
        latency = self._latency_summary(vocal)
        metadata = {
            "source": str(source),
            "decoder": decoder,
            "sample_rate": sample_rate,
            "audio_duration_seconds": duration,
            "device": str(self.device),
            "cached_dct_bank": True,
            "cached_dct_bank_shape": list(self.cached_dct_bank.shape),
            "articulation_mode": mode,
            "pitch_shift_semitones": self.config.pitch_shift_semitones,
            "slap_crossover_midi": self.config.slap_crossover_midi,
            "slap_crossover_hz": 440.0
            * 2.0 ** ((self.config.slap_crossover_midi - 69.0) / 12.0),
            "listen_output_scale": listen_scale,
            "setup_times": self.setup_times,
            "timings": timings,
            "latency": latency,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self.config).items()
            },
        }
        with (output_dir / "run.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)

        return {
            "audio": audio,
            "reconstruction": reconstruction,
            "listen_audio": listen_audio,
            "branches": branches,
            "controls": controls,
            "sample_rate": sample_rate,
            "output_dir": output_dir,
            "input_path": input_path,
            "raw_path": raw_path,
            "listen_path": listen_path,
            "timings": timings,
            "latency": latency,
            "metadata": metadata,
        }

    @staticmethod
    def _latency_summary(controls: dict) -> dict:
        aubio = controls["aubio_onsets"]
        event = np.asarray(aubio.get("complex_event_times", []), dtype=np.float64)
        decision = np.asarray(
            aubio.get("complex_decision_times", []), dtype=np.float64
        )
        count = min(len(event), len(decision))
        delays = (decision[:count] - event[:count]) * 1000.0
        hop_ms = float(controls["config"].hop_seconds * 1000.0)
        frame_ms = float(controls["config"].frame_seconds * 1000.0)
        return {
            "control_hop_ms": hop_ms,
            "analysis_frame_ms": frame_ms,
            "aubio_complex_events": int(count),
            "aubio_decision_delay_median_ms": (
                float(np.median(delays)) if count else None
            ),
            "aubio_decision_delay_max_ms": (
                float(np.max(delays)) if count else None
            ),
            "algorithmic_parallel_path_bound_ms": (
                float(max(frame_ms, np.max(delays))) if count else frame_ms
            ),
            "note": (
                "The bound excludes audio-driver and block scheduling latency. "
                "The causal 80 ms gate release and offset hold are state rules, "
                "not mandatory look-ahead latency."
            ),
        }

    def print_timing(self, result: dict) -> None:
        timing = result["timings"]
        latency = result["latency"]
        duration = result["metadata"]["audio_duration_seconds"]
        print(f"Device: {self.device}; cached DCT bank: yes")
        print(f"Audio duration: {duration:.3f} s")
        print(
            "Observed processing: "
            f"controls {timing['control_extraction_seconds']:.3f} s | "
            f"mapping {timing['control_mapping_seconds']:.3f} s | "
            f"synthesis {timing['synthesis_seconds']:.3f} s "
            f"(RTF {timing['synthesis_rtf']:.3f}) | "
            f"total {timing['total_seconds']:.3f} s "
            f"(RTF {timing['total_rtf']:.3f})"
        )
        median = latency["aubio_decision_delay_median_ms"]
        maximum = latency["aubio_decision_delay_max_ms"]
        if median is None:
            aubio_text = "no accepted aubio events"
        else:
            aubio_text = f"aubio onset median {median:.1f} ms, max {maximum:.1f} ms"
        print(
            "Algorithmic timing: "
            f"{latency['analysis_frame_ms']:.0f} ms frame, "
            f"{latency['control_hop_ms']:.0f} ms hop, {aubio_text}; "
            f"parallel-path bound about "
            f"{latency['algorithmic_parallel_path_bound_ms']:.1f} ms"
        )
        print(
            "The algorithmic bound excludes microphone/audio-driver buffering. "
            "The wall times above are offline notebook measurements."
        )
        print(f"Output: {result['output_dir']}")
