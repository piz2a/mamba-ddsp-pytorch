from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import math
import random

import librosa as li
import numpy as np
import torch

from ddsp.core import extract_loudness, extract_pitch


PLUCKING_STYLES = ("FS", "MU", "PK", "SP", "ST")
EXPRESSION_STYLES = ("NO", "BE", "DN", "HA", "VI")
OBSERVED_ARTICULATION_STYLES = (
    "FS_NO",
    "MU_NO",
    "PK_NO",
    "SP_NO",
    "ST_NO",
    "FS_BE",
    "FS_DN",
    "FS_HA",
    "FS_VI",
)


@dataclass(frozen=True)
class BassNote:
    path: Path
    pluck: str
    expression: str
    string: int
    fret: int
    frequency: float

    @property
    def articulation(self):
        return f"{self.pluck}_{self.expression}"


def bass_note_frequency(string_number, fret_number):
    open_string_midi = {
        1: 28,  # E1
        2: 33,  # A1
        3: 38,  # D2
        4: 43,  # G2
        5: 23,  # B0, present in a few IDMT files despite the 4-string note.
    }
    midi = open_string_midi[int(string_number)] + int(fret_number)
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def parse_idmt_bass_note(path):
    path = Path(path)
    group = path.parent.parent.name
    category = path.parent.name

    if group == "PS":
        pluck = category
        expression = "NO"
    elif group == "ES":
        pluck = "FS"
        expression = category
    else:
        raise ValueError(f"expected path under PS or ES, got {path}")

    tokens = path.stem.split("_")
    try:
        string_number = int(tokens[-2])
        fret_number = int(tokens[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"could not parse string/fret from {path.name}") from exc

    return BassNote(
        path=path,
        pluck=pluck,
        expression=expression,
        string=string_number,
        fret=fret_number,
        frequency=bass_note_frequency(string_number, fret_number),
    )


def _ordered_labels(values, preferred_order):
    present = set(values)
    labels = [value for value in preferred_order if value in present]
    labels.extend(sorted(present.difference(labels)))
    return labels


class IDMTBassRiffDataset(torch.utils.data.Dataset):
    """Online riff generator for the IDMT-SMT-BASS one-note dataset.

    Each item trims note-level leading/trailing silence, randomly crops musical
    note durations, and stitches notes with equal-power overlap-add crossfades.
    """

    def __init__(
        self,
        data_location,
        sampling_rate,
        block_size,
        signal_length,
        examples_per_epoch=2048,
        min_note_seconds=0.20,
        max_note_seconds=0.90,
        min_crossfade_seconds=0.025,
        max_crossfade_seconds=0.070,
        trim_top_db=35.0,
        trim_onset_top_db=25.0,
        trim_noise_percentile=20.0,
        trim_noise_margin_db=12.0,
        trim_frame_size=512,
        trim_hop_size=128,
        trim_pad_seconds=0.012,
        edge_fade_seconds=0.004,
        release_fade_seconds=0.035,
        event_width_seconds=0.032,
        include_expression_styles=EXPRESSION_STYLES,
        include_string_numbers=(1, 2, 3, 4),
        cache_size=384,
        pitch_source="torchcrepe",
        pitch_fmin=40.0,
        pitch_fmax=600.0,
        peak_normalize=True,
        label_mode="pluck_expression",
        use_note_shape_controls=False,
        note_age_clip_seconds=1.0,
        deduplicate=True,
        seed=None,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(data_location).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"IDMT-SMT-BASS root does not exist: {self.root}")

        self.sampling_rate = int(sampling_rate)
        self.block_size = int(block_size)
        self.signal_length = int(signal_length)
        self.examples_per_epoch = int(examples_per_epoch)
        self.min_note_samples = max(1, int(float(min_note_seconds) * sampling_rate))
        self.max_note_samples = max(
            self.min_note_samples,
            int(float(max_note_seconds) * sampling_rate),
        )
        self.min_crossfade_samples = max(
            1,
            int(float(min_crossfade_seconds) * sampling_rate),
        )
        self.max_crossfade_samples = max(
            self.min_crossfade_samples,
            int(float(max_crossfade_seconds) * sampling_rate),
        )
        self.trim_top_db = float(trim_top_db)
        self.trim_onset_top_db = float(trim_onset_top_db)
        self.trim_noise_percentile = float(trim_noise_percentile)
        self.trim_noise_margin_db = float(trim_noise_margin_db)
        self.trim_frame_size = int(trim_frame_size)
        self.trim_hop_size = int(trim_hop_size)
        self.trim_pad_samples = int(float(trim_pad_seconds) * sampling_rate)
        self.edge_fade_samples = int(float(edge_fade_seconds) * sampling_rate)
        self.release_fade_samples = int(float(release_fade_seconds) * sampling_rate)
        self.event_width_samples = max(
            1,
            int(float(event_width_seconds) * sampling_rate),
        )
        self.include_expression_styles = {
            str(expression)
            for expression in include_expression_styles
        } if include_expression_styles else None
        self.include_string_numbers = {
            int(string_number)
            for string_number in include_string_numbers
        } if include_string_numbers else None
        self.cache_size = int(cache_size)
        self.pitch_source = pitch_source
        self.pitch_fmin = float(pitch_fmin)
        self.pitch_fmax = float(pitch_fmax)
        self.peak_normalize = bool(peak_normalize)
        self.label_mode = str(label_mode)
        if self.label_mode not in {"pluck_expression", "observed_articulation"}:
            raise ValueError(
                "label_mode must be 'pluck_expression' or "
                f"'observed_articulation', got {self.label_mode!r}"
            )
        self.use_note_shape_controls = bool(use_note_shape_controls)
        self.note_age_clip_seconds = max(1e-4, float(note_age_clip_seconds))
        self.deduplicate = bool(deduplicate)
        self.seed = None if seed is None else int(seed)
        self.frames = self.signal_length // self.block_size
        self._audio_cache = OrderedDict()

        self.notes = []
        seen_names = set()
        for path in sorted(self.root.glob("*/*/*.wav")):
            if path.parent.parent.name not in {"PS", "ES"}:
                continue
            if self.deduplicate and path.name in seen_names:
                continue
            note = parse_idmt_bass_note(path)
            if (self.include_expression_styles is not None
                    and note.expression not in self.include_expression_styles):
                continue
            if (self.include_string_numbers is not None
                    and note.string not in self.include_string_numbers):
                continue
            self.notes.append(note)
            seen_names.add(path.name)
        if not self.notes:
            raise FileNotFoundError(f"found no IDMT-SMT-BASS wav files under {self.root}")

        self.pluck_labels = _ordered_labels(
            [note.pluck for note in self.notes],
            PLUCKING_STYLES,
        )
        self.expression_labels = _ordered_labels(
            [note.expression for note in self.notes],
            EXPRESSION_STYLES,
        )
        self.pluck_to_id = {
            label: idx for idx, label in enumerate(self.pluck_labels)
        }
        self.expression_to_id = {
            label: idx for idx, label in enumerate(self.expression_labels)
        }
        self.articulation_labels = _ordered_labels(
            [note.articulation for note in self.notes],
            OBSERVED_ARTICULATION_STYLES,
        )
        self.articulation_to_id = {
            label: idx for idx, label in enumerate(self.articulation_labels)
        }

    @property
    def n_pluck(self):
        return len(self.pluck_labels)

    @property
    def n_expression(self):
        return len(self.expression_labels)

    @property
    def n_articulation(self):
        return len(self.articulation_labels)

    def __len__(self):
        return self.examples_per_epoch

    def _activity_bounds(self, audio):
        if audio.size == 0:
            return 0, 0, {
                "trim_peak_rms": 0.0,
                "trim_noise_rms": 0.0,
                "trim_threshold": 0.0,
            }

        frame_length = min(self.trim_frame_size, max(16, audio.shape[0]))
        rms = li.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=self.trim_hop_size,
            center=False,
        )[0]
        peak = float(rms.max()) if rms.size else 0.0
        if peak <= 1e-8:
            return 0, audio.shape[0], {
                "trim_peak_rms": peak,
                "trim_noise_rms": 0.0,
                "trim_threshold": 0.0,
            }

        edge_count = max(1, int(math.ceil(rms.size * 0.1)))
        edge_rms = np.concatenate([rms[:edge_count], rms[-edge_count:]])
        noise = float(np.percentile(edge_rms, self.trim_noise_percentile))

        relative_threshold = peak * (10.0 ** (-self.trim_top_db / 20.0))
        onset_threshold = peak * (10.0 ** (-self.trim_onset_top_db / 20.0))
        noise_threshold = noise * (10.0 ** (self.trim_noise_margin_db / 20.0))
        threshold = max(relative_threshold, noise_threshold, 1e-6)
        onset_threshold = max(onset_threshold, noise_threshold, 1e-6)
        active = np.flatnonzero(rms >= threshold)
        if active.size == 0:
            return 0, audio.shape[0], {
                "trim_peak_rms": peak,
                "trim_noise_rms": noise,
                "trim_threshold": threshold,
                "trim_onset_threshold": onset_threshold,
            }

        onset_active = np.flatnonzero(rms >= onset_threshold)
        start_frame = int(onset_active[0] if onset_active.size else active[0])
        start = start_frame * self.trim_hop_size - self.trim_pad_samples
        end = int(active[-1] * self.trim_hop_size) + frame_length + self.trim_pad_samples
        start = max(0, start)
        end = min(audio.shape[0], end)
        return start, end, {
            "trim_peak_rms": peak,
            "trim_noise_rms": noise,
            "trim_threshold": threshold,
            "trim_onset_threshold": onset_threshold,
        }

    def _trim_active_region(self, audio):
        start, end, info = self._activity_bounds(audio)
        info.update({
            "original_samples": int(audio.shape[0]),
            "trim_start_sample": int(start),
            "trim_end_sample": int(end),
            "trimmed_samples": int(max(0, end - start)),
        })
        return audio[start:end], info

    def _apply_edge_fades(self, audio):
        fade = min(self.edge_fade_samples, audio.shape[0] // 4)
        if fade <= 1:
            return audio

        audio = audio.copy()
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
        return audio

    def _apply_release_fade(self, audio):
        fade = min(self.release_fade_samples, audio.shape[0] // 3)
        if fade <= 1:
            return audio

        audio = audio.copy()
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return audio

    def _load_note_audio(self, note):
        key = str(note.path)
        cached = self._audio_cache.get(key)
        if cached is not None:
            self._audio_cache.move_to_end(key)
            return cached

        audio, _ = li.load(note.path, sr=self.sampling_rate, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        audio, trim_info = self._trim_active_region(audio)
        audio = self._apply_edge_fades(audio)
        if audio.size == 0:
            audio = np.zeros(max(1, self.edge_fade_samples), dtype=np.float32)

        cached = (audio, trim_info)
        self._audio_cache[key] = cached
        if self.cache_size > 0 and len(self._audio_cache) > self.cache_size:
            self._audio_cache.popitem(last=False)
        return cached

    def _choose_rng(self, idx):
        if self.seed is not None:
            return random.Random((self.seed + int(idx) * 1000003) % (2**63 - 1))

        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        return random.Random((seed + int(idx) * 1000003) % (2**63 - 1))

    def _random_note_audio(self, rng, note):
        audio, trim_info = self._load_note_audio(note)
        target = rng.randint(self.min_note_samples, self.max_note_samples)
        cropped = False
        if audio.shape[0] >= target:
            segment = audio[:target].copy()
            cropped = audio.shape[0] > target
            if cropped:
                segment = self._apply_release_fade(segment)
        else:
            segment = audio.copy()

        info = dict(trim_info)
        info.update({
            "source_samples": int(audio.shape[0]),
            "segment_samples": int(segment.shape[0]),
            "target_samples": int(target),
            "cropped": bool(cropped),
        })
        return segment, info

    def _event(self, start, note, trim_info, crossfade, transition_start=None):
        transition_start = start if transition_start is None else transition_start
        return {
            "start_sample": int(start),
            "transition_start_sample": int(transition_start),
            "transition_midpoint_sample": int(start),
            "transition_end_sample": int(transition_start + crossfade),
            "pluck_id": int(self.pluck_to_id[note.pluck]),
            "expression_id": int(self.expression_to_id[note.expression]),
            "articulation_id": int(self.articulation_to_id[note.articulation]),
            "pluck": note.pluck,
            "expression": note.expression,
            "articulation": note.articulation,
            "frequency": float(note.frequency),
            "string": int(note.string),
            "fret": int(note.fret),
            "source_path": str(note.path),
            "crossfade_samples": int(crossfade),
            **trim_info,
        }

    def _append_note(self, audio, events, note_audio, note, rng, trim_info):
        if audio.size == 0:
            events.append(self._event(0, note, trim_info, 0, 0))
            return note_audio.copy()

        crossfade = rng.randint(
            self.min_crossfade_samples,
            self.max_crossfade_samples,
        )
        crossfade = min(crossfade, audio.shape[0], note_audio.shape[0])
        start = audio.shape[0] - crossfade
        theta = np.linspace(0.0, math.pi / 2.0, crossfade, dtype=np.float32)
        fade_out = np.cos(theta)
        fade_in = np.sin(theta)

        out = audio.copy()
        out[start:] = out[start:] * fade_out + note_audio[:crossfade] * fade_in
        out = np.concatenate([out, note_audio[crossfade:]], axis=0)

        label_start = start + crossfade // 2
        events.append(self._event(label_start, note, trim_info, crossfade, start))
        return out

    def _intervals(self, events):
        intervals = []
        for idx, event in enumerate(events):
            start = int(event["start_sample"])
            end = (
                int(events[idx + 1]["start_sample"])
                if idx + 1 < len(events)
                else self.signal_length
            )
            start = min(max(start, 0), self.signal_length)
            end = min(max(end, 0), self.signal_length)
            if end <= start:
                continue
            interval = dict(event)
            interval["end_sample"] = end
            interval["start_seconds"] = start / self.sampling_rate
            interval["end_seconds"] = end / self.sampling_rate
            interval["duration_seconds"] = (end - start) / self.sampling_rate
            intervals.append(interval)
        return intervals

    def _label_tracks(self, events):
        starts = np.asarray([event["start_sample"] for event in events], dtype=np.int64)
        plucks = np.asarray([event["pluck_id"] for event in events], dtype=np.int64)
        expressions = np.asarray(
            [event["expression_id"] for event in events],
            dtype=np.int64,
        )
        articulations = np.asarray(
            [event["articulation_id"] for event in events],
            dtype=np.int64,
        )
        frequencies = np.asarray(
            [event["frequency"] for event in events],
            dtype=np.float32,
        )

        frame_positions = (
            np.arange(self.frames, dtype=np.int64) * self.block_size
            + self.block_size // 2
        )
        frame_events = np.searchsorted(starts, frame_positions, side="right") - 1
        frame_events = np.clip(frame_events, 0, len(events) - 1)
        return (
            plucks[frame_events],
            expressions[frame_events],
            articulations[frame_events],
            frequencies[frame_events],
        )

    def _event_pulse_track(self, samples):
        frame_positions = (
            np.arange(self.frames, dtype=np.float32) * self.block_size
            + self.block_size / 2
        )
        track = np.zeros(self.frames, dtype=np.float32)
        width = float(self.event_width_samples)
        for sample in samples:
            distance = np.abs(frame_positions - float(sample))
            pulse = np.maximum(0.0, 1.0 - distance / width)
            track = np.maximum(track, pulse.astype(np.float32))
        return track

    def _note_event_tracks(self, intervals):
        onset_samples = [interval["start_sample"] for interval in intervals]
        offset_samples = [interval["end_sample"] for interval in intervals]
        return (
            self._event_pulse_track(onset_samples),
            self._event_pulse_track(offset_samples),
        )

    def _note_shape_tracks(self, intervals):
        frame_positions = (
            np.arange(self.frames, dtype=np.float32) * self.block_size
            + self.block_size / 2
        )
        gate = np.zeros(self.frames, dtype=np.float32)
        note_age = np.zeros(self.frames, dtype=np.float32)
        note_progress = np.zeros(self.frames, dtype=np.float32)

        clip_seconds = self.note_age_clip_seconds
        for interval in intervals:
            start = float(interval["start_sample"])
            end = float(interval["end_sample"])
            duration = max(1.0, end - start)
            active = (frame_positions >= start) & (frame_positions < end)
            if not np.any(active):
                continue

            age_seconds = (frame_positions[active] - start) / self.sampling_rate
            gate[active] = 1.0
            note_age[active] = np.minimum(age_seconds, clip_seconds)
            note_progress[active] = np.clip(
                (frame_positions[active] - start) / duration,
                0.0,
                1.0,
            )

        return gate, note_age, note_progress

    def _pitch_track(self, audio, label_pitch):
        if self.pitch_source == "labels":
            return label_pitch.astype(np.float32)
        if self.pitch_source == "torchcrepe":
            return extract_pitch(
                audio,
                self.sampling_rate,
                self.block_size,
                fmin=self.pitch_fmin,
                fmax=self.pitch_fmax,
            ).astype(np.float32)
        raise ValueError(
            "pitch_source must be 'torchcrepe' or 'labels', "
            f"got {self.pitch_source!r}"
        )

    def _format_item(
        self,
        audio,
        pitch,
        loudness,
        pluck,
        expression,
        articulation,
        onset,
        offset,
        gate,
        note_age,
        note_progress,
    ):
        if self.label_mode == "observed_articulation":
            if self.use_note_shape_controls:
                return (
                    torch.from_numpy(audio),
                    torch.from_numpy(pitch),
                    torch.from_numpy(loudness),
                    torch.from_numpy(articulation),
                    torch.from_numpy(onset),
                    torch.from_numpy(offset),
                    torch.from_numpy(gate),
                    torch.from_numpy(note_age),
                    torch.from_numpy(note_progress),
                )
            return (
                torch.from_numpy(audio),
                torch.from_numpy(pitch),
                torch.from_numpy(loudness),
                torch.from_numpy(articulation),
            )

        return (
            torch.from_numpy(audio),
            torch.from_numpy(pitch),
            torch.from_numpy(loudness),
            torch.from_numpy(pluck),
            torch.from_numpy(expression),
            torch.from_numpy(onset),
            torch.from_numpy(offset),
        )

    def _riff(self, rng):
        audio = np.zeros(0, dtype=np.float32)
        events = []
        while audio.shape[0] < self.signal_length:
            note = rng.choice(self.notes)
            note_audio, trim_info = self._random_note_audio(rng, note)
            if note_audio.size <= 1:
                continue
            audio = self._append_note(audio, events, note_audio, note, rng, trim_info)

        audio = audio[:self.signal_length].astype(np.float32, copy=False)
        if self.peak_normalize:
            peak = float(np.max(np.abs(audio)))
            if peak > 0.99:
                audio = audio * (0.99 / peak)

        intervals = self._intervals(events)
        pluck, expression, articulation, label_pitch = self._label_tracks(events)
        onset, offset = self._note_event_tracks(intervals)
        gate, note_age, note_progress = self._note_shape_tracks(intervals)
        return (
            audio,
            pluck,
            expression,
            articulation,
            label_pitch,
            onset,
            offset,
            gate,
            note_age,
            note_progress,
            intervals,
        )

    def __getitem__(self, idx):
        rng = self._choose_rng(idx)
        (
            audio,
            pluck,
            expression,
            articulation,
            label_pitch,
            onset,
            offset,
            gate,
            note_age,
            note_progress,
            _,
        ) = self._riff(rng)

        loudness = extract_loudness(
            audio,
            self.sampling_rate,
            self.block_size,
        ).astype(np.float32)

        pitch = self._pitch_track(audio, label_pitch)
        return self._format_item(
            audio,
            pitch,
            loudness,
            pluck,
            expression,
            articulation,
            onset,
            offset,
            gate,
            note_age,
            note_progress,
        )

    def generate_debug_example(self, idx=0, pitch_source=None):
        rng = self._choose_rng(idx)
        (
            audio,
            pluck,
            expression,
            articulation,
            label_pitch,
            onset,
            offset,
            gate,
            note_age,
            note_progress,
            intervals,
        ) = self._riff(rng)
        loudness = extract_loudness(
            audio,
            self.sampling_rate,
            self.block_size,
        ).astype(np.float32)

        pitch_source = pitch_source or self.pitch_source
        if pitch_source == "labels":
            pitch = label_pitch.astype(np.float32)
        elif pitch_source == "torchcrepe":
            pitch = extract_pitch(
                audio,
                self.sampling_rate,
                self.block_size,
                fmin=self.pitch_fmin,
                fmax=self.pitch_fmax,
            ).astype(np.float32)
        else:
            raise ValueError(
                "pitch_source must be 'torchcrepe' or 'labels', "
                f"got {pitch_source!r}"
            )

        return {
            "audio": audio,
            "pitch": pitch,
            "label_pitch": label_pitch,
            "loudness": loudness,
            "pluck": pluck,
            "expression": expression,
            "articulation": articulation,
            "onset": onset,
            "offset": offset,
            "gate": gate,
            "note_age": note_age,
            "note_progress": note_progress,
            "intervals": intervals,
            "pluck_labels": list(self.pluck_labels),
            "expression_labels": list(self.expression_labels),
            "articulation_labels": list(self.articulation_labels),
            "sampling_rate": self.sampling_rate,
            "block_size": self.block_size,
            "pitch_source": pitch_source,
        }

    def generate_debug_riff(self, idx=0, pitch_source=None):
        return self.generate_debug_example(idx, pitch_source)


class IDMTBassNoteDataset(IDMTBassRiffDataset):
    """Single-note IDMT-SMT-BASS dataset for Bass-DDSP v2 bootstrapping."""

    def _single_note(self, rng):
        note = rng.choice(self.notes)
        note_audio, trim_info = self._random_note_audio(rng, note)
        if note_audio.size <= 1:
            note_audio = np.zeros(max(1, self.edge_fade_samples), dtype=np.float32)

        active_samples = min(note_audio.shape[0], self.signal_length)
        segment = note_audio[:active_samples].astype(np.float32, copy=True)
        if note_audio.shape[0] > active_samples:
            segment = self._apply_release_fade(segment)

        audio = np.zeros(self.signal_length, dtype=np.float32)
        audio[:active_samples] = segment
        if self.peak_normalize:
            peak = float(np.max(np.abs(audio)))
            if peak > 0.99:
                audio = audio * (0.99 / peak)

        event = self._event(0, note, trim_info, 0, 0)
        interval = dict(event)
        interval["end_sample"] = int(active_samples)
        interval["start_seconds"] = 0.0
        interval["end_seconds"] = active_samples / self.sampling_rate
        interval["duration_seconds"] = active_samples / self.sampling_rate
        intervals = [interval]

        pluck = np.full(
            self.frames,
            self.pluck_to_id[note.pluck],
            dtype=np.int64,
        )
        expression = np.full(
            self.frames,
            self.expression_to_id[note.expression],
            dtype=np.int64,
        )
        articulation = np.full(
            self.frames,
            self.articulation_to_id[note.articulation],
            dtype=np.int64,
        )
        label_pitch = np.full(self.frames, note.frequency, dtype=np.float32)
        onset, offset = self._note_event_tracks(intervals)
        gate, note_age, note_progress = self._note_shape_tracks(intervals)
        return (
            audio,
            pluck,
            expression,
            articulation,
            label_pitch,
            onset,
            offset,
            gate,
            note_age,
            note_progress,
            intervals,
        )

    def __getitem__(self, idx):
        rng = self._choose_rng(idx)
        (
            audio,
            pluck,
            expression,
            articulation,
            label_pitch,
            onset,
            offset,
            gate,
            note_age,
            note_progress,
            _,
        ) = self._single_note(rng)

        loudness = extract_loudness(
            audio,
            self.sampling_rate,
            self.block_size,
        ).astype(np.float32)

        pitch = self._pitch_track(audio, label_pitch)
        return self._format_item(
            audio,
            pitch,
            loudness,
            pluck,
            expression,
            articulation,
            onset,
            offset,
            gate,
            note_age,
            note_progress,
        )

    def generate_debug_example(self, idx=0, pitch_source=None):
        rng = self._choose_rng(idx)
        (
            audio,
            pluck,
            expression,
            articulation,
            label_pitch,
            onset,
            offset,
            gate,
            note_age,
            note_progress,
            intervals,
        ) = self._single_note(rng)
        loudness = extract_loudness(
            audio,
            self.sampling_rate,
            self.block_size,
        ).astype(np.float32)

        pitch_source = pitch_source or self.pitch_source
        old_pitch_source = self.pitch_source
        self.pitch_source = pitch_source
        try:
            pitch = self._pitch_track(audio, label_pitch)
        finally:
            self.pitch_source = old_pitch_source

        return {
            "audio": audio,
            "pitch": pitch,
            "label_pitch": label_pitch,
            "loudness": loudness,
            "pluck": pluck,
            "expression": expression,
            "articulation": articulation,
            "onset": onset,
            "offset": offset,
            "gate": gate,
            "note_age": note_age,
            "note_progress": note_progress,
            "intervals": intervals,
            "pluck_labels": list(self.pluck_labels),
            "expression_labels": list(self.expression_labels),
            "articulation_labels": list(self.articulation_labels),
            "sampling_rate": self.sampling_rate,
            "block_size": self.block_size,
            "pitch_source": pitch_source,
        }
