from __future__ import annotations

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


STYLE_COLORS = {
    "FS_NO": "#2176ae",
    "MU_NO": "#5c677d",
    "PK_NO": "#d1495b",
    "SP_NO": "#f79256",
    "ST_NO": "#50a060",
    "FS_DN": "#6f4e7c",
}


def _shade_notes(axis, times, gate):
    gate = np.asarray(gate) > 0.5
    changes = np.diff(np.pad(gate.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.016
    for start, end in zip(starts, ends):
        left = float(times[min(start, len(times) - 1)])
        right = float(times[min(end, len(times) - 1)]) + hop
        axis.axvspan(left, right, color="black", alpha=0.08, linewidth=0)


def plot_demo_result(result):
    controls = result["controls"]
    sr = result["sample_rate"]
    audio = result["audio"]
    reconstruction = result["reconstruction"]
    times = np.asarray(controls["times"])
    audio_times = np.arange(len(audio)) / sr

    figure = plt.figure(figsize=(15, 12), constrained_layout=True)
    grid = figure.add_gridspec(5, 2, height_ratios=[1.0, 1.25, 1.25, 1.0, 1.0])
    input_wave = figure.add_subplot(grid[0, :])
    vocal_spec = figure.add_subplot(grid[1, 0])
    bass_spec = figure.add_subplot(grid[1, 1], sharex=vocal_spec, sharey=vocal_spec)
    pitch_axis = figure.add_subplot(grid[2, :])
    event_axis = figure.add_subplot(grid[3, :], sharex=pitch_axis)
    style_axis = figure.add_subplot(grid[4, :], sharex=pitch_axis)

    input_wave.plot(audio_times, audio, color="#30343f", linewidth=0.65)
    input_wave.plot(
        audio_times,
        reconstruction,
        color="#d1495b",
        linewidth=0.55,
        alpha=0.75,
        label="Bass output",
    )
    _shade_notes(input_wave, times, controls["gate"])
    input_wave.set(title="Input vocal and synthesized bass", ylabel="Amplitude")
    input_wave.legend(loc="upper right", frameon=False)

    for axis, signal, title in (
        (vocal_spec, audio, "Vocal log-mel spectrogram"),
        (bass_spec, reconstruction, "Bass log-mel spectrogram"),
    ):
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=96,
            fmin=30,
            fmax=sr / 2,
            power=2.0,
        )
        db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(
            db,
            sr=sr,
            hop_length=256,
            x_axis="time",
            y_axis="mel",
            fmin=30,
            fmax=sr / 2,
            cmap="magma",
            vmin=-80,
            vmax=0,
            ax=axis,
        )
        axis.set_title(title)

    pitch_axis.plot(times, controls["f0_hz"], color="#2176ae", label="Vocal F0")
    pitch_axis.plot(
        times,
        controls["bass_f0_hz"],
        color="#d1495b",
        linewidth=1.5,
        label="Mapped bass F0",
    )
    periodicity_axis = pitch_axis.twinx()
    periodicity_axis.plot(
        times,
        controls["periodicity"],
        color="#50a060",
        alpha=0.55,
        label="Periodicity",
    )
    pitch_axis.set(title="Pitch mapping and CREPE confidence", ylabel="Hz")
    periodicity_axis.set(ylabel="Periodicity", ylim=(-0.02, 1.02))
    lines = pitch_axis.lines + periodicity_axis.lines
    pitch_axis.legend(lines, [line.get_label() for line in lines], loc="upper right", ncol=3, frameon=False)

    event_axis.plot(
        times,
        controls["bass_loudness_z"],
        color="#2176ae",
        label="Loudness z",
    )
    event_axis.plot(
        times,
        controls["bass_onset_strength"],
        color="#d1495b",
        linewidth=1.5,
        label="Bass onset envelope",
    )
    event_axis.fill_between(
        times,
        0,
        controls["gate"],
        color="#50a060",
        alpha=0.18,
        label="Gate",
    )
    for frame in np.flatnonzero(np.asarray(controls["note_onset"]) > 0.5):
        event_axis.axvline(times[frame], color="#f79256", linewidth=1.0)
    for frame in np.flatnonzero(np.asarray(controls["offset"]) > 0.5):
        event_axis.axvline(times[frame], color="#30343f", linewidth=0.8, linestyle="--")
    event_axis.set(
        title="Causal events and synthesis controls",
        ylabel="Control value",
    )
    event_axis.legend(loc="upper right", ncol=4, frameon=False)

    names = np.asarray(controls["bass_articulation_name"], dtype=object)
    gate = np.asarray(controls["gate"]) > 0.5
    for label, color in STYLE_COLORS.items():
        mask = (names == label) & gate
        style_axis.fill_between(
            times,
            0,
            mask.astype(float),
            where=mask,
            step="post",
            color=color,
            alpha=0.85,
            label=label,
        )
    style_axis.set(
        title=f"Latched articulation ({controls['articulation_mode']})",
        xlabel="Time (s)",
        yticks=[],
        ylim=(0, 1.05),
    )
    handles, labels = style_axis.get_legend_handles_labels()
    if handles:
        style_axis.legend(handles, labels, loc="upper right", ncol=6, frameon=False)
    for axis in (input_wave, pitch_axis, event_axis, style_axis):
        axis.grid(alpha=0.16, linewidth=0.6)
        axis.set_xlim(0, len(audio) / sr)
    return figure
