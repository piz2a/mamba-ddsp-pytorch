import argparse
import csv
import json
import math
import random
from pathlib import Path

import librosa as li
import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bass_ddsp.export_branch_debug import load_model, make_dataset, reconstruct
from bass_ddsp.train import multiscale_spectral_loss


def _load_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _frame_rms(audio, block_size):
    usable = audio.shape[-1] - (audio.shape[-1] % block_size)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[:usable].reshape(-1, block_size)
    return np.sqrt(np.mean(frames * frames, axis=-1) + 1e-12).astype(np.float32)


def _rms(audio):
    return float(np.sqrt(np.mean(audio * audio) + 1e-12))


def _peak(audio):
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def _safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _align_frames(values, target_frames):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape[0] == target_frames:
        return values
    if values.shape[0] == 0:
        return np.full(target_frames, np.nan, dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        fill = float(values[finite][0]) if finite.any() else float("nan")
        return np.full(target_frames, fill, dtype=np.float32)
    return np.interp(new_x, old_x[finite], values[finite]).astype(np.float32)


def _spectrogram_mag(audio, sr, n_fft=2048, hop_length=256):
    n_fft = min(int(n_fft), max(256, audio.shape[-1]))
    return np.abs(
        li.stft(
            audio.astype(np.float32),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            center=True,
        )
    ).astype(np.float32)


def _lsd_db(target, recon, sr, hop_length):
    target_mag = _spectrogram_mag(target, sr, hop_length=hop_length)
    recon_mag = _spectrogram_mag(recon, sr, hop_length=hop_length)
    frames = min(target_mag.shape[1], recon_mag.shape[1])
    bins = min(target_mag.shape[0], recon_mag.shape[0])
    target_db = 20.0 * np.log10(np.maximum(target_mag[:bins, :frames], 1e-7))
    recon_db = 20.0 * np.log10(np.maximum(recon_mag[:bins, :frames], 1e-7))
    return float(np.sqrt(np.mean((target_db - recon_db) ** 2)))


def _mss_loss(target, recon, config, device):
    target_t = torch.from_numpy(target).float().unsqueeze(0).to(device)
    recon_t = torch.from_numpy(recon).float().unsqueeze(0).to(device)
    with torch.no_grad():
        loss = multiscale_spectral_loss(
            target_t,
            recon_t,
            config["train"]["scales"],
            config["train"]["overlap"],
        )
    return float(loss.detach().cpu().item())


def _onset_mask(intervals, n_samples, sr, seconds):
    mask = np.zeros(n_samples, dtype=bool)
    width = max(1, int(round(float(seconds) * sr)))
    for interval in intervals:
        start = int(round(float(interval["start_seconds"]) * sr))
        end = min(n_samples, start + width)
        if end > start:
            mask[start:end] = True
    return mask


def _high_frequency_rms(audio, sr, mask, cutoff_hz=700.0):
    if not mask.any():
        return 0.0
    mag = _spectrogram_mag(audio, sr, hop_length=256)
    freqs = li.fft_frequencies(sr=sr, n_fft=min(2048, max(256, audio.shape[-1])))
    frame_times = li.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=256)
    frame_samples = np.clip((frame_times * sr).astype(np.int64), 0, len(mask) - 1)
    frame_mask = mask[frame_samples]
    bin_mask = freqs >= float(cutoff_hz)
    if not frame_mask.any() or not bin_mask.any():
        return 0.0
    return float(np.sqrt(np.mean(mag[bin_mask][:, frame_mask] ** 2) + 1e-12))


def _pitch_metrics(recon, label_pitch, gate, sr, block_size, fmin, fmax):
    try:
        f0, _, _ = li.pyin(
            recon.astype(np.float32),
            fmin=float(fmin),
            fmax=float(fmax),
            sr=int(sr),
            frame_length=2048,
            hop_length=int(block_size),
            center=True,
        )
    except Exception:
        f0 = np.full_like(label_pitch, np.nan, dtype=np.float32)
    f0 = _align_frames(f0, len(label_pitch))
    label_pitch = np.asarray(label_pitch, dtype=np.float32)
    gate = np.asarray(gate, dtype=np.float32)
    active = (gate > 0.5) & np.isfinite(label_pitch) & (label_pitch > 1.0)
    valid = active & np.isfinite(f0) & (f0 > 1.0)
    active_count = int(active.sum())
    valid_count = int(valid.sum())
    if valid_count:
        cents = np.abs(1200.0 * np.log2(np.maximum(f0[valid], 1e-6) / label_pitch[valid]))
        median_cents = float(np.median(cents))
        mean_cents = float(np.mean(cents))
        gross_valid = np.zeros(active_count, dtype=bool)
        active_indices = np.flatnonzero(active)
        valid_lookup = {int(idx): float(value) for idx, value in zip(np.flatnonzero(valid), cents)}
        for pos, idx in enumerate(active_indices):
            value = valid_lookup.get(int(idx))
            gross_valid[pos] = True if value is None else value > 100.0
        gross_pct = float(np.mean(gross_valid) * 100.0)
    else:
        median_cents = float("nan")
        mean_cents = float("nan")
        gross_pct = 100.0 if active_count else float("nan")
    valid_pct = float(valid_count / active_count * 100.0) if active_count else float("nan")
    return {
        "f0_median_cents": median_cents,
        "f0_mean_cents": mean_cents,
        "gross_pitch_error_pct": gross_pct,
        "f0_valid_pct": valid_pct,
    }


def _branch_metrics(branches):
    names = ["sustain", "noise", "transient", "signal"]
    values = {}
    signal_rms = max(_rms(branches.get("signal", np.zeros(1, dtype=np.float32))), 1e-12)
    for name in names:
        audio = branches.get(name)
        if audio is None:
            values[f"{name}_rms"] = 0.0
            values[f"{name}_rms_vs_signal_pct"] = 0.0
        else:
            value = _rms(audio.astype(np.float32))
            values[f"{name}_rms"] = value
            values[f"{name}_rms_vs_signal_pct"] = value / signal_rms * 100.0
    return values


def _audio_metrics(target, recon, data, config, device, onset_seconds):
    sr = int(data["sampling_rate"])
    block_size = int(data["block_size"])
    n = min(target.shape[-1], recon.shape[-1])
    target = target[:n].astype(np.float32)
    recon = recon[:n].astype(np.float32)
    target_frame = _frame_rms(target, block_size)
    recon_frame = _frame_rms(recon, block_size)
    frames = min(target_frame.shape[0], recon_frame.shape[0], len(data["gate"]))
    gate = data["gate"][:frames]
    active = gate > 0.5
    if not active.any():
        active = np.ones(frames, dtype=bool)
    target_active = target_frame[:frames][active]
    recon_active = recon_frame[:frames][active]
    rms_ratio = float(np.mean(recon_active) / max(np.mean(target_active), 1e-12))
    rms_corr = _safe_corr(np.log(target_active + 1e-7), np.log(recon_active + 1e-7))

    onset = _onset_mask(data.get("intervals", []), n, sr, onset_seconds)
    target_onset = target[onset] if onset.any() else target
    recon_onset = recon[onset] if onset.any() else recon
    onset_energy_ratio = _rms(recon_onset) / max(_rms(target_onset), 1e-12)
    target_hf = _high_frequency_rms(target, sr, onset)
    recon_hf = _high_frequency_rms(recon, sr, onset)
    onset_hf_log_error = abs(math.log(max(recon_hf, 1e-12) / max(target_hf, 1e-12)))

    pitch_config = config.get("idmt_bass", {})
    pitch_metrics = _pitch_metrics(
        recon,
        data["label_pitch"][:frames],
        data["gate"][:frames],
        sr,
        block_size,
        pitch_config.get("pitch_fmin", config["model"].get("f0_min_hz", 30.0)),
        pitch_config.get("pitch_fmax", config["model"].get("f0_max_hz", 330.0)),
    )

    return {
        "mss": _mss_loss(target, recon, config, device),
        "lsd_db": _lsd_db(target, recon, sr, block_size),
        "rms_ratio": rms_ratio,
        "rms_corr": rms_corr,
        "onset_energy_ratio": onset_energy_ratio,
        "onset_hf_log_error": onset_hf_log_error,
        "peak": _peak(recon),
        "rms": _rms(recon),
        **pitch_metrics,
    }


def _mean_std(values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=0))


def _summarize(rows, models):
    keys = [
        "mss",
        "lsd_db",
        "rms_ratio",
        "rms_corr",
        "f0_median_cents",
        "f0_mean_cents",
        "gross_pitch_error_pct",
        "f0_valid_pct",
        "onset_hf_log_error",
        "onset_energy_ratio",
        "rms",
        "peak",
        "sustain_rms_vs_signal_pct",
        "noise_rms_vs_signal_pct",
        "transient_rms_vs_signal_pct",
    ]
    summary = {}
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        summary[model] = {"num_samples": len(model_rows)}
        for key in keys:
            mean, std = _mean_std([float(row[key]) for row in model_rows])
            summary[model][f"{key}_mean"] = mean
            summary[model][f"{key}_std"] = std
    return summary


def _read_loss_tail(run_dir, tail=1000):
    path = Path(run_dir) / "loss.csv"
    if not path.exists():
        return {}
    rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    tail_rows = rows[-min(tail, len(rows)) :]
    keys = [key for key in rows[0].keys() if key != "step"]
    out = {"steps": int(float(rows[-1]["step"])) + 1}
    for key in keys:
        values = []
        for row in tail_rows:
            try:
                values.append(float(row[key]))
            except ValueError:
                pass
        mean, std = _mean_std(values)
        out[f"tail_{key}_mean"] = mean
        out[f"tail_{key}_std"] = std
    return out


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric_bars(path, summary):
    metrics = [
        ("mss", "MSS loss"),
        ("lsd_db", "LSD dB"),
        ("rms_ratio", "RMS ratio"),
        ("rms_corr", "RMS corr"),
        ("f0_median_cents", "F0 median cents"),
        ("gross_pitch_error_pct", "Gross pitch %"),
        ("onset_hf_log_error", "Onset HF log error"),
        ("onset_energy_ratio", "Onset energy ratio"),
    ]
    models = list(summary.keys())
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    axes = axes.reshape(-1)
    colors = {"Bass-DDSP v2": "#1f77b4", "Vanilla DDSP": "#ff7f0e"}
    for ax, (key, title) in zip(axes, metrics):
        means = [summary[model][f"{key}_mean"] for model in models]
        stds = [summary[model][f"{key}_std"] for model in models]
        ax.bar(models, means, yerr=stds, color=[colors.get(model, "0.5") for model in models], alpha=0.85)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_loss_curves(path, runs):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    fields = [("loss", axes[0]), ("spectral_loss", axes[0]), ("rms_loss", axes[1])]
    for label, run_dir in runs.items():
        loss_path = Path(run_dir) / "loss.csv"
        if not loss_path.exists():
            continue
        steps = []
        values = {field: [] for field, _ in fields}
        with open(loss_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                step = int(float(row["step"]))
                if step % 200 != 0:
                    continue
                steps.append(step)
                for field, _ in fields:
                    values[field].append(float(row[field]))
        for field, ax in fields:
            ax.plot(steps, values[field], label=f"{label} {field}", linewidth=0.9)
    axes[0].set_title("Training losses sampled every 200 steps")
    axes[1].set_title("Frame log-RMS loss")
    for ax in axes:
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _frame_times(frames, sr, block_size):
    return (np.arange(frames) * block_size + block_size * 0.5) / float(sr)


def _spectrogram_db(audio, sr, hop_length):
    mag = _spectrogram_mag(audio, sr, hop_length=hop_length)
    db = li.amplitude_to_db(mag, ref=np.max if np.max(mag) > 0 else 1.0)
    freqs = li.fft_frequencies(sr=sr, n_fft=min(2048, max(256, audio.shape[-1])))
    times = li.frames_to_time(np.arange(db.shape[1]), sr=sr, hop_length=hop_length)
    return db, freqs, times


def _draw_intervals(ax, intervals, labels):
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(idx % cmap.N) for idx, label in enumerate(labels)}
    for interval in intervals:
        label = interval.get("articulation", "")
        ax.axvspan(
            float(interval["start_seconds"]),
            float(interval["end_seconds"]),
            color=colors.get(label, "0.85"),
            alpha=0.12,
            linewidth=0,
        )


def _plot_sample(path, sample_name, data, bass, vanilla):
    sr = int(data["sampling_rate"])
    block_size = int(data["block_size"])
    target = data["audio"].astype(np.float32)
    bass_audio = bass["signal"].astype(np.float32)
    vanilla_audio = vanilla["signal"].astype(np.float32)
    n = min(len(target), len(bass_audio), len(vanilla_audio))
    target = target[:n]
    bass_audio = bass_audio[:n]
    vanilla_audio = vanilla_audio[:n]
    audio_t = np.arange(n) / float(sr)
    frame_count = min(len(data["pitch"]), len(data["gate"]))
    frame_t = _frame_times(frame_count, sr, block_size)
    labels = data.get("articulation_labels", [])
    intervals = data.get("intervals", [])

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(15, 16),
        sharex=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.1, 1.35, 1.35, 1.35, 0.85, 0.95, 0.85]},
    )

    ax = axes[0]
    _draw_intervals(ax, intervals, labels)
    ax.plot(audio_t, target, color="black", linewidth=0.55, label="target")
    ax.plot(audio_t, bass_audio, color="#1f77b4", linewidth=0.55, alpha=0.82, label="Bass-DDSP v2")
    ax.plot(audio_t, vanilla_audio, color="#ff7f0e", linewidth=0.55, alpha=0.82, label="Vanilla DDSP")
    ax.set_title(f"{sample_name}: waveform")
    ax.set_ylabel("amp")
    ax.set_xlim(0, n / float(sr))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", ncol=3)

    for ax, audio, title in [
        (axes[1], target, "target STFT"),
        (axes[2], bass_audio, "Bass-DDSP v2 STFT"),
        (axes[3], vanilla_audio, "Vanilla DDSP STFT"),
    ]:
        db, freqs, times = _spectrogram_db(audio, sr, block_size)
        image = ax.pcolormesh(times, freqs, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        _draw_intervals(ax, intervals, labels)
        ax.set_title(title)
        ax.set_ylabel("Hz")
        ax.set_ylim(0, min(2200, sr / 2))
        ax.set_xlim(0, n / float(sr))
        fig.colorbar(image, ax=ax, label="dB")

    ax = axes[4]
    target_rms = _frame_rms(target, block_size)
    bass_rms = _frame_rms(bass_audio, block_size)
    vanilla_rms = _frame_rms(vanilla_audio, block_size)
    frames = min(len(target_rms), len(bass_rms), len(vanilla_rms), frame_count)
    _draw_intervals(ax, intervals, labels)
    ax.plot(frame_t[:frames], target_rms[:frames], color="black", linewidth=1.1, label="target")
    ax.plot(frame_t[:frames], bass_rms[:frames], color="#1f77b4", linewidth=1.1, label="Bass-DDSP v2")
    ax.plot(frame_t[:frames], vanilla_rms[:frames], color="#ff7f0e", linewidth=1.1, label="Vanilla DDSP")
    ax.set_title("frame RMS envelope")
    ax.set_ylabel("RMS")
    ax.set_xlim(0, n / float(sr))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", ncol=3)

    ax = axes[5]
    _draw_intervals(ax, intervals, labels)
    ax.plot(frame_t[:frame_count], data["pitch"][:frame_count], label="f0 label", linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(frame_t[:frame_count], data["onset_strength"][:frame_count], color="#d62728", label="onset", linewidth=0.9)
    ax2.plot(frame_t[:frame_count], data["gate"][:frame_count], color="#2ca02c", label="gate", linewidth=0.9)
    ax.set_title("controls")
    ax.set_ylabel("Hz")
    ax2.set_ylabel("control")
    ax.set_xlim(0, n / float(sr))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[6]
    _draw_intervals(ax, intervals, labels)
    ax.plot(audio_t, bass.get("sustain", np.zeros(n, dtype=np.float32))[:n], color="#1f77b4", linewidth=0.55, label="bass sustain")
    ax.plot(audio_t, bass.get("transient", np.zeros(n, dtype=np.float32))[:n], color="#9467bd", linewidth=0.55, label="bass transient")
    ax.plot(audio_t, bass.get("noise", np.zeros(n, dtype=np.float32))[:n], color="#17becf", linewidth=0.55, label="bass noise")
    ax.plot(audio_t, vanilla.get("sustain", np.zeros(n, dtype=np.float32))[:n], color="#ff7f0e", linewidth=0.55, alpha=0.8, label="vanilla sustain")
    ax.plot(audio_t, vanilla.get("noise", np.zeros(n, dtype=np.float32))[:n], color="#8c564b", linewidth=0.55, alpha=0.8, label="vanilla noise")
    ax.set_title("branch waveforms")
    ax.set_ylabel("amp")
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, n / float(sr))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", ncol=5, fontsize=8)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_report(path, args, summary, loss_summary, out_dir):
    bass = summary["Bass-DDSP v2"]
    vanilla = summary["Vanilla DDSP"]

    metric_rows = [
        ("MSS loss", "lower", "mss_mean", 4),
        ("LSD dB", "lower", "lsd_db_mean", 3),
        ("Frame RMS ratio", "close to 1", "rms_ratio_mean", 3),
        ("Frame RMS correlation", "higher", "rms_corr_mean", 3),
        ("F0 median cents", "lower", "f0_median_cents_mean", 1),
        ("Gross pitch error %", "lower", "gross_pitch_error_pct_mean", 2),
        ("Onset HF log error", "lower", "onset_hf_log_error_mean", 3),
        ("Onset energy ratio", "close to 1", "onset_energy_ratio_mean", 3),
    ]

    def winner(key, direction):
        b = bass[key]
        v = vanilla[key]
        if not np.isfinite(b) or not np.isfinite(v):
            return "n/a"
        if direction == "lower":
            return "Bass-DDSP v2" if b < v else "Vanilla DDSP"
        if direction == "higher":
            return "Bass-DDSP v2" if b > v else "Vanilla DDSP"
        if direction == "close to 1":
            return "Bass-DDSP v2" if abs(b - 1.0) < abs(v - 1.0) else "Vanilla DDSP"
        return "n/a"

    lines = [
        "# Bass-DDSP v2 vs Vanilla DDSP Comparison",
        "",
        "This report compares the two completed riff-stage models on the same deterministic generated-riff evaluation set.",
        "",
        "## Runs",
        "",
        f"- Bass-DDSP v2: `{args.bass_run}`",
        f"- Vanilla DDSP: `{args.vanilla_run}`",
        f"- Samples evaluated: `{bass['num_samples']}`",
        f"- Evaluation seed: `{args.seed}`",
        f"- Pitch source: `{args.pitch_source}`",
        f"- Output directory: `{out_dir}`",
        "",
        "## Objective Metrics",
        "",
        "| Metric | Direction | Bass-DDSP v2 | Vanilla DDSP | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for name, direction, key, digits in metric_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    direction,
                    _format_float(bass[key], digits),
                    _format_float(vanilla[key], digits),
                    winner(key, direction),
                ]
            )
            + " |"
        )

    lines.extend([
        "",
        "## Training Tail",
        "",
        "These are means over the last logged training steps, useful as a rough convergence check. They are not a replacement for the deterministic comparison set above.",
        "",
        "| Metric | Bass-DDSP v2 | Vanilla DDSP |",
        "|---|---:|---:|",
        f"| steps | {_format_float(loss_summary['Bass-DDSP v2'].get('steps'), 0)} | {_format_float(loss_summary['Vanilla DDSP'].get('steps'), 0)} |",
        f"| tail loss | {_format_float(loss_summary['Bass-DDSP v2'].get('tail_loss_mean'), 4)} | {_format_float(loss_summary['Vanilla DDSP'].get('tail_loss_mean'), 4)} |",
        f"| tail spectral loss | {_format_float(loss_summary['Bass-DDSP v2'].get('tail_spectral_loss_mean'), 4)} | {_format_float(loss_summary['Vanilla DDSP'].get('tail_spectral_loss_mean'), 4)} |",
        f"| tail RMS loss | {_format_float(loss_summary['Bass-DDSP v2'].get('tail_rms_loss_mean'), 4)} | {_format_float(loss_summary['Vanilla DDSP'].get('tail_rms_loss_mean'), 4)} |",
        "",
        "## Branch Diagnostics",
        "",
        "Branch percentages are RMS relative to each model's final signal. Vanilla has no transient branch, so its transient value is zero by construction.",
        "",
        "| Branch metric | Bass-DDSP v2 | Vanilla DDSP |",
        "|---|---:|---:|",
        f"| sustain RMS / signal % | {_format_float(bass['sustain_rms_vs_signal_pct_mean'], 2)} | {_format_float(vanilla['sustain_rms_vs_signal_pct_mean'], 2)} |",
        f"| noise RMS / signal % | {_format_float(bass['noise_rms_vs_signal_pct_mean'], 2)} | {_format_float(vanilla['noise_rms_vs_signal_pct_mean'], 2)} |",
        f"| transient RMS / signal % | {_format_float(bass['transient_rms_vs_signal_pct_mean'], 2)} | {_format_float(vanilla['transient_rms_vs_signal_pct_mean'], 2)} |",
        "",
        "## Visualizations",
        "",
        f"- Metric bars: [`metric_bars.png`](metric_bars.png)",
        f"- Training loss curves: [`loss_curves.png`](loss_curves.png)",
        "",
        "Sample comparison plots:",
        "",
    ])

    for sample_dir in sorted(Path(out_dir).glob("sample_*")):
        plot = sample_dir / "comparison.png"
        if plot.exists():
            lines.append(f"- [`{sample_dir.name}`]({sample_dir.name}/comparison.png)")

    lines.extend([
        "",
        "## Interpretation Rule",
        "",
        "Bass-DDSP v2 is only a useful architectural change if it improves attack, loudness tracking, pitch correctness, or listening preference. A lower training loss alone is not enough.",
        "",
        "FAD is not computed in this run. It should be added later with a fixed audio embedding model and a larger held-out audio set; otherwise the number would be difficult to interpret.",
        "",
    ])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bass-run", required=True)
    parser.add_argument("--vanilla-run", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-plots", type=int, default=4)
    parser.add_argument("--pitch-source", choices=["labels", "torchcrepe"], default="labels")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onset-seconds", type=float, default=0.15)
    args = parser.parse_args()

    bass_run = Path(args.bass_run)
    vanilla_run = Path(args.vanilla_run)
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"model_comparison_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=False)

    bass_config = _load_yaml(bass_run / "config.yaml")
    vanilla_config = _load_yaml(vanilla_run / "config.yaml")
    dataset = make_dataset(bass_config, args.seed, args.pitch_source)
    device = torch.device(args.device)
    metric_device = torch.device("cpu")

    bass_model = load_model(bass_config, bass_run, dataset, device)
    vanilla_model = load_model(vanilla_config, vanilla_run, dataset, device)

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    rows = []
    samples = []

    for position, idx in enumerate(indices):
        data = dataset.generate_debug_example(idx, pitch_source=args.pitch_source)
        target = data["audio"].astype(np.float32)

        torch.manual_seed(args.seed + idx)
        bass_branches = reconstruct(bass_model, bass_config, data, device)
        torch.manual_seed(args.seed + idx)
        vanilla_branches = reconstruct(vanilla_model, vanilla_config, data, device)

        for model_name, branches, config in [
            ("Bass-DDSP v2", bass_branches, bass_config),
            ("Vanilla DDSP", vanilla_branches, vanilla_config),
        ]:
            metrics = _audio_metrics(
                target,
                branches["signal"].astype(np.float32),
                data,
                config,
                metric_device,
                args.onset_seconds,
            )
            rows.append({
                "sample_position": position,
                "index": idx,
                "model": model_name,
                **metrics,
                **_branch_metrics(branches),
            })

        sample_name = f"sample_{position:02d}_idx_{idx:04d}"
        sample_dir = out_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sr = int(data["sampling_rate"])
        sf.write(sample_dir / "target.wav", target, sr, subtype="FLOAT")
        sf.write(sample_dir / "bass_ddsp_v2.wav", bass_branches["signal"].astype(np.float32), sr, subtype="FLOAT")
        sf.write(sample_dir / "vanilla_ddsp.wav", vanilla_branches["signal"].astype(np.float32), sr, subtype="FLOAT")
        sf.write(
            sample_dir / "target_bass_vanilla.wav",
            np.concatenate([
                target,
                np.zeros(int(0.25 * sr), dtype=np.float32),
                bass_branches["signal"].astype(np.float32),
                np.zeros(int(0.25 * sr), dtype=np.float32),
                vanilla_branches["signal"].astype(np.float32),
            ]),
            sr,
            subtype="FLOAT",
        )
        if position < args.num_plots:
            _plot_sample(sample_dir / "comparison.png", sample_name, data, bass_branches, vanilla_branches)
        samples.append({"index": idx, "directory": str(sample_dir)})

    summary = _summarize(rows, ["Bass-DDSP v2", "Vanilla DDSP"])
    loss_summary = {
        "Bass-DDSP v2": _read_loss_tail(bass_run),
        "Vanilla DDSP": _read_loss_tail(vanilla_run),
    }
    _write_csv(out_dir / "per_sample_metrics.csv", rows)
    summary_blob = {
        "args": vars(args),
        "indices": indices,
        "summary": summary,
        "loss_summary": loss_summary,
        "samples": samples,
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary_blob, handle, indent=2)
    _plot_metric_bars(out_dir / "metric_bars.png", summary)
    _plot_loss_curves(
        out_dir / "loss_curves.png",
        {"Bass-DDSP v2": bass_run, "Vanilla DDSP": vanilla_run},
    )
    _write_report(out_dir / "REPORT.md", args, summary, loss_summary, out_dir)

    print(json.dumps({
        "out_dir": str(out_dir),
        "report": str(out_dir / "REPORT.md"),
        "summary": summary,
        "indices": indices,
    }, indent=2))


if __name__ == "__main__":
    main()
