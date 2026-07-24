from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent
BASE_NAME = "bass_ddsp_v2_overall_diagram"

COLORS = {
    "data": "#e8f0fe",
    "control": "#d9ead3",
    "neural": "#fce5cd",
    "dsp": "#d0e0e3",
    "audio": "#eadcf8",
    "loss": "#f4cccc",
    "note": "#f7f7f7",
    "edge": "#222222",
}


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    fc,
    fontsize=8.5,
    weight="normal",
    radius=0.035,
    lw=1.05,
    ec="#222222",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        family="DejaVu Sans",
        linespacing=1.18,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, label=None, color="#222222", lw=1.0, rad=0.0, ms=8):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(patch)
    if label:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.3,
            family="DejaVu Sans",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
            zorder=4,
        )


def line(ax, start, end, color="#555555", lw=0.9):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw, zorder=1)


def group_label(ax, x, y, text):
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        family="DejaVu Sans",
        color="#444444",
    )


def draw_diagram():
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.linewidth": 0.0,
        }
    )

    fig, ax = plt.subplots(figsize=(18.4, 9.8))
    ax.set_xlim(0, 18.4)
    ax.set_ylim(0, 9.8)
    ax.axis("off")

    ax.text(
        0.35,
        9.35,
        "Bass-DDSP v2: label-conditioned transient-assisted hybrid synthesizer",
        ha="left",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        family="DejaVu Sans",
    )
    ax.text(
        0.36,
        9.02,
        "T = control frames, block_size = 256 audio samples, N = T * block_size. "
        "Active controls are causal; note_progress, string_id, fret_id, and centroid are excluded.",
        ha="left",
        va="center",
        fontsize=8.7,
        family="DejaVu Sans",
        color="#444444",
    )

    group_label(ax, 0.45, 8.48, "Data and controls")
    group_label(ax, 4.15, 8.48, "Frame-rate encoder / decoder")
    group_label(ax, 9.65, 8.48, "Audio-rate differentiable synthesis")
    group_label(ax, 15.75, 8.48, "Output and losses")

    # Left column.
    box(
        ax,
        0.35,
        7.45,
        3.30,
        0.68,
        "IDMT-SMT-BASS / riff generator\ntrim -> crossfade -> aligned labels",
        COLORS["data"],
        fontsize=8.2,
        weight="bold",
    )
    box(
        ax,
        0.35,
        4.05,
        3.30,
        2.88,
        "Frame controls, all at (B,T,*)\n\n"
        "f0(t): label Hz, (B,T,1)\n"
        "loudness(t): z-score, (B,T,1)\n"
        "articulation_id(t): observed class, (B,T)\n"
        "gate(t): active note mask\n"
        "onset_strength(t): HPSS [0,1]\n"
        "offset(t): note-end pulse\n"
        "note_age(t): causal seconds since onset\n"
        "periodicity(t): harmonicity prior/conf.",
        COLORS["control"],
        fontsize=7.4,
        radius=0.025,
    )
    box(
        ax,
        0.35,
        2.60,
        3.30,
        0.72,
        "Target waveform\nx_target: (B,N,1)",
        COLORS["audio"],
        fontsize=8.5,
        weight="bold",
    )
    arrow(ax, (2.00, 7.45), (2.00, 6.93), "frame extraction")
    arrow(ax, (2.00, 4.05), (2.00, 3.32))

    # Frame-rate encoder.
    box(
        ax,
        4.10,
        7.35,
        2.78,
        0.66,
        "f0 Hz -> MIDI [0,1]\nPitch MLP -> 64 ch",
        COLORS["neural"],
        fontsize=8.0,
    )
    box(
        ax,
        4.10,
        6.43,
        2.78,
        0.66,
        "Loudness z-score\nLoudness MLP -> 64 ch",
        COLORS["neural"],
        fontsize=8.0,
    )
    box(
        ax,
        4.10,
        5.06,
        2.78,
        0.98,
        "ArticulationEncoder\nembedding(24) + onset/offset/gate/\nnote_age/periodicity -> z(t), 64 ch",
        COLORS["neural"],
        fontsize=7.4,
    )
    box(
        ax,
        4.10,
        4.14,
        2.78,
        0.62,
        "z(t) MLP -> 256 ch",
        COLORS["neural"],
        fontsize=8.0,
    )
    box(
        ax,
        7.35,
        6.05,
        2.00,
        0.72,
        "concat features\n64 + 64 + 256 = 384",
        COLORS["note"],
        fontsize=8.0,
        weight="bold",
    )
    box(
        ax,
        7.30,
        4.95,
        2.10,
        0.78,
        "causal sequence model\nGRU H=256\n(Mamba option)",
        COLORS["neural"],
        fontsize=7.8,
    )
    box(
        ax,
        7.20,
        3.72,
        2.30,
        0.90,
        "Out MLP\n[GRU, f0, loudness, z]\n-> hidden h(t): (B,T,256)",
        COLORS["neural"],
        fontsize=7.5,
    )

    # Input routing into encoder.
    arrow(ax, (3.65, 5.92), (4.10, 7.68), "f0")
    arrow(ax, (3.65, 5.56), (4.10, 6.76), "loudness")
    arrow(ax, (3.65, 5.16), (4.10, 5.55), "articulation + events")
    arrow(ax, (5.49, 5.06), (5.49, 4.76))
    arrow(ax, (6.88, 7.68), (7.35, 6.48))
    arrow(ax, (6.88, 6.76), (7.35, 6.38))
    arrow(ax, (6.88, 4.45), (7.35, 6.17))
    arrow(ax, (8.35, 6.05), (8.35, 5.73))
    arrow(ax, (8.35, 4.95), (8.35, 4.62))

    # Audio-rate control bus.
    box(
        ax,
        9.75,
        5.82,
        1.82,
        1.36,
        "audio-rate\ncontrol expansion\n\nupsample/interp\nnote_age + sample offset",
        COLORS["control"],
        fontsize=7.2,
    )
    box(
        ax,
        9.75,
        4.20,
        1.82,
        1.05,
        "branch modulators\n\nloudness_gain\nharmonic_gate\ntransient_window\nfixed branch_gain",
        COLORS["control"],
        fontsize=7.0,
    )
    arrow(ax, (9.50, 4.17), (9.75, 6.50), "h(t)", rad=0.10)
    arrow(ax, (3.65, 5.00), (9.75, 6.45), color="#555555", lw=0.85, rad=0.08)
    arrow(ax, (3.65, 5.00), (9.75, 4.75), color="#555555", lw=0.85, rad=-0.04)

    # Branches.
    box(
        ax,
        12.05,
        6.55,
        3.20,
        1.28,
        "Sustain branch: DWTS-style wavetable\n\n"
        "h(t) -> 16-table attention + amplitude\n"
        "f0_audio -> phase accumulator\n"
        "512-sample learned wavetable lookup",
        COLORS["dsp"],
        fontsize=7.3,
    )
    box(
        ax,
        12.05,
        5.46,
        3.20,
        0.66,
        "sustain *= gate * fade_in(age)\n"
        "* loudness_gain * harmonic_gate",
        COLORS["control"],
        fontsize=7.2,
        radius=0.025,
    )
    box(
        ax,
        12.05,
        3.86,
        3.20,
        1.04,
        "Noise branch: subtractive noise\n\n"
        "h(t) -> 65 band magnitudes -> IR\n"
        "white noise -> FFT convolution",
        COLORS["dsp"],
        fontsize=7.3,
    )
    box(
        ax,
        12.05,
        3.00,
        3.20,
        0.58,
        "noise *= gate * fixed branch_gain\n"
        "continuous residual noise path",
        COLORS["control"],
        fontsize=7.2,
        radius=0.025,
    )
    box(
        ax,
        12.05,
        1.52,
        3.20,
        1.10,
        "Transient branch: DCT-bank prototype\n\n"
        "articulation_id -> class DCT coeff row\n"
        "IDCT -> 300 ms style waveform; h(t) -> gain",
        COLORS["dsp"],
        fontsize=7.2,
    )
    box(
        ax,
        12.05,
        0.70,
        3.20,
        0.56,
        "transient *= gate * fixed window(note_age)\n"
        "* onset-strength velocity",
        COLORS["control"],
        fontsize=7.0,
        radius=0.025,
    )

    # Hidden/control buses into branches.
    line(ax, (11.72, 0.98), (11.72, 7.25), lw=1.0)
    arrow(ax, (9.50, 4.15), (11.72, 4.15), "h(t)")
    arrow(ax, (11.72, 7.25), (12.05, 7.25))
    arrow(ax, (11.72, 4.35), (12.05, 4.35))
    arrow(ax, (11.72, 2.08), (12.05, 2.08))
    arrow(ax, (11.57, 6.50), (12.05, 6.98), "controls", color="#555555", lw=0.85)
    arrow(ax, (11.57, 4.72), (12.05, 3.30), color="#555555", lw=0.85)
    arrow(ax, (11.57, 4.72), (12.05, 0.98), color="#555555", lw=0.85)
    arrow(ax, (13.65, 6.55), (13.65, 6.12))
    arrow(ax, (13.65, 3.86), (13.65, 3.58))
    arrow(ax, (13.65, 1.52), (13.65, 1.26))

    # Output side.
    box(
        ax,
        16.00,
        4.52,
        1.42,
        1.06,
        "sum\n\nsustain\n+ noise\n+ transient",
        COLORS["audio"],
        fontsize=8.0,
        weight="bold",
    )
    box(
        ax,
        15.88,
        3.45,
        1.66,
        0.58,
        "optional reverb\nOFF by default",
        COLORS["note"],
        fontsize=7.4,
    )
    box(
        ax,
        15.78,
        2.45,
        1.86,
        0.72,
        "Reconstruction\nx_hat: (B,N,1)",
        COLORS["audio"],
        fontsize=8.0,
        weight="bold",
    )
    box(
        ax,
        15.65,
        6.72,
        2.08,
        1.02,
        "Training loss\nvs x_target\n\nmulti-scale STFT\n+ log-frame-RMS",
        COLORS["loss"],
        fontsize=7.6,
        weight="bold",
    )
    box(
        ax,
        15.60,
        1.22,
        2.18,
        0.78,
        "Diagnostics\nbranch RMS, fixed gains,\nattention, gates, W&B",
        COLORS["note"],
        fontsize=7.4,
    )

    arrow(ax, (15.25, 5.78), (16.00, 5.26), "sustain")
    arrow(ax, (15.25, 3.29), (16.00, 5.05), "noise")
    arrow(ax, (15.25, 0.98), (16.00, 4.78), "transient", rad=0.03)
    arrow(ax, (16.71, 4.52), (16.71, 4.03))
    arrow(ax, (16.71, 3.45), (16.71, 3.17))
    arrow(ax, (16.71, 2.45), (16.71, 2.00))
    arrow(ax, (17.64, 2.82), (17.72, 6.72), color="#8a1c1c", rad=-0.20)

    # Legend.
    legend_y = 0.28
    legend = [
        ("data", COLORS["data"]),
        ("controls", COLORS["control"]),
        ("neural layers", COLORS["neural"]),
        ("differentiable DSP", COLORS["dsp"]),
        ("audio", COLORS["audio"]),
        ("loss", COLORS["loss"]),
    ]
    x = 0.45
    for label, color in legend:
        box(ax, x, legend_y, 0.50, 0.22, "", color, radius=0.008, lw=0.7)
        ax.text(
            x + 0.60,
            legend_y + 0.11,
            label,
            va="center",
            ha="left",
            fontsize=7.2,
            family="DejaVu Sans",
        )
        x += 1.75 if label != "differentiable DSP" else 2.25

    ax.text(
        12.10,
        0.25,
        "Source: /workspace/figures/draw_bass_ddsp_v2_overall_diagram.py; "
        "matches /workspace/bass_ddsp/model.py and configs/bass_ddsp_v2_riff.yaml.",
        ha="left",
        va="center",
        fontsize=6.8,
        family="DejaVu Sans",
        color="#555555",
    )

    for ext in ("svg", "pdf", "png"):
        path = OUT_DIR / f"{BASE_NAME}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    draw_diagram()
