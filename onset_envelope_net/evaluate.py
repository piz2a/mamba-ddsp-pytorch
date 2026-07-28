from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bass_ddsp.dataset import IDMTBassRiffDataset
from onset_envelope_net.model import BassOnsetEnvelopeNet
from onset_envelope_net.train import collect_examples


def pearson_rows(pred, target):
    values = []
    for p, t in zip(pred, target):
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            continue
        values.append(float(np.corrcoef(p, t)[0, 1]))
    return float(np.mean(values)) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="/workspace/runs/bass_onset_envelope_v1")
    parser.add_argument("--data", default="/disk1/ahnjiho/IDMT-SMT-BASS")
    parser.add_argument("--examples", type=int, default=128)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    dataset = IDMTBassRiffDataset(
        data_location=args.data, sampling_rate=16000, block_size=256,
        signal_length=32768, examples_per_epoch=2048, seed=9102026,
        pitch_source="labels", label_mode="observed_articulation",
        include_expression_styles=("NO", "BE", "DN", "VI"),
    )
    labels, controls, targets = collect_examples(dataset, args.examples, checkpoint["envelope_frames"], 9102026)
    model = BassOnsetEnvelopeNet(checkpoint["num_articulations"], checkpoint["envelope_frames"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        prediction = model(labels, controls).numpy()
        template_only = torch.sigmoid(model.base_template_logits(labels)).numpy()
    target = targets.numpy()
    train_tensors = torch.load(run_dir / "training_tensors.pt", map_location="cpu")
    template_baseline = train_tensors["targets"].numpy()
    baseline_by_style = np.zeros((checkpoint["num_articulations"], target.shape[1]), dtype=np.float32)
    for style in range(checkpoint["num_articulations"]):
        rows = train_tensors["targets"].numpy()[train_tensors["labels"].numpy() == style]
        baseline_by_style[style] = rows.mean(axis=0) if len(rows) else 0.0
    baseline = baseline_by_style[labels.numpy()]
    metrics = {
        "heldout_examples": int(len(target)),
        "envelope_frames": int(target.shape[1]),
        "model_mse": float(np.mean((prediction - target) ** 2)),
        "model_mae": float(np.mean(np.abs(prediction - target))),
        "model_first_3_frame_mse": float(np.mean((prediction[:, :3] - target[:, :3]) ** 2)),
        "model_row_correlation": pearson_rows(prediction, target),
        "template_only_mse": float(np.mean((template_only - target) ** 2)),
        "mean_template_baseline_mse": float(np.mean((baseline - target) ** 2)),
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
    }
    (run_dir / "evaluation_metrics.json").write_text(__import__("json").dumps(metrics, indent=2) + "\n")

    steps = np.arange(target.shape[1]) * 16
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for style, name in enumerate(checkpoint["articulation_labels"]):
        axes[0].plot(steps, torch.sigmoid(model.base_template_logits.weight[style]).detach().numpy(), label=name)
    axes[0].set_title("Learned articulation lookup templates")
    axes[0].set_xlabel("milliseconds after onset")
    axes[0].set_ylabel("onset strength")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.2)
    for i in range(min(6, len(target))):
        axes[1].plot(steps, target[i], color="0.65", alpha=0.55)
        axes[1].plot(steps, prediction[i], linewidth=1.2, label=checkpoint["articulation_labels"][int(labels[i])])
    axes[1].set_title("Held-out target and predicted onset envelopes")
    axes[1].set_xlabel("milliseconds after onset")
    axes[1].set_ylabel("onset strength")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.2)
    fig.savefig(run_dir / "lookup_table_and_predictions.png", dpi=160)
    plt.close(fig)

    report = [
        "# Bass Onset-Strength Envelope Evaluation",
        "",
        "The model was trained on 512 generated IDMT riffs for 20,000 steps. Evaluation uses 128 newly generated riffs with a different seed.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    report.extend(f"| `{key}` | {value:.6f} |" if isinstance(value, float) else f"| `{key}` | {value} |" for key, value in metrics.items())
    report += [
        "",
        "## Interpretation",
        "",
        "The lookup-only baseline measures how much articulation alone explains the attack shape. The full model adds onset-time F0, loudness, and periodicity modulation. `model_first_3_frame_mse` checks the sharpest attack region, while row correlation checks envelope shape rather than only average amplitude.",
        "",
        "The figure `lookup_table_and_predictions.png` shows the learned articulation templates and held-out predictions. This is an envelope-learning smoke test, not evidence that the vocal classifier or offset detector is complete.",
    ]
    (run_dir / "EVALUATION_REPORT.md").write_text("\n".join(report) + "\n")
    print((run_dir / "EVALUATION_REPORT.md").read_text())
    print(f"saved: {run_dir / 'lookup_table_and_predictions.png'}")


if __name__ == "__main__":
    main()
