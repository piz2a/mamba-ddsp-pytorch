from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bass_ddsp.dataset import IDMTBassNoteDataset
from onset_envelope_net.model import StructuredBassOnsetEnvelopeNet


CASES = {
    "articulation": (),
    "articulation_f0": (0,),
    "articulation_loudness": (1,),
    "articulation_periodicity": (2,),
    "articulation_f0_loudness": (0, 1),
    "articulation_f0_periodicity": (0, 2),
    "articulation_loudness_periodicity": (1, 2),
    "articulation_all": (0, 1, 2),
}
CONTROL_NAMES = ("f0", "loudness", "periodicity")


def collect_single_notes(dataset, count, envelope_frames, seed):
    rng = random.Random(seed)
    labels, controls, targets = [], [], []
    for example_index in range(count):
        sample = dataset.generate_debug_example(rng.randrange(len(dataset)), pitch_source="labels")
        strength = np.asarray(sample["onset_strength"], dtype=np.float32)
        pitch = np.asarray(sample["label_pitch"], dtype=np.float32)
        loudness = np.asarray(sample["loudness"], dtype=np.float32)
        periodicity = np.asarray(sample["periodicity"], dtype=np.float32)
        end_frame = int(np.ceil(float(sample["intervals"][0]["end_sample"]) / dataset.block_size))
        target = np.zeros(envelope_frames, dtype=np.float32)
        stop = min(envelope_frames, end_frame, len(strength))
        target[:stop] = strength[:stop]
        f0 = max(float(pitch[0]), 40.0)
        labels.append(int(sample["articulation"][0]))
        controls.append([
            np.log2(f0 / 40.0) / np.log2(600.0 / 40.0),
            float(loudness[0]),
            float(periodicity[0]),
        ])
        targets.append(target)
        if (example_index + 1) % 64 == 0 or example_index + 1 == count:
            print(f"preprocess single notes: {example_index + 1}/{count}", flush=True)
    controls = np.asarray(controls, dtype=np.float32)
    controls[:, 1] = (controls[:, 1] - controls[:, 1].mean()) / max(float(controls[:, 1].std()), 1e-6)
    return (torch.tensor(labels, dtype=torch.long), torch.tensor(controls, dtype=torch.float32),
            torch.tensor(np.asarray(targets), dtype=torch.float32))


def fit_case(case_name, control_indices, train, valid, num_articulations, args, run_dir, wandb_project):
    train_labels, train_controls, train_targets = train
    valid_labels, valid_controls, valid_targets = valid
    model = StructuredBassOnsetEnvelopeNet(num_articulations, len(control_indices), args.envelope_frames)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    train_loader = DataLoader(TensorDataset(train_labels, train_controls[:, list(control_indices)], train_targets),
                              batch_size=args.batch_size, shuffle=True)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=wandb_project, group="single-note-ablation",
                               name=f"{run_dir.name}_{case_name}", dir=str(run_dir),
                               config={"case": case_name, "controls": [CONTROL_NAMES[i] for i in control_indices], **vars(args)})
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    iterator = iter(train_loader)
    for step in range(args.steps):
        try:
            batch_labels, batch_controls, batch_targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch_labels, batch_controls, batch_targets = next(iterator)
        prediction = model(batch_labels, batch_controls)
        weights = torch.ones_like(batch_targets)
        weights[:, :3] = 2.0
        loss = ((prediction - batch_targets).square() * weights).mean()
        smoothness = (prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2]).square().mean()
        total = loss + 0.02 * smoothness
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                valid_prediction = model(valid_labels, valid_controls[:, list(control_indices)])
                valid_loss = float(torch.mean((valid_prediction - valid_targets).square()))
            record = {"step": step, "train_loss": float(total.detach()), "valid_mse": valid_loss}
            history.append(record)
            if wandb_run is not None:
                wandb_run.log({f"{k}": v for k, v in record.items()}, step=step)
            print(f"{case_name} step={step:06d} train={record['train_loss']:.6f} valid={valid_loss:.6f}", flush=True)
            if valid_loss < best_loss - args.min_delta:
                best_loss, best_state, stale = valid_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= args.patience_evals:
                    print(f"{case_name} early_stop step={step} best_valid={best_loss:.6f}", flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(valid_labels, valid_controls[:, list(control_indices)]).numpy()
    target = valid_targets.numpy()
    metrics = {
        "case": case_name,
        "controls": [CONTROL_NAMES[i] for i in control_indices],
        "best_valid_mse": float(np.mean((pred - target) ** 2)),
        "valid_mae": float(np.mean(np.abs(pred - target))),
        "first_3_frame_mse": float(np.mean((pred[:, :3] - target[:, :3]) ** 2)),
        "valid_correlation": float(np.mean([np.corrcoef(p, t)[0, 1] for p, t in zip(pred, target) if np.std(t) > 1e-8])),
    }
    torch.save({"model": model.state_dict(), "num_articulations": num_articulations,
                "envelope_frames": args.envelope_frames, "control_indices": list(control_indices),
                "control_names": [CONTROL_NAMES[i] for i in control_indices], "history": history}, run_dir / f"{case_name}.pt")
    (run_dir / f"{case_name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.summary.update(metrics)
        wandb_run.finish()
    return metrics, pred, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/disk1/ahnjiho/IDMT-SMT-BASS")
    parser.add_argument("--run-dir", default="/workspace/runs/bass_onset_single_note_ablation_v1")
    parser.add_argument("--examples", type=int, default=2048)
    parser.add_argument("--envelope-frames", type=int, default=20)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="bass-ddsp-onset-envelope")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset = IDMTBassNoteDataset(
        data_location=args.data, sampling_rate=16000, block_size=256, signal_length=32768,
        examples_per_epoch=args.examples, seed=args.seed, pitch_source="labels",
        label_mode="observed_articulation", include_expression_styles=("NO", "BE", "DN", "VI"),
    )
    labels, controls, targets = collect_single_notes(dataset, args.examples, args.envelope_frames, args.seed)
    split = int(len(labels) * 0.8)
    permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(args.seed))
    train_idx, valid_idx = permutation[:split], permutation[split:]
    train = (labels[train_idx], controls[train_idx], targets[train_idx])
    valid = (labels[valid_idx], controls[valid_idx], targets[valid_idx])
    torch.save({"train": train, "valid": valid, "articulation_labels": dataset.articulation_labels}, run_dir / "single_note_tensors.pt")
    all_metrics = []
    all_predictions = {}
    for case_name, control_indices in CASES.items():
        metrics, pred, target = fit_case(case_name, control_indices, train, valid, dataset.n_articulation, args, run_dir, args.wandb_project)
        all_metrics.append(metrics)
        all_predictions[case_name] = (pred, target)
    (run_dir / "ablation_metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    best = min(all_metrics, key=lambda item: item["best_valid_mse"])
    lines = ["# Single-Note Bass Onset Envelope Ablation", "", "Dataset: `IDMTBassNoteDataset` with trimmed, onset-aligned IDMT notes.", "", "| Case | Controls | MSE | MAE | First 3-frame MSE | Correlation |", "|---|---|---:|---:|---:|---:|"]
    for item in all_metrics:
        lines.append(f"| `{item['case']}` | {', '.join(item['controls']) or 'none'} | {item['best_valid_mse']:.6f} | {item['valid_mae']:.6f} | {item['first_3_frame_mse']:.6f} | {item['valid_correlation']:.4f} |")
    lines += ["", f"Best validation case: `{best['case']}` with MSE `{best['best_valid_mse']:.6f}`.", "", "The selected control set is determined by held-out MSE, not training loss alone. Early stopping is enabled per case."]
    (run_dir / "EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True, sharey=True, constrained_layout=True)
    time_ms = np.arange(args.envelope_frames) * 16
    for ax, (case_name, (pred, target)) in zip(axes.flat, all_predictions.items()):
        for row in target[:6]: ax.plot(time_ms, row, color="0.7", alpha=0.4)
        for row in pred[:6]: ax.plot(time_ms, row, linewidth=1.0)
        ax.set_title(case_name)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
    fig.savefig(run_dir / "eight_case_envelopes.png", dpi=160)
    plt.close(fig)
    print((run_dir / "EVALUATION_REPORT.md").read_text())


if __name__ == "__main__":
    main()
