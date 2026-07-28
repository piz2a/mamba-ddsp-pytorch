from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bass_ddsp.dataset import IDMTBassRiffDataset
from onset_envelope_net.model import BassOnsetEnvelopeNet


def collect_examples(dataset, count, envelope_frames, seed):
    rng = random.Random(seed)
    controls, labels, targets = [], [], []
    for _ in range(count):
        sample_index = rng.randrange(len(dataset))
        sample = dataset.generate_debug_riff(sample_index, pitch_source="labels")
        strength = np.asarray(sample["onset_strength"], dtype=np.float32)
        pitch = np.asarray(sample["label_pitch"], dtype=np.float32)
        loudness = np.asarray(sample["loudness"], dtype=np.float32)
        periodicity = np.asarray(sample["periodicity"], dtype=np.float32)
        for interval in sample["intervals"]:
            frame = int(np.floor(float(interval["start_sample"]) / dataset.block_size))
            frame = min(max(frame, 0), len(strength) - 1)
            end_frame = int(np.ceil(float(interval["end_sample"]) / dataset.block_size))
            target = np.zeros(envelope_frames, dtype=np.float32)
            stop = min(envelope_frames, max(0, end_frame - frame), len(strength) - frame)
            if stop:
                target[:stop] = strength[frame:frame + stop]
            f0 = max(float(pitch[frame]), 40.0)
            controls.append([np.log2(f0 / 40.0) / np.log2(600.0 / 40.0),
                             float(loudness[min(frame, len(loudness) - 1)]),
                             float(periodicity[min(frame, len(periodicity) - 1)])])
            labels.append(int(sample["articulation"][frame]))
            targets.append(target)
    controls = np.asarray(controls, dtype=np.float32)
    controls[:, 1] = (controls[:, 1] - controls[:, 1].mean()) / max(float(controls[:, 1].std()), 1e-6)
    return (torch.tensor(labels, dtype=torch.long),
            torch.tensor(controls, dtype=torch.float32),
            torch.tensor(np.asarray(targets), dtype=torch.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/disk1/ahnjiho/IDMT-SMT-BASS")
    parser.add_argument("--run-dir", default="/workspace/runs/bass_onset_envelope_v1")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--envelope-frames", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="bass-ddsp-onset-envelope")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=run_dir.name,
                               dir=str(run_dir), config=vars(args))
    dataset = IDMTBassRiffDataset(
        data_location=args.data, sampling_rate=16000, block_size=256,
        signal_length=32768, examples_per_epoch=2048, seed=args.seed,
        pitch_source="labels", label_mode="observed_articulation",
        include_expression_styles=("NO", "BE", "DN", "VI"),
    )
    labels, controls, targets = collect_examples(dataset, args.examples, args.envelope_frames, args.seed)
    torch.save({"labels": labels, "controls": controls, "targets": targets}, run_dir / "training_tensors.pt")
    model = BassOnsetEnvelopeNet(dataset.n_articulation, args.envelope_frames)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(labels, controls, targets), batch_size=args.batch_size, shuffle=True, drop_last=False)
    iterator = iter(loader)
    history = []
    for step in range(args.steps):
        try:
            batch_labels, batch_controls, batch_targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch_labels, batch_controls, batch_targets = next(iterator)
        prediction = model(batch_labels, batch_controls)
        weights = torch.ones_like(batch_targets)
        weights[:, :3] = 2.0
        loss = ((prediction - batch_targets).square() * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0 or step == args.steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
            if wandb_run is not None:
                wandb_run.log({"train/loss": float(loss.detach())}, step=step)
            print(f"step={step:06d} loss={float(loss.detach()):.6f}", flush=True)
    torch.save({"model": model.state_dict(), "num_articulations": dataset.n_articulation,
                "envelope_frames": args.envelope_frames, "articulation_labels": dataset.articulation_labels,
                "history": history}, run_dir / "checkpoint.pt")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"saved: {run_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
