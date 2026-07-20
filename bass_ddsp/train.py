import argparse
import csv
import itertools
from os import listdir, makedirs, path

import numpy as np
import soundfile as sf
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from bass_ddsp.dataset import IDMTBassNoteDataset, IDMTBassRiffDataset
from bass_ddsp.model import BassDDSPV2
from ddsp.core import mean_std_loudness, multiscale_fft, safe_log


def frame_log_rms(signal, block_size):
    usable = signal.shape[-1] - (signal.shape[-1] % block_size)
    signal = signal[..., :usable]
    frames = signal.reshape(signal.shape[0], -1, block_size)
    rms = torch.sqrt(torch.mean(frames * frames, dim=-1) + 1e-7)
    return torch.log(rms + 1e-7)


def multiscale_spectral_loss(target, reconstruction, scales, overlap):
    target_stft = multiscale_fft(target, scales, overlap)
    reconstruction_stft = multiscale_fft(reconstruction, scales, overlap)
    loss = 0
    for s_x, s_y in zip(target_stft, reconstruction_stft):
        loss = loss + (s_x - s_y).abs().mean()
        loss = loss + (safe_log(s_x) - safe_log(s_y)).abs().mean()
    return loss


def branch_rms(model, name, device):
    branch = getattr(model, "last_branch_outputs", {}).get(name)
    if branch is None:
        return torch.tensor(0.0, device=device)
    return torch.sqrt(torch.mean(branch.detach() * branch.detach()) + 1e-7)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bass_ddsp_v2_riff.yaml")
    parser.add_argument("--name", default="debug")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--start-lr", type=float, default=1e-3)
    parser.add_argument("--stop-lr", type=float, default=1e-4)
    parser.add_argument("--decay-over", type=int, default=400000)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--init-state", help="Optional checkpoint to load before training.")
    parser.add_argument("--wandb", action="store_true", help="Log scalar metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="bass-ddsp-v2")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


def maybe_init_wandb(args, config, run_dir):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Install it or run without --wandb."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or args.name,
        dir=run_dir,
        config=config,
    )


def make_dataset(config):
    data_config = config.get("data", {})
    dataset_type = data_config.get("dataset", "idmt_bass_riff")
    dataset_cls = {
        "idmt_bass_note": IDMTBassNoteDataset,
        "idmt_bass_riff": IDMTBassRiffDataset,
    }.get(dataset_type)
    if dataset_cls is None:
        raise ValueError(
            "data.dataset must be 'idmt_bass_note' or 'idmt_bass_riff', "
            f"got {dataset_type!r}"
        )
    return dataset_cls(
        data_location=data_config["data_location"],
        sampling_rate=config["preprocess"]["sampling_rate"],
        block_size=config["preprocess"]["block_size"],
        signal_length=config["preprocess"]["signal_length"],
        **config.get("idmt_bass", {}),
    )


def set_lr(opt, start_lr, stop_lr, decay_over, step):
    if decay_over <= 0:
        lr = stop_lr
    else:
        ratio = min(1.0, step / float(decay_over))
        lr = start_lr * ((stop_lr / start_lr) ** ratio)
    for group in opt.param_groups:
        group["lr"] = lr
    return lr


def unpack_batch(batch, device):
    if len(batch) != 9:
        raise ValueError(
            "BassDDSPV2 expects 9 tensors: audio, pitch, loudness, articulation, "
            f"onset_strength, offset, gate, note_age, periodicity. Got {len(batch)}."
        )
    (
        audio,
        pitch,
        loudness,
        articulation,
        onset_strength,
        offset,
        gate,
        note_age,
        periodicity,
    ) = batch
    return {
        "audio": audio.to(device),
        "pitch": pitch.unsqueeze(-1).to(device),
        "loudness": loudness.unsqueeze(-1).to(device),
        "articulation": articulation.to(device),
        "onset_strength": onset_strength.unsqueeze(-1).to(device),
        "offset": offset.unsqueeze(-1).to(device),
        "gate": gate.unsqueeze(-1).to(device),
        "note_age": note_age.unsqueeze(-1).to(device),
        "periodicity": periodicity.unsqueeze(-1).to(device),
    }


def main():
    args = parse_args()
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    dataset = make_dataset(config)
    if len(dataset) < args.batch:
        raise ValueError(f"dataset length {len(dataset)} is smaller than batch {args.batch}")

    config["model"]["n_articulation"] = dataset.n_articulation
    config["data"]["pluck_labels"] = list(dataset.pluck_labels)
    config["data"]["expression_labels"] = list(dataset.expression_labels)
    config["data"]["articulation_labels"] = list(dataset.articulation_labels)

    model = BassDDSPV2(**config["model"]).to(device)
    if args.init_state:
        state = torch.load(args.init_state, map_location=device)
        model.load_state_dict(state)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=config["train"].get("num_workers", 0),
    )

    stats_batches = config["train"].get("loudness_stats_batches")
    if stats_batches:
        stats_loader = torch.utils.data.DataLoader(
            dataset,
            args.batch,
            shuffle=True,
            drop_last=True,
            num_workers=config["train"].get("num_workers", 0),
        )
        mean_loudness, std_loudness = mean_std_loudness(
            itertools.islice(stats_loader, int(stats_batches))
        )
    else:
        mean_loudness, std_loudness = mean_std_loudness(dataloader)
    std_loudness = max(float(std_loudness), 1e-8)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

    run_dir = path.join(args.root, args.name)
    if path.isdir(run_dir) and listdir(run_dir) and not args.overwrite:
        raise FileExistsError(
            f"run directory already exists and is not empty: {run_dir}. "
            "Use a new --name or pass --overwrite intentionally."
        )
    makedirs(run_dir, exist_ok=True)
    with open(path.join(run_dir, "config.yaml"), "w") as out_config:
        yaml.safe_dump(config, out_config)

    writer = SummaryWriter(run_dir, flush_secs=20)
    wandb_run = maybe_init_wandb(args, config, run_dir)
    opt = torch.optim.Adam(model.parameters(), lr=args.start_lr)
    loss_csv = open(path.join(run_dir, "loss.csv"), "w", newline="")
    loss_writer = csv.writer(loss_csv)
    loss_writer.writerow([
        "step",
        "loss",
        "spectral_loss",
        "rms_loss",
        "onset_spectral_loss",
        "transient_loss",
        "transient_branch_loss",
        "target_rms",
        "reconstruction_rms",
        "transient_branch_rms",
        "sustain_branch_rms",
        "noise_branch_rms",
        "sustain_gain_db",
        "noise_gain_db",
        "transient_gain_db",
        "lr",
    ])

    best_loss = float("inf")
    mean_loss = 0.0
    n_element = 0
    step = 0
    epochs = int(np.ceil(args.steps / len(dataloader)))

    for epoch in tqdm(range(epochs)):
        for batch in dataloader:
            batch = unpack_batch(batch, device)
            loudness = (batch["loudness"] - mean_loudness) / std_loudness
            y = model(
                batch["pitch"],
                loudness,
                articulation=batch["articulation"],
                onset_strength=batch["onset_strength"],
                offset=batch["offset"],
                gate=batch["gate"],
                note_age=batch["note_age"],
                periodicity=batch["periodicity"],
            ).squeeze(-1)
            target = batch["audio"]

            spectral_loss = multiscale_spectral_loss(
                target,
                y,
                config["train"]["scales"],
                config["train"]["overlap"],
            )
            rms_loss = torch.tensor(0.0, device=device)
            rms_loss_weight = float(config["train"].get("rms_loss_weight", 0.0))
            if rms_loss_weight > 0.0:
                block_size = config["preprocess"]["block_size"]
                rms_loss = (
                    frame_log_rms(target, block_size)
                    - frame_log_rms(y, block_size)
                ).abs().mean()
            onset_spectral_loss = torch.tensor(0.0, device=device)
            transient_loss = torch.tensor(0.0, device=device)
            transient_branch_loss = torch.tensor(0.0, device=device)
            loss = spectral_loss + rms_loss_weight * rms_loss

            lr = set_lr(opt, args.start_lr, args.stop_lr, args.decay_over, step)
            opt.zero_grad()
            loss.backward()
            grad_clip_norm = config["train"].get("grad_clip_norm")
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            opt.step()

            target_rms = torch.sqrt(torch.mean(target * target) + 1e-7)
            reconstruction_rms = torch.sqrt(torch.mean(y * y) + 1e-7)
            transient_branch_rms = branch_rms(model, "transient", device)
            sustain_branch_rms = branch_rms(model, "sustain", device)
            noise_branch_rms = branch_rms(model, "noise", device)
            branch_gain_db = model.branch_gain_db()

            metrics = {
                "loss": loss.item(),
                "spectral_loss": spectral_loss.item(),
                "rms_loss": rms_loss.item(),
                "onset_spectral_loss": onset_spectral_loss.item(),
                "transient_loss": transient_loss.item(),
                "transient_branch_loss": transient_branch_loss.item(),
                "target_rms": target_rms.item(),
                "reconstruction_rms": reconstruction_rms.item(),
                "transient_branch_rms": transient_branch_rms.item(),
                "sustain_branch_rms": sustain_branch_rms.item(),
                "noise_branch_rms": noise_branch_rms.item(),
                "lr": lr,
                **branch_gain_db,
            }
            for key, value in metrics.items():
                writer.add_scalar(key, value, step)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            loss_writer.writerow([
                step,
                metrics["loss"],
                metrics["spectral_loss"],
                metrics["rms_loss"],
                metrics["onset_spectral_loss"],
                metrics["transient_loss"],
                metrics["transient_branch_loss"],
                metrics["target_rms"],
                metrics["reconstruction_rms"],
                metrics["transient_branch_rms"],
                metrics["sustain_branch_rms"],
                metrics["noise_branch_rms"],
                metrics["sustain_gain_db"],
                metrics["noise_gain_db"],
                metrics["transient_gain_db"],
                metrics["lr"],
            ])
            loss_csv.flush()

            step += 1
            n_element += 1
            mean_loss += (loss.item() - mean_loss) / n_element
            if step >= args.steps:
                break

        should_finish = step >= args.steps
        eval_every_epochs = int(config["train"].get("eval_every_epochs", 10))
        if (not epoch % eval_every_epochs) or should_finish:
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), path.join(run_dir, "state.pth"))
            mean_loss = 0.0
            n_element = 0
            audio = torch.cat([target, y], dim=-1).reshape(-1).detach().cpu().numpy()
            sf.write(
                path.join(run_dir, f"eval_{epoch:06d}.wav"),
                audio,
                config["preprocess"]["sampling_rate"],
            )
        if should_finish:
            break

    loss_csv.close()
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
