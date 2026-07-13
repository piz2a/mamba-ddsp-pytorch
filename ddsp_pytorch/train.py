import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSP
from effortless_config import Config
from os import listdir, makedirs, path
import itertools
import csv
from idmt_bass import IDMTBassNoteDataset, IDMTBassRiffDataset
from preprocess import Dataset as PreprocessedDataset
from tqdm import tqdm
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from ddsp.utils import get_scheduler
import numpy as np


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
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss
    return loss


def onset_frame_mask(onset, window_seconds, sampling_rate, block_size):
    if onset is None or window_seconds <= 0:
        return None

    onset_frames = onset.squeeze(-1)
    window_frames = max(1, int(np.ceil(window_seconds * sampling_rate / block_size)))
    mask = torch.zeros_like(onset_frames)
    for batch_idx in range(onset_frames.shape[0]):
        starts = torch.nonzero(onset_frames[batch_idx] > 0.05, as_tuple=False)
        for start in starts.flatten():
            start_idx = int(start.item())
            end_idx = min(mask.shape[1], start_idx + window_frames)
            mask[batch_idx, start_idx:end_idx] = 1.0
    return mask


def frame_mask_to_audio(mask, block_size, length):
    if mask is None:
        return None
    audio_mask = mask.repeat_interleave(block_size, dim=1)
    if audio_mask.shape[-1] < length:
        audio_mask = torch.nn.functional.pad(
            audio_mask,
            (0, length - audio_mask.shape[-1]),
        )
    return audio_mask[:, :length]


def highpass_transient_loss(target, reconstruction, frame_mask, block_size):
    audio_mask = frame_mask_to_audio(frame_mask, block_size, target.shape[-1])
    if audio_mask is None or not torch.any(audio_mask > 0):
        return torch.tensor(0.0, device=target.device)

    target_hp = target[:, 1:] - target[:, :-1]
    reconstruction_hp = reconstruction[:, 1:] - reconstruction[:, :-1]
    return ((target_hp - reconstruction_hp).abs() * audio_mask[:, 1:]).mean()


def branch_rms(model, name, device):
    branches = getattr(model, "last_branch_outputs", {})
    branch = branches.get(name)
    if branch is None:
        return torch.tensor(0.0, device=device)
    return torch.sqrt(torch.mean(branch.detach() * branch.detach()) + 1e-7)


class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    STEPS = 500000
    BATCH = 16
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000
    DEVICE = None
    OVERWRITE = False


args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

device = torch.device(
    args.DEVICE
    if args.DEVICE is not None
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
)


def make_dataset(config):
    data_config = config.get("data", {})
    dataset_type = data_config.get("dataset", "preprocessed")
    if dataset_type == "idmt_bass_note":
        return IDMTBassNoteDataset(
            data_location=data_config["data_location"],
            sampling_rate=config["preprocess"]["sampling_rate"],
            block_size=config["preprocess"]["block_size"],
            signal_length=config["preprocess"]["signal_length"],
            **config.get("idmt_bass", {}),
        )
    if dataset_type == "idmt_bass_riff":
        return IDMTBassRiffDataset(
            data_location=data_config["data_location"],
            sampling_rate=config["preprocess"]["sampling_rate"],
            block_size=config["preprocess"]["block_size"],
            signal_length=config["preprocess"]["signal_length"],
            **config.get("idmt_bass", {}),
        )
    if dataset_type == "preprocessed":
        return PreprocessedDataset(config["preprocess"]["out_dir"])

    raise ValueError(
        "data.dataset must be 'preprocessed', 'idmt_bass_note', or "
        "'idmt_bass_riff', "
        f"got {dataset_type!r}"
    )


dataset = make_dataset(config)
if len(dataset) < args.BATCH:
    raise ValueError(
        f"dataset has {len(dataset)} examples, but batch size is "
        f"{args.BATCH}. Add more audio, lower --batch, or set oneshot: false "
        "with longer source files before training."
    )

if isinstance(dataset, (IDMTBassRiffDataset, IDMTBassNoteDataset)):
    config["model"]["n_pluck"] = dataset.n_pluck
    config["model"]["n_expression"] = dataset.n_expression
    config["model"]["n_articulation"] = dataset.n_articulation
    config["data"]["pluck_labels"] = list(dataset.pluck_labels)
    config["data"]["expression_labels"] = list(dataset.expression_labels)
    config["data"]["articulation_labels"] = list(dataset.articulation_labels)

model = DDSP(**config["model"]).to(device)

dataloader = torch.utils.data.DataLoader(
    dataset,
    args.BATCH,
    True,
    drop_last=True,
    num_workers=config["train"].get("num_workers", 0),
)

stats_batches = config["train"].get("loudness_stats_batches")
if stats_batches:
    stats_loader = torch.utils.data.DataLoader(
        dataset,
        args.BATCH,
        True,
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

run_dir = path.join(args.ROOT, args.NAME)
if path.isdir(run_dir) and listdir(run_dir) and not args.OVERWRITE:
    raise FileExistsError(
        f"run directory already exists and is not empty: {run_dir}. "
        "Use a new --name, or pass --overwrite true intentionally."
    )
makedirs(run_dir, exist_ok=True)
writer = SummaryWriter(run_dir, flush_secs=20)
loss_csv_path = path.join(run_dir, "loss.csv")
loss_csv = open(loss_csv_path, "w", newline="")
loss_writer = csv.writer(loss_csv)
loss_writer.writerow([
    "step",
    "loss",
    "spectral_loss",
    "rms_loss",
    "onset_spectral_loss",
    "transient_loss",
    "target_rms",
    "reconstruction_rms",
    "transient_branch_rms",
    "sustain_branch_rms",
    "noise_branch_rms",
])

with open(path.join(run_dir, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

schedule = get_scheduler(
    len(dataloader),
    args.START_LR,
    args.STOP_LR,
    args.DECAY_OVER,
)

# scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

best_loss = float("inf")
mean_loss = 0
n_element = 0
step = 0
epochs = int(np.ceil(args.STEPS / len(dataloader)))

for e in tqdm(range(epochs)):
    for batch in dataloader:
        pluck = None
        expression = None
        articulation = None
        onset = None
        offset = None
        gate = None
        note_age = None
        note_progress = None

        if len(batch) == 3:
            s, p, l = batch
        elif len(batch) == 4:
            s, p, l, articulation = batch
            articulation = articulation.to(device)
        elif len(batch) == 5:
            s, p, l, pluck, expression = batch
            pluck = pluck.to(device)
            expression = expression.to(device)
        elif len(batch) == 7:
            s, p, l, pluck, expression, onset, offset = batch
            pluck = pluck.to(device)
            expression = expression.to(device)
            onset = onset.unsqueeze(-1).to(device)
            offset = offset.unsqueeze(-1).to(device)
        elif len(batch) == 9:
            (
                s,
                p,
                l,
                articulation,
                onset,
                offset,
                gate,
                note_age,
                note_progress,
            ) = batch
            articulation = articulation.to(device)
            onset = onset.unsqueeze(-1).to(device)
            offset = offset.unsqueeze(-1).to(device)
            gate = gate.unsqueeze(-1).to(device)
            note_age = note_age.unsqueeze(-1).to(device)
            note_progress = note_progress.unsqueeze(-1).to(device)
        else:
            raise ValueError(f"unexpected batch with {len(batch)} tensors")

        s = s.to(device)
        p = p.unsqueeze(-1).to(device)
        l = l.unsqueeze(-1).to(device)

        l = (l - mean_loudness) / std_loudness

        y = model(
            p,
            l,
            pluck,
            expression,
            onset,
            offset,
            articulation=articulation,
            gate=gate,
            note_age=note_age,
            note_progress=note_progress,
        ).squeeze(-1)

        spectral_loss = multiscale_spectral_loss(
            s,
            y,
            config["train"]["scales"],
            config["train"]["overlap"],
        )

        rms_loss = torch.tensor(0.0, device=device)
        rms_loss_weight = float(config["train"].get("rms_loss_weight", 0.0))
        if rms_loss_weight:
            rms_loss = (
                frame_log_rms(s, config["preprocess"]["block_size"])
                - frame_log_rms(y, config["preprocess"]["block_size"])
            ).abs().mean()

        onset_spectral_loss = torch.tensor(0.0, device=device)
        transient_loss = torch.tensor(0.0, device=device)
        onset_loss_weight = float(config["train"].get("onset_loss_weight", 0.0))
        transient_loss_weight = float(
            config["train"].get("transient_loss_weight", 0.0)
        )
        onset_mask = None
        if onset_loss_weight or transient_loss_weight:
            onset_mask = onset_frame_mask(
                onset,
                float(config["train"].get("onset_loss_seconds", 0.15)),
                config["preprocess"]["sampling_rate"],
                config["preprocess"]["block_size"],
            )
        if onset_loss_weight and onset_mask is not None:
            audio_mask = frame_mask_to_audio(
                onset_mask,
                config["preprocess"]["block_size"],
                s.shape[-1],
            )
            onset_spectral_loss = multiscale_spectral_loss(
                s * audio_mask,
                y * audio_mask,
                config["train"]["scales"],
                config["train"]["overlap"],
            )
        if transient_loss_weight and onset_mask is not None:
            transient_loss = highpass_transient_loss(
                s,
                y,
                onset_mask,
                config["preprocess"]["block_size"],
            )

        loss = (
            spectral_loss
            + rms_loss_weight * rms_loss
            + onset_loss_weight * onset_spectral_loss
            + transient_loss_weight * transient_loss
        )

        opt.zero_grad()
        loss.backward()
        grad_clip_norm = config["train"].get("grad_clip_norm")
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(grad_clip_norm),
            )
        opt.step()

        target_rms = torch.sqrt(torch.mean(s * s) + 1e-7)
        reconstruction_rms = torch.sqrt(torch.mean(y * y) + 1e-7)

        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("spectral_loss", spectral_loss.item(), step)
        writer.add_scalar("rms_loss", rms_loss.item(), step)
        writer.add_scalar("onset_spectral_loss", onset_spectral_loss.item(), step)
        writer.add_scalar("transient_loss", transient_loss.item(), step)
        writer.add_scalar("target_rms", target_rms.item(), step)
        writer.add_scalar("reconstruction_rms", reconstruction_rms.item(), step)
        transient_branch_rms = branch_rms(model, "transient", device)
        sustain_branch_rms = branch_rms(model, "sustain", device)
        noise_branch_rms = branch_rms(model, "noise", device)
        writer.add_scalar("transient_branch_rms", transient_branch_rms.item(), step)
        writer.add_scalar("sustain_branch_rms", sustain_branch_rms.item(), step)
        writer.add_scalar("noise_branch_rms", noise_branch_rms.item(), step)
        loss_writer.writerow([
            step,
            loss.item(),
            spectral_loss.item(),
            rms_loss.item(),
            onset_spectral_loss.item(),
            transient_loss.item(),
            target_rms.item(),
            reconstruction_rms.item(),
            transient_branch_rms.item(),
            sustain_branch_rms.item(),
            noise_branch_rms.item(),
        ])
        loss_csv.flush()

        step += 1

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element

        if step >= args.STEPS:
            break

    should_finish = step >= args.STEPS
    eval_every_epochs = int(config["train"].get("eval_every_epochs", 10))
    if (not e % eval_every_epochs) or should_finish:
        writer.add_scalar("lr", schedule(step), step)
        writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
        writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)
        # scheduler.step()
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                model.state_dict(),
                path.join(run_dir, "state.pth"),
            )

        mean_loss = 0
        n_element = 0

        audio = torch.cat([s, y], -1).reshape(-1).detach().cpu().numpy()

        sf.write(
            path.join(run_dir, f"eval_{e:06d}.wav"),
            audio,
            config["preprocess"]["sampling_rate"],
        )

    if should_finish:
        break

loss_csv.close()
