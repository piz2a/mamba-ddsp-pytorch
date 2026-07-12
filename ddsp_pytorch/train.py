import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSP
from effortless_config import Config
from os import makedirs, path
import itertools
import csv
from idmt_bass import IDMTBassRiffDataset
from preprocess import Dataset as PreprocessedDataset
from tqdm import tqdm
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from ddsp.utils import get_scheduler
import numpy as np


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
        "data.dataset must be 'preprocessed' or 'idmt_bass_riff', "
        f"got {dataset_type!r}"
    )


dataset = make_dataset(config)
if len(dataset) < args.BATCH:
    raise ValueError(
        f"dataset has {len(dataset)} examples, but batch size is "
        f"{args.BATCH}. Add more audio, lower --batch, or set oneshot: false "
        "with longer source files before training."
    )

if isinstance(dataset, IDMTBassRiffDataset):
    config["model"]["n_pluck"] = dataset.n_pluck
    config["model"]["n_expression"] = dataset.n_expression
    config["data"]["pluck_labels"] = list(dataset.pluck_labels)
    config["data"]["expression_labels"] = list(dataset.expression_labels)

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
makedirs(run_dir, exist_ok=True)
writer = SummaryWriter(run_dir, flush_secs=20)
loss_csv_path = path.join(run_dir, "loss.csv")
loss_csv = open(loss_csv_path, "w", newline="")
loss_writer = csv.writer(loss_csv)
loss_writer.writerow(["step", "loss"])

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
        if len(batch) == 3:
            s, p, l = batch
            pluck = None
            expression = None
        elif len(batch) == 5:
            s, p, l, pluck, expression = batch
            pluck = pluck.to(device)
            expression = expression.to(device)
        else:
            raise ValueError(f"unexpected batch with {len(batch)} tensors")

        s = s.to(device)
        p = p.unsqueeze(-1).to(device)
        l = l.unsqueeze(-1).to(device)

        l = (l - mean_loudness) / std_loudness

        y = model(p, l, pluck, expression).squeeze(-1)

        ori_stft = multiscale_fft(
            s,
            config["train"]["scales"],
            config["train"]["overlap"],
        )
        rec_stft = multiscale_fft(
            y,
            config["train"]["scales"],
            config["train"]["overlap"],
        )

        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            lin_loss = (s_x - s_y).abs().mean()
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss = loss + lin_loss + log_loss

        opt.zero_grad()
        loss.backward()
        grad_clip_norm = config["train"].get("grad_clip_norm")
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(grad_clip_norm),
            )
        opt.step()

        writer.add_scalar("loss", loss.item(), step)
        loss_writer.writerow([step, loss.item()])
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
