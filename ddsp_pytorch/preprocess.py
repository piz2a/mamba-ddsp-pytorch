import yaml
import pathlib
import librosa as li
from ddsp.core import extract_loudness, extract_pitch
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
from scipy.io import wavfile


def get_files(data_location, extension, **kwargs):
    data_path = pathlib.Path(data_location).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(
            f"data.data_location does not exist: {data_path}"
        )

    extension = extension.lstrip(".")
    files = sorted(data_path.rglob(f"*.{extension}"))
    if not files:
        raise FileNotFoundError(
            f"found no '*.{extension}' files under data.data_location: "
            f"{data_path}"
        )

    return files


def preprocess(f, sampling_rate, block_size, signal_length, oneshot, **kwargs):
    x, sr = li.load(f, sr=sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)
    
    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        files = {
            "signals": path.join(out_dir, "signals.npy"),
            "pitchs": path.join(out_dir, "pitchs.npy"),
            "loudness": path.join(out_dir, "loudness.npy"),
        }
        missing = [file_path for file_path in files.values()
                   if not path.exists(file_path)]
        if missing:
            raise FileNotFoundError(
                "missing preprocessed dataset files: "
                f"{', '.join(missing)}. Run `python preprocess.py` after "
                "setting data.data_location in config.yaml to a directory "
                "containing audio files."
            )

        self.signals = np.load(files["signals"])
        self.pitchs = np.load(files["pitchs"])
        self.loudness = np.load(files["loudness"])

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s, p, l


def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    files = get_files(**config["data"])
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []

    for f in pb:
        pb.set_description(str(f))
        x, p, l = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)


if __name__ == "__main__":
    main()
