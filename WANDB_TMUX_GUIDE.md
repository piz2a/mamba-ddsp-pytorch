# W&B and Long-Running Training Guide

## What W&B Does In This Project

W&B is useful here for:

- remote metric dashboards: loss, spectral loss, RMS loss, transient loss, branch RMS, learning rate;
- remote audio inspection: eval WAVs are logged as W&B audio panels;
- config snapshots: training config and CLI arguments are stored with the run;
- run comparison: compare experiments such as v1 riff training vs Bass-DDSP v2 single-note training;
- optional model artifacts: save `state.pth` checkpoints to W&B when `--wandb-log-model true` is passed.

W&B does not keep training alive by itself. It is a logging/control service, not compute hosting. If the Docker container stops, training stops. If you only disconnect SSH or close VS Code while the container keeps running, `tmux` can keep the training process alive.

## One-Time W&B Login

From inside the Docker container:

```bash
python -m pip install wandb
wandb login
```

Paste your API key from:

```text
https://wandb.ai/authorize
```

Check login:

```bash
wandb status
```

## Start Training With W&B

Use a new run folder name every time unless you intentionally pass `--overwrite true`.

```bash
cd /workspace/ddsp_pytorch
python train.py \
  --config config_idmt_bass_v2_single_note.yaml \
  --name idmt_bass_v2_single_note_wandb_001 \
  --steps 10000 \
  --batch 4 \
  --wandb true \
  --wandb-project bass-ddsp \
  --wandb-tags idmt,bass-ddsp-v2,single-note
```

Optional checkpoint artifact upload:

```bash
python train.py \
  --config config_idmt_bass_v2_single_note.yaml \
  --name idmt_bass_v2_single_note_wandb_001 \
  --steps 10000 \
  --batch 4 \
  --wandb true \
  --wandb-project bass-ddsp \
  --wandb-log-model true
```

Artifacts can be large. Keep `--wandb-log-model false` for fast experiments unless you specifically want remote checkpoint storage.

## Run Training Through tmux

Start a persistent terminal:

```bash
tmux new -s bass_train
```

Run training inside tmux:

```bash
cd /workspace/ddsp_pytorch
python train.py \
  --config config_idmt_bass_v2_single_note.yaml \
  --name idmt_bass_v2_single_note_wandb_001 \
  --steps 10000 \
  --batch 4 \
  --wandb true \
  --wandb-project bass-ddsp \
  --wandb-tags idmt,bass-ddsp-v2,single-note
```

Detach without stopping training:

```text
Ctrl-b, then d
```

Reattach later:

```bash
tmux attach -t bass_train
```

List sessions:

```bash
tmux ls
```

Stop the run:

```bash
tmux attach -t bass_train
```

Then press:

```text
Ctrl-c
```

## What Happens If You Log Out

Works:

- closing the SSH terminal while tmux is running;
- closing VS Code while the Docker container keeps running;
- reconnecting later and reattaching with `tmux attach`.

Does not work:

- stopping the Docker container;
- rebooting the host;
- killing the tmux session;
- running training in a normal terminal and closing that terminal.

W&B will keep the metrics already uploaded, but it cannot continue training if the process dies.

## Resume Behavior

`train.py` writes the W&B run id to:

```text
runs/<run-name>/wandb_run_id.txt
```

If you restart with the same `--name` and `--overwrite true`, W&B can attach logs to the same W&B run id. Be careful: local files in the run directory are still your responsibility. The safer default is to use a new run name for each experiment.

For local model weights, the training script saves the best state to:

```text
runs/<run-name>/state.pth
```

Current `train.py` does not fully restore optimizer/model state and continue from an arbitrary step. W&B resume resumes the dashboard identity, not the PyTorch optimizer state.

## Remote Control

Practical remote control:

- watch metrics/audio from the W&B web UI;
- compare runs;
- stop training by SSH/tmux and `Ctrl-c`;
- start new runs by SSH/tmux;
- use W&B artifacts to download checkpoints.

Not currently implemented:

- changing learning rate live from the W&B UI;
- pausing/resuming training from W&B UI;
- launching training jobs from W&B Launch.

Those can be added later, but the first useful setup is stable tmux execution plus W&B logging.

## Offline Mode

If internet is unreliable:

```bash
python train.py ... --wandb true --wandb-mode offline
```

Later sync:

```bash
wandb sync /workspace/ddsp_pytorch/runs/<run-name>/wandb/offline-run-*
```

## Recommended First Test

Run a short smoke training:

```bash
cd /workspace/ddsp_pytorch
python train.py \
  --config config_idmt_bass_v2_single_note.yaml \
  --name wandb_smoke_001 \
  --steps 20 \
  --batch 2 \
  --wandb true \
  --wandb-project bass-ddsp \
  --wandb-tags smoke
```

Confirm in W&B:

- `train/loss` updates;
- `train/target_rms` and `train/reconstruction_rms` update;
- `eval/reconstruction_audio` appears after evaluation;
- local files also exist under `runs/wandb_smoke_001/`.
