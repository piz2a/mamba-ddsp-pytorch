# Codex Workspace Rules

- Treat `/workspace` as the main repository root.
- First-party code belongs directly under `/workspace` or a clearly named first-party package such as `/workspace/bass_ddsp`.
- Do not add new first-party implementation code inside cloned/reference repositories.
- The cloned/reference repositories are read-only references for Codex work:
  - `ddsp_pytorch`
  - `ddsp_pytorch-modified`
  - `ddsp-guitar`
  - `diff-wave-synth`
  - `mamba`
- If reference code is needed in the first-party project, copy or reimplement the required concept into first-party files and document the source.
- Keep generated runs, checkpoints, caches, and datasets out of git unless explicitly requested.
