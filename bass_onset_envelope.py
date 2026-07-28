from __future__ import annotations

import torch
from torch import nn


class BassOnsetEnvelopeNet(nn.Module):
    """Articulation template plus causal control-conditioned modulation."""

    def __init__(self, num_articulations: int, envelope_frames: int = 20, hidden: int = 128):
        super().__init__()
        self.envelope_frames = int(envelope_frames)
        self.articulation_embedding = nn.Embedding(int(num_articulations), hidden)
        self.base_template_logits = nn.Embedding(int(num_articulations), self.envelope_frames)
        self.modulation = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.envelope_frames),
        )

    def forward(self, articulation_id, controls):
        # articulation_id: (B,)
        # controls: (B, 3) = normalized f0, loudness, periodicity
        embedding = self.articulation_embedding(articulation_id)  # (B, H)
        base = self.base_template_logits(articulation_id)  # (B, K)
        modulation = self.modulation(torch.cat([embedding, controls], dim=-1))  # (B, K)
        return torch.sigmoid(base + modulation)  # (B, K)
