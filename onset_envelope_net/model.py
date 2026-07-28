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


class StructuredBassOnsetEnvelopeNet(nn.Module):
    """Causal articulation template with smooth peak/attack/decay parameters."""

    def __init__(self, num_articulations: int, control_dim: int, envelope_frames: int = 20, hidden: int = 64):
        super().__init__()
        self.envelope_frames = int(envelope_frames)
        self.control_dim = int(control_dim)
        self.articulation_embedding = nn.Embedding(int(num_articulations), hidden)
        self.base_parameters = nn.Embedding(int(num_articulations), 3)
        self.modulation = nn.Sequential(
            nn.Linear(hidden + self.control_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, articulation_id, controls):
        # articulation_id: (B,)
        # controls: (B, C), selected normalized onset controls
        embedding = self.articulation_embedding(articulation_id)  # (B, H)
        parameters = self.base_parameters(articulation_id)  # (B, 3)
        if self.control_dim:
            parameters = parameters + 0.25 * self.modulation(torch.cat([embedding, controls], dim=-1))  # (B, 3)
        else:
            parameters = parameters + 0.25 * self.modulation(embedding.new_zeros(embedding.shape[0], embedding.shape[1]))[:, :3]
        peak = torch.sigmoid(parameters[:, 0:1])  # (B, 1)
        attack_frames = 0.5 + 6.0 * torch.sigmoid(parameters[:, 1:2])  # (B, 1)
        decay_frames = 1.0 + 18.0 * torch.sigmoid(parameters[:, 2:3])  # (B, 1)
        time = torch.arange(self.envelope_frames, device=parameters.device, dtype=parameters.dtype)[None, :]  # (1, K)
        attack = 1.0 - torch.exp(-(time + 1.0) / attack_frames)  # (B, K)
        decay = torch.exp(-time / decay_frames)  # (B, K)
        return (peak * attack * decay).clamp(0.0, 1.0)  # (B, K)
