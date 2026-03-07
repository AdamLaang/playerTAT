from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    def __init__(self, d_model: int, n_cont_features: int) -> None:
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_cont_features),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.psi(h)
