from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TATEncoder(nn.Module):
    def __init__(
        self,
        n_cont_features: int,
        n_positions: int,
        n_teams: int,
        window_size: int,
        d_model: int = 384,
        d_z: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        use_position_embedding: bool = True,
        use_team_embeddings: bool = True,
        causal_attention: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.use_position_embedding = use_position_embedding
        self.use_team_embeddings = use_team_embeddings
        self.causal_attention = causal_attention

        self.phi = nn.Sequential(
            nn.Linear(n_cont_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.position_emb = nn.Embedding(n_positions, d_model)
        self.home_emb = nn.Embedding(2, d_model)
        self.recency_emb = nn.Embedding(window_size, d_model)

        self.gap_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        if use_team_embeddings:
            self.team_emb = nn.Embedding(n_teams, d_model)
            self.opp_emb = nn.Embedding(n_teams, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.z_proj = nn.Linear(d_model, d_z)

    @staticmethod
    def _last_valid_state(h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # `pad_mask` is True for padded tokens.
        lengths = (~pad_mask).sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(dim=1, index=idx).squeeze(1)

    def forward(
        self,
        x_cont: torch.Tensor,
        position_id: torch.Tensor,
        home_away: torch.Tensor,
        gap_days: torch.Tensor,
        pad_mask: torch.Tensor,
        team_id: Optional[torch.Tensor] = None,
        opponent_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x_cont.shape
        if seq_len != self.window_size:
            raise ValueError(f"Expected window size {self.window_size}, got {seq_len}")

        h = self.phi(x_cont)
        if self.use_position_embedding:
            h = h + self.position_emb(position_id)
        h = h + self.home_emb(home_away)

        recency = torch.arange(seq_len, device=x_cont.device).unsqueeze(0).expand(bsz, -1)
        h = h + self.recency_emb(recency)

        gap_input = torch.log1p(gap_days.clamp(min=0.0)).unsqueeze(-1)
        h = h + self.gap_proj(gap_input)

        if self.use_team_embeddings and team_id is not None and opponent_id is not None:
            h = h + self.team_emb(team_id) + self.opp_emb(opponent_id)

        attn_mask = None
        if self.causal_attention:
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x_cont.device),
                diagonal=1,
            )

        h = self.encoder(h, mask=attn_mask, src_key_padding_mask=pad_mask)
        h_last = self._last_valid_state(h, pad_mask)

        z = F.normalize(self.z_proj(h_last), p=2, dim=-1)
        return h, z
