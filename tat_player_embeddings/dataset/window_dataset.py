from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from tat_player_embeddings.dataset.sequences import PlayerSequence


@dataclass
class CorruptionConfig:
    p_feature_mask: float = 0.25
    p_match_mask: float = 0.10
    noise_sigma: float = 0.02


class PlayerWindowDataset(Dataset):
    def __init__(
        self,
        sequences: Dict[int, PlayerSequence],
        window_size: int,
        target_splits: Sequence[str],
        corruption: CorruptionConfig,
        use_corruption: bool,
        cutoff_shift: int = 0,
        seed: int = 42,
    ) -> None:
        self.sequences = sequences
        self.window_size = int(window_size)
        self.target_splits = set(target_splits)
        self.corruption = corruption
        self.use_corruption = use_corruption
        self.cutoff_shift = int(cutoff_shift)
        self.rng = np.random.default_rng(seed)

        self.index: List[Tuple[int, int]] = []
        for player_id, seq in self.sequences.items():
            for t in range(len(seq.split)):
                cutoff_t = t + self.cutoff_shift
                if seq.split[t] in self.target_splits and 0 <= cutoff_t < len(seq.split):
                    self.index.append((player_id, t))

        if not self.index:
            raise ValueError(f"No samples found for splits={target_splits}")

    def __len__(self) -> int:
        return len(self.index)

    def _extract_window(self, seq: PlayerSequence, t: int) -> Dict[str, np.ndarray]:
        L = self.window_size
        F = seq.x_cont.shape[1]

        start = t - L + 1
        real_start = max(start, 0)

        x = np.zeros((L, F), dtype=np.float32)
        position = np.zeros((L,), dtype=np.int64)
        home = np.zeros((L,), dtype=np.int64)
        team = np.zeros((L,), dtype=np.int64)
        opp = np.zeros((L,), dtype=np.int64)
        gap = np.zeros((L,), dtype=np.float32)
        pad_mask = np.ones((L,), dtype=bool)

        sl = slice(real_start, t + 1)
        length = (t + 1) - real_start
        # Right padding is safer for causal attention than left padding:
        # padded query tokens (at the tail) still have visible non-padded keys.
        dst = slice(0, length)

        x[dst] = seq.x_cont[sl]
        position[dst] = seq.position_id[sl]
        home[dst] = seq.home_away[sl]
        team[dst] = seq.team_id[sl]
        opp[dst] = seq.opponent_id[sl]
        gap[dst] = seq.gap_days[sl]
        pad_mask[dst] = False

        return {
            "x": x,
            "position": position,
            "home": home,
            "team": team,
            "opp": opp,
            "gap": gap,
            "pad_mask": pad_mask,
        }

    def _corrupt_view(self, x_true: np.ndarray, pad_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L, F = x_true.shape
        valid = ~pad_mask

        feat_mask = self.rng.random((L, F)) < self.corruption.p_feature_mask
        feat_mask &= valid[:, None]

        row_mask = self.rng.random(L) < self.corruption.p_match_mask
        row_mask &= valid
        match_mask = np.repeat(row_mask[:, None], F, axis=1)

        mask = feat_mask | match_mask
        if mask.sum() == 0:
            valid_rows = np.where(valid)[0]
            if len(valid_rows) > 0:
                r = int(self.rng.choice(valid_rows))
                c = int(self.rng.integers(0, F))
                mask[r, c] = True

        x_tilde = x_true.copy()
        if self.corruption.noise_sigma > 0:
            noise = self.rng.normal(0.0, self.corruption.noise_sigma, size=x_tilde.shape).astype(np.float32)
            add_noise = valid[:, None] & ~mask
            x_tilde[add_noise] += noise[add_noise]

        x_tilde[mask] = 0.0
        return x_tilde, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        player_id, t_target = self.index[idx]
        t_cutoff = t_target + self.cutoff_shift
        seq = self.sequences[player_id]
        w = self._extract_window(seq, t_cutoff)

        x_true = w["x"]
        if self.use_corruption:
            x1, m1 = self._corrupt_view(x_true, w["pad_mask"])
            x2, m2 = self._corrupt_view(x_true, w["pad_mask"])
        else:
            x1 = x_true.copy()
            x2 = x_true.copy()
            m1 = np.zeros_like(x_true, dtype=bool)
            m2 = np.zeros_like(x_true, dtype=bool)

        return {
            "x_true": torch.from_numpy(x_true),
            "x1": torch.from_numpy(x1),
            "m1": torch.from_numpy(m1),
            "x2": torch.from_numpy(x2),
            "m2": torch.from_numpy(m2),
            "position_id": torch.from_numpy(w["position"]),
            "home_away": torch.from_numpy(w["home"]),
            "team_id": torch.from_numpy(w["team"]),
            "opponent_id": torch.from_numpy(w["opp"]),
            "gap_days": torch.from_numpy(w["gap"]),
            "pad_mask": torch.from_numpy(w["pad_mask"]),
            "player_id": torch.tensor(player_id, dtype=torch.long),
            "t_index": torch.tensor(t_target, dtype=torch.long),
            "target_position_id": torch.tensor(int(seq.position_id[t_target]), dtype=torch.long),
            "match_id": torch.tensor(int(seq.match_id[t_target]), dtype=torch.long),
            "match_date_ns": torch.tensor(seq.match_date[t_target].astype("int64"), dtype=torch.long),
        }
