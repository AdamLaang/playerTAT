from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class PlayerSequence:
    player_id: int
    x_cont: np.ndarray
    position_id: np.ndarray
    home_away: np.ndarray
    team_id: np.ndarray
    opponent_id: np.ndarray
    gap_days: np.ndarray
    split: np.ndarray
    match_id: np.ndarray
    match_date: np.ndarray


def apply_scaler(
    df: pd.DataFrame,
    continuous_features: List[str],
    medians: Dict[str, float],
    scaler,
) -> pd.DataFrame:
    out = df.copy()
    fill_values = {col: medians.get(col, 0.0) for col in continuous_features}
    imputed = out[continuous_features].fillna(fill_values)
    scaled = scaler.transform(imputed.values)
    scaled_df = pd.DataFrame(scaled, columns=continuous_features, index=out.index).astype(np.float32)
    for col in continuous_features:
        out[col] = scaled_df[col]
    return out


def build_player_sequences(df: pd.DataFrame, continuous_features: List[str]) -> Dict[int, PlayerSequence]:
    required = [
        "player_id",
        "match_id",
        "match_date",
        "split",
        "position_id",
        "home_away",
        "days_since_prev_match",
    ] + continuous_features

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for sequence build: {missing}")

    work = df[required].copy()
    if "team_id" not in df.columns:
        work["team_id"] = 0
    else:
        work["team_id"] = df["team_id"]
    if "opponent_id" not in df.columns:
        work["opponent_id"] = 0
    else:
        work["opponent_id"] = df["opponent_id"]
    work["match_date"] = pd.to_datetime(work["match_date"], utc=True, errors="coerce")
    work = work[work["match_date"].notna()].copy()
    work = work.sort_values(["player_id", "match_date", "match_id"]).reset_index(drop=True)

    sequences: Dict[int, PlayerSequence] = {}
    for player_id, g in work.groupby("player_id", sort=False):
        x_cont = g[continuous_features].to_numpy(dtype=np.float32)
        seq = PlayerSequence(
            player_id=int(player_id),
            x_cont=x_cont,
            position_id=g["position_id"].to_numpy(dtype=np.int64),
            home_away=g["home_away"].to_numpy(dtype=np.int64),
            team_id=g["team_id"].to_numpy(dtype=np.int64),
            opponent_id=g["opponent_id"].to_numpy(dtype=np.int64),
            gap_days=g["days_since_prev_match"].to_numpy(dtype=np.float32),
            split=g["split"].to_numpy(dtype=object),
            match_id=g["match_id"].to_numpy(dtype=np.int64),
            match_date=g["match_date"].to_numpy(dtype="datetime64[ns]"),
        )
        sequences[seq.player_id] = seq

    return sequences
