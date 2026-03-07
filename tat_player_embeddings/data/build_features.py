from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


POSITION_ALIASES = {
    "Goalkeeper": "Goalkeeper",
    "Defender": "Defender",
    "Midfielder": "Midfielder",
    "Attacker": "Attacker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player-match features for TAT.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tat_player_embeddings/configs/tat_base.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _build_position_vocab(series: pd.Series) -> Dict[str, int]:
    values = sorted(v for v in series.dropna().unique().tolist())
    vocab = {"Unknown": 0}
    for value in values:
        if value == "Unknown":
            continue
        vocab[value] = len(vocab)
    return vocab


def _derive_player_position(df: pd.DataFrame) -> pd.Series:
    base = df["position"].astype("string").str.strip()
    normalized = base.map(POSITION_ALIASES)
    normalized = normalized.fillna("Unknown")
    return normalized


def build_features(cfg: Dict) -> None:
    players_path = Path(cfg["data"]["players_csv"])
    fixtures_path = Path(cfg["data"]["fixtures_csv"])
    processed_path = Path(cfg["data"]["processed_csv"])
    feature_cfg_path = Path(cfg["data"]["feature_config_yaml"])

    players = pd.read_csv(players_path, low_memory=False)
    fixtures = pd.read_csv(fixtures_path, low_memory=False)

    players = players.copy()
    fixtures = fixtures.copy()

    players["match_date"] = pd.to_datetime(players["starting_at"], utc=True, errors="coerce")
    players = players[players["match_date"].notna()].copy()

    players["player_id"] = pd.to_numeric(players["sportmonks_player_id"], errors="coerce")
    players["match_id"] = pd.to_numeric(players["fixture_id"], errors="coerce")
    players = players[players["player_id"].notna() & players["match_id"].notna()].copy()

    players["player_id"] = players["player_id"].astype(np.int64)
    players["match_id"] = players["match_id"].astype(np.int64)

    players["position_group"] = _derive_player_position(players)
    players = players[players["position_group"] != "Coach"].copy()

    # Resolve minutes from understat first, then fallback to provider minutes.
    _safe_numeric(players, ["understat_minutes", "minutes_played", "marketvalue", "goals", "age"])
    players["minutes_played_resolved"] = players["understat_minutes"].fillna(players["minutes_played"])
    min_minutes_required = float(cfg["features"].get("min_minutes_required", 0.0))
    if min_minutes_required > 0:
        players = players[players["minutes_played_resolved"].fillna(0.0) >= min_minutes_required].copy()

    # Match-level team context from fixture table.
    fixture_keep = [
        "fixture_id",
        "home_name",
        "away_name",
        "home_goals",
        "away_goals",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_possesion",
        "away_possesion",
    ]
    fixtures = fixtures[fixture_keep].rename(columns={"fixture_id": "match_id"})
    _safe_numeric(
        fixtures,
        [
            "home_goals",
            "away_goals",
            "home_shots",
            "away_shots",
            "home_shots_on_target",
            "away_shots_on_target",
            "home_possesion",
            "away_possesion",
        ],
    )

    players = players.merge(fixtures, how="left", on="match_id")

    is_home = players["team_name"].astype("string") == players["home_name"].astype("string")
    players["home_away"] = is_home.astype(np.int64)
    players["opponent_name"] = np.where(is_home, players["away_name"], players["home_name"])

    players["team_goals"] = np.where(is_home, players["home_goals"], players["away_goals"])
    players["opp_goals"] = np.where(is_home, players["away_goals"], players["home_goals"])
    players["team_shots"] = np.where(is_home, players["home_shots"], players["away_shots"])
    players["opp_shots"] = np.where(is_home, players["away_shots"], players["home_shots"])
    players["team_shots_on_target"] = np.where(
        is_home, players["home_shots_on_target"], players["away_shots_on_target"]
    )
    players["opp_shots_on_target"] = np.where(
        is_home, players["away_shots_on_target"], players["home_shots_on_target"]
    )
    players["team_possession"] = np.where(is_home, players["home_possesion"], players["away_possesion"])
    players["opp_possession"] = np.where(is_home, players["away_possesion"], players["home_possesion"])

    # Key engineered fields.
    players["own_goal"] = players["own_goal"].astype("boolean").fillna(False).astype(np.float32)
    players["log_marketvalue"] = np.log1p(players["marketvalue"].clip(lower=0))

    understat_cols = cfg["features"]["understat_numeric"]
    _safe_numeric(players, understat_cols)

    min_minutes = float(cfg["features"]["min_minutes_for_per90"])
    minute_denom = players["minutes_played_resolved"].fillna(0.0).clip(lower=min_minutes)
    per90_source_cols = [
        "goals",
        "own_goal",
        "understat_xg",
        "understat_xa",
        "understat_npxg",
        "understat_xgchain",
        "understat_xgbuildup",
        "understat_shots",
        "understat_key_passes",
        "understat_goals",
        "understat_assists",
    ]
    per90_cols: List[str] = []
    for col in per90_source_cols:
        if col not in players.columns:
            continue
        per_col = f"{col}_per90"
        players[per_col] = players[col] * 90.0 / minute_denom
        per90_cols.append(per_col)

    players = players.sort_values(["player_id", "match_date", "match_id"]).reset_index(drop=True)

    # Days since previous match per player.
    prev = players.groupby("player_id")["match_date"].shift(1)
    gap_days = (players["match_date"] - prev).dt.total_seconds() / 86400.0
    players["days_since_prev_match"] = gap_days.fillna(14.0).clip(lower=0.0, upper=180.0)

    # Stable vocab IDs for categorical context.
    players["position_group"] = players["position_group"].fillna("Unknown")
    position_vocab = _build_position_vocab(players["position_group"])
    players["position_id"] = players["position_group"].map(position_vocab).fillna(0).astype(np.int64)

    team_vocab = {name: idx + 1 for idx, name in enumerate(sorted(players["team_name"].dropna().astype(str).unique()))}
    team_vocab["Unknown"] = 0
    players["team_id"] = players["team_name"].astype(str).map(team_vocab).fillna(0).astype(np.int64)
    players["opponent_id"] = players["opponent_name"].astype(str).map(team_vocab).fillna(0).astype(np.int64)

    # Season index by first observed date.
    season_order = (
        players.groupby("season", as_index=False)["match_date"].min().sort_values("match_date")["season"].tolist()
    )
    season_vocab = {season: idx for idx, season in enumerate(season_order)}
    players["season_id"] = players["season"].map(season_vocab).astype(np.int64)

    # Missing indicators for sparse features.
    indicator_cols: List[str] = []
    for col in cfg["features"]["add_missing_indicators_for"]:
        if col in players.columns:
            ind_col = f"{col}_is_missing"
            players[ind_col] = players[col].isna().astype(np.float32)
            indicator_cols.append(ind_col)

    base_numeric = [c for c in cfg["features"]["base_numeric"] if c in players.columns]
    understat_numeric = [c for c in cfg["features"]["understat_numeric"] if c in players.columns]
    fixture_context_numeric = [
        c for c in cfg["features"]["fixture_context_numeric"] if c in players.columns
    ]
    continuous_cols = base_numeric + understat_numeric + fixture_context_numeric + per90_cols + indicator_cols

    keep_cols = [
        "player_id",
        "match_id",
        "match_date",
        "season",
        "season_id",
        "team_id",
        "opponent_id",
        "position_id",
        "home_away",
        "days_since_prev_match",
        "team_name",
        "opponent_name",
        "player_name",
        "dataset_split",
    ] + continuous_cols

    out_df = players[keep_cols].copy()
    out_df.to_csv(processed_path, index=False)

    feature_cfg = {
        "continuous_features": continuous_cols,
        "categorical_features": ["position_id", "home_away"],
        "gap_feature": "days_since_prev_match",
        "id_columns": {
            "player_id": "player_id",
            "match_id": "match_id",
            "match_date": "match_date",
            "season": "season",
            "season_id": "season_id",
        },
        "vocab_sizes": {
            "position": int(max(position_vocab.values()) + 1),
            "team": int(max(team_vocab.values()) + 1),
        },
        "position_vocab": position_vocab,
        "season_vocab": season_vocab,
        "per90_source_columns": per90_source_cols,
        "min_minutes_for_per90": min_minutes,
        "min_minutes_required": min_minutes_required,
    }

    feature_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(feature_cfg, f, sort_keys=False)

    team_vocab_path = feature_cfg_path.with_suffix(".team_vocab.json")
    with team_vocab_path.open("w", encoding="utf-8") as f:
        json.dump(team_vocab, f, ensure_ascii=True)

    print(f"Wrote processed features: {processed_path}")
    print(f"Rows={len(out_df):,} Columns={len(out_df.columns)}")
    print(f"Continuous feature count: {len(continuous_cols)}")
    print(f"Wrote feature config: {feature_cfg_path}")


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    build_features(cfg)


if __name__ == "__main__":
    main()
