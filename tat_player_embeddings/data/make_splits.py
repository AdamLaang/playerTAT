from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create strict time-based train/val/test splits.")
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


def make_splits(cfg: Dict) -> None:
    processed_path = Path(cfg["data"]["processed_csv"])
    split_path = Path(cfg["data"]["split_csv"])

    df = pd.read_csv(processed_path, parse_dates=["match_date"], low_memory=False)
    if df.empty:
        raise ValueError("Processed feature file is empty.")

    season_last_date = (
        df.groupby("season", as_index=False)["match_date"]
        .max()
        .sort_values("match_date")
        .reset_index(drop=True)
    )

    if len(season_last_date) < 3:
        raise ValueError("Need at least 3 seasons for train/validation/test split.")

    test_season = season_last_date.iloc[-1]["season"]
    val_season = season_last_date.iloc[-2]["season"]

    split = pd.Series("train", index=df.index, dtype="object")
    split.loc[df["season"] == val_season] = "validation"
    split.loc[df["season"] == test_season] = "test"
    df["split"] = split

    # Keep hint column only as reference, model code reads `split`.
    if "dataset_split" in df.columns:
        df.rename(columns={"dataset_split": "upstream_dataset_split"}, inplace=True)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_path, index=False)

    counts = df["split"].value_counts().to_dict()
    print(f"Test season={test_season} Val season={val_season}")
    print(f"Split counts={counts}")
    print(f"Wrote split dataset: {split_path}")


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    make_splits(cfg)


if __name__ == "__main__":
    main()
