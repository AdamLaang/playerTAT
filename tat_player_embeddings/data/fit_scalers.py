from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit train-only scaler + imputers.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tat_player_embeddings/configs/tat_base.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fit_scaler(cfg: Dict) -> None:
    split_path = Path(cfg["data"]["split_csv"])
    feature_cfg_path = Path(cfg["data"]["feature_config_yaml"])
    scaler_path = Path(cfg["data"]["scaler_joblib"])

    df = pd.read_csv(split_path, low_memory=False)
    feature_cfg = _load_yaml(feature_cfg_path)
    continuous = feature_cfg["continuous_features"]

    missing = [c for c in continuous if c not in df.columns]
    if missing:
        raise ValueError(f"Missing continuous columns in split dataset: {missing}")

    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("No rows in split=train. Cannot fit scaler.")

    medians = train_df[continuous].median(axis=0, skipna=True).fillna(0.0)
    train_imputed = train_df[continuous].fillna(medians)

    scaler = StandardScaler()
    scaler.fit(train_imputed.values)

    payload = {
        "continuous_features": continuous,
        "medians": medians.to_dict(),
        "scaler": scaler,
    }

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, scaler_path)
    print(f"Wrote scaler artifact: {scaler_path}")
    print(f"Train rows used: {len(train_df):,}")
    print(f"Continuous feature count: {len(continuous)}")


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    fit_scaler(cfg)


if __name__ == "__main__":
    main()
