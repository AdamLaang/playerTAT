from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tat_player_embeddings.dataset.collate import collate_windows
from tat_player_embeddings.dataset.sequences import apply_scaler, build_player_sequences
from tat_player_embeddings.dataset.window_dataset import CorruptionConfig, PlayerWindowDataset
from tat_player_embeddings.models.tat_encoder import TATEncoder
from tat_player_embeddings.utils import get_device, load_feature_data_and_scaler, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate player-match embeddings from trained TAT model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tat_player_embeddings/configs/tat_base.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Checkpoint path (default from config).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Comma-separated splits to export (train,validation,test or all).",
    )
    parser.add_argument("--from-date", type=str, default=None, help="Inclusive lower date bound (YYYY-MM-DD).")
    parser.add_argument("--to-date", type=str, default=None, help="Inclusive upper date bound (YYYY-MM-DD).")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/tat/embeddings.csv"),
        help="Output file for match-level embeddings.",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.98, help="EMA alpha for per-player embeddings.")
    parser.add_argument(
        "--player-output-csv",
        type=Path,
        default=Path("artifacts/tat/player_embeddings_ema.csv"),
        help="Output file for per-player EMA embeddings.",
    )
    parser.add_argument(
        "--write-player-ema",
        action="store_true",
        help="Also write a per-player EMA embedding table.",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for quick smoke tests.")
    return parser.parse_args()


def _parse_splits(value: str) -> List[str]:
    value = value.strip().lower()
    if value == "all":
        return ["train", "validation", "test"]
    return [p.strip() for p in value.split(",") if p.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    model_path = args.model_path
    if model_path is None:
        model_path = Path(cfg["output"]["model_dir"]) / cfg["output"]["model_file"]

    df, feature_cfg, scaler_payload = load_feature_data_and_scaler(cfg)
    continuous = feature_cfg["continuous_features"]

    df = apply_scaler(
        df=df,
        continuous_features=continuous,
        medians=scaler_payload["medians"],
        scaler=scaler_payload["scaler"],
    )

    sequences = build_player_sequences(df, continuous)

    target_splits = _parse_splits(args.split)
    dataset = PlayerWindowDataset(
        sequences=sequences,
        window_size=int(cfg["sequence"]["window_size"]),
        target_splits=target_splits,
        corruption=CorruptionConfig(0.0, 0.0, 0.0),
        use_corruption=False,
        cutoff_shift=int(cfg["sequence"].get("cutoff_shift", 0)),
        seed=int(cfg.get("seed", 42)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_windows,
    )

    device = get_device()
    encoder = TATEncoder(
        n_cont_features=len(continuous),
        n_positions=int(feature_cfg["vocab_sizes"]["position"]),
        n_teams=int(feature_cfg["vocab_sizes"]["team"]),
        window_size=int(cfg["sequence"]["window_size"]),
        d_model=int(cfg["model"]["d_model"]),
        d_z=int(cfg["model"]["d_z"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        use_position_embedding=bool(cfg["model"].get("use_position_embedding", True)),
        use_team_embeddings=bool(cfg["model"].get("use_team_embeddings", True)),
        causal_attention=bool(cfg["model"].get("causal_attention", False)),
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            x = batch["x_true"].to(device)
            position = batch["position_id"].to(device)
            home_away = batch["home_away"].to(device)
            gap = batch["gap_days"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            _, z = encoder(
                x_cont=x,
                position_id=position,
                home_away=home_away,
                gap_days=gap,
                pad_mask=pad_mask,
                team_id=None,
                opponent_id=None,
            )

            z_np = z.cpu().numpy()
            player_ids = batch["player_id"].cpu().numpy()
            match_ids = batch["match_id"].cpu().numpy()
            date_ns = batch["match_date_ns"].cpu().numpy()

            for i in range(len(player_ids)):
                rows.append(
                    {
                        "player_id": int(player_ids[i]),
                        "match_id": int(match_ids[i]),
                        "match_date": pd.to_datetime(int(date_ns[i]), utc=True),
                        "z": z_np[i],
                    }
                )

    emb_df = pd.DataFrame(rows)

    if args.from_date:
        start = pd.Timestamp(args.from_date, tz="UTC")
        emb_df = emb_df[emb_df["match_date"] >= start].copy()
    if args.to_date:
        end = pd.Timestamp(args.to_date, tz="UTC")
        emb_df = emb_df[emb_df["match_date"] <= end].copy()

    emb_df = emb_df.sort_values(["player_id", "match_date", "match_id"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_df = emb_df.copy()
    save_df["z"] = save_df["z"].map(lambda v: json.dumps(v.tolist()))
    save_df.to_csv(args.output_csv, index=False)
    print(f"Wrote match embeddings: {args.output_csv} (rows={len(save_df):,})")

    if args.write_player_ema:
        alpha = float(args.ema_alpha)
        player_rows = []
        for player_id, g in emb_df.groupby("player_id", sort=False):
            e = None
            for vec in g["z"]:
                if e is None:
                    e = vec.copy()
                else:
                    e = alpha * e + (1.0 - alpha) * vec
            player_rows.append(
                {
                    "player_id": int(player_id),
                    "last_match_date": g["match_date"].iloc[-1],
                    "n_matches": int(len(g)),
                    "ema_alpha": alpha,
                    "e": json.dumps(e.tolist()),
                }
            )

        player_df = pd.DataFrame(player_rows)
        args.player_output_csv.parent.mkdir(parents=True, exist_ok=True)
        player_df.to_csv(args.player_output_csv, index=False)
        print(f"Wrote player EMA embeddings: {args.player_output_csv} (rows={len(player_df):,})")


if __name__ == "__main__":
    main()
