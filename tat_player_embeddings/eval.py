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
    parser = argparse.ArgumentParser(description="Evaluate embedding quality on test split.")
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
    parser.add_argument("--k", type=int, default=10, help="k for kNN retrieval.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for quick smoke tests.")
    return parser.parse_args()


def _last_valid_token(values: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    lengths = (~pad_mask).sum(dim=1).clamp(min=1)
    idx = (lengths - 1).view(-1, 1)
    return values.gather(1, idx).squeeze(1)


def compute_retrieval_same_position(emb: np.ndarray, position_ids: np.ndarray, k: int) -> float:
    n = emb.shape[0]
    if n <= 1:
        return 0.0

    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)

    topk = np.argpartition(-sim, kth=min(k, n - 1) - 1, axis=1)[:, : min(k, n - 1)]
    matches = (position_ids[topk] == position_ids[:, None]).astype(np.float32)
    return float(matches.mean())


def compute_within_player_coherence(emb: np.ndarray, player_ids: np.ndarray) -> float:
    scores: List[float] = []
    for pid in np.unique(player_ids):
        idx = np.where(player_ids == pid)[0]
        if len(idx) < 2:
            continue
        e = emb[idx]
        sim = e @ e.T
        upper = np.triu_indices(len(idx), k=1)
        if len(upper[0]) == 0:
            continue
        scores.append(float(sim[upper].mean()))

    return float(np.mean(scores)) if scores else 0.0


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

    dataset = PlayerWindowDataset(
        sequences=sequences,
        window_size=int(cfg["sequence"]["window_size"]),
        target_splits=["test"],
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

    all_z: List[np.ndarray] = []
    all_player_ids: List[np.ndarray] = []
    all_match_ids: List[np.ndarray] = []
    all_dates: List[np.ndarray] = []
    all_pos: List[np.ndarray] = []

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

            if "target_position_id" in batch:
                last_pos = batch["target_position_id"].to(device)
            else:
                last_pos = _last_valid_token(position, pad_mask)
            all_z.append(z.cpu().numpy())
            all_player_ids.append(batch["player_id"].cpu().numpy())
            all_match_ids.append(batch["match_id"].cpu().numpy())
            all_dates.append(batch["match_date_ns"].cpu().numpy())
            all_pos.append(last_pos.cpu().numpy())

    z = np.concatenate(all_z, axis=0)
    player_ids = np.concatenate(all_player_ids, axis=0)
    match_ids = np.concatenate(all_match_ids, axis=0)
    date_ns = np.concatenate(all_dates, axis=0)
    pos_ids = np.concatenate(all_pos, axis=0)

    same_pos_at_k = compute_retrieval_same_position(z, pos_ids, k=args.k)
    coherence = compute_within_player_coherence(z, player_ids)

    out_dir = Path(cfg["output"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "n_test_embeddings": int(z.shape[0]),
        "embedding_dim": int(z.shape[1]),
        "same_position_at_k": same_pos_at_k,
        "within_player_coherence": coherence,
        "k": int(args.k),
    }

    metrics_path = out_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    embed_df = pd.DataFrame(
        {
            "player_id": player_ids,
            "match_id": match_ids,
            "match_date": pd.to_datetime(date_ns, utc=True),
            "position_id": pos_ids,
            "z": [json.dumps(row.tolist()) for row in z],
        }
    )
    embed_path = out_dir / "test_embeddings.csv"
    embed_df.to_csv(embed_path, index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote embeddings: {embed_path}")


if __name__ == "__main__":
    main()
