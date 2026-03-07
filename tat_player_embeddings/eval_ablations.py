from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from tat_player_embeddings.dataset.collate import collate_windows
from tat_player_embeddings.dataset.sequences import apply_scaler, build_player_sequences
from tat_player_embeddings.dataset.window_dataset import CorruptionConfig, PlayerWindowDataset
from tat_player_embeddings.models.tat_encoder import TATEncoder
from tat_player_embeddings.utils import get_device, load_feature_data_and_scaler, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding quality ablations on test split.")
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output json path (default: artifacts/tat/eval_ablations.json).",
    )
    return parser.parse_args()


def compute_retrieval_same_position(emb: np.ndarray, position_ids: np.ndarray, k: int) -> float:
    n = emb.shape[0]
    if n <= 1:
        return 0.0

    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)

    k_eff = min(k, n - 1)
    topk = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
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


def evaluate_variant(
    *,
    name: str,
    encoder: TATEncoder,
    sequences,
    cfg: Dict,
    device: torch.device,
    k: int,
    max_batches: int | None,
    cutoff_shift: int,
    disable_position_input: bool,
    disable_team_context: bool,
) -> Dict:
    dataset = PlayerWindowDataset(
        sequences=sequences,
        window_size=int(cfg["sequence"]["window_size"]),
        target_splits=["test"],
        corruption=CorruptionConfig(0.0, 0.0, 0.0),
        use_corruption=False,
        cutoff_shift=cutoff_shift,
        seed=int(cfg.get("seed", 42)),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_windows,
    )

    all_z: List[np.ndarray] = []
    all_player_ids: List[np.ndarray] = []
    all_pos: List[np.ndarray] = []

    encoder.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = batch["x_true"].to(device)
            position = batch["position_id"].to(device)
            if disable_position_input:
                position = torch.zeros_like(position)

            home_away = batch["home_away"].to(device)
            gap = batch["gap_days"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            if disable_team_context or not encoder.use_team_embeddings:
                team_id = None
                opp_id = None
            else:
                team_id = batch["team_id"].to(device)
                opp_id = batch["opponent_id"].to(device)

            _, z = encoder(
                x_cont=x,
                position_id=position,
                home_away=home_away,
                gap_days=gap,
                pad_mask=pad_mask,
                team_id=team_id,
                opponent_id=opp_id,
            )

            all_z.append(z.cpu().numpy())
            all_player_ids.append(batch["player_id"].cpu().numpy())
            all_pos.append(batch["target_position_id"].cpu().numpy())

    if not all_z:
        raise ValueError(f"No embeddings produced for variant={name}")

    z = np.concatenate(all_z, axis=0)
    player_ids = np.concatenate(all_player_ids, axis=0)
    pos_ids = np.concatenate(all_pos, axis=0)

    return {
        "variant": name,
        "cutoff_shift": int(cutoff_shift),
        "disable_position_input": bool(disable_position_input),
        "disable_team_context": bool(disable_team_context),
        "n_embeddings": int(z.shape[0]),
        "embedding_dim": int(z.shape[1]),
        "same_position_at_k": compute_retrieval_same_position(z, pos_ids, k=k),
        "within_player_coherence": compute_within_player_coherence(z, player_ids),
    }


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

    baseline_shift = int(cfg["sequence"].get("cutoff_shift", 0))
    variants = [
        {
            "name": "baseline",
            "cutoff_shift": baseline_shift,
            "disable_position_input": False,
            "disable_team_context": False,
        },
    ]

    if bool(cfg["model"].get("use_position_embedding", True)):
        variants.append(
            {
                "name": "no_position_input",
                "cutoff_shift": baseline_shift,
                "disable_position_input": True,
                "disable_team_context": False,
            }
        )

    if bool(cfg["model"].get("use_team_embeddings", True)):
        variants.append(
            {
                "name": "no_team_context",
                "cutoff_shift": baseline_shift,
                "disable_position_input": False,
                "disable_team_context": True,
            }
        )

    if baseline_shift != -1:
        variants.append(
            {
                "name": "t_minus_1_history",
                "cutoff_shift": -1,
                "disable_position_input": False,
                "disable_team_context": False,
            }
        )

    results: List[Dict] = []
    for variant in variants:
        metrics = evaluate_variant(
            name=variant["name"],
            encoder=encoder,
            sequences=sequences,
            cfg=cfg,
            device=device,
            k=int(args.k),
            max_batches=args.max_batches,
            cutoff_shift=int(variant["cutoff_shift"]),
            disable_position_input=bool(variant["disable_position_input"]),
            disable_team_context=bool(variant["disable_team_context"]),
        )
        results.append(metrics)
        print(
            f"variant={metrics['variant']} "
            f"n={metrics['n_embeddings']} "
            f"same_position_at_k={metrics['same_position_at_k']:.4f} "
            f"within_player_coherence={metrics['within_player_coherence']:.4f}"
        )

    out_path = args.output_json
    if out_path is None:
        out_path = Path(cfg["output"]["model_dir"]) / "eval_ablations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "k": int(args.k),
        "max_batches": args.max_batches,
        "model_path": str(model_path),
        "variants": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote ablation metrics: {out_path}")


if __name__ == "__main__":
    main()
