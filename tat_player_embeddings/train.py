from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tat_player_embeddings.data.build_features import build_features
from tat_player_embeddings.data.fit_scalers import fit_scaler
from tat_player_embeddings.data.make_splits import make_splits
from tat_player_embeddings.dataset.collate import collate_windows
from tat_player_embeddings.dataset.sequences import apply_scaler, build_player_sequences
from tat_player_embeddings.dataset.window_dataset import CorruptionConfig, PlayerWindowDataset
from tat_player_embeddings.losses.info_nce import mixed_role_info_nce_loss
from tat_player_embeddings.losses.reconstruction import masked_huber_loss
from tat_player_embeddings.models.heads import ReconstructionHead
from tat_player_embeddings.models.tat_encoder import TATEncoder
from tat_player_embeddings.utils import get_device, load_feature_data_and_scaler, load_yaml, seed_everything


def _log(msg: str, fh) -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def _write_loss_plot(history: list[Dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["eval_loss"] for row in history]
    train_rec = [row["train_rec"] for row in history]
    val_rec = [row["eval_rec"] for row in history]
    train_con = [row["train_con"] for row in history]
    val_con = [row["eval_con"] for row in history]
    panels = [
        ("Total Loss", train_loss, val_loss),
        ("Reconstruction Loss", train_rec, val_rec),
        ("Contrastive Loss", train_con, val_con),
    ]

    width = 1300
    height = 420
    margin_left = 50
    margin_right = 20
    margin_top = 35
    margin_bottom = 40
    panel_gap = 24
    panel_width = (width - margin_left - margin_right - panel_gap * 2) / 3.0
    panel_height = height - margin_top - margin_bottom

    def _polyline(xs, ys, x0, y0, w, h) -> str:
        if len(xs) <= 1:
            return ""
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if y_min == y_max:
            y_min -= 1e-6
            y_max += 1e-6
        if x_min == x_max:
            x_max = x_min + 1
        pts = []
        for x, y in zip(xs, ys):
            px = x0 + (x - x_min) / (x_max - x_min) * w
            py = y0 + h - (y - y_min) / (y_max - y_min) * h
            pts.append(f"{px:.2f},{py:.2f}")
        return " ".join(pts)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white" />',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937} .axis{stroke:#9ca3af;stroke-width:1} .grid{stroke:#e5e7eb;stroke-width:1}</style>',
        '<text x="20" y="22" font-size="14" font-weight="600">TAT Training Curves</text>',
    ]

    for i, (title, ys_train, ys_val) in enumerate(panels):
        x0 = margin_left + i * (panel_width + panel_gap)
        y0 = margin_top
        all_y = ys_train + ys_val
        y_min, y_max = min(all_y), max(all_y)
        if y_min == y_max:
            y_min -= 1e-6
            y_max += 1e-6

        parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{panel_width:.2f}" height="{panel_height:.2f}" fill="none" class="axis" />')
        for frac in [0.25, 0.5, 0.75]:
            gy = y0 + panel_height * frac
            parts.append(f'<line x1="{x0:.2f}" y1="{gy:.2f}" x2="{x0 + panel_width:.2f}" y2="{gy:.2f}" class="grid" />')
        parts.append(f'<text x="{x0:.2f}" y="{y0 - 8:.2f}" font-size="12" font-weight="600">{title}</text>')
        parts.append(
            f'<text x="{x0:.2f}" y="{y0 + panel_height + 16:.2f}" font-size="10">y[{y_min:.4f}, {y_max:.4f}]</text>'
        )

        tr_points = _polyline(epochs, ys_train, x0, y0, panel_width, panel_height)
        va_points = _polyline(epochs, ys_val, x0, y0, panel_width, panel_height)
        if tr_points:
            parts.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{tr_points}" />')
        if va_points:
            parts.append(f'<polyline fill="none" stroke="#dc2626" stroke-width="2" points="{va_points}" />')

        lx = x0 + 8
        ly = y0 + 14
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 16}" y2="{ly}" stroke="#2563eb" stroke-width="2" />')
        parts.append(f'<text x="{lx + 20}" y="{ly + 3}" font-size="10">train</text>')
        parts.append(f'<line x1="{lx + 60}" y1="{ly}" x2="{lx + 76}" y2="{ly}" stroke="#dc2626" stroke-width="2" />')
        parts.append(f'<text x="{lx + 80}" y="{ly + 3}" font-size="10">val</text>')

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal autoencoder transformer for player embeddings.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tat_player_embeddings/configs/tat_base.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Run build_features -> make_splits -> fit_scalers before training.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Cap batches per train epoch.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Cap batches per validation epoch.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to plain-text training log file (default: <model_dir>/train.log).",
    )
    parser.add_argument(
        "--csv-log",
        type=Path,
        default=None,
        help="Path to epoch metrics CSV (default: <model_dir>/training_history.csv).",
    )
    parser.add_argument(
        "--jsonl-log",
        type=Path,
        default=None,
        help="Path to epoch metrics JSONL (default: <model_dir>/training_history.jsonl).",
    )
    parser.add_argument(
        "--plot-losses",
        action="store_true",
        help="Write/update loss curve SVG during training.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard scalars under <model_dir>/tensorboard.",
    )
    return parser.parse_args()


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, _lr_lambda)


def contrastive_weight(epoch: int, cfg: Dict) -> float:
    target = float(cfg["train"]["lambda_contrastive"])
    warmup_epochs = int(cfg["train"].get("contrastive_warmup_epochs", 0))
    ramp_epochs = int(cfg["train"].get("contrastive_ramp_epochs", max(1, warmup_epochs)))

    if epoch < warmup_epochs:
        return 0.0

    ramp_progress = min(1.0, float(epoch - warmup_epochs + 1) / float(max(1, ramp_epochs)))
    return target * ramp_progress


def run_epoch(
    encoder: TATEncoder,
    recon_head: ReconstructionHead,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    tau: float,
    cross_role_weight: float,
    in_role_weight: float,
    lambda_con: float,
    grad_clip: float,
    use_amp: bool,
    training: bool,
    goalkeeper_position_id: int,
    max_batches: int | None,
) -> Dict[str, float]:
    mode = "train" if training else "eval"
    encoder.train(training)
    recon_head.train(training)

    total_loss = 0.0
    total_rec = 0.0
    total_con = 0.0
    n_batches = 0
    skipped_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x_true = batch["x_true"].to(device)
        x1 = batch["x1"].to(device)
        x2 = batch["x2"].to(device)
        m1 = batch["m1"].to(device)
        m2 = batch["m2"].to(device)

        position_id = batch["position_id"].to(device)
        home_away = batch["home_away"].to(device)
        gap_days = batch["gap_days"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        target_position_id = batch["target_position_id"].to(device)
        is_goalkeeper = target_position_id == int(goalkeeper_position_id)

        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = use_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            h1, z1 = encoder(
                x_cont=x1,
                position_id=position_id,
                home_away=home_away,
                gap_days=gap_days,
                pad_mask=pad_mask,
                team_id=None,
                opponent_id=None,
            )
            h2, z2 = encoder(
                x_cont=x2,
                position_id=position_id,
                home_away=home_away,
                gap_days=gap_days,
                pad_mask=pad_mask,
                team_id=None,
                opponent_id=None,
            )

            xhat1 = recon_head(h1)
            xhat2 = recon_head(h2)

            rec1 = masked_huber_loss(xhat1, x_true, m1)
            rec2 = masked_huber_loss(xhat2, x_true, m2)
            rec = 0.5 * (rec1 + rec2)

            con = mixed_role_info_nce_loss(
                z1,
                z2,
                is_goalkeeper=is_goalkeeper,
                tau=tau,
                cross_role_weight=cross_role_weight,
                in_role_weight=in_role_weight,
            )
            loss = rec + lambda_con * con

        if not torch.isfinite(loss) or not torch.isfinite(rec) or not torch.isfinite(con):
            # Skip numerically unstable batches instead of poisoning epoch metrics.
            skipped_batches += 1
            continue

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(list(encoder.parameters()) + list(recon_head.parameters()), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(list(encoder.parameters()) + list(recon_head.parameters()), grad_clip)
                optimizer.step()
            scheduler.step()

        total_loss += float(loss.item())
        total_rec += float(rec.item())
        total_con += float(con.item())
        n_batches += 1

    if n_batches == 0:
        raise ValueError(f"No batches in {mode} loader.")

    return {
        f"{mode}_loss": total_loss / n_batches,
        f"{mode}_rec": total_rec / n_batches,
        f"{mode}_con": total_con / n_batches,
        f"{mode}_skipped_batches": skipped_batches,
    }


def maybe_prepare_data(cfg: Dict) -> None:
    build_features(cfg)
    make_splits(cfg)
    fit_scaler(cfg)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)

    seed_everything(int(cfg.get("seed", 42)))

    if args.prepare_data:
        maybe_prepare_data(cfg)

    df, feature_cfg, scaler_payload = load_feature_data_and_scaler(cfg)
    continuous = feature_cfg["continuous_features"]

    df = apply_scaler(
        df=df,
        continuous_features=continuous,
        medians=scaler_payload["medians"],
        scaler=scaler_payload["scaler"],
    )

    sequences = build_player_sequences(df, continuous)

    corruption = CorruptionConfig(
        p_feature_mask=float(cfg["train"]["p_feature_mask"]),
        p_match_mask=float(cfg["train"]["p_match_mask"]),
        noise_sigma=float(cfg["train"]["noise_sigma"]),
    )

    window_size = int(cfg["sequence"]["window_size"])
    cutoff_shift = int(cfg["sequence"].get("cutoff_shift", 0))
    train_ds = PlayerWindowDataset(
        sequences=sequences,
        window_size=window_size,
        target_splits=["train"],
        corruption=corruption,
        use_corruption=True,
        cutoff_shift=cutoff_shift,
        seed=int(cfg.get("seed", 42)),
    )
    val_ds = PlayerWindowDataset(
        sequences=sequences,
        window_size=window_size,
        target_splits=["validation"],
        corruption=corruption,
        use_corruption=True,
        cutoff_shift=cutoff_shift,
        seed=int(cfg.get("seed", 42)) + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_windows,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_windows,
        drop_last=False,
    )

    device = get_device()

    encoder = TATEncoder(
        n_cont_features=len(continuous),
        n_positions=int(feature_cfg["vocab_sizes"]["position"]),
        n_teams=int(feature_cfg["vocab_sizes"]["team"]),
        window_size=window_size,
        d_model=int(cfg["model"]["d_model"]),
        d_z=int(cfg["model"]["d_z"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        use_position_embedding=bool(cfg["model"].get("use_position_embedding", True)),
        use_team_embeddings=bool(cfg["model"].get("use_team_embeddings", True)),
        causal_attention=bool(cfg["model"].get("causal_attention", False)),
    ).to(device)

    position_vocab = feature_cfg.get("position_vocab", {})
    goalkeeper_position_id = position_vocab.get("Goalkeeper", None)
    if goalkeeper_position_id is None:
        raise ValueError("Could not find 'Goalkeeper' in feature config position_vocab.")

    recon_head = ReconstructionHead(
        d_model=int(cfg["model"]["d_model"]),
        n_cont_features=len(continuous),
    ).to(device)

    params = list(encoder.parameters()) + list(recon_head.parameters())
    optimizer = AdamW(
        params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    total_steps = epochs * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=float(cfg["train"]["warmup_ratio"]))

    use_amp = bool(cfg["train"].get("amp", True))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    tau = float(cfg["train"]["tau"])
    cross_role_weight = float(cfg["train"].get("contrastive_cross_role_weight", 1.0))
    in_role_weight = float(cfg["train"].get("contrastive_in_role_weight", 1.0))
    grad_clip = float(cfg["train"]["grad_clip"])

    out_dir = Path(cfg["output"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / cfg["output"]["model_file"]
    log_path = args.log_file if args.log_file is not None else out_dir / "train.log"
    csv_log_path = args.csv_log if args.csv_log is not None else out_dir / "training_history.csv"
    jsonl_log_path = args.jsonl_log if args.jsonl_log is not None else out_dir / "training_history.jsonl"
    plot_path = out_dir / "loss_curves.svg"
    hist_path = out_dir / "training_history.json"

    best_val = float("inf")
    history = []
    with (
        log_path.open("w", encoding="utf-8") as log_fh,
        csv_log_path.open("w", encoding="utf-8", newline="") as csv_fh,
        jsonl_log_path.open("w", encoding="utf-8") as jsonl_fh,
    ):
        _log(f"Using device: {device}", log_fh)
        _log(f"Output directory: {out_dir}", log_fh)
        _log(
            f"Train batches/epoch={len(train_loader)} Val batches/epoch={len(val_loader)} Epochs={epochs}",
            log_fh,
        )
        _log(f"Causal cutoff shift={cutoff_shift}", log_fh)

        metric_fields = [
            "epoch",
            "epoch_time_sec",
            "lambda_con",
            "train_loss",
            "train_rec",
            "train_con",
            "train_skipped_batches",
            "eval_loss",
            "eval_rec",
            "eval_con",
            "eval_skipped_batches",
        ]
        csv_writer = csv.DictWriter(csv_fh, fieldnames=metric_fields)
        csv_writer.writeheader()
        csv_fh.flush()

        tb_writer = None
        if args.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
                _log(f"TensorBoard enabled: {out_dir / 'tensorboard'}", log_fh)
            except Exception as exc:  # pragma: no cover
                _log(f"TensorBoard disabled (import failed): {exc}", log_fh)

        plot_enabled = bool(args.plot_losses)
        if plot_enabled:
            _log(f"Loss plot will be updated at: {plot_path}", log_fh)

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            lambda_con = contrastive_weight(epoch, cfg)
            train_metrics = run_epoch(
                encoder=encoder,
                recon_head=recon_head,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                tau=tau,
                cross_role_weight=cross_role_weight,
                in_role_weight=in_role_weight,
                lambda_con=lambda_con,
                grad_clip=grad_clip,
                use_amp=use_amp,
                training=True,
                goalkeeper_position_id=int(goalkeeper_position_id),
                max_batches=args.max_train_batches,
            )

            with torch.no_grad():
                val_metrics = run_epoch(
                    encoder=encoder,
                    recon_head=recon_head,
                    loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    device=device,
                    tau=tau,
                    cross_role_weight=cross_role_weight,
                    in_role_weight=in_role_weight,
                    lambda_con=lambda_con,
                    grad_clip=grad_clip,
                    use_amp=use_amp,
                    training=False,
                    goalkeeper_position_id=int(goalkeeper_position_id),
                    max_batches=args.max_val_batches,
                )

            merged = {"epoch": epoch + 1, "lambda_con": lambda_con, **train_metrics, **val_metrics}
            merged["epoch_time_sec"] = round(time.perf_counter() - epoch_start, 3)
            history.append(merged)

            csv_writer.writerow({k: merged[k] for k in metric_fields})
            csv_fh.flush()
            jsonl_fh.write(json.dumps(merged) + "\n")
            jsonl_fh.flush()

            if tb_writer is not None:
                tb_writer.add_scalar("loss/train_total", merged["train_loss"], epoch + 1)
                tb_writer.add_scalar("loss/val_total", merged["eval_loss"], epoch + 1)
                tb_writer.add_scalar("loss/train_rec", merged["train_rec"], epoch + 1)
                tb_writer.add_scalar("loss/val_rec", merged["eval_rec"], epoch + 1)
                tb_writer.add_scalar("loss/train_con", merged["train_con"], epoch + 1)
                tb_writer.add_scalar("loss/val_con", merged["eval_con"], epoch + 1)
                tb_writer.add_scalar("train/lambda_con", merged["lambda_con"], epoch + 1)
                tb_writer.add_scalar("train/epoch_time_sec", merged["epoch_time_sec"], epoch + 1)
                tb_writer.add_scalar("train/skipped_batches", merged["train_skipped_batches"], epoch + 1)
                tb_writer.add_scalar("val/skipped_batches", merged["eval_skipped_batches"], epoch + 1)

            if plot_enabled:
                _write_loss_plot(history, plot_path)

            _log(
                f"epoch={epoch + 1:02d} "
                f"train_loss={train_metrics['train_loss']:.4f} "
                f"val_loss={val_metrics['eval_loss']:.4f} "
                f"train_rec={train_metrics['train_rec']:.4f} "
                f"val_rec={val_metrics['eval_rec']:.4f} "
                f"train_con={train_metrics['train_con']:.6e} "
                f"val_con={val_metrics['eval_con']:.6e} "
                f"train_skipped={train_metrics['train_skipped_batches']} "
                f"val_skipped={val_metrics['eval_skipped_batches']} "
                f"epoch_time_sec={merged['epoch_time_sec']:.2f}",
                log_fh,
            )

            if val_metrics["eval_loss"] < best_val:
                best_val = val_metrics["eval_loss"]
                torch.save(
                    {
                        "encoder_state_dict": encoder.state_dict(),
                        "recon_head_state_dict": recon_head.state_dict(),
                        "feature_config": feature_cfg,
                        "config": cfg,
                        "best_val_loss": best_val,
                    },
                    model_path,
                )
                _log(f"New best checkpoint saved: {model_path} (val_loss={best_val:.6f})", log_fh)

        with hist_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if tb_writer is not None:
            tb_writer.close()

        _log(f"Saved best model: {model_path}", log_fh)
        _log(f"Saved history JSON: {hist_path}", log_fh)
        _log(f"Saved history CSV: {csv_log_path}", log_fh)
        _log(f"Saved history JSONL: {jsonl_log_path}", log_fh)
        if args.plot_losses:
            _log(f"Loss plot path: {plot_path}", log_fh)


if __name__ == "__main__":
    main()
