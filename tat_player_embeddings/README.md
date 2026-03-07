# Temporal Autoencoder Transformer (TAT) for Player Embeddings

Technical report (formal notation and derivations): `docs/TAT_TECHNICAL_REPORT.md`

This package trains a masked denoising temporal transformer with a contrastive objective to generate player-match embeddings.

Current defaults:

- Causal attention (`model.causal_attention: true`)
- Pre-match style cutoffs (`sequence.cutoff_shift: -1`, so history is up to `t-1`)
- Position embedding disabled by default (`model.use_position_embedding: false`)
- No team/opponent identity embeddings (`model.use_team_embeddings: false`)
- Mixed-role negatives in contrastive loss (both cross-role and in-role negatives, configurable weights)

## Feature Set (default)

The default config (`tat_player_embeddings/configs/tat_base.yaml`) uses:

- Core player state: `minutes_played_resolved`, `goals`, `own_goal`, `age`, `log_marketvalue`
- Understat features: `understat_xg`, `understat_xa`, `understat_xg90`, `understat_xa90`, `understat_npxg`, `understat_xgchain`, `understat_xgbuildup`, `understat_shots`, `understat_key_passes`, `understat_goals`, `understat_assists`
- Fixture context: team/opponent goals, shots, shots-on-target, possession
- Engineered features: per-90 variants for key event stats, missing indicators for sparse understat columns
- Temporal/context channels: `days_since_prev_match`, `position_id`, `home_away`

By default, rows with `minutes_played_resolved < 1` are dropped so understat coverage is stronger and bench-only rows do not dominate windows.

## Pipeline

1. Build features:

```bash
python3 -m tat_player_embeddings.data.build_features --config tat_player_embeddings/configs/tat_base.yaml
```

2. Build strict time-based splits:

```bash
python3 -m tat_player_embeddings.data.make_splits --config tat_player_embeddings/configs/tat_base.yaml
```

3. Fit train-only scaler:

```bash
python3 -m tat_player_embeddings.data.fit_scalers --config tat_player_embeddings/configs/tat_base.yaml
```

4. Train model:

```bash
python3 -m tat_player_embeddings.train --config tat_player_embeddings/configs/tat_base.yaml
```

Train with live logging + plots:

```bash
python3 -m tat_player_embeddings.train \
  --config tat_player_embeddings/configs/tat_base.yaml \
  --tensorboard \
  --plot-losses
```

This writes:

- `train.log` (timestamped text logs)
- `training_history.csv`
- `training_history.jsonl`
- `training_history.json`
- `loss_curves.svg`
- `tensorboard/` (if `--tensorboard` is enabled)

All under `output.model_dir` (default `artifacts/tat/`).

For a quick smoke run:

```bash
python3 -m tat_player_embeddings.train \
  --config tat_player_embeddings/configs/tat_base.yaml \
  --epochs 1 \
  --batch-size 32 \
  --max-train-batches 2 \
  --max-val-batches 2
```

5. Evaluate embeddings on test split:

```bash
python3 -m tat_player_embeddings.eval --config tat_player_embeddings/configs/tat_base.yaml --k 10
```

Run targeted ablations (baseline, no-position input, no-team context, t-1 history):

```bash
python3 -m tat_player_embeddings.eval_ablations \
  --config tat_player_embeddings/configs/tat_base.yaml \
  --k 10
```

6. Export embeddings:

```bash
python3 -m tat_player_embeddings.embed \
  --config tat_player_embeddings/configs/tat_base.yaml \
  --split test \
  --write-player-ema
```

Outputs are written to `artifacts/tat/` by default, using the best saved checkpoint
`artifacts/tat/model.pt` unless `--model-path` is overridden.

7. Visualize player embeddings (PCA + t-SNE):

```bash
python3 -m tat_player_embeddings.visualize_embeddings \
  --player-embeddings-csv artifacts/tat/player_embeddings_ema.csv \
  --features-csv data/tat_player_features_with_splits.csv \
  --feature-config-yaml data/tat_feature_config.yaml \
  --output-csv artifacts/tat/player_embedding_projection.csv \
  --output-svg artifacts/tat/player_embedding_projection.svg \
  --output-html artifacts/tat/player_embedding_projection.html
```

This writes:

- PCA: `artifacts/tat/player_embedding_projection.csv`, `artifacts/tat/player_embedding_projection.svg`, `artifacts/tat/player_embedding_projection.html`
- t-SNE: `artifacts/tat/player_embedding_projection_tsne.csv`, `artifacts/tat/player_embedding_projection_tsne.svg`, `artifacts/tat/player_embedding_projection_tsne.html`

The HTML outputs are interactive and include hover details for player name, team, seasons, split coverage, match count, and aggregated summaries of core model input features.

8. Search nearest neighbours for a player:

```bash
python3 -m tat_player_embeddings.player_neighbors \
  --query "Mohamed Salah" \
  --top-k 10
```

This searches over cosine similarity on `artifacts/tat/player_embeddings_ema.csv`. If a name is duplicated in the dataset, pass `--player-id` to disambiguate it.

## Post-run plotting

Generate/refresh loss plots from saved logs:

```bash
python3 -m tat_player_embeddings.plot_training \
  --history artifacts/tat/training_history.csv \
  --output artifacts/tat/loss_curves.svg
```
