from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import plotly.graph_objects as go
from sklearn.manifold import TSNE
import yaml

COLOR_MAP = {
    "Goalkeeper": "#1f77b4",
    "Defender": "#2ca02c",
    "Midfielder": "#ff7f0e",
    "Attacker": "#d62728",
    "Unknown": "#7f7f7f",
}
DEFAULT_HOVER_FEATURES = [
    "minutes_played_resolved",
    "age",
    "log_marketvalue",
    "goals_per90",
    "understat_xg_per90",
    "understat_xa_per90",
    "understat_shots_per90",
    "understat_key_passes_per90",
    "team_possession",
    "opp_possession",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project and visualize player embeddings.")
    parser.add_argument(
        "--player-embeddings-csv",
        type=Path,
        default=Path("artifacts/tat/player_embeddings_ema.csv"),
        help="CSV from embed.py containing one embedding vector per player.",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("data/tat_player_features_with_splits.csv"),
        help="Feature table used to derive dominant player position.",
    )
    parser.add_argument(
        "--feature-config-yaml",
        type=Path,
        default=Path("data/tat_feature_config.yaml"),
        help="Feature config with position vocab.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/tat/player_embedding_projection.csv"),
        help="Output CSV with 2D projected coordinates.",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("artifacts/tat/player_embedding_projection.svg"),
        help="Output SVG scatter plot.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("artifacts/tat/player_embedding_projection.html"),
        help="Output HTML interactive scatter plot.",
    )
    parser.add_argument(
        "--label-top-n",
        type=int,
        default=35,
        help="Label top N players by n_matches.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=1,
        help="Drop players with fewer than this many matches before projection.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="pca,tsne",
        help="Comma-separated projection methods to run. Supported: pca,tsne.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Target t-SNE perplexity. Will be clipped for small sample sizes.",
    )
    parser.add_argument(
        "--tsne-learning-rate",
        type=str,
        default="auto",
        help="t-SNE learning rate. Use 'auto' or a positive float.",
    )
    parser.add_argument(
        "--tsne-max-iter",
        type=int,
        default=1000,
        help="Maximum number of t-SNE iterations.",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=42,
        help="Random seed for t-SNE reproducibility.",
    )
    parser.add_argument(
        "--tsne-pre-pca-dim",
        type=int,
        default=50,
        help="Reduce to this many PCA dimensions before t-SNE. Set 0 to disable.",
    )
    parser.add_argument(
        "--hover-feature-columns",
        type=str,
        default=",".join(DEFAULT_HOVER_FEATURES),
        help="Comma-separated numeric feature columns to summarize in interactive hover cards.",
    )
    return parser.parse_args()


def load_feature_config(cfg_path: Path) -> Dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_position_mapping(cfg: Dict) -> Dict[int, str]:
    pos_vocab = cfg.get("position_vocab", {})
    inv = {int(v): str(k) for k, v in pos_vocab.items()}
    return inv


def dominant_position(features_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(features_csv, usecols=["player_id", "position_id"], low_memory=False)
    df = df.dropna(subset=["player_id", "position_id"]).copy()
    df["player_id"] = df["player_id"].astype(int)
    df["position_id"] = df["position_id"].astype(int)

    def _mode(values: pd.Series) -> int:
        vc = values.value_counts()
        return int(vc.index[0]) if not vc.empty else 0

    out = df.groupby("player_id", as_index=False)["position_id"].agg(_mode)
    out.rename(columns={"position_id": "dominant_position_id"}, inplace=True)
    return out


def parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_methods(value: str) -> List[str]:
    methods = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not methods:
        raise ValueError("At least one projection method is required.")

    supported = {"pca", "tsne"}
    invalid = [method for method in methods if method not in supported]
    if invalid:
        raise ValueError(f"Unsupported projection methods: {invalid}. Supported: {sorted(supported)}")
    return methods


def parse_learning_rate(value: str) -> str | float:
    value = value.strip().lower()
    if value == "auto":
        return "auto"
    learning_rate = float(value)
    if learning_rate <= 0.0:
        raise ValueError("t-SNE learning rate must be positive.")
    return learning_rate


def pca_project(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    return coords.astype(np.float32)


def mode_string(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return "Unknown"
    vc = values.value_counts()
    return str(vc.index[0])


def join_unique(values: pd.Series) -> str:
    unique_values = sorted({str(v) for v in values.dropna() if str(v).strip()})
    return ", ".join(unique_values) if unique_values else "Unknown"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 100.0:
        return f"{value:.1f}"
    if abs(value) >= 10.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def feature_label(name: str) -> str:
    label = name.replace("_per90", " per90")
    label = label.replace("_", " ")
    return label.title()


def summarize_feature_list(values: List[str], preview_count: int = 8) -> str:
    if not values:
        return "None"
    if len(values) <= preview_count:
        return ", ".join(values)
    preview = ", ".join(values[:preview_count])
    return f"{preview}, ... (+{len(values) - preview_count} more)"


def build_feature_overview(feature_cfg: Dict) -> Dict[str, str]:
    continuous = [str(v) for v in feature_cfg.get("continuous_features", [])]
    categorical = [str(v) for v in feature_cfg.get("categorical_features", [])]
    gap_feature = str(feature_cfg.get("gap_feature", "days_since_prev_match"))
    all_features = continuous + categorical + [gap_feature]

    return {
        "count_line": (
            f"Model inputs: {len(continuous)} continuous, {len(categorical)} categorical, "
            f"1 gap feature ({len(all_features)} total)"
        ),
        "preview_line": f"Feature preview: {summarize_feature_list(all_features)}",
        "hover_line": (
            f"{len(continuous)} continuous + {len(categorical)} categorical + gap feature "
            f"({gap_feature})"
        ),
    }


def aggregate_player_metadata(
    features_csv: Path,
    hover_feature_columns: List[str],
) -> tuple[pd.DataFrame, List[str]]:
    available_columns = pd.read_csv(features_csv, nrows=0).columns.tolist()
    selected_hover_columns = [col for col in hover_feature_columns if col in available_columns]

    usecols = ["player_id", "player_name", "team_name", "season", "split"] + selected_hover_columns
    df = pd.read_csv(features_csv, usecols=usecols, low_memory=False)
    df["player_id"] = df["player_id"].astype(int)

    agg_spec: Dict[str, str | callable] = {
        "player_name": mode_string,
        "team_name": mode_string,
        "season": join_unique,
        "split": join_unique,
    }
    for col in selected_hover_columns:
        agg_spec[col] = "mean"

    out = df.groupby("player_id", as_index=False).agg(agg_spec)
    out.rename(
        columns={
            "team_name": "primary_team_name",
            "season": "seasons",
            "split": "splits",
        },
        inplace=True,
    )
    return out, selected_hover_columns


def tsne_2d(
    x: np.ndarray,
    perplexity: float,
    learning_rate: str | float,
    max_iter: int,
    random_state: int,
    pre_pca_dim: int,
) -> np.ndarray:
    n_samples, n_features = x.shape
    if n_samples < 3:
        raise ValueError("t-SNE requires at least 3 player embeddings.")

    x_tsne = x.astype(np.float64)
    if pre_pca_dim > 0:
        reduced_dim = min(int(pre_pca_dim), n_samples, n_features)
        if 2 < reduced_dim < n_features:
            x_tsne = pca_project(x_tsne, n_components=reduced_dim)

    clipped_perplexity = min(float(perplexity), float(n_samples - 1))
    if clipped_perplexity <= 0.0:
        raise ValueError("t-SNE perplexity must be positive after clipping.")

    coords = TSNE(
        n_components=2,
        perplexity=clipped_perplexity,
        learning_rate=learning_rate,
        max_iter=int(max_iter),
        init="pca",
        random_state=int(random_state),
        method="barnes_hut",
        angle=0.5,
        n_jobs=1,
    ).fit_transform(x_tsne)
    return coords.astype(np.float32)


def output_path_for_method(base_path: Path, method: str, methods: List[str]) -> Path:
    if len(methods) == 1:
        return base_path
    if method == "pca":
        return base_path
    return base_path.with_name(f"{base_path.stem}_{method}{base_path.suffix}")


def build_hover_text(
    df: pd.DataFrame,
    hover_feature_columns: List[str],
    feature_overview: Dict[str, str],
) -> pd.Series:
    hover_values: List[str] = []
    for row in df.itertuples(index=False):
        lines = [
            f"<b>{row.player_name}</b>",
            f"Player ID: {int(row.player_id)}",
            f"Position: {row.position_name}",
            f"Primary team: {row.primary_team_name}",
            f"Seasons: {row.seasons}",
            f"Splits: {row.splits}",
            f"Matches in EMA: {int(row.n_matches)}",
            f"Last match: {row.last_match_date}",
        ]
        for col in hover_feature_columns:
            lines.append(f"Avg {feature_label(col)}: {format_number(getattr(row, col))}")
        lines.append(f"Feature set: {feature_overview['hover_line']}")
        hover_values.append("<br>".join(lines))
    return pd.Series(hover_values, index=df.index, dtype="object")


def write_svg(
    df: pd.DataFrame,
    out_path: Path,
    label_top_n: int,
    title_suffix: str,
    axis_x_label: str,
    axis_y_label: str,
) -> None:
    width, height = 1300, 900
    margin_left, margin_right, margin_top, margin_bottom = 80, 260, 60, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    def map_x(v: float) -> float:
        return margin_left + (v - x_min) / (x_max - x_min) * plot_w

    def map_y(v: float) -> float:
        return margin_top + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    parts.append('<style>text{font-family:Arial,sans-serif;fill:#111827}</style>')
    parts.append(f'<text x="24" y="34" font-size="22" font-weight="700">Player Embedding Projection ({title_suffix})</text>')
    parts.append(
        '<text x="24" y="56" font-size="12">Each dot = one player EMA embedding, color = dominant position</text>'
    )

    # Axes
    x0, y0 = margin_left, margin_top + plot_h
    x1, y1 = margin_left + plot_w, margin_top
    parts.append(f'<line x1="{margin_left}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#9ca3af"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{y0}" stroke="#9ca3af"/>')
    parts.append(f'<text x="{margin_left + plot_w/2:.1f}" y="{height-20}" font-size="12">{axis_x_label}</text>')
    parts.append(
        f'<text x="18" y="{margin_top + plot_h/2:.1f}" font-size="12" transform="rotate(-90 18 {margin_top + plot_h/2:.1f})">{axis_y_label}</text>'
    )

    # Points
    for row in df.itertuples(index=False):
        cx = map_x(float(row.x))
        cy = map_y(float(row.y))
        color = COLOR_MAP.get(str(row.position_name), "#7f7f7f")
        parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.2" fill="{color}" fill-opacity="0.72"/>')

    # Labels (top N by matches)
    if label_top_n > 0:
        top = df.sort_values("n_matches", ascending=False).head(label_top_n)
        for row in top.itertuples(index=False):
            cx = map_x(float(row.x))
            cy = map_y(float(row.y))
            parts.append(f'<text x="{cx + 4:.2f}" y="{cy - 4:.2f}" font-size="10" fill="#111827">{int(row.player_id)}</text>')

    # Legend
    legend_x = margin_left + plot_w + 24
    legend_y = margin_top + 24
    parts.append(f'<text x="{legend_x}" y="{legend_y - 10}" font-size="14" font-weight="700">Position</text>')
    for i, (name, color) in enumerate(COLOR_MAP.items()):
        yy = legend_y + i * 26
        parts.append(f'<circle cx="{legend_x}" cy="{yy}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 14}" y="{yy + 4}" font-size="12">{name}</text>')

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_interactive_html(
    df: pd.DataFrame,
    out_path: Path,
    method: str,
    feature_overview: Dict[str, str],
) -> None:
    if method == "pca":
        title_suffix = "PCA 2D"
        axis_x_label, axis_y_label = "PC1", "PC2"
    else:
        title_suffix = "t-SNE 2D"
        axis_x_label, axis_y_label = "t-SNE 1", "t-SNE 2"

    fig = go.Figure()
    ordered_positions = list(COLOR_MAP.keys())
    remaining_positions = [p for p in df["position_name"].dropna().astype(str).unique() if p not in COLOR_MAP]
    for position_name in ordered_positions + sorted(remaining_positions):
        sub = df[df["position_name"] == position_name]
        if sub.empty:
            continue
        marker_size = np.clip(5.0 + np.log1p(sub["n_matches"].to_numpy(dtype=np.float32)) * 2.0, 6.0, 14.0)
        fig.add_trace(
            go.Scattergl(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                name=position_name,
                text=sub["player_name"],
                hovertext=sub["hover_html"],
                hovertemplate="%{hovertext}<extra></extra>",
                marker={
                    "color": COLOR_MAP.get(position_name, "#7f7f7f"),
                    "size": marker_size,
                    "opacity": 0.74,
                    "line": {"width": 0},
                },
            )
        )

    fig.update_layout(
        title={
            "text": (
                f"Player Embedding Projection ({title_suffix})"
                f"<br><sup>{feature_overview['count_line']}<br>{feature_overview['preview_line']}</sup>"
            ),
            "x": 0.03,
        },
        template="plotly_white",
        width=1300,
        height=900,
        legend_title_text="Position",
        hoverlabel={"align": "left"},
        margin={"l": 80, "r": 40, "t": 100, "b": 70},
    )
    fig.update_xaxes(title_text=axis_x_label, showgrid=True, gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#d1d5db")
    fig.update_yaxes(title_text=axis_y_label, showgrid=True, gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#d1d5db")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    hover_feature_columns = parse_csv_list(args.hover_feature_columns)
    feature_cfg = load_feature_config(args.feature_config_yaml)

    emb = pd.read_csv(args.player_embeddings_csv, low_memory=False)
    emb = emb[emb["n_matches"] >= int(args.min_matches)].copy()
    if emb.empty:
        raise ValueError("No players left after min_matches filter.")

    emb["player_id"] = emb["player_id"].astype(int)
    emb["vector"] = emb["e"].map(json.loads)
    x = np.asarray(emb["vector"].tolist(), dtype=np.float32)

    pos_df = dominant_position(args.features_csv)
    id_to_name = load_position_mapping(feature_cfg)
    feature_overview = build_feature_overview(feature_cfg)
    player_meta, selected_hover_features = aggregate_player_metadata(args.features_csv, hover_feature_columns)

    out = emb.merge(pos_df, on="player_id", how="left")
    out["dominant_position_id"] = out["dominant_position_id"].fillna(0).astype(int)
    out["position_name"] = out["dominant_position_id"].map(id_to_name).fillna("Unknown")
    out = out.merge(player_meta, on="player_id", how="left")
    out["player_name"] = out["player_name"].fillna(out["player_id"].astype(str))
    out["primary_team_name"] = out["primary_team_name"].fillna("Unknown")
    out["seasons"] = out["seasons"].fillna("Unknown")
    out["splits"] = out["splits"].fillna("Unknown")
    base = out[
        [
            "player_id",
            "player_name",
            "last_match_date",
            "n_matches",
            "ema_alpha",
            "dominant_position_id",
            "position_name",
            "primary_team_name",
            "seasons",
            "splits",
            *selected_hover_features,
        ]
    ].copy()

    projections: Dict[str, np.ndarray] = {}
    if "pca" in methods:
        projections["pca"] = pca_project(x, n_components=2)
    if "tsne" in methods:
        projections["tsne"] = tsne_2d(
            x=x,
            perplexity=float(args.tsne_perplexity),
            learning_rate=parse_learning_rate(args.tsne_learning_rate),
            max_iter=int(args.tsne_max_iter),
            random_state=int(args.tsne_random_state),
            pre_pca_dim=int(args.tsne_pre_pca_dim),
        )

    for method in methods:
        coords = projections[method]
        proj = base.copy()
        proj["projection_method"] = method
        proj["x"] = coords[:, 0]
        proj["y"] = coords[:, 1]
        proj["hover_html"] = build_hover_text(proj, selected_hover_features, feature_overview)

        output_csv = output_path_for_method(args.output_csv, method, methods)
        output_svg = output_path_for_method(args.output_svg, method, methods)
        output_html = output_path_for_method(args.output_html, method, methods)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        proj.to_csv(output_csv, index=False)

        if method == "pca":
            title_suffix = "PCA 2D"
            axis_x_label, axis_y_label = "PC1", "PC2"
        else:
            title_suffix = "t-SNE 2D"
            axis_x_label, axis_y_label = "t-SNE 1", "t-SNE 2"

        write_svg(
            proj,
            output_svg,
            label_top_n=int(args.label_top_n),
            title_suffix=title_suffix,
            axis_x_label=axis_x_label,
            axis_y_label=axis_y_label,
        )
        write_interactive_html(
            proj,
            output_html,
            method=method,
            feature_overview=feature_overview,
        )

        print(f"Wrote {method.upper()} projection CSV: {output_csv} (rows={len(proj):,})")
        print(f"Wrote {method.upper()} projection SVG: {output_svg}")
        print(f"Wrote {method.upper()} interactive HTML: {output_html}")


if __name__ == "__main__":
    main()
