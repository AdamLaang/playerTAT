from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search nearest neighbours for a player embedding.")
    parser.add_argument(
        "--player-embeddings-csv",
        type=Path,
        default=Path("artifacts/tat/player_embeddings_ema.csv"),
        help="CSV from embed.py containing one EMA embedding vector per player.",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("data/tat_player_features_with_splits.csv"),
        help="Feature table used to recover player names and context metadata.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Player name to search for (case-insensitive).",
    )
    parser.add_argument(
        "--player-id",
        type=int,
        default=None,
        help="Optional player_id to disambiguate duplicate player names.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest neighbours to return.",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include the queried player in the output ranking.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save the neighbour table as CSV.",
    )
    return parser.parse_args()


def mode_string(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return "Unknown"
    counts = values.value_counts()
    return str(counts.index[0])


def join_unique(values: pd.Series) -> str:
    unique_values = sorted({str(v) for v in values.dropna() if str(v).strip()})
    return ", ".join(unique_values) if unique_values else "Unknown"


def normalize_name(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def aggregate_player_metadata(features_csv: Path) -> pd.DataFrame:
    usecols = ["player_id", "player_name", "team_name", "season", "split"]
    df = pd.read_csv(features_csv, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["player_id"]).copy()
    df["player_id"] = df["player_id"].astype(int)

    out = (
        df.groupby("player_id", as_index=False)
        .agg(
            {
                "player_name": mode_string,
                "team_name": mode_string,
                "season": join_unique,
                "split": join_unique,
            }
        )
        .rename(
            columns={
                "team_name": "primary_team_name",
                "season": "seasons",
                "split": "splits",
            }
        )
    )
    return out


def parse_embedding_series(values: pd.Series) -> np.ndarray:
    vectors = [np.asarray(json.loads(value), dtype=np.float32) for value in values]
    if not vectors:
        raise ValueError("No player embeddings were found in the provided CSV.")
    return np.vstack(vectors)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return x / norms


@dataclass
class PlayerEmbeddingIndex:
    players: pd.DataFrame
    embeddings: np.ndarray
    normalized_embeddings: np.ndarray


def build_player_embedding_index(
    player_embeddings_csv: Path,
    features_csv: Path,
) -> PlayerEmbeddingIndex:
    emb = pd.read_csv(player_embeddings_csv, low_memory=False)
    required_columns = {"player_id", "last_match_date", "n_matches", "ema_alpha", "e"}
    missing = required_columns - set(emb.columns)
    if missing:
        raise ValueError(f"Player embedding CSV is missing required columns: {sorted(missing)}")

    emb = emb.dropna(subset=["player_id", "e"]).copy()
    emb["player_id"] = emb["player_id"].astype(int)
    emb["last_match_date"] = pd.to_datetime(emb["last_match_date"], utc=True, errors="coerce")

    metadata = aggregate_player_metadata(features_csv)
    players = emb.merge(metadata, on="player_id", how="left")
    players["player_name"] = players["player_name"].fillna(players["player_id"].astype(str))
    players["primary_team_name"] = players["primary_team_name"].fillna("Unknown")
    players["seasons"] = players["seasons"].fillna("Unknown")
    players["splits"] = players["splits"].fillna("Unknown")
    players["name_key"] = players["player_name"].map(normalize_name)

    embeddings = parse_embedding_series(players["e"])
    normalized_embeddings = normalize_rows(embeddings)
    return PlayerEmbeddingIndex(
        players=players.reset_index(drop=True),
        embeddings=embeddings,
        normalized_embeddings=normalized_embeddings,
    )


def _format_candidate_table(candidates: pd.DataFrame) -> str:
    preview = candidates[
        ["player_id", "player_name", "primary_team_name", "seasons", "n_matches"]
    ].sort_values(["n_matches", "player_id"], ascending=[False, True])
    return preview.to_string(index=False)


def resolve_player(
    index: PlayerEmbeddingIndex,
    query: str,
    player_id: Optional[int] = None,
) -> pd.Series:
    players = index.players

    if player_id is not None:
        matches = players[players["player_id"] == int(player_id)]
        if matches.empty:
            raise ValueError(f"player_id={player_id} was not found in the player embedding index.")
        if query and normalize_name(matches.iloc[0]["player_name"]) != normalize_name(query):
            raise ValueError(
                f'player_id={player_id} does not match query="{query}". '
                f'Found "{matches.iloc[0]["player_name"]}" instead.'
            )
        return matches.iloc[0]

    query_key = normalize_name(query)
    exact = players[players["name_key"] == query_key]
    if len(exact) == 1:
        return exact.iloc[0]
    if len(exact) > 1:
        raise ValueError(
            "Multiple players match this exact name. Re-run with --player-id.\n"
            f"{_format_candidate_table(exact)}"
        )

    partial = players[players["name_key"].str.contains(query_key, regex=False)]
    if len(partial) == 1:
        return partial.iloc[0]
    if len(partial) > 1:
        raise ValueError(
            "Multiple players partially match this query. Re-run with a more specific name or --player-id.\n"
            f"{_format_candidate_table(partial)}"
        )

    unique_names = players["player_name"].dropna().astype(str).unique().tolist()
    suggestions = difflib.get_close_matches(query, unique_names, n=5, cutoff=0.6)
    suggestion_text = f" Close matches: {', '.join(suggestions)}." if suggestions else ""
    raise ValueError(f'No player found for query "{query}".{suggestion_text}')


def search_player_neighbors(
    query: str,
    index: PlayerEmbeddingIndex,
    top_k: int = 10,
    player_id: Optional[int] = None,
    include_self: bool = False,
) -> Tuple[pd.Series, pd.DataFrame]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    query_row = resolve_player(index=index, query=query, player_id=player_id)
    query_idx = int(query_row.name)

    scores = index.normalized_embeddings @ index.normalized_embeddings[query_idx]
    if not include_self:
        scores[query_idx] = -np.inf

    available = len(scores) if include_self else max(len(scores) - 1, 0)
    if available == 0:
        raise ValueError("The player embedding index must contain at least two players.")

    top_k = min(int(top_k), available)
    top_idx = np.argpartition(-scores, kth=top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    neighbors = index.players.iloc[top_idx].copy()
    neighbors.insert(0, "rank", np.arange(1, len(neighbors) + 1))
    neighbors.insert(1, "cosine_similarity", scores[top_idx].astype(np.float32))

    output_columns = [
        "rank",
        "cosine_similarity",
        "player_id",
        "player_name",
        "primary_team_name",
        "seasons",
        "splits",
        "n_matches",
        "last_match_date",
        "ema_alpha",
    ]
    return query_row, neighbors[output_columns].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    index = build_player_embedding_index(
        player_embeddings_csv=args.player_embeddings_csv,
        features_csv=args.features_csv,
    )
    query_row, neighbors = search_player_neighbors(
        query=args.query,
        index=index,
        top_k=int(args.top_k),
        player_id=args.player_id,
        include_self=bool(args.include_self),
    )

    print(
        f'Query player: {query_row["player_name"]} '
        f'(player_id={int(query_row["player_id"])}, '
        f'team={query_row["primary_team_name"]}, '
        f'n_matches={int(query_row["n_matches"])})'
    )
    print(neighbors.to_string(index=False))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        neighbors.to_csv(args.output_csv, index=False)
        print(f"Wrote neighbours: {args.output_csv}")


if __name__ == "__main__":
    main()
