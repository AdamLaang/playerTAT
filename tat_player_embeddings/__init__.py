__all__ = [
    "PlayerEmbeddingIndex",
    "build_player_embedding_index",
    "resolve_player",
    "search_player_neighbors",
]


def __getattr__(name: str):
    if name in __all__:
        from tat_player_embeddings import player_neighbors

        return getattr(player_neighbors, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
