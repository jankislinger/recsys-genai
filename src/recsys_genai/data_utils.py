"""Data loading and preprocessing utilities for MovieLens dataset.

This module provides helper functions for loading and preparing MovieLens data
for recommendation system experiments.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


def load_movielens(
    data_dir: str = "data",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load MovieLens datasets (movies, ratings, tags, links).

    Args:
        data_dir: Directory containing parquet files

    Returns:
        Tuple of (movies, ratings, tags, links) DataFrames

    Example:
        >>> movies, ratings, tags, links = load_movielens("data")  # doctest: +SKIP
        >>> print(movies.columns)  # doctest: +SKIP
        ['movie_id', 'title', 'genres']
    """
    data_path = Path(data_dir)
    movies = pl.read_parquet(data_path / "movies.parquet")
    ratings = pl.read_parquet(data_path / "ratings.parquet")
    tags = pl.read_parquet(data_path / "tags.parquet")
    links = pl.read_parquet(data_path / "links.parquet")
    return movies, ratings, tags, links


def create_user_item_matrix(ratings: pl.DataFrame, min_rating: float = 4.0) -> pl.DataFrame:
    """Create binary user-item interaction matrix.

    Args:
        ratings: Ratings DataFrame with user_id, movie_id, rating columns
        min_rating: Minimum rating to consider as positive interaction

    Returns:
        DataFrame with user_id, movie_id for positive interactions

    Example:
        >>> ratings = pl.DataFrame({
        ...     "user_id": [1, 1, 2],
        ...     "movie_id": [10, 20, 10],
        ...     "rating": [5.0, 3.0, 4.5]
        ... })
        >>> interactions = create_user_item_matrix(ratings, min_rating=4.0)
        >>> len(interactions)
        2
    """
    return ratings.filter(pl.col("rating") >= min_rating).select(["user_id", "movie_id"])


def get_user_sequences(
    ratings: pl.DataFrame, max_len: int = 50, min_len: int = 5
) -> dict[int, list[int]]:
    """Extract sequential user interaction histories.

    Args:
        ratings: Ratings DataFrame with user_id, movie_id, timestamp
        max_len: Maximum sequence length
        min_len: Minimum sequence length (filter out shorter sequences)

    Returns:
        Dictionary mapping user_id to list of movie_ids in chronological order

    Example:
        >>> ratings = pl.DataFrame({
        ...     "user_id": [1, 1, 1],
        ...     "movie_id": [10, 20, 30],
        ...     "timestamp": [1000, 2000, 3000]
        ... })
        >>> seqs = get_user_sequences(ratings, max_len=10, min_len=2)
        >>> seqs[1]
        [10, 20, 30]
    """
    sequences = {}
    for user_id, group in ratings.sort("timestamp").group_by("user_id"):
        seq = group.select("movie_id").to_series().to_list()
        if len(seq) >= min_len:
            sequences[user_id[0]] = seq[-max_len:]  # Keep most recent
    return sequences


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequential recommendation.

    Creates training examples for next-item prediction tasks.
    Each example is a sequence with the last item as the target.

    Args:
        sequences: Dictionary mapping user_id to item sequences
        max_len: Maximum sequence length (pad/truncate)

    Example:
        >>> sequences = {1: [10, 20, 30, 40], 2: [5, 15]}
        >>> dataset = SequenceDataset(sequences, max_len=3)
        >>> len(dataset)
        2
        >>> seq, target = dataset[0]
        >>> seq.shape
        torch.Size([3])
    """

    def __init__(self, sequences: dict[int, list[int]], max_len: int = 50):
        self.sequences = list(sequences.values())
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]

        # Use all but last as input, last as target
        if len(seq) > self.max_len:
            seq = seq[-self.max_len - 1 :]  # Keep last max_len+1 items

        input_seq = seq[:-1]
        target = seq[-1]

        # Pad sequence
        if len(input_seq) < self.max_len:
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def sample_negative_items(
    positive_items: set[int], num_items: int, num_negatives: int = 100, seed: Optional[int] = None
) -> list[int]:
    """Sample negative items for evaluation.

    Args:
        positive_items: Set of items the user has interacted with
        num_items: Total number of items in catalog
        num_negatives: Number of negative samples to draw
        seed: Random seed for reproducibility

    Returns:
        List of negative item IDs

    Example:
        >>> positives = {1, 2, 3}
        >>> negatives = sample_negative_items(positives, num_items=10, num_negatives=3, seed=42)
        >>> len(negatives)
        3
        >>> any(n in positives for n in negatives)
        False
    """
    rng = np.random.RandomState(seed)
    candidates = list(set(range(1, num_items + 1)) - positive_items)
    return rng.choice(candidates, size=min(num_negatives, len(candidates)), replace=False).tolist()
