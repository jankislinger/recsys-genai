"""Tests for data_utils module."""

import polars as pl

from recsys_genai.data_utils import (
    SequenceDataset,
    create_user_item_matrix,
    get_user_sequences,
    sample_negative_items,
)


def test_create_user_item_matrix():
    """Test binary interaction matrix creation."""
    ratings = pl.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "movie_id": [10, 20, 10, 30, 40],
            "rating": [5.0, 3.0, 4.5, 2.0, 4.0],
        }
    )

    interactions = create_user_item_matrix(ratings, min_rating=4.0)

    assert len(interactions) == 3  # Only ratings >= 4.0
    assert set(interactions["user_id"].to_list()) == {1, 2, 3}
    assert 20 not in interactions["movie_id"].to_list()  # Rating 3.0 excluded


def test_get_user_sequences():
    """Test user sequence extraction."""
    ratings = pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3],
            "movie_id": [10, 20, 30, 40, 50, 60],
            "timestamp": [1000, 2000, 3000, 1500, 2500, 1000],
        }
    )

    sequences = get_user_sequences(ratings, max_len=10, min_len=2)

    assert 1 in sequences
    assert 2 in sequences
    assert 3 not in sequences  # Only 1 item, below min_len

    assert sequences[1] == [10, 20, 30]  # Chronological order
    assert sequences[2] == [40, 50]


def test_get_user_sequences_truncation():
    """Test sequence truncation to max_len."""
    ratings = pl.DataFrame(
        {
            "user_id": [1] * 10,
            "movie_id": list(range(100, 110)),
            "timestamp": list(range(1000, 1010)),
        }
    )

    sequences = get_user_sequences(ratings, max_len=5, min_len=2)

    assert len(sequences[1]) == 5
    assert sequences[1] == [105, 106, 107, 108, 109]  # Most recent 5


def test_sequence_dataset():
    """Test PyTorch dataset for sequences."""
    sequences = {1: [10, 20, 30, 40], 2: [5, 15, 25]}

    dataset = SequenceDataset(sequences, max_len=3)

    assert len(dataset) == 2

    # Check first sequence
    seq, target = dataset[0]
    assert seq.shape == (3,)
    assert target.item() == 40  # Last item is target
    assert seq.tolist() == [10, 20, 30]  # All but last

    # Check second sequence (should be padded)
    seq, target = dataset[1]
    assert seq.shape == (3,)
    assert target.item() == 25
    assert seq.tolist() == [0, 5, 15]  # Padded with 0


def test_sequence_dataset_long_sequence():
    """Test dataset with sequence longer than max_len."""
    sequences = {1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    dataset = SequenceDataset(sequences, max_len=4)

    seq, target = dataset[0]
    assert seq.shape == (4,)
    assert target.item() == 10
    # Should keep last max_len items before target
    assert seq.tolist() == [6, 7, 8, 9]


def test_sample_negative_items():
    """Test negative item sampling."""
    positives = {1, 2, 3}
    negatives = sample_negative_items(positives, num_items=10, num_negatives=5, seed=42)

    assert len(negatives) == 5
    assert len(set(negatives) & positives) == 0  # No overlap with positives
    assert all(1 <= item <= 10 for item in negatives)


def test_sample_negative_items_limited():
    """Test negative sampling when few candidates available."""
    positives = {1, 2, 3, 4, 5, 6, 7, 8}
    negatives = sample_negative_items(positives, num_items=10, num_negatives=5, seed=42)

    # Only 2 candidates available (9, 10), but requested 5
    assert len(negatives) <= 2
    assert len(set(negatives) & positives) == 0


def test_sample_negative_items_reproducibility():
    """Test that sampling is reproducible with same seed."""
    positives = {1, 2, 3}

    negatives1 = sample_negative_items(positives, num_items=20, num_negatives=10, seed=42)
    negatives2 = sample_negative_items(positives, num_items=20, num_negatives=10, seed=42)

    assert negatives1 == negatives2
