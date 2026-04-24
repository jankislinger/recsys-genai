"""Tests for metrics module."""

import pytest

from recsys_genai.metrics import (
    coverage,
    diversity_at_k,
    evaluate_ranking,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_perfect():
    """Test recall with perfect predictions."""
    preds = [1, 2, 3, 4, 5]
    targets = {1, 2, 3}

    recall = recall_at_k(preds, targets, k=5)
    assert recall == 1.0  # All targets in top-5


def test_recall_at_k_partial():
    """Test recall with partial matches."""
    preds = [1, 2, 3, 4, 5]
    targets = {2, 6, 7}  # Only 2 is in predictions

    recall = recall_at_k(preds, targets, k=5)
    assert recall == pytest.approx(1 / 3)


def test_recall_at_k_position_independent():
    """Test that recall is position-independent."""
    preds = [1, 2, 3, 4, 5]
    targets = {2}

    recall1 = recall_at_k(preds, targets, k=5)

    preds2 = [5, 4, 3, 2, 1]  # Different order
    recall2 = recall_at_k(preds2, targets, k=5)

    assert recall1 == recall2  # Same recall regardless of order


def test_recall_at_k_with_k():
    """Test recall respects k parameter."""
    preds = [1, 2, 3, 4, 5]
    targets = {3}

    assert recall_at_k(preds, targets, k=2) == 0.0  # Target not in top-2
    assert recall_at_k(preds, targets, k=3) == 1.0  # Target in top-3


def test_ndcg_at_k_perfect():
    """Test NDCG with perfect ranking."""
    preds = [1, 2, 3, 4, 5]
    targets = {1, 2}  # Both at top

    ndcg = ndcg_at_k(preds, targets, k=5)
    assert ndcg == 1.0  # Perfect ranking


def test_ndcg_at_k_position_matters():
    """Test that NDCG is position-aware."""
    targets = {1}

    # Target at position 0 (best)
    preds1 = [1, 2, 3, 4, 5]
    ndcg1 = ndcg_at_k(preds1, targets, k=5)

    # Target at position 4 (worse)
    preds2 = [2, 3, 4, 5, 1]
    ndcg2 = ndcg_at_k(preds2, targets, k=5)

    assert ndcg1 > ndcg2  # Earlier position = higher NDCG


def test_ndcg_at_k_empty_targets():
    """Test NDCG with empty targets."""
    preds = [1, 2, 3]
    targets = set()

    assert ndcg_at_k(preds, targets, k=3) == 0.0


def test_mean_reciprocal_rank():
    """Test MRR calculation."""
    preds = [1, 2, 3, 4, 5]

    # Target at position 0
    assert mean_reciprocal_rank(preds, {1}) == 1.0

    # Target at position 2
    assert mean_reciprocal_rank(preds, {3}) == pytest.approx(1 / 3)

    # Target not in list
    assert mean_reciprocal_rank(preds, {10}) == 0.0


def test_coverage():
    """Test catalog coverage calculation."""
    all_preds = [[1, 2, 3], [2, 3, 4], [1, 5, 6]]

    # Unique items: {1, 2, 3, 4, 5, 6} = 6 out of 10
    cov = coverage(all_preds, num_items=10)
    assert cov == 0.6


def test_coverage_full():
    """Test 100% coverage."""
    all_preds = [[i] for i in range(1, 11)]

    cov = coverage(all_preds, num_items=10)
    assert cov == 1.0


def test_diversity_at_k():
    """Test diversity calculation."""
    preds = [1, 2, 3]
    features = {
        1: {"A", "B"},  # Shares A with 2
        2: {"A", "C"},  # Shares A with 1
        3: {"D", "E"},  # Completely different
    }

    div = diversity_at_k(preds, features, k=3)

    # Pairs: (1,2) have Jaccard dist = 1 - 1/3 = 2/3
    #        (1,3) have Jaccard dist = 1 - 0/4 = 1.0
    #        (2,3) have Jaccard dist = 1 - 0/4 = 1.0
    # Average: (2/3 + 1.0 + 1.0) / 3 ≈ 0.889

    assert 0.8 < div < 0.9


def test_diversity_at_k_identical():
    """Test diversity with identical items."""
    preds = [1, 2]
    features = {
        1: {"A", "B"},
        2: {"A", "B"},  # Identical
    }

    div = diversity_at_k(preds, features, k=2)
    assert div == 0.0  # No diversity


def test_diversity_at_k_single_item():
    """Test diversity with single item."""
    preds = [1]
    features = {1: {"A"}}

    div = diversity_at_k(preds, features, k=1)
    assert div == 0.0  # Cannot measure diversity with 1 item


def test_evaluate_ranking():
    """Test comprehensive ranking evaluation."""
    predictions = {1: [10, 20, 30], 2: [40, 50, 60]}
    ground_truth = {1: {20, 30}, 2: {40}}

    metrics = evaluate_ranking(predictions, ground_truth, k_values=[3])

    assert "recall@3" in metrics
    assert "ndcg@3" in metrics
    assert "mrr" in metrics

    # User 1: recall = 2/2 = 1.0
    # User 2: recall = 1/1 = 1.0
    # Average: 1.0
    assert metrics["recall@3"] == 1.0


def test_evaluate_ranking_partial():
    """Test ranking evaluation with partial matches."""
    predictions = {1: [1, 2, 3], 2: [4, 5, 6]}
    ground_truth = {
        1: {2},  # One match
        2: {99},  # No match
    }

    metrics = evaluate_ranking(predictions, ground_truth, k_values=[3])

    # User 1: recall = 1.0, User 2: recall = 0.0
    # Average: 0.5
    assert metrics["recall@3"] == 0.5
