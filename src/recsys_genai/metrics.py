"""Evaluation metrics for recommendation systems.

Implements common offline metrics including Recall@K, NDCG@K, and diversity measures.
"""

import numpy as np


def recall_at_k(predictions: list[int], targets: set[int], k: int = 10) -> float:
    """Calculate Recall@K for a single user.

    Measures the proportion of relevant items that appear in top-K recommendations.

    Args:
        predictions: Ordered list of recommended item IDs
        targets: Set of ground truth relevant item IDs
        k: Number of top recommendations to consider

    Returns:
        Recall@K score between 0 and 1

    Example:
        >>> preds = [1, 2, 3, 4, 5]
        >>> targets = {2, 6, 7}
        >>> recall_at_k(preds, targets, k=5)
        0.3333333333333333
        >>> recall_at_k(preds, targets, k=2)
        0.3333333333333333
    """
    if not targets:
        return 0.0

    top_k = set(predictions[:k])
    hits = len(top_k & targets)
    return hits / len(targets)


def ndcg_at_k(predictions: list[int], targets: set[int], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    Measures ranking quality with position-based discounting.

    Args:
        predictions: Ordered list of recommended item IDs
        targets: Set of ground truth relevant item IDs
        k: Number of top recommendations to consider

    Returns:
        NDCG@K score between 0 and 1

    Example:
        >>> preds = [1, 2, 3, 4, 5]
        >>> targets = {1, 2}
        >>> score = ndcg_at_k(preds, targets, k=5)
        >>> bool(score == 1.0)
        True
    """
    if not targets:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(predictions[:k]):
        if item in targets:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed

    # Calculate ideal DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(targets))))

    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(predictions: list[int], targets: set[int]) -> float:
    """Calculate Mean Reciprocal Rank.

    Returns the reciprocal of the rank of the first relevant item.

    Args:
        predictions: Ordered list of recommended item IDs
        targets: Set of ground truth relevant item IDs

    Returns:
        MRR score between 0 and 1

    Example:
        >>> preds = [1, 2, 3, 4, 5]
        >>> targets = {3}
        >>> mean_reciprocal_rank(preds, targets)
        0.3333333333333333
    """
    for i, item in enumerate(predictions):
        if item in targets:
            return 1.0 / (i + 1)
    return 0.0


def coverage(all_predictions: list[list[int]], num_items: int) -> float:
    """Calculate catalog coverage.

    Measures the percentage of catalog items that appear in recommendations.

    Args:
        all_predictions: List of recommendation lists for all users
        num_items: Total number of items in catalog

    Returns:
        Coverage score between 0 and 1

    Example:
        >>> preds = [[1, 2, 3], [2, 3, 4], [1, 5, 6]]
        >>> coverage(preds, num_items=10)
        0.6
    """
    recommended_items = set()
    for pred_list in all_predictions:
        recommended_items.update(pred_list)

    return len(recommended_items) / num_items


def diversity_at_k(predictions: list[int], item_features: dict[int, set], k: int = 10) -> float:
    """Calculate diversity of top-K recommendations.

    Measures average pairwise dissimilarity based on item features (e.g., genres).

    Args:
        predictions: Ordered list of recommended item IDs
        item_features: Dictionary mapping item_id to set of features
        k: Number of top recommendations to consider

    Returns:
        Diversity score between 0 and 1

    Example:
        >>> preds = [1, 2, 3]
        >>> features = {1: {'A', 'B'}, 2: {'A', 'C'}, 3: {'D', 'E'}}
        >>> score = diversity_at_k(preds, features, k=3)
        >>> 0 <= score <= 1
        True
    """
    top_k = predictions[:k]

    if len(top_k) < 2:
        return 0.0

    total_dissimilarity = 0.0
    pairs = 0

    for i in range(len(top_k)):
        for j in range(i + 1, len(top_k)):
            item_i = top_k[i]
            item_j = top_k[j]

            if item_i in item_features and item_j in item_features:
                features_i = item_features[item_i]
                features_j = item_features[item_j]

                # Jaccard dissimilarity
                intersection = len(features_i & features_j)
                union = len(features_i | features_j)

                dissimilarity = 1 - (intersection / union if union > 0 else 0)
                total_dissimilarity += dissimilarity
                pairs += 1

    return total_dissimilarity / pairs if pairs > 0 else 0.0


def evaluate_ranking(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k_values: list[int] = [5, 10, 20],
) -> dict[str, float]:
    """Evaluate ranking metrics across all users.

    Args:
        predictions: Dictionary mapping user_id to ordered recommendation list
        ground_truth: Dictionary mapping user_id to set of relevant items
        k_values: List of K values to evaluate

    Returns:
        Dictionary of metric names to average scores

    Example:
        >>> preds = {1: [10, 20, 30], 2: [40, 50, 60]}
        >>> truth = {1: {20, 30}, 2: {40}}
        >>> metrics = evaluate_ranking(preds, truth, k_values=[3])
        >>> 'recall@3' in metrics and 'ndcg@3' in metrics
        True
    """
    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"ndcg@{k}": [] for k in k_values})
    results["mrr"] = []

    for user_id in predictions:
        if user_id not in ground_truth:
            continue

        preds = predictions[user_id]
        targets = ground_truth[user_id]

        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(preds, targets, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(preds, targets, k))

        results["mrr"].append(mean_reciprocal_rank(preds, targets))

    # Average across users
    return {metric: np.mean(scores) for metric, scores in results.items()}
