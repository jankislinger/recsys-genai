"""Visualization utilities for recommendation analysis."""

from typing import Optional

import numpy as np
import plotnine as p9
import polars as pl


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: Optional[list[str]] = None,
    highlight_indices: Optional[list[int]] = None,
    method: str = "pca",
) -> p9.ggplot:
    """Plot 2D projection of embeddings.

    Args:
        embeddings: (n_items, embedding_dim) array
        labels: Optional list of item labels
        highlight_indices: Optional indices to highlight
        method: Dimensionality reduction method ("pca" or "tsne")

    Returns:
        plotnine ggplot object

    Example:
        >>> embeddings = np.random.randn(10, 50)
        >>> plot = plot_embeddings_2d(embeddings, method="pca")
        >>> isinstance(plot, p9.ggplot)
        True
    """
    from sklearn.decomposition import PCA

    if method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
    else:
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)

    plot_df = pl.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": labels if labels else [""] * len(coords),
            "highlight": [i in (highlight_indices or []) for i in range(len(coords))],
        }
    )

    plot = (
        p9.ggplot(plot_df, p9.aes(x="x", y="y", color="highlight"))
        + p9.geom_point(alpha=0.6, size=2)
        + p9.scale_color_manual(values=["steelblue", "orange"])
        + p9.labs(
            title=f"Item Embeddings ({method.upper()})",
            x=f"{method.upper()} 1",
            y=f"{method.upper()} 2",
        )
        + p9.theme_minimal()
        + p9.theme(legend_position="none")
    )

    return plot


def plot_metrics_comparison(
    metrics_dict: dict[str, dict[str, float]], metric_name: str = "recall"
) -> p9.ggplot:
    """Plot comparison of metrics across different models.

    Args:
        metrics_dict: Dictionary mapping model_name to metrics dictionary
        metric_name: Name of metric to plot (e.g., "recall", "ndcg")

    Returns:
        plotnine ggplot object

    Example:
        >>> metrics = {
        ...     "MF": {"recall@10": 0.15, "ndcg@10": 0.12},
        ...     "EASE": {"recall@10": 0.18, "ndcg@10": 0.14}
        ... }
        >>> plot = plot_metrics_comparison(metrics, metric_name="recall")
        >>> isinstance(plot, p9.ggplot)
        True
    """
    rows = []
    for model_name, metrics in metrics_dict.items():
        for metric_key, value in metrics.items():
            if metric_name in metric_key.lower():
                rows.append({"model": model_name, "metric": metric_key, "value": value})

    plot_df = pl.DataFrame(rows)

    return (
        p9.ggplot(plot_df, p9.aes(x="metric", y="value", fill="model"))
        + p9.geom_col(position="dodge", alpha=0.8)
        + p9.labs(
            title=f"Model Comparison: {metric_name.capitalize()}",
            x="Metric",
            y="Score",
            fill="Model",
        )
        + p9.theme_minimal()
        + p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1))
    )
