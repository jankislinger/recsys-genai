"""Recommender Systems in the Age of Generative AI - Workshop Materials.

This package contains utilities, models, and helper functions for the workshop
"From Matrix Factorization to Generative Recommendation Pages".
"""

from recsys_genai.data_utils import (
    SequenceDataset,
    create_user_item_matrix,
    get_user_sequences,
    load_movielens,
    sample_negative_items,
)
from recsys_genai.llm_utils import (
    check_ollama_available,
    ollama_embed,
    ollama_generate,
    ollama_generate_json,
)
from recsys_genai.metrics import (
    coverage,
    diversity_at_k,
    evaluate_ranking,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from recsys_genai.models import (
    EASE,
    MatrixFactorization,
)
from recsys_genai.sasrec import (
    SASRec,
    train_sasrec,
)

__all__ = [
    # Data utilities
    "load_movielens",
    "create_user_item_matrix",
    "get_user_sequences",
    "SequenceDataset",
    "sample_negative_items",
    # LLM utilities
    "ollama_generate",
    "ollama_generate_json",
    "ollama_embed",
    "check_ollama_available",
    # Metrics
    "recall_at_k",
    "ndcg_at_k",
    "mean_reciprocal_rank",
    "coverage",
    "diversity_at_k",
    "evaluate_ranking",
    # Models
    "MatrixFactorization",
    "EASE",
    "SASRec",
    "train_sasrec",
]
