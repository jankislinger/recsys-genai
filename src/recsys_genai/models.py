"""Classical and neural recommendation models.

Implements EASE and SASRec architectures.
"""

import numpy as np


class EASE:
    """Embarrassingly Shallow Autoencoder for Sparse data.

    Simple linear model that learns item-item similarity matrix.
    Based on Steck 2019 (WWW).

    Args:
        reg: Regularization parameter (controls diagonal suppression)

    Example:
        >>> import scipy.sparse as sp
        >>> X = sp.csr_matrix([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        >>> model = EASE(reg=100.0)
        >>> model.fit(X)
        >>> recs = model.recommend_for_user(user_idx=0, X=X, k=2)
        >>> len(recs)
        2
    """

    def __init__(self, reg: float = 100.0):
        self.reg = reg
        self.B = None

    def fit(self, X):
        """Fit EASE model.

        Args:
            X: User-item interaction matrix (sparse or dense)
        """
        # Gram matrix
        G = X.T @ X

        # Convert to dense if sparse
        if hasattr(G, "toarray"):
            G = G.toarray()

        # Ensure float64
        G = G.astype(np.float64)

        # Add regularization to diagonal
        diag_indices = np.diag_indices_from(G)
        G[diag_indices] += self.reg

        # Inverse
        P = np.linalg.inv(G)

        # EASE weights
        self.B = P / -np.diag(P)
        self.B[diag_indices] = 0.0

    def recommend_for_user(self, user_idx: int, X, k: int = 10) -> list[int]:
        """Generate top-K recommendations for a user.

        Args:
            user_idx: User index
            X: User-item interaction matrix
            k: Number of recommendations

        Returns:
            List of item indices
        """
        if hasattr(X, "toarray"):
            X = X.toarray()

        scores = X[user_idx] @ self.B

        # Mask already interacted items
        scores[X[user_idx] > 0] = -np.inf

        # Top-K
        top_k_items = np.argsort(-scores)[:k]
        return top_k_items.tolist()
