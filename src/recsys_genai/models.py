"""Classical and neural recommendation models.

Implements Matrix Factorization, EASE, and SASRec architectures.
"""


import numpy as np


class MatrixFactorization:
    """Matrix Factorization for collaborative filtering.

    Factorizes the user-item matrix into user and item latent factors.
    Based on Koren et al. 2009 (Netflix Prize).

    Args:
        num_users: Number of users
        num_items: Number of items
        num_factors: Dimensionality of latent factors
        learning_rate: SGD learning rate
        reg: L2 regularization parameter

    Example:
        >>> model = MatrixFactorization(num_users=100, num_items=50, num_factors=10)
        >>> model.fit(user_ids=[1, 2], item_ids=[10, 20], ratings=[5.0, 4.0], epochs=1)
        >>> score = model.predict(user_id=1, item_id=10)
        >>> isinstance(score, float)
        True
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 20,
        learning_rate: float = 0.01,
        reg: float = 0.01,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.lr = learning_rate
        self.reg = reg

        # Initialize factor matrices
        self.user_factors = np.random.normal(0, 0.1, (num_users, num_factors))
        self.item_factors = np.random.normal(0, 0.1, (num_items, num_factors))
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_bias = 0.0

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair."""
        pred = self.global_bias
        pred += self.user_bias[user_id] + self.item_bias[item_id]
        pred += np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return float(pred)

    def fit(
        self,
        user_ids: list[int],
        item_ids: list[int],
        ratings: list[float],
        epochs: int = 10,
        verbose: bool = False,
    ):
        """Train model using SGD."""
        self.global_bias = np.mean(ratings)

        for epoch in range(epochs):
            total_error = 0.0

            for u, i, r in zip(user_ids, item_ids, ratings):
                # Prediction error
                pred = self.predict(u, i)
                error = r - pred
                total_error += error**2

                # Update biases
                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])

                # Update factors
                u_factors = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (
                    error * self.item_factors[i] - self.reg * u_factors
                )
                self.item_factors[i] += self.lr * (
                    error * u_factors - self.reg * self.item_factors[i]
                )

            if verbose and (epoch + 1) % 5 == 0:
                rmse = np.sqrt(total_error / len(ratings))
                print(f"Epoch {epoch + 1}: RMSE = {rmse:.4f}")


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
        # Convert to dense if sparse
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Gram matrix (ensure float type)
        G = (X.T @ X).astype(np.float64)

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
