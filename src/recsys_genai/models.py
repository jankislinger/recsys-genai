"""Classical and neural recommendation models.

Implements Matrix Factorization, EASE, and SASRec architectures.
"""

from itertools import batched

import numpy as np
import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
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
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.lr = learning_rate
        self.reg = reg

        # Initialize embeddings as nn.Embedding layers
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)
        self.user_bias_embedding = nn.Embedding(num_users, 1)
        self.item_bias_embedding = nn.Embedding(num_items, 1)

        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.user_bias_embedding.weight)
        nn.init.zeros_(self.item_bias_embedding.weight)

        # Global bias as a parameter
        self._global_bias = nn.Parameter(torch.zeros(1))

    @property
    def global_bias(self) -> float:
        """Get global bias as float."""
        return self._global_bias.item()

    @property
    def user_factors(self) -> np.ndarray:
        """Get user factors as numpy array.

        Note: This is read-only. To modify, use:
            model.user_embedding.weight.data[idx] = torch.tensor(...)
        """
        return self.user_embedding.weight.detach().cpu().numpy()

    @property
    def item_factors(self) -> np.ndarray:
        """Get item factors as numpy array.

        Note: This is read-only. To modify, use:
            model.item_embedding.weight.data[idx] = torch.tensor(...)
        """
        return self.item_embedding.weight.detach().cpu().numpy()

    @property
    def user_bias(self) -> np.ndarray:
        """Get user bias as numpy array.

        Note: This is read-only. To modify, use:
            model.user_bias_embedding.weight.data[idx, 0] = value
        """
        return self.user_bias_embedding.weight.detach().cpu().numpy().squeeze()

    @property
    def item_bias(self) -> np.ndarray:
        """Get item bias as numpy array.

        Note: This is read-only. To modify, use:
            model.item_bias_embedding.weight.data[idx, 0] = value
        """
        return self.item_bias_embedding.weight.detach().cpu().numpy().squeeze()

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair."""
        with torch.no_grad():
            user_factor = self.user_embedding.weight[user_id]
            item_factor = self.item_embedding.weight[item_id]
            u_bias = self.user_bias_embedding.weight[user_id, 0]
            i_bias = self.item_bias_embedding.weight[item_id, 0]

            pred = self._global_bias.item()
            pred += u_bias.item() + i_bias.item()
            pred += torch.dot(user_factor, item_factor).item()

        return float(pred)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute predictions.

        Args:
            user_ids: Tensor of user indices (batch_size,)
            item_ids: Tensor of item indices (batch_size,)

        Returns:
            Predicted ratings (batch_size,)
        """
        # Get embeddings
        user_factors = self.user_embedding(user_ids)  # (batch_size, num_factors)
        item_factors = self.item_embedding(item_ids)  # (batch_size, num_factors)
        user_biases = self.user_bias_embedding(user_ids).squeeze()  # (batch_size,)
        item_biases = self.item_bias_embedding(item_ids).squeeze()  # (batch_size,)

        # Compute predictions
        preds = self._global_bias + user_biases + item_biases
        preds += (user_factors * item_factors).sum(dim=1)

        return preds

    def fit(
        self,
        user_ids: list[int],
        item_ids: list[int],
        ratings: list[float],
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = False,
    ):
        """Train model using SGD with mini-batch support and autograd.

        Args:
            user_ids: List of user indices
            item_ids: List of item indices
            ratings: List of ratings
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Whether to print training progress
        """
        # Set global bias
        self._global_bias.data = torch.tensor([np.mean(ratings)], dtype=torch.float32)

        # Convert to tensors
        users_t = torch.tensor(user_ids, dtype=torch.long)
        items_t = torch.tensor(item_ids, dtype=torch.long)
        ratings_t = torch.tensor(ratings, dtype=torch.float32)

        n_samples = len(user_ids)

        # Create optimizer with L2 regularization (weight decay)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.reg)

        for epoch in range(epochs):
            total_loss = 0.0

            # Process in batches
            for batch_indices in batched(range(n_samples), batch_size):
                # print("batch", batch_indices)
                batch_indices = list(batch_indices)  # Convert to list for indexing

                # Get batch data
                batch_users = users_t[batch_indices]
                batch_items = items_t[batch_indices]
                batch_ratings = ratings_t[batch_indices]

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                preds = self.forward(batch_users, batch_items)

                # Compute MSE loss
                loss = ((batch_ratings - preds) ** 2).mean()

                # Backward pass (compute gradients)
                loss.backward()

                # Update parameters
                optimizer.step()

                total_loss += loss.item() * len(batch_indices)

            if verbose:
                rmse = np.sqrt(total_loss / n_samples)
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
