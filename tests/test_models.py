"""Tests for models module."""

import numpy as np
import pytest
import scipy.sparse as sp

from recsys_genai.models import EASE, MatrixFactorization


def test_matrix_factorization_init():
    """Test MF model initialization."""
    model = MatrixFactorization(num_users=100, num_items=50, num_factors=10)

    assert model.user_factors.shape == (100, 10)
    assert model.item_factors.shape == (50, 10)
    assert len(model.user_bias) == 100
    assert len(model.item_bias) == 50


def test_matrix_factorization_predict():
    """Test MF prediction."""
    model = MatrixFactorization(num_users=10, num_items=5, num_factors=3)

    # Set known factors for testing
    model.user_factors[0] = [1.0, 0.0, 0.0]
    model.item_factors[0] = [1.0, 0.0, 0.0]
    model.user_bias[0] = 0.5
    model.item_bias[0] = 0.3
    model.global_bias = 3.5

    pred = model.predict(0, 0)
    # Expected: 3.5 + 0.5 + 0.3 + 1.0*1.0 = 5.3
    assert pred == pytest.approx(5.3)


def test_matrix_factorization_fit():
    """Test MF training."""
    model = MatrixFactorization(
        num_users=5, num_items=5, num_factors=2, learning_rate=0.1, reg=0.01
    )

    # Simple training data
    user_ids = [0, 1, 2]
    item_ids = [0, 1, 2]
    ratings = [5.0, 4.0, 3.0]

    # Train for a few epochs
    model.fit(user_ids, item_ids, ratings, epochs=5, verbose=False)

    # Check that predictions are reasonable
    pred_0 = model.predict(0, 0)
    assert 2.0 < pred_0 < 6.0  # Should be roughly near 5.0


def test_matrix_factorization_convergence():
    """Test that MF loss decreases during training."""
    model = MatrixFactorization(
        num_users=10, num_items=10, num_factors=5, learning_rate=0.01, reg=0.001
    )

    # Generate synthetic data
    np.random.seed(42)
    num_samples = 50
    user_ids = np.random.randint(0, 10, num_samples).tolist()
    item_ids = np.random.randint(0, 10, num_samples).tolist()
    ratings = np.random.uniform(1, 5, num_samples).tolist()

    # Calculate initial error
    initial_errors = []
    for u, i, r in zip(user_ids, item_ids, ratings):
        pred = model.predict(u, i)
        initial_errors.append((r - pred) ** 2)
    initial_mse = np.mean(initial_errors)

    # Train
    model.fit(user_ids, item_ids, ratings, epochs=20, verbose=False)

    # Calculate final error
    final_errors = []
    for u, i, r in zip(user_ids, item_ids, ratings):
        pred = model.predict(u, i)
        final_errors.append((r - pred) ** 2)
    final_mse = np.mean(final_errors)

    # Error should decrease
    assert final_mse < initial_mse


def test_ease_init():
    """Test EASE model initialization."""
    model = EASE(reg=100.0)

    assert model.reg == 100.0
    assert model.B is None


def test_ease_fit():
    """Test EASE model training."""
    # Create simple interaction matrix
    X = sp.csr_matrix([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1]])

    model = EASE(reg=100.0)
    model.fit(X)

    assert model.B is not None
    assert model.B.shape == (4, 4)

    # Diagonal should be zero
    assert np.allclose(np.diag(model.B), 0.0)


def test_ease_recommend():
    """Test EASE recommendation generation."""
    # Create simple interaction matrix
    X = sp.csr_matrix([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1]])

    model = EASE(reg=100.0)
    model.fit(X)

    # Get recommendations for user 0
    recs = model.recommend_for_user(0, X, k=2)

    assert len(recs) == 2
    assert all(isinstance(r, (int, np.integer)) for r in recs)


def test_ease_recommend_excludes_interacted():
    """Test that EASE doesn't recommend already interacted items."""
    # User 0 has interacted with items 0 and 2
    X = sp.csr_matrix([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0]])

    model = EASE(reg=100.0)
    model.fit(X)

    recs = model.recommend_for_user(0, X, k=3)

    # Recommendations should not include items 0 and 2
    assert 0 not in recs
    assert 2 not in recs


def test_ease_dense_input():
    """Test EASE with dense array input."""
    X_dense = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1]])

    model = EASE(reg=100.0)
    model.fit(X_dense)

    assert model.B is not None

    recs = model.recommend_for_user(0, X_dense, k=2)
    assert len(recs) == 2


def test_ease_regularization_effect():
    """Test that different regularization affects results."""
    X = sp.csr_matrix([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1]])

    model_low_reg = EASE(reg=1.0)
    model_low_reg.fit(X)

    model_high_reg = EASE(reg=1000.0)
    model_high_reg.fit(X)

    # Different regularization should give different B matrices
    assert not np.allclose(model_low_reg.B, model_high_reg.B)
