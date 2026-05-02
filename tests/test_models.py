"""Tests for models module."""

import numpy as np
import scipy.sparse as sp

from recsys_genai.models import EASE


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
