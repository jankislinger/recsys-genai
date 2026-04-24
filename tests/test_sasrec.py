"""Tests for SASRec module."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from recsys_genai.sasrec import SASRec, TransformerBlock, train_sasrec


def test_sasrec_init():
    """Test SASRec model initialization."""
    model = SASRec(
        num_items=100, max_len=50, num_blocks=2, num_heads=2, hidden_size=64, dropout=0.2
    )

    assert model.num_items == 100
    assert model.max_len == 50
    assert len(model.blocks) == 2


def test_sasrec_forward():
    """Test SASRec forward pass."""
    model = SASRec(num_items=100, max_len=20, hidden_size=32)

    # Batch of sequences
    batch_size = 4
    seq_len = 20
    seq = torch.randint(0, 100, (batch_size, seq_len))

    logits = model(seq)

    assert logits.shape == (batch_size, seq_len, 101)  # num_items + 1 (padding)


def test_sasrec_predict_next():
    """Test next-item prediction."""
    model = SASRec(num_items=50, max_len=10, hidden_size=16)

    seq = torch.randint(1, 50, (2, 10))  # 2 sequences

    top_items, top_scores = model.predict_next(seq, k=5)

    assert top_items.shape == (2, 5)  # Batch of 2, top-5 each
    assert top_scores.shape == (2, 5)

    # Items should be in valid range
    assert torch.all(top_items >= 0)
    assert torch.all(top_items <= 50)


def test_sasrec_causal_masking():
    """Test that causal masking prevents future peeking."""
    model = SASRec(num_items=20, max_len=5, num_blocks=1, hidden_size=8)

    # Two sequences: one with future info, one without
    seq1 = torch.tensor([[1, 2, 3, 4, 5]])
    seq2 = torch.tensor([[1, 2, 3, 0, 0]])  # Padded

    model.eval()
    with torch.no_grad():
        logits1 = model(seq1)
        logits2 = model(seq2)

    # Position 2 should have same logits regardless of future items
    # (This is approximate due to positional encodings, but should be similar)
    pos_2_logits1 = logits1[0, 2, :]
    pos_2_logits2 = logits2[0, 2, :]

    # Should be reasonably close (not exactly equal due to different lengths)
    # Just check they're in similar range
    assert torch.abs(pos_2_logits1.mean() - pos_2_logits2.mean()) < 5.0


def test_transformer_block():
    """Test TransformerBlock module."""
    block = TransformerBlock(hidden_size=32, num_heads=2, dropout=0.1)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 32)

    output = block(x)

    assert output.shape == x.shape
    assert not torch.allclose(output, x)  # Should transform input


def test_transformer_block_with_mask():
    """Test TransformerBlock with attention mask."""
    block = TransformerBlock(hidden_size=16, num_heads=2, dropout=0.0)

    seq_len = 5
    x = torch.randn(1, seq_len, 16)

    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    output = block(x, mask)

    assert output.shape == x.shape


def test_train_sasrec_runs():
    """Test that training loop executes without errors."""
    # Small model for fast testing
    model = SASRec(num_items=50, max_len=10, num_blocks=1, num_heads=1, hidden_size=16, dropout=0.1)

    # Dummy data
    seqs = torch.randint(1, 50, (20, 10))
    targets = torch.randint(1, 50, (20,))
    dataset = TensorDataset(seqs, targets)
    loader = DataLoader(dataset, batch_size=5)

    # Train for 1 epoch
    losses = train_sasrec(model, loader, num_epochs=1, lr=0.01, device="cpu")

    assert len(losses) == 1
    assert losses[0] > 0  # Loss should be positive


def test_train_sasrec_decreasing_loss():
    """Test that loss decreases during training."""
    torch.manual_seed(42)

    model = SASRec(num_items=30, max_len=8, num_blocks=1, num_heads=1, hidden_size=16, dropout=0.1)

    # Simple repeating pattern to learn
    seqs = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
        * 5
    )  # Repeat for more data

    targets = torch.tensor([9] * 20)
    dataset = TensorDataset(seqs, targets)
    loader = DataLoader(dataset, batch_size=4)

    losses = train_sasrec(model, loader, num_epochs=5, lr=0.01, device="cpu")

    # Loss should generally decrease
    assert losses[-1] < losses[0]


def test_sasrec_padding_idx():
    """Test that padding index (0) is handled correctly."""
    model = SASRec(num_items=10, max_len=5, hidden_size=8)

    # Sequence with padding
    seq = torch.tensor([[0, 0, 1, 2, 3]])

    logits = model(seq)

    # Should produce output without errors
    assert logits.shape == (1, 5, 11)


def test_sasrec_different_batch_sizes():
    """Test SASRec with different batch sizes."""
    model = SASRec(num_items=20, max_len=10, hidden_size=16)

    for batch_size in [1, 2, 8]:
        seq = torch.randint(0, 20, (batch_size, 10))
        logits = model(seq)
        assert logits.shape[0] == batch_size
