"""Self-Attentive Sequential Recommendation (SASRec).

Implementation based on Kang & McAuley (2018) ICDM paper.
"""

from typing import Optional

import torch
import torch.nn as nn


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model.

    Uses self-attention to model sequential user behavior for next-item prediction.
    Based on "Self-Attentive Sequential Recommendation" (Kang & McAuley, 2018).

    Args:
        num_items: Number of items in catalog (+ padding idx 0)
        max_len: Maximum sequence length
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        hidden_size: Dimensionality of embeddings
        dropout: Dropout probability

    Example:
        >>> model = SASRec(num_items=100, max_len=50, hidden_size=64)
        >>> seq = torch.randint(0, 100, (2, 50))  # batch_size=2, seq_len=50
        >>> logits = model(seq)
        >>> logits.shape
        torch.Size([2, 50, 101])
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        num_blocks: int = 2,
        num_heads: int = 2,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len

        # Item embeddings (0 is padding)
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, dropout) for _ in range(num_blocks)]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            seq: (batch_size, seq_len) Item sequence tensor

        Returns:
            (batch_size, seq_len, num_items) Logits for next item prediction
        """
        batch_size, seq_len = seq.shape

        # Positions
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.layer_norm(x)

        # Output logits (reuse item embeddings)
        logits = torch.matmul(x, self.item_emb.weight.T)

        return logits

    def predict_next(self, seq: torch.Tensor, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict top-K next items for a sequence.

        Args:
            seq: (batch_size, seq_len) Item sequence
            k: Number of items to return

        Returns:
            top_items: (batch_size, k) Top-K item indices
            top_scores: (batch_size, k) Corresponding scores
        """
        self.eval()
        with torch.no_grad():
            logits = self(seq)
            # Get logits for last position
            last_logits = logits[:, -1, :]
            top_scores, top_items = torch.topk(last_logits, k, dim=-1)
        return top_items, top_scores

    def forward_with_attention(self, seq: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass that captures attention weights from all transformer blocks.

        Args:
            seq: (batch_size, seq_len) Item sequence tensor

        Returns:
            x: (batch_size, seq_len, hidden_size) Final hidden states
            attention_weights: List of attention weight tensors, one per block.
                Each tensor has shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = seq.shape

        # Positions
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()

        # Collect attention weights from all blocks
        attention_weights = []

        for block in self.blocks:
            # Get attention from multi-head attention (per-head weights)
            attn_output, attn_weights = block.attention(
                x, x, x, attn_mask=mask, need_weights=True, average_attn_weights=False
            )

            # Save weights from this block
            attention_weights.append(attn_weights)

            # Continue block forward pass
            x = block.ln1(x + block.dropout(attn_output))
            ffn_out = block.ffn(x)
            x = block.ln2(x + ffn_out)

        return x, attention_weights


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head self-attention and FFN."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.2):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)

        return x


def train_sasrec(
    model: SASRec, dataloader, num_epochs: int = 5, lr: float = 0.001, device: str = "cpu"
) -> list[float]:
    """Train SASRec model.

    Args:
        model: SASRec model instance
        dataloader: PyTorch DataLoader with (seq, target) pairs
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        List of average losses per epoch

    Example:
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> seqs = torch.randint(1, 100, (10, 50))
        >>> targets = torch.randint(1, 100, (10,))
        >>> loader = DataLoader(TensorDataset(seqs, targets), batch_size=2)
        >>> model = SASRec(num_items=100, max_len=50, hidden_size=32)
        >>> losses = train_sasrec(model, loader, num_epochs=1)  # doctest: +ELLIPSIS
        Epoch 1/1, Loss: ...
        >>> len(losses) == 1
        True
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for seq, target in dataloader:
            seq = seq.to(device)
            target = target.to(device)

            # Forward
            logits = model(seq)

            # Loss on last position
            loss = criterion(logits[:, -1, :], target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return losses
