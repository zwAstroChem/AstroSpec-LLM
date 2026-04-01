"""
@author AFelixLiu
@date 2026 3月 05
"""

import torch.nn as nn

from .embedding import BERTEmbedding
from .layers import EncoderLayer
from .utils import clone_module


class BERT(nn.Module):
    """BERT model implementation: stack of Transformer Encoder layers."""

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, use_rope=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # Feed-forward hidden size is typically 4 times the model hidden size
        self.feed_forward_hidden = hidden * 4
        self.use_rope = use_rope

        # Initialize core embedding and transformer layers
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            dropout=dropout,
            use_rope=use_rope
        )

        self.layers = clone_module(
            EncoderLayer(hidden, attn_heads, self.feed_forward_hidden, dropout, use_rope),
            n_layers
        )

    def forward(self, x):
        """Forward pass through embeddings and sequential encoder layers."""

        # Resulting shape: [batch_size, 1, seq_len]
        mask = (x > 0).unsqueeze(1)

        # Initial embedding pass
        x = self.embedding(x)

        # Sequential pass through stacked encoder blocks
        for layer in self.layers:
            x = layer(x, mask)

        return x
