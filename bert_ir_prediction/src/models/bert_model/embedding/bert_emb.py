"""
@author AFelixLiu
@date 2026 3月 05
"""

import torch.nn as nn

from .pos_emb import SinPositionalEmbedding
from .token_emb import TokenEmbedding


class BERTEmbedding(nn.Module):
    """Combines Token and Positional Embeddings, with optional support for RoPE."""

    def __init__(self, vocab_size, embed_size, dropout=0.1, use_rope=True):
        super().__init__()
        self.use_rope = use_rope

        # Core embedding layers
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = SinPositionalEmbedding(embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        """Calculates embeddings. If RoPE is enabled, absolute positions are skipped."""

        x = self.token(sequence)

        # RoPE handles positions within the Attention layer, so skip absolute addition here
        if not self.use_rope:
            x = x + self.position(sequence)

        return self.dropout(x)
