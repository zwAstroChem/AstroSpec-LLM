"""
@author AFelixLiu
@date 2026 3月 05
"""

import torch.nn as nn

from .feed_forward import PositionwiseFeedForward
from .sublayer import Sublayer
from ..attention import MultiHeadedAttention
from ..utils import clone_module


class EncoderLayer(nn.Module):
    """Transformer Encoder layer: Multi-Headed Attention followed by Feed-Forward."""

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, use_rope):
        super().__init__()
        # Initializing core sub-components
        self.self_attn = MultiHeadedAttention(
            n_heads=attn_heads, d_model=hidden, dropout=dropout, use_rope=use_rope
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )

        # Using clone_module to create distinct layers for residual connections
        self.sublayer = clone_module(Sublayer(size=hidden, dropout=dropout), 2)

    def forward(self, x, mask):
        """Apply self-attention and feed-forward operations with residual connections."""

        # First sublayer: Multi-Headed Self-Attention
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))

        # Second sublayer: Position-wise Feed-Forward Network
        return self.sublayer[1](x, self.feed_forward)
