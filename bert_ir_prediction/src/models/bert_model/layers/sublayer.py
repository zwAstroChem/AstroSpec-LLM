"""
@author AFelixLiu
@date 2026 3月 05
"""

import torch.nn as nn

from .layer_norm import LayerNorm


class Sublayer(nn.Module):
    """Implements a residual connection followed by a LayerNorm (Post-Norm)."""

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection and normalization to the sublayer output."""

        # Residual connection with dropout on sublayer output
        return self.norm(x + self.dropout(sublayer(x)))
