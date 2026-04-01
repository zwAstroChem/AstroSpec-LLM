"""
@author AFelixLiu
@date 2026 3月 05
"""

import torch.nn as nn

from ..utils import GELU


class PositionwiseFeedForward(nn.Module):
    """Implements the two-layer linear transformation with a GELU activation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # First linear layer expands dimensionality to d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        # Second linear layer projects back to d_model
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """Apply position-wise feed-forward transformations."""

        return self.w_2(self.dropout(self.activation(self.w_1(x))))
