"""
@author AFelixLiu
@date 2026 3月 05
"""

import math

import torch
import torch.nn as nn


class SinPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding as described in 'Attention Is All You Need'."""

    def __init__(self, max_len=512, embed_size=768):
        super().__init__()

        # Precompute positional encodings in log space for numerical stability
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term calculation: 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer to ensure it's saved with state_dict but not trained
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """Return precomputed encodings cropped to input sequence length."""

        return self.pe[:, :x.size(1)]


class TrainablePositionalEmbedding(nn.Embedding):
    """Learned positional embedding layer."""

    def __init__(self, max_len=512, embed_size=768):
        # Initializing trainable embeddings with padding_idx=0
        super().__init__(max_len, embed_size, padding_idx=0)
