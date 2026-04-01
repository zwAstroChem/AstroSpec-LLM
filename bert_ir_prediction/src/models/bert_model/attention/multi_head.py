"""
@author AFelixLiu
@date 2026 3月 05
"""

import math

import torch
import torch.nn as nn

from ..utils import clone_module


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'.

    Args:
        query (Tensor): Query tensor of shape [batch, heads, seq_len, head_dim].
        key (Tensor): Key tensor of shape [batch, heads, seq_len, head_dim].
        value (Tensor): Value tensor of shape [batch, heads, seq_len, head_dim].
        mask (Tensor, optional): Mask tensor to filter out specific positions.
        dropout (nn.Dropout, optional): Dropout layer to apply on attention weights.

    Returns:
        Tuple[Tensor, Tensor]: Output tensor and attention weight matrix.
    """

    head_dim = query.size(-1)
    # Scaled dot-product between Query and Key
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention with Rotary Positional Embeddings (RoPE) support."""

    def __init__(self, n_heads, d_model, dropout=0.1, use_rope=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.use_rope = use_rope

        # Linears for Q, K, V and Output projection
        self.linears = clone_module(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Broadcast mask across all heads
            mask = mask.unsqueeze(1)  # [batch, 1, 1, seq_len]

        batch_size = query.size(0)
        seq_len = query.size(1)

        # 1) Linear projections and reshape to [batch, heads, seq_len, head_dim]
        query, key, value = [
            l(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            for l, x in zip(self.linears[:3], (query, key, value))
        ]

        # 2) Apply RoPE if enabled
        if self.use_rope:
            query, key = self.apply_rope(query, key)

        # 3) Scaled Dot-Product Attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 4) Concatenate heads and project output
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.linears[-1](x)

    def apply_rope(self, q, k):
        """Apply Rotary Positional Embeddings via complex number multiplication."""

        seq_len = q.size(2)
        # Precompute rotation factors
        pos_emb = self.get_rotary_embedding(seq_len, self.head_dim, q.device)

        # Reshape to complex-ready format: [..., head_dim//2, 2]
        q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

        # Rotate: (a+bi)(cos+isin)
        q_rotated = q_complex * pos_emb
        k_rotated = k_complex * pos_emb

        # Back to real: [batch, heads, seq_len, head_dim]
        q_out = torch.view_as_real(q_rotated).flatten(3)
        k_out = torch.view_as_real(k_rotated).flatten(3)

        return q_out.type_as(q), k_out.type_as(k)

    @staticmethod
    def get_rotary_embedding(seq_len, head_dim, device):
        """Generates complex rotation factors e^(i*theta)."""

        # Theta calculation for half of the head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
        t = torch.arange(seq_len, device=device).float()

        # Outer product to get angles for each position
        freqs = torch.outer(t, inv_freq)
        # Use polar coordinates to create complex rotation unit vectors
        return torch.polar(torch.ones_like(freqs), freqs)
