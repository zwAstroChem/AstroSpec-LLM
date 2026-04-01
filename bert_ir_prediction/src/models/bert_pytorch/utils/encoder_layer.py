import torch.nn as nn

from ..attention import MultiHeadedAttention
from ..utils import PositionwiseFeedForward, Sublayer
from ..utils import clone_module


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward."""

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, use_rope):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        :param use_rope: whether to use RoPE
        """

        super().__init__()
        self.self_attn = MultiHeadedAttention(n_heads=attn_heads, d_model=hidden, dropout=dropout, use_rope=use_rope)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.sublayer = clone_module(Sublayer(size=hidden, dropout=dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        return self.sublayer[1](x, self.feed_forward)
