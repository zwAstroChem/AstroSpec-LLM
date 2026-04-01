import math

import torch
import torch.nn as nn

from ..utils import clone_module


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'.
    Args:
        query: [batch, heads, seq_len, head_dim]
        key: [batch, heads, seq_len, head_dim]
        value: [batch, heads, seq_len, head_dim]
        mask: ...
        dropout: ...
    """

    head_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """The implementation of multi-head attention mechanism."""

    def __init__(self, n_heads, d_model, dropout=0.1, use_rope=True):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        # 4个用于可学习线性转换权重矩阵 W^Q, W^K, W^V, W^O
        self.linears = clone_module(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.use_rope = use_rope

    def forward(self, query, key, value, mask=None):
        # query, key, value: [batch, seq_len, d_model]

        if mask is not None:
            # Same mask applied to all heads.
            mask = mask.unsqueeze(2)  # [batch_size, 1, 1, seq_len]

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => n_heads x head_dim.
        query, key, value = [
            lin(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]  # query, key, value: [batch, n_heads, seq_len, head_dim]

        # 2) Apply RoPE.
        if self.use_rope:
            query, key = self.apply_rope(query, key)

        # 3) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 4) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.head_dim)
        )  # x: [batch, seq_len, d_model]

        # 删除不再需要的中间变量，以节省内存
        del query
        del key
        del value

        return self.linears[-1](x)

    def apply_rope(self, q, k):
        """Apply RoPE.
        Args:
            q: [batch, n_heads, seq_len, head_dim]
            k: [batch, n_heads, seq_len, head_dim]
        Returns:
            Rotated query and key.
        """

        pos_emb = self.rotary_position_embedding(q.size(-2), q.size(-1), q.device)  # [seq_len, head_dim//2]

        q_ = q.float().reshape(*q.shape[:-1], -1, 2)  # [batch, n_heads, seq_len, head_dim//2, 2]
        k_ = k.float().reshape(*k.shape[:-1], -1, 2)  # [batch, n_heads, seq_len, head_dim//2, 2]

        # 转为复数域
        q_ = torch.view_as_complex(q_)  # [batch, n_heads, seq_len, head_dim//2]
        k_ = torch.view_as_complex(k_)  # [batch, n_heads, seq_len, head_dim//2]

        # 复数乘法实现旋转操作
        # pos_emb: [seq_len, head_dim//2] -> 广播为 [batch, n_heads, seq_len, head_dim//2]
        q_rot = q_ * pos_emb
        k_rot = k_ * pos_emb

        # 转换回实数域并展平
        q_rot = torch.view_as_real(q_rot).flatten(3)  # [batch, n_heads, seq_len, head_dim]
        k_rot = torch.view_as_real(k_rot).flatten(3)  # [batch, n_heads, seq_len, head_dim]

        return q_rot.type_as(q), k_rot.type_as(k)

    @staticmethod
    def rotary_position_embedding(seq_len, head_dim, device):
        # 计算旋转频率
        div_term = torch.exp(
            torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) *
            -(math.log(10000.0) / (head_dim // 2))
        )

        # 生成位置索引
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 计算旋转角度
        embeddings = torch.outer(position, div_term).float()  # [seq_len, head_dim//2]
        # 使用极坐标方式构造复数旋转因子
        pos_emb = torch.polar(torch.ones_like(embeddings), embeddings)  # [seq_len, head_dim//2]

        return pos_emb
