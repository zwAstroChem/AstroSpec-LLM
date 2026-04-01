import math

import torch
import torch.nn as nn


class SinPositionalEmbedding(nn.Module):

    def __init__(self, max_len=512, embed_size=768):
        super(SinPositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].requires_grad_(False)


class TrainablePositionalEmbedding(nn.Embedding):

    def __init__(self, max_len=512, embed_size=768):
        super().__init__(max_len, embed_size, padding_idx=0)
