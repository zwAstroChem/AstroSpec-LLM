import torch.nn as nn

from .position import SinPositionalEmbedding
from .token import TokenEmbedding


class BERTEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout=0.1, use_rope=True):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        :param use_rope: use RoPE
        """
        super().__init__()
        self.use_rope = use_rope

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = SinPositionalEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        if self.use_rope:
            x = self.token(sequence)
        else:
            x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)
