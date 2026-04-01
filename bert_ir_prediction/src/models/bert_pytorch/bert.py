import torch.nn as nn

from .embedding import BERTEmbedding
from .utils.encoder_layer import EncoderLayer
from .utils import clone_module


class BERT(nn.Module):
    """
    BERT model: Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, use_rope=True):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        :param use_rope: use RoPE
        """

        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.use_rope = use_rope

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of token and positional embeddings (PE is optional.)
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout, use_rope=use_rope)

        self.layers = clone_module(
            EncoderLayer(self.hidden, self.attn_heads, self.feed_forward_hidden, self.dropout, self.use_rope),
            n_layers)

    def forward(self, x):
        """Pass the input (and mask) through each layer in turn."""

        # attention masking for padded token
        mask = (x > 0).unsqueeze(1).to(x.device)  # [batch_size, 1, seq_len]

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x
