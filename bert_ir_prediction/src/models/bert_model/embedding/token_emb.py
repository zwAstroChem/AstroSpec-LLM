"""
@author AFelixLiu
@date 2026 3月 05
"""

import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """Standard Token Embedding layer with a fixed padding index."""

    def __init__(self, vocab_size, embed_size=768):
        super().__init__(vocab_size, embed_size, padding_idx=0)
