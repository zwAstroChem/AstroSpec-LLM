"""
@author AFelixLiu
@date 2026 3月 05
"""

import torch.nn as nn


class BERTLM(nn.Module):
    """BERT Language Model for pre-training, including the Masked LM head."""

    def __init__(self, bert, vocab_size):
        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x):
        """Pass input through BERT and then the MLM head."""

        x = self.bert(x)
        return self.mask_lm(x)


class MaskedLanguageModel(nn.Module):
    """Predicts original tokens from masked input sequences (N-class classification)."""

    def __init__(self, hidden, vocab_size):
        super().__init__()
        # Linear layer to project hidden states to vocabulary size
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """Apply linear transformation and log-softmax for classification."""

        return self.softmax(self.linear(x))
