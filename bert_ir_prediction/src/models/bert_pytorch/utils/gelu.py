import math

import torch
import torch.nn as nn


class GELU(nn.Module):
    """BERT used the GELU instead of RELU."""

    @staticmethod
    def forward(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
