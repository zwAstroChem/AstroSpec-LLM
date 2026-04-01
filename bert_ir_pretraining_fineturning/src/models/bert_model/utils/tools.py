"""
@author AFelixLiu
@date 2026 3月 05
"""

import copy
import math

import torch
import torch.nn as nn


def clone_module(module, n):
    """
    Creates a ModuleList containing n deep copies of the input module.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function."""

    @staticmethod
    def forward(x):
        # Implementation using the tanh approximation formula
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
