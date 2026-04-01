"""
@author AFelixLiu
@date 2026 3月 05
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Legacy Layer Normalization module.

    Keep this class to ensure compatibility with pretrained weights.
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """Normalize using std and eps outside the radical."""

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class LayerNormStandard(nn.Module):
    """
    Standard Layer Normalization module.

    Using this will cause slight numerical drift.
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """Normalize using sqrt(var + eps) following standard conventions."""

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)

        return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2
