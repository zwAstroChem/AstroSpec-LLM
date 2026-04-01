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


def normalize(y_hat, y, eps=1e-8):
    """
    Normalizes predictions and targets to ensure proper types, dimensions, and sum-to-one.

    Args:
        y_hat (Tensor/Array): Predicted values.
        y (Tensor/Array): Target values.
        eps (float): Small constant to avoid division by zero.

    Returns:
        Tuple[Tensor, Tensor]: Normalized (y_hat, y).
    """

    # Cast to float tensors if necessary
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat, dtype=torch.float)
    elif y_hat.dtype != torch.float:
        y_hat = y_hat.type(torch.float)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float)
    elif y.dtype != torch.float:
        y = y.type(torch.float)

    # Ensure batch dimension (rank-2)
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # Sum-based normalization along the last dimension
    normed_y_hat = y_hat / (y_hat.sum(dim=-1, keepdim=True) + eps)
    normed_y = y / (y.sum(dim=-1, keepdim=True) + eps)

    return normed_y_hat, normed_y
