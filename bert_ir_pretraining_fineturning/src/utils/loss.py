"""
@author AFelixLiu
@date 2026 3月 05
"""

import torch
import torch.nn as nn


class MLMAcc(nn.Module):
    """Accuracy for Masked Language Modeling, ignoring specified indices."""

    def __init__(self, ignore_index: int = 0):
        super(MLMAcc, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        Args:
            logits: Predicted probabilities from the MLM head.
                    Shape: [Batch_Size, Seq_Len, Vocab_Size]
            labels: Ground truth token IDs.
                    Shape: [Batch_Size, Seq_Len]
        """

        device = logits.device
        preds = torch.argmax(logits, dim=-1)
        mask = (labels > self.ignore_index).to(device)
        correct = torch.sum((preds == labels) * mask).float()

        return correct / (torch.sum(mask).float() + 1e-8)


class EMDLoss(nn.Module):
    """
    Earth Mover's Distance (EMD) Loss for spectral profile matching.
    Calculates the integral of the absolute difference between cumulative distributions.
    """

    def __init__(self):
        super(EMDLoss, self).__init__()

    @staticmethod
    def forward(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted spectrum [Batch, Out_Dim]
            label: Ground truth spectrum [Batch, Out_Dim]
        """

        # Calculate cumulative sum of differences across the spectral dimension
        diff = pred - label
        cumsum_diff = torch.cumsum(diff, dim=-1)

        # Sum of absolute cumulative differences for the entire batch
        emd = torch.sum(torch.abs(cumsum_diff))
        return emd


class SISLoss(nn.Module):
    """
    Spectral Information Similarity (SIS) Loss based on Symmetric Itakura-Saito Divergence.
    Measures the statistical distance between predicted and target spectral distributions.
    """

    def __init__(self, epsilon=1e-8):
        super(SISLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted spectrum [Batch, Out_Dim]
            label: Ground truth spectrum [Batch, Out_Dim]
        """

        # Numerical stability: clamp values to avoid log(0)
        pred = torch.clamp(pred, min=self.epsilon)
        label = torch.clamp(label, min=self.epsilon)

        # Calculate Symmetric Itakura-Saito Divergence (SID)
        sid = torch.sum(pred * torch.log(pred / label) + label * torch.log(label / pred), dim=-1)

        # Map SID to similarity score and sum across the batch
        sis = torch.sum(1 / (1 + sid))
        return sis
