"""
@author AFelixLiu
@date 2026 3月 05
"""

import torch
import torch.nn.functional as F
from torch import nn

from .bert_model import BERT
from ..utils import EMDLoss, SISLoss, normalize


class BERT4IR(nn.Module):
    """
    Downstream BERT model specifically designed for Infrared (IR) spectra prediction.
    It supports neutral and charged molecules with multiple encoding strategies.
    """

    def __init__(self, bert, ir_bins, normed=False, support_charge=False, charge_vocab=None,
                 charge_encoding="emb", charge_dim=16, onehot_repeat=1, plot=False):
        """
        Args:
            bert (BERT): The pre-trained backbone BERT model.
            ir_bins (int): Dimension of the output IR spectrum (number of frequency bins).
            normed (bool): If True, assumes input spectra are already normalized.
            support_charge (bool): Whether to incorporate ionic charge information.
            charge_vocab (list, optional): List of supported charge values (e.g., [-1, 0, 1, 2]).
            charge_encoding (str): Method to encode charge: 'emb' (Embedding) or 'onehot'.
            charge_dim (int): Size of the charge embedding vector (used if encoding='emb').
            onehot_repeat (int): Number of times to tile the one-hot vector (used if encoding='onehot').
            plot (bool): If True, returns both predictions and individual loss metrics.
        """

        super().__init__()
        self.bert = bert
        self.normed = normed
        self.support_charge = support_charge
        self.plot = plot

        # Head selection based on charge encoding configuration
        if not support_charge:
            self.head = IRHead(self.bert.hidden, ir_bins)
        elif charge_encoding == "emb":
            self.head = IRHeadWithChargeEmb(
                hidden=self.bert.hidden,
                ir_bins=ir_bins,
                charge_vocab=charge_vocab,
                charge_dim=charge_dim
            )
        elif charge_encoding == "onehot":
            self.head = IRHeadWithChargeOneHot(
                hidden=self.bert.hidden,
                ir_bins=ir_bins,
                charge_vocab=charge_vocab,
                onehot_repeat=onehot_repeat
            )
        else:
            raise ValueError(f"Unsupported charge encoding: {charge_encoding}")

        # Regression loss metrics for spectral distribution
        self.criterion_emd = EMDLoss()
        self.criterion_sis = SISLoss()
        self.criterion_mse = nn.MSELoss(reduction="sum")

    def forward(self, x, y, charges=None):
        """Forward pass through BERT and prediction head, followed by multi-metric loss calculation."""

        bert_output = self.bert(x)

        # Select y_hat calculation path based on charge support
        if self.support_charge:
            y_hat = self.head(bert_output, charges)
        else:
            y_hat = self.head(bert_output)

        # Spectral normalization and loss evaluation
        y_hat_norm, y_norm = (y_hat, y) if self.normed else normalize(y_hat, y)

        emd_loss = self.criterion_emd(y_hat_norm, y_norm)
        sis_loss = self.criterion_sis(y_hat_norm, y_norm)
        mse_loss = self.criterion_mse(y_hat_norm, y_norm)

        if self.plot:
            return y_hat_norm, y_norm, emd_loss, sis_loss, mse_loss
        return emd_loss, sis_loss, mse_loss


class IRHead(nn.Module):
    """Basic MLP head for neutral molecules using the [CLS] token."""

    def __init__(self, hidden, ir_bins):
        super().__init__()
        half = hidden // 2
        self.linear1 = nn.Linear(hidden, half)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(half, ir_bins)

    def forward(self, x):
        # Using [CLS] vector for global molecular representation
        cls_vec = x[:, 0]
        out = self.relu1(self.linear1(cls_vec))

        return torch.abs(self.linear2(out))


class IRHeadWithChargeEmb(nn.Module):
    """IR prediction head using trainable charge embeddings."""

    def __init__(self, hidden, ir_bins, charge_vocab=None, charge_dim=4):
        super().__init__()
        self.charge_vocab = charge_vocab or [-2, -1, 0, 1]
        self.charge_embedding = nn.Embedding(len(self.charge_vocab), charge_dim)

        total_dim = hidden + charge_dim
        half = total_dim // 2
        self.linear1 = nn.Linear(total_dim, half)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(half, ir_bins)

    def forward(self, x, charges):
        device = x.device
        # Vectorized charge index mapping
        c_indices = torch.tensor([self.charge_vocab.index(c) for c in charges], device=device)

        cls_vec = x[:, 0]
        c_emb = self.charge_embedding(c_indices)
        combined = torch.cat([cls_vec, c_emb], dim=-1)

        out = self.relu1(self.linear1(combined))
        return torch.abs(self.linear2(out))


class IRHeadWithChargeOneHot(nn.Module):
    """IR prediction head using repeated one-hot encoded charges."""

    def __init__(self, hidden, ir_bins, charge_vocab=None, onehot_repeat=1):
        super().__init__()
        self.charge_vocab = charge_vocab or [-2, -1, 0, 1]
        self.onehot_repeat = onehot_repeat
        self.charge_dim = len(self.charge_vocab)
        self.onehot_dim = self.charge_dim * onehot_repeat

        total_dim = hidden + self.onehot_dim
        half = total_dim // 2
        self.linear1 = nn.Linear(total_dim, half)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(half, ir_bins)

    def forward(self, x, charges):
        device = x.device
        c_indices = torch.tensor([self.charge_vocab.index(c) for c in charges], device=device)

        # Generate and repeat one-hot encoding
        c_onehot = F.one_hot(c_indices, num_classes=self.charge_dim).float()
        c_expanded = c_onehot.repeat(1, self.onehot_repeat)

        combined = torch.cat([x[:, 0], c_expanded], dim=-1)
        out = self.relu1(self.linear1(combined))
        return torch.abs(self.linear2(out))
