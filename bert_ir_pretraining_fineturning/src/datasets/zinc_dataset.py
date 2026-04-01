"""
@author AFelixLiu
@date 2026 3月 05
"""

import random

import torch
from torch.utils.data import Dataset

from ..utils import morgan_tokenize, smiles_tokenize


class ZINCDataset(Dataset):
    """Dataset for ZINC molecules implementing Masked Language Modeling (MLM)."""

    def __init__(self,data, scheme, vocab, seq_len):
        self.data = data["canonical_smiles"]
        self.scheme = scheme
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.seq_len = seq_len

        # Select tokenization function once during initialization
        if self.scheme == "morgan":
            self.tokenize_fn = morgan_tokenize
        else:
            self.tokenize_fn = smiles_tokenize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        tokens = self.tokenize_fn(smiles)

        # Use efficient slicing for Morgan fingerprint substructures
        if self.scheme == "morgan":
            tokens = tokens[1::2]

        masked_indices, label_indices = self.random_mask(tokens)

        # Build input/label sequences with special token boundaries
        input_seq = [self.vocab["<cls>"]] + masked_indices + [self.vocab["<sep>"]]
        label_seq = [self.vocab["<pad>"]] + label_indices + [self.vocab["<pad>"]]

        # Truncate to maximum sequence length
        input_seq = input_seq[:self.seq_len]
        label_seq = label_seq[:self.seq_len]

        return {
            "input": torch.tensor(input_seq, dtype=torch.long),
            "label": torch.tensor(label_seq, dtype=torch.long),
        }

    def random_mask(self, tokens):
        """Implements random masking strategy (15% probability)."""

        masked_indices = []
        label_indices = []

        for token in tokens:
            token_idx = self.vocab.get(token, self.vocab["<unk>"])
            prob = random.random()

            if prob < 0.15:
                # 15% of tokens are chosen for potential masking/replacement
                sub_prob = random.random()

                if sub_prob < 0.8:
                    # 80% replace with <mask>
                    masked_indices.append(self.vocab["<mask>"])
                elif sub_prob < 0.9:
                    # 10% replace with random token from vocabulary
                    masked_indices.append(random.randrange(self.vocab_size))
                else:
                    # 10% keep the original token
                    masked_indices.append(token_idx)

                label_indices.append(token_idx)
            else:
                # 85% of tokens are left unchanged and ignored in loss (label 0)
                masked_indices.append(token_idx)
                label_indices.append(0)

        return masked_indices, label_indices
