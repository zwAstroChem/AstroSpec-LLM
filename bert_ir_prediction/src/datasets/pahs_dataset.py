"""
@author AFelixLiu
@date 2026 3月 05
"""

import json

import torch
from torch.utils.data import Dataset as TorchDataset

from ..utils import morgan_tokenize, smiles_tokenize


class PAHsDataset(TorchDataset):
    """Dataset handler for PAHs molecular data with Morgan or SMILES tokenization."""

    def __init__(self, label_col, data, scheme, vocab, seq_len):
        self.label_col = label_col
        # Pre-select required columns to reduce memory overhead
        self.data = data.select_columns(["canonical_smiles", self.label_col, "charge"])
        self.scheme = scheme
        self.vocab = vocab
        self.seq_len = seq_len

        # Select tokenization function once during initialization
        if self.scheme == "morgan":
            self.tokenize_fn = morgan_tokenize
        else:
            self.tokenize_fn = smiles_tokenize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Fetch data row once to minimize indexing overhead
        row = self.data[item]
        raw_input = row["canonical_smiles"]
        label = json.loads(row[self.label_col])
        charge = int(row["charge"])

        tokens = self.tokenize_fn(raw_input)

        # Extract odd-indexed tokens for Morgan fingerprint method
        if self.scheme == "morgan":
            tokens = tokens[1::2]

        # Convert tokens to indices with unknown token fallback
        tokens_idx = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]

        # Construct input sequence with special tokens and fixed length truncation
        token_ids = [self.vocab["<cls>"]] + tokens_idx + [self.vocab["<sep>"]]
        token_ids = token_ids[:self.seq_len]

        return {
            "input": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "charge": torch.tensor(charge, dtype=torch.long),
        }
