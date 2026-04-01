"""
@author AFelixLiu
@date 2026 3月 07
"""

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .pahs_dataset import PAHsDataset
from .zinc_dataset import ZINCDataset
from ..utils import collate_zinc, collate_pahs


def load_zinc(data_path, scheme, vocab, seq_len, bs, seed=42):
    dataset = load_dataset('csv', data_files=data_path)
    train_test = dataset['train'].train_test_split(test_size=0.2, seed=seed)

    train_set = ZINCDataset(train_test['train'], scheme, vocab, seq_len)
    test_set = ZINCDataset(train_test['test'], scheme, vocab, seq_len)

    params = {"batch_size": bs, "pin_memory": True, "num_workers": 6, "collate_fn": collate_zinc}
    train_loader = DataLoader(train_set, shuffle=True, **params)
    test_loader = DataLoader(test_set, shuffle=False, **params)

    return train_loader, test_loader


def load_pahs(data_path, label_col, scheme, vocab, seq_len, bs, split="random", scale=1.0, seed=42):
    dataset = load_dataset('csv', data_files=data_path)

    if split == "c100":
        test_data = dataset['train'].filter(lambda x: x['n_c'] >= 100)
        train_val_pool = dataset['train'].filter(lambda x: x['n_c'] < 100)
        train_val_data = train_val_pool.train_test_split(test_size=0.2, seed=seed)
        train_data, val_data = train_val_data['train'], train_val_data['test']
    else:
        train_test_data = dataset['train'].train_test_split(test_size=0.2, seed=seed)
        train_data, val_test_pool = train_test_data['train'], train_test_data['test']
        val_test_data = val_test_pool.train_test_split(test_size=0.5, seed=seed)
        val_data, test_data = val_test_data['train'], val_test_data['test']

    train_data = train_data.train_test_split(train_size=scale, seed=seed)['train']

    train_set = PAHsDataset(label_col, train_data, scheme, vocab, seq_len)
    val_set = PAHsDataset(label_col, val_data, scheme, vocab, seq_len)
    test_set = PAHsDataset(label_col, test_data, scheme, vocab, seq_len)

    # Dynamic dimensionality discovery
    sample = train_set[0]
    out_dim = len(sample['label'])

    params = {"batch_size": bs, "pin_memory": True, "num_workers": 6, "collate_fn": collate_pahs}
    train_loader = DataLoader(train_set, shuffle=True, **params)
    val_loader = DataLoader(val_set, shuffle=False, **params)
    test_loader = DataLoader(test_set, shuffle=False, **params)

    return (train_loader, val_loader, test_loader), out_dim


def load_pahs_kfold(fold_idx, n_splits, data_path, label_col, scheme, vocab,
                    seq_len, bs, scale=1.0, seed=42):
    full_data = load_dataset('csv', data_files=data_path)['train']
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_indices = [val_idx for _, val_idx in k_fold.split(range(len(full_data)))]

    test_idx = folds_indices[fold_idx]
    valid_idx = folds_indices[(fold_idx + 1) % n_splits]
    train_indices_list = [folds_indices[i] for i in range(n_splits) if
                          i != fold_idx and i != (fold_idx + 1) % n_splits]
    train_idx = np.concatenate(train_indices_list)

    train_data = full_data.select(train_idx)
    val_data = full_data.select(valid_idx)
    test_data = full_data.select(test_idx)

    train_data = train_data.train_test_split(train_size=scale, seed=seed)['train']

    train_set = PAHsDataset(label_col, train_data, scheme, vocab, seq_len)
    val_set = PAHsDataset(label_col, val_data, scheme, vocab, seq_len)
    test_set = PAHsDataset(label_col, test_data, scheme, vocab, seq_len)

    # Dynamic dimensionality discovery
    sample = train_set[0]
    out_dim = len(sample['label'])

    params = {"batch_size": bs, "pin_memory": True, "num_workers": 6, "collate_fn": collate_pahs}
    train_loader = DataLoader(train_set, shuffle=True, **params)
    val_loader = DataLoader(val_set, shuffle=False, **params)
    test_loader = DataLoader(test_set, shuffle=False, **params)

    return (train_loader, val_loader, test_loader), out_dim
