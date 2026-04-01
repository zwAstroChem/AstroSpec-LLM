"""
@author AFelixLiu
@date 2026 3月 05
"""

import re

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.nn.utils.rnn import pad_sequence


# =================================================================
# Chemical Structure Tokenization (RDKit / SMILES)
# =================================================================

def morgan_tokenize(smi, radius=1, ignore_null_token=True):
    """
    Converts a SMILES string into a list of Morgan fingerprint tokens.

    Args:
        smi (str): SMILES string of the molecule.
        radius (int): Radius for Morgan fingerprinting. Defaults to 1.
        ignore_null_token (bool): Whether to skip atoms with None tokens. Defaults to True.

    Returns:
        List[str]: A list of tokens representing atom environments at different radii.
    """

    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    radius_range = list(range(int(radius) + 1))

    token_info = {}
    AllChem.GetMorganGenerator()
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=token_info)

    atom_indices = [a.GetIdx() for a in mol.GetAtoms()]
    atom_token_map = {atom_idx: {r: None for r in radius_range} for atom_idx in atom_indices}

    for token, atom_radius_pairs in token_info.items():
        for atom_idx, r in atom_radius_pairs:
            atom_token_map[atom_idx][r] = token

    tokens_list = [
        str(atom_token_map[atom_idx][r])
        for atom_idx in atom_token_map
        for r in radius_range
        if atom_token_map[atom_idx][r] is not None or not ignore_null_token
    ]

    return tokens_list


def smiles_tokenize(smi):
    """
    Tokenizes a SMILES string using a regular expression pattern.

    Args:
        smi (str): SMILES string to be tokenized.

    Returns:
        List[str]: A list of individual SMILES tokens (atoms, bonds, brackets, etc.).
    """

    REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%\([0-9]{3}\)|\%[0-9]{2}|[0-9])"
    regex = re.compile(REGEX_PATTERN)
    tokens = [token for token in regex.findall(smi)]

    return tokens


# =================================================================
# Data Loading and Batch Processing (PyTorch DataLoader)
# =================================================================

def collate_zinc(data):
    """
    Collates a batch of data for pre-training with padding.

    Args:
        data (List[Dict]): A list of samples containing 'input' and 'label' tensors.

    Returns:
        Dict[str, Tensor]: Dictionary with padded input and label batch tensors.
    """

    input_list = [d['input'] for d in data]
    label_list = [d['label'] for d in data]
    input_tensor = pad_sequence(input_list, batch_first=True)
    label_tensor = pad_sequence(label_list, batch_first=True)

    return {
        "input": input_tensor,
        "label": label_tensor
    }


def collate_pahs(data):
    """
    Collates a batch of data for fine-tuning, supporting sequence padding and labels.

    Args:
        data (List[Dict]): A list of samples containing 'input', 'label', and 'charge' tensors.

    Returns:
        Dict[str, Tensor]: Dictionary with padded inputs and stacked labels/charges.
    """

    input_list = [d['input'] for d in data]
    label_list = [d['label'] for d in data]
    charge_list = [d['charge'] for d in data]
    input_tensor = pad_sequence(input_list, batch_first=True)
    label_tensor = torch.stack(label_list, dim=0)
    charge_tensor = torch.stack(charge_list, dim=0)

    return {
        "input": input_tensor,
        "label": label_tensor,
        "charge": charge_tensor
    }


# =================================================================
# Other tools
# =================================================================

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
