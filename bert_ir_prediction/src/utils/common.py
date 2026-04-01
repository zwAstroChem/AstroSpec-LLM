"""
@author AFelixLiu
@date 2026 3月 05
"""

import copy
import logging
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml


class EarlyStopping:
    """
    Early stopping monitor to terminate training when a metric stops improving.

    Args:
        patience (int): How many epochs to wait after last time the monitor improved.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        mode (str): One of ['min', 'max']. In 'min' mode, training stops when the quantity
                    monitored has stopped decreasing; in 'max' mode it stops when the
                    quantity monitored has stopped increasing.
    """

    def __init__(self, patience=20, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        # Initialize best_score based on mode
        if self.mode == 'min':
            self.val_min = np.inf
        elif self.mode == 'max':
            self.val_max = -np.inf
        else:
            raise ValueError("Mode must be 'min' or 'max'")

    def __call__(self, val):
        """
        Returns:
            bool: True if an improvement is detected, False otherwise.
        """

        if self.best_score is None:
            self.best_score = val
            return True

        if self.mode == 'min':
            # Improvement: current value is smaller than (best - delta)
            if val < self.best_score - self.delta:
                self.best_score = val
                self.counter = 0
                return True
            else:
                self.counter += 1

        else:  # mode == 'max'
            # Improvement: current value is larger than (best + delta)
            if val > self.best_score + self.delta:
                self.best_score = val
                self.counter = 0
                return True
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return False


def load_exps_from_yaml(yaml_file):
    """Parses a pre-training YAML config into a list of merged SimpleNamespace objects."""

    # Check if the configuration file exists to avoid FileNotFoundError
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"YAML file not found at: {yaml_file}")

    with open(yaml_file, 'r', encoding='utf-8') as f:
        # Load the YAML file into a dictionary
        # Use safe_load to prevent execution of arbitrary code within YAML
        data = yaml.safe_load(f)

    # Extract base settings and experiment variations
    base_config = data.get('base_settings', {})
    experiments = data.get('experiments', [])

    def deep_merge(base, args):
        """Recursively update the base_config with specific experiment parameters."""

        for key, value in args.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def wrap_namespace(d):
        """Recursively wrap dictionaries to SimpleNamespace for dot-notation access."""

        if isinstance(d, dict):
            return SimpleNamespace(**{k: wrap_namespace(v) for k, v in d.items()})
        return d

    all_exps = []
    for exp_args in experiments:
        # Use deepcopy to ensure each exp starts with an isolated base_config
        merged_dict = deep_merge(copy.deepcopy(base_config), exp_args)
        all_exps.append(wrap_namespace(merged_dict))

    return all_exps


def set_seed(seed=42):
    """Ensure reproducible results by seeding all random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(path, run_id):
    """
    Configures and returns a logger that outputs to both console and file.

    Args:
        path (str): Directory where the log file will be saved.
        run_id (str): Unique identifier for the current run.
    """

    os.makedirs(path, exist_ok=True)
    log_path = os.path.join(path, f"{run_id}.log")

    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)

    # Clean up existing handlers for re-run safety
    if logger.handlers:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    # Create handlers
    file_handler = logging.FileHandler(log_path)
    # stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')

    for h in [file_handler]:
        h.setFormatter(formatter)
        logger.addHandler(h)

    return logger


def calculate_steps(len_train_loader, epochs, warmup_ratio=0.1):
    """Computes total training steps and warmup steps for the scheduler."""

    training_steps = len_train_loader * epochs
    warmup_steps = int(training_steps * warmup_ratio)

    return training_steps, warmup_steps


def set_default(args):
    """
    Standardize the experiment configuration by injecting default values for optional fields,
    ensuring cross-compatibility between different experiment types.
    """

    # 1. Initialize 'support_charge' based on the presence of the 'charge' config block
    args.support_charge = hasattr(args, "charge")

    # 2. Set default values for data processing and splitting strategies
    # 'split': Strategy for data division (e.g., "random", "c100")
    # 'scales': List of ratios for training set size ablation (e.g., [0.25, 0.5])
    # 'kfold': Number of folds for cross-validation; None means standard split
    d_vars = vars(args.data)
    d_vars.setdefault("split", "random")
    d_vars.setdefault("scales", [1.0])
    d_vars.setdefault("kfold", None)

    # 3. Handle charge system defaults to maintain structure consistency
    # Even if charge is not supported, initialize the namespace to prevent AttributeError
    if not args.support_charge:
        args.charge = SimpleNamespace()

    c_vars = vars(args.charge)
    # Default vocabulary for molecular charges
    c_vars.setdefault("vocab", [-1, 0, 1, 2])
    # Default encoding method: "emb" (Embedding) or "onehot"
    c_vars.setdefault("enc", "emb")
    # Hyperparameters for charge encoding dimensionality/repetition
    c_vars.setdefault("onehot_repeat", [9])
    c_vars.setdefault("emb_dim", [56])

    # 4. Set seed
    args.seed = 42

    return args


def build_run_id(args, scale, fold_idx=None):
    charge_info = ""
    split_info = ""
    scale_info = ""
    fold_info = ""

    if args.support_charge:
        if args.charge.enc == "emb":
            charge_info = f"{args.charge_dim}"
        if args.charge.enc == "onehot":
            charge_info = f"[{len(args.charge.vocab)}x{args.charge_dim}]"

    if not args.data.split == "random":
        split_info = f"_{args.data.split}"

    if not args.data.scales == [1.0]:
        scale_info = f"_{scale}x"

    if args.data.kfold:
        fold_info = f"_fold[{fold_idx}-{args.data.kfold}]"

    return args.run_id + charge_info + split_info + scale_info + fold_info
