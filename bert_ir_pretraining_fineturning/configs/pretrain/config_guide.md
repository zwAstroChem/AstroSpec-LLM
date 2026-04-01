# Configuration Guide (`config.yaml`)

**This document provides a detailed reference for all parameters used in the configuration file, ensuring clarity on their functions and types.**

---

## 1. Global Settings (`base_settings`)

The global parameters are defined using the YAML anchor `&defaults`, providing baseline architecture and optimizer settings for all runs.

### 1.1 Model Architecture (`model`)

* **`n_layer`** $\rightarrow$ Number of Transformer blocks (e.g., 6).
* **`n_head`** $\rightarrow$ Number of self-attention heads (e.g., 12).
* **`hid_dim`** $\rightarrow$ Dimensionality of the hidden layers / Embedding size (e.g., 768).
* **`seq_len`** $\rightarrow$ Maximum token count per molecular sequence (e.g., 384).
* **`dropout`** $\rightarrow$ Probability of zeroing neurons for regularization.

### 1.2 Optimizer & Encoding (`optimizer` & `encode`)

* **`lr`** $\rightarrow$ Initial learning rate for the training process.
* **`scheme`** $\rightarrow$ Molecular featurization strategy (e.g., `morgan`, `smiles`).
* **`vocab`** $\rightarrow$ Path to the vocabulary file (e.g., `.pickle` format).

### 1.3 Project (`project`)

* **`name`** $\rightarrow$ General project name used for tracking and identification.
* **`seed`** $\rightarrow$ Random seed.

---

## 2. Experiment Matrix (`experiments`)

**This section defines the specific configurations for each individual training run.**

### 2.1 Execution & Identity

* **`run_id`** $\rightarrow$ **Unique identifier** for the run, used for logging and saving checkpoints.
* **`trained`** $\rightarrow$ **Execution Switch**: If `true`, the task is marked as completed; if `false`, the task is pending.
* **`ds_id`** $\rightarrow$ Dataset group identifier (e.g., `ds1` to `ds6`).

### 2.2 Hyperparameters & Logic

* **`rope`** $\rightarrow$ **Rotary Positional Embeddings**: Whether to use RoPE (`true`) or Absolute Positional Embeddings (APE, `false`).
* **`epochs`** $\rightarrow$ Total training iterations (varies from 8 to 90 based on dataset size).
* **`bs`** $\rightarrow$ Batch size (number of samples per training step).

### 2.3 Data Path

* **`data_path`** $\rightarrow$ Local path to the source CSV dataset file containing molecular SMILES.
