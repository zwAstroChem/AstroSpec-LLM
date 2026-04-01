## Configuration Guide

---

## 1. Fine-tuning Tasks List

| File Name                                 | Description                                                                                                     |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **`task1_morgan_neutral.yaml`**           | Uses **Morgan** fingerprinting based on the **pure neutral** dataset.                                           |
| **`task2_smiles_neutral.yaml`**           | Uses **SMILES** tokenization based on the **pure neutral** dataset.                                             |
| **`task3_smiles_all.yaml`**               | Uses **SMILES** tokenization based on the **complete (all)** dataset.                                           |
| **`task4_smiles_all_charge.yaml`**        | Uses **SMILES** tokenization based on the **complete dataset**, including **charge information**.               |
| **`task5_smiles_neutral_c100split.yaml`** | Uses **SMILES** tokenization on **neutral data**; test set restricted to molecules with $\ge 100$ Carbon atoms. |
| **`task6_smiles_neutral_scale.yaml`**     | Uses **SMILES** tokenization on **neutral data** to evaluate the impact of **training set scale**.              |
| **`task7_smiles_all_charge_kfold.yaml`**  | Uses **SMILES** tokenization on the **complete dataset** with **charge info** and **K-Fold cross-validation**.  |

---

## 2. Configuration Parameters Reference

### 2.1 Global Settings (`base_settings`)

**Model Architecture (`model`)**

* **`n_layer`**: Number of Transformer blocks (e.g., 6).
* **`n_head`**: Number of self-attention heads (e.g., 12).
* **`hid_dim`**: Dimensionality of the hidden layers and embedding size (e.g., 768).
* **`seq_len`**: Maximum token count per molecular sequence (e.g., 384).
* **`dropout`**: Probability of zeroing neurons for regularization.

**Optimizer & Encoding (`optimizer` & `encode`)**

* **`lr`**: Initial learning rate for the training process.
* **`scheme`**: Molecular featurization strategy, such as `smiles` or `morgan`.
* **`vocab`**: Path to the vocabulary file in `.pickle` format.

**Data & Project (`data` & `project`)**

* **`label_normed`**: Boolean flag indicating if target labels (IR intensities) are pre-normalized.
* **`path`**: Relative path to the source CSV dataset file.
* **`scales`**: A list of values defining different training set sizes for scaling tests.
* **`kfold`**: Integer specifying the total number of cross-validation folds.
* **`split`**: Partitioning strategy: c100 (train on Carbon < 100 only) or random (standard split).
* **`name`**: General project name used for tracking and identification.

**Storage (`save`)**

* **`model`** / **`log`**: Directory paths for saving trained model checkpoints and execution logs.

---

### 2.2 Experiment Matrix (`experiments`)

**Execution & Identity**

* **`run_id`**: Unique identifier for the run, used for logging and saving.
* **`trained`**: Execution Switch: If `true`, the experiment is marked as completed; if `false`, it's pending.
* **`bert_weight`**: Path to the pre-trained BERT weights used for fine-tuning initialization.

**Hyperparameters & Training Logic**

* **`rope`**: **Rotary Positional Embeddings**: Whether to use RoPE (`true`) or Absolute Positional Embeddings (
  `false`).
* **`bs`**: Batch size (number of samples per training step).
* **`label_col`**: The specific target column in the dataset (e.g., `norm_inten_low`).
* **`early_stop`**: Criteria including `patience` (epochs to wait) and `delta` (minimum improvement).
* **`scheduler`**: Learning rate adjustment policy using `factor` and `threshold`.

---

### 2.3 Charge Module (Optional)

This section is only required for tasks involving charge info.

**Global Charge Definition**

* **`vocab`**: A defined list of supported molecular charge states, typically $[-1, 0, 1, 2]$.

**Experimental Encoding Strategies**

* **One-Hot Encoding (`enc: "onehot"`)**:
    * **`onehot_repeat`**: Number of times the one-hot vector.

* **Embedding Encoding (`enc: "emb"`)**:
    * **`emb_dim`**: The dimensionality of the learnable embedding layer dedicated to charge states.
