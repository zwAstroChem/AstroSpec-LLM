AstroSpec-LLM: Usage Guide 

Author: Yuan Liu, Zhao Wang and Dong Qiu @GXU

=======================================================

This guide provides instructions for the AstroSpec-LLM framework, a deep-learning system designed to train molecular representations and predict the infrared (IR) spectra of Polycyclic Aromatic Hydrocarbons (PAHs). By treating SMILES strings as 'chemical sentences,' the model enables high-throughput spectral analysis for astrochemistry.

=======================================================

1. Project Introduction

AstroSpec-LLM is a transformer-based framework optimized for the James Webb Space Telescope (JWST) era. It utilizes self-supervised pre-training to master universal molecular features, which are subsequently fine-tuned to predict interstellar PAH spectra. This approach achieves high precision while bypassing the "quartic scaling" bottleneck and extreme computational costs of traditional quantum chemical calculations (e.g., DFT).

=======================================================

2. Workflow

The AstroSpec-LLM pipeline operates in three distinct phases:

(1) Pre-training: The model is trained on a massive, diverse collection of molecular SMILES strings. During this phase, it learns fundamental "chemical grammar" and structural patterns through self-supervised tasks.

(2) Fine-tuning: A task-specific Multi-Layer Perceptron (MLP) head is attached to the pre-trained encoder. The model is then trained on curated PAH datasets (such as NASA Ames PAHdb) to map molecular sequences to specific reference IR spectra.

(3) Inference/Prediction: The finalized model performs end-to-end spectral inference. It generates full IR spectra for novel PAH molecules based solely on their textual SMILES representation and charge state.

=======================================================

3. Code structure

The project is organized into two primary modules based on the workflow stage:

(1) Training Module (./bert_ir_pretraining_finetuning):

Contains the core logic for Phase 1 (Pre-training) and Phase 2 (Fine-tuning). This includes the BERT encoder implementation, dataset loaders, and the .yaml configuration files used to manage hyperparameter experiments.

(2) Prediction Module (./bert_ir_prediction):

Contains the standalone inference scripts (e.g., predict_by_csv.py) and pre-trained weights. This module is designed for researchers who wish to deploy the model directly to predict spectra for new molecular lists without re-training the underlying architecture.

Please see the README files in each folder for detailed Usage Guide.

=======================================================

4. Important: Model Weights

Note: Due to their large size, the model weight files (.pth) in both modules are provided in a compressed ZIP format. You must unzip these files into their respective save/models/ directories before attempting to run fine-tuning or prediction scripts.

=======================================================

5. License

MIT License

Copyright (c) 2026 zwAstroChem
