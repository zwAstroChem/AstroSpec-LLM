AstroSpec-LLM Usage Guide 1: Pretraining and Fine-turning 
Author: Yuan Liu, Zhao Wang and Dong Qiu @GXU

================================================================================
PAH-BERT: MOLECULAR REPRESENTATION LEARNING FOR ASTROCHEMISTRY
INTRODUCTION

This project is a molecular representation learning and property prediction
framework based on the BERT architecture. It is specifically designed for
researching Polycyclic Aromatic Hydrocarbons (PAHs) in astrochemistry.

Project Core: Through large-scale self-supervised pre-training on chemical
molecules, the framework constructs a universal representation system with
high generalization capabilities.

Application Value: By fine-tuning the pre-trained model, the framework
achieves high-precision prediction of interstellar PAH infrared (IR) spectra.
This provides a high-performance computational tool for deep-space chemical
research where laboratory experimental data is often scarce.

DIRECTORY STRUCTURE

configs/      : Model hyperparameters and experimental configurations.

pretrain/ : Configurations for the pre-training phase.

finetune/ : Configurations for various fine-tuning tasks.

data/         : Data storage directory.

corpus/   : Molecular datasets used for large-scale pre-training.

pahdb/    : PAHdb IR spectral datasets for fine-tuning.

vocabs/   : Vocabulary files for molecular tokenization.

main/         : Task execution entries.

pretrain.py: Script to launch pre-training tasks.

finetune.py: Script to launch fine-tuning tasks.

run.sh    : Automation script for server-side execution.

save/         : Stores model weights and logs during training.

src/          : Core source code.

datasets/ : Data loading and processing logic.

models/   : Implementation of model architectures.

trainer/  : Training loops, validation logic, and optimizer strategies.

utils/    : Auxiliary utility functions.

environment.yml: Conda environment configuration file.

QUICK START: PRE-TRAINING

STEP 1: Data Preparation (Optional)
To use a custom dataset for pre-training:

Format: Files must be .csv with the data column named 'canonical_smiles'.

Path: Place files in 'data/corpus/'.

Config: Update 'data_path' in 'configs/pretrain/*.yaml' to point to your file.

STEP 2: Experiment Tracking
The project integrates SwanLab by default for visual tracking.

SwanLab (Default):

Obtain an API Key from https://swanlab.cn/

Insert Key in 'main/pretrain.py':
swanlab.login(api_key="YOUR_API_KEY", save=True)

WandB: Replace SwanLab code in 'main/pretrain.py' and 'src/trainer/zinc_trainer.py'.

Disable Tracking: Comment out SwanLab calls in the scripts mentioned above.

STEP 3: Environment Setup
Run the following in your terminal:
conda env create -f environment.yml -y
conda activate pah_bert_pt210_smiles

STEP 4: Parameter Tuning
Modify .yaml files in 'configs/pretrain/'. Note the 'trained' flag:

trained: true  -> Scheme is complete and will be skipped.

trained: false -> Scheme is pending execution.

STEP 5: Execution
Ensure your terminal is in the project root:
Option A (Direct): python3 main/pretrain.py
Option B (Shell) : bash main/run.sh

QUICK START: FINE-TUNING

STEP 1: Logging & Setup
Follow the same Experiment Tracking and Environment Setup steps as described
in the Pre-training section, ensuring API keys are set in 'main/finetune.py'.

STEP 2: Parameter Tuning
Modify .yaml files in 'configs/finetune/'. Refer to 'configs/finetune/config_guide.md'
for detailed parameter meanings.

STEP 3: Specify Tasks
Manually define the configuration files to be executed in the
"if name == 'main':" section of 'main/finetune.py':

files_to_run = [
"configs/finetune/task1_morgan_neutral.yaml",
"configs/finetune/task2_smiles_neutral.yaml"
]

STEP 4: Execution
Ensure your terminal is in the project root:
Option A (Direct): python3 main/finetune.py
Option B (Shell) : bash main/run.sh (Ensure script points to finetune.py)

================================================================================
License
MIT License
Copyright (c) 2026 zwAstroChem