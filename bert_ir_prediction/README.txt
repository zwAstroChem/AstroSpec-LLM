AstroSpec-LLM: Usage Guide 2: Spectrum Prediction
Author: Yuan Liu, Zhao Wang and Dong Qiu @GXU
=================================================================

This set of Python scripts enable the direct prediction of Infrared (IR) spectra for Polycyclic Aromatic Hydrocarbon (PAH) molecules using their SMILES strings. 
By leveraging a pre-trained BERT-based model, the script automates the inference process to generate spectral data from molecular structures.

1. Overview
The script performs inference on a batch of molecules provided in a CSV file. 
It utilizes a BERT architecture with Rotary Positional Embeddings (RoPE) to extract structural features from SMILES and combines them with Charge Information to output predicted IR spectra.
The prediction is split into two frequency domains:
Low Frequency: Outputs a 105-dimensional spectrum vector.
High Frequency: Outputs a 71-dimensional spectrum vector.

2. Prerequisites
a. Environment Requirements
Python: 3.8+PyTorch: Compatible with your CUDA version (for GPU acceleration).
Dependencies: pandas, numpy, tqdm, pickle.
Project Structure: The script assumes it is located within a specific directory structure (two levels deep from the root) to correctly resolve imports from src.
b. Required Data Files: 
Ensure the following files exist at the paths defined in the PATH dictionary within the script:
Vocab File: data/vocabs/vocab_smiles.pickle (SMILES token mapping).
Model Weights (Low): save/models/finetune/task4_smiles_all_charge/BERT4IR_..._low_...best.pth.
Model Weights (High): save/models/finetune/task4_smiles_all_charge/BERT4IR_..._high_...best.pth.
Important: Please note that the two model weight .pth files are provided in a compressed (.zip) format due to their large file size. You must unzip them and place the resulting .pth files in the save/models/finetune/task4_smiles_all_charge/ directory before running the script.

3. Input File Format
The input must be a CSV file (default: data/pahdb/pahdb_w24146_all_test_quicktest.csv) containing at least the following columns:
Column Name : Description
canonical_smiles : The standardized SMILES string of the molecule.
charge : Integer value of the formal charge (supported: -1, 0, 1, 2).

4. Configuration and Customization
You can modify the following variables in the if __name__ == '__main__': block or the PATH dictionary:
file_path: Change this to point to your specific input CSV file.
output_dir: The directory where the results will be saved (default: predicted_spectra).
device: The script automatically detects cuda but can be forced to cpu if needed.
charge_encoding: The current model configuration uses onehot encoding with a onehot_repeat of 9 to emphasize charge features.

5. Running the Script
Execute the script from the command line from the main folder: python ./infer/pred/predict_by_csv.py
Execution Logic:
Initialization: Sets the working directory and appends the project root to the system path.
Model Loading: Initializes the BERT4IR model with 6 layers and 12 attention heads.
Tokenization: Converts SMILES into tokens and adds [CLS] and [SEP] tags.
Inference:Iterates through each row in the CSV.
Passes the tokenized sequence and the charge value into the model.
Uses torch.abs() on the final output to ensure non-negative IR intensities.
Output: Generates a new file named pahdb_with_predictions.csv.

6. Understanding the Output
The output CSV contains all original columns plus two new prediction columns:
predicted_spectrum_low: A list of 105 floating-point values representing the low-frequency IR intensities.
predicted_spectrum_high: A list of 71 floating-point values representing the high-frequency IR intensities.

7. Troubleshooting
"FileNotFoundError": Ensure your PATH dictionary matches the actual file locations on your disk.
"Unsupported charge value": The model only supports charges defined in charge_vocab (e.g., [-1, 0, 1, 2]). Ensure your CSV data is within this range.
Memory Issues: If running out of VRAM, consider reducing the batch size (though this script currently processes molecules one by one via a loop).

8. License
MIT License
Copyright (c) 2026 zwAstroChem