"""
@author AFelixLiu
@date 2026 3月 06
"""

import os
import pickle
import sys
from pathlib import Path

import swanlab
import torch
from rdkit import rdBase

from src import BERT, BERTLM, ZINCTrainer
from src import load_exps_from_yaml, set_seed, setup_logger
from src import load_zinc, calculate_steps

sys.path.append('.')

rdBase.DisableLog('rdApp.warning')


def training(args, device):
    # --- 1. Experiment Environment Setup ---
    run_id = args.run_id

    # Establish directory structure for logs and model checkpoints
    log_root = os.path.join("save", "logs", "pretrain")
    model_root = os.path.join("save", "models", "pretrain")
    os.makedirs(model_root, exist_ok=True)

    # SwanLab local tracking directory
    swanlab_root = os.path.join("save", "swanlab")
    os.makedirs(swanlab_root, exist_ok=True)

    # Initialize file-based logger and model storage path
    logger = setup_logger(log_root, run_id)
    model_path = os.path.join(model_root, f"bert_{run_id}.pt")

    # Initialize SwanLab Experiment Tracking
    swanlab.init(
        project=args.project.name,
        experiment_name=run_id,
        config={
            "encode_scheme": args.encode.scheme,
            "dataset_id": args.ds_id,
            "rope": args.rope,
            "epochs": args.epochs,
            "batch_size": args.bs,
            "init_lr": args.optimizer.lr,
            "model": args.model,
        },
        logdir=swanlab_root,
        mode="cloud"
    )

    # --- 2. Data & Model Preparation ---
    # Load vocabulary for molecular tokenization
    with open(args.encode.vocab, 'rb') as file:
        vocab = pickle.load(file)

    # Initialize DataLoaders
    train_loader, test_loader = load_zinc(
        args.data_path, args.encode.scheme, vocab, args.model.seq_len, args.bs, args.seed
    )

    # Initialize model
    bert = BERT(len(vocab), args.model.hid_dim, args.model.n_layer, args.model.n_head, args.model.dropout, args.rope)
    model = BERTLM(bert, len(vocab))

    # Calculate optimization steps for linear scheduler
    training_steps, warmup_steps = calculate_steps(len(train_loader), args.epochs)
    trainer = ZINCTrainer(model, args.optimizer.lr, warmup_steps, training_steps, device)

    # --- 3. Training Loop with Resume Support ---
    if os.path.exists(model_path):
        next_epoch = trainer.load_checkpoint(model_path)
        logger.info(f"Resuming training from epoch {next_epoch}")
    else:
        next_epoch = 1

    for epoch in range(next_epoch, args.epochs + 1):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        trainer.save_checkpoint(model_path, epoch)
        logger.info(f"Epoch {epoch:03d} | Loss={train_loss:.6f}, Acc={train_acc:.6f}")

    # --- 4. Final Evaluation ---
    print("\n>>> [TEST] Evaluating model...")
    test_loss, test_acc = trainer.evaluate(test_loader)

    test_info = f"Final Test Result | Loss: {test_loss:.4f} | Acc: {test_acc:.4f}"
    print(f"{'=' * 65}\n{test_info}\n{'=' * 65}")
    logger.info(f"\n{test_info}")

    # --- 5. Resource Cleanup ---
    swanlab.finish()

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    del bert, model, trainer, train_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main(path):
    device_available = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = Path(path)

    if not folder.exists():
        print(f"Error: Path '{path}' not found.")
        return

    files_to_run = sorted([f for f in folder.iterdir() if f.suffix in {'.yaml', '.yml'}])

    for yaml_file in files_to_run:
        print(f">>> Loading YAML File: {yaml_file.name}")
        try:
            all_exps = load_exps_from_yaml(str(yaml_file))

            for exp_args in all_exps:
                try:
                    if not getattr(exp_args, "trained", False):
                        set_seed(getattr(exp_args, "seed", 42))

                        print(f"\n[EXEC] | Exp: {exp_args.run_id}")
                        training(exp_args, device_available)
                    else:
                        print(f"[SKIP] | {exp_args.run_id} | Marked as trained.")
                except Exception as e:
                    print(f"[FAILED] | Experiment {exp_args.run_id} failed: {e}")
        except Exception as file_error:
            print(f"[FILE ERROR] | Could not parse {yaml_file.name}: {file_error}")


if __name__ == '__main__':
    swanlab.login(api_key="SWANLAB_API_KEY", save=True)

    main(path=r"./configs/pretrain/")
