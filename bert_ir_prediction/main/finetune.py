"""
@author AFelixLiu
@date 2026 3月 08
"""

import os
import pickle

import swanlab
import torch

from src import BERT, BERT4IR, PAHsTrainer, EarlyStopping
from src import load_exps_from_yaml, set_seed, set_default, setup_logger
from src import load_pahs, load_pahs_kfold, build_run_id


def training(args, run_id, loaders, out_dim):
    print(f"\n[EXEC] | Exp: {run_id}")

    # --- 1. Experiment Environment Setup ---
    train_loader, val_loader, test_loader = loaders

    # Establish directory structure for logs and model checkpoints
    log_root = os.path.join(str(args.save.log), str(args.project.name))
    model_root = os.path.join(str(args.save.model), str(args.project.name))
    os.makedirs(model_root, exist_ok=True)

    # SwanLab local tracking directory
    swanlab_root = os.path.join("save", "swanlab")
    os.makedirs(swanlab_root, exist_ok=True)

    # Initialize file-based logger and model storage path
    logger = setup_logger(log_root, run_id)
    model_path = os.path.join(model_root, f"bert4ir_{run_id}.pt")

    # Initialize SwanLab Experiment Tracking
    swanlab.init(
        project=args.project.name,
        experiment_name=run_id,
        config={
            "support_charge": args.support_charge,
            "encode_scheme": args.encode.scheme,
            "rope": args.rope,
            "label_col": args.label_col,
            "init_lr": args.optimizer.lr,
            "early_stop": args.early_stop,
            "scheduler": args.scheduler,
            "batch_size": args.bs,
            "model": args.model,
        },
        logdir=swanlab_root,
        mode="cloud"
    )

    # --- 2. Model Preparation ---
    # Load vocabulary for molecular tokenization
    with open(args.encode.vocab, 'rb') as file:
        vocab = pickle.load(file)
    charge_emb_dim = args.charge_dim if args.support_charge else 56
    charge_onehot_repeat = args.charge_dim if args.support_charge else 9

    bert = BERT(len(vocab), args.model.hid_dim, args.model.n_layer, args.model.n_head, args.model.dropout, args.rope)
    model = BERT4IR(bert, out_dim, args.data.label_normed, args.support_charge, args.charge.vocab,
                    args.charge.enc, charge_emb_dim, charge_onehot_repeat)
    trainer = PAHsTrainer(model, args.optimizer.lr, args.scheduler, args.device)

    # EarlyStopping monitors Validation EMD loss to prevent overfitting
    early_stopping = EarlyStopping(patience=args.early_stop.patience, delta=args.early_stop.delta, mode="min")

    # --- 3. Training Loop ---
    next_epoch = trainer.load_checkpoint(args.bert_weight, from_scratch=True)

    for epoch in range(next_epoch, 999):
        train_emd, train_sis, train_mse = trainer.train_epoch(train_loader, epoch)
        val_emd, val_sis, val_mse = trainer.evaluate(val_loader, is_val=True)

        # Sync SwanLab
        swanlab.log({
            "train/epochs": epoch,
            "train/emd_loss": train_emd,
            "train/sis_loss": train_sis,
            "train/mse_loss": train_mse,
            "val/epochs": epoch,
            "val/emd_loss": val_emd,
            "val/sis_loss": val_sis,
            "val/mse_loss": val_mse,
        })

        # Determine status message
        status = ""
        if early_stopping(val_emd):
            trainer.save_checkpoint(model_path, epoch)
            status = "| [SAVE]"

        # Log full precision to file
        logger.info(
            f"Epoch {epoch:03d} | T-EMD={train_emd:.4f}, T-SIS={train_sis:.4f}, T-MSE={train_mse:.4f},"
            f"V-EMD={val_emd:.6f}, V-SIS={val_sis:.4f}, V-MSE={val_mse:.4f}, {status}")

        if early_stopping.early_stop:
            msg = f"\n[!] Early stopping at epoch {epoch}."
            print(msg)
            logger.info(msg)
            break

    # --- 4. Final Testing ---
    print("\n>>> [TEST] Evaluating best model...")
    if os.path.exists(model_path):
        trainer.load_checkpoint(model_path, from_scratch=False)

    test_emd, test_sis, test_mse = trainer.evaluate(test_loader)
    swanlab.log({
        "test/emd_loss": test_emd,
        "test/sis_loss": test_sis,
        "test/mse_loss": test_mse,
    })

    test_res = f"Final Test Result | EMD: {test_emd:.4f} | SIS: {test_sis:.4f} | MSE: {test_mse:.4f}"
    print(f"{'=' * 65}\n{test_res}\n{'=' * 65}")
    logger.info(f"\n{test_res}")

    # --- 5. Resource Cleanup ---
    swanlab.finish()

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    del bert, model, trainer, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def kfold_loop(args, scale=1.0):
    if not args.kfold:
        run_id = build_run_id(args, scale)
        loaders, out_dim = load_pahs(args.data.path, args.label_col, args.encode.scheme, args.encode.vocab,
                                     args.model.seq_len, args.bs, args.data.split, scale, args.seed)
        training(args, run_id, loaders, out_dim)
        return

    for fold in range(1, args.kfold + 1):
        run_id = build_run_id(args, scale, fold)
        loaders, out_dim = load_pahs_kfold(fold, args.kfold, args.data.path, args.label_col,
                                           args.encode.scheme, args.encode.vocab,
                                           args.model.seq_len, args.bs, scale, args.seed)
        training(args, run_id, loaders, out_dim)


def run_exp(args):
    args = set_default(args)

    # 1. Validate charge encoding configuration
    if args.support_charge and args.charge.enc not in ["onehot", "emb"]:
        raise ValueError(f"Unsupported charge encoding: {args.charge.enc}. Options: [onehot, emb]")

    # 2. Define internal scale loop to bridge parameters and kfold
    def _run_scale_loop():
        for scale in args.data.scales:
            kfold_loop(args, scale)

    # 3. Determine charge parameter iterations and execute
    if args.support_charge:
        charge_dim_list = args.charge.onehot_repeat if args.charge.enc == "onehot" else args.charge.emb_dim
        for value in charge_dim_list:
            args.charge_dim = value
            _run_scale_loop()
    else:
        # Run standard loop without charge parameter variations
        _run_scale_loop()


def main(file_list):
    device_available = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for yaml_file in file_list:
        print(f">>> Loading YAML File: {yaml_file.name}")
        try:
            all_exps = load_exps_from_yaml(str(yaml_file))

            for exp_args in all_exps:
                try:
                    if not getattr(exp_args, "trained", False):
                        set_seed(getattr(exp_args, "seed", 42))
                        exp_args.device = device_available

                        run_exp(exp_args)
                    else:
                        print(f"[SKIP] | {exp_args.run_id} | Marked as trained.")
                except Exception as e:
                    print(f"[FAILED] | Experiment {exp_args.run_id} failed: {e}")
        except Exception as file_error:
            print(f"[FILE ERROR] | Could not parse {yaml_file.name}: {file_error}")


if __name__ == '__main__':
    swanlab.login(api_key="SWANLAB_API_KEY", save=True)

    files_to_run = ["configs/finetune/task1_morgan_neutral.yaml", "configs/finetune/task2_smiles_neutral.yaml"]
    main(files_to_run)
