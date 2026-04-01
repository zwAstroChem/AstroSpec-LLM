"""
@author AFelixLiu
@date 2026 3月 06
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import BERT4IR


class PAHsTrainer:
    """Trainer for PAH Infrared spectra regression."""

    def __init__(self, model: BERT4IR, lr, scheduler_args, device):
        self.device = device
        self.model = model.to(self.device)

        self.optimizer = AdamW(model.parameters(), lr=lr)
        # Dynamic LR adjustment based on validation performance
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_args.factor,
            patience=scheduler_args.patience,
            threshold=scheduler_args.threshold,
            threshold_mode="abs"
        )

    def train_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        epoch_emd_sum, epoch_sis_sum, epoch_mse_sum = 0.0, 0.0, 0.0

        # tqdm progress bar with live loss monitoring
        pbar = tqdm(loader, desc=f"Train Epoch {epoch:03d}", leave=False)

        for i, batch_data in enumerate(pbar):
            batch_data: dict

            inputs = batch_data["input"].to(self.device)
            labels = batch_data["label"].to(self.device)
            charges = batch_data["charge"].to(self.device)

            self.optimizer.zero_grad()
            emd_loss, sis_loss, mse_loss = self.model(inputs, labels, charges)

            # Optimization based on EMD loss
            emd_loss.backward()
            self.optimizer.step()

            epoch_emd_sum += emd_loss.item()
            epoch_sis_sum += sis_loss.item()
            epoch_mse_sum += mse_loss.item()

            # Batch-level normalization for display
            num = inputs.size(0)
            pbar.set_postfix(
                emd=f"{emd_loss.item() / num:.3f}",
                sis=f"{sis_loss.item() / num:.3f}",
                mse=f"{mse_loss.item() / num:.3f}",
            )

        d_size = len(loader.dataset)  # type: ignore
        return epoch_emd_sum / d_size, epoch_sis_sum / d_size, epoch_mse_sum / d_size

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, is_val: bool = False):
        self.model.eval()
        epoch_emd_sum, epoch_sis_sum, epoch_mse_sum = 0.0, 0.0, 0.0

        for batch_data in loader:
            batch_data: dict

            inputs = batch_data["input"].to(self.device)
            labels = batch_data["label"].to(self.device)
            charges = batch_data["charge"].to(self.device)

            emd_loss, sis_loss, mse_loss = self.model(inputs, labels, charges)

            epoch_emd_sum += emd_loss.item()
            epoch_sis_sum += sis_loss.item()
            epoch_mse_sum += mse_loss.item()

        d_size = len(loader.dataset)  # type: ignore
        avg_emd = epoch_emd_sum / d_size

        # Step the scheduler based on EMD loss during validation
        if is_val:
            self.scheduler.step(avg_emd)

        return avg_emd, epoch_sis_sum / d_size, epoch_mse_sum / d_size

    def save_checkpoint(self, path: str, epoch: int):
        """Saves the full model (BERT + Head) and optimizer state."""

        state = {
            "epoch": epoch,
            "bert4ir_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(state, path)

    def load_checkpoint(self, path: str, from_scratch: bool):
        """
        Loads weights.

        If from_scratch is True, only the BERT backbone from a pre-trained (ZINC) checkpoint is loaded.
        """

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if from_scratch:
            # Transfer learning: load backbone weights into the new architecture
            self.model.bert.load_state_dict(checkpoint["bert_state_dict"])
        else:
            # Resume training: load full model and optimizer states
            self.model.load_state_dict(checkpoint["bert4ir_state_dict"])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return 1 if from_scratch else checkpoint["epoch"] + 1
