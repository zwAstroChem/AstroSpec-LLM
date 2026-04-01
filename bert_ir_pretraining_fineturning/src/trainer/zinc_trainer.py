"""
@author AFelixLiu
@date 2026 3月 06
"""

import swanlab
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..models import BERTLM
from ..utils import MLMAcc


class ZINCTrainer:
    """Trainer for BERT Masked Language Modeling (MLM) on ZINC molecular dataset."""

    def __init__(self, model: BERTLM, lr, warmup_steps, training_steps, device):
        self.device = device
        self.model = model.to(self.device)

        # Optimizer with weight decay correction
        self.optimizer = AdamW(model.parameters(), lr=lr)

        # Learning rate scheduler with warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, training_steps,
        )

        # Loss and Accuracy metrics
        self.criterion_loss = nn.NLLLoss(ignore_index=0)
        self.criterion_acc = MLMAcc(ignore_index=0)

    def train_epoch(self, loader: DataLoader, epoch: int):
        """Performs one epoch of training with gradient updates."""

        self.model.train()
        total_loss, total_acc = 0.0, 0.0

        # tqdm progress bar with live loss monitoring
        pbar = tqdm(loader, desc=f"Train Epoch {epoch:03d}", leave=False)

        for i, batch_data in enumerate(pbar):
            batch_data: dict

            # 1. Data loading and device placement
            inputs = batch_data["input"].to(self.device)
            labels = batch_data["label"].to(self.device)

            # 2. Forward pass
            self.optimizer.zero_grad()
            # Forward pass: out shape [batch, seq_len, vocab_size]
            out = self.model(inputs)

            # 3. Loss & Metric calculation
            # NOTE: Transpose to match the input format [batch_size, num_classes, sequence_length] for NLLLoss
            loss = self.criterion_loss(out.transpose(1, 2), labels)
            acc = self.criterion_acc(out, labels)

            # 4. Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # 5. Statistics tracking
            total_loss += loss.item()
            total_acc += acc.item()

            curr_loss = total_loss / (i + 1)
            curr_acc = (total_acc / (i + 1)) * 100

            # Update tqdm status
            pbar.set_postfix(loss=f"{curr_loss:.4f}", acc=f"{curr_acc:.2f}%")

            # 6. Logging to SwanLab (Stream metrics)
            swanlab.log({
                "train/loss": curr_loss,
                "train/acc": curr_acc,
                "train/lr": self.optimizer.param_groups[0]['lr']
            })

        return total_loss / len(loader), total_acc / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        """Evaluates the model on validation/test set."""

        self.model.eval()
        total_loss, total_acc = 0.0, 0.0

        pbar = tqdm(loader, desc=f"Eval", leave=False)

        for i, batch_data in enumerate(pbar):
            batch_data: dict

            # 1. Data loading and device placement
            inputs = batch_data["input"].to(self.device)
            labels = batch_data["label"].to(self.device)

            # 2. Forward pass only
            output = self.model(inputs)

            # 3. Statistics tracking
            total_loss += self.criterion_loss(output.transpose(1, 2), labels).item()
            total_acc += self.criterion_acc(output, labels).item()

            curr_loss = total_loss / (i + 1)
            curr_acc = (total_acc / (i + 1)) * 100

            # Update progress bar
            pbar.set_postfix(loss=f"{curr_loss:.4f}", acc=f"{curr_acc:.2f}%")

            # 4. Logging to SwanLab (Stream metrics)
            swanlab.log({
                "eval/loss": curr_loss,
                "eval/acc": curr_acc,
            })

        return total_loss / len(loader), total_acc / len(loader)

    def save_checkpoint(self, path: str, epoch: int):
        """Saves BERT backbone and optimizer state."""

        state = {
            "epoch": epoch,
            "bert_state_dict": self.model.bert.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(state, path)

    def load_checkpoint(self, path: str):
        """Loads BERT backbone and optimizer."""

        checkpoint = torch.load(path, map_location=self.device)
        self.model.bert.load_state_dict(checkpoint['bert_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'] + 1
