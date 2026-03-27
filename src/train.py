"""
Training pipeline for deep learning sentiment models.

Supports training with early stopping, learning rate scheduling,
gradient clipping, and comprehensive metric logging.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Hyperparameters and training configuration.

    Attributes:
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for the optimizer.
        weight_decay: L2 regularization strength.
        grad_clip: Maximum gradient norm for clipping.
        patience: Early stopping patience (epochs without improvement).
        min_delta: Minimum validation loss improvement to reset patience.
        checkpoint_dir: Directory to save model checkpoints.
    """

    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 3
    min_delta: float = 1e-4
    checkpoint_dir: str = "checkpoints"


@dataclass
class TrainingHistory:
    """Stores training and validation metrics across epochs."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)


class EarlyStopping:
    """Monitors validation loss and stops training when it stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum improvement required to reset the counter.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop based on validation loss."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, float]:
    """Run a single training epoch.

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a DataLoader without gradient computation.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    model_name: str = "model",
) -> TrainingHistory:
    """Full training loop with early stopping, LR scheduling, and checkpointing.

    Args:
        model: PyTorch model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training hyperparameters.
        device: Device to train on (cpu/cuda/mps).
        model_name: Name prefix for checkpoint files.

    Returns:
        TrainingHistory with per-epoch metrics.
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    early_stop = EarlyStopping(config.patience, config.min_delta)
    history = TrainingHistory()

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Training {model_name} ({param_count:,} parameters)")
    print(f"Device: {device} | Epochs: {config.epochs} | LR: {config.learning_rate}")
    print(f"{'='*60}")

    for epoch in range(1, config.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config.grad_clip
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)
        history.learning_rates.append(current_lr)
        history.epoch_times.append(elapsed)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                checkpoint_dir / f"{model_name}_best.pt",
            )

        scheduler.step(val_loss)

        if early_stop.should_stop(val_loss):
            print(f"\nEarly stopping at epoch {epoch} (patience={config.patience})")
            break

    # Load best checkpoint
    best_ckpt = torch.load(checkpoint_dir / f"{model_name}_best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"\nLoaded best model from epoch {best_ckpt['epoch']} "
          f"(val_loss={best_ckpt['val_loss']:.4f}, val_acc={best_ckpt['val_acc']:.4f})")

    return history
