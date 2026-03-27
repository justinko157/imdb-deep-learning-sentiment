"""
Visualization utilities for model training and evaluation.

Generates publication-quality plots for training curves, confusion matrices,
ROC curves, and model comparison charts.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

from .train import TrainingHistory

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

COLORS = {
    "train": "#2B579A",
    "val": "#E85D3A",
    "lstm": "#2B579A",
    "cnn": "#E85D3A",
    "lr": "#4CAF50",
    "rf": "#9C27B0",
    "svm": "#FF9800",
}


def plot_training_curves(
    histories: Dict[str, TrainingHistory], save_path: Optional[str] = None
) -> None:
    """Plot training and validation loss/accuracy curves for all models.

    Args:
        histories: Dictionary mapping model names to TrainingHistory objects.
        save_path: If provided, saves the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = list(COLORS.values())

    for i, (name, hist) in enumerate(histories.items()):
        c = colors[i % len(colors)]
        epochs = range(1, len(hist.train_loss) + 1)

        axes[0].plot(epochs, hist.train_loss, f"-", color=c, alpha=0.7, label=f"{name} (train)")
        axes[0].plot(epochs, hist.val_loss, f"--", color=c, label=f"{name} (val)")

        axes[1].plot(epochs, hist.train_acc, f"-", color=c, alpha=0.7, label=f"{name} (train)")
        axes[1].plot(epochs, hist.val_acc, f"--", color=c, label=f"{name} (val)")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = save_path or str(FIGURES_DIR / "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {path}")


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        labels: True labels.
        preds: Predicted labels.
        model_name: Name of the model for the title.
        save_path: If provided, saves the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13)

    plt.tight_layout()
    path = save_path or str(FIGURES_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {path}")


def plot_roc_curves(
    roc_data: Dict[str, tuple],
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC curves for multiple models on a single figure.

    Args:
        roc_data: Dictionary mapping model names to (labels, probabilities) tuples.
        save_path: If provided, saves the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = list(COLORS.values())

    for i, (name, (labels, probs)) in enumerate(roc_data.items()):
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Model Comparison")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    path = save_path or str(FIGURES_DIR / "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves to {path}")


def plot_model_comparison(
    all_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot a grouped bar chart comparing all models across metrics.

    Args:
        all_results: Dictionary mapping model names to metric dictionaries.
        save_path: If provided, saves the figure to this path.
    """
    models = list(all_results.keys())
    metrics = ["accuracy", "f1", "auc_roc", "precision", "recall"]
    metric_labels = ["Accuracy", "F1", "AUC-ROC", "Precision", "Recall"]

    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    colors = list(COLORS.values())

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        values = [all_results[model][m] for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9, label=model, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.set_ylim(0.6, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    path = save_path or str(FIGURES_DIR / "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison chart to {path}")
