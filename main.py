"""
IMDB Deep Learning Sentiment Analysis
======================================

End-to-end pipeline that:
1. Downloads and preprocesses the IMDB movie review dataset (50K reviews)
2. Trains two deep learning architectures (BiLSTM with Attention, Multi-Kernel CNN)
3. Compares against traditional ML baselines (Logistic Regression, Random Forest, SVM)
4. Generates publication-quality visualizations and a performance summary

Usage:
    python main.py                    # Full pipeline (default: GPU if available)
    python main.py --epochs 10        # Custom epoch count
    python main.py --skip-baselines   # Skip traditional ML baselines
    python main.py --device cpu       # Force CPU training

Author: Justin Ko
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.dataset import download_imdb, get_dataloaders, load_split
from src.evaluate import evaluate_model, get_predictions, print_comparison, run_baselines
from src.model import CNNClassifier, LSTMClassifier
from src.train import TrainingConfig, train_model
from src.visualize import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves,
    plot_training_curves,
)


def get_device(requested: str = "auto") -> torch.device:
    """Select the best available compute device.

    Priority: CUDA GPU > Apple MPS > CPU

    Args:
        requested: Specific device string, or 'auto' for automatic selection.

    Returns:
        torch.device for training.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="IMDB Sentiment Analysis with Deep Learning")
    parser.add_argument("--epochs", type=int, default=15, help="Max training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--max-len", type=int, default=300, help="Max sequence length (default: 300)")
    parser.add_argument("--max-vocab", type=int, default=25000, help="Vocabulary size (default: 25000)")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension (default: 128)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden dimension (default: 256)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip traditional ML baselines")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # ── Reproducibility ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  IMDB SENTIMENT ANALYSIS WITH DEEP LEARNING")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Vocab Size: {args.max_vocab:,}")
    print(f"  Seq Length: {args.max_len}")
    print(f"  Seed:       {args.seed}")
    print("=" * 60)

    # ── 1. Data ──
    print("\n[1/5] Preparing data...")
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_vocab=args.max_vocab,
    )

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    all_results = {}
    histories = {}
    roc_data = {}

    # ── 2. BiLSTM with Attention ──
    print("\n[2/5] Training BiLSTM with Attention...")
    lstm_model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        dropout=0.3,
    )
    lstm_hist = train_model(lstm_model, train_loader, val_loader, config, device, "lstm")
    histories["BiLSTM + Attention"] = lstm_hist

    lstm_metrics = evaluate_model(lstm_model, test_loader, device, "BiLSTM + Attention")
    all_results["BiLSTM + Attention"] = lstm_metrics

    preds, probs, labels = get_predictions(lstm_model, test_loader, device)
    roc_data["BiLSTM + Attention"] = (labels, probs)
    plot_confusion_matrix(labels, preds, "BiLSTM + Attention")

    # ── 3. Multi-Kernel CNN ──
    print("\n[3/5] Training Multi-Kernel CNN...")
    cnn_model = CNNClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_filters=100,
        kernel_sizes=[2, 3, 4, 5],
        dropout=0.3,
    )
    cnn_hist = train_model(cnn_model, train_loader, val_loader, config, device, "cnn")
    histories["Multi-Kernel CNN"] = cnn_hist

    cnn_metrics = evaluate_model(cnn_model, test_loader, device, "Multi-Kernel CNN")
    all_results["Multi-Kernel CNN"] = cnn_metrics

    preds, probs, labels = get_predictions(cnn_model, test_loader, device)
    roc_data["Multi-Kernel CNN"] = (labels, probs)
    plot_confusion_matrix(labels, preds, "Multi-Kernel CNN")

    # ── 4. Traditional ML baselines ──
    if not args.skip_baselines:
        print("\n[4/5] Running traditional ML baselines...")
        imdb_path = download_imdb()
        train_texts, train_labels = load_split(imdb_path, "train")
        test_texts, test_labels = load_split(imdb_path, "test")

        baseline_results = run_baselines(train_texts, train_labels, test_texts, test_labels)
        all_results.update(baseline_results)
    else:
        print("\n[4/5] Skipping baselines (--skip-baselines)")

    # ── 5. Visualizations and summary ──
    print("\n[5/5] Generating visualizations...")
    plot_training_curves(histories)
    plot_roc_curves(roc_data)
    plot_model_comparison(all_results)
    print_comparison(all_results)

    # Save results to JSON
    json_path = results_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── Summary ──
    best_model = max(all_results, key=lambda k: all_results[k]["accuracy"])
    print(f"\nBest model: {best_model} (Accuracy: {all_results[best_model]['accuracy']:.4f})")
    print("\nDone! Check figures/ for visualizations and results/ for metrics.")


if __name__ == "__main__":
    main()
