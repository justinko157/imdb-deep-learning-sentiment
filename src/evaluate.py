"""
Model evaluation and baseline comparison utilities.

Computes classification metrics (accuracy, precision, recall, F1, AUC-ROC),
runs traditional ML baselines (Logistic Regression, Random Forest, SVM),
and generates a comparison summary.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────
# Deep learning evaluation
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract predictions, probabilities, and true labels from a DataLoader.

    Returns:
        Tuple of (predictions, probabilities, true_labels) as numpy arrays.
    """
    model.eval()
    all_probs, all_labels = [], []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(targets.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)
    return preds, probs, labels


def compute_metrics(
    preds: np.ndarray, probs: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Compute a full set of binary classification metrics.

    Returns:
        Dictionary with accuracy, precision, recall, f1, and auc_roc.
    """
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc_roc": roc_auc_score(labels, probs),
    }


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device, model_name: str
) -> Dict[str, float]:
    """Evaluate a deep learning model and print a classification report.

    Args:
        model: Trained PyTorch model.
        loader: Test DataLoader.
        device: Device for inference.
        model_name: Label for printed output.

    Returns:
        Dictionary of classification metrics.
    """
    preds, probs, labels = get_predictions(model, loader, device)
    metrics = compute_metrics(preds, probs, labels)

    print(f"\n{'='*60}")
    print(f"  {model_name} - Test Results")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Negative", "Positive"]))

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Traditional ML baselines
# ──────────────────────────────────────────────────────────────────────
def run_baselines(
    train_texts: List[List[str]],
    train_labels: List[int],
    test_texts: List[List[str]],
    test_labels: List[int],
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate traditional ML baselines using TF-IDF features.

    Trains Logistic Regression, Random Forest, and Linear SVM on TF-IDF
    representations of the text data.

    Args:
        train_texts: Tokenized training reviews (list of token lists).
        train_labels: Training labels.
        test_texts: Tokenized test reviews.
        test_labels: Test labels.

    Returns:
        Dictionary mapping model names to their metric dictionaries.
    """
    # Convert tokenized texts back to strings for TF-IDF
    train_strings = [" ".join(t) for t in train_texts]
    test_strings = [" ".join(t) for t in test_texts]

    print("\nFitting TF-IDF vectorizer (max 25,000 features)...")
    tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_strings)
    X_test = tfidf.transform(test_strings)

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=50, n_jobs=-1),
        "Linear SVM": LinearSVC(max_iter=2000, C=1.0),
    }

    results = {}
    for name, clf in baselines.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        # SVM does not support predict_proba, use decision function
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_test)
            # Normalize to [0, 1] for AUC calculation
            probs = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            probs = preds.astype(float)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "auc_roc": roc_auc_score(y_test, probs),
        }
        results[name] = metrics
        print(f"  {name}: Accuracy={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f} | AUC={metrics['auc_roc']:.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────────────────────────────
def print_comparison(all_results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted comparison table of all model results.

    Args:
        all_results: Dictionary mapping model names to metric dictionaries.
    """
    print(f"\n{'='*75}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10} {'Precision':>10}")
    print(f"  {'-'*65}")
    for name, metrics in sorted(all_results.items(), key=lambda x: -x[1]["accuracy"]):
        print(
            f"  {name:<25} {metrics['accuracy']:>10.4f} {metrics['f1']:>10.4f} "
            f"{metrics['auc_roc']:>10.4f} {metrics['precision']:>10.4f}"
        )
    print(f"{'='*75}")
