"""Tests for the IMDB sentiment analysis pipeline."""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import Vocabulary, IMDBDataset, clean_text
from src.model import LSTMClassifier, CNNClassifier
from src.train import TrainingConfig, TrainingHistory, EarlyStopping
from src.evaluate import compute_metrics
from src.visualize import FIGURES_DIR


# ── Dataset Tests ──

def test_clean_text():
    result = clean_text("<br/>Hello, World! 123")
    assert isinstance(result, list), "clean_text should return a list"
    assert all(tok.isalpha() for tok in result), "tokens should only contain letters"
    assert all(tok.islower() for tok in result), "tokens should be lowercase"
    assert "hello" in result
    assert "world" in result
    print("PASS: test_clean_text")


def test_clean_text_html():
    result = clean_text("<p>Some <b>bold</b> text</p>")
    assert "p" not in result, "HTML tags should be stripped"
    assert "b" not in result
    assert "some" in result
    assert "bold" in result
    assert "text" in result
    print("PASS: test_clean_text_html")


def test_vocabulary():
    texts = [["the", "cat", "sat"], ["the", "dog", "sat"], ["the", "cat", "ran"]]
    vocab = Vocabulary(max_size=100, min_freq=1).build(texts)
    assert len(vocab) > 2, "vocab should have more than just PAD and UNK"
    assert vocab.token2idx["<PAD>"] == 0
    assert vocab.token2idx["<UNK>"] == 1
    assert "the" in vocab.token2idx
    assert "cat" in vocab.token2idx
    print("PASS: test_vocabulary")


def test_vocabulary_min_freq():
    texts = [["common", "common", "rare"], ["common"]]
    vocab = Vocabulary(max_size=100, min_freq=2).build(texts)
    assert "common" in vocab.token2idx
    assert "rare" not in vocab.token2idx, "rare tokens below min_freq should be excluded"
    print("PASS: test_vocabulary_min_freq")


def test_vocabulary_encode():
    texts = [["hello", "world"]]
    vocab = Vocabulary(max_size=100, min_freq=1).build(texts)
    encoded = vocab.encode(["hello", "unknown_token", "world"])
    assert encoded[0] == vocab.token2idx["hello"]
    assert encoded[1] == vocab.token2idx["<UNK>"]
    assert encoded[2] == vocab.token2idx["world"]
    print("PASS: test_vocabulary_encode")


def test_imdb_dataset():
    texts = [["hello", "world"], ["foo", "bar", "baz"]]
    labels = [0, 1]
    vocab = Vocabulary(max_size=100, min_freq=1).build(texts)
    dataset = IMDBDataset(texts, labels, vocab, max_len=5)
    assert len(dataset) == 2
    seq, label = dataset[0]
    assert seq.shape == (5,), f"Expected shape (5,), got {seq.shape}"
    assert label.item() == 0.0
    print("PASS: test_imdb_dataset")


def test_imdb_dataset_padding():
    texts = [["a"]]
    labels = [1]
    vocab = Vocabulary(max_size=100, min_freq=1).build(texts)
    dataset = IMDBDataset(texts, labels, vocab, max_len=10)
    seq, _ = dataset[0]
    assert seq.shape == (10,)
    assert seq[1:].sum().item() == 0, "padding positions should be 0"
    print("PASS: test_imdb_dataset_padding")


def test_imdb_dataset_truncation():
    texts = [["a", "b", "c", "d", "e"]]
    labels = [0]
    vocab = Vocabulary(max_size=100, min_freq=1).build(texts)
    dataset = IMDBDataset(texts, labels, vocab, max_len=3)
    seq, _ = dataset[0]
    assert seq.shape == (3,), "sequence should be truncated to max_len"
    print("PASS: test_imdb_dataset_truncation")


# ── Model Tests ──

def test_lstm_forward():
    model = LSTMClassifier(vocab_size=100, embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
    model.eval()
    x = torch.randint(0, 100, (4, 10))  # batch=4, seq_len=10
    out = model(x)
    assert out.shape == (4,), f"Expected shape (4,), got {out.shape}"
    print("PASS: test_lstm_forward")


def test_cnn_forward():
    model = CNNClassifier(vocab_size=100, embed_dim=16, num_filters=8, kernel_sizes=[2, 3], dropout=0.0)
    model.eval()
    x = torch.randint(0, 100, (4, 10))
    out = model(x)
    assert out.shape == (4,), f"Expected shape (4,), got {out.shape}"
    print("PASS: test_cnn_forward")


def test_lstm_output_range():
    model = LSTMClassifier(vocab_size=50, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
    model.eval()
    x = torch.randint(0, 50, (8, 20))
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    assert (probs >= 0).all() and (probs <= 1).all(), "sigmoid output should be in [0,1]"
    print("PASS: test_lstm_output_range")


def test_cnn_output_range():
    model = CNNClassifier(vocab_size=50, embed_dim=8, num_filters=8, kernel_sizes=[2, 3], dropout=0.0)
    model.eval()
    x = torch.randint(0, 50, (8, 20))
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    assert (probs >= 0).all() and (probs <= 1).all(), "sigmoid output should be in [0,1]"
    print("PASS: test_cnn_output_range")


# ── Training Utilities Tests ──

def test_early_stopping():
    es = EarlyStopping(patience=2, min_delta=0.01)
    assert not es.should_stop(1.0)
    assert not es.should_stop(0.5)
    assert not es.should_stop(0.5)   # no improvement, counter=1
    assert es.should_stop(0.5)       # no improvement, counter=2 >= patience
    print("PASS: test_early_stopping")


def test_early_stopping_reset():
    es = EarlyStopping(patience=2, min_delta=0.01)
    es.should_stop(1.0)
    es.should_stop(1.0)  # counter=1
    es.should_stop(0.5)  # improvement, counter resets
    assert not es.should_stop(0.5)  # counter=1
    assert es.should_stop(0.5)      # counter=2
    print("PASS: test_early_stopping_reset")


def test_training_history():
    hist = TrainingHistory()
    assert hist.train_loss == []
    hist.train_loss.append(0.5)
    hist.val_loss.append(0.6)
    assert len(hist.train_loss) == 1
    assert len(hist.val_loss) == 1
    print("PASS: test_training_history")


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.epochs == 20
    assert config.learning_rate == 1e-3
    assert config.patience == 3
    assert config.grad_clip == 1.0
    print("PASS: test_training_config_defaults")


# ── Evaluation Tests ──

def test_compute_metrics():
    labels = np.array([0, 0, 1, 1, 1])
    preds = np.array([0, 1, 1, 1, 0])
    probs = np.array([0.2, 0.6, 0.8, 0.9, 0.3])
    metrics = compute_metrics(preds, probs, labels)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "auc_roc" in metrics
    assert metrics["accuracy"] == 0.6
    print("PASS: test_compute_metrics")


def test_compute_metrics_perfect():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(preds, probs, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    print("PASS: test_compute_metrics_perfect")


# ── Visualization Tests ──

def test_figures_dir():
    assert FIGURES_DIR == Path("figures")
    print("PASS: test_figures_dir")


# ── Run all tests ──

if __name__ == "__main__":
    tests = [
        test_clean_text,
        test_clean_text_html,
        test_vocabulary,
        test_vocabulary_min_freq,
        test_vocabulary_encode,
        test_imdb_dataset,
        test_imdb_dataset_padding,
        test_imdb_dataset_truncation,
        test_lstm_forward,
        test_cnn_forward,
        test_lstm_output_range,
        test_cnn_output_range,
        test_early_stopping,
        test_early_stopping_reset,
        test_training_history,
        test_training_config_defaults,
        test_compute_metrics,
        test_compute_metrics_perfect,
        test_figures_dir,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*40}")
    sys.exit(1 if failed else 0)
