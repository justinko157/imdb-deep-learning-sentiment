"""
Data loading, preprocessing, and vocabulary management for IMDB reviews.

Handles downloading the IMDB dataset, tokenization, vocabulary construction,
sequence padding, and PyTorch DataLoader creation.
"""

import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = Path("data")
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ──────────────────────────────────────────────────────────────────────
# Download and extraction
# ──────────────────────────────────────────────────────────────────────
def download_imdb(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract the IMDB dataset if not already present.

    Returns:
        Path to the extracted aclImdb directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_path = data_dir / "aclImdb"

    if extract_path.exists():
        print(f"Dataset already exists at {extract_path}")
        return extract_path

    tar_path = data_dir / "aclImdb_v1.tar.gz"
    print("Downloading IMDB dataset...")
    urllib.request.urlretrieve(IMDB_URL, tar_path)

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir, filter="data")

    tar_path.unlink()
    print(f"Dataset ready at {extract_path}")
    return extract_path


# ──────────────────────────────────────────────────────────────────────
# Text preprocessing
# ──────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> list:
    """Clean and normalize a single review string.

    Removes HTML tags, non-alphabetic characters, and lowercases the text.
    """
    text = re.sub(r"<[^>]+>", " ", text)       # strip HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text)   # keep only letters
    return text.lower().split()


def load_split(path: Path, split: str) -> Tuple[list, list]:
    """Load reviews and labels for a given split (train/test).

    Args:
        path: Root path to aclImdb directory.
        split: Either 'train' or 'test'.

    Returns:
        Tuple of (tokenized_reviews, labels) where each review is a list
        of lowercase tokens and each label is 0 (negative) or 1 (positive).
    """
    texts, labels = [], []
    for label_dir, label_val in [("neg", 0), ("pos", 1)]:
        folder = path / split / label_dir
        for file in sorted(folder.glob("*.txt")):
            texts.append(clean_text(file.read_text(encoding="utf-8")))
            labels.append(label_val)
    return texts, labels


# ──────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────
class Vocabulary:
    """Maps tokens to integer indices with special PAD and UNK tokens.

    Args:
        max_size: Maximum vocabulary size (most frequent tokens kept).
        min_freq: Minimum token frequency to be included.
    """

    def __init__(self, max_size: int = 25000, min_freq: int = 5):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2token = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, tokenized_texts: list) -> "Vocabulary":
        """Build vocabulary from a list of tokenized documents."""
        counter = Counter(tok for doc in tokenized_texts for tok in doc)
        frequent = [
            tok
            for tok, count in counter.most_common(self.max_size)
            if count >= self.min_freq
        ]
        for tok in frequent:
            idx = len(self.token2idx)
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok
        print(f"Vocabulary built: {len(self.token2idx):,} tokens")
        return self

    def encode(self, tokens: list) -> list:
        """Convert a list of tokens to a list of integer indices."""
        unk = self.token2idx[UNK_TOKEN]
        return [self.token2idx.get(t, unk) for t in tokens]

    def __len__(self) -> int:
        return len(self.token2idx)


# ──────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────
class IMDBDataset(Dataset):
    """PyTorch Dataset for IMDB reviews.

    Encodes tokenized reviews using the provided vocabulary and pads/truncates
    sequences to a fixed maximum length.

    Args:
        texts: List of tokenized reviews.
        labels: List of integer labels (0 or 1).
        vocab: Vocabulary instance for encoding.
        max_len: Maximum sequence length.
    """

    def __init__(self, texts: list, labels: list, vocab: Vocabulary, max_len: int = 300):
        self.labels = labels
        self.max_len = max_len
        self.encoded = [vocab.encode(t) for t in texts]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.encoded[idx][: self.max_len]
        padded = seq + [0] * (self.max_len - len(seq))
        return torch.tensor(padded, dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.float
        )


# ──────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────
def get_dataloaders(
    batch_size: int = 64,
    max_len: int = 300,
    max_vocab: int = 25000,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """Prepare train, validation, and test DataLoaders.

    Downloads the IMDB dataset if needed, builds the vocabulary from the
    training set, and splits training data into train/validation sets.

    Args:
        batch_size: Batch size for all DataLoaders.
        max_len: Maximum sequence length after padding/truncation.
        max_vocab: Maximum vocabulary size.
        val_split: Fraction of training data used for validation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocab, raw_splits)
        where raw_splits is (train_texts, train_labels, test_texts, test_labels).
    """
    path = download_imdb()

    print("Loading training data...")
    train_texts, train_labels = load_split(path, "train")
    print("Loading test data...")
    test_texts, test_labels = load_split(path, "test")

    vocab = Vocabulary(max_size=max_vocab).build(train_texts)

    full_train = IMDBDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_len)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train: {train_size:,} | Val: {val_size:,} | Test: {len(test_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, vocab, (train_texts, train_labels, test_texts, test_labels)
