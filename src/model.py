"""
Deep learning model architectures for sentiment classification.

Implements LSTM and CNN-based classifiers with embedding layers,
dropout regularization, and configurable hyperparameters.
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM with attention for binary sentiment classification.

    Architecture:
        Embedding -> Bidirectional LSTM -> Self-Attention -> FC -> Sigmoid

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Dimensionality of word embeddings.
        hidden_dim: Number of LSTM hidden units.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between layers.
        pad_idx: Index of the padding token in the vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)

        # Self-attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.fc(self.dropout(context)).squeeze(1)


class CNNClassifier(nn.Module):
    """Multi-kernel CNN for binary sentiment classification.

    Architecture:
        Embedding -> Parallel Conv1D (multiple kernel sizes) -> MaxPool -> FC -> Sigmoid

    Captures n-gram patterns at different scales (e.g., bigrams, trigrams,
    4-grams) and concatenates the features for classification.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Dimensionality of word embeddings.
        num_filters: Number of convolutional filters per kernel size.
        kernel_sizes: List of kernel sizes for parallel convolutions.
        dropout: Dropout probability.
        pad_idx: Index of the padding token.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 100,
        kernel_sizes: list = None,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x)).permute(0, 2, 1)

        conv_outputs = [
            torch.relu(conv(embedded)).max(dim=2)[0] for conv in self.convs
        ]
        concatenated = torch.cat(conv_outputs, dim=1)

        return self.fc(self.dropout(concatenated)).squeeze(1)
