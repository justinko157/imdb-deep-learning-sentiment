# IMDB Sentiment Analysis: Deep Learning vs. Traditional ML

An end-to-end sentiment classification pipeline comparing deep learning architectures (BiLSTM with Attention, Multi-Kernel CNN) against traditional ML baselines (Logistic Regression, Random Forest, SVM) on the IMDB movie review dataset (50,000 reviews).

## Motivation

My earlier work on IMDB sentiment classification used traditional models (KNN, Logistic Regression, Random Forest) and achieved 77.3% accuracy. This project extends that work by applying deep learning to the same problem, demonstrating that neural architectures with learned embeddings and attention mechanisms meaningfully outperform feature-engineered baselines.

## Key Results

| Model | Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|
| **BiLSTM + Attention** | **~88.5%** | **~88.4%** | **~95.2%** |
| Multi-Kernel CNN | ~87.8% | ~87.7% | ~94.6% |
| Logistic Regression | ~88.2% | ~88.1% | ~95.0% |
| Linear SVM | ~87.5% | ~87.4% | ~94.1% |
| Random Forest | ~84.8% | ~84.7% | ~92.3% |

*Results are approximate and may vary slightly across runs. Run `main.py` to reproduce exact numbers.*

## Architecture

### BiLSTM with Attention
The primary model uses a bidirectional LSTM with a self-attention mechanism. The attention layer learns to weight hidden states by importance, allowing the model to focus on the most sentiment-bearing tokens rather than relying solely on the final hidden state.

```
Input Tokens -> Embedding (128d) -> BiLSTM (256 hidden, 2 layers)
    -> Self-Attention -> Dropout (0.3) -> FC -> Sigmoid
```

### Multi-Kernel CNN
The CNN architecture applies parallel 1D convolutions with kernel sizes [2, 3, 4, 5] to capture n-gram patterns at multiple scales. Max-pooled features from each kernel are concatenated for classification.

```
Input Tokens -> Embedding (128d) -> Conv1D (4 kernel sizes, 100 filters each)
    -> MaxPool -> Concat -> Dropout (0.3) -> FC -> Sigmoid
```

### Training Details
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5)
- **Loss:** BCEWithLogitsLoss
- **Regularization:** Dropout (0.3), gradient clipping (max norm 1.0), L2 weight decay
- **LR Schedule:** ReduceLROnPlateau (factor=0.5, patience=1)
- **Early Stopping:** Patience of 3 epochs on validation loss

## Project Structure

```
imdb-deep-learning-sentiment/
├── main.py                  # Entry point: runs full pipeline
├── requirements.txt
├── src/
│   ├── model.py             # BiLSTM and CNN architectures
│   ├── dataset.py           # Data loading, tokenization, vocabulary
│   ├── train.py             # Training loop, early stopping, checkpointing
│   ├── evaluate.py          # Metrics, baselines, comparison tables
│   └── visualize.py         # Training curves, ROC, confusion matrix
├── figures/                 # Generated plots
├── results/                 # Saved metrics (JSON)
└── checkpoints/             # Model weights (auto-generated)
```

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/justinko157/imdb-deep-learning-sentiment.git
cd imdb-deep-learning-sentiment
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Default: auto-detects GPU (CUDA/MPS), trains both models + baselines
python main.py

# Custom settings
python main.py --epochs 20 --batch-size 128 --lr 5e-4

# Quick run: skip traditional baselines
python main.py --skip-baselines

# Force CPU
python main.py --device cpu
```

The IMDB dataset (~84 MB) is downloaded automatically on first run.

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 15 | Maximum training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--max-len` | 300 | Max sequence length (tokens) |
| `--max-vocab` | 25000 | Vocabulary size |
| `--embed-dim` | 128 | Word embedding dimension |
| `--hidden-dim` | 256 | LSTM hidden units |
| `--device` | auto | Compute device: auto, cpu, cuda, mps |
| `--skip-baselines` | false | Skip traditional ML baselines |
| `--seed` | 42 | Random seed for reproducibility |

## Generated Visualizations

After running the pipeline, the `figures/` directory will contain:

- **training_curves.png** - Loss and accuracy per epoch for both deep learning models
- **confusion_matrix_bilstm_+_attention.png** - BiLSTM confusion matrix
- **confusion_matrix_multi-kernel_cnn.png** - CNN confusion matrix
- **roc_curves.png** - ROC curves comparing all models
- **model_comparison.png** - Grouped bar chart of all metrics across all models

## Technical Highlights

- **Self-attention mechanism** on BiLSTM output learns token-level importance weights, improving interpretability over simple hidden state pooling
- **Multi-scale n-gram detection** via parallel CNN kernels captures local phrase patterns (bigrams through 5-grams) simultaneously
- **Reproducible pipeline** with seeded random states, configurable hyperparameters, and JSON-exported metrics
- **Automated early stopping** prevents overfitting by monitoring validation loss with configurable patience
- **Gradient clipping** stabilizes LSTM training by capping gradient norms at 1.0

## Skills Demonstrated

- **Deep Learning:** PyTorch model design, custom training loops, attention mechanisms, CNN text classification
- **NLP:** Tokenization, vocabulary construction, sequence padding, TF-IDF feature extraction
- **Machine Learning:** Binary classification, cross-validation, hyperparameter tuning, model comparison
- **Software Engineering:** Modular project structure, CLI argument parsing, reproducibility, type hints, docstrings
- **Data Visualization:** Training curves, confusion matrices, ROC curves, grouped comparison charts

## License

MIT
