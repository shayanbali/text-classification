# Multiclass Text Classification with DistilBERT

## Overview

This project implements a multiclass text classification system using a fine-tuned DistilBERT transformer model. The model is trained on labeled text data and evaluated using standard classification metrics such as accuracy and F1-score.

The notebook provides an end-to-end Natural Language Processing (NLP) pipeline, from data preparation to model evaluation and saving.

---

## Features

- End-to-end transformer-based text classification pipeline
- Train / validation / test split with stratification
- Fine-tuning of a pre-trained DistilBERT model
- Evaluation using multiple metrics
- Confusion matrix visualization
- Model saving for later inference

---

## Dataset

The expected dataset contains two main columns:

- **text** — the input sentence or document
- **label** — integer class identifier 

Expected dataset file:



Example format:

| text | label |
|------|--------|
| "example sentence" | 3 |

---

## Model

- Base model: `distilbert-base-uncased`
- Task: Sequence classification
- Framework: PyTorch + Hugging Face Transformers

---

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy torch transformers scikit-learn matplotlib seaborn
```

---

## Usage

### 1. Load Dataset

Make sure the dataset file path is correct:

```python
df = pd.read_csv("dataset_path")
```

---

### 2. Data Splitting

The dataset is split into:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

Stratified sampling is used to preserve class distribution.

---

### 3. Tokenization

Text is tokenized using the DistilBERT tokenizer:

```python
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
```

---

### 4. Training

Training is performed using Hugging Face's `Trainer` API with:

- Train batch size: 16
- Evaluation batch size: 32
- Epochs: 10
- Weight decay: 0.01
- Early stopping (patience = 3)
- Best model selected by F1-score

Start training:

```python
trainer.train()
```

---

### 5. Evaluation

Model performance is evaluated using:

- Accuracy
- Weighted F1-score
- Classification report
- Confusion matrix

Evaluate on the test set:

```python
trainer.predict(test_ds)
```

---

## Visualization

A confusion matrix is generated using Seaborn to analyze performance across classes.

---

## Model Saving

The best model is saved to:

```
./final_intent_model
```

This directory contains:

- Model weights
- Configuration files
- Tokenizer

---

## Reproducibility

Random seeds are fixed to ensure reproducible results:

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```

---

## Future Improvements

Possible enhancements include:

- Hyperparameter tuning
- Larger training datasets
- Data augmentation
- Alternative transformer architectures
- Ensemble methods
- Detailed error analysis
- Deployment as an API or application

---

## References

- Hugging Face Transformers Library
- DistilBERT: *DistilBERT, a distilled version of BERT*
- PyTorch Documentation

---

## Notes

This project provides a baseline implementation for multiclass text classification using transformer models and can be extended for research or production use.
