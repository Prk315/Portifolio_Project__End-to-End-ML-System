"""Transformer model: DistilBERT for text classification."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import TRANSFORMER_CONFIG, MODELS_DIR
from src.utils.logger import log
from src.utils.metrics import calculate_metrics, print_metrics
from src.preprocessing.text_processor import create_label_mapping


class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: DistilBertTokenizer,
        max_length: int = 256
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TransformerModel:
    """DistilBERT-based text classification model."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        device: str = None
    ):
        """
        Initialize transformer model.

        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Warmup steps for scheduler
            weight_decay: Weight decay for optimizer
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        log.info(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        self.model = None
        self.label2id = None
        self.id2label = None
        self.classes = None

    def train(
        self,
        X_train: List[str],
        y_train: pd.Series,
        X_val: List[str] = None,
        y_val: pd.Series = None
    ):
        """
        Train the transformer model.

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
        """
        log.info("Training transformer model...")
        log.info(f"Training samples: {len(X_train)}")

        # Create label mappings
        self.label2id, self.id2label = create_label_mapping(y_train)
        self.classes = list(self.label2id.keys())
        num_labels = len(self.classes)

        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        ).to(self.device)

        # Convert labels to numeric
        y_train_numeric = y_train.map(self.label2id).tolist()

        # Create datasets
        train_dataset = TextDataset(
            X_train, y_train_numeric, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Create validation dataset if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            y_val_numeric = y_val.map(self.label2id).tolist()
            val_dataset = TextDataset(
                X_val, y_val_numeric, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        for epoch in range(self.num_epochs):
            log.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            progress_bar = tqdm(train_loader, desc=f"Training")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Track metrics
                train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': train_correct / train_total
                })

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            log.info(f"Training Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")

            # Validation
            if val_loader is not None:
                val_metrics = self._evaluate_loader(val_loader)
                log.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
                log.info(f"Validation F1 (macro): {val_metrics['f1_macro']:.4f}")

        log.info("Training completed!")

    def _evaluate_loader(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on a data loader."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            self.classes
        )

        return metrics

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        self.model.eval()

        predictions = []
        batch_size = self.batch_size

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                encodings = self.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_preds)

        # Convert to labels
        predictions_labels = [self.id2label[p] for p in predictions]
        return np.array(predictions_labels)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities for texts."""
        self.model.eval()

        probabilities = []
        batch_size = self.batch_size

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                encodings = self.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.extend(probs)

        return np.array(probabilities)

    def evaluate(self, X_test: List[str], y_test: pd.Series) -> Dict:
        """Evaluate model on test set."""
        y_test_numeric = y_test.map(self.label2id).tolist()

        test_dataset = TextDataset(
            X_test, y_test_numeric, self.tokenizer, self.max_length
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        metrics = self._evaluate_loader(test_loader)
        return metrics

    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / "transformer_model"
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save label mappings
        import json
        mappings = {
            'label2id': self.label2id,
            'id2label': {int(k): v for k, v in self.id2label.items()},
            'classes': self.classes
        }
        with open(path / 'label_mappings.json', 'w') as f:
            json.dump(mappings, f, indent=2)

        log.info(f"Model saved to {path}")

    def load(self, path: str = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / "transformer_model"
        else:
            path = Path(path)

        # Load model and tokenizer
        self.model = DistilBertForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)

        # Load label mappings
        import json
        with open(path / 'label_mappings.json', 'r') as f:
            mappings = json.load(f)

        self.label2id = mappings['label2id']
        self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
        self.classes = mappings['classes']

        log.info(f"Model loaded from {path}")


def train_transformer_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label',
    save_path: str = None
) -> TransformerModel:
    """Train and evaluate transformer model."""
    # Initialize model
    model = TransformerModel(**TRANSFORMER_CONFIG)

    # Train
    model.train(
        train_df[text_column].tolist(),
        train_df[label_column],
        val_df[text_column].tolist(),
        val_df[label_column]
    )

    # Evaluate on test set
    log.info("\nEvaluating on test set...")
    test_metrics = model.evaluate(
        test_df[text_column].tolist(),
        test_df[label_column]
    )
    print_metrics(test_metrics, "Test Set - Transformer")

    # Save model
    if save_path:
        model.save(save_path)
    else:
        model.save()

    return model
