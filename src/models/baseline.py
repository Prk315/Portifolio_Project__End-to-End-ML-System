"""Baseline model: TF-IDF + Logistic Regression."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import BASELINE_CONFIG, MODELS_DIR
from src.utils.logger import log
from src.utils.metrics import calculate_metrics, print_metrics
from src.preprocessing.text_processor import create_label_mapping


class BaselineModel:
    """TF-IDF + Logistic Regression baseline model."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize baseline model.

        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for TF-IDF
            max_iter: Maximum iterations for LogisticRegression
            random_state: Random seed
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.random_state = random_state

        # Initialize pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )),
            ('classifier', LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                class_weight='balanced',  # Handle imbalanced classes
                n_jobs=-1
            ))
        ])

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
        Train the baseline model.

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        log.info("Training baseline model...")
        log.info(f"Training samples: {len(X_train)}")

        # Create label mappings
        self.label2id, self.id2label = create_label_mapping(y_train)
        self.classes = list(self.label2id.keys())

        # Convert labels to numeric
        y_train_numeric = y_train.map(self.label2id)

        # Train pipeline
        self.pipeline.fit(X_train, y_train_numeric)

        log.info("Training completed!")

        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train)
        print_metrics(train_metrics, "Training Set")

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print_metrics(val_metrics, "Validation Set")

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict labels for texts.

        Args:
            texts: Input texts

        Returns:
            Predicted labels
        """
        predictions_numeric = self.pipeline.predict(texts)
        predictions = [self.id2label[p] for p in predictions_numeric]
        return np.array(predictions)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for texts.

        Args:
            texts: Input texts

        Returns:
            Prediction probabilities
        """
        return self.pipeline.predict_proba(texts)

    def evaluate(self, X_test: List[str], y_test: pd.Series) -> Dict:
        """
        Evaluate model on test set.

        Args:
            X_test: Test texts
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Convert labels to numeric for metrics
        y_test_numeric = y_test.map(self.label2id).values
        y_pred_numeric = np.array([self.label2id[p] for p in y_pred])

        metrics = calculate_metrics(
            y_test_numeric,
            y_pred_numeric,
            y_proba,
            self.classes
        )

        return metrics

    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / "baseline_model.pkl"
        else:
            path = Path(path)

        # Save pipeline and mappings
        model_data = {
            'pipeline': self.pipeline,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'classes': self.classes
        }

        joblib.dump(model_data, path)
        log.info(f"Model saved to {path}")

    def load(self, path: str = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / "baseline_model.pkl"
        else:
            path = Path(path)

        model_data = joblib.load(path)
        self.pipeline = model_data['pipeline']
        self.label2id = model_data['label2id']
        self.id2label = model_data['id2label']
        self.classes = model_data['classes']

        log.info(f"Model loaded from {path}")

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top features for each class.

        Args:
            top_n: Number of top features per class

        Returns:
            DataFrame with feature importances
        """
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        coefficients = self.pipeline.named_steps['classifier'].coef_

        results = []
        for idx, class_name in enumerate(self.classes):
            # Get coefficients for this class
            class_coef = coefficients[idx]

            # Get top positive features
            top_positive_idx = np.argsort(class_coef)[-top_n:][::-1]
            top_negative_idx = np.argsort(class_coef)[:top_n]

            for i in top_positive_idx:
                results.append({
                    'class': class_name,
                    'feature': feature_names[i],
                    'coefficient': class_coef[i],
                    'type': 'positive'
                })

            for i in top_negative_idx:
                results.append({
                    'class': class_name,
                    'feature': feature_names[i],
                    'coefficient': class_coef[i],
                    'type': 'negative'
                })

        return pd.DataFrame(results)


def train_baseline_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label',
    save_path: str = None
) -> BaselineModel:
    """
    Train and evaluate baseline model.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        text_column: Name of text column
        label_column: Name of label column
        save_path: Path to save model

    Returns:
        Trained model
    """
    # Initialize model
    model = BaselineModel(**BASELINE_CONFIG)

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
    print_metrics(test_metrics, "Test Set")

    # Save model
    if save_path:
        model.save(save_path)
    else:
        model.save()

    return model
