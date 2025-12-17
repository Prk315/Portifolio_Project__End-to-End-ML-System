"""Evaluation metrics and utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from typing import Dict, List, Any
import pandas as pd


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_proba: np.ndarray = None,
                     labels: List[str] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        labels: Label names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Weighted average
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    metrics['precision_weighted'] = precision_w
    metrics['recall_weighted'] = recall_w
    metrics['f1_weighted'] = f1_w

    # Macro average
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    metrics['precision_macro'] = precision_m
    metrics['recall_macro'] = recall_m
    metrics['f1_macro'] = f1_m

    # Per-class details
    metrics['per_class'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist()
    }

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # Classification report
    if labels:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=labels, zero_division=0
        )
    else:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, zero_division=0
        )

    # AUC if probabilities provided
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            pass

    return metrics


def print_metrics(metrics: Dict[str, Any], model_name: str = "Model"):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro):  {metrics['f1_macro']:.4f}")
    print(f"\nPrecision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (weighted):  {metrics['f1_weighted']:.4f}")

    if 'auc_roc' in metrics:
        print(f"\nAUC-ROC:           {metrics['auc_roc']:.4f}")

    print(f"\n{'-'*60}")
    print("Classification Report:")
    print(f"{'-'*60}")
    print(metrics['classification_report'])
    print(f"{'='*60}\n")


def confidence_analysis(y_proba: np.ndarray, y_true: np.ndarray,
                       y_pred: np.ndarray) -> pd.DataFrame:
    """
    Analyze model confidence vs accuracy.

    Args:
        y_proba: Prediction probabilities
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        DataFrame with confidence analysis
    """
    max_probs = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(int)

    # Create bins
    bins = [0.0, 0.5, 0.7, 0.85, 0.95, 1.0]
    labels = ['0.0-0.5', '0.5-0.7', '0.7-0.85', '0.85-0.95', '0.95-1.0']

    df = pd.DataFrame({
        'confidence': max_probs,
        'correct': correct,
        'confidence_bin': pd.cut(max_probs, bins=bins, labels=labels)
    })

    # Group by confidence bins
    analysis = df.groupby('confidence_bin', observed=True).agg({
        'correct': ['mean', 'count']
    }).round(4)

    analysis.columns = ['accuracy', 'count']
    return analysis
