"""Error analysis utilities."""

import numpy as np
import pandas as pd
from typing import List, Dict
from collections import defaultdict


def analyze_errors(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: List[str]
) -> pd.DataFrame:
    """
    Analyze prediction errors.

    Args:
        texts: Input texts
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        labels: Label names

    Returns:
        DataFrame with error analysis
    """
    errors = []

    for i, (text, true_label, pred_label, proba) in enumerate(
        zip(texts, y_true, y_pred, y_proba)
    ):
        if true_label != pred_label:
            confidence = np.max(proba)
            true_prob = proba[true_label]
            pred_prob = proba[pred_label]

            errors.append({
                'index': i,
                'text': text,
                'true_label': labels[true_label],
                'predicted_label': labels[pred_label],
                'true_probability': true_prob,
                'predicted_probability': pred_prob,
                'confidence': confidence,
                'error_margin': pred_prob - true_prob
            })

    return pd.DataFrame(errors)


def get_confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get most common confusion pairs.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        top_n: Number of top pairs

    Returns:
        DataFrame with confusion pairs
    """
    pairs = defaultdict(int)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            pair = (labels[true_label], labels[pred_label])
            pairs[pair] += 1

    # Sort by frequency
    sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return pd.DataFrame(
        sorted_pairs,
        columns=['pair', 'count']
    )


def analyze_by_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    bins: List[float] = None
) -> pd.DataFrame:
    """
    Analyze errors by confidence level.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        bins: Confidence bins

    Returns:
        DataFrame with confidence analysis
    """
    if bins is None:
        bins = [0.0, 0.5, 0.7, 0.85, 0.95, 1.0]

    max_probs = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(int)

    df = pd.DataFrame({
        'confidence': max_probs,
        'correct': correct
    })

    df['bin'] = pd.cut(df['confidence'], bins=bins)

    analysis = df.groupby('bin', observed=True).agg({
        'correct': ['mean', 'sum', 'count']
    }).round(4)

    analysis.columns = ['accuracy', 'correct_count', 'total_count']
    analysis['error_count'] = analysis['total_count'] - analysis['correct_count']

    return analysis


def get_hardest_examples(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: List[str],
    n: int = 10
) -> pd.DataFrame:
    """
    Get hardest examples (lowest confidence on correct predictions
    or highest confidence on errors).

    Args:
        texts: Input texts
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        labels: Label names
        n: Number of examples

    Returns:
        DataFrame with hardest examples
    """
    examples = []

    for i, (text, true_label, pred_label, proba) in enumerate(
        zip(texts, y_true, y_pred, y_proba)
    ):
        confidence = np.max(proba)
        is_correct = (true_label == pred_label)

        # For errors, higher confidence = harder (confident but wrong)
        # For correct, lower confidence = harder (unsure but right)
        if is_correct:
            difficulty = 1 - confidence  # Lower confidence = harder
        else:
            difficulty = confidence  # Higher confidence = harder

        examples.append({
            'text': text,
            'true_label': labels[true_label],
            'predicted_label': labels[pred_label],
            'confidence': confidence,
            'is_correct': is_correct,
            'difficulty': difficulty
        })

    df = pd.DataFrame(examples)
    return df.nlargest(n, 'difficulty')


def analyze_per_class_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str]
) -> pd.DataFrame:
    """
    Analyze errors per class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names

    Returns:
        DataFrame with per-class error analysis
    """
    results = []

    for i, label in enumerate(labels):
        # Get samples for this class
        mask = (y_true == i)
        if mask.sum() == 0:
            continue

        true_positives = ((y_true == i) & (y_pred == i)).sum()
        false_positives = ((y_true != i) & (y_pred == i)).sum()
        false_negatives = ((y_true == i) & (y_pred != i)).sum()
        true_negatives = ((y_true != i) & (y_pred != i)).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'class': label,
            'samples': mask.sum(),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'error_rate': false_negatives / mask.sum() if mask.sum() > 0 else 0
        })

    return pd.DataFrame(results)
