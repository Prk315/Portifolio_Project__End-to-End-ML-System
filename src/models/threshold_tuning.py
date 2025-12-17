import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
import seaborn as sns

class ThresholdOptimizer:
    """
    Optimizes classification threshold based on business costs.
    """

    def __init__(self, cost_fp=1, cost_fn=5):
        """
        Initialize with cost parameters.

        Args:
            cost_fp: Cost of false positive (approving bad loan)
            cost_fn: Cost of false negative (rejecting good loan)
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.optimal_threshold = None
        self.threshold_metrics = None

    def find_optimal_threshold(self, y_true, y_proba, metric='cost'):
        """
        Finds optimal threshold based on specified metric.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Optimization metric ('cost', 'f1', 'youden')

        Returns:
            Optimal threshold value
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        metrics = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            total_cost = (fp * self.cost_fp) + (fn * self.cost_fn)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden_index = sensitivity + specificity - 1

            metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cost': total_cost,
                'youden': youden_index,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })

        self.threshold_metrics = pd.DataFrame(metrics)

        if metric == 'cost':
            optimal_idx = self.threshold_metrics['cost'].idxmin()
        elif metric == 'f1':
            optimal_idx = self.threshold_metrics['f1'].idxmax()
        elif metric == 'youden':
            optimal_idx = self.threshold_metrics['youden'].idxmax()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        self.optimal_threshold = self.threshold_metrics.loc[optimal_idx, 'threshold']

        print(f"\nOptimal threshold ({metric}): {self.optimal_threshold:.3f}")
        print("\nMetrics at optimal threshold:")
        optimal_row = self.threshold_metrics.loc[optimal_idx]
        print(f"  Precision: {optimal_row['precision']:.3f}")
        print(f"  Recall: {optimal_row['recall']:.3f}")
        print(f"  F1: {optimal_row['f1']:.3f}")
        print(f"  Total Cost: {optimal_row['cost']:.0f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {optimal_row['tn']:.0f}, FP: {optimal_row['fp']:.0f}")
        print(f"  FN: {optimal_row['fn']:.0f}, TP: {optimal_row['tp']:.0f}")

        return self.optimal_threshold

    def plot_threshold_analysis(self):
        """
        Visualizes threshold optimization results.
        """
        if self.threshold_metrics is None:
            raise ValueError("Run find_optimal_threshold first")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].plot(self.threshold_metrics['threshold'], self.threshold_metrics['f1'], label='F1')
        axes[0, 0].plot(self.threshold_metrics['threshold'], self.threshold_metrics['precision'], label='Precision')
        axes[0, 0].plot(self.threshold_metrics['threshold'], self.threshold_metrics['recall'], label='Recall')
        axes[0, 0].axvline(self.optimal_threshold, color='red', linestyle='--', label='Optimal')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Precision, Recall, and F1 vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.threshold_metrics['threshold'], self.threshold_metrics['cost'])
        axes[0, 1].axvline(self.optimal_threshold, color='red', linestyle='--', label='Optimal')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Total Cost')
        axes[0, 1].set_title(f'Total Cost vs Threshold (FP cost={self.cost_fp}, FN cost={self.cost_fn})')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(self.threshold_metrics['threshold'], self.threshold_metrics['tp'], label='True Positives')
        axes[1, 0].plot(self.threshold_metrics['threshold'], self.threshold_metrics['fp'], label='False Positives')
        axes[1, 0].plot(self.threshold_metrics['threshold'], self.threshold_metrics['tn'], label='True Negatives')
        axes[1, 0].plot(self.threshold_metrics['threshold'], self.threshold_metrics['fn'], label='False Negatives')
        axes[1, 0].axvline(self.optimal_threshold, color='red', linestyle='--', label='Optimal')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Confusion Matrix Components vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.threshold_metrics['threshold'], self.threshold_metrics['youden'])
        axes[1, 1].axvline(self.optimal_threshold, color='red', linestyle='--', label='Optimal')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel("Youden's Index")
        axes[1, 1].set_title("Youden's Index vs Threshold")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def evaluate_at_threshold(self, y_true, y_proba, threshold=None):
        """
        Evaluates model performance at a specific threshold.
        """
        if threshold is None:
            threshold = self.optimal_threshold

        if threshold is None:
            raise ValueError("No threshold specified. Run find_optimal_threshold first.")

        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_cost = (fp * self.cost_fp) + (fn * self.cost_fn)

        results = {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_cost': total_cost,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }

        return results

    def compare_thresholds(self, y_true, y_proba, thresholds=[0.3, 0.5, 0.7]):
        """
        Compares model performance at different thresholds.
        """
        results = []

        for threshold in thresholds:
            result = self.evaluate_at_threshold(y_true, y_proba, threshold)
            results.append(result)

        comparison = pd.DataFrame(results)

        print("\nThreshold Comparison:")
        print(comparison[['threshold', 'precision', 'recall', 'f1', 'total_cost']])

        return comparison

if __name__ == "__main__":
    print("Threshold tuning module loaded")
