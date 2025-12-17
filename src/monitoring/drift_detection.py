import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from pathlib import Path
import json

class DriftDetector:
    """
    Monitors data drift using Evidently.
    Compares production data against reference data.
    """

    def __init__(self, reference_data_path=None):
        self.reference_data = None
        if reference_data_path:
            self.load_reference_data(reference_data_path)

    def load_reference_data(self, path):
        """
        Loads reference dataset (e.g., training data).
        """
        self.reference_data = pd.read_csv(path)
        print(f"Loaded reference data: {len(self.reference_data)} samples")

    def set_reference_data(self, df):
        """
        Sets reference data from DataFrame.
        """
        self.reference_data = df.copy()
        print(f"Set reference data: {len(self.reference_data)} samples")

    def detect_drift(self, current_data, save_report=True, report_path='reports/drift_report.html'):
        """
        Detects data drift between reference and current data.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded. Call load_reference_data() first.")

        print("Analyzing data drift...")

        data_drift_report = Report(metrics=[
            DataDriftPreset(),
        ])

        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        if save_report:
            report_dir = Path(report_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)
            data_drift_report.save_html(report_path)
            print(f"Drift report saved to: {report_path}")

        drift_results = data_drift_report.as_dict()

        return self._extract_drift_summary(drift_results)

    def run_drift_tests(self, current_data, save_results=True, results_path='reports/drift_tests.json'):
        """
        Runs drift detection tests.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded.")

        print("Running drift tests...")

        test_suite = TestSuite(tests=[
            DataDriftTestPreset(),
        ])

        test_suite.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        if save_results:
            results_dir = Path(results_path).parent
            results_dir.mkdir(parents=True, exist_ok=True)

            results_dict = test_suite.as_dict()
            with open(results_path, 'w') as f:
                json.dump(results_dict, f, indent=2)

            print(f"Test results saved to: {results_path}")

        return test_suite.as_dict()

    def _extract_drift_summary(self, drift_results):
        """
        Extracts key drift metrics from Evidently report.
        """
        summary = {
            'drift_detected': False,
            'drifted_features': [],
            'drift_share': 0.0,
            'total_features': 0
        }

        try:
            metrics = drift_results.get('metrics', [])

            for metric in metrics:
                metric_result = metric.get('result', {})

                if 'dataset_drift' in metric_result:
                    summary['drift_detected'] = metric_result['dataset_drift']

                if 'drift_by_columns' in metric_result:
                    drift_by_col = metric_result['drift_by_columns']
                    summary['drifted_features'] = [
                        col for col, drifted in drift_by_col.items() if drifted
                    ]
                    summary['total_features'] = len(drift_by_col)
                    if summary['total_features'] > 0:
                        summary['drift_share'] = len(summary['drifted_features']) / summary['total_features']

        except Exception as e:
            print(f"Warning: Could not extract drift summary: {e}")

        return summary

    def monitor_predictions(self, reference_predictions, current_predictions,
                          save_report=True, report_path='reports/target_drift_report.html'):
        """
        Monitors drift in model predictions.
        """
        ref_df = pd.DataFrame({'prediction': reference_predictions})
        curr_df = pd.DataFrame({'prediction': current_predictions})

        print("Analyzing prediction drift...")

        target_drift_report = Report(metrics=[
            TargetDriftPreset(),
        ])

        target_drift_report.run(
            reference_data=ref_df,
            current_data=curr_df
        )

        if save_report:
            report_dir = Path(report_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)
            target_drift_report.save_html(report_path)
            print(f"Target drift report saved to: {report_path}")

        return target_drift_report.as_dict()

class PerformanceMonitor:
    """
    Monitors model performance over time.
    """

    def __init__(self, log_file='logs/performance_log.csv'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_file.exists():
            self._initialize_log()

    def _initialize_log(self):
        """
        Creates performance log file.
        """
        df = pd.DataFrame(columns=[
            'timestamp', 'num_predictions', 'mean_probability',
            'prediction_rate', 'avg_confidence'
        ])
        df.to_csv(self.log_file, index=False)

    def log_predictions(self, predictions, probabilities):
        """
        Logs prediction statistics.
        """
        timestamp = pd.Timestamp.now()
        num_predictions = len(predictions)
        mean_probability = float(np.mean(probabilities))
        prediction_rate = float(np.mean(predictions))
        avg_confidence = float(np.mean(np.max(probabilities, axis=1)) if len(probabilities.shape) > 1 else np.mean(probabilities))

        log_entry = {
            'timestamp': timestamp,
            'num_predictions': num_predictions,
            'mean_probability': mean_probability,
            'prediction_rate': prediction_rate,
            'avg_confidence': avg_confidence
        }

        df_log = pd.read_csv(self.log_file)
        df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
        df_log.to_csv(self.log_file, index=False)

    def get_performance_stats(self, days=30):
        """
        Retrieves performance statistics for the last N days.
        """
        df_log = pd.read_csv(self.log_file)
        df_log['timestamp'] = pd.to_datetime(df_log['timestamp'])

        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        recent_data = df_log[df_log['timestamp'] >= cutoff_date]

        if len(recent_data) == 0:
            return None

        stats = {
            'total_predictions': int(recent_data['num_predictions'].sum()),
            'avg_prediction_rate': float(recent_data['prediction_rate'].mean()),
            'avg_confidence': float(recent_data['avg_confidence'].mean()),
            'days_monitored': days
        }

        return stats

if __name__ == "__main__":
    print("Monitoring module loaded")
