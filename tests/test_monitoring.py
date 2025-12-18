import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from monitoring.drift_detection import DriftDetector, PerformanceMonitor


class TestDriftDetector:
    """
    Tests for DriftDetector class.
    """

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.randint(0, 10, 100)
        })

        # Similar distribution (no drift)
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50),
            'feature3': np.random.randint(0, 10, 50)
        })

        # Drifted distribution
        drifted_data = pd.DataFrame({
            'feature1': np.random.normal(5, 2, 50),  # Different mean and std
            'feature2': np.random.normal(0, 1, 50),
            'feature3': np.random.randint(0, 10, 50)
        })

        return reference_data, current_data, drifted_data

    def test_drift_detector_initialization(self):
        """Test DriftDetector initialization."""
        detector = DriftDetector()
        assert detector.reference_data is None

    def test_set_reference_data(self, sample_data):
        """Test setting reference data."""
        reference_data, _, _ = sample_data
        detector = DriftDetector()
        detector.set_reference_data(reference_data)

        assert detector.reference_data is not None
        assert len(detector.reference_data) == len(reference_data)

    def test_detect_drift_without_reference(self, sample_data):
        """Test drift detection without reference data."""
        _, current_data, _ = sample_data
        detector = DriftDetector()

        with pytest.raises(ValueError, match="Reference data not loaded"):
            detector.detect_drift(current_data, save_report=False)

    def test_detect_drift_with_reference(self, sample_data):
        """Test drift detection with reference data."""
        reference_data, current_data, _ = sample_data
        detector = DriftDetector()
        detector.set_reference_data(reference_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'drift_report.html'
            drift_summary = detector.detect_drift(
                current_data,
                save_report=True,
                report_path=str(report_path)
            )

            assert isinstance(drift_summary, dict)
            assert 'drift_detected' in drift_summary
            assert 'drifted_features' in drift_summary
            assert 'drift_share' in drift_summary
            assert 'total_features' in drift_summary
            assert report_path.exists()

    def test_drift_summary_structure(self, sample_data):
        """Test structure of drift summary."""
        reference_data, current_data, _ = sample_data
        detector = DriftDetector()
        detector.set_reference_data(reference_data)

        drift_summary = detector.detect_drift(current_data, save_report=False)

        assert isinstance(drift_summary['drift_detected'], bool)
        assert isinstance(drift_summary['drifted_features'], list)
        assert isinstance(drift_summary['drift_share'], float)
        assert isinstance(drift_summary['total_features'], int)
        assert drift_summary['drift_share'] >= 0.0
        assert drift_summary['drift_share'] <= 1.0


class TestPerformanceMonitor:
    """
    Tests for PerformanceMonitor class.
    """

    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'performance_log.csv'
            monitor = PerformanceMonitor(log_file=str(log_file))

            assert log_file.exists()
            df = pd.read_csv(log_file)
            assert 'timestamp' in df.columns
            assert 'num_predictions' in df.columns

    def test_log_predictions(self):
        """Test logging predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'performance_log.csv'
            monitor = PerformanceMonitor(log_file=str(log_file))

            predictions = np.array([0, 1, 0, 1, 0])
            probabilities = np.array([
                [0.8, 0.2],
                [0.3, 0.7],
                [0.9, 0.1],
                [0.4, 0.6],
                [0.7, 0.3]
            ])

            monitor.log_predictions(predictions, probabilities)

            df = pd.read_csv(log_file)
            assert len(df) == 1
            assert df['num_predictions'].iloc[0] == 5

    def test_get_performance_stats_no_data(self):
        """Test getting performance stats with no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'performance_log.csv'
            monitor = PerformanceMonitor(log_file=str(log_file))

            stats = monitor.get_performance_stats(days=30)
            assert stats is None

    def test_get_performance_stats_with_data(self):
        """Test getting performance stats with data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'performance_log.csv'
            monitor = PerformanceMonitor(log_file=str(log_file))

            # Log some predictions
            predictions = np.array([0, 1, 0])
            probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
            monitor.log_predictions(predictions, probabilities)

            stats = monitor.get_performance_stats(days=30)

            assert stats is not None
            assert 'total_predictions' in stats
            assert 'avg_prediction_rate' in stats
            assert 'avg_confidence' in stats
            assert stats['total_predictions'] == 3

    def test_multiple_log_entries(self):
        """Test multiple log entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'performance_log.csv'
            monitor = PerformanceMonitor(log_file=str(log_file))

            # Log first batch
            monitor.log_predictions(
                np.array([0, 1]),
                np.array([[0.8, 0.2], [0.3, 0.7]])
            )

            # Log second batch
            monitor.log_predictions(
                np.array([1, 0, 1]),
                np.array([[0.2, 0.8], [0.9, 0.1], [0.4, 0.6]])
            )

            df = pd.read_csv(log_file)
            assert len(df) == 2
            assert df['num_predictions'].sum() == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
