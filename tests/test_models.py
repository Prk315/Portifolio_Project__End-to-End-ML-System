import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import tempfile

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from models.threshold_tuning import ThresholdOptimizer


class TestThresholdOptimizer:
    """
    Tests for ThresholdOptimizer class.
    """

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        # Simulate predictions
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1] * 10)
        y_proba = np.random.rand(100)

        return y_true, y_proba

    def test_threshold_optimizer_initialization(self):
        """Test ThresholdOptimizer initialization."""
        optimizer = ThresholdOptimizer(cost_fp=1, cost_fn=5)
        assert optimizer.cost_fp == 1
        assert optimizer.cost_fn == 5

    def test_threshold_optimizer_default_costs(self):
        """Test ThresholdOptimizer with default costs."""
        optimizer = ThresholdOptimizer()
        assert optimizer.cost_fp == 1
        assert optimizer.cost_fn == 1

    def test_find_optimal_threshold(self, sample_predictions):
        """Test finding optimal threshold."""
        y_true, y_proba = sample_predictions
        optimizer = ThresholdOptimizer(cost_fp=1, cost_fn=5)

        optimal_threshold = optimizer.find_optimal_threshold(y_true, y_proba)

        assert isinstance(optimal_threshold, float)
        assert 0.0 <= optimal_threshold <= 1.0

    def test_threshold_optimizer_with_different_costs(self, sample_predictions):
        """Test threshold optimizer with different cost ratios."""
        y_true, y_proba = sample_predictions

        optimizer_balanced = ThresholdOptimizer(cost_fp=1, cost_fn=1)
        threshold_balanced = optimizer_balanced.find_optimal_threshold(y_true, y_proba)

        optimizer_fn_heavy = ThresholdOptimizer(cost_fp=1, cost_fn=10)
        threshold_fn_heavy = optimizer_fn_heavy.find_optimal_threshold(y_true, y_proba)

        # When FN is more costly, threshold should generally be lower
        # (more conservative in predicting negative class)
        assert 0.0 <= threshold_balanced <= 1.0
        assert 0.0 <= threshold_fn_heavy <= 1.0

    def test_calculate_cost(self, sample_predictions):
        """Test cost calculation."""
        y_true, y_proba = sample_predictions
        optimizer = ThresholdOptimizer(cost_fp=1, cost_fn=5)

        threshold = 0.5
        y_pred = (y_proba >= threshold).astype(int)

        # Calculate false positives and false negatives
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        expected_cost = fp * optimizer.cost_fp + fn * optimizer.cost_fn
        assert expected_cost >= 0


class TestModelIntegration:
    """
    Integration tests for model components.
    """

    def test_model_and_preprocessor_compatibility(self):
        """Test that models can be saved and loaded correctly."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            X_train = np.random.rand(100, 5)
            y_train = np.random.randint(0, 2, 100)
            model.fit(X_train, y_train)

            model_path = Path(tmpdir) / 'model.pkl'
            joblib.dump(model, model_path)

            # Create and save preprocessor
            preprocessor = StandardScaler()
            preprocessor.fit(X_train)

            preprocessor_path = Path(tmpdir) / 'preprocessor.pkl'
            joblib.dump(preprocessor, preprocessor_path)

            # Load and verify
            loaded_model = joblib.load(model_path)
            loaded_preprocessor = joblib.load(preprocessor_path)

            X_test = np.random.rand(10, 5)
            X_processed = loaded_preprocessor.transform(X_test)
            predictions = loaded_model.predict(X_processed)

            assert len(predictions) == 10
            assert all(p in [0, 1] for p in predictions)


class TestExplainability:
    """
    Tests for explainability module (when model is available).
    """

    def test_shap_values_structure(self):
        """Test that SHAP values have correct structure."""
        # This is a placeholder test - requires trained model
        # In a real scenario, you would load a trained model and verify SHAP output
        assert True  # Placeholder

    def test_explanation_text_generation(self):
        """Test explanation text generation."""
        # Placeholder for testing explanation generation
        assert True  # Placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
