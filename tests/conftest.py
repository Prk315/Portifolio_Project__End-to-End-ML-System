import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def sample_loan_data():
    """
    Generate sample loan data for testing.
    """
    np.random.seed(42)
    n_samples = 100

    data = {
        'loan_amnt': np.random.uniform(1000, 40000, n_samples),
        'term': np.random.choice(['36 months', '60 months'], n_samples),
        'int_rate': np.random.uniform(5, 25, n_samples),
        'installment': np.random.uniform(50, 1500, n_samples),
        'grade': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
        'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '5 years', '10+ years'], n_samples),
        'annual_inc': np.random.uniform(20000, 150000, n_samples),
        'dti': np.random.uniform(0, 40, n_samples),
        'delinq_2yrs': np.random.randint(0, 5, n_samples),
        'fico_range_high': np.random.randint(600, 850, n_samples),
        'revol_bal': np.random.uniform(0, 50000, n_samples),
        'revol_util': np.random.uniform(0, 100, n_samples),
        'loan_status': np.random.choice([0, 1], n_samples)  # Target variable
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for testing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_loan_application():
    """
    Single loan application for API testing.
    """
    return {
        "loan_amnt": 10000,
        "term": "36 months",
        "int_rate": 10.5,
        "installment": 325.0,
        "grade": "B",
        "emp_length": "5 years",
        "annual_inc": 60000,
        "dti": 15.5,
        "delinq_2yrs": 0,
        "fico_range_high": 720,
        "revol_bal": 5000,
        "revol_util": 30.0
    }


@pytest.fixture
def sample_predictions():
    """
    Sample predictions and probabilities for testing.
    """
    np.random.seed(42)
    predictions = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1] * 10)
    probabilities = np.random.rand(100, 2)
    # Normalize probabilities
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    return predictions, probabilities


@pytest.fixture
def mock_model_paths(temp_dir):
    """
    Create mock model and preprocessor paths.
    """
    model_path = temp_dir / 'model.pkl'
    preprocessor_path = temp_dir / 'preprocessor.pkl'

    return {
        'model_path': model_path,
        'preprocessor_path': preprocessor_path,
        'temp_dir': temp_dir
    }
