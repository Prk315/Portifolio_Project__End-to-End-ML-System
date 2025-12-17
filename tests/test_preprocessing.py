import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from data.preprocessing import LoanDataPreprocessor, split_data

def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = LoanDataPreprocessor(target_col='loan_status')
    assert preprocessor.target_col == 'loan_status'
    assert preprocessor.feature_names is None

def test_leakage_detection():
    """Test leakage feature detection."""
    preprocessor = LoanDataPreprocessor()

    test_df = pd.DataFrame({
        'loan_amnt': [10000, 15000],
        'total_pymnt': [5000, 7000],
        'recoveries': [100, 200],
        'annual_inc': [50000, 60000]
    })

    leakage_features = preprocessor.detect_leakage_features(test_df)

    assert 'total_pymnt' in leakage_features
    assert 'recoveries' in leakage_features
    assert 'loan_amnt' not in leakage_features
    assert 'annual_inc' not in leakage_features

def test_feature_engineering():
    """Test feature engineering."""
    preprocessor = LoanDataPreprocessor()

    test_df = pd.DataFrame({
        'loan_amnt': [10000, 15000],
        'annual_inc': [50000, 60000],
        'int_rate': ['10.5%', '12.0%'],
        'term': ['36 months', '60 months']
    })

    result = preprocessor.engineer_features(test_df)

    assert 'loan_to_income' in result.columns
    assert 'term_months' in result.columns
    assert result['int_rate'].dtype == float

def test_data_splitting():
    """Test train/val/test split."""
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.1, random_state=42
    )

    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)

def test_fit_transform():
    """Test preprocessor fit_transform."""
    preprocessor = LoanDataPreprocessor(target_col='target')

    test_df = pd.DataFrame({
        'numeric_col': [1, 2, 3, np.nan],
        'category_col': ['A', 'B', 'A', 'C'],
        'target': [0, 1, 0, 1]
    })

    X, y = preprocessor.fit_transform(test_df)

    assert len(X) == len(test_df)
    assert len(y) == len(test_df)
    assert preprocessor.feature_names is not None
    assert X['numeric_col'].isna().sum() == 0

if __name__ == "__main__":
    pytest.main([__file__])
