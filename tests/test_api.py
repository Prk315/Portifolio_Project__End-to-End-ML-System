import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from api.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_predict_endpoint_structure():
    """Test prediction endpoint structure."""
    loan_data = {
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

    response = client.post("/predict", json=loan_data)

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability_default" in data
        assert "risk_level" in data
    elif response.status_code == 503:
        pass

def test_predict_validation():
    """Test prediction endpoint validation."""
    invalid_data = {
        "loan_amnt": "invalid",
    }

    response = client.post("/predict", json=invalid_data)
    assert response.status_code in [422, 503]

if __name__ == "__main__":
    pytest.main([__file__])
