from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Optional, List, Dict
import sys
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))
from models.explainability import ModelExplainer
from monitoring.drift_detection import DriftDetector, PerformanceMonitor

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Loan Default Prediction API",
    description="REST API for predicting loan defaults with explainability",
    version="1.0.0"
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = None
preprocessor = None
explainer = None
drift_detector = None
performance_monitor = None
model_metadata = {}

@app.on_event("startup")
async def load_model():
    """
    Loads model and preprocessor on startup.
    """
    global model, preprocessor, explainer, drift_detector, performance_monitor, model_metadata

    model_path = Path(config['api']['model_path'])
    preprocessor_path = Path(config['api']['preprocessor_path'])

    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        print("Please train a model first using train_advanced.py")
        return

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    explainer = ModelExplainer(str(model_path), str(preprocessor_path))

    # Initialize monitoring
    performance_monitor = PerformanceMonitor()
    drift_detector = DriftDetector()

    # Store model metadata
    model_metadata = {
        "model_type": type(model).__name__,
        "model_path": str(model_path),
        "loaded_at": datetime.now().isoformat(),
        "version": "1.0.0"
    }

    print("Model, preprocessor, and monitoring tools loaded successfully")

class LoanApplication(BaseModel):
    """
    Schema for loan application data.
    """
    loan_amnt: float = Field(..., description="Loan amount requested")
    term: str = Field(..., description="Loan term (e.g., '36 months', '60 months')")
    int_rate: float = Field(..., description="Interest rate")
    installment: Optional[float] = Field(None, description="Monthly installment")
    grade: Optional[str] = Field(None, description="Loan grade")
    emp_length: Optional[str] = Field(None, description="Employment length")
    annual_inc: float = Field(..., description="Annual income")
    dti: float = Field(..., description="Debt-to-income ratio")
    delinq_2yrs: Optional[int] = Field(0, description="Number of delinquencies in past 2 years")
    fico_range_high: Optional[int] = Field(None, description="Upper FICO score range")
    revol_bal: Optional[float] = Field(None, description="Revolving balance")
    revol_util: Optional[float] = Field(None, description="Revolving line utilization rate")

    class Config:
        schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    """
    prediction: int
    prediction_label: str
    probability_default: float
    probability_no_default: float
    risk_level: str

class ExplainedPredictionResponse(PredictionResponse):
    """
    Extended response with explainability.
    """
    explanation: str
    top_factors: List[dict]

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Loan Default Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Make a prediction",
            "/predict/explain": "Make a prediction with explanation",
            "/health": "Health check",
            "/metrics": "Get performance metrics",
            "/model/info": "Get model information",
            "/drift/detect": "Detect data drift",
            "/drift/set-reference": "Set reference data for drift detection"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    if model is None or preprocessor is None:
        return {
            "status": "unhealthy",
            "message": "Model not loaded. Please train a model first."
        }

    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("100/minute")
async def predict(request: Request, loan: LoanApplication):
    """
    Predicts loan default probability.
    Rate limited to 100 requests per minute.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    try:
        loan_data = loan.dict()
        df = pd.DataFrame([loan_data])

        X = preprocessor.transform(df)

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        prob_no_default = float(probabilities[0])
        prob_default = float(probabilities[1])

        if prob_default < 0.3:
            risk_level = "Low"
        elif prob_default < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Log predictions to performance monitor
        if performance_monitor is not None:
            try:
                performance_monitor.log_predictions(
                    predictions=np.array([prediction]),
                    probabilities=np.array([probabilities])
                )
            except Exception as log_error:
                print(f"Warning: Failed to log prediction: {log_error}")

        return PredictionResponse(
            prediction=int(prediction),
            prediction_label="Default" if prediction == 1 else "No Default",
            probability_default=prob_default,
            probability_no_default=prob_no_default,
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/explain", response_model=ExplainedPredictionResponse)
@limiter.limit("50/minute")
async def predict_with_explanation(request: Request, loan: LoanApplication):
    """
    Predicts loan default with SHAP-based explanation.
    Rate limited to 50 requests per minute (SHAP is computationally expensive).
    """
    if model is None or preprocessor is None or explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    try:
        loan_data = loan.dict()

        explanation_text, feature_contrib = explainer.explain_prediction(loan_data)

        df = pd.DataFrame([loan_data])
        X = preprocessor.transform(df)

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        prob_no_default = float(probabilities[0])
        prob_default = float(probabilities[1])

        if prob_default < 0.3:
            risk_level = "Low"
        elif prob_default < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Log predictions to performance monitor
        if performance_monitor is not None:
            try:
                performance_monitor.log_predictions(
                    predictions=np.array([prediction]),
                    probabilities=np.array([probabilities])
                )
            except Exception as log_error:
                print(f"Warning: Failed to log prediction: {log_error}")

        top_factors = feature_contrib.head(5)[['feature', 'value', 'shap']].to_dict('records')

        return ExplainedPredictionResponse(
            prediction=int(prediction),
            prediction_label="Default" if prediction == 1 else "No Default",
            probability_default=prob_default,
            probability_no_default=prob_no_default,
            risk_level=risk_level,
            explanation=explanation_text,
            top_factors=top_factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics(days: int = 30):
    """
    Get performance metrics for the last N days.
    """
    if performance_monitor is None:
        raise HTTPException(
            status_code=503,
            detail="Performance monitor not initialized"
        )

    try:
        stats = performance_monitor.get_performance_stats(days=days)

        if stats is None:
            return {
                "message": "No metrics available",
                "days_requested": days
            }

        return {
            "metrics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "model_metadata": model_metadata,
        "model_loaded": True,
        "explainability_enabled": explainer is not None,
        "monitoring_enabled": performance_monitor is not None
    }

@app.post("/drift/detect")
async def detect_drift(current_data: List[Dict]):
    """
    Detect data drift between reference and current data.
    """
    if drift_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Drift detector not initialized"
        )

    try:
        # Convert input data to DataFrame
        df_current = pd.DataFrame(current_data)

        # Check if reference data is set
        if drift_detector.reference_data is None:
            # Try to load from default location
            ref_data_path = Path("data/processed/train.csv")
            if ref_data_path.exists():
                drift_detector.load_reference_data(str(ref_data_path))
            else:
                return {
                    "message": "Reference data not set. Please set reference data first.",
                    "status": "error"
                }

        # Detect drift
        drift_summary = drift_detector.detect_drift(
            current_data=df_current,
            save_report=True,
            report_path="reports/drift_report.html"
        )

        return {
            "drift_summary": drift_summary,
            "timestamp": datetime.now().isoformat(),
            "report_path": "reports/drift_report.html"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift/set-reference")
async def set_reference_data(reference_data: List[Dict]):
    """
    Set reference data for drift detection.
    """
    if drift_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Drift detector not initialized"
        )

    try:
        df_reference = pd.DataFrame(reference_data)
        drift_detector.set_reference_data(df_reference)

        return {
            "message": "Reference data set successfully",
            "num_samples": len(df_reference),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port']
    )
