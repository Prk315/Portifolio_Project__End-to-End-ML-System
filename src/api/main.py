from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Optional, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.explainability import ModelExplainer

app = FastAPI(
    title="Loan Default Prediction API",
    description="REST API for predicting loan defaults with explainability",
    version="1.0.0"
)

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = None
preprocessor = None
explainer = None

@app.on_event("startup")
async def load_model():
    """
    Loads model and preprocessor on startup.
    """
    global model, preprocessor, explainer

    model_path = Path(config['api']['model_path'])
    preprocessor_path = Path(config['api']['preprocessor_path'])

    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        print("Please train a model first using train_advanced.py")
        return

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    explainer = ModelExplainer(str(model_path), str(preprocessor_path))

    print("Model and preprocessor loaded successfully")

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
            "/health": "Health check"
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
async def predict(loan: LoanApplication):
    """
    Predicts loan default probability.
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
async def predict_with_explanation(loan: LoanApplication):
    """
    Predicts loan default with SHAP-based explanation.
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port']
    )
