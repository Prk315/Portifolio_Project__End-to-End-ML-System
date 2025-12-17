"""FastAPI main application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import API_CONFIG, MODELS_DIR
from src.api.predictor import TicketPredictor
from src.utils.logger import log

# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG['title'],
    description=API_CONFIG['description'],
    version=API_CONFIG['version']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = None


# Request/Response models
class TicketRequest(BaseModel):
    """Single ticket prediction request."""
    text: str = Field(..., description="Ticket text", min_length=1)
    use_rag: bool = Field(True, description="Use RAG for response suggestion")
    model_type: str = Field("transformer", description="Model type: 'baseline' or 'transformer'")


class BatchTicketRequest(BaseModel):
    """Batch ticket prediction request."""
    texts: List[str] = Field(..., description="List of ticket texts")
    use_rag: bool = Field(True, description="Use RAG for response suggestions")
    model_type: str = Field("transformer", description="Model type: 'baseline' or 'transformer'")


class PredictionResponse(BaseModel):
    """Prediction response."""
    text: str
    predicted_category: str
    confidence: float
    all_probabilities: Dict[str, float]
    routing_decision: str
    suggested_response: Optional[str] = None
    similar_tickets: Optional[List[Dict]] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    rag_enabled: bool


class MetricsResponse(BaseModel):
    """Metrics response."""
    model_type: str
    metrics: Dict


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global predictor

    log.info("Starting up API...")

    try:
        predictor = TicketPredictor(
            baseline_model_path=MODELS_DIR / "baseline_model.pkl",
            transformer_model_path=MODELS_DIR / "transformer_model",
            rag_retriever_path=MODELS_DIR / "rag_retriever",
            confidence_threshold_high=API_CONFIG['confidence_threshold_high'],
            confidence_threshold_low=API_CONFIG['confidence_threshold_low']
        )
        log.info("Models loaded successfully")
    except Exception as e:
        log.error(f"Error loading models: {e}")
        log.warning("API will start but predictions may fail until models are trained")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Support Triage API",
        "version": API_CONFIG['version'],
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    model_loaded = predictor is not None and predictor.model_loaded
    rag_enabled = predictor is not None and predictor.rag_enabled

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "rag_enabled": rag_enabled
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_ticket(request: TicketRequest):
    """
    Predict category and suggest response for a support ticket.

    - **text**: The support ticket text
    - **use_rag**: Whether to use RAG for response suggestion
    - **model_type**: Which model to use ('baseline' or 'transformer')
    """
    if predictor is None or not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train models first."
        )

    try:
        result = predictor.predict(
            request.text,
            use_rag=request.use_rag,
            model_type=request.model_type
        )
        return result
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict_tickets(request: BatchTicketRequest):
    """
    Batch predict categories for multiple support tickets.

    - **texts**: List of support ticket texts
    - **use_rag**: Whether to use RAG for response suggestions
    - **model_type**: Which model to use ('baseline' or 'transformer')
    """
    if predictor is None or not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train models first."
        )

    try:
        predictions = []
        for text in request.texts:
            result = predictor.predict(
                text,
                use_rag=request.use_rag,
                model_type=request.model_type
            )
            predictions.append(result)

        return {"predictions": predictions}
    except Exception as e:
        log.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories", tags=["Information"])
async def get_categories():
    """Get list of supported ticket categories."""
    if predictor is None or not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded."
        )

    return {
        "categories": predictor.get_categories()
    }


@app.get("/metrics", response_model=MetricsResponse, tags=["Information"])
async def get_metrics():
    """Get model performance metrics (if available)."""
    # This would load saved metrics from training
    # For now, return placeholder
    return {
        "model_type": "transformer",
        "metrics": {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "note": "Train model to see actual metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
