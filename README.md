# Loan Default Prediction System

A production-ready machine learning system for predicting loan defaults with full explainability and monitoring capabilities.

## Overview

This project demonstrates a complete ML pipeline from data exploration to deployment, including:

- Comprehensive exploratory data analysis with leakage detection
- Baseline and advanced modeling approaches
- Model explainability using SHAP values
- Production-ready REST API
- Interactive web interface
- Data drift detection and model monitoring
- Business-driven threshold optimization
- Containerized deployment

## Features

### Core ML Capabilities
- **Baseline Model**: Logistic regression for quick iteration
- **Advanced Model**: XGBoost with hyperparameter tuning
- **Explainability**: SHAP values for transparent predictions
- **Threshold Tuning**: Cost-based optimization for business objectives

### Production Features
- **REST API**: FastAPI endpoint for scalable inference
- **Web UI**: Streamlit interface for interactive predictions
- **Monitoring**: Data drift detection using Evidently
- **Deployment**: Docker containers for easy deployment

## Project Structure

```
.
├── config/                 # Configuration files
│   └── config.yaml
├── data/                   # Data storage
│   ├── raw/               # Raw data files
│   └── processed/         # Processed datasets
├── models/                 # Trained model artifacts
├── notebooks/              # Jupyter notebooks
│   └── 01_EDA_and_Leakage_Detection.ipynb
├── src/                    # Source code
│   ├── data/              # Data processing
│   │   ├── download_data.py
│   │   └── preprocessing.py
│   ├── models/            # Model training
│   │   ├── train_baseline.py
│   │   ├── train_advanced.py
│   │   ├── explainability.py
│   │   └── threshold_tuning.py
│   ├── api/               # API service
│   │   └── main.py
│   └── monitoring/        # Monitoring tools
│       └── drift_detection.py
├── ui/                     # Web interface
│   └── app.py
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── Dockerfile.api          # API container
├── Dockerfile.ui           # UI container
└── docker-compose.yml      # Multi-container setup
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PortefolioProject__End-to-End-ML-System
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Setup

Download the Lending Club dataset:

**Option 1: Kaggle API**
```bash
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d data/raw/
```

**Option 2: Manual Download**
1. Visit https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Download `accepted_2007_to_2018Q4.csv.gz`
3. Extract to `data/raw/`

**Option 3: Credit Card Default (Alternative)**
```bash
python src/data/download_data.py credit
```

### Training Models

1. **Explore the data**:
```bash
jupyter notebook notebooks/01_EDA_and_Leakage_Detection.ipynb
```

2. **Train baseline model**:
```bash
python src/models/train_baseline.py --data data/raw/accepted_2007_to_2018Q4.csv
```

3. **Train advanced model**:
```bash
python src/models/train_advanced.py --data data/raw/accepted_2007_to_2018Q4.csv
```

4. **With hyperparameter tuning**:
```bash
python src/models/train_advanced.py --data data/raw/accepted_2007_to_2018Q4.csv --tune
```

## Usage

### API Service

Start the FastAPI server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Access API documentation: http://localhost:8000/docs

**Example API request**:
```python
import requests

loan_data = {
    "loan_amnt": 10000,
    "term": "36 months",
    "int_rate": 10.5,
    "annual_inc": 60000,
    "dti": 15.5,
    "fico_range_high": 720,
    "revol_bal": 5000,
    "revol_util": 30.0
}

response = requests.post(
    "http://localhost:8000/predict/explain",
    json=loan_data
)

print(response.json())
```

### Web Interface

Launch the Streamlit UI:
```bash
streamlit run ui/app.py
```

Access the interface: http://localhost:8501

### Docker Deployment

Build and run with Docker Compose:
```bash
docker-compose up --build
```

Services will be available at:
- API: http://localhost:8000
- UI: http://localhost:8501

## Model Performance

The system includes comprehensive evaluation metrics:

- **F1 Score**: Balanced measure of precision and recall
- **ROC-AUC**: Overall classification performance
- **Precision/Recall**: Optimized for business objectives
- **Cost-based metrics**: Threshold tuning by false positive/negative costs

### Key Results
- Detects and removes data leakage features
- Handles class imbalance with appropriate techniques
- Provides feature importance and SHAP explanations
- Monitors data drift in production

## Explainability

The system uses SHAP (SHapley Additive exPlanations) to provide:

- **Global explanations**: Overall feature importance
- **Local explanations**: Individual prediction reasoning
- **Waterfall plots**: Feature contribution breakdown
- **Summary plots**: Population-level insights

## Monitoring

### Data Drift Detection
```python
from src.monitoring.drift_detection import DriftDetector

detector = DriftDetector()
detector.load_reference_data('data/processed/train.csv')
drift_summary = detector.detect_drift(production_data)
```

### Performance Monitoring
```python
from src.monitoring.drift_detection import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.log_predictions(predictions, probabilities)
stats = monitor.get_performance_stats(days=30)
```

## Threshold Optimization

Optimize classification threshold for business objectives:

```python
from src.models.threshold_tuning import ThresholdOptimizer

optimizer = ThresholdOptimizer(cost_fp=1, cost_fn=5)
optimal_threshold = optimizer.find_optimal_threshold(y_test, y_proba)
optimizer.plot_threshold_analysis()
```

## Testing

Run tests:
```bash
pytest tests/ -v --cov=src
```

## Configuration

Edit `config/config.yaml` to customize:
- Model hyperparameters
- Data paths
- API settings
- Monitoring thresholds

## Technical Stack

- **ML/Data**: scikit-learn, XGBoost, pandas, numpy
- **Explainability**: SHAP
- **API**: FastAPI, uvicorn
- **UI**: Streamlit
- **Monitoring**: Evidently
- **Visualization**: matplotlib, seaborn, plotly
- **Deployment**: Docker

## Business Value

This system demonstrates:

1. **Full ML Lifecycle**: From data exploration to deployment
2. **Production Readiness**: API, monitoring, and containerization
3. **Explainability**: Transparent, interpretable predictions
4. **Business Alignment**: Cost-based optimization
5. **Best Practices**: Leakage detection, drift monitoring, testing

## Use Cases

- **Lending Decisions**: Automated loan approval recommendations
- **Risk Assessment**: Portfolio risk evaluation
- **Credit Scoring**: Alternative credit evaluation
- **Fraud Detection**: Suspicious application flagging

## Future Enhancements

- Real-time prediction pipeline
- A/B testing framework
- Automated retraining triggers
- Fairness analysis and bias mitigation
- Multi-model ensemble
- Feature store integration

## License

This project is intended for portfolio demonstration purposes.

## Contact

For questions or collaboration opportunities, please reach out through the repository.

---

**Note**: This is a demonstration system. For production lending decisions, always consult with financial and legal professionals.
