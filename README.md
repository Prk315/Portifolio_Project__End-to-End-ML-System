# Customer Support Ticket Triage & Response System

An end-to-end NLP system that automatically triages customer support tickets and suggests appropriate responses using modern NLP techniques including transformers and retrieval-augmented generation (RAG).

## Problem Statement

Customer support teams receive thousands of tickets daily. Manual triage is time-consuming and inconsistent. This system:
- **Automatically categorizes** support tickets by urgency and topic
- **Suggests responses** based on similar historical tickets
- **Routes tickets** to appropriate teams with confidence scoring
- **Reduces response time** and improves customer satisfaction

## Key Features

- **Multi-Model Architecture**: Baseline (TF-IDF + Logistic) and Transformer (DistilBERT)
- **Confidence-Based Routing**: High-confidence predictions go straight to suggested responses, low-confidence to human review
- **RAG Integration**: Retrieves similar tickets and generates contextual responses
- **Production-Ready API**: FastAPI endpoint for real-time inference
- **Comprehensive Error Analysis**: Detailed model performance insights

## Project Structure

```
.
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned and preprocessed data
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_baseline_model.ipynb         # TF-IDF + Logistic Regression
│   ├── 03_transformer_model.ipynb      # DistilBERT fine-tuning
│   └── 04_error_analysis.ipynb         # Model evaluation
├── src/
│   ├── preprocessing/    # Text preprocessing utilities
│   ├── models/          # Model implementations
│   ├── rag/             # RAG system components
│   ├── api/             # FastAPI application
│   └── utils/           # Helper functions
├── models/              # Saved model artifacts
├── config/              # Configuration files
└── tests/               # Unit tests
```

## Datasets

1. **Customer Support on Twitter** (Kaggle)
   - Real customer service conversations
   - Multi-turn dialogues

2. **Amazon Reviews** (Kaggle)
   - Product feedback with sentiment
   - Rich text for classification

## Methodology

### 1. Text Preprocessing & EDA
- Text cleaning and normalization
- Exploratory analysis of ticket categories
- Class imbalance detection
- Text length and vocabulary analysis

### 2. Baseline Model
- **Features**: TF-IDF vectorization
- **Model**: Logistic Regression with class weighting
- **Purpose**: Establish performance floor

### 3. Transformer Model
- **Architecture**: DistilBERT (lightweight, fast)
- **Fine-tuning**: On customer support domain
- **Optimization**: Learning rate scheduling, gradient clipping

### 4. Error Analysis
- Confusion matrix analysis
- Per-class performance metrics
- Failure case examination
- Bias detection

### 5. RAG System
- **Vector Store**: FAISS for similarity search
- **Embeddings**: sentence-transformers
- **Response Generation**: Template-based + retrieval

### 6. API Deployment
- FastAPI with async support
- Confidence-based routing logic
- Model versioning
- Response caching

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd PortefolioProject__End-to-End-ML-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Data Preparation
```bash
# Download datasets (instructions in data/README.md)
python src/preprocessing/download_data.py

# Preprocess data
python src/preprocessing/preprocess.py
```

### 2. Train Models
```bash
# Train baseline
python src/models/train_baseline.py

# Train transformer
python src/models/train_transformer.py
```

### 3. Run API
```bash
# Start FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "My order hasn't arrived and I need it urgently!"}'
```

## API Endpoints

- `POST /predict` - Classify ticket and suggest response
- `POST /batch_predict` - Batch prediction
- `GET /health` - Service health check
- `GET /metrics` - Model performance metrics

## Model Performance

| Model | Accuracy | F1-Score | Inference Time |
|-------|----------|----------|----------------|
| Baseline (TF-IDF + LR) | TBD | TBD | ~5ms |
| DistilBERT | TBD | TBD | ~50ms |

## Key Learnings

- Modern NLP pipeline development
- Transformer fine-tuning and optimization
- Production API design patterns
- RAG system implementation
- Model monitoring and error analysis

## Future Enhancements

- [ ] Multi-language support
- [ ] Sentiment analysis integration
- [ ] Real-time model retraining
- [ ] A/B testing framework
- [ ] Advanced RAG with GPT integration

## License

MIT License
