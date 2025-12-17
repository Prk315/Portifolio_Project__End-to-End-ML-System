#!/bin/bash

# Quick start script for Customer Support Triage System

echo "=================================="
echo "Customer Support Triage System"
echo "Quick Start Setup"
echo "=================================="
echo ""

# Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
echo "[3/5] Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
echo "[4/5] Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Train models (using sample data)
echo "[5/5] Training models with sample data..."
echo "This may take a few minutes..."
python train.py --model all

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To start the API server:"
echo "  source venv/bin/activate"
echo "  uvicorn src.api.main:app --reload"
echo ""
echo "Then visit: http://localhost:8000/docs"
echo ""
echo "To explore notebooks:"
echo "  jupyter notebook notebooks/"
echo ""
