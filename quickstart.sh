#!/bin/bash

echo "================================================"
echo "Loan Default Prediction System - Quick Start"
echo "================================================"

echo -e "\n1. Creating virtual environment..."
python3 -m venv venv

echo -e "\n2. Activating virtual environment..."
source venv/bin/activate

echo -e "\n3. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "\n4. Creating directory structure..."
mkdir -p data/raw data/processed models logs reports

echo -e "\n5. Setting up environment file..."
cp .env.example .env

echo -e "\n================================================"
echo "Setup complete!"
echo "================================================"
echo -e "\nNext steps:"
echo "  1. Download dataset from Kaggle:"
echo "     https://www.kaggle.com/datasets/wordsforthewise/lending-club"
echo "  2. Place data in: data/raw/"
echo "  3. Explore data: jupyter notebook notebooks/01_EDA_and_Leakage_Detection.ipynb"
echo "  4. Train model: python src/models/train_advanced.py --data data/raw/your_file.csv"
echo "  5. Start API: make api"
echo "  6. Start UI: make ui"
echo -e "\nFor help: make help"
