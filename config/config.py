"""Configuration file for the NLP triage system."""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset URLs
TWITTER_SUPPORT_URL = "https://www.kaggle.com/datasets/skywalker123/customer-support-on-twitter"
AMAZON_REVIEWS_URL = "https://www.kaggle.com/datasets/bittlingmayer/amazonreviews"

# Model parameters
BASELINE_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "max_iter": 1000,
    "random_state": 42,
    "test_size": 0.2,
    "val_size": 0.1
}

TRANSFORMER_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42
}

# RAG configuration
RAG_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 5,
    "similarity_threshold": 0.7,
    "index_type": "Flat"
}

# API configuration
API_CONFIG = {
    "title": "Customer Support Triage API",
    "description": "NLP-powered ticket classification and response suggestion",
    "version": "1.0.0",
    "confidence_threshold_high": 0.85,
    "confidence_threshold_low": 0.5,
    "default_model": "transformer"
}

# Ticket categories (customizable based on dataset)
TICKET_CATEGORIES = [
    "account_issue",
    "billing",
    "technical_support",
    "product_inquiry",
    "shipping_delivery",
    "returns_refunds",
    "general_inquiry",
    "complaint"
]

# Preprocessing parameters
PREPROCESSING_CONFIG = {
    "lowercase": True,
    "remove_urls": True,
    "remove_emails": True,
    "remove_special_chars": False,
    "remove_numbers": False,
    "remove_stopwords": False,  # Keep for context
    "lemmatize": False,  # BERT handles this
    "min_length": 10,  # Minimum character length
    "max_length": 512   # Maximum character length
}
