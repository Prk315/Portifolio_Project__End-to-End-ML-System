"""Main training script for all models."""

import argparse
import pandas as pd
from pathlib import Path

from src.preprocessing.data_loader import DataLoader
from src.preprocessing.text_processor import TextPreprocessor
from src.models.baseline import train_baseline_model
from src.models.transformer import train_transformer_model
from src.rag.retriever import RAGRetriever
from config.config import (
    PREPROCESSING_CONFIG,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RAG_CONFIG
)
from src.utils.logger import setup_logger

# Setup logger
log = setup_logger("training.log")


def prepare_data():
    """Load and preprocess data."""
    log.info("Loading and preprocessing data...")

    loader = DataLoader()

    # Load data (will create sample if not available)
    df = loader.load_twitter_support()
    log.info(f"Loaded {len(df)} samples")

    # Preprocess
    preprocessor = TextPreprocessor(**PREPROCESSING_CONFIG)
    df['text'] = preprocessor.process(df['text'].tolist())

    # Filter by length
    df = preprocessor.filter_by_length(df, 'text')
    log.info(f"After filtering: {len(df)} samples")

    # Split data
    train_df, val_df, test_df = loader.prepare_dataset(df)

    # Save processed data
    loader.save_processed_data(train_df, val_df, test_df)

    return train_df, val_df, test_df


def train_baseline(train_df, val_df, test_df):
    """Train baseline model."""
    log.info("\n" + "="*80)
    log.info("Training Baseline Model (TF-IDF + Logistic Regression)")
    log.info("="*80)

    model = train_baseline_model(train_df, val_df, test_df)
    log.info("Baseline model training completed!")

    return model


def train_transformer(train_df, val_df, test_df):
    """Train transformer model."""
    log.info("\n" + "="*80)
    log.info("Training Transformer Model (DistilBERT)")
    log.info("="*80)

    model = train_transformer_model(train_df, val_df, test_df)
    log.info("Transformer model training completed!")

    return model


def build_rag_index(train_df):
    """Build RAG retriever index."""
    log.info("\n" + "="*80)
    log.info("Building RAG Retriever Index")
    log.info("="*80)

    retriever = RAGRetriever(**RAG_CONFIG)
    retriever.build_index(train_df, text_column='text', label_column='label')
    retriever.save()

    log.info("RAG retriever built and saved!")

    return retriever


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train customer support triage models')
    parser.add_argument(
        '--model',
        type=str,
        choices=['baseline', 'transformer', 'rag', 'all'],
        default='all',
        help='Which model to train'
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing (use existing processed data)'
    )

    args = parser.parse_args()

    # Prepare data
    if args.skip_preprocessing:
        log.info("Loading existing processed data...")
        train_df = pd.read_csv(PROCESSED_DATA_DIR / 'processed_train.csv')
        val_df = pd.read_csv(PROCESSED_DATA_DIR / 'processed_val.csv')
        test_df = pd.read_csv(PROCESSED_DATA_DIR / 'processed_test.csv')
    else:
        train_df, val_df, test_df = prepare_data()

    log.info(f"\nDataset sizes:")
    log.info(f"Train: {len(train_df)}")
    log.info(f"Validation: {len(val_df)}")
    log.info(f"Test: {len(test_df)}")

    # Train models
    if args.model in ['baseline', 'all']:
        train_baseline(train_df, val_df, test_df)

    if args.model in ['transformer', 'all']:
        train_transformer(train_df, val_df, test_df)

    if args.model in ['rag', 'all']:
        build_rag_index(train_df)

    log.info("\n" + "="*80)
    log.info("Training pipeline completed!")
    log.info("="*80)
    log.info(f"\nModels saved to: {MODELS_DIR}")
    log.info("\nTo start the API server, run:")
    log.info("  uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
