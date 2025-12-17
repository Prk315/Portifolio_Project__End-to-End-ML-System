"""Data loading and preparation utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logger import log


class DataLoader:
    """Load and prepare datasets for training."""

    def __init__(self):
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR

    def load_twitter_support(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load Twitter customer support dataset.

        Args:
            file_path: Path to CSV file. If None, use default location.

        Returns:
            DataFrame with 'text' and 'label' columns
        """
        if file_path is None:
            file_path = self.raw_dir / "customer_support_twitter.csv"

        log.info(f"Loading Twitter support data from {file_path}")

        # Check if file exists
        if not Path(file_path).exists():
            log.warning(f"File not found: {file_path}")
            # Return sample data for development
            return self._create_sample_data()

        try:
            df = pd.read_csv(file_path)
            log.info(f"Loaded {len(df)} records")

            # Process based on actual column structure
            # Twitter support typically has: tweet_id, author_id, text, in_response_to_tweet_id
            # We'll need to identify issues and categorize them
            # For now, create a simple categorization based on keywords

            if 'text' not in df.columns:
                # Find the text column
                text_col = [col for col in df.columns if 'text' in col.lower()]
                if text_col:
                    df['text'] = df[text_col[0]]
                else:
                    raise ValueError("No text column found")

            # Create labels based on keywords (this is a simplified approach)
            df['label'] = df['text'].apply(self._categorize_tweet)

            # Clean up
            df = df[['text', 'label']].dropna()

            return df

        except Exception as e:
            log.error(f"Error loading data: {e}")
            return self._create_sample_data()

    def load_amazon_reviews(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load Amazon reviews dataset.

        Args:
            file_path: Path to file

        Returns:
            DataFrame with 'text' and 'label' columns
        """
        if file_path is None:
            file_path = self.raw_dir / "amazon_reviews.csv"

        log.info(f"Loading Amazon reviews from {file_path}")

        if not Path(file_path).exists():
            log.warning(f"File not found: {file_path}")
            return self._create_sample_data()

        try:
            df = pd.read_csv(file_path)

            # Amazon reviews typically have: rating, review_text
            if 'rating' in df.columns:
                # Convert rating to categories
                df['label'] = pd.cut(
                    df['rating'],
                    bins=[0, 2, 3, 5],
                    labels=['negative', 'neutral', 'positive']
                )

            if 'review_text' in df.columns:
                df['text'] = df['review_text']
            elif 'text' not in df.columns:
                text_col = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower()]
                if text_col:
                    df['text'] = df[text_col[0]]

            df = df[['text', 'label']].dropna()
            return df

        except Exception as e:
            log.error(f"Error loading Amazon reviews: {e}")
            return self._create_sample_data()

    def _categorize_tweet(self, text: str) -> str:
        """Categorize tweet based on keywords."""
        text_lower = str(text).lower()

        if any(word in text_lower for word in ['account', 'login', 'password', 'access']):
            return 'account_issue'
        elif any(word in text_lower for word in ['bill', 'charge', 'payment', 'refund']):
            return 'billing'
        elif any(word in text_lower for word in ['error', 'bug', 'broken', 'not working', 'issue']):
            return 'technical_support'
        elif any(word in text_lower for word in ['ship', 'delivery', 'arrived', 'tracking']):
            return 'shipping_delivery'
        elif any(word in text_lower for word in ['return', 'exchange', 'refund']):
            return 'returns_refunds'
        elif any(word in text_lower for word in ['product', 'feature', 'how to', 'what is']):
            return 'product_inquiry'
        elif any(word in text_lower for word in ['complaint', 'disappointed', 'terrible', 'worst']):
            return 'complaint'
        else:
            return 'general_inquiry'

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for development."""
        log.info("Creating sample dataset")

        samples = [
            ("I can't log into my account. Keep getting error message.", "account_issue"),
            ("Where is my order? It's been 2 weeks!", "shipping_delivery"),
            ("Your app keeps crashing when I try to upload photos", "technical_support"),
            ("I was charged twice for the same order", "billing"),
            ("How do I return this item? It doesn't fit.", "returns_refunds"),
            ("What are the features of the premium plan?", "product_inquiry"),
            ("This is the worst customer service I've ever experienced!", "complaint"),
            ("Can you help me with something?", "general_inquiry"),
            ("My password reset link isn't working", "account_issue"),
            ("The product arrived damaged, need replacement", "complaint"),
            ("How much does shipping cost?", "shipping_delivery"),
            ("I want to cancel my subscription", "billing"),
            ("The website is not loading properly", "technical_support"),
            ("Do you have this in a different color?", "product_inquiry"),
            ("I haven't received my refund yet", "returns_refunds"),
            ("Thanks for the quick response!", "general_inquiry"),
            ("My order number ABC123 still shows as processing", "shipping_delivery"),
            ("I need to update my billing address", "billing"),
            ("The mobile app keeps logging me out", "technical_support"),
            ("What's your return policy?", "returns_refunds")
        ] * 50  # Repeat to create more samples

        df = pd.DataFrame(samples, columns=['text', 'label'])
        return df

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        log.info(f"Splitting dataset: test_size={test_size}, val_size={val_size}")

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_df['label']
        )

        log.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prefix: str = "processed"
    ):
        """Save processed datasets."""
        train_path = self.processed_dir / f"{prefix}_train.csv"
        val_path = self.processed_dir / f"{prefix}_val.csv"
        test_path = self.processed_dir / f"{prefix}_test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        log.info(f"Saved processed data to {self.processed_dir}")
