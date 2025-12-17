"""Text preprocessing utilities."""

import re
import string
from typing import List, Optional
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Text preprocessing pipeline."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_chars: bool = False,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        min_length: int = 10,
        max_length: int = 512
    ):
        """
        Initialize text preprocessor.

        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_special_chars: Remove special characters
            remove_numbers: Remove numbers
            remove_stopwords: Remove stopwords
            lemmatize: Apply lemmatization
            min_length: Minimum text length
            max_length: Maximum text length
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_length = min_length
        self.max_length = max_length

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)

        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove special characters (keep spaces and basic punctuation)
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove stopwords and lemmatize
        if self.remove_stopwords or self.lemmatize:
            tokens = word_tokenize(text)

            if self.remove_stopwords:
                tokens = [t for t in tokens if t not in self.stop_words]

            if self.lemmatize:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

            text = ' '.join(tokens)

        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]

        return text.strip()

    def process(self, texts: List[str]) -> List[str]:
        """
        Process a list of texts.

        Args:
            texts: List of input texts

        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]

    def filter_by_length(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Filter DataFrame by text length.

        Args:
            df: Input DataFrame
            text_column: Name of text column

        Returns:
            Filtered DataFrame
        """
        df['text_length'] = df[text_column].str.len()
        df_filtered = df[
            (df['text_length'] >= self.min_length) &
            (df['text_length'] <= self.max_length)
        ].copy()
        df_filtered.drop('text_length', axis=1, inplace=True)
        return df_filtered


def create_label_mapping(labels: pd.Series) -> tuple:
    """
    Create label to index mapping.

    Args:
        labels: Series of labels

    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    unique_labels = sorted(labels.unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def balance_dataset(
    df: pd.DataFrame,
    label_column: str,
    strategy: str = 'undersample',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance dataset by class.

    Args:
        df: Input DataFrame
        label_column: Name of label column
        strategy: 'undersample' or 'oversample'
        random_state: Random seed

    Returns:
        Balanced DataFrame
    """
    if strategy == 'undersample':
        # Undersample majority classes
        min_count = df[label_column].value_counts().min()
        df_balanced = df.groupby(label_column, group_keys=False).apply(
            lambda x: x.sample(min_count, random_state=random_state)
        )
    elif strategy == 'oversample':
        # Oversample minority classes
        max_count = df[label_column].value_counts().max()
        df_balanced = df.groupby(label_column, group_keys=False).apply(
            lambda x: x.sample(max_count, replace=True, random_state=random_state)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df_balanced.reset_index(drop=True)
