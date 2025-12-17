import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

class LoanDataPreprocessor:
    """
    Preprocesses loan data for modeling.
    Handles missing values, encoding, and feature engineering.
    """

    def __init__(self, target_col='loan_status'):
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None

    def detect_leakage_features(self, df):
        """
        Identifies features that may cause data leakage.
        Returns list of suspicious features to review.
        """
        leakage_keywords = [
            'total_pymnt', 'recoveries', 'collection_recovery',
            'last_pymnt', 'next_pymnt', 'debt_settlement'
        ]

        suspicious = []
        for col in df.columns:
            col_lower = col.lower()
            for keyword in leakage_keywords:
                if keyword in col_lower:
                    suspicious.append(col)
                    break

        return suspicious

    def prepare_lending_club_data(self, df):
        """
        Specific preprocessing for Lending Club dataset.
        """
        df = df.copy()

        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].apply(
                lambda x: 1 if x in ['Charged Off', 'Default'] else 0
            )

        leakage_cols = self.detect_leakage_features(df)
        if leakage_cols:
            print(f"Removing potential leakage features: {leakage_cols}")
            df = df.drop(columns=leakage_cols, errors='ignore')

        high_missing = df.columns[df.isnull().mean() > 0.5].tolist()
        if high_missing:
            print(f"Removing high-missing columns: {high_missing}")
            df = df.drop(columns=high_missing)

        return df

    def engineer_features(self, df):
        """
        Creates additional features from existing ones.
        """
        df = df.copy()

        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)

        if 'int_rate' in df.columns:
            df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float)

        if 'revol_util' in df.columns:
            df['revol_util'] = df['revol_util'].str.rstrip('%').astype(float)

        if 'term' in df.columns:
            df['term_months'] = df['term'].str.extract('(\d+)').astype(float)

        if 'emp_length' in df.columns:
            emp_map = {
                '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                '8 years': 8, '9 years': 9, '10+ years': 10
            }
            df['emp_length_years'] = df['emp_length'].map(emp_map)

        return df

    def fit_transform(self, df, target_col=None):
        """
        Fits preprocessor and transforms data.
        """
        df = df.copy()

        if target_col:
            self.target_col = target_col

        if self.target_col in df.columns:
            y = df[self.target_col]
            X = df.drop(columns=[self.target_col])
        else:
            y = None
            X = df

        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        for col in self.numeric_features:
            X[col] = X[col].fillna(X[col].median())

        for col in self.categorical_features:
            X[col] = X[col].fillna('missing')

            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        X[self.numeric_features] = self.scaler.fit_transform(X[self.numeric_features])

        self.feature_names = X.columns.tolist()

        return X, y

    def transform(self, df):
        """
        Transforms new data using fitted preprocessor.
        """
        df = df.copy()
        X = df.copy()

        for col in self.numeric_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())

        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].fillna('missing')

                le = self.label_encoders.get(col)
                if le:
                    X[col] = X[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        if self.numeric_features:
            X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])

        return X[self.feature_names]

    def save(self, path):
        """Saves preprocessor to disk."""
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path):
        """Loads preprocessor from disk."""
        return joblib.load(path)

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits data into train, validation, and test sets.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    print("Preprocessing module loaded successfully")
