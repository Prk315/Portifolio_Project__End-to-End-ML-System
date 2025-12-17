import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.preprocessing import LoanDataPreprocessor, split_data

class BaselineModel:
    """
    Logistic Regression baseline for loan default prediction.
    """

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = LogisticRegression(
            max_iter=self.config['model']['baseline']['max_iter'],
            random_state=self.config['model']['random_state'],
            class_weight='balanced'
        )
        self.preprocessor = None
        self.metrics = {}

    def train(self, data_path):
        """
        Trains baseline logistic regression model.
        """
        print("=" * 60)
        print("BASELINE MODEL TRAINING")
        print("=" * 60)

        print("\nLoading data...")
        df = pd.read_csv(data_path, low_memory=False)
        print(f"Loaded {len(df):,} records")

        print("\nPreprocessing...")
        self.preprocessor = LoanDataPreprocessor(
            target_col=self.config['model']['target_column']
        )

        if 'loan_status' in df.columns:
            df['loan_status'] = df['loan_status'].apply(
                lambda x: 1 if x in ['Charged Off', 'Default'] else 0
            )

        df = self.preprocessor.prepare_lending_club_data(df)
        df = self.preprocessor.engineer_features(df)

        X, y = self.preprocessor.fit_transform(df)

        print(f"\nFeatures: {X.shape[1]}")
        print(f"Target distribution: {y.value_counts().to_dict()}")

        print("\nSplitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y,
            test_size=self.config['model']['test_size'],
            val_size=self.config['model']['val_size'],
            random_state=self.config['model']['random_state']
        )

        print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

        print("\nTraining model...")
        self.model.fit(X_train, y_train)

        print("\nEvaluating...")
        self._evaluate(X_train, y_train, "Train")
        self._evaluate(X_val, y_val, "Validation")
        self._evaluate(X_test, y_test, "Test")

        models_dir = Path(self.config['paths']['models_dir'])
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / 'baseline_model.joblib'
        preprocessor_path = models_dir / 'baseline_preprocessor.joblib'

        joblib.dump(self.model, model_path)
        self.preprocessor.save(preprocessor_path)

        print(f"\nModel saved to: {model_path}")
        print(f"Preprocessor saved to: {preprocessor_path}")

        return self.metrics

    def _evaluate(self, X, y, dataset_name):
        """
        Evaluates model performance.
        """
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_proba)

        self.metrics[dataset_name] = {
            'f1': f1,
            'roc_auc': roc_auc
        }

        print(f"\n{dataset_name} Set Performance:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))

    def predict(self, X):
        """
        Makes predictions on new data.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Returns prediction probabilities.
        """
        return self.model.predict_proba(X)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    args = parser.parse_args()

    model = BaselineModel()
    model.train(args.data)
