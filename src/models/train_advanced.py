import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.preprocessing import LoanDataPreprocessor, split_data

class AdvancedModel:
    """
    XGBoost model for loan default prediction with hyperparameter tuning.
    """

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.preprocessor = None
        self.metrics = {}
        self.best_params = None

    def train(self, data_path, tune=False):
        """
        Trains advanced XGBoost model with optional hyperparameter tuning.
        """
        print("=" * 60)
        print("ADVANCED MODEL TRAINING (XGBoost)")
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

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        if tune:
            print("\nTuning hyperparameters...")
            self._tune_hyperparameters(X_train, y_train, scale_pos_weight)
        else:
            print("\nUsing default parameters from config...")
            self.best_params = {
                'n_estimators': self.config['model']['advanced']['n_estimators'],
                'max_depth': self.config['model']['advanced']['max_depth'],
                'learning_rate': self.config['model']['advanced']['learning_rate'],
                'subsample': self.config['model']['advanced']['subsample'],
                'colsample_bytree': self.config['model']['advanced']['colsample_bytree'],
                'reg_alpha': self.config['model']['advanced']['reg_alpha'],
                'reg_lambda': self.config['model']['advanced']['reg_lambda'],
            }

        print("\nTraining final model...")
        self.model = xgb.XGBClassifier(
            **self.best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config['model']['random_state'],
            eval_metric='logloss'
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        print("\nEvaluating...")
        self._evaluate(X_train, y_train, "Train")
        self._evaluate(X_val, y_val, "Validation")
        self._evaluate(X_test, y_test, "Test")

        self._analyze_feature_importance(X_train)

        models_dir = Path(self.config['paths']['models_dir'])
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / 'final_model.joblib'
        preprocessor_path = models_dir / 'preprocessor.joblib'

        joblib.dump(self.model, model_path)
        self.preprocessor.save(preprocessor_path)

        print(f"\nModel saved to: {model_path}")
        print(f"Preprocessor saved to: {preprocessor_path}")

        return self.metrics

    def _tune_hyperparameters(self, X_train, y_train, scale_pos_weight):
        """
        Tunes hyperparameters using GridSearchCV.
        """
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
        }

        base_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=self.config['model']['random_state']
        )

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

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

    def _analyze_feature_importance(self, X_train):
        """
        Analyzes and displays feature importance.
        """
        feature_importance = pd.DataFrame({
            'feature': self.preprocessor.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))

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

    parser = argparse.ArgumentParser(description='Train advanced model')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    args = parser.parse_args()

    model = AdvancedModel()
    model.train(args.data, tune=args.tune)
