import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

class ModelExplainer:
    """
    Provides model explainability using SHAP values.
    """

    def __init__(self, model_path, preprocessor_path):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.explainer = None
        self.shap_values = None

    def initialize_explainer(self, X_sample=None):
        """
        Initializes SHAP explainer.
        For tree-based models, uses TreeExplainer.
        """
        print("Initializing SHAP explainer...")

        if hasattr(self.model, 'get_booster'):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            if X_sample is None:
                raise ValueError("X_sample required for non-tree models")
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X_sample
            )

        print("Explainer initialized successfully")

    def compute_shap_values(self, X):
        """
        Computes SHAP values for given data.
        """
        if self.explainer is None:
            self.initialize_explainer(X)

        print(f"Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X)

        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]

        print("SHAP values computed")
        return self.shap_values

    def plot_summary(self, X, max_display=20):
        """
        Creates SHAP summary plot showing feature importance.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.preprocessor.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.show()

    def plot_waterfall(self, X, idx=0):
        """
        Creates waterfall plot for a single prediction.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[idx],
                base_values=self.explainer.expected_value,
                data=X.iloc[idx],
                feature_names=self.preprocessor.feature_names
            )
        )

    def plot_force(self, X, idx=0):
        """
        Creates force plot for a single prediction.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[idx],
            X.iloc[idx],
            feature_names=self.preprocessor.feature_names,
            matplotlib=True
        )
        plt.show()

    def get_top_features(self, X, n_features=10):
        """
        Returns top features by average absolute SHAP value.
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        feature_importance = pd.DataFrame({
            'feature': self.preprocessor.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        return feature_importance.head(n_features)

    def explain_prediction(self, loan_data, return_text=True):
        """
        Explains a single loan prediction with SHAP values.
        Returns human-readable explanation.
        """
        X = self.preprocessor.transform(pd.DataFrame([loan_data]))

        if self.shap_values is None:
            self.compute_shap_values(X)

        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]

        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1][0]
        else:
            shap_vals = shap_vals[0]

        feature_contributions = pd.DataFrame({
            'feature': self.preprocessor.feature_names,
            'value': X.iloc[0],
            'shap': shap_vals
        }).sort_values('shap', key=abs, ascending=False)

        if return_text:
            explanation = f"Prediction: {'Default' if prediction == 1 else 'No Default'}\n"
            explanation += f"Probability of Default: {probability:.2%}\n\n"
            explanation += "Top factors influencing this prediction:\n"

            for idx, row in feature_contributions.head(5).iterrows():
                direction = "increases" if row['shap'] > 0 else "decreases"
                explanation += f"- {row['feature']} (value: {row['value']:.2f}) {direction} default risk by {abs(row['shap']):.4f}\n"

            return explanation, feature_contributions
        else:
            return prediction, probability, feature_contributions

if __name__ == "__main__":
    print("Explainability module loaded")
