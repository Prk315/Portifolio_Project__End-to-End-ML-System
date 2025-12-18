import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import shutil


class ModelRegistry:
    """
    Model registry for versioning and tracking ML models.
    Manages model artifacts, metadata, and performance metrics.
    """

    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "registry_metadata.json"
        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize registry metadata file if it doesn't exist."""
        if not self.metadata_file.exists():
            initial_data = {
                "models": {},
                "latest_version": None,
                "production_version": None
            }
            self._save_metadata(initial_data)

    def _load_metadata(self) -> Dict:
        """Load registry metadata."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata: Dict):
        """Save registry metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def register_model(
        self,
        model,
        preprocessor,
        version: str,
        model_type: str,
        metrics: Dict[str, float],
        hyperparameters: Optional[Dict] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a new model version.

        Args:
            model: Trained model object
            preprocessor: Fitted preprocessor object
            version: Version string (e.g., "1.0.0", "2.1.0")
            model_type: Type of model (e.g., "XGBoost", "LogisticRegression")
            metrics: Performance metrics dict (e.g., {"f1": 0.85, "roc_auc": 0.92})
            hyperparameters: Model hyperparameters
            description: Model description
            tags: Additional tags (e.g., {"dataset": "lending_club_2023"})

        Returns:
            str: Path to registered model directory
        """
        # Create version directory
        version_dir = self.registry_path / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model and preprocessor
        model_path = version_dir / "model.pkl"
        preprocessor_path = version_dir / "preprocessor.pkl"
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        # Create model metadata
        model_metadata = {
            "version": version,
            "model_type": model_type,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparameters": hyperparameters or {},
            "description": description or "",
            "tags": tags or {},
            "status": "registered",
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path)
        }

        # Update registry metadata
        registry_metadata = self._load_metadata()
        registry_metadata["models"][version] = model_metadata
        registry_metadata["latest_version"] = version
        self._save_metadata(registry_metadata)

        # Save version-specific metadata
        version_metadata_path = version_dir / "metadata.json"
        with open(version_metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        print(f"Model version {version} registered successfully at {version_dir}")
        return str(version_dir)

    def promote_to_production(self, version: str):
        """
        Promote a model version to production.

        Args:
            version: Version to promote
        """
        registry_metadata = self._load_metadata()

        if version not in registry_metadata["models"]:
            raise ValueError(f"Version {version} not found in registry")

        # Update previous production model status
        if registry_metadata["production_version"]:
            prev_version = registry_metadata["production_version"]
            registry_metadata["models"][prev_version]["status"] = "archived"

        # Set new production version
        registry_metadata["models"][version]["status"] = "production"
        registry_metadata["models"][version]["promoted_at"] = datetime.now().isoformat()
        registry_metadata["production_version"] = version

        self._save_metadata(registry_metadata)

        # Copy production model to standard location
        production_dir = self.registry_path.parent
        version_dir = self.registry_path / f"v{version}"

        shutil.copy(version_dir / "model.pkl", production_dir / "model.pkl")
        shutil.copy(version_dir / "preprocessor.pkl", production_dir / "preprocessor.pkl")

        print(f"Model version {version} promoted to production")

    def get_model(self, version: Optional[str] = None):
        """
        Load a specific model version or production model.

        Args:
            version: Model version to load. If None, loads production model.

        Returns:
            Tuple of (model, preprocessor, metadata)
        """
        registry_metadata = self._load_metadata()

        if version is None:
            version = registry_metadata.get("production_version")
            if version is None:
                raise ValueError("No production model set")

        if version not in registry_metadata["models"]:
            raise ValueError(f"Version {version} not found in registry")

        version_dir = self.registry_path / f"v{version}"
        model = joblib.load(version_dir / "model.pkl")
        preprocessor = joblib.load(version_dir / "preprocessor.pkl")
        metadata = registry_metadata["models"][version]

        return model, preprocessor, metadata

    def list_models(self) -> pd.DataFrame:
        """
        List all registered models with their metadata.

        Returns:
            DataFrame with model versions and metadata
        """
        registry_metadata = self._load_metadata()

        if not registry_metadata["models"]:
            return pd.DataFrame()

        models_data = []
        for version, metadata in registry_metadata["models"].items():
            row = {
                "version": version,
                "model_type": metadata["model_type"],
                "registered_at": metadata["registered_at"],
                "status": metadata["status"],
                **{f"metric_{k}": v for k, v in metadata["metrics"].items()}
            }
            models_data.append(row)

        df = pd.DataFrame(models_data)
        return df.sort_values("registered_at", ascending=False)

    def compare_models(self, version1: str, version2: str) -> Dict:
        """
        Compare two model versions.

        Args:
            version1: First version
            version2: Second version

        Returns:
            Comparison dictionary
        """
        registry_metadata = self._load_metadata()

        if version1 not in registry_metadata["models"]:
            raise ValueError(f"Version {version1} not found")
        if version2 not in registry_metadata["models"]:
            raise ValueError(f"Version {version2} not found")

        model1_meta = registry_metadata["models"][version1]
        model2_meta = registry_metadata["models"][version2]

        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {
                metric: {
                    "version1": model1_meta["metrics"].get(metric),
                    "version2": model2_meta["metrics"].get(metric),
                    "improvement": (
                        model2_meta["metrics"].get(metric, 0) -
                        model1_meta["metrics"].get(metric, 0)
                    )
                }
                for metric in set(list(model1_meta["metrics"].keys()) +
                                list(model2_meta["metrics"].keys()))
            },
            "model_type_v1": model1_meta["model_type"],
            "model_type_v2": model2_meta["model_type"]
        }

        return comparison

    def delete_version(self, version: str, force: bool = False):
        """
        Delete a model version.

        Args:
            version: Version to delete
            force: If True, allows deletion of production model
        """
        registry_metadata = self._load_metadata()

        if version not in registry_metadata["models"]:
            raise ValueError(f"Version {version} not found")

        if registry_metadata["models"][version]["status"] == "production" and not force:
            raise ValueError("Cannot delete production model. Use force=True to override.")

        # Delete version directory
        version_dir = self.registry_path / f"v{version}"
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # Update registry metadata
        del registry_metadata["models"][version]

        if registry_metadata["production_version"] == version:
            registry_metadata["production_version"] = None

        if registry_metadata["latest_version"] == version:
            # Set latest to most recent remaining version
            if registry_metadata["models"]:
                latest = max(
                    registry_metadata["models"].items(),
                    key=lambda x: x[1]["registered_at"]
                )[0]
                registry_metadata["latest_version"] = latest
            else:
                registry_metadata["latest_version"] = None

        self._save_metadata(registry_metadata)
        print(f"Model version {version} deleted successfully")


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    print("Model Registry initialized")
    print(registry.list_models())
