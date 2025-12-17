from setuptools import setup, find_packages

setup(
    name="loan-default-prediction",
    version="1.0.0",
    description="End-to-End ML System for Loan Default Prediction",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.1.4",
        "numpy>=1.26.2",
        "scikit-learn>=1.3.2",
        "xgboost>=2.0.3",
        "lightgbm>=4.1.0",
        "shap>=0.44.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.25.0",
        "streamlit>=1.29.0",
        "evidently>=0.4.11",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "joblib>=1.3.2",
        "pyyaml>=6.0.1",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
