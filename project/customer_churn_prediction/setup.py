from setuptools import setup, find_packages

setup(
    name="customer_churn_prediction",
    version="0.1.0",
    description="Production-grade customer churn prediction and retention model",
    author="Vamsi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "shap>=0.41.0",
        "scipy>=1.9.0",
        "joblib>=1.2.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "churn-train=pipelines.train_pipeline:main",
            "churn-predict=pipelines.predict_pipeline:main",
            "churn-serve=api.main:start",
        ]
    },
)
