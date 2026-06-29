"""
Model training module.
Handles model instantiation, class imbalance, training, and probability calibration.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)


def build_model(config: dict) -> Any:
    """
    Instantiate the model specified in the config.
    Supported: xgboost, lightgbm, logistic_regression
    """
    algorithm = config["model"]["algorithm"]
    logger.info(f"Building model: {algorithm}")

    if algorithm == "xgboost":
        from xgboost import XGBClassifier
        params = config["model"]["xgboost"].copy()
        # Handle auto scale_pos_weight
        if params.get("scale_pos_weight") == "auto":
            params.pop("scale_pos_weight")
        return XGBClassifier(random_state=config["project"]["random_seed"], **params)

    elif algorithm == "lightgbm":
        from lightgbm import LGBMClassifier
        params = config["model"]["lightgbm"].copy()
        return LGBMClassifier(random_state=config["project"]["random_seed"], **params)

    elif algorithm == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        params = config["model"]["logistic_regression"].copy()
        return LogisticRegression(random_state=config["project"]["random_seed"], **params)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from: xgboost, lightgbm, logistic_regression")


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    config: dict = {},
) -> Any:
    """
    Train the model with optional early stopping for XGBoost/LightGBM.
    Applies sample weights for class imbalance.
    """
    algorithm = config.get("model", {}).get("algorithm", "xgboost")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    logger.info(f"Training {algorithm} on {X_train.shape[0]} samples, {X_train.shape[1]} features...")

    if algorithm in ("xgboost", "lightgbm") and X_val is not None:
        eval_set = [(X_val, y_val)]
        model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=eval_set, verbose=50)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights)

    logger.info("Model training complete.")
    return model


def calibrate_model(model: Any, X_cal: pd.DataFrame, y_cal: pd.Series, method: str = "isotonic") -> Any:
    """
    Wrap a trained model with Platt scaling or isotonic regression for probability calibration.
    
    Args:
        method: 'isotonic' (non-parametric) or 'sigmoid' (Platt scaling)
    """
    if method == "none":
        logger.info("Skipping calibration.")
        return model
    
    logger.info(f"Calibrating probabilities using: {method}")
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    logger.info("Calibration complete.")
    return calibrated


def save_model(model: Any, feature_names: list, config: dict, metrics: dict):
    """
    Save model artifact, feature names, and training metadata.
    """
    model_dir = Path(config["artifacts"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "churn_model_latest.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")

    feature_path = model_dir / "feature_names.json"
    with open(feature_path, "w") as f:
        json.dump(feature_names, f, indent=4)
    logger.info(f"Feature names saved to: {feature_path}")

    metadata = {
        "algorithm": config["model"]["algorithm"],
        "calibration": config["model"].get("calibration_method", "none"),
        "num_features": len(feature_names),
        "train_metrics": metrics,
        "project_version": config["project"]["version"],
    }
    meta_path = model_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata saved to: {meta_path}")


def load_model(model_dir: str) -> Tuple[Any, list]:
    """Load a saved model and its feature names."""
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "churn_model_latest.pkl")
    with open(model_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    logger.info(f"Model loaded from: {model_dir}")
    return model, feature_names
