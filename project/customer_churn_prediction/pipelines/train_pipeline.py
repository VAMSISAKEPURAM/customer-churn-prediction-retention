"""
train_pipeline.py
=================
End-to-end training pipeline for the Customer Churn Prediction model.

Run with:
    python pipelines/train_pipeline.py --config configs/train_config.yaml

Steps:
    1. Load & validate raw data
    2. Join all source tables
    3. Encode categorical features
    4. Feature selection
    5. Train/val split
    6. Model training
    7. Probability calibration
    8. Evaluation & reporting
    9. Save all artifacts
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path so imports work whether or not package is installed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.helpers import load_config, setup_logging, set_seed
from src.data.ingest import DataLoader, DataValidator, join_all_sources
from src.features.encode import CategoricalEncoderPipeline
from src.features.feature_selection import FeatureSelector
from src.models.train import build_model, train_model, calibrate_model, save_model
from src.evaluation.metrics import (
    compute_metrics,
    save_metrics_report,
    plot_roc_curve,
    plot_calibration_curve,
)

from sklearn.model_selection import train_test_split

logger = logging.getLogger("pipelines.train")


def run(config_path: str):
    # ---- Setup ----
    config = load_config(config_path)
    setup_logging()
    set_seed(config["project"]["random_seed"])
    logger.info(f"Starting training pipeline | Project: {config['project']['name']} v{config['project']['version']}")

    # ---- Step 1: Load & Validate Data ----
    loader = DataLoader(config)
    validator = DataValidator()
    raw_dfs = loader.load_all()
    for name, df in raw_dfs.items():
        validator.validate(df, name)

    # ---- Step 2: Join Sources ----
    df = join_all_sources(raw_dfs)

    # ---- Step 3: Train/Test Split ----
    target = config["data"]["target_column"]
    split_cfg = config["data"]["train_test_split"]
    df_train, df_test = train_test_split(
        df,
        test_size=split_cfg["test_size"],
        random_state=config["project"]["random_seed"],
        stratify=df[target] if split_cfg["stratify"] else None,
    )
    logger.info(f"Train: {df_train.shape}, Test: {df_test.shape}")

    # ---- Step 4: Encode Categorical Features ----
    encoder = CategoricalEncoderPipeline(config)
    df_train_enc, df_test_enc = encoder.fit_transform(df_train, df_test)
    encoder.save("models/registry/categorical_encoders.pkl")

    # ---- Step 5: Feature Selection ----
    id_col = config["data"]["id_column"]
    X_train = df_train_enc.drop(columns=[target, id_col], errors="ignore")
    y_train = df_train_enc[target]
    X_test = df_test_enc.drop(columns=[target, id_col], errors="ignore")
    y_test = df_test_enc[target]

    selector = FeatureSelector(config)
    X_train_sel, selected_features = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    logger.info(f"Selected {len(selected_features)} features for modeling.")

    # ---- Step 6: Train Model ----
    # Use 80% of train for fitting, 20% for calibration
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_sel, y_train, test_size=0.2,
        random_state=config["project"]["random_seed"],
        stratify=y_train,
    )
    model = build_model(config)
    model = train_model(model, X_fit, y_fit, X_cal, y_cal, config=config)

    # ---- Step 7: Calibrate Probabilities ----
    calibration_method = config["model"].get("calibration_method", "isotonic")
    model = calibrate_model(model, X_cal, y_cal, method=calibration_method)

    # ---- Step 8: Evaluate ----
    y_prob_test = model.predict_proba(X_test_sel)[:, 1]
    metrics = compute_metrics(y_test.values, y_prob_test, threshold=config["evaluation"]["threshold"])

    reports_dir = config["artifacts"]["reports_dir"]
    save_metrics_report(metrics, reports_dir, split="test")
    plot_roc_curve(y_test.values, y_prob_test, reports_dir)
    plot_calibration_curve(y_test.values, y_prob_test, reports_dir)

    # ---- Step 9: Save Artifacts ----
    save_model(model, selected_features, config, metrics)
    logger.info("Training pipeline complete. All artifacts saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the churn prediction model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    run(args.config)
