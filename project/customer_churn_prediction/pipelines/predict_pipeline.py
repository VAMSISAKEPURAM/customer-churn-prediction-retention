"""
predict_pipeline.py
===================
Batch scoring pipeline for the Customer Churn Prediction model.

Run with:
    python pipelines/predict_pipeline.py --config configs/predict_config.yaml

Steps:
    1. Load scoring input CSV
    2. Load model & encoder artifacts
    3. Apply encoding
    4. Score all customers
    5. Export scored CSV with churn_probability, risk_decile, score_date
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.helpers import load_config, setup_logging
from src.models.predict import ChurnScorer

import pandas as pd

logger = logging.getLogger("pipelines.predict")


def run(config_path: str):
    config = load_config(config_path)
    setup_logging()
    logger.info("Starting batch scoring pipeline...")

    # Load scoring data
    input_path = config["data"]["input_path"]
    logger.info(f"Loading scoring input: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"  → {len(df)} customers to score")

    # Score
    scorer = ChurnScorer.from_config(config)
    scored_df = scorer.score(df)

    # Export
    output_path = config["data"]["output_path"]
    scorer.export(scored_df, output_path)
    logger.info(f"Batch scoring complete. Results at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score customers with the churn model")
    parser.add_argument("--config", type=str, default="configs/predict_config.yaml")
    args = parser.parse_args()
    run(args.config)
