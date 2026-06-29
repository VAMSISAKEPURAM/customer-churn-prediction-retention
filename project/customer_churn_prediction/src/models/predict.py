"""
Model inference / scoring module.
Loads a trained model artifact and scores new customer records.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class ChurnScorer:
    """
    Production-ready scoring class.
    Loads model and encoder artifacts and produces calibrated churn probability scores.

    Usage:
        scorer = ChurnScorer.from_config(config)
        scored_df = scorer.score(df_new_customers)
        scorer.export(scored_df, "reports/scored_customers.csv")
    """

    def __init__(self, model: Any, encoder: Any, feature_names: List[str], config: dict):
        self.model = model
        self.encoder = encoder
        self.feature_names = feature_names
        self.config = config

    @classmethod
    def from_config(cls, config: dict) -> "ChurnScorer":
        """Initialize scorer from config paths."""
        model_dir = Path(config["model"]["artifact_path"]).parent
        model = joblib.load(config["model"]["artifact_path"])
        encoder = joblib.load(config["model"]["encoder_path"])
        with open(config["model"]["feature_names_path"]) as f:
            feature_names = json.load(f)
        logger.info("ChurnScorer loaded all artifacts.")
        return cls(model=model, encoder=encoder, feature_names=feature_names, config=config)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run end-to-end scoring on a new customer DataFrame.
        Returns the original DataFrame with churn_probability and risk_decile appended.
        """
        id_col = self.config["data"]["id_column"]
        score_col = self.config["scoring"]["score_column"]
        decile_col = self.config["scoring"]["decile_column"]
        score_date_col = self.config["scoring"]["score_date_column"]

        logger.info(f"Scoring {len(df)} customers...")

        # Select only the features the model was trained on
        X = df[[c for c in self.feature_names if c in df.columns]]
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            logger.warning(f"Missing {len(missing)} expected features — filling with 0: {missing[:10]}")
            for col in missing:
                X[col] = 0
        X = X[self.feature_names]

        # Predict probabilities
        probabilities = self.model.predict_proba(X)[:, 1]

        # Build output
        output_df = df[[id_col]].copy()
        output_df[score_col] = probabilities
        output_df[decile_col] = pd.qcut(probabilities, q=10, labels=False, duplicates="drop") + 1
        output_df[score_date_col] = str(date.today())

        logger.info(f"Scoring complete. Avg churn probability: {probabilities.mean():.4f}")
        return output_df

    def export(self, scored_df: pd.DataFrame, output_path: str):
        """Save scored output to CSV."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(output_path, index=False)
        logger.info(f"Scored output exported to: {output_path}")
